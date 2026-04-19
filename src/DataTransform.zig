const std = @import("std");
const gguf = @import("Gguf.zig");
const types = @import("types.zig");
const ggml = @import("ggml.h");

pub const Quantizer = struct {
    // Main entry point: Source -> F32 -> Dest
    pub fn convertTensorData(
        allocator: std.mem.Allocator,
        src_data: []const u8,
        src_type: types.DataType,
        dst_type: types.DataType,
        element_count: u64,
        pool: *std.Thread.Pool,
    ) ![]u8 {
        // Optimization: Direct copy if types match
        if (src_type.equivalentType(@tagName(dst_type))) {
            const out = try allocator.alloc(u8, src_data.len);
            @memcpy(out, src_data);
            return out;
        }

        // 1. Dequantize to F32 (Intermediate Buffer)
        // We allocate this temporarily
        const f32_buffer = try allocator.alloc(f32, @intCast(element_count));
        defer allocator.free(f32_buffer);

        try dequantizeToF32(src_data, f32_buffer, src_type, pool);

        // 2. Quantize from F32 to Target
        const out_size = dst_type.calcSizeInBytes(element_count);
        const out_buffer = try allocator.alloc(u8, out_size);
        errdefer allocator.free(out_buffer); // Free on error, otherwise return ownership

        try quantizeFromF32(f32_buffer, out_buffer, dst_type, pool);

        return out_buffer;
    }

    fn dequantizeToF32(
        input_bytes: []const u8,
        output_f32: []f32,
        src_type: types.DataType,
        pool: *std.Thread.Pool,
    ) !void {
        switch (src_type) {
            .F8_E4M3 => {
                if (input_bytes.len != output_f32.len)
                    return error.InputSizeMismatch;
                try dequantizeSimple(input_bytes, output_f32, pool, .F8_E4M3);
            },
            .F8_E5M2 => {
                if (input_bytes.len != output_f32.len)
                    return error.InputSizeMismatch;
                try dequantizeSimple(input_bytes, output_f32, pool, .F8_E5M2);
            },
            .BF16, .bf16 => {
                if (input_bytes.len / 2 != output_f32.len) return error.InputSizeMismatch;
                const in_ptr: [*]const ggml.ggml_bf16_t = @ptrCast(@alignCast(input_bytes.ptr));
                ggml.ggml_bf16_to_fp32_row(in_ptr, output_f32.ptr, @intCast(output_f32.len));
            },
            .F16, .f16 => {
                if (input_bytes.len / 2 != output_f32.len) return error.InputSizeMismatch;
                const in_ptr: [*]const ggml.ggml_fp16_t = @ptrCast(@alignCast(input_bytes.ptr));
                ggml.ggml_fp16_to_fp32_row(in_ptr, output_f32.ptr, @intCast(output_f32.len));
            },
            .F32, .f32 => {
                const input_vals = std.mem.bytesAsSlice(f32, input_bytes);
                @memcpy(output_f32, input_vals);
            },
            .F64, .f64 => {
                const f64_count = input_bytes.len / 8;
                if (f64_count != output_f32.len) return error.InputSizeMismatch;
                try dequantizeSimple(input_bytes, output_f32, pool, .F64);
            },
            else => return error.UnsupportedSourceType,
        }
    }

    fn quantizeFromF32(
        input_f32: []const f32,
        output_bytes: []u8,
        dst_type: types.DataType,
        pool: *std.Thread.Pool,
    ) !void {
        switch (dst_type) {
            .f32, .F32 => {
                const out_slice = std.mem.bytesAsSlice(f32, output_bytes);
                @memcpy(out_slice, input_f32);
            },
            .BF16, .bf16 => {
                const out_ptr: [*]ggml.ggml_bf16_t = @ptrCast(@alignCast(output_bytes.ptr));
                ggml.ggml_fp32_to_bf16_row(input_f32.ptr, out_ptr, @intCast(input_f32.len));
            },
            .f16, .F16 => {
                const out_ptr: [*]ggml.ggml_fp16_t = @ptrCast(@alignCast(output_bytes.ptr));
                ggml.ggml_fp32_to_fp16_row(input_f32.ptr, out_ptr, @intCast(input_f32.len));
            },
            .F8_E4M3, .F8_E5M2 => {
                if (output_bytes.len != input_f32.len)
                    return error.OutputBufferSizeMismatch;
                try convertTypeSimple(input_f32, output_bytes, pool, dst_type);
            },
            .q8_0, .q5_0, .q4_0,
            .q5_1, .q4_1,
            .q6_k, .q5_k, .q4_k, .q3_k, .q2_k => {
                const gguf_type = try gguf.GgmlType.fromString(@tagName(dst_type));
                const block_elements = gguf_type.getBlockSize();
                const block_size = gguf_type.getBytesPerBlock();

                try convertTypeGguf(
                    input_f32,
                    output_bytes,
                    pool,
                    gguf_type,
                    block_elements,
                    block_size,
                );
            },
            else => return error.UnsupportedDestinationType,
        }
    }

    fn convertTypeGguf(
        input_f32: []const f32,
        output_bytes: []u8,
        pool: *std.Thread.Pool,
        q_type: gguf.GgmlType,
        block_elements: u64,
        block_size: u64,
    ) !void {
        const element_count: u64 = @intCast(input_f32.len);
        const block_count = @divExact(element_count, block_elements);
        const threads_u64: u64 = @intCast(pool.threads.len);

        // Ensure output buffer is large enough
        if (output_bytes.len < block_count * block_size) return error.OutputBufferTooSmall;

        // divide blocks up for threads
        const blocks_per_thread = @divTrunc(block_count, threads_u64);
        const leftover = block_count - (blocks_per_thread * threads_u64);

        var wg: std.Thread.WaitGroup = .{};

        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * blocks_per_thread;
            var end = start + blocks_per_thread;
            if (i == threads_u64 - 1) {
                end += leftover;
            }
            //std.log.debug("Spawning a task for blocks {} - {} of {}", .{ start, end, block_count });
            pool.spawnWg(&wg, processBlocks, .{ input_f32, output_bytes, start, end, block_elements, block_size, q_type });
        }
        wg.wait();
    }

    fn processBlocks(input_f32: []const f32, output_bytes: []u8, start: u64, end: u64, block_elements: u64, block_size: u64, q_type: gguf.GgmlType) void {
        const size = end - start;
        const src_offset: usize = @intCast(start * block_elements);
        const dst_offset: usize = @intCast(start * block_size);
        const block_elements_usize: usize = @intCast(block_elements);
        const block_size_usize: usize = @intCast(block_size);
        const src_block = input_f32[src_offset .. src_offset + block_elements_usize];
        const dst_block = output_bytes[dst_offset .. dst_offset + block_size_usize];

        _ = ggml.ggml_quantize_chunk(
            @as(ggml.enum_ggml_type, @intCast(@intFromEnum(q_type))),
            src_block.ptr,
            dst_block.ptr,
            0,
            @intCast(size),
            @intCast(block_elements),
            null,
        );
    }

    fn convertTypeSimple(
        input_f32: []const f32,
        output_bytes: []u8,
        pool: *std.Thread.Pool,
        dst_type: types.DataType,
    ) !void {
        const element_count = input_f32.len;
        const threads_count = @min(pool.threads.len, element_count);
        const elems_per_thread = element_count / threads_count;
        const leftover = element_count - (elems_per_thread * threads_count);

        var wg: std.Thread.WaitGroup = .{};

        var i: usize = 0;
        while (i < threads_count) : (i += 1) {
            const start = i * elems_per_thread;
            const end = start + elems_per_thread + (if (i == threads_count - 1) leftover else 0);
            pool.spawnWg(&wg, processSimple, .{ input_f32, output_bytes, start, end, dst_type });
        }
        wg.wait();
    }

    fn processSimple(input_f32: []const f32, output_bytes: []u8, start: usize, end: usize, dst_type: types.DataType) void {
        switch (dst_type) {
            .BF16, .bf16 => {
                const out_slice = std.mem.bytesAsSlice(u16, output_bytes);
                for (input_f32[start..end], start..) |val, i| {
                    out_slice[i] = f32_to_bf16(val);
                }
            },
            .F16, .f16 => {
                const out_slice = std.mem.bytesAsSlice(f16, output_bytes);
                for (input_f32[start..end], start..) |val, i| {
                    out_slice[i] = @floatCast(val);
                }
            },
            .F8_E4M3 => quantizeF8Row(.F8_E4M3, input_f32[start..end], output_bytes[start..end]),
            .F8_E5M2 => quantizeF8Row(.F8_E5M2, input_f32[start..end], output_bytes[start..end]),
            else => unreachable,
        }
    }

    const fp8_vec_width = 8;

    fn quantizeF8Row(comptime fp8_type: types.DataType, input: []const f32, output: []u8) void {
        const W = fp8_vec_width;
        var i: usize = 0;
        while (i + W <= input.len) : (i += W) {
            const chunk: @Vector(W, f32) = input[i..][0..W].*;
            const vec_result: @Vector(W, u8) = switch (fp8_type) {
                .F8_E4M3 => f32_to_fp8_e4m3_chunk(chunk),
                .F8_E5M2 => f32_to_fp8_e5m2_chunk(chunk),
                else => unreachable,
            };
            output[i..][0..W].* = vec_result;
        }
        while (i < input.len) : (i += 1) {
            output[i] = switch (fp8_type) {
                .F8_E4M3 => f32_to_fp8_e4m3(input[i]),
                .F8_E5M2 => f32_to_fp8_e5m2(input[i]),
                else => unreachable,
            };
        }
    }

    pub fn f32_to_fp8_e4m3_chunk(chunk: @Vector(fp8_vec_width, f32)) @Vector(fp8_vec_width, u8) {
        const W = fp8_vec_width;
        const U32V = @Vector(W, u32);
        const I32V = @Vector(W, i32);
        const F32V = @Vector(W, f32);

        const bits: U32V = @bitCast(chunk);
        const sign: U32V = bits >> @as(U32V, @splat(31));
        const f32_exp: U32V = (bits >> @as(U32V, @splat(23))) & @as(U32V, @splat(0xFF));
        const f32_mant: U32V = bits & @as(U32V, @splat(0x7FFFFF));

        // f8_exp_biased = f32_exp_biased - 127 + 7 = f32_exp_biased - 120
        const f8_exp_i: I32V = @as(I32V, @intCast(f32_exp)) - @as(I32V, @splat(120));
        // Round-to-nearest: add round bit (f32 mantissa bit 19) before extracting top 3 bits
        const f8_mant_r: U32V = ((f32_mant >> @as(U32V, @splat(20))) & @as(U32V, @splat(7))) + ((f32_mant >> @as(U32V, @splat(19))) & @as(U32V, @splat(1)));
        const f8_mant_carry: U32V = f8_mant_r >> @as(U32V, @splat(3)); // 1 if rounded up to 8
        const f8_mant: U32V = f8_mant_r & @as(U32V, @splat(7));
        const f8_exp_adj: I32V = f8_exp_i + @as(I32V, @intCast(f8_mant_carry));

        const is_f32_special: @Vector(W, bool) = f32_exp == @as(U32V, @splat(255));
        // Clamp if exp >= 16, or exp == 15 with mant == 7 (which would encode as NaN 0x7F)
        const would_encode_nan: @Vector(W, bool) = (f8_exp_adj == @as(I32V, @splat(15))) & (f8_mant == @as(U32V, @splat(7)));
        const is_overflow: @Vector(W, bool) = ((f8_exp_adj >= @as(I32V, @splat(16))) | would_encode_nan) & !is_f32_special;
        const is_small: @Vector(W, bool) = (f8_exp_adj <= @as(I32V, @splat(0))) & !is_f32_special;

        // Subnormal F8 E4M3: mant = round(|x| * 2^9), clamped to [0, 7]
        const abs_v: F32V = @abs(chunk);
        const subnorm_mant: U32V = @intFromFloat(@min(abs_v * @as(F32V, @splat(512.0)) + @as(F32V, @splat(0.5)), @as(F32V, @splat(7.0))));

        // Clamp exponent for normal path [1..15]
        const f8_exp_u: U32V = @intCast(@min(@max(f8_exp_adj, @as(I32V, @splat(1))), @as(I32V, @splat(15))));
        const normal: U32V = (sign << @as(U32V, @splat(7))) | (f8_exp_u << @as(U32V, @splat(3))) | f8_mant;

        var result: U32V = normal;
        result = @select(u32, is_small,    (sign << @as(U32V, @splat(7))) | subnorm_mant,       result);
        result = @select(u32, is_overflow, (sign << @as(U32V, @splat(7))) | @as(U32V, @splat(0x7E)), result);
        result = @select(u32, is_f32_special, @as(U32V, @splat(0x7F)), result);

        return @truncate(result);
    }

    pub fn f32_to_fp8_e5m2_chunk(chunk: @Vector(fp8_vec_width, f32)) @Vector(fp8_vec_width, u8) {
        const W = fp8_vec_width;
        const U32V = @Vector(W, u32);
        const I32V = @Vector(W, i32);
        const F32V = @Vector(W, f32);

        const bits: U32V = @bitCast(chunk);
        const sign: U32V = bits >> @as(U32V, @splat(31));
        const f32_exp: U32V = (bits >> @as(U32V, @splat(23))) & @as(U32V, @splat(0xFF));
        const f32_mant: U32V = bits & @as(U32V, @splat(0x7FFFFF));

        // f8_exp_biased = f32_exp_biased - 127 + 15 = f32_exp_biased - 112
        const f8_exp_i: I32V = @as(I32V, @intCast(f32_exp)) - @as(I32V, @splat(112));
        // Round-to-nearest: add round bit (f32 mantissa bit 20) before extracting top 2 bits
        const f8_mant_r: U32V = ((f32_mant >> @as(U32V, @splat(21))) & @as(U32V, @splat(3))) + ((f32_mant >> @as(U32V, @splat(20))) & @as(U32V, @splat(1)));
        const f8_mant_carry: U32V = f8_mant_r >> @as(U32V, @splat(2)); // 1 if rounded up to 4
        const f8_mant: U32V = f8_mant_r & @as(U32V, @splat(3));
        const f8_exp_adj: I32V = f8_exp_i + @as(I32V, @intCast(f8_mant_carry));

        const is_f32_special: @Vector(W, bool) = f32_exp == @as(U32V, @splat(255));
        const is_nan: @Vector(W, bool) = is_f32_special & (f32_mant != @as(U32V, @splat(0)));
        const is_inf: @Vector(W, bool) = is_f32_special & (f32_mant == @as(U32V, @splat(0)));
        const is_overflow: @Vector(W, bool) = (f8_exp_adj >= @as(I32V, @splat(31))) & !is_f32_special;
        const is_small: @Vector(W, bool) = (f8_exp_adj <= @as(I32V, @splat(0))) & !is_f32_special;

        // Subnormal F8 E5M2: mant = round(|x| * 2^16), clamped to [0, 3]
        const abs_v: F32V = @abs(chunk);
        const subnorm_mant: U32V = @intFromFloat(@min(abs_v * @as(F32V, @splat(65536.0)) + @as(F32V, @splat(0.5)), @as(F32V, @splat(3.0))));

        // Clamp exponent for normal path [1..30]
        const f8_exp_u: U32V = @intCast(@min(@max(f8_exp_adj, @as(I32V, @splat(1))), @as(I32V, @splat(30))));
        const normal: U32V = (sign << @as(U32V, @splat(7))) | (f8_exp_u << @as(U32V, @splat(2))) | f8_mant;

        var result: U32V = normal;
        result = @select(u32, is_small,    (sign << @as(U32V, @splat(7))) | subnorm_mant,        result);
        result = @select(u32, is_overflow, (sign << @as(U32V, @splat(7))) | @as(U32V, @splat(0x7B)), result);
        result = @select(u32, is_inf,      (sign << @as(U32V, @splat(7))) | @as(U32V, @splat(0x7C)), result);
        result = @select(u32, is_nan,      @as(U32V, @splat(0x7F)), result);

        return @truncate(result);
    }

    fn dequantizeSimple(
        input_bytes: []const u8,
        output_f32: []f32,
        pool: *std.Thread.Pool,
        src_type: types.DataType,
    ) !void {
        const element_count = output_f32.len;
        const threads_count = @min(pool.threads.len, element_count);
        const elems_per_thread = element_count / threads_count;
        const leftover = element_count - (elems_per_thread * threads_count);

        var wg: std.Thread.WaitGroup = .{};

        var i: usize = 0;
        while (i < threads_count) : (i += 1) {
            const start = i * elems_per_thread;
            const end = start + elems_per_thread + (if (i == threads_count - 1) leftover else 0);
            pool.spawnWg(&wg, processDequantize, .{ input_bytes, output_f32, start, end, src_type });
        }
        wg.wait();
    }

    fn processDequantize(input_bytes: []const u8, output_f32: []f32, start: usize, end: usize, src_type: types.DataType) void {
        switch (src_type) {
            .F8_E4M3 => {
                for (input_bytes[start..end], start..) |b, i| {
                    output_f32[i] = lut_e4m3[b];
                }
            },
            .F8_E5M2 => {
                for (input_bytes[start..end], start..) |b, i| {
                    output_f32[i] = lut_e5m2[b];
                }
            },
            .BF16, .bf16 => {
                // input slice for this thread: each element is 2 bytes, so byte offsets are doubled
                    const in_slice = std.mem.bytesAsSlice(u16, input_bytes);
                for (in_slice[start..end], start..) |val, i| {
                    output_f32[i] = bf16_to_f32(val);
                }
            },
            .F16, .f16 => {
                const in_slice = std.mem.bytesAsSlice(f16, input_bytes);
                for (in_slice[start..end], start..) |val, i| {
                    output_f32[i] = @floatCast(val);
                }
            },
            .F64, .f64 => {
                const in_slice = std.mem.bytesAsSlice(f64, input_bytes);
                for (in_slice[start..end], start..) |val, i| {
                    output_f32[i] = @floatCast(val);
                }
            },
            else => unreachable,
        }
    }

    pub fn fp8_e4m3_to_f32(x: u8) f32 {
        const sign: f32 = @floatFromInt((x >> 7) & 0x1);
        const exp = (x >> 3) & 0xF;
        const mant = x & 0x7;
        const sign_mult = 1.0 - 2.0 * sign;

        if (exp == 0) {
            // Subnormal: ±mant * 2^(-9)
            return sign_mult * @as(f32, @floatFromInt(mant)) / 8.0 * @exp2(@as(f32, -6.0));
        }
        if (exp == 0xF and mant == 0x7) {
            // E4M3FN: only 0x7F/0xFF are NaN; no Inf representation
            return std.math.nan(f32);
        }
        // Normal (includes exp=0xF with mant 0–6, which encode values up to 448)
        const e = @as(f32, @floatFromInt(exp)) - 7.0;
        const m = 1.0 + @as(f32, @floatFromInt(mant)) / 8.0;
        return sign_mult * m * @exp2(e);
    }

    pub fn fp8_e5m2_to_f32(x: u8) f32 {
        const sign = @as(f32, @floatFromInt((x >> 7) & 0x1));
        const exp = (x >> 2) & 0x1F;
        const mant = x & 0x3;

        if (exp == 0) {
            const m = @as(f32, @floatFromInt(mant)) / 4.0;
            return (1.0 - 2.0 * sign) * m * @exp2(@as(f32, -14.0));
        } else if (exp == 0x1F) {
            if (mant == 0) return std.math.inf(f32) * (1.0 - 2.0 * sign);
            return std.math.nan(f32);
        } else {
            const e = @as(f32, @floatFromInt(exp)) - 15.0;
            const m = 1.0 + @as(f32, @floatFromInt(mant)) / 4.0;
            return (1.0 - 2.0 * sign) * m * @exp2(e);
        }
    }

    pub const lut_e4m3: [256]f32 = blk: {
        @setEvalBranchQuota(10000);
        var t: [256]f32 = undefined;
        var i: u32 = 0;
        while (i < 256) : (i += 1) t[i] = fp8_e4m3_to_f32(@intCast(i));
        break :blk t;
    };
    pub const lut_e5m2: [256]f32 = blk: {
        @setEvalBranchQuota(10000);
        var t: [256]f32 = undefined;
        var i: u32 = 0;
        while (i < 256) : (i += 1) t[i] = fp8_e5m2_to_f32(@intCast(i));
        break :blk t;
    };

    pub fn f32_to_fp8_e4m3(x: f32) u8 {
        if (std.math.isNan(x)) return 0x7F;

        const sign_bit: u8 = if (x < 0.0 or (x == 0.0 and std.math.signbit(x))) 0x80 else 0x00;
        const abs_x = @abs(x);

        if (std.math.isInf(abs_x) or abs_x > 448.0) {
            return sign_bit | 0x7E;
        }

        if (abs_x == 0.0) return sign_bit;

        const log2_val = std.math.log2(abs_x);
        const e: i32 = @intFromFloat(@floor(log2_val));

        if (e < -6) {
            // Subnormal: encoded value = mant * 2^(-9)
            const mant_f = abs_x * 512.0;
            const mant: u8 = @intFromFloat(@min(7.0, @max(0.0, @round(mant_f))));
            return sign_bit | mant;
        }

        // Normal: biased_exp in [1..15]; max representable e=8 (biased 15, mant≤6 = 448)
        const e_clamped: i32 = @min(8, e);
        const biased_exp: u8 = @intCast(e_clamped + 7);
        const scale = std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e_clamped)));
        const mant_f = (abs_x / scale - 1.0) * 8.0;
        const mant: u8 = @intFromFloat(@min(7.0, @max(0.0, @round(mant_f))));
        return sign_bit | (biased_exp << 3) | mant;
    }

    pub fn f32_to_fp8_e5m2(x: f32) u8 {
        if (std.math.isNan(x)) return 0x7F;

        const sign_bit: u8 = if (x < 0.0 or (x == 0.0 and std.math.signbit(x))) 0x80 else 0x00;
        const abs_x = @abs(x);

        if (std.math.isInf(abs_x)) return sign_bit | 0x7C;
        if (abs_x > 57344.0) return sign_bit | 0x7B;

        if (abs_x == 0.0) return sign_bit;

        const log2_val = std.math.log2(abs_x);
        const e: i32 = @intFromFloat(@floor(log2_val));

        if (e < -14) {
            // Subnormal: encoded value = mant * 2^(-16)
            const mant_f = abs_x * 65536.0;
            const mant: u8 = @intFromFloat(@min(3.0, @max(0.0, @round(mant_f))));
            return sign_bit | mant;
        }

        // Normal: biased_exp in [1..30]; max e=15 (biased 30, max normal = 57344)
        const e_clamped: i32 = @min(15, e);
        const biased_exp: u8 = @intCast(e_clamped + 15);
        const scale = std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e_clamped)));
        const mant_f = (abs_x / scale - 1.0) * 4.0;
        const mant: u8 = @intFromFloat(@min(3.0, @max(0.0, @round(mant_f))));
        return sign_bit | (biased_exp << 2) | mant;
    }

    fn bf16_to_f32(x: u16) f32 {
        const bits = (@as(u32, x) << 16);
        return @bitCast(bits);
    }

    fn f32_to_bf16(x: f32) u16 {
        const bits: u32 = @bitCast(x);
        return @truncate(bits >> 16);
    }
};

test "transform f16 to q8_0" {
    const allocator = std.testing.allocator;

    // Load the f16 source file (skip test if artifacts not present)
    const f16_file = std.fs.cwd().openFile("test-artifact/output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight.f16", .{}) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer f16_file.close();

    const f16_data = try f16_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(f16_data);

    // Calculate element count (f16 is 2 bytes per element)
    const element_count: u64 = @intCast(f16_data.len / 2);

    var pool: std.Thread.Pool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    // Convert f16 to q8_0
    const q8_0_data = try Quantizer.convertTensorData(
        allocator,
        f16_data,
        types.DataType.f16,
        types.DataType.q8_0,
        element_count,
        &pool,
    );
    defer allocator.free(q8_0_data);

    try std.testing.expectEqual(q8_0_data.len, 1740800);

    // Load the expected q8_0 file
    const expected_file = try std.fs.cwd().openFile("test-artifact/output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight.q8_0", .{});
    defer expected_file.close();

    const expected_data = try expected_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(expected_data);

    // Compare the results
    try std.testing.expectEqual(expected_data.len, q8_0_data.len);
    try std.testing.expectEqualSlices(u8, expected_data, q8_0_data);
}

// ============================================================================
// F8 conversion tests
// ============================================================================

test "F8_E4M3 scalar encode: exact values" {
    // 1.0 → 0_0111_000 = 0x38
    try std.testing.expectEqual(@as(u8, 0x38), Quantizer.f32_to_fp8_e4m3(1.0));
    // -1.0 → 0xB8
    try std.testing.expectEqual(@as(u8, 0xB8), Quantizer.f32_to_fp8_e4m3(-1.0));
    // 2.0 → 0_1000_000 = 0x40
    try std.testing.expectEqual(@as(u8, 0x40), Quantizer.f32_to_fp8_e4m3(2.0));
    // 0.5 → 0_0110_000 = 0x30
    try std.testing.expectEqual(@as(u8, 0x30), Quantizer.f32_to_fp8_e4m3(0.5));
    // 0.0 → 0x00
    try std.testing.expectEqual(@as(u8, 0x00), Quantizer.f32_to_fp8_e4m3(0.0));
    // -0.0 → 0x80
    try std.testing.expectEqual(@as(u8, 0x80), Quantizer.f32_to_fp8_e4m3(-0.0));
    // NaN → 0x7F
    try std.testing.expectEqual(@as(u8, 0x7F), Quantizer.f32_to_fp8_e4m3(std.math.nan(f32)));
    // 448.0 (max) → 0x7E
    try std.testing.expectEqual(@as(u8, 0x7E), Quantizer.f32_to_fp8_e4m3(448.0));
    // Overflow → clamp to 0x7E
    try std.testing.expectEqual(@as(u8, 0x7E), Quantizer.f32_to_fp8_e4m3(1e10));
    // Negative overflow → 0xFE
    try std.testing.expectEqual(@as(u8, 0xFE), Quantizer.f32_to_fp8_e4m3(-1e10));
}

test "F8_E5M2 scalar encode: exact values" {
    // 1.0 → 0_01111_00 = 0x3C
    try std.testing.expectEqual(@as(u8, 0x3C), Quantizer.f32_to_fp8_e5m2(1.0));
    // -1.0 → 0xBC
    try std.testing.expectEqual(@as(u8, 0xBC), Quantizer.f32_to_fp8_e5m2(-1.0));
    // 2.0 → 0_10000_00 = 0x40
    try std.testing.expectEqual(@as(u8, 0x40), Quantizer.f32_to_fp8_e5m2(2.0));
    // 0.5 → 0_01110_00 = 0x38
    try std.testing.expectEqual(@as(u8, 0x38), Quantizer.f32_to_fp8_e5m2(0.5));
    // 0.0 → 0x00
    try std.testing.expectEqual(@as(u8, 0x00), Quantizer.f32_to_fp8_e5m2(0.0));
    // -0.0 → 0x80
    try std.testing.expectEqual(@as(u8, 0x80), Quantizer.f32_to_fp8_e5m2(-0.0));
    // +Inf → 0x7C
    try std.testing.expectEqual(@as(u8, 0x7C), Quantizer.f32_to_fp8_e5m2(std.math.inf(f32)));
    // -Inf → 0xFC
    try std.testing.expectEqual(@as(u8, 0xFC), Quantizer.f32_to_fp8_e5m2(-std.math.inf(f32)));
    // NaN → 0x7F
    try std.testing.expectEqual(@as(u8, 0x7F), Quantizer.f32_to_fp8_e5m2(std.math.nan(f32)));
    // 57344.0 (max normal) → 0x7B
    try std.testing.expectEqual(@as(u8, 0x7B), Quantizer.f32_to_fp8_e5m2(57344.0));
    // Overflow → 0x7B (max, not Inf)
    try std.testing.expectEqual(@as(u8, 0x7B), Quantizer.f32_to_fp8_e5m2(1e10));
}

test "F8_E4M3 round-trip: exact representable values" {
    const cases = [_]f32{ 1.0, -1.0, 2.0, 0.5, 0.25, 4.0, 0.125, -2.0, -0.5, 448.0, -448.0 };
    for (cases) |v| {
        const encoded = Quantizer.f32_to_fp8_e4m3(v);
        const decoded = Quantizer.lut_e4m3[encoded];
        try std.testing.expectApproxEqRel(v, decoded, 0.001);
    }
}

test "F8_E5M2 round-trip: exact representable values" {
    const cases = [_]f32{ 1.0, -1.0, 2.0, 0.5, 0.25, 4.0, 0.125, -2.0, -0.5, 57344.0 };
    for (cases) |v| {
        const encoded = Quantizer.f32_to_fp8_e5m2(v);
        const decoded = Quantizer.lut_e5m2[encoded];
        try std.testing.expectApproxEqRel(v, decoded, 0.001);
    }
}

test "F8_E4M3 decode: LUT matches scalar function" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const from_lut = Quantizer.lut_e4m3[b];
        const from_fn = Quantizer.fp8_e4m3_to_f32(b);
        if (std.math.isNan(from_fn)) {
            try std.testing.expect(std.math.isNan(from_lut));
        } else {
            try std.testing.expectEqual(from_fn, from_lut);
        }
    }
}

test "F8_E5M2 decode: LUT matches scalar function" {
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        const from_lut = Quantizer.lut_e5m2[b];
        const from_fn = Quantizer.fp8_e5m2_to_f32(b);
        if (std.math.isNan(from_fn)) {
            try std.testing.expect(std.math.isNan(from_lut));
        } else {
            try std.testing.expectEqual(from_fn, from_lut);
        }
    }
}

test "F8_E4M3 SIMD chunk matches scalar" {
    // Build a varied input covering normals, subnormals, zero, large values
    var input: [16]f32 = .{ 1.0, -1.0, 0.5, 2.0, 0.0, -0.0, 448.0, -448.0, 1e10, -1e10, 0.001, -0.001, 3.14, -2.71, 100.0, 0.0625 };
    var scalar_out: [16]u8 = undefined;
    for (input, 0..) |v, i| scalar_out[i] = Quantizer.f32_to_fp8_e4m3(v);

    var simd_out: [16]u8 = undefined;
    const chunk0: @Vector(8, f32) = input[0..8].*;
    const chunk1: @Vector(8, f32) = input[8..16].*;
    simd_out[0..8].* = @as([8]u8, Quantizer.f32_to_fp8_e4m3_chunk(chunk0));
    simd_out[8..16].* = @as([8]u8, Quantizer.f32_to_fp8_e4m3_chunk(chunk1));

    try std.testing.expectEqualSlices(u8, &scalar_out, &simd_out);
}

test "F8_E5M2 SIMD chunk matches scalar" {
    var input: [16]f32 = .{ 1.0, -1.0, 0.5, 2.0, 0.0, -0.0, 57344.0, -57344.0, 1e10, -1e10, 0.001, -0.001, 3.14, -2.71, 100.0, 0.0625 };
    var scalar_out: [16]u8 = undefined;
    for (input, 0..) |v, i| scalar_out[i] = Quantizer.f32_to_fp8_e5m2(v);

    var simd_out: [16]u8 = undefined;
    const chunk0: @Vector(8, f32) = input[0..8].*;
    const chunk1: @Vector(8, f32) = input[8..16].*;
    simd_out[0..8].* = @as([8]u8, Quantizer.f32_to_fp8_e5m2_chunk(chunk0));
    simd_out[8..16].* = @as([8]u8, Quantizer.f32_to_fp8_e5m2_chunk(chunk1));

    try std.testing.expectEqualSlices(u8, &scalar_out, &simd_out);
}
