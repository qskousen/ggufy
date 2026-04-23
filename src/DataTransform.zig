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
        // Vectorized ml_dtypes float8_e4m3fn ConvertFrom<float> (non-saturating, round-to-nearest-even).
        //
        // Uses a fixed shift of 20 for the normal path to avoid vpsrlvd (slow variable-shift).
        // Subnormal f8 values (tbe < 0) are handled with the IEEE 754 add-magic RTE trick.
        const W = fp8_vec_width;
        const U32V = @Vector(W, u32);
        const I32V = @Vector(W, i32);
        const F32V = @Vector(W, f32);

        const bits: U32V = @bitCast(chunk);
        const sign: U32V = bits >> @as(U32V, @splat(31));
        const abs_bits: U32V = bits & @as(U32V, @splat(0x7FFF_FFFF));

        const is_special: @Vector(W, bool) = abs_bits >= @as(U32V, @splat(0x7F80_0000));

        const f32_biased_exp: U32V = abs_bits >> @as(U32V, @splat(23));
        const norm_mant: U32V = @as(U32V, @splat(0x80_0000)) | (abs_bits & @as(U32V, @splat(0x7F_FFFF)));

        // tbe = (f32_biased_exp - 127) + 6 = f32_biased_exp - 121
        const tbe: I32V = @as(I32V, @intCast(f32_biased_exp)) - @as(I32V, @splat(121));
        const is_subnorm: @Vector(W, bool) = tbe < @as(I32V, @splat(0));

        // Normal path: fixed ashift = 20 → compiles to vpsrld/vpslld (fast).
        const L: U32V = (norm_mant >> @as(U32V, @splat(20))) & @as(U32V, @splat(1));
        const rounded: U32V = norm_mant + (L + @as(U32V, @splat(0x7FFFF)));
        const aligned: U32V = rounded >> @as(U32V, @splat(20));
        const exp_bits: U32V = @intCast(@max(@as(I32V, @splat(0)), tbe));
        const result_normal: U32V = aligned + (exp_bits << @as(U32V, @splat(3)));

        // Subnormal path: mant = RTE(|x| * 512).
        // IEEE 754 addition with magic constant performs round-to-nearest-even.
        // No upper clamp: values that round up to 8 correctly become the smallest normal (0x08).
        // Cap abs_bits at 2^-6 (0x3B800000) before float arithmetic so that NaN/Inf
        // elements (handled by is_special above) produce a safe finite scaled value
        // instead of causing @intFromFloat to panic — the @select discards these lanes.
        const magic: F32V = @splat(0x1p23); // 2^23: forces integer rounding in f32 mantissa
        const capped_abs: F32V = @bitCast(@as(U32V, @min(abs_bits, @as(U32V, @splat(0x3C80_0000)))));
        const subnorm_mant: U32V = @intFromFloat(capped_abs * @as(F32V, @splat(512.0)) + magic - magic);

        var result_pre: U32V = @select(u32, is_subnorm, subnorm_mant, result_normal);

        // Overflow: tbe >= 16 OR result > 0x7E → 0x7F (E4M3FN has no infinity, overflow = NaN).
        const is_overflow: @Vector(W, bool) = (tbe >= @as(I32V, @splat(16))) | (result_pre > @as(U32V, @splat(0x7E)));
        result_pre = @select(u32, is_overflow, @as(U32V, @splat(0x7F)), result_pre);

        // Apply sign; NaN/Inf override to (sign << 7) | 0x7F.
        var result: U32V = (sign << @as(U32V, @splat(7))) | result_pre;
        result = @select(u32, is_special, (sign << @as(U32V, @splat(7))) | @as(U32V, @splat(0x7F)), result);

        return @truncate(result);
    }

    pub fn f32_to_fp8_e5m2_chunk(chunk: @Vector(fp8_vec_width, f32)) @Vector(fp8_vec_width, u8) {
        // Vectorized ml_dtypes float8_e5m2 ConvertFrom<float> (non-saturating, round-to-nearest-even).
        //
        // Uses a fixed shift of 21 for the normal path to avoid vpsrlvd (slow variable-shift).
        // Subnormal f8 values (tbe < 0) are handled with the IEEE 754 add-magic RTE trick.
        const W = fp8_vec_width;
        const U32V = @Vector(W, u32);
        const I32V = @Vector(W, i32);
        const F32V = @Vector(W, f32);

        const bits: U32V = @bitCast(chunk);
        const sign: U32V = bits >> @as(U32V, @splat(31));
        const abs_bits: U32V = bits & @as(U32V, @splat(0x7FFF_FFFF));

        const is_nan: @Vector(W, bool) = abs_bits > @as(U32V, @splat(0x7F80_0000));
        const is_inf: @Vector(W, bool) = abs_bits == @as(U32V, @splat(0x7F80_0000));

        const f32_biased_exp: U32V = abs_bits >> @as(U32V, @splat(23));
        const norm_mant: U32V = @as(U32V, @splat(0x80_0000)) | (abs_bits & @as(U32V, @splat(0x7F_FFFF)));

        // tbe = (f32_biased_exp - 127) + 14 = f32_biased_exp - 113
        const tbe: I32V = @as(I32V, @intCast(f32_biased_exp)) - @as(I32V, @splat(113));
        const is_subnorm: @Vector(W, bool) = tbe < @as(I32V, @splat(0));

        // Normal path: fixed ashift = 21 → compiles to vpsrld/vpslld (fast).
        const L: U32V = (norm_mant >> @as(U32V, @splat(21))) & @as(U32V, @splat(1));
        const rounded: U32V = norm_mant + (L + @as(U32V, @splat(0xFFFFF)));
        const aligned: U32V = rounded >> @as(U32V, @splat(21));
        const exp_bits: U32V = @intCast(@max(@as(I32V, @splat(0)), tbe));
        const result_normal: U32V = aligned + (exp_bits << @as(U32V, @splat(2)));

        // Subnormal path: mant = RTE(|x| * 65536).
        // No upper clamp: values that round up to 4 correctly become the smallest normal (0x04).
        // Cap abs_bits at 2^-14 (0x38800000) before float arithmetic so NaN/Inf elements
        // produce a safe finite scaled value — the @select discards those lanes anyway.
        const magic: F32V = @splat(0x1p23);
        const capped_abs: F32V = @bitCast(@as(U32V, @min(abs_bits, @as(U32V, @splat(0x3880_0000)))));
        const subnorm_mant: U32V = @intFromFloat(capped_abs * @as(F32V, @splat(65536.0)) + magic - magic);

        var result_pre: U32V = @select(u32, is_subnorm, subnorm_mant, result_normal);

        // Overflow: tbe >= 31 OR result > 0x7B → 0x7C (Inf for E5M2).
        const is_overflow: @Vector(W, bool) = (tbe >= @as(I32V, @splat(31))) | (result_pre > @as(U32V, @splat(0x7B)));
        result_pre = @select(u32, is_overflow, @as(U32V, @splat(0x7C)), result_pre);

        // Apply sign; then override Inf and NaN.
        var result: U32V = (sign << @as(U32V, @splat(7))) | result_pre;
        result = @select(u32, is_inf, (sign << @as(U32V, @splat(7))) | @as(U32V, @splat(0x7C)), result);
        result = @select(u32, is_nan, (sign << @as(U32V, @splat(7))) | @as(U32V, @splat(0x7E)), result);

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
        // Matches ml_dtypes float8_e4m3fn ConvertFrom<float> (non-saturating, round-to-nearest-even).
        // E4M3FN: bias=7, no infinity encoding — overflow maps to NaN (0x7F).
        const bits: u32 = @bitCast(x);
        const from_sign: u8 = @truncate(bits >> 31);
        const abs_bits: u32 = bits & 0x7FFF_FFFF;

        // NaN or Inf → NaN/overflow encoding (0x7F), with sign applied.
        if (abs_bits >= 0x7F80_0000) return (from_sign << 7) | 0x7F;

        // Zero
        if (abs_bits == 0) return from_sign << 7;

        const from_biased_exp: u32 = abs_bits >> 23;
        const from_fraction: u32 = abs_bits & 0x7F_FFFF;

        var unbiased_exp: i32 = undefined;
        var norm_mant: u32 = undefined;

        if (from_biased_exp != 0) {
            unbiased_exp = @as(i32, @intCast(from_biased_exp)) - 127;
            norm_mant = 0x80_0000 | from_fraction;
        } else {
            // Subnormal f32: normalize by shifting until implicit 1 is at bit 23.
            const lz: i32 = @clz(from_fraction);
            const frac_lz: i32 = lz - 9; // leading zeros within 23-bit field
            const norm_shift: i32 = frac_lz + 1;
            norm_mant = from_fraction << @intCast(norm_shift);
            unbiased_exp = (1 - 127) - norm_shift;
        }

        // target_biased_exponent_base = unbiased_exp + kToExponentBias - 1 = unbiased_exp + 6
        const tbe: i32 = unbiased_exp + 6;

        // Shift to align 23-bit source mantissa onto 3-bit target mantissa.
        const denorm_adj: i32 = @max(0, -tbe);
        const ashift: i32 = @min(20 + denorm_adj, 25);
        const roundoff: u5 = @intCast(ashift);

        // Round-to-nearest-even (ml_dtypes RoundBitsToNearestEven).
        const bias: u32 = ((norm_mant >> roundoff) & 1) + (@as(u32, 1) << (roundoff - 1)) - 1;
        const rounded: u32 = norm_mant + bias;
        const aligned: u8 = @truncate(rounded >> roundoff);

        const exp_bits: u8 = @intCast(@max(0, tbe));
        var result: u8 = aligned +% (exp_bits << 3);

        // Overflow: tbe >= max_exponent(9) + kToExponentBias(7) = 16, or result > max_finite(0x7E).
        if (tbe >= 16 or result > 0x7E) result = 0x7F;

        return (from_sign << 7) | result;
    }

    pub fn f32_to_fp8_e5m2(x: f32) u8 {
        // Matches ml_dtypes float8_e5m2 ConvertFrom<float> (non-saturating, round-to-nearest-even).
        // E5M2: bias=15, infinity=0x7C, overflow maps to infinity.
        const bits: u32 = @bitCast(x);
        const from_sign: u8 = @truncate(bits >> 31);
        const abs_bits: u32 = bits & 0x7FFF_FFFF;

        // NaN → quiet NaN (0x7E for E5M2), with sign.
        if (abs_bits > 0x7F80_0000) return (from_sign << 7) | 0x7E;
        // Inf → ±Inf (0x7C), with sign.
        if (abs_bits == 0x7F80_0000) return (from_sign << 7) | 0x7C;
        // Zero
        if (abs_bits == 0) return from_sign << 7;

        const from_biased_exp: u32 = abs_bits >> 23;
        const from_fraction: u32 = abs_bits & 0x7F_FFFF;

        var unbiased_exp: i32 = undefined;
        var norm_mant: u32 = undefined;

        if (from_biased_exp != 0) {
            unbiased_exp = @as(i32, @intCast(from_biased_exp)) - 127;
            norm_mant = 0x80_0000 | from_fraction;
        } else {
            // Subnormal f32: normalize by shifting until implicit 1 is at bit 23.
            const lz: i32 = @clz(from_fraction);
            const frac_lz: i32 = lz - 9;
            const norm_shift: i32 = frac_lz + 1;
            norm_mant = from_fraction << @intCast(norm_shift);
            unbiased_exp = (1 - 127) - norm_shift;
        }

        // tbe = unbiased_exp + kToExponentBias - 1 = unbiased_exp + 14
        const tbe: i32 = unbiased_exp + 14;

        const denorm_adj: i32 = @max(0, -tbe);
        const ashift: i32 = @min(21 + denorm_adj, 25);
        const roundoff: u5 = @intCast(ashift);

        // Round-to-nearest-even (ml_dtypes RoundBitsToNearestEven).
        const bias: u32 = ((norm_mant >> roundoff) & 1) + (@as(u32, 1) << (roundoff - 1)) - 1;
        const rounded: u32 = norm_mant + bias;
        const aligned: u8 = @truncate(rounded >> roundoff);

        const exp_bits: u8 = @intCast(@max(0, tbe));
        var result: u8 = aligned +% (exp_bits << 2);

        // Overflow: tbe >= max_exponent(16) + kToExponentBias(15) = 31, or result > max_finite(0x7B).
        if (tbe >= 31 or result > 0x7B) result = 0x7C;

        return (from_sign << 7) | result;
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
// ml_dtypes reference fixture tests
// ============================================================================
//
// Fixtures are generated by gen_fp8_fixtures.py (venv/bin/python3 gen_fp8_fixtures.py).
// All tests skip gracefully when fixtures are absent.

const fixture_dir = "src/test_fixtures";

fn loadFixture(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    var path_buf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ fixture_dir, name });
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        if (err == error.FileNotFound) return null;
        return err;
    };
    defer file.close();
    return try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
}

// Returns the number of mismatches, printing the first few.
fn checkEncodeResults(inputs: []const f32, got: []const u8, expected: []const u8, label: []const u8) usize {
    var mismatches: usize = 0;
    for (inputs, got, expected, 0..) |val, g, e, i| {
        if (g != e) {
            if (mismatches < 8) {
                std.debug.print("  {s}[{}]: f32={d:.6} got=0x{X:0>2} expected=0x{X:0>2}\n", .{ label, i, val, g, e });
            }
            mismatches += 1;
        }
    }
    return mismatches;
}

test "F8_E4M3FN scalar encode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const inputs_bytes = (try loadFixture(allocator, "fp8_test_inputs.f32")) orelse return error.SkipZigTest;
    defer allocator.free(inputs_bytes);
    const expected = (try loadFixture(allocator, "fp8_e4m3fn_encoded.u8")) orelse return error.SkipZigTest;
    defer allocator.free(expected);

    const inputs: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(inputs_bytes)));
    try std.testing.expectEqual(inputs.len, expected.len);

    const got = try allocator.alloc(u8, inputs.len);
    defer allocator.free(got);
    for (inputs, got) |val, *out| out.* = Quantizer.f32_to_fp8_e4m3(val);

    const mismatches = checkEncodeResults(inputs, got, expected, "E4M3FN scalar");
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E5M2 scalar encode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const inputs_bytes = (try loadFixture(allocator, "fp8_test_inputs.f32")) orelse return error.SkipZigTest;
    defer allocator.free(inputs_bytes);
    const expected = (try loadFixture(allocator, "fp8_e5m2_encoded.u8")) orelse return error.SkipZigTest;
    defer allocator.free(expected);

    const inputs: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(inputs_bytes)));
    try std.testing.expectEqual(inputs.len, expected.len);

    const got = try allocator.alloc(u8, inputs.len);
    defer allocator.free(got);
    for (inputs, got) |val, *out| out.* = Quantizer.f32_to_fp8_e5m2(val);

    const mismatches = checkEncodeResults(inputs, got, expected, "E5M2 scalar");
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E4M3FN SIMD encode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const inputs_bytes = (try loadFixture(allocator, "fp8_test_inputs.f32")) orelse return error.SkipZigTest;
    defer allocator.free(inputs_bytes);
    const expected = (try loadFixture(allocator, "fp8_e4m3fn_encoded.u8")) orelse return error.SkipZigTest;
    defer allocator.free(expected);

    const inputs: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(inputs_bytes)));
    try std.testing.expectEqual(inputs.len, expected.len);

    const got = try allocator.alloc(u8, inputs.len);
    defer allocator.free(got);

    const W = Quantizer.fp8_vec_width;
    var i: usize = 0;
    while (i + W <= inputs.len) : (i += W) {
        const chunk: @Vector(W, f32) = inputs[i..][0..W].*;
        got[i..][0..W].* = @as([W]u8, Quantizer.f32_to_fp8_e4m3_chunk(chunk));
    }
    while (i < inputs.len) : (i += 1) {
        got[i] = Quantizer.f32_to_fp8_e4m3(inputs[i]);
    }

    const mismatches = checkEncodeResults(inputs, got, expected, "E4M3FN SIMD");
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E5M2 SIMD encode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const inputs_bytes = (try loadFixture(allocator, "fp8_test_inputs.f32")) orelse return error.SkipZigTest;
    defer allocator.free(inputs_bytes);
    const expected = (try loadFixture(allocator, "fp8_e5m2_encoded.u8")) orelse return error.SkipZigTest;
    defer allocator.free(expected);

    const inputs: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(inputs_bytes)));
    try std.testing.expectEqual(inputs.len, expected.len);

    const got = try allocator.alloc(u8, inputs.len);
    defer allocator.free(got);

    const W = Quantizer.fp8_vec_width;
    var i: usize = 0;
    while (i + W <= inputs.len) : (i += W) {
        const chunk: @Vector(W, f32) = inputs[i..][0..W].*;
        got[i..][0..W].* = @as([W]u8, Quantizer.f32_to_fp8_e5m2_chunk(chunk));
    }
    while (i < inputs.len) : (i += 1) {
        got[i] = Quantizer.f32_to_fp8_e5m2(inputs[i]);
    }

    const mismatches = checkEncodeResults(inputs, got, expected, "E5M2 SIMD");
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E4M3FN decode: LUT matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const expected_bytes = (try loadFixture(allocator, "fp8_e4m3fn_decode.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(@as(usize, 256), expected.len);

    var mismatches: usize = 0;
    for (expected, 0..) |exp_val, i| {
        const got = Quantizer.lut_e4m3[i];
        const both_nan = std.math.isNan(exp_val) and std.math.isNan(got);
        if (!both_nan and got != exp_val) {
            if (mismatches < 8) {
                std.debug.print("  E4M3FN LUT[0x{X:0>2}]: got={d:.6} expected={d:.6}\n", .{ i, got, exp_val });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E5M2 decode: LUT matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const expected_bytes = (try loadFixture(allocator, "fp8_e5m2_decode.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(@as(usize, 256), expected.len);

    var mismatches: usize = 0;
    for (expected, 0..) |exp_val, i| {
        const got = Quantizer.lut_e5m2[i];
        const both_nan = std.math.isNan(exp_val) and std.math.isNan(got);
        const both_inf = std.math.isInf(exp_val) and std.math.isInf(got) and
            std.math.signbit(exp_val) == std.math.signbit(got);
        if (!both_nan and !both_inf and got != exp_val) {
            if (mismatches < 8) {
                std.debug.print("  E5M2 LUT[0x{X:0>2}]: got={d:.6} expected={d:.6}\n", .{ i, got, exp_val });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E4M3FN scalar decode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const expected_bytes = (try loadFixture(allocator, "fp8_e4m3fn_decode.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(@as(usize, 256), expected.len);

    var mismatches: usize = 0;
    for (expected, 0..) |exp_val, i| {
        const got = Quantizer.fp8_e4m3_to_f32(@intCast(i));
        const both_nan = std.math.isNan(exp_val) and std.math.isNan(got);
        if (!both_nan and got != exp_val) {
            if (mismatches < 8) {
                std.debug.print("  E4M3FN scalar decode[0x{X:0>2}]: got={d:.6} expected={d:.6}\n", .{ i, got, exp_val });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "F8_E5M2 scalar decode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const expected_bytes = (try loadFixture(allocator, "fp8_e5m2_decode.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(@as(usize, 256), expected.len);

    var mismatches: usize = 0;
    for (expected, 0..) |exp_val, i| {
        const got = Quantizer.fp8_e5m2_to_f32(@intCast(i));
        const both_nan = std.math.isNan(exp_val) and std.math.isNan(got);
        const both_inf = std.math.isInf(exp_val) and std.math.isInf(got) and
            std.math.signbit(exp_val) == std.math.signbit(got);
        if (!both_nan and !both_inf and got != exp_val) {
            if (mismatches < 8) {
                std.debug.print("  E5M2 scalar decode[0x{X:0>2}]: got={d:.6} expected={d:.6}\n", .{ i, got, exp_val });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}
