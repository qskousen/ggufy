const std = @import("std");
const gguf = @import("Gguf.zig");
const types = @import("types.zig");
const ggml = @import("ggml.h");
const thread_pool_mod = @import("ThreadPool.zig");

pub const Quantizer = struct {
    // Main entry point: Source -> F32 -> Dest
    pub fn convertTensorData(
        allocator: std.mem.Allocator,
        src_data: []const u8,
        src_type: types.DataType,
        dst_type: types.DataType,
        element_count: u64,
        pool: *thread_pool_mod.ThreadPool,
    ) ![]u8 {
        return convertTensorDataWeighted(allocator, src_data, src_type, dst_type, element_count, pool, null);
    }

    /// The same, with per-column importance weights for the destination type's
    /// scale search. `imatrix` is one weight per column of a row, so its length
    /// is the row width and the k-quant encoders are handed whole rows.
    /// Unweighted callers keep the bit-exact path their golden fixtures pin.
    pub fn convertTensorDataWeighted(
        allocator: std.mem.Allocator,
        src_data: []const u8,
        src_type: types.DataType,
        dst_type: types.DataType,
        element_count: u64,
        pool: *thread_pool_mod.ThreadPool,
        imatrix: ?[]const f32,
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

        try quantizeFromF32(f32_buffer, out_buffer, dst_type, pool, imatrix);

        return out_buffer;
    }

    fn dequantizeToF32(
        input_bytes: []const u8,
        output_f32: []f32,
        src_type: types.DataType,
        pool: *thread_pool_mod.ThreadPool,
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
            .F4_E2M1, .MXFP4 => {
                if (input_bytes.len * 2 != output_f32.len)
                    return error.InputSizeMismatch;
                dequantizeFP4(input_bytes, output_f32, pool);
            },
            .mxfp4 => {
                // GGUF block format: [scale: E8M0 u8][qs[0..15]: u8×16] per 32 elements
                const n_blocks = input_bytes.len / 17;
                if (n_blocks * 32 != output_f32.len) return error.InputSizeMismatch;
                dequantizeMXFP4Gguf(input_bytes, output_f32, pool);
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
            else => {
                // Generic GGUF block-type dequantization via GGML type traits
                if (src_type.formatType() != .gguf) return error.UnsupportedSourceType;
                const gguf_type = gguf.GgmlType.fromString(@tagName(src_type)) catch
                    return error.UnsupportedSourceType;
                const expected_size: usize = @intCast(src_type.calcSizeInBytes(@intCast(output_f32.len)));
                if (input_bytes.len != expected_size) return error.InputSizeMismatch;
                const traits = ggml.ggml_get_type_traits(
                    @as(ggml.enum_ggml_type, @intCast(@intFromEnum(gguf_type))),
                );
                const to_float_fn = traits.*.to_float orelse return error.UnsupportedSourceType;
                to_float_fn(input_bytes.ptr, output_f32.ptr, @intCast(output_f32.len));
            },
        }
    }

    fn quantizeFromF32(
        input_f32: []const f32,
        output_bytes: []u8,
        dst_type: types.DataType,
        pool: *thread_pool_mod.ThreadPool,
        imatrix: ?[]const f32,
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
            .F4_E2M1, .MXFP4 => {
                if (output_bytes.len * 2 != input_f32.len)
                    return error.OutputBufferSizeMismatch;
                quantizeFP4(input_f32, output_bytes, pool);
            },
            .q8_0, .q5_0, .q4_0,
            .q5_1, .q4_1,
            .q6_k, .q5_k, .q4_k, .q3_k, .q2_k,
            // The IQ family goes through the same ggml entry point. Several of
            // their encoders assert on a missing imatrix (iq2_xxs, iq2_xs,
            // iq1_s, and q2_k's weighted impl), so a tensor the collector never
            // saw cannot be given one of those types - that aborts rather than
            // degrades.
            .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s,
            .iq1_s, .iq1_m, .iq4_nl, .iq4_xs,
            .mxfp4 => {
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
                    imatrix,
                );
            },
            else => return error.UnsupportedDestinationType,
        }
    }

    fn convertTypeGguf(
        input_f32: []const f32,
        output_bytes: []u8,
        pool: *thread_pool_mod.ThreadPool,
        q_type: gguf.GgmlType,
        block_elements: u64,
        block_size: u64,
        imatrix: ?[]const f32,
    ) !void {
        const element_count: u64 = @intCast(input_f32.len);
        const block_count = @divExact(element_count, block_elements);
        const threads_u64: u64 = @intCast(pool.threads.len);

        // Ensure output buffer is large enough
        if (output_bytes.len < block_count * block_size) return error.OutputBufferTooSmall;

        // The work splits into equal units, each handed to ggml as one call.
        //
        //   - Unweighted: a unit is one block. Blocks quantize independently, so
        //     this can ignore the tensor's real row structure - which matters,
        //     because ggufy blocks over the flat element count and some rows are
        //     not a whole number of blocks.
        //   - Weighted: a unit is a whole row. ggml indexes quant_weights by
        //     position within the row it was handed, so the weights only line up
        //     if it is told the true row width. Getting this wrong does not fail;
        //     it applies column 0's importance to every 256th weight and quietly
        //     produces a worse model than no imatrix at all.
        // A few encoders abort outright without weights rather than falling back,
        // so refuse here with a name the caller can report. Reaching ggml with
        // this combination kills the process mid-file.
        if (imatrix == null and ggml.ggml_quantize_requires_imatrix(@intCast(@intFromEnum(q_type)))) {
            return error.TypeRequiresImatrix;
        }
        const unit_elems: u64 = if (imatrix) |im| @intCast(im.len) else block_elements;
        if (unit_elems == 0) return error.InvalidImatrix;
        if (imatrix != null) {
            if (unit_elems % block_elements != 0) return error.ImatrixNotBlockAligned;
            if (element_count % unit_elems != 0) return error.ImatrixWidthMismatch;
        }
        const unit_bytes: u64 = (unit_elems / block_elements) * block_size;
        const units = @divExact(element_count, unit_elems);

        const units_per_thread = @divTrunc(units, threads_u64);
        const leftover = units - (units_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};

        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * units_per_thread;
            var end = start + units_per_thread;
            if (i == threads_u64 - 1) {
                end += leftover;
            }
            pool.spawnWg(&wg, processBlocks, .{ input_f32, output_bytes, start, end, unit_elems, unit_bytes, q_type, imatrix });
        }
        wg.wait();
    }

    fn processBlocks(
        input_f32: []const f32,
        output_bytes: []u8,
        start: u64,
        end: u64,
        unit_elems: u64,
        unit_bytes: u64,
        q_type: gguf.GgmlType,
        imatrix: ?[]const f32,
    ) void {
        const units = end - start;
        const unit_elems_usize: usize = @intCast(unit_elems);
        const unit_bytes_usize: usize = @intCast(unit_bytes);
        const src_offset: usize = @intCast(start * unit_elems);
        const dst_offset: usize = @intCast(start * unit_bytes);
        // The slices must span all `units` this worker owns: ggml writes through
        // the raw pointer, so bounds cut to one unit are a lie that a
        // bounds-checked API would reject.
        const src_block = input_f32[src_offset..][0 .. units * unit_elems_usize];
        const dst_block = output_bytes[dst_offset..][0 .. units * unit_bytes_usize];

        // Every unit in this call is the same width, so one weight vector serves
        // all of them - which is ggml's own contract for quant_weights.
        _ = ggml.ggml_quantize_chunk(
            @as(ggml.enum_ggml_type, @intCast(@intFromEnum(q_type))),
            src_block.ptr,
            dst_block.ptr,
            0,
            @intCast(units),
            @intCast(unit_elems),
            if (imatrix) |im| im.ptr else null,
        );
    }

    fn convertTypeSimple(
        input_f32: []const f32,
        output_bytes: []u8,
        pool: *thread_pool_mod.ThreadPool,
        dst_type: types.DataType,
    ) !void {
        const element_count = input_f32.len;
        const threads_count = @min(pool.threads.len, element_count);
        const elems_per_thread = element_count / threads_count;
        const leftover = element_count - (elems_per_thread * threads_count);

        var wg: thread_pool_mod.WaitGroup = .{};

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
        pool: *thread_pool_mod.ThreadPool,
        src_type: types.DataType,
    ) !void {
        const element_count = output_f32.len;
        const threads_count = @min(pool.threads.len, element_count);
        const elems_per_thread = element_count / threads_count;
        const leftover = element_count - (elems_per_thread * threads_count);

        var wg: thread_pool_mod.WaitGroup = .{};

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

    // -------------------------------------------------------------------------
    // E8M0: 8-bit unsigned exponent-only scale used in MX formats.
    // Value = 2^(x - 127); x=0 maps to the subnormal 2^-127; x=255 is NaN.
    // -------------------------------------------------------------------------

    pub fn e8m0_to_f32(x: u8) f32 {
        if (x == 0) return @bitCast(@as(u32, 0x0040_0000)); // 2^-127 as f32 subnormal
        if (x == 255) return std.math.nan(f32); // E8M0: x=255 is NaN
        return @bitCast(@as(u32, x) << 23);
    }

    /// Encode an f32 value to E8M0 (8-bit exponent-only) format.
    /// Extracts the f32 biased exponent (range 0-255) and returns it directly.
    /// Special cases: NaN/Inf → 255, zero/subnormal → 0.
    pub fn f32_to_e8m0(x: f32) u8 {
        const bits: u32 = @bitCast(x);
        const abs_bits: u32 = bits & 0x7FFF_FFFF;

        // Zero → 0
        if (abs_bits == 0) return 0;
        // Extract biased exponent (bits 23-30)
        const biased_exp: u32 = (abs_bits >> 23) & 0xFF;

        // NaN or Inf (exp == 255) → 255
        // Subnormal (exp == 0) → 0
        // Normal → biased_exp
        if (biased_exp == 0) return 0;
        if (biased_exp == 255) return 255;

        return @truncate(biased_exp);
    }

    // -------------------------------------------------------------------------
    // FP4 / E2M1: 1 sign | 2 exp (bias=1) | 1 mantissa, 2 nibbles/byte.
    // Positive values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}.
    // Used as the element type for FP4, NV FP4, and MX FP4; block-level scaling
    // (if any) is stored externally and is not part of this element encoding.
    // Packing: element[2i] in low nibble, element[2i+1] in high nibble.
    // -------------------------------------------------------------------------

    pub const lut_fp4_e2m1: [16]f32 = blk: {
        var t: [16]f32 = undefined;
        var i: u32 = 0;
        while (i < 16) : (i += 1) t[i] = fp4_e2m1_to_f32(@intCast(i));
        break :blk t;
    };

    pub fn fp4_e2m1_to_f32(nibble: u4) f32 {
        const positives = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };
        const sign: f32 = if ((nibble >> 3) != 0) -1.0 else 1.0;
        return sign * positives[nibble & 0x7];
    }

    pub fn f32_to_fp4_e2m1(x: f32) u4 {
        const bits: u32 = @bitCast(x);
        const sign: u4 = @truncate(bits >> 31);
        const abs_bits: u32 = bits & 0x7FFF_FFFF;
        // NaN/Inf → saturate to max magnitude (6.0)
        if (abs_bits >= 0x7F80_0000) return (sign << 3) | 0x7;
        const abs: f32 = @bitCast(abs_bits);
        // Round-to-nearest-even over the 8 representable magnitudes.
        // At midpoints, even code (0,2,4,6) wins: use <= for even-lower, < for odd-lower.
        const code: u4 = if (abs <= 0.25) 0
            else if (abs < 0.75) 1
            else if (abs <= 1.25) 2
            else if (abs < 1.75) 3
            else if (abs <= 2.5) 4
            else if (abs < 3.5) 5
            else if (abs <= 5.0) 6
            else 7;
        return (sign << 3) | code;
    }

    fn dequantizeFP4(input_bytes: []const u8, output_f32: []f32, pool: *thread_pool_mod.ThreadPool) void {
        const element_count = output_f32.len;
        if (element_count == 0) return;
        const threads_count = @min(pool.threads.len, element_count);
        // Round chunk size up to even so byte boundaries don't straddle threads.
        const raw_per = element_count / threads_count;
        const elems_per_thread = @max(2, (raw_per + 1) & ~@as(usize, 1));
        var wg: thread_pool_mod.WaitGroup = .{};
        var start: usize = 0;
        while (start < element_count) : (start += elems_per_thread) {
            const end = @min(start + elems_per_thread, element_count);
            pool.spawnWg(&wg, processDequantizeFP4, .{ input_bytes, output_f32, start, end });
        }
        wg.wait();
    }

    fn processDequantizeFP4(input_bytes: []const u8, output_f32: []f32, start: usize, end: usize) void {
        var i: usize = start;
        while (i + 1 < end) : (i += 2) {
            const byte = input_bytes[i / 2];
            output_f32[i] = lut_fp4_e2m1[byte & 0xF];
            output_f32[i + 1] = lut_fp4_e2m1[byte >> 4];
        }
        if (i < end) output_f32[i] = lut_fp4_e2m1[input_bytes[i / 2] & 0xF];
    }

    fn quantizeFP4(input_f32: []const f32, output_bytes: []u8, pool: *thread_pool_mod.ThreadPool) void {
        const element_count = input_f32.len;
        if (element_count == 0) return;
        const threads_count = @min(pool.threads.len, element_count);
        const raw_per = element_count / threads_count;
        const elems_per_thread = @max(2, (raw_per + 1) & ~@as(usize, 1));
        var wg: thread_pool_mod.WaitGroup = .{};
        var start: usize = 0;
        while (start < element_count) : (start += elems_per_thread) {
            const end = @min(start + elems_per_thread, element_count);
            pool.spawnWg(&wg, processQuantizeFP4, .{ input_f32, output_bytes, start, end });
        }
        wg.wait();
    }

    fn processQuantizeFP4(input_f32: []const f32, output_bytes: []u8, start: usize, end: usize) void {
        var i: usize = start;
        while (i + 1 < end) : (i += 2) {
            const lo: u8 = @as(u8, f32_to_fp4_e2m1(input_f32[i]));
            const hi: u8 = @as(u8, f32_to_fp4_e2m1(input_f32[i + 1]));
            output_bytes[i / 2] = (hi << 4) | lo;
        }
        if (i < end) output_bytes[i / 2] = @as(u8, f32_to_fp4_e2m1(input_f32[i]));
    }

    // -------------------------------------------------------------------------
    // ComfyUI FP8 cluster quantization.
    // weight: F8_E4M3 elements; weight_scale: single F32 global scalar.
    // -------------------------------------------------------------------------

    pub const ComfyFp8Data = struct { weight: []u8, scale: f32 };

    /// Quantize F32 input to ComfyUI FP8 cluster format.
    /// Computes a global scalar scale = amax / 448.0 and converts elements to F8_E4M3.
    /// Caller owns the returned weight slice.
    pub fn quantizeToComfyFp8(
        allocator: std.mem.Allocator,
        input: []const f32,
        pool: *thread_pool_mod.ThreadPool,
    ) !ComfyFp8Data {
        var amax: f32 = 0.0;
        for (input) |v| amax = @max(amax, @abs(v));

        const fp8_max: f32 = 448.0;
        const scale: f32 = if (amax > 0.0) amax / fp8_max else 1.0;
        const inv_scale = 1.0 / scale;

        // Pre-scale so that the F8 conversion round-trips correctly via *scale
        const scaled = try allocator.alloc(f32, input.len);
        defer allocator.free(scaled);
        for (input, 0..) |x, i| scaled[i] = x * inv_scale;

        const weight = try allocator.alloc(u8, input.len);
        errdefer allocator.free(weight);
        try convertTypeSimple(scaled, weight, pool, .F8_E4M3);

        return .{ .weight = weight, .scale = scale };
    }

    // -------------------------------------------------------------------------
    // ConvRot INT8 (ComfyUI "int8_tensorwise" with convrot + per_row).
    //
    // Weights are rotated by a normalized *regular* Hadamard matrix (a Kronecker
    // power of H4) in groups along the input dimension, then per-row INT8 quantized.
    // The rotation spreads per-channel outliers within each group, tightening the
    // per-row dynamic range and cutting quantization error. Because the matrix is
    // symmetric and orthogonal (H @ H = I), the same transform both applies the
    // rotation (quantize) and undoes it (dequantize).
    //
    // A dense group matmul would cost rows*cols*group_size FLOPs. Since the matrix
    // is H4^{⊗k}, we instead use the fast radix-4 Hadamard transform (O(N·log₄N)),
    // which computes the identical linear map. `buildHadamard` returns the dense
    // matrix and exists only as a reference for validating the fast transform.
    // -------------------------------------------------------------------------

    /// Regular order-4 Hadamard (symmetric, entries ±1), row-major.
    /// H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] — matches comfy_kitchen.
    const h4_raw = [16]f32{ 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1 };

    /// True for sizes that are a power of 4 and ≥ 4 (valid regular-Hadamard orders).
    pub fn isValidHadamardSize(size: usize) bool {
        return size >= 4 and (size & (size - 1)) == 0 and (@ctz(size) & 1) == 0;
    }

    /// Build a normalized regular Hadamard matrix of the given power-of-4 `size`,
    /// row-major [size*size]. Entries are ±1/√size. Reference implementation used to
    /// validate `hadamardTransformInPlace`; the hot path uses the fast transform.
    /// Caller owns the returned slice.
    pub fn buildHadamard(allocator: std.mem.Allocator, size: usize) ![]f32 {
        if (!isValidHadamardSize(size)) return error.InvalidHadamardSize;

        var cur: usize = 4;
        var h = try allocator.alloc(f32, 16);
        @memcpy(h, &h4_raw);
        errdefer allocator.free(h);

        while (cur < size) {
            const next = cur * 4;
            const nh = try allocator.alloc(f32, next * next);
            // nh = kron(h, H4): nh[(ra*4+rb), (ca*4+cb)] = h[ra,ca] * H4[rb,cb]
            for (0..cur) |ra| {
                for (0..cur) |ca| {
                    const a = h[ra * cur + ca];
                    for (0..4) |rb| {
                        for (0..4) |cb| {
                            nh[(ra * 4 + rb) * next + (ca * 4 + cb)] = a * h4_raw[rb * 4 + cb];
                        }
                    }
                }
            }
            allocator.free(h);
            h = nh;
            cur = next;
        }

        const norm = 1.0 / @sqrt(@as(f32, @floatFromInt(size)));
        for (h) |*v| v.* *= norm;
        return h;
    }

    /// Apply the normalized regular Hadamard transform to `v` in place.
    /// `v.len` must be a power of 4. Radix-4 butterfly (iterative, natural order);
    /// computes exactly H4^{⊗k} @ v then scales by 1/√len. Its own inverse.
    pub fn hadamardTransformInPlace(v: []f32) void {
        const n = v.len;
        var h: usize = 1;
        while (h < n) : (h *= 4) {
            var i: usize = 0;
            while (i < n) : (i += h * 4) {
                var j = i;
                while (j < i + h) : (j += 1) {
                    const a = v[j];
                    const b = v[j + h];
                    const c = v[j + 2 * h];
                    const d = v[j + 3 * h];
                    v[j] = a + b + c - d;
                    v[j + h] = a + b - c + d;
                    v[j + 2 * h] = a - b + c + d;
                    v[j + 3 * h] = -a + b + c + d;
                }
            }
        }
        const norm = 1.0 / @sqrt(@as(f32, @floatFromInt(n)));
        for (v) |*x| x.* *= norm;
    }

    fn rotateGroupwiseRows(
        buf: []f32,
        cols: usize,
        group_size: usize,
        start_row: usize,
        end_row: usize,
    ) void {
        const n_groups = cols / group_size;
        for (start_row..end_row) |row| {
            for (0..n_groups) |g| {
                const base = row * cols + g * group_size;
                hadamardTransformInPlace(buf[base .. base + group_size]);
            }
        }
    }

    /// Rotate a [rows*cols] matrix in place: apply the Hadamard transform to each
    /// contiguous group of `group_size` elements along the column (input) dimension.
    /// Serves both directions (quantize rotate / dequantize un-rotate). Threaded over rows.
    pub fn rotateGroupwiseInPlace(
        buf: []f32,
        rows: usize,
        cols: usize,
        group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
    ) !void {
        if (!isValidHadamardSize(group_size)) return error.InvalidHadamardSize;
        if (cols % group_size != 0) return error.ColsNotDivisibleByGroupSize;

        const threads_u64: u64 = @intCast(pool.threads.len);
        const rows_u64: u64 = @intCast(rows);
        const rows_per_thread = @divTrunc(rows_u64, threads_u64);
        const leftover = rows_u64 - (rows_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * rows_per_thread;
            var end = start + rows_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            if (start == end) continue;
            pool.spawnWg(&wg, rotateGroupwiseRows, .{ buf, cols, group_size, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();
    }

    /// Round half-to-even (banker's rounding), matching torch's `.round()`.
    pub fn roundHalfToEven(x: f32) f32 {
        const fl = @floor(x);
        const diff = x - fl;
        if (diff < 0.5) return fl;
        if (diff > 0.5) return fl + 1.0;
        return if (@mod(fl, 2.0) == 0.0) fl else fl + 1.0;
    }

    /// splitmix64 finalizer — a strong 64→64-bit hash used to derive per-element
    /// randomness for stochastic rounding independently of thread scheduling.
    fn splitmix64(x: u64) u64 {
        var z = x +% 0x9E3779B97F4A7C15;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    /// Reproducible uniform draw in [0, 1) for element `idx` under `seed`. Hashing the
    /// index before combining with the seed decorrelates neighbouring elements, so the
    /// result depends only on (seed, idx) — never on how rows are split across threads.
    fn stochasticUniform(seed: u64, idx: u64) f32 {
        const h = splitmix64(seed ^ splitmix64(idx));
        // Top 24 bits → an f32 in [0, 1) with full mantissa precision.
        return @as(f32, @floatFromInt(h >> 40)) * 0x1p-24;
    }

    pub const ConvrotInt8Data = struct {
        weight: []u8, // int8 bit patterns, [rows*cols]
        scale: []f32, // per-row scale, [rows]
    };

    /// Quantize a [rows*cols] F32 matrix to ComfyUI int8_tensorwise (per-row) INT8.
    /// When `convrot`, first rotate group-wise with the Hadamard transform (cols must be
    /// divisible by `group_size`, a power of 4). Matches comfy_kitchen's quantize path:
    ///   scale[r] = max(amax(row[r]) / 127, 1e-30)
    ///   q[r,c]   = clamp(round_half_even(row[r,c] / scale[r]), -128, 127)
    /// Caller owns both slices.
    pub fn quantizeToInt8(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot: bool,
        group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
    ) !ConvrotInt8Data {
        if (input.len != rows * cols) return error.InputSizeMismatch;

        // Rotation is in-place, so it needs a mutable copy; plain int8 reads the input directly.
        var rotated: []f32 = &.{};
        defer if (convrot) allocator.free(rotated);
        if (convrot) {
            rotated = try allocator.alloc(f32, input.len);
            @memcpy(rotated, input);
            try rotateGroupwiseInPlace(rotated, rows, cols, group_size, pool);
        }
        const work: []const f32 = if (convrot) rotated else input;

        const weight = try allocator.alloc(u8, rows * cols);
        errdefer allocator.free(weight);
        const scale = try allocator.alloc(f32, rows);
        errdefer allocator.free(scale);

        // Rows are independent (each carries its own scale) — quantize them in parallel.
        const threads_u64: u64 = @intCast(pool.threads.len);
        const rows_u64: u64 = @intCast(rows);
        const rows_per_thread = @divTrunc(rows_u64, threads_u64);
        const leftover = rows_u64 - (rows_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * rows_per_thread;
            var end = start + rows_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            if (start == end) continue;
            pool.spawnWg(&wg, quantizeInt8Rows, .{ work, weight, scale, cols, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();

        return .{ .weight = weight, .scale = scale };
    }

    fn quantizeInt8Rows(
        work: []const f32,
        weight: []u8,
        scale: []f32,
        cols: usize,
        start_row: usize,
        end_row: usize,
    ) void {
        for (start_row..end_row) |r| {
            const row = work[r * cols .. r * cols + cols];
            var amax: f32 = 0.0;
            for (row) |v| {
                if (!std.math.isNan(v) and !std.math.isInf(v)) amax = @max(amax, @abs(v));
            }
            const s: f32 = @max(amax / 127.0, 1e-30);
            scale[r] = s;
            for (row, 0..) |v, c| {
                // True division (not multiply-by-reciprocal) to match torch's x/scale bit-for-bit.
                const q = std.math.clamp(roundHalfToEven(v / s), -128.0, 127.0);
                weight[r * cols + c] = @bitCast(@as(i8, @intFromFloat(q)));
            }
        }
    }

    /// Convenience wrapper: ConvRot INT8 (always rotates). See `quantizeToInt8`.
    pub fn quantizeToConvrotInt8(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
    ) !ConvrotInt8Data {
        return quantizeToInt8(allocator, input, rows, cols, true, group_size, pool);
    }

    // -------------------------------------------------------------------------
    // int4 ConvRot (ComfyUI "convrot_w4a4").
    //
    // Symmetric per-row signed 4-bit weight, Hadamard-rotated group-wise before
    // quantization, packed two nibbles per byte along the column dimension. Matches
    // comfy_kitchen's quantize_convrot_w4a4_weight:
    //   scale[r] = max(amax(row[r]) / 7, 1e-30)               (per output row)
    //   q[r,c]   = clamp(round(row[r,c] / scale[r]), -7, 7)   (symmetric; -8 not emitted)
    // Packing: element 2k → low nibble of byte k, element 2k+1 → high nibble; each nibble
    // is the value's two's-complement low 4 bits (identical to _pack_int4_row_major).
    //
    // `stochastic_rounding` is a seed: 0 selects deterministic round-half-to-even, which is
    // bit-compatible with comfy_kitchen's default (stochastic_rounding=0) and is the
    // validation contract. Any nonzero value enables stochastic rounding — round(x) becomes
    // floor(x + u), u ~ U[0,1) from a reproducible per-element PRNG keyed by (seed, index).
    // This mirrors comfy_kitchen's stochastic path but uses ggufy's own RNG, so nonzero
    // seeds are NOT bit-compatible with torch (statistically equivalent quality only).
    // `cols` must be even. Caller owns both slices.
    // -------------------------------------------------------------------------

    pub const Int4Data = struct {
        weight: []u8, // nibble-packed signed 4-bit values, [rows * cols / 2]
        scale: []f32, // per-row scale, [rows]
    };

    pub fn quantizeToInt4(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot: bool,
        group_size: usize,
        stochastic_rounding: u64,
        pool: *thread_pool_mod.ThreadPool,
    ) !Int4Data {
        if (input.len != rows * cols) return error.InputSizeMismatch;
        if (cols % 2 != 0) return error.ColsNotEven;

        // Rotation is in-place, so it needs a mutable copy; plain int4 reads the input directly.
        var rotated: []f32 = &.{};
        defer if (convrot) allocator.free(rotated);
        if (convrot) {
            rotated = try allocator.alloc(f32, input.len);
            @memcpy(rotated, input);
            try rotateGroupwiseInPlace(rotated, rows, cols, group_size, pool);
        }
        const work: []const f32 = if (convrot) rotated else input;

        const weight = try allocator.alloc(u8, rows * (cols / 2));
        errdefer allocator.free(weight);
        const scale = try allocator.alloc(f32, rows);
        errdefer allocator.free(scale);

        // Rows are independent (each carries its own scale) — quantize them in parallel.
        const threads_u64: u64 = @intCast(pool.threads.len);
        const rows_u64: u64 = @intCast(rows);
        const rows_per_thread = @divTrunc(rows_u64, threads_u64);
        const leftover = rows_u64 - (rows_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * rows_per_thread;
            var end = start + rows_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            if (start == end) continue;
            pool.spawnWg(&wg, quantizeInt4Rows, .{ work, weight, scale, cols, stochastic_rounding, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();

        return .{ .weight = weight, .scale = scale };
    }

    fn quantizeInt4Rows(
        work: []const f32,
        weight: []u8,
        scale: []f32,
        cols: usize,
        stochastic_rounding: u64,
        start_row: usize,
        end_row: usize,
    ) void {
        const packed_cols = cols / 2;
        for (start_row..end_row) |r| {
            const row = work[r * cols .. r * cols + cols];
            var amax: f32 = 0.0;
            for (row) |v| {
                if (!std.math.isNan(v) and !std.math.isInf(v)) amax = @max(amax, @abs(v));
            }
            const s: f32 = @max(amax / 7.0, 1e-30);
            scale[r] = s;
            const row_base = r * cols; // flat element index of column 0, for the per-element RNG
            for (0..packed_cols) |pc| {
                const lo = quantizeInt4Nibble(row[2 * pc], s, stochastic_rounding, row_base + 2 * pc);
                const hi = quantizeInt4Nibble(row[2 * pc + 1], s, stochastic_rounding, row_base + 2 * pc + 1);
                weight[r * packed_cols + pc] = lo | (hi << 4);
            }
        }
    }

    /// Quantize one value to a signed-4-bit nibble (two's-complement low 4 bits).
    /// `seed` 0 → deterministic round-half-to-even; nonzero → stochastic rounding via a
    /// reproducible per-element draw keyed by (seed, idx). Clamp is [-7, 7] (symmetric,
    /// matching comfy_kitchen's _INT4_MAX contract — -8 is representable but never emitted).
    fn quantizeInt4Nibble(v: f32, s: f32, seed: u64, idx: u64) u8 {
        // True division (not multiply-by-reciprocal) to match torch's x/scale bit-for-bit.
        const scaled = v / s;
        const rounded = if (seed == 0)
            roundHalfToEven(scaled)
        else
            @floor(scaled + stochasticUniform(seed, idx));
        const q = std.math.clamp(rounded, -7.0, 7.0);
        return @as(u8, @bitCast(@as(i8, @intFromFloat(q)))) & 0x0F;
    }

    /// Convenience wrapper: ConvRot int4 (always rotates). See `quantizeToInt4`.
    pub fn quantizeToConvrotInt4(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        group_size: usize,
        stochastic_rounding: u64,
        pool: *thread_pool_mod.ThreadPool,
    ) !Int4Data {
        return quantizeToInt4(allocator, input, rows, cols, true, group_size, stochastic_rounding, pool);
    }

    // asym_w4a8_int8 (ComfyUI AsymW4A8Int8Layout).
    //
    // 4 bits of storage, 8 bits of arithmetic. Where convrot_w4a4 quantizes weights and
    // activations to 4 bits, this format decodes each 4-bit code to a full int8 value at
    // load time, so the matrix multiply is int8 against an int8 activation. That is why the
    // decode rounds to whole numbers instead of reconstructing a float.
    //
    // Matches comfy_kitchen quantize_w4a8_int8_weight in symmetric codebook mode, applied to
    // the rotated weight, per group of w4a8_group_size columns:
    //
    //   group_scale = max(amax(group), 1e-8)
    //   code        = nearest(codebook, w / group_scale)
    //   twice:
    //       group_scale = max(dot(w, c) / max(dot(c, c), 1e-8), 1e-8)   with c = codebook[code]
    //       code        = nearest(codebook, w / group_scale)
    //   s_channel   = max(amax_row(codebook[code] * group_scale) / 127, 1e-8)
    //   s_rel       = fp8_e4m3(group_scale / s_channel)
    //   levels[j]   = clamp(round(codebook[j] * s_rel), -127, 127)
    //   code        = nearest(levels, w / s_channel)
    //
    // Ties go to the lower index in both assignment steps. cols must be even and a multiple
    // of the group size. Caller owns every returned slice.
    //
    // Output is not byte for byte the same as comfy_kitchen on a bf16 checkpoint. It builds
    // the rotation matrix in the weight's own dtype, so a bf16 weight rotates in bf16. This
    // code rotates in f32, which is slightly more accurate and moves a few percent of the
    // s_rel bytes by one step.

    /// ComfyUI's default W4A8 scale group, the group_size field of the comfy_quant JSON.
    pub const w4a8_group_size: usize = 16;

    /// The 16 levels comfy_kitchen uses, copied exactly. They are Lloyd-Max optimal for a
    /// Gaussian, which is what the rotation makes each group look like, so one fixed table
    /// works about as well as fitting one per tensor.
    ///
    /// comfy_kitchen fits its own table when a group is heavy-tailed instead of Gaussian.
    /// This code always writes the fixed table and warns when that test would have tripped.
    /// The table is stored in the file, so decoding is correct either way; only byte equality
    /// with comfy_kitchen's output is lost on such a tensor.
    pub const w4a8_codebook = [16]f32{
        -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
        0.052541,  0.156985,  0.265284,  0.379533,  0.502636,  0.638953,  0.794876,  0.980671,
    };

    /// Excess kurtosis above which comfy_kitchen fits its own codebook instead of the fixed one.
    const w4a8_gate_kurtosis: f64 = -0.1;

    pub const AsymW4a8Data = struct {
        weight: []u8, // packed 4-bit codebook indices, [rows * cols / 2]
        s_rel: []u8, // per-group scale, raw fp8 e4m3 bytes, [rows * cols / group_size]
        s_channel: []f32, // per-output-row scale, [rows]
        /// True when comfy_kitchen would have fitted its own codebook for this weight.
        codebook_fit_recommended: bool,
    };

    pub fn quantizeToAsymW4a8(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot_group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
    ) !AsymW4a8Data {
        if (input.len != rows * cols) return error.InputSizeMismatch;
        if (cols % w4a8_group_size != 0) return error.ColsNotDivisibleByGroupSize;
        if (cols % convrot_group_size != 0) return error.ColsNotDivisibleByGroupSize;

        // Rotation is in-place, so it needs a mutable copy. This format is always rotated.
        const rotated = try allocator.alloc(f32, input.len);
        defer allocator.free(rotated);
        @memcpy(rotated, input);
        try rotateGroupwiseInPlace(rotated, rows, cols, convrot_group_size, pool);

        const groups_per_row = cols / w4a8_group_size;

        const weight = try allocator.alloc(u8, rows * (cols / 2));
        errdefer allocator.free(weight);
        const s_rel = try allocator.alloc(u8, rows * groups_per_row);
        errdefer allocator.free(s_rel);
        const s_channel = try allocator.alloc(f32, rows);
        errdefer allocator.free(s_channel);

        // Scratch for the settled per-group scales, so the workers allocate nothing.
        const group_scales = try allocator.alloc(f32, rows * groups_per_row);
        defer allocator.free(group_scales);

        // Each row has its own s_channel, so rows are independent and the split over threads
        // does not change the result.
        const threads_u64: u64 = @intCast(pool.threads.len);
        const rows_u64: u64 = @intCast(rows);
        const rows_per_thread = @divTrunc(rows_u64, threads_u64);
        const leftover = rows_u64 - (rows_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * rows_per_thread;
            var end = start + rows_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            if (start == end) continue;
            pool.spawnWg(&wg, quantizeAsymW4a8Rows, .{ rotated, weight, s_rel, s_channel, group_scales, cols, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();

        return .{
            .weight = weight,
            .s_rel = s_rel,
            .s_channel = s_channel,
            .codebook_fit_recommended = w4a8CodebookFitRecommended(rotated, cols),
        };
    }

    /// Elements sampled by the kurtosis check. comfy_kitchen samples the same count randomly;
    /// an evenly spaced sample estimates the same shape.
    const w4a8_gate_sample_elems: usize = 1 << 19;

    /// Advisory check: would comfy_kitchen have fitted a per-tensor codebook for this
    /// (already rotated) weight?
    fn w4a8CodebookFitRecommended(rotated: []const f32, cols: usize) bool {
        const n_groups = rotated.len / w4a8_group_size;
        if (n_groups == 0 or cols == 0) return false;

        // Stride whole groups so each sampled value is normalized by its own group's amax,
        // and so the sample spans the tensor rather than its first rows.
        const want_groups = @max(1, w4a8_gate_sample_elems / w4a8_group_size);
        const stride = @max(1, n_groups / want_groups);

        var sum: f64 = 0.0;
        var count: f64 = 0.0;
        var m2: f64 = 0.0;
        var m4: f64 = 0.0;

        // Two passes over the SAMPLE (mean, then central moments), not over the tensor.
        var g: usize = 0;
        while (g < n_groups) : (g += stride) {
            const group = rotated[g * w4a8_group_size ..][0..w4a8_group_size];
            var amax: f32 = 0.0;
            for (group) |v| {
                if (!std.math.isNan(v) and !std.math.isInf(v)) amax = @max(amax, @abs(v));
            }
            const gs: f32 = @max(amax, 1e-8);
            for (group) |v| {
                sum += @as(f64, v / gs);
                count += 1.0;
            }
        }
        if (count < 2.0) return false;
        const mean = sum / count;

        g = 0;
        while (g < n_groups) : (g += stride) {
            const group = rotated[g * w4a8_group_size ..][0..w4a8_group_size];
            var amax: f32 = 0.0;
            for (group) |v| {
                if (!std.math.isNan(v) and !std.math.isInf(v)) amax = @max(amax, @abs(v));
            }
            const gs: f32 = @max(amax, 1e-8);
            for (group) |v| {
                const d = @as(f64, v / gs) - mean;
                m2 += d * d;
                m4 += d * d * d * d;
            }
        }
        // torch's .std() is the sample (n-1) estimator; comfy divides by it before the
        // 4th moment, so match that rather than the population form.
        const variance = m2 / (count - 1.0);
        const sd = @sqrt(variance) + 1e-9;
        const excess = (m4 / count) / (sd * sd * sd * sd) - 3.0;
        return excess > w4a8_gate_kurtosis;
    }

    /// One scale group as a single vector. Every per-element step of the sweep does the same
    /// thing to all 16 lanes, so the group is the natural unit to vectorize over.
    const GroupVec = @Vector(w4a8_group_size, f32);
    const GroupIdx = @Vector(w4a8_group_size, u8);

    /// Halfway points between neighbouring levels, which are the boundaries a nearest-level
    /// assignment decides on. The nearest level is then just the count of boundaries below
    /// the value, so this replaces a lower bound search plus a distance comparison with one
    /// count. Ties still fall to the lower level, since a value sitting exactly on a boundary
    /// is not below it.
    ///
    /// This matches comfy_kitchen's searchsorted-then-closer-neighbour rule in exact arithmetic
    /// but not always in f32, because averaging two levels rounds. A value within one ulp of a
    /// boundary can land on either side, which changes about one code in a million by a single
    /// level and does not measurably change the error.
    fn levelMidpoints(levels: [16]f32) [15]f32 {
        var mids: [15]f32 = undefined;
        for (0..15) |j| mids[j] = (levels[j] + levels[j + 1]) * 0.5;
        return mids;
    }

    /// Boundaries for the fixed codebook. Same for every group, so computed once.
    const w4a8_codebook_mids: [15]f32 = levelMidpoints(w4a8_codebook);

    /// Assign every lane to its nearest level, given that table's boundaries, which must be
    /// finite. Same result as counting `mid < val` with vector compares, which a Debug build
    /// runs about 5x slower: the sign bit of mid - val decides it, since distinct floats never
    /// subtract to zero, the + 0.0 turns a -0 mid into +0 so equal values give +0, and a NaN
    /// lane is forced to 0 as a compare would leave it.
    inline fn assignGroup(vals: GroupVec, mids: [15]f32) GroupIdx {
        const U = @Vector(w4a8_group_size, u32);
        var count: U = @splat(0);
        inline for (0..15) |j| {
            const d = @as(GroupVec, @splat(mids[j] + 0.0)) - vals;
            count +%= @as(U, @bitCast(d)) >> @splat(31);
        }
        const bits: U = @bitCast(vals);
        const not_nan = ((bits & @as(U, @splat(0x7FFF_FFFF))) -% @as(U, @splat(0x7F80_0001))) >> @splat(31);
        return @truncate(count *% not_nan);
    }

    /// Largest finite magnitude in the group. Max does not depend on the order it is taken
    /// in, so this matches a scalar loop exactly. NaN and infinity are masked to zero first;
    /// both compare false against a finite bound, so one comparison covers them.
    inline fn groupAmax(gv: GroupVec) f32 {
        const av = @abs(gv);
        const finite = av < @as(GroupVec, @splat(std.math.floatMax(f32)));
        return @reduce(.Max, @select(f32, finite, av, @as(GroupVec, @splat(0.0))));
    }

    /// Quantize a range of rows. Two passes, because every group's s_rel is stored relative
    /// to the row's s_channel, and s_channel is not known until every group has been seen.
    /// group_scales is scratch owned by the caller, one f32 per group.
    fn quantizeAsymW4a8Rows(
        rotated: []const f32,
        weight: []u8,
        s_rel: []u8,
        s_channel: []f32,
        group_scales: []f32,
        cols: usize,
        start_row: usize,
        end_row: usize,
    ) void {
        const gs_n = w4a8_group_size;
        const groups_per_row = cols / gs_n;
        const packed_cols = cols / 2;
        const cb = w4a8_codebook;

        // The pass 2 grid depends only on the stored s_rel byte, so build all 256 once.
        var mids_by_s_rel: [256][15]f32 = undefined;
        for (&mids_by_s_rel, 0..) |*mids, b| {
            const s_rel_dec = fp8_e4m3_to_f32(@intCast(b));
            var levels: [16]f32 = undefined;
            for (cb, 0..) |c, j| {
                levels[j] = std.math.clamp(roundHalfToEven(c * s_rel_dec), -127.0, 127.0);
            }
            mids.* = levelMidpoints(levels);
        }

        for (start_row..end_row) |r| {
            const row = rotated[r * cols ..][0..cols];
            const row_scales = group_scales[r * groups_per_row ..][0..groups_per_row];

            // Pass 1: settle each group's scale and track the row's peak decoded magnitude,
            // which sets s_channel.
            var max_shifted: f32 = 0.0;

            for (0..groups_per_row) |g| {
                const grp = row[g * gs_n ..][0..gs_n];
                const gv: GroupVec = grp.*;

                var gs: f32 = @max(groupAmax(gv), 1e-8);
                // Dividing a vector does the same thing to each lane as dividing one value,
                // so this matches the reference. It is not a multiply by a reciprocal.
                var idx: [gs_n]u8 = assignGroup(gv / @as(GroupVec, @splat(gs)), w4a8_codebook_mids);

                // Two refit passes: fit the scale to the levels just chosen, then choose
                // levels again. comfy_kitchen uses two.
                //
                // The sums stay scalar and in lane order. Adding f32 in a different order
                // gives a different total, which would change gs, and gs decides the stored
                // s_rel byte.
                for (0..2) |_| {
                    var num: f32 = 0.0;
                    var den: f32 = 0.0;
                    for (grp, 0..) |v, j| {
                        const c = cb[idx[j]];
                        num += v * c;
                        den += c * c;
                    }
                    gs = @max(num / @max(den, 1e-8), 1e-8);
                    idx = assignGroup(gv / @as(GroupVec, @splat(gs)), w4a8_codebook_mids);
                }

                row_scales[g] = gs;
                // s_channel is taken from the magnitudes here, before the final assignment
                // against the whole number grid. The order matters.
                for (idx) |j| max_shifted = @max(max_shifted, @abs(cb[j] * gs));
            }

            const sc: f32 = @max(max_shifted / 127.0, 1e-8);
            s_channel[r] = sc;

            // Pass 2: store s_rel as fp8, rebuild the level grid, assign, and pack.
            for (0..groups_per_row) |g| {
                const gv: GroupVec = row[g * gs_n ..][0..gs_n].*;

                // The grid comes from the fp8 byte, not the f32 scale, so it is what a loader sees.
                const s_rel_byte = f32_to_fp8_e4m3(row_scales[g] / sc);
                s_rel[r * groups_per_row + g] = s_rel_byte;

                const idx: [gs_n]u8 = assignGroup(gv / @as(GroupVec, @splat(sc)), mids_by_s_rel[s_rel_byte]);

                // Element 2k goes in the low nibble of byte k, 2k+1 in the high nibble. The
                // nibble is an unsigned codebook index from 0 to 15, so no sign handling.
                const base = g * gs_n;
                for (0..gs_n / 2) |p| {
                    const flat = base + 2 * p;
                    weight[r * packed_cols + flat / 2] = idx[2 * p] | (idx[2 * p + 1] << 4);
                }
            }
        }
    }

    // asym_w4a8_int8 in its asymmetric form: uniform codes over each group's own [min, max],
    // and a per-group correction that puts the zero point back. Matches comfy_kitchen
    // quantize_w4a8_int8_weight(symmetric=False, codebook=False) on the rotated weight:
    //
    //   group_scale = max((max(group) - min(group)) / 15, 1e-8)
    //   code        = clamp(round((w - min(group)) / group_scale), 0, 15)
    //   s_channel   = max(amax_row((code - 8) * group_scale) / 127, 1e-8)
    //   s_rel       = fp8_e4m3(group_scale / s_channel)
    //   correction  = 8 * group_scale + min(group)
    //
    // The codes are not reassigned against the rounded int8 grid, unlike the codebook form.
    // correction is laid out [group][row], as ComfyUI stores it.

    pub const AsymW4a8ZpData = struct {
        weight: []u8, // packed 4-bit codes, [rows * cols / 2]
        s_rel: []u8, // per-group scale, raw fp8 e4m3 bytes, [rows * cols / group_size]
        s_channel: []f32, // per-output-row scale, [rows]
        correction: []f32, // [cols / group_size * rows]
    };

    /// A convrot_group_size of 0 skips the rotation, which only tests want.
    pub fn quantizeToAsymW4a8Zp(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot_group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
    ) !AsymW4a8ZpData {
        if (input.len != rows * cols) return error.InputSizeMismatch;
        if (cols % w4a8_group_size != 0) return error.ColsNotDivisibleByGroupSize;
        if (convrot_group_size != 0 and cols % convrot_group_size != 0) return error.ColsNotDivisibleByGroupSize;

        const rotated = try allocator.alloc(f32, input.len);
        defer allocator.free(rotated);
        @memcpy(rotated, input);
        if (convrot_group_size != 0) try rotateGroupwiseInPlace(rotated, rows, cols, convrot_group_size, pool);

        const groups_per_row = cols / w4a8_group_size;

        const weight = try allocator.alloc(u8, rows * (cols / 2));
        errdefer allocator.free(weight);
        const s_rel = try allocator.alloc(u8, rows * groups_per_row);
        errdefer allocator.free(s_rel);
        const s_channel = try allocator.alloc(f32, rows);
        errdefer allocator.free(s_channel);
        const correction = try allocator.alloc(f32, rows * groups_per_row);
        errdefer allocator.free(correction);

        const group_scales = try allocator.alloc(f32, rows * groups_per_row);
        defer allocator.free(group_scales);

        const threads_u64: u64 = @intCast(pool.threads.len);
        const rows_u64: u64 = @intCast(rows);
        const rows_per_thread = @divTrunc(rows_u64, threads_u64);
        const leftover = rows_u64 - (rows_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * rows_per_thread;
            var end = start + rows_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            if (start == end) continue;
            pool.spawnWg(&wg, quantizeAsymW4a8ZpRows, .{ rotated, weight, s_rel, s_channel, correction, group_scales, rows, cols, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();

        return .{ .weight = weight, .s_rel = s_rel, .s_channel = s_channel, .correction = correction };
    }

    fn quantizeAsymW4a8ZpRows(
        rotated: []const f32,
        weight: []u8,
        s_rel: []u8,
        s_channel: []f32,
        correction: []f32,
        group_scales: []f32,
        rows: usize,
        cols: usize,
        start_row: usize,
        end_row: usize,
    ) void {
        const gs_n = w4a8_group_size;
        const groups_per_row = cols / gs_n;
        const packed_cols = cols / 2;

        for (start_row..end_row) |r| {
            const row = rotated[r * cols ..][0..cols];
            const row_scales = group_scales[r * groups_per_row ..][0..groups_per_row];

            var max_shifted: f32 = 0.0;
            for (0..groups_per_row) |g| {
                const grp = row[g * gs_n ..][0..gs_n];
                const gv: GroupVec = grp.*;
                const mn = @reduce(.Min, gv);
                const gs: f32 = @max((@reduce(.Max, gv) - mn) / 15.0, 1e-8);
                row_scales[g] = gs;
                correction[g * rows + r] = 8.0 * gs + mn;

                var codes: [gs_n]u8 = undefined;
                for (&codes, 0..) |*c, j| {
                    const q = std.math.clamp(roundHalfToEven((grp[j] - mn) / gs), 0.0, 15.0);
                    c.* = @intFromFloat(q);
                    max_shifted = @max(max_shifted, @abs((q - 8.0) * gs));
                }
                const base = g * gs_n;
                for (0..gs_n / 2) |p| {
                    weight[r * packed_cols + (base + 2 * p) / 2] = codes[2 * p] | (codes[2 * p + 1] << 4);
                }
            }

            const sc: f32 = @max(max_shifted / 127.0, 1e-8);
            s_channel[r] = sc;
            for (0..groups_per_row) |g| s_rel[r * groups_per_row + g] = f32_to_fp8_e4m3(row_scales[g] / sc);
        }
    }

    // w6a8_int8: the same ComfyUI layout as asym_w4a8_int8 at bits=6. Uniform levels -31..31
    // instead of a codebook, stored as code q+32, and the fp8 group scale is then searched over
    // its e4m3 neighbours for the one whose rounded int8 grid fits the group best. Matches
    // comfy_kitchen quantize_w4a8_int8_weight(bits=6):
    //
    //   group_scale = max(amax(group) / 31, 1e-8)
    //   q           = clamp(round(w / group_scale), -31, 31)
    //   twice:
    //       group_scale = max(dot(w, q) / max(dot(q, q), 1e-8), 1e-8)
    //       q           = clamp(round(w / group_scale), -31, 31)
    //   s_channel   = max(amax_row(q * group_scale) / 127, 1e-8)
    //   for b in fp8(group_scale / s_channel), b - 1, b + 1:
    //       levels[j] = clamp(round(j * fp8_decode(b)), -127, 127)   for j in -31..31
    //       q         = nearest(levels, w / s_channel)
    //   keep the first b with the smallest sum((levels[q] - w / s_channel)^2)
    //
    // A row is 3*cols/4 bytes: cols/2 bytes of low nibbles (even column low), then cols/4 bytes
    // holding each code's top two bits, column c at bits 2*(c%4) of byte c/4.

    pub const w6a8_levels: i32 = 31;
    const w6a8_n_levels: usize = 2 * w6a8_levels + 1;
    /// Largest finite e4m3fn byte; 0x7F is NaN.
    const e4m3_max_finite_byte: u8 = 0x7E;

    pub const W6a8Data = struct {
        weight: []u8, // [rows * 3 * cols / 4]
        s_rel: []u8, // per-group scale, raw fp8 e4m3 bytes, [rows * cols / group_size]
        s_channel: []f32, // per-output-row scale, [rows]
    };

    /// The int8 grid every s_rel byte decodes to, indexed [byte][level index].
    const W6a8Grids = [256][w6a8_n_levels]f32;

    /// comfy_kitchen's nearest-level rule for a whole group against one ascending grid: the
    /// lower bound (torch.searchsorted) is the count of levels below t, then the closer of the
    /// two neighbours wins, ties going low. Rounding can repeat a level at a small s_rel, and
    /// this still picks the index the search would. A NaN lane gets 0. Also returns the chosen
    /// level for each lane.
    inline fn w6a8NearestGroup(levels: *const [w6a8_n_levels]f32, t: GroupVec, idx_out: *GroupIdx) GroupVec {
        const I = @Vector(w4a8_group_size, i32);
        var pos: I = @splat(0);
        for (levels) |l| {
            pos += @select(i32, @as(GroupVec, @splat(l)) < t, @as(I, @splat(1)), @as(I, @splat(0)));
        }
        const lo_i = @max(pos - @as(I, @splat(1)), @as(I, @splat(0)));
        const hi_i = @min(pos, @as(I, @splat(w6a8_n_levels - 1)));
        var below: GroupVec = undefined;
        var above: GroupVec = undefined;
        inline for (0..w4a8_group_size) |j| {
            below[j] = levels[@intCast(lo_i[j])];
            above[j] = levels[@intCast(hi_i[j])];
        }
        const take_hi = @abs(t - above) < @abs(t - below);
        idx_out.* = @intCast(@select(i32, take_hi, hi_i, lo_i));
        return @select(f32, take_hi, above, below);
    }

    /// A convrot_group_size of 0 skips the rotation, which only tests want.
    pub fn quantizeToW6a8(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot_group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
    ) !W6a8Data {
        if (input.len != rows * cols) return error.InputSizeMismatch;
        // cols % 32 keeps the two-bit plane whole bytes and both planes 8-byte aligned.
        if (cols % 32 != 0 or cols % w4a8_group_size != 0) return error.ColsNotDivisibleByGroupSize;
        if (convrot_group_size != 0 and cols % convrot_group_size != 0) return error.ColsNotDivisibleByGroupSize;

        const rotated = try allocator.alloc(f32, input.len);
        defer allocator.free(rotated);
        @memcpy(rotated, input);
        if (convrot_group_size != 0) try rotateGroupwiseInPlace(rotated, rows, cols, convrot_group_size, pool);

        const groups_per_row = cols / w4a8_group_size;

        const weight = try allocator.alloc(u8, rows * (cols / 4 * 3));
        errdefer allocator.free(weight);
        const s_rel = try allocator.alloc(u8, rows * groups_per_row);
        errdefer allocator.free(s_rel);
        const s_channel = try allocator.alloc(f32, rows);
        errdefer allocator.free(s_channel);

        const group_scales = try allocator.alloc(f32, rows * groups_per_row);
        defer allocator.free(group_scales);

        const grids = try allocator.create(W6a8Grids);
        defer allocator.destroy(grids);
        for (grids, 0..) |*levels, b| {
            const s = fp8_e4m3_to_f32(@intCast(b));
            for (levels, 0..) |*l, j| {
                const q: f32 = @floatFromInt(@as(i32, @intCast(j)) - w6a8_levels);
                l.* = std.math.clamp(roundHalfToEven(q * s), -127.0, 127.0);
            }
        }

        const threads_u64: u64 = @intCast(pool.threads.len);
        const rows_u64: u64 = @intCast(rows);
        const rows_per_thread = @divTrunc(rows_u64, threads_u64);
        const leftover = rows_u64 - (rows_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * rows_per_thread;
            var end = start + rows_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            if (start == end) continue;
            pool.spawnWg(&wg, quantizeW6a8Rows, .{ rotated, weight, s_rel, s_channel, group_scales, grids, cols, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();

        return .{ .weight = weight, .s_rel = s_rel, .s_channel = s_channel };
    }

    /// Level index 0..62 of round(v), clamped to -31..31.
    /// Adding and subtracting 1.5 * 2^23 rounds half to even like roundHalfToEven for
    /// |v| < 2^22. Anything larger clamps to the same end either way, and a NaN lane comes out
    /// as +31, as roundHalfToEven then std.math.clamp would give.
    inline fn w6a8AssignUniform(vals: GroupVec) [w4a8_group_size]u8 {
        const lim: GroupVec = @splat(@floatFromInt(w6a8_levels));
        const magic: GroupVec = @splat(12582912.0);
        const in_range = @abs(vals) < @as(GroupVec, @splat(4194304.0));
        const rounded = @select(f32, in_range, (vals + magic) - magic, vals);
        const q = @max(@min(rounded, lim), -lim);
        const idx: GroupIdx = @intFromFloat(q + lim);
        return idx;
    }

    fn quantizeW6a8Rows(
        rotated: []const f32,
        weight: []u8,
        s_rel: []u8,
        s_channel: []f32,
        group_scales: []f32,
        grids: *const W6a8Grids,
        cols: usize,
        start_row: usize,
        end_row: usize,
    ) void {
        const gs_n = w4a8_group_size;
        const groups_per_row = cols / gs_n;
        const row_bytes = cols / 4 * 3;
        const hi_off = cols / 2;
        const lim: f32 = @floatFromInt(w6a8_levels);

        for (start_row..end_row) |r| {
            const row = rotated[r * cols ..][0..cols];
            const row_scales = group_scales[r * groups_per_row ..][0..groups_per_row];

            var max_shifted: f32 = 0.0;
            for (0..groups_per_row) |g| {
                const grp = row[g * gs_n ..][0..gs_n];
                const gv: GroupVec = grp.*;

                var gs: f32 = @max(groupAmax(gv) / lim, 1e-8);
                var idx = w6a8AssignUniform(gv / @as(GroupVec, @splat(gs)));
                // Scalar sums in lane order, as in quantizeAsymW4a8Rows.
                for (0..2) |_| {
                    var num: f32 = 0.0;
                    var den: f32 = 0.0;
                    for (grp, 0..) |v, j| {
                        const q: f32 = @as(f32, @floatFromInt(idx[j])) - lim;
                        num += v * q;
                        den += q * q;
                    }
                    gs = @max(num / @max(den, 1e-8), 1e-8);
                    idx = w6a8AssignUniform(gv / @as(GroupVec, @splat(gs)));
                }
                row_scales[g] = gs;
                for (idx) |j| max_shifted = @max(max_shifted, @abs((@as(f32, @floatFromInt(j)) - lim) * gs));
            }

            const sc: f32 = @max(max_shifted / 127.0, 1e-8);
            s_channel[r] = sc;

            const out = weight[r * row_bytes ..][0..row_bytes];
            for (0..groups_per_row) |g| {
                const gv: GroupVec = row[g * gs_n ..][0..gs_n].*;
                const target = gv / @as(GroupVec, @splat(sc));

                // Candidates in comfy's order, ties keeping the earlier one. Below byte 2 the
                // lower neighbour is either the base again or, wrapped, a NaN scale; neither
                // can win, so it is skipped. The upper one saturates at the largest finite.
                const base = f32_to_fp8_e4m3(row_scales[g] / sc);
                var cands: [3]u8 = undefined;
                var n_cands: usize = 0;
                cands[n_cands] = base;
                n_cands += 1;
                if (base >= 2) {
                    cands[n_cands] = base - 1;
                    n_cands += 1;
                }
                cands[n_cands] = @min(base +| 1, e4m3_max_finite_byte);
                n_cands += 1;

                var best_byte = base;
                var best_err: f32 = std.math.inf(f32);
                var best_idx: [gs_n]u8 = undefined;
                for (cands[0..n_cands]) |b| {
                    var cand_idx: GroupIdx = undefined;
                    const chosen: [gs_n]f32 = w6a8NearestGroup(&grids[b], target, &cand_idx);
                    // Summed in lane order, as the reference does.
                    var err: f32 = 0.0;
                    const target_arr: [gs_n]f32 = target;
                    for (chosen, target_arr) |l, t| {
                        const d = l - t;
                        err += d * d;
                    }
                    if (err < best_err) {
                        best_err = err;
                        best_byte = b;
                        best_idx = cand_idx;
                    }
                }
                s_rel[r * groups_per_row + g] = best_byte;

                // Level index 0..62 is stored as code 1..63.
                var codes: [gs_n]u8 = undefined;
                for (&codes, best_idx) |*c, k| c.* = k + 1;
                const col0 = g * gs_n;
                for (0..gs_n / 2) |p| {
                    out[(col0 + 2 * p) / 2] = (codes[2 * p] & 0x0F) | ((codes[2 * p + 1] & 0x0F) << 4);
                }
                for (0..gs_n / 4) |p| {
                    var hi: u8 = 0;
                    inline for (0..4) |s| hi |= ((codes[4 * p + s] >> 4) & 0x3) << (2 * s);
                    out[hi_off + (col0 + 4 * p) / 4] = hi;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // ComfyUI MXFP cluster quantization.
    // Produces weight and scale.
    // -------------------------------------------------------------------------

    pub const ComfyMxfpData = struct { weight: []u8, scale: []u8 };

    /// Quantize F32 input to MXFP4 cluster format
    ///   weight: sequential OCP nibbles (8 per U32, stored as raw bytes)
    ///   scale:  E8M0 byte per 32-element block
    /// Caller owns both returned slices.
    pub fn quantizeToComfyMxfp4(
        allocator: std.mem.Allocator,
        input: []const f32,
        pool: *thread_pool_mod.ThreadPool,
    ) !ComfyMxfpData {
        const n = input.len;
        if (n % 32 != 0) return error.ElementCountNotMultipleOf32;
        const n_blocks = n / 32;

        // Quantize via GGML to get GGUF mxfp4 blocks: [E8M0 byte][16 × packed nibbles]
        const gguf_buf = try allocator.alloc(u8, n_blocks * 17);
        defer allocator.free(gguf_buf);
        try convertTypeGguf(input, gguf_buf, pool, .mxfp4, 32, 17, null);

        const weight = try allocator.alloc(u8, n / 2);
        errdefer allocator.free(weight);
        const scale = try allocator.alloc(u8, n_blocks);
        errdefer allocator.free(scale);

        // Repack: GGUF first-half/second-half → sequential (low nibble = earlier element)
        // GGUF:    qs[j] = elem[j] | (elem[j+16] << 4)  for j in 0..15
        // output:  qs[k] = elem[2k] | (elem[2k+1] << 4) for k in 0..15
        for (0..n_blocks) |bi| {
            const block = gguf_buf[bi * 17 .. bi * 17 + 17];
            scale[bi] = block[0];
            var nibbles: [32]u8 = undefined;
            for (0..16) |j| {
                nibbles[j]      = block[1 + j] & 0xF;
                nibbles[j + 16] = block[1 + j] >> 4;
            }
            const base = bi * 16;
            for (0..16) |k| {
                weight[base + k] = nibbles[2 * k] | (nibbles[2 * k + 1] << 4);
            }
        }

        return .{ .weight = weight, .scale = scale };
    }

    /// Quantize F32 data to ComfyUI MXFP8 cluster format.
        /// Returns weight (F8_E4M3, 1 byte per element) and scale (E8M0, 1 byte per block of 32).
        pub fn quantizeToComfyMxfp8(
        allocator: std.mem.Allocator,
        f32_slice: []const f32,
        pool: *thread_pool_mod.ThreadPool,
    ) !ComfyMxfpData {
        const n_elements = f32_slice.len;
        if (n_elements % 32 != 0) return error.InvalidMxfp8Size;

        const n_blocks = n_elements / 32;
        const weight = try allocator.alloc(u8, n_elements);
        errdefer allocator.free(weight);
        const scale = try allocator.alloc(u8, n_blocks);
        errdefer allocator.free(scale);

        const threads_u64: u64 = @intCast(pool.threads.len);
        const blocks_per_thread = @divTrunc(n_blocks, threads_u64);
        const leftover = n_blocks - (blocks_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * blocks_per_thread;
            var end = start + blocks_per_thread;
            if (i == threads_u64 - 1) end += leftover;
            pool.spawnWg(&wg, processMxfp8Blocks, .{ f32_slice, weight, scale, start, end });
        }
        wg.wait();

        return .{ .weight = weight, .scale = scale };
    }

    /// Apply cuBLAS MXFP8 scale blocking (equivalent to Python's to_blocked).
    /// Input:  row-major scales [n_rows * n_scale_cols], one E8M0 byte per 32-element block.
    /// Output: cuBLAS-tiled scales [n_row_blocks*128 * n_col_blocks*4], zero-padded.
    /// Mapping: ik=rb*n_col_blocks+cb, flat=ik*512+b*16+a*4+c_within,
    ///          output[flat/padded_cols][flat%padded_cols] = input[r][c]
    pub fn toBlockedMxfp8(
        allocator: std.mem.Allocator,
        scale_raw: []const u8,
        n_rows: usize,
        n_scale_cols: usize,
    ) ![]u8 {
        const n_row_blocks = (n_rows + 127) / 128;
        const n_col_blocks = (n_scale_cols + 3) / 4;
        const padded_rows = n_row_blocks * 128;
        const padded_cols = n_col_blocks * 4;
        const out = try allocator.alloc(u8, padded_rows * padded_cols);
        @memset(out, 0);
        for (0..n_rows) |r| {
            const rb = r / 128;
            const r_within = r % 128;
            const a = r_within / 32; // a ∈ [0,4)
            const b = r_within % 32; // b ∈ [0,32)
            for (0..n_scale_cols) |c| {
                const cb = c / 4;
                const c_within = c % 4;
                const ik = rb * n_col_blocks + cb;
                const flat = ik * 512 + b * 16 + a * 4 + c_within;
                out[flat] = scale_raw[r * n_scale_cols + c];
            }
        }
        return out;
    }

    fn processMxfp8Blocks(input: []const f32, weight: []u8, scale: []u8, start: u64, end: u64) void {
        const start_usize: usize = @intCast(start);
        const end_usize: usize = @intCast(end);
        for (start_usize..end_usize) |block_idx| {
            const elem_start = block_idx * 32;
            const block = input[elem_start..][0..32];

            // Find max absolute value in this block
            var amax: f32 = 0.0;
            for (block) |val| {
                amax = @max(amax, @abs(val));
            }

            // Handle zero/near-zero blocks
            if (amax < 1e-30) {
                scale[block_idx] = 0;
                for (0..32) |i| weight[elem_start + i] = 0;
                continue;
            }

            // OCP MX spec: shared_exp = floor(log2(amax)) + 1
            // This gives us the scale as 2^shared_exp
            // E8M0 stores this directly as (shared_exp + 127)
            const log2_amax = @log2(amax);
            const shared_exp_unbiased: i32 = @as(i32, @intFromFloat(@floor(log2_amax))) + 1;
            const shared_exp_biased: i32 = shared_exp_unbiased + 127;

            // Clamp to valid E8M0 range [0, 255]
            const scale_byte: u8 = @intCast(std.math.clamp(shared_exp_biased, 0, 254));
            scale[block_idx] = scale_byte;

            // Decode the scale for quantization
            const scale_f32 = e8m0_to_f32(scale_byte);
            const inv_scale = 1.0 / scale_f32;

            // Quantize elements: divide by scale then encode as F8_E4M3
            for (block, 0..) |val, i| {
                const scaled = val * inv_scale;
                weight[elem_start + i] = f32_to_fp8_e4m3(scaled);
            }
        }
    }

    // -------------------------------------------------------------------------
    // GGUF mxfp4 block dequantization.
    // Block layout (17 bytes, 32 elements):
    //   [scale: E8M0 u8][qs[0..15]: u8]
    // qs[j] low nibble  → element j      (j in 0..15)
    // qs[j] high nibble → element j + 16
    // -------------------------------------------------------------------------

    fn dequantizeMXFP4Gguf(input_bytes: []const u8, output_f32: []f32, pool: *thread_pool_mod.ThreadPool) void {
        const block_count = output_f32.len / 32;
        if (block_count == 0) return;
        const threads_count = @min(pool.threads.len, block_count);
        const blocks_per_thread = block_count / threads_count;
        const leftover = block_count - (blocks_per_thread * threads_count);

        var wg: thread_pool_mod.WaitGroup = .{};
        var i: usize = 0;
        while (i < threads_count) : (i += 1) {
            const start_block = i * blocks_per_thread;
            const end_block = start_block + blocks_per_thread + (if (i == threads_count - 1) leftover else 0);
            pool.spawnWg(&wg, processDequantizeMXFP4Gguf, .{ input_bytes, output_f32, start_block, end_block });
        }
        wg.wait();
    }

    fn processDequantizeMXFP4Gguf(input_bytes: []const u8, output_f32: []f32, start_block: usize, end_block: usize) void {
        for (start_block..end_block) |b| {
            const scale = e8m0_to_f32(input_bytes[b * 17]);
            const qs = input_bytes[b * 17 + 1 .. b * 17 + 17];
            const elem_base = b * 32;
            for (0..16) |j| {
                output_f32[elem_base + j]      = lut_fp4_e2m1[qs[j] & 0xF] * scale;
                output_f32[elem_base + j + 16] = lut_fp4_e2m1[qs[j] >> 4]  * scale;
            }
        }
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

fn readFileToOwnedSlice(allocator: std.mem.Allocator, path: []const u8, max_size: usize) ![]u8 {
    const io = std.testing.io;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const file_len = try file.length(io);
    if (file_len > max_size) return error.FileTooLarge;
    const buf = try allocator.alloc(u8, @intCast(file_len));
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(io, buf, 0);
    return buf;
}

test "ConvRot Hadamard: fast transform matches dense matrix, and H@H = I" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC04710);
    const rand = prng.random();

    for ([_]usize{ 4, 16, 64, 256 }) |size| {
        try std.testing.expect(Quantizer.isValidHadamardSize(size));
        const h = try Quantizer.buildHadamard(allocator, size);
        defer allocator.free(h);

        // H symmetric and H@H == I (orthogonal + symmetric ⇒ involution).
        for (0..size) |i| {
            for (0..size) |j| {
                try std.testing.expectApproxEqAbs(h[i * size + j], h[j * size + i], 1e-6);
                var dot: f32 = 0;
                for (0..size) |k| dot += h[i * size + k] * h[k * size + j];
                const expected: f32 = if (i == j) 1.0 else 0.0;
                try std.testing.expectApproxEqAbs(expected, dot, 1e-4);
            }
        }

        // Fast transform must equal the dense matrix-vector product H @ v.
        const v = try allocator.alloc(f32, size);
        defer allocator.free(v);
        for (v) |*x| x.* = rand.float(f32) * 2.0 - 1.0;

        const dense = try allocator.alloc(f32, size);
        defer allocator.free(dense);
        for (0..size) |i| {
            var acc: f32 = 0;
            for (0..size) |j| acc += h[i * size + j] * v[j];
            dense[i] = acc;
        }

        const fast = try allocator.dupe(f32, v);
        defer allocator.free(fast);
        Quantizer.hadamardTransformInPlace(fast);

        for (dense, fast) |d, f| try std.testing.expectApproxEqAbs(d, f, 1e-4);

        // Involution: applying twice returns the original.
        Quantizer.hadamardTransformInPlace(fast);
        for (v, fast) |orig, back| try std.testing.expectApproxEqAbs(orig, back, 1e-4);
    }
}

test "ConvRot INT8 quantize→dequantize round-trip beats plain per-row INT8" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rand = prng.random();

    const rows: usize = 8;
    const cols: usize = 256;
    const gs: usize = 256;

    // Build a weight with per-channel outliers — the case ConvRot is designed for.
    const w = try allocator.alloc(f32, rows * cols);
    defer allocator.free(w);
    for (0..rows) |r| {
        for (0..cols) |c| {
            var v = (rand.float(f32) * 2.0 - 1.0) * 0.05;
            if (c % 64 == 0) v += 1.0; // outlier columns
            w[r * cols + c] = v;
        }
    }

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 2 });
    defer pool.deinit();

    // ConvRot path.
    const enc = try Quantizer.quantizeToConvrotInt8(allocator, w, rows, cols, gs, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    const deq = try allocator.alloc(f32, rows * cols);
    defer allocator.free(deq);
    for (0..rows) |r| {
        for (0..cols) |c| {
            deq[r * cols + c] = @as(f32, @floatFromInt(@as(i8, @bitCast(enc.weight[r * cols + c])))) * enc.scale[r];
        }
    }
    try Quantizer.rotateGroupwiseInPlace(deq, rows, cols, gs, &pool);

    // Plain per-row INT8 (no rotation) for comparison.
    var convrot_err: f64 = 0;
    var plain_err: f64 = 0;
    for (0..rows) |r| {
        var amax: f32 = 0;
        for (w[r * cols .. r * cols + cols]) |v| amax = @max(amax, @abs(v));
        const s: f32 = @max(amax / 127.0, 1e-30);
        for (0..cols) |c| {
            const idx = r * cols + c;
            const q = std.math.clamp(@round(w[idx] / s), -128.0, 127.0);
            const plain = q * s;
            plain_err += @abs(plain - w[idx]);
            convrot_err += @abs(deq[idx] - w[idx]);
        }
    }
    // Rotation should meaningfully reduce error on outlier-heavy weights.
    try std.testing.expect(convrot_err < plain_err);
}

test "transform f16 to q8_0" {
    const allocator = std.testing.allocator;

    // Load the f16 source file (skip test if artifacts not present)
    const f16_data = readFileToOwnedSlice(
        allocator,
        "test-artifact/output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight.f16",
        10 * 1024 * 1024,
    ) catch |err| {
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer allocator.free(f16_data);

    // Calculate element count (f16 is 2 bytes per element)
    const element_count: u64 = @intCast(f16_data.len / 2);

    var pool: thread_pool_mod.ThreadPool = undefined;
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
    const expected_data = try readFileToOwnedSlice(
        allocator,
        "test-artifact/output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight.q8_0",
        10 * 1024 * 1024,
    );
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
    return readFileToOwnedSlice(allocator, path, 64 * 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) return null;
        return err;
    };
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

test "E8M0 decode: all 256 values match ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const expected_bytes = (try loadFixture(allocator, "e8m0_decode.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(@as(usize, 256), expected.len);

    var mismatches: usize = 0;
    for (expected, 0..) |exp_val, i| {
        const got = Quantizer.e8m0_to_f32(@intCast(i));
        const both_nan = std.math.isNan(exp_val) and std.math.isNan(got);
        if (!both_nan and got != exp_val) {
            if (mismatches < 8) {
                std.debug.print("  E8M0[0x{X:0>2}]: got={e} expected={e}\n", .{ i, got, exp_val });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "FP4/E2M1 LUT decode: all 16 values match ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const expected_bytes = (try loadFixture(allocator, "fp4_e2m1_decode.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(@as(usize, 16), expected.len);

    var mismatches: usize = 0;
    for (expected, 0..) |exp_val, i| {
        const got = Quantizer.lut_fp4_e2m1[i];
        // -0.0 == 0.0 in IEEE 754; only flag truly different magnitudes/signs
        if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(exp_val))) {
            if (mismatches < 8) {
                std.debug.print("  FP4 LUT[{}]: got={d} expected={d}\n", .{ i, got, exp_val });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "FP4/E2M1 scalar encode: matches ml_dtypes reference" {
    const allocator = std.testing.allocator;

    const inputs_bytes = (try loadFixture(allocator, "fp4_e2m1_encode_inputs.f32")) orelse return error.SkipZigTest;
    defer allocator.free(inputs_bytes);
    const expected = (try loadFixture(allocator, "fp4_e2m1_encode_expected.u8")) orelse return error.SkipZigTest;
    defer allocator.free(expected);

    const inputs: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(inputs_bytes)));
    try std.testing.expectEqual(inputs.len, expected.len);

    var mismatches: usize = 0;
    for (inputs, expected, 0..) |val, exp, i| {
        const got: u8 = Quantizer.f32_to_fp4_e2m1(val);
        if (got != exp) {
            if (mismatches < 8) {
                std.debug.print("  FP4 encode[{}]: f32={d:.4} got=0x{X} expected=0x{X}\n", .{ i, val, got, exp });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "MXFP4 GGUF block decode: matches reference" {
    const allocator = std.testing.allocator;

    const blocks = (try loadFixture(allocator, "mxfp4_gguf_test_blocks.bin")) orelse return error.SkipZigTest;
    defer allocator.free(blocks);
    const expected_bytes = (try loadFixture(allocator, "mxfp4_gguf_test_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const n_blocks = blocks.len / 17;
    const n_elements = n_blocks * 32;
    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try std.testing.expectEqual(n_elements, expected.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    const got_bytes = try Quantizer.convertTensorData(
        allocator,
        blocks,
        types.DataType.mxfp4,
        types.DataType.f32,
        n_elements,
        &pool,
    );
    defer allocator.free(got_bytes);

    const got: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(got_bytes)));
    try std.testing.expectEqual(n_elements, got.len);

    var mismatches: usize = 0;
    for (got, expected, 0..) |g, e, i| {
        if (g != e) {
            if (mismatches < 8) {
                std.debug.print("  MXFP4-GGUF[{}]: got={d} expected={d}\n", .{ i, g, e });
            }
            mismatches += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}

test "MXFP8 toBlockedMxfp8: matches Python to_blocked reference" {
    // Validates toBlockedMxfp8 against fixtures generated by gen_quantization_fixtures.py.
    // Three cases:
    //   1. [128,128] weight → scales [128,4]:  exact fit, no padding needed
    //   2. [3840, 64] weight → scales [3840,2]: n_scale_cols=2, column padding needed
    //   3. [200,  96] weight → scales [200, 3]: both dims need padding
    const allocator = std.testing.allocator;
    const Case = struct { n_rows: usize, n_cols: usize, fixture_num: u8 };
    const cases = [_]Case{
        .{ .n_rows = 128,  .n_cols = 128, .fixture_num = 1 },
        .{ .n_rows = 3840, .n_cols = 64,  .fixture_num = 2 },
        .{ .n_rows = 200,  .n_cols = 96,  .fixture_num = 3 },
    };
    inline for (cases) |c| {
        var inp_name: [32]u8 = undefined;
        var exp_name: [32]u8 = undefined;
        const inp_path = try std.fmt.bufPrint(&inp_name, "mxfp8_blocking_input_{d}.u8",    .{c.fixture_num});
        const exp_path = try std.fmt.bufPrint(&exp_name, "mxfp8_blocking_expected_{d}.u8", .{c.fixture_num});

        const input_bytes    = (try loadFixture(allocator, inp_path)) orelse return error.SkipZigTest;
        defer allocator.free(input_bytes);
        const expected_bytes = (try loadFixture(allocator, exp_path)) orelse return error.SkipZigTest;
        defer allocator.free(expected_bytes);

        const n_scale_cols = (c.n_cols + 31) / 32;
        try std.testing.expectEqual(c.n_rows * n_scale_cols, input_bytes.len);

        const got = try Quantizer.toBlockedMxfp8(allocator, input_bytes, c.n_rows, n_scale_cols);
        defer allocator.free(got);

        try std.testing.expectEqual(expected_bytes.len, got.len);
        try std.testing.expectEqualSlices(u8, expected_bytes, got);
    }
}

// ============================================================================
// Activation-aware quantization (ggml imatrix)
// ============================================================================

fn fillDeterministicWeights(dst: []f32) void {
    var s: u64 = 0x243F6A8885A308D3; // pi digits, as good a seed as any
    for (dst, 0..) |*v, i| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const bits24: f32 = @floatFromInt(s >> 40);
        const x = bits24 / 8388608.0 - 1.0; // [-1, 1)
        v.* = x * 0.05 + (if (i % 512 == 0) @as(f32, 1.0) else 0.0);
    }
}

/// A plausible importance spread: lognormal with an occasional outlier channel,
/// rescaled to mean 1. `sigma` widens it.
fn fillLognormalImportance(dst: []f32, sigma: f32) void {
    var s: u64 = 0xDEADBEEF12345678;
    for (dst, 0..) |*v, j| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        const u: f32 = @as(f32, @floatFromInt(s >> 40)) / 8388608.0 - 1.0; // [-1, 1)
        v.* = @exp(sigma * u * 2.0) * (if (j % 97 == 0) @as(f32, 30.0) else 1.0);
    }
    var mean: f64 = 0;
    for (dst) |v| mean += v;
    mean /= @floatFromInt(dst.len);
    for (dst) |*v| v.* = @floatCast(@as(f64, v.*) / mean);
}

/// Quantize `w` to `dst_type` and dequantize straight back, so the caller can
/// measure what the format cost.
fn roundtripWeighted(
    allocator: std.mem.Allocator,
    w: []const f32,
    dst_type: types.DataType,
    pool: *thread_pool_mod.ThreadPool,
    imatrix: ?[]const f32,
) ![]f32 {
    const q = try Quantizer.convertTensorDataWeighted(
        allocator,
        std.mem.sliceAsBytes(w),
        .F32,
        dst_type,
        w.len,
        pool,
        imatrix,
    );
    defer allocator.free(q);
    const back = try Quantizer.convertTensorData(allocator, q, dst_type, .F32, w.len, pool);
    defer allocator.free(back);
    const as_f32: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, back));
    return allocator.dupe(f32, as_f32);
}

/// Σ_j w_j · (a_j − b_j)² over every row, weights cycling with the row width.
/// Exactly the objective ggml's weighted scale search minimizes, which is what
/// makes it the right yardstick.
fn weightedSqErr(a: []const f32, b: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (a, b, 0..) |x, y, i| {
        const d: f64 = @as(f64, x) - @as(f64, y);
        acc += @as(f64, weights[i % weights.len]) * d * d;
    }
    return acc;
}

test "an imatrix lowers the weighted error it is given to minimize" {
    // The receipt that the weights actually reach ggml's scale search, measured
    // on ggml's own objective - Σ w_j (W-Ŵ)² - because that is what an imatrix
    // promises to improve. It makes the *plain* squared error worse by
    // construction; that trade is the entire point.
    //
    // q2_k is absent: it is the one type that gets worse on this objective, a
    // real property of ggml's q2_K encoder rather than a reason to withhold the
    // weights from it.
    const allocator = std.testing.allocator;
    const rows = 16;
    const cols = 512;
    const n = rows * cols;

    const w = try allocator.alloc(f32, n);
    defer allocator.free(w);
    fillDeterministicWeights(w);

    const imat = try allocator.alloc(f32, cols);
    defer allocator.free(imat);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    // Two spreads, because the benefit is spread-dependent.
    for ([_]f32{ 1.0, 2.0 }) |sigma| {
        fillLognormalImportance(imat, sigma);
        for ([_]types.DataType{ .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q4_1, .q5_0, .q5_1 }) |dt| {
            const plain = try roundtripWeighted(allocator, w, dt, &pool, null);
            defer allocator.free(plain);
            const weighted = try roundtripWeighted(allocator, w, dt, &pool, imat);
            defer allocator.free(weighted);

            const e_plain = weightedSqErr(w, plain, imat);
            const e_weighted = weightedSqErr(w, weighted, imat);
            if (!(e_weighted < e_plain)) {
                std.debug.print(
                    "sigma {d}: {s}: imatrix did not reduce the weighted error: plain {e:.6} weighted {e:.6}\n",
                    .{ sigma, @tagName(dt), e_plain, e_weighted },
                );
                return error.ImatrixNoBenefit;
            }
        }
    }
}

test "the GGUF block tables agree with ggml's own for every type we emit" {
    // getBlockSize and getBytesPerBlock are hand-maintained numbers that decide
    // how large an output buffer is and how many blocks fit in it. Wrong values
    // do not fail loudly - they write a file whose tensors are the wrong length.
    // ggml already knows the answers, so ask it rather than trusting the table.
    const emitted = [_]types.DataType{
        .q4_0,    .q4_1,   .q5_0,    .q5_1,  .q8_0,
        .q2_k,    .q3_k,   .q4_k,    .q5_k,  .q6_k,
        .iq2_xxs, .iq2_xs, .iq2_s,   .iq3_xxs, .iq3_s,
        .iq1_s,   .iq1_m,  .iq4_nl,  .iq4_xs,
        .mxfp4,
    };
    for (emitted) |dt| {
        const t = try gguf.GgmlType.fromString(@tagName(dt));
        const gt: ggml.enum_ggml_type = @intCast(@intFromEnum(t));
        const want_block: u64 = @intCast(ggml.ggml_blck_size(gt));
        const want_bytes: u64 = @intCast(ggml.ggml_type_size(gt));
        if (t.getBlockSize() != want_block or t.getBytesPerBlock() != want_bytes) {
            std.debug.print(
                "{s}: block {d} (ggml {d}), bytes/block {d} (ggml {d})\n",
                .{ @tagName(dt), t.getBlockSize(), want_block, t.getBytesPerBlock(), want_bytes },
            );
            return error.BlockTableMismatch;
        }
    }
}

test "W4A8 assignGroup counts boundaries below each lane exactly as a float compare does" {
    const Q = Quantizer;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rnd = prng.random();
    const inf = std.math.inf(f32);
    const specials = [_]f32{
        0.0,                         -0.0,                           inf, -inf,
        std.math.nan(f32),           -std.math.nan(f32),             std.math.floatTrueMin(f32),
        -std.math.floatTrueMin(f32), std.math.floatMax(f32),         -std.math.floatMax(f32),
        127.0,                       -127.0,
    };

    for (0..50_000) |it| {
        // Tables shaped like the real ones: the fixed codebook, a scaled whole-number grid with
        // collapsed duplicates and signed zeros, and the all-zero grid of a zero s_rel byte.
        var levels: [16]f32 = undefined;
        for (&levels, Q.w4a8_codebook) |*l, c| {
            l.* = switch (it % 3) {
                0 => c,
                1 => std.math.clamp(Q.roundHalfToEven(c * rnd.float(f32) * 140.0), -127.0, 127.0),
                else => c * 0.0,
            };
        }
        const mids = Q.levelMidpoints(levels);

        var vals: [16]f32 = undefined;
        for (&vals, 0..) |*v, j| {
            const m = mids[j % 15];
            v.* = switch (rnd.uintLessThan(u8, 5)) {
                0 => specials[rnd.uintLessThan(usize, specials.len)],
                1 => m,
                2 => std.math.nextAfter(f32, m, inf),
                3 => std.math.nextAfter(f32, m, -inf),
                else => rnd.floatNorm(f32) * 60.0,
            };
        }

        const got: [16]u8 = Q.assignGroup(vals, mids);
        for (vals, got) |v, g| {
            var want: u8 = 0;
            for (mids) |m| want += @intFromBool(m < v);
            if (g != want) std.debug.print("assignGroup({e}) = {d}, compare gives {d}, mids {any}\n", .{ v, g, want, mids });
            try std.testing.expectEqual(want, g);
        }
    }
}
