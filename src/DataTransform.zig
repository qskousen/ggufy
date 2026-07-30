const std = @import("std");
const gguf = @import("Gguf.zig");
const types = @import("types.zig");
const tp = @import("tp_core");
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

    /// `convertTensorData` with ggml's activation-aware hook wired up.
    ///
    /// `imatrix`, when given, is one importance weight per **column** of the
    /// tensor — so its length is the row width, and `element_count` must be a
    /// whole number of those rows. The k-quant and legacy qX_0/qX_1 encoders
    /// minimize the weighted squared error instead of the plain one; every other
    /// destination type ignores it. Callers are expected to have checked
    /// applicability already (`Imatrix.decide`); a weight vector that does not
    /// fit the data is an error here rather than something to quietly drop, since
    /// silently reverting to unweighted quantization would make an
    /// activation-aware run indistinguishable from a plain one.
    pub fn convertTensorDataWeighted(
        allocator: std.mem.Allocator,
        src_data: []const u8,
        src_type: types.DataType,
        dst_type: types.DataType,
        element_count: u64,
        pool: *thread_pool_mod.ThreadPool,
        imatrix: ?[]const f32,
    ) ![]u8 {
        // Optimization: Direct copy if types match. Nothing is being quantized,
        // so there is no scale search for an imatrix to steer.
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
                tp.dtype.bf16ToF32Row(input_bytes, output_f32, 1.0);
            },
            .F16, .f16 => {
                if (input_bytes.len / 2 != output_f32.len) return error.InputSizeMismatch;
                tp.dtype.f16ToF32Row(input_bytes, output_f32, 1.0);
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
                tp.quants.raw.dequantRow(
                    @intFromEnum(gguf_type),
                    input_bytes,
                    output_f32.len,
                    output_f32,
                ) catch |err| return switch (err) {
                    // Keep this function's error surface as it was: callers
                    // (precision_realdata, the converter) match on these names.
                    error.UnknownGgmlType, error.UnsupportedGgmlType => error.UnsupportedSourceType,
                    error.NotBlockAligned, error.BufferSizeMismatch => error.InputSizeMismatch,
                    else => err,
                };
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
                if (output_bytes.len < input_f32.len * 2) return error.OutputBufferSizeMismatch;
                tp.dtype.f32ToBf16Row(input_f32, output_bytes[0 .. input_f32.len * 2]);
            },
            .f16, .F16 => {
                if (output_bytes.len < input_f32.len * 2) return error.OutputBufferSizeMismatch;
                tp.dtype.f32ToF16Row(input_f32, output_bytes[0 .. input_f32.len * 2]);
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

        // The work is split into equal "units", each handed to ggml as one call.
        //
        //   - Unweighted: a unit is a single block. Blocks are quantized
        //     independently, so this is free to ignore the tensor's real row
        //     structure — which matters, because ggufy quantizes over the flat
        //     element count and some tensors' rows are not a whole number of
        //     blocks.
        //   - Weighted: a unit is a whole **row**. ggml indexes `quant_weights`
        //     by position within the row it was handed (`quant_weights + QK_K*i +
        //     32*j`), so the weights only line up if it is told the true row
        //     width. Getting this wrong would not fail — it would apply column
        //     0's importance to every 256th weight and quietly produce a worse
        //     model than no imatrix at all.
        const unit_elems: u64 = if (imatrix) |im| im.len else block_elements;
        if (unit_elems == 0) return error.InvalidImatrix;
        if (imatrix != null) {
            if (unit_elems % block_elements != 0) return error.ImatrixNotBlockAligned;
            if (element_count % unit_elems != 0) return error.ImatrixWidthMismatch;
        }
        const unit_bytes: u64 = (unit_elems / block_elements) * block_size;
        const units = @divExact(element_count, unit_elems);

        // Ensure output buffer is large enough
        if (output_bytes.len < block_count * block_size) return error.OutputBufferTooSmall;

        const threads_u64: u64 = @intCast(pool.threads.len);

        // divide units up for threads
        const units_per_thread = @divTrunc(units, threads_u64);
        const leftover = units - (units_per_thread * threads_u64);

        var wg: thread_pool_mod.WaitGroup = .{};
        // The workers cannot return an error through spawnWg, so they flag failure
        // here and we surface it after the join.
        var failed = std.atomic.Value(bool).init(false);

        // Build this type's quantization tables once, on this thread, before the
        // fan-out. ggml documents init as thread-safe (and it is a no-op for every
        // type we emit), but doing it up front keeps it off the hot path.
        tp.quants.raw.ensureQuantizeInit(@intFromEnum(q_type)) catch
            return error.UnsupportedDestinationType;

        var i: u64 = 0;
        while (i < threads_u64) : (i += 1) {
            const start = i * units_per_thread;
            var end = start + units_per_thread;
            if (i == threads_u64 - 1) {
                end += leftover;
            }
            pool.spawnWg(&wg, processBlocks, .{ input_f32, output_bytes, start, end, unit_elems, unit_bytes, q_type, imatrix, &failed });
        }
        wg.wait();
        if (failed.load(.acquire)) return error.QuantizationFailed;
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
        failed: *std.atomic.Value(bool),
    ) void {
        const units = end - start;
        const unit_elems_usize: usize = @intCast(unit_elems);
        const unit_bytes_usize: usize = @intCast(unit_bytes);
        const src_offset: usize = @intCast(start * unit_elems);
        const dst_offset: usize = @intCast(start * unit_bytes);
        // This worker owns `units` whole units, not one — the slices must span all
        // of them. (They used to be cut to a single block's length while ggml was
        // told to write `blocks` of them: correct output, because the parent buffers
        // are contiguous and ggml works off the raw pointer, but the slice bounds
        // were a lie and any bounds-checked API would have rejected them.)
        const src_block = input_f32[src_offset..][0 .. units * unit_elems_usize];
        const dst_block = output_bytes[dst_offset..][0 .. units * unit_bytes_usize];

        // Every unit in this call has the same width, so one imatrix serves all
        // of them — which is exactly ggml's own contract for `quant_weights`.
        _ = tp.quants.raw.quantizeChunk(
            @intFromEnum(q_type),
            src_block,
            dst_block,
            @intCast(units),
            unit_elems_usize,
            imatrix,
        ) catch {
            failed.store(true, .release);
        };
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
        return quantizeToComfyFp8Weighted(allocator, input, pool, null, 0);
    }

    /// `quantizeToComfyFp8` with the activation-weighted clipping search.
    ///
    /// The scale here is a single scalar over the whole tensor, so the search is
    /// one-dimensional and the weighting is the tensor's per-column energy tiled
    /// across rows. `weights` is per input column (length `cols`); null
    /// reproduces `quantizeToComfyFp8` exactly.
    ///
    /// Note this searches against the *rounded fp8 grid*, not a uniform one — the
    /// error model is `f32 → e4m3 → f32`, which is what the format actually does.
    pub fn quantizeToComfyFp8Weighted(
        allocator: std.mem.Allocator,
        input: []const f32,
        pool: *thread_pool_mod.ThreadPool,
        weights: ?[]const f32,
        cols: usize,
    ) !ComfyFp8Data {
        var amax: f32 = 0.0;
        for (input) |v| amax = @max(amax, @abs(v));

        const fp8_max: f32 = 448.0;
        var scale: f32 = if (amax > 0.0) amax / fp8_max else 1.0;
        if (weights) |w| {
            if (cols == 0 or w.len != cols or input.len % cols != 0) return error.WeightsWidthMismatch;
            const chunks = @max(1, @min(fp8_search_chunks, input.len));
            const partials = try allocator.alloc(f64, chunks * clip_ratios.len);
            defer allocator.free(partials);
            scale = searchFp8Scale(input, w, cols, amax, pool, partials);
        }
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

    /// One-dimensional clip search for the FP8 cluster's single global scale.
    /// Separate from `searchScale` because the reconstruction is an fp8 round-trip
    /// rather than a uniform grid, so the error has to be evaluated through the
    /// actual e4m3 encoder.
    ///
    /// Element ranges the fp8 error reduction is split into.
    ///
    /// **Fixed, not derived from the pool size.** The reduction is a global sum of
    /// f64 partials, so its result depends on the partition; deriving the
    /// partition from the thread count would make a tensor quantize differently
    /// on an 8-core machine than on a 16-core one. A constant chunk count keeps
    /// both the summation order and the answer identical everywhere, and the pool
    /// simply schedules however many chunks it can at a time.
    const fp8_search_chunks: usize = 64;

    /// Threaded over element ranges rather than over candidate scales: each chunk
    /// evaluates every candidate on its own slice and the partials are summed in
    /// chunk order afterwards. Splitting by candidate instead would leave most
    /// threads idle once the grid is shorter than the core count.
    fn searchFp8Scale(
        input: []const f32,
        w: []const f32,
        cols: usize,
        amax: f32,
        pool: *thread_pool_mod.ThreadPool,
        partials: []f64,
    ) f32 {
        if (amax <= 0.0) return 1.0;
        const fp8_max: f32 = 448.0;

        const chunks = @max(1, @min(fp8_search_chunks, input.len));
        @memset(partials, 0);

        var wg: thread_pool_mod.WaitGroup = .{};
        const per = input.len / chunks;
        for (0..chunks) |i| {
            const start = i * per;
            const end = if (i == chunks - 1) input.len else start + per;
            pool.spawnWg(&wg, fp8ScaleErrors, .{
                input, w, cols, amax, start, end, partials[i * clip_ratios.len ..][0..clip_ratios.len],
            });
        }
        wg.wait();

        var best_s: f32 = amax / fp8_max;
        var best_e: f64 = std.math.inf(f64);
        for (clip_ratios, 0..) |alpha, k| {
            var acc: f64 = 0;
            for (0..chunks) |i| acc += partials[i * clip_ratios.len + k];
            if (acc < best_e) {
                best_e = acc;
                best_s = alpha * amax / fp8_max;
            }
        }
        return best_s;
    }

    fn fp8ScaleErrors(
        input: []const f32,
        w: []const f32,
        cols: usize,
        amax: f32,
        start: usize,
        end: usize,
        out: []f64,
    ) void {
        const fp8_max: f32 = 448.0;
        for (clip_ratios, 0..) |alpha, k| {
            const s: f32 = alpha * amax / fp8_max;
            if (s <= 0.0) {
                out[k] = std.math.inf(f64);
                continue;
            }
            var acc: f64 = 0;
            for (input[start..end], start..) |v, i| {
                if (std.math.isNan(v) or std.math.isInf(v)) continue;
                const back = fp8_e4m3_to_f32(f32_to_fp8_e4m3(v / s)) * s;
                const d: f64 = @as(f64, v) - @as(f64, back);
                acc += @as(f64, w[i % cols]) * d * d;
            }
            out[k] = acc;
        }
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
    fn roundHalfToEven(x: f32) f32 {
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

    // -------------------------------------------------------------------------
    // Activation-weighted clipping search (plan §8A.2)
    //
    // Every symmetric-integer format here picks its scale as `amax / qmax`: the
    // smallest scale that clips nothing. That is the right answer when all
    // weights matter equally, and the wrong one when they do not — a single
    // outlier channel the network barely reads stretches the grid for every
    // channel it does read. Clipping it costs one large error on a column with
    // low activation energy and buys finer resolution everywhere else.
    //
    // So: search a clipping ratio α over `scale = α · amax / qmax`, and keep the
    // α that minimizes Σ_i w_i (x_i − ŝ·q_i)², where w is per-element activation
    // energy from a calibration cache.
    //
    // **The default is untouched.** With no weights the search does not run at
    // all, so `ggufy convert` without `--calib` still produces the ComfyUI-
    // bit-exact bytes the fixtures pin. This is opt-in precisely because those
    // fixtures are the correctness contract for these formats.
    // -------------------------------------------------------------------------

    /// Clipping ratios searched, from "clip nothing" down to "clip hard". α = 1.0
    /// is first so it is the incumbent: the search uses a strict improvement test,
    /// so it can never return a scale worse than the default *on its own
    /// objective*, and ties keep the default.
    ///
    /// The range reaches **0.10**, not the 0.5 the plan first sketched. A row's
    /// scale is set by its largest element, so clipping an outlier of magnitude
    /// k× the bulk down to the bulk needs α ≈ 1/k — and activation-outlier
    /// channels in transformers routinely sit 10–100× above the median. A grid
    /// stopping at 0.5 can only ever halve the range, which is nowhere near
    /// enough to act on the very rows the search exists for. (Measured: on a row
    /// with a 100× low-importance outlier, a [0.5, 1.0] grid could not improve
    /// the weighted error at all.) Widening is safe because the search only moves
    /// on a strict improvement.
    ///
    /// Resolution is fine near 1.0, where well-behaved rows land, and coarser
    /// below, where the decision is "clip this outlier or not" rather than a
    /// precise ratio.
    pub const clip_ratios = blk: {
        const fine = 21; // 1.000 … 0.500 in 0.025 steps
        const coarse = 8; // 0.450 … 0.100 in 0.050 steps
        var out: [fine + coarse]f32 = undefined;
        for (0..fine) |i| out[i] = 1.0 - 0.025 * @as(f32, @floatFromInt(i));
        for (0..coarse) |i| out[fine + i] = 0.45 - 0.05 * @as(f32, @floatFromInt(i));
        break :blk out;
    };

    /// Weighted squared error of quantizing `vals` at scale `s`.
    /// Lanes the clipping-error evaluation is vectorized over.
    ///
    /// Fixed at comptime, so the f64 accumulation order is the same on every
    /// machine — a search whose answer depended on the vector width would
    /// quantize differently per host, the same reproducibility trap
    /// `fp8_search_chunks` avoids.
    const clip_lanes = 8;

    /// Round half to even, elementwise. Same rule as `roundHalfToEven` — the
    /// search has to score candidates under the rounding the quantizer will
    /// actually apply, or it optimizes for a quantization that never happens.
    ///
    /// `fl - 2·⌊fl/2⌋` is the parity test; it agrees with the scalar's
    /// `@mod(fl, 2)` on negatives too, since both land in [0, 2).
    fn roundHalfToEvenVec(x: @Vector(clip_lanes, f32)) @Vector(clip_lanes, f32) {
        const zero: @Vector(clip_lanes, f32) = @splat(0.0);
        const one: @Vector(clip_lanes, f32) = @splat(1.0);
        const two: @Vector(clip_lanes, f32) = @splat(2.0);
        const half: @Vector(clip_lanes, f32) = @splat(0.5);

        const fl = @floor(x);
        const diff = x - fl;
        const odd = (fl - two * @floor(fl / two)) != zero;
        const tie = @select(f32, odd, one, zero);
        return fl + @select(f32, diff > half, one, @select(f32, diff < half, zero, tie));
    }

    /// Weighted squared error of quantizing `vals` at scale `s`.
    ///
    /// This is the hot loop of the whole clipping search — every candidate ratio
    /// walks every element of every row — so it is vectorized. Note the f64
    /// accumulation is per lane and reduced at the end, which is a different
    /// summation order from a scalar loop; the value can differ in the last bits.
    /// That is fine because it is deterministic and only ever used to *rank*
    /// candidates, but it does mean this is not bit-identical to a naive scalar
    /// implementation of the same formula.
    fn clipError(vals: []const f32, w: []const f32, s: f32, qlo: f32, qhi: f32) f64 {
        const sv: @Vector(clip_lanes, f32) = @splat(s);
        const lo: @Vector(clip_lanes, f32) = @splat(qlo);
        const hi: @Vector(clip_lanes, f32) = @splat(qhi);
        var acc: @Vector(clip_lanes, f64) = @splat(0.0);

        var i: usize = 0;
        while (i + clip_lanes <= vals.len) : (i += clip_lanes) {
            const v: @Vector(clip_lanes, f32) = vals[i..][0..clip_lanes].*;
            const wi: @Vector(clip_lanes, f32) = w[i..][0..clip_lanes].*;
            const q = @min(@max(roundHalfToEvenVec(v / sv), lo), hi);
            // Widen before the subtraction, exactly as the scalar path does, so
            // only the accumulation order differs between the two.
            const vd: @Vector(clip_lanes, f64) = @floatCast(v);
            const qd: @Vector(clip_lanes, f64) = @floatCast(q);
            const sd: @Vector(clip_lanes, f64) = @splat(@as(f64, s));
            const d = vd - sd * qd;
            // A non-finite input contributes nothing, matching the scalar guard.
            // One comparison covers both cases: NaN fails every comparison, and
            // `inf < inf` is false.
            const finite = @abs(v) < @as(@Vector(clip_lanes, f32), @splat(std.math.inf(f32)));
            const term = @as(@Vector(clip_lanes, f64), @floatCast(wi)) * d * d;
            acc += @select(f64, finite, term, @as(@Vector(clip_lanes, f64), @splat(0.0)));
        }

        var total = @reduce(.Add, acc);
        while (i < vals.len) : (i += 1) {
            const v = vals[i];
            if (std.math.isNan(v) or std.math.isInf(v)) continue;
            const q = std.math.clamp(roundHalfToEven(v / s), qlo, qhi);
            const d: f64 = @as(f64, v) - @as(f64, s) * @as(f64, q);
            total += @as(f64, w[i]) * d * d;
        }
        return total;
    }

    /// Pick the scale for one symmetric-integer group. `w` is per-element
    /// importance in the same basis as `vals`; null falls back to the plain
    /// `amax / qmax` the reference implementations use.
    ///
    /// `qdiv` is the divisor the format's default uses (127 for int8, 7 for
    /// int4); `qlo`/`qhi` are the clamp bounds, which are not always symmetric
    /// with it (int8 clamps to [-128, 127]).
    fn searchScale(vals: []const f32, w: ?[]const f32, qdiv: f32, qlo: f32, qhi: f32) f32 {
        var amax: f32 = 0.0;
        for (vals) |v| {
            if (!std.math.isNan(v) and !std.math.isInf(v)) amax = @max(amax, @abs(v));
        }
        const base: f32 = @max(amax / qdiv, 1e-30);
        const weights = w orelse return base;

        var best_s = base;
        var best_e = clipError(vals, weights, base, qlo, qhi);
        // Skip index 0: α = 1.0 is `base`, already the incumbent.
        for (clip_ratios[1..]) |alpha| {
            const s: f32 = @max(alpha * amax / qdiv, 1e-30);
            const e = clipError(vals, weights, s, qlo, qhi);
            if (e < best_e) {
                best_e = e;
                best_s = s;
            }
        }
        return best_s;
    }

    /// Per-element importance in the basis the quantizer actually rounds in.
    ///
    /// Without rotation that is just the per-column activation energy. With
    /// ConvRot the quantizer rounds `H·x` group-wise, so the objective `eᵀDe`
    /// becomes `εᵀ(HᵀDH)ε`. Every entry of the normalized regular Hadamard has
    /// magnitude `1/√g`, so
    ///
    ///     diag(HᵀDH)_k = Σ_j d_j·H[j,k]² = (1/g)·Σ_j d_j
    ///
    /// — **the same value for every k in the group**. The rotated importance is
    /// therefore piecewise constant: one weight per group, equal to that group's
    /// mean energy. That is weaker than per-channel weighting (96 distinct values
    /// per row at g=64, cols=6144, not 6144) but it is the correct diagonal
    /// treatment, and it is what lets a row's scale be set by the groups the
    /// network actually reads.
    ///
    /// The off-diagonal terms are dropped. Keeping them is what §8C's GPTQ solve
    /// is for; a per-row scalar scale cannot exploit them anyway.
    ///
    /// Returns null when there is nothing to do (no weights), so callers keep the
    /// reference path. Caller owns the result.
    pub fn rotatedWeights(
        allocator: std.mem.Allocator,
        weights: ?[]const f32,
        cols: usize,
        convrot: bool,
        group_size: usize,
    ) !?[]f32 {
        const w = weights orelse return null;
        if (w.len != cols) return error.WeightsWidthMismatch;

        const out = try allocator.alloc(f32, cols);
        errdefer allocator.free(out);
        if (!convrot) {
            @memcpy(out, w);
            return out;
        }
        if (group_size == 0 or cols % group_size != 0) return error.ColsNotDivisibleByGroupSize;

        const inv_g = 1.0 / @as(f64, @floatFromInt(group_size));
        var g: usize = 0;
        while (g < cols) : (g += group_size) {
            var sum: f64 = 0;
            for (w[g .. g + group_size]) |v| sum += v;
            const mean: f32 = @floatCast(sum * inv_g);
            @memset(out[g .. g + group_size], mean);
        }
        return out;
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
        return quantizeToInt8Weighted(allocator, input, rows, cols, convrot, group_size, pool, null);
    }

    /// `quantizeToInt8` with the activation-weighted clipping search enabled.
    /// `weights` is per **input column** (length `cols`); null reproduces
    /// `quantizeToInt8` exactly, including its reference-bit-exact scales.
    pub fn quantizeToInt8Weighted(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot: bool,
        group_size: usize,
        pool: *thread_pool_mod.ThreadPool,
        weights: ?[]const f32,
    ) !ConvrotInt8Data {
        if (input.len != rows * cols) return error.InputSizeMismatch;

        const rot_w = try rotatedWeights(allocator, weights, cols, convrot, group_size);
        defer if (rot_w) |rw| allocator.free(rw);

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
            pool.spawnWg(&wg, quantizeInt8Rows, .{ work, weight, scale, cols, rot_w, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
        }
        wg.wait();

        return .{ .weight = weight, .scale = scale };
    }

    fn quantizeInt8Rows(
        work: []const f32,
        weight: []u8,
        scale: []f32,
        cols: usize,
        rot_w: ?[]const f32,
        start_row: usize,
        end_row: usize,
    ) void {
        for (start_row..end_row) |r| {
            const row = work[r * cols .. r * cols + cols];
            const s = searchScale(row, rot_w, 127.0, -128.0, 127.0);
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
        return quantizeToInt4Weighted(allocator, input, rows, cols, convrot, group_size, stochastic_rounding, pool, null);
    }

    /// `quantizeToInt4` with the activation-weighted clipping search enabled.
    /// `weights` is per **input column** (length `cols`); null reproduces
    /// `quantizeToInt4` exactly.
    ///
    /// The search always evaluates candidates under deterministic rounding, even
    /// when `stochastic_rounding` is nonzero. The two are separable: the search
    /// picks where to put the grid, SR then dithers within it, and SR's expected
    /// error at a fixed scale tracks the deterministic error closely enough that
    /// the argmin over α does not move. Searching under a random rounding would
    /// also make the chosen scale depend on the dither, which is worse.
    pub fn quantizeToInt4Weighted(
        allocator: std.mem.Allocator,
        input: []const f32,
        rows: usize,
        cols: usize,
        convrot: bool,
        group_size: usize,
        stochastic_rounding: u64,
        pool: *thread_pool_mod.ThreadPool,
        weights: ?[]const f32,
    ) !Int4Data {
        if (input.len != rows * cols) return error.InputSizeMismatch;
        if (cols % 2 != 0) return error.ColsNotEven;

        const rot_w = try rotatedWeights(allocator, weights, cols, convrot, group_size);
        defer if (rot_w) |rw| allocator.free(rw);

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
            pool.spawnWg(&wg, quantizeInt4Rows, .{ work, weight, scale, cols, stochastic_rounding, rot_w, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
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
        rot_w: ?[]const f32,
        start_row: usize,
        end_row: usize,
    ) void {
        const packed_cols = cols / 2;
        for (start_row..end_row) |r| {
            const row = work[r * cols .. r * cols + cols];
            const s = searchScale(row, rot_w, 7.0, -7.0, 7.0);
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
        // mxfp4's encoder discards quant_weights, so there is nothing to pass.
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
// GGML block-quant golden bytes
// ============================================================================
//
// The ggml block-quant encoders are the one part of the pipeline whose output we
// never pinned: every other format has a fixture, but q4_K et al. were only ever
// checked by round-trip error. That makes any change to the ggml backend — a
// version bump, a build-flag change, moving to a different vendored copy —
// invisible until a model renders wrong. (One such difference was already there:
// q6_k/q5_k/q2_k encode differently at different ggml optimization levels.)
//
// These fixtures are the pin: a fixed input (generated by an explicit xorshift so
// it does not depend on std's PRNG staying put) encoded to every ggml type the
// converter emits, byte for byte. Regenerate them ONLY on a deliberate, reviewed
// ggml change (flip the GENERATE test below), and record it in CLAUDE.md — a diff
// here means every GGUF we write from then on differs from every one written
// before it.

/// Deterministic, self-contained test input: xorshift64 noise at a realistic
/// trained-weight scale, with a periodic outlier so block scale selection is
/// actually exercised. Reproducible across Zig versions and platforms.
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

/// Every ggml type `quantizeFromF32` routes through `ggml_quantize_chunk`.
const ggml_golden_types = [_]types.DataType{
    .q8_0, .q5_0, .q4_0, .q5_1, .q4_1, .q6_k, .q5_k, .q4_k, .q3_k, .q2_k, .mxfp4,
};

const ggml_golden_elems: u64 = 4096; // 16 k-quant super-blocks / 128 legacy blocks

fn quantizeGolden(allocator: std.mem.Allocator, dst_type: types.DataType) ![]u8 {
    const input = try allocator.alloc(f32, @intCast(ggml_golden_elems));
    defer allocator.free(input);
    fillDeterministicWeights(input);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    return Quantizer.convertTensorData(
        allocator,
        std.mem.sliceAsBytes(input),
        .F32,
        dst_type,
        ggml_golden_elems,
        &pool,
    );
}

test "ggml block-quant encoders produce the pinned golden bytes" {
    const allocator = std.testing.allocator;
    var missing: usize = 0;
    var failed: usize = 0;
    // Check every type before failing: which types diverge is the diagnostic, and
    // stopping at the first one hides the rest.
    for (ggml_golden_types) |dst_type| {
        var name_buf: [64]u8 = undefined;
        const fixture_name = try std.fmt.bufPrint(&name_buf, "ggml_{s}.bin", .{@tagName(dst_type)});

        const expected = try loadFixture(allocator, fixture_name) orelse {
            missing += 1;
            continue;
        };
        defer allocator.free(expected);

        const got = try quantizeGolden(allocator, dst_type);
        defer allocator.free(got);

        if (expected.len != got.len) {
            std.debug.print("{t}: size changed, {d} -> {d} bytes\n", .{ dst_type, expected.len, got.len });
            failed += 1;
            continue;
        }
        var diff_bytes: usize = 0;
        var first_diff: usize = 0;
        for (expected, got, 0..) |e, g, i| {
            if (e == g) continue;
            if (diff_bytes == 0) first_diff = i;
            diff_bytes += 1;
        }
        if (diff_bytes != 0) {
            std.debug.print("{t}: {d}/{d} bytes differ (first at {d}: 0x{X:0>2} -> 0x{X:0>2})\n", .{
                dst_type, diff_bytes, expected.len, first_diff, expected[first_diff], got[first_diff],
            });
            failed += 1;
        }
    }
    if (failed != 0) return error.GgmlGoldenBytesChanged;
    // All fixtures absent is a fresh checkout (they are committed, so this means
    // the working dir is wrong); a PARTIAL set means someone deleted one, which
    // would silently stop pinning that type.
    if (missing != 0 and missing != ggml_golden_types.len) {
        std.debug.print("{d} of {d} ggml golden fixtures missing\n", .{ missing, ggml_golden_types.len });
        return error.IncompleteFixtureSet;
    }
    if (missing != 0) return error.SkipZigTest;
}

test "ggufy's block-size tables agree with ggml's own layout" {
    // ggufy sizes output buffers from `GgmlType.getBlockSize`/`getBytesPerBlock`
    // (its own table, also used to write GGUF headers) while ggml writes according
    // to its internal layout. A disagreement is a buffer overrun or a corrupt file,
    // not a test failure — so pin the two against each other now that ggml's answer
    // is reachable directly.
    for (ggml_golden_types) |dst_type| {
        const gt = try gguf.GgmlType.fromString(@tagName(dst_type));
        const id: u32 = @intFromEnum(gt);

        errdefer std.debug.print("block-layout disagreement for {t} (ggml id {d})\n", .{ dst_type, id });
        try std.testing.expectEqual(try tp.quants.raw.blockElems(id), @as(usize, @intCast(gt.getBlockSize())));
        try std.testing.expectEqual(try tp.quants.raw.blockBytes(id), @as(usize, @intCast(gt.getBytesPerBlock())));
        // And the size ggufy predicts for a whole tensor must match ggml's row size.
        const elems: usize = @intCast(gt.getBlockSize() * 4);
        try std.testing.expectEqual(
            try tp.quants.raw.rowBytes(id, elems),
            @as(usize, @intCast(dst_type.calcSizeInBytes(elems))),
        );
    }
}

test "GENERATE ggml golden fixtures" {
    if (true) return error.SkipZigTest; // flip to run; see the comment above
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for (ggml_golden_types) |dst_type| {
        const got = try quantizeGolden(allocator, dst_type);
        defer allocator.free(got);
        var path_buf: [256]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/ggml_{s}.bin", .{ fixture_dir, @tagName(dst_type) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = got });
        std.debug.print("wrote {s} ({d} bytes)\n", .{ path, got.len });
    }
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
// Activation-aware quantization (ggml imatrix) — plan §8A
// ============================================================================

/// Quantize `w` to `dst_type` and dequantize straight back, so the caller can
/// measure what the format cost. `imatrix` steers the scale search where the
/// encoder honours it.
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

/// Σ_j w_j · (a_j − b_j)² summed over every row, with the per-column weights
/// cycling with `cols`. This is exactly the objective ggml's weighted scale
/// search minimizes, which is what makes it the right yardstick.
fn weightedSqErr(a: []const f32, b: []const f32, weights: []const f32) f64 {
    var acc: f64 = 0;
    for (a, b, 0..) |x, y, i| {
        const d: f64 = @as(f64, x) - @as(f64, y);
        acc += @as(f64, weights[i % weights.len]) * d * d;
    }
    return acc;
}

test "row-major decomposition is a pure regrouping: same bytes as block-at-a-time" {
    // The imatrix path hands ggml whole rows instead of single blocks, because
    // that is the only way `quant_weights` lines up with the data. Without
    // weights the two groupings must produce identical bytes — otherwise any
    // difference measured later could be the regrouping rather than the
    // weighting, and the experiment would prove nothing.
    const allocator = std.testing.allocator;
    const rows = 8;
    const cols = 512; // two q4_k blocks per row
    const n = rows * cols;

    const w = try allocator.alloc(f32, n);
    defer allocator.free(w);
    fillDeterministicWeights(w);

    const block_elems: u64 = 256;
    const block_bytes: u64 = 144; // q4_k
    const out_flat = try allocator.alloc(u8, (n / block_elems) * block_bytes);
    defer allocator.free(out_flat);
    const out_rows = try allocator.alloc(u8, out_flat.len);
    defer allocator.free(out_rows);

    try tp.quants.raw.ensureQuantizeInit(@intFromEnum(gguf.GgmlType.q4_k));
    var failed = std.atomic.Value(bool).init(false);

    // One block per unit — what the unweighted path does.
    Quantizer.processBlocks(w, out_flat, 0, n / block_elems, block_elems, block_bytes, .q4_k, null, &failed);
    // One row per unit — what the weighted path does.
    Quantizer.processBlocks(w, out_rows, 0, rows, cols, (cols / block_elems) * block_bytes, .q4_k, null, &failed);

    try std.testing.expect(!failed.load(.acquire));
    try std.testing.expectEqualSlices(u8, out_flat, out_rows);
}

/// Per-channel importance shaped like real activation energy: lognormal with a
/// few heavy outliers, normalized to mean 1.0 exactly as `Imatrix.fromCache`
/// does. `sigma` sets the spread — the variable q2_k turns out to be sensitive to.
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

test "an imatrix lowers the weighted error it is given to minimize" {
    // The receipt that the weights actually reach ggml's scale search, measured
    // on ggml's own objective — Σ w_j (W−Ŵ)² — because that is precisely what an
    // imatrix promises to improve. (It makes the *plain* squared error worse by
    // construction; that trade is the entire point.)
    //
    // q2_k is deliberately absent: it is the one type that gets *worse* on this
    // objective. That is a real property of ggml's q2_K encoder, pinned by the
    // test below — but it is emphatically **not** a reason to withhold an imatrix
    // from q2_k, because on real activations q2_k benefits more than anything
    // else. See `Imatrix.shipsWeighted` for what that gap means.
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

    // Two spreads, because the benefit is spread-dependent and a single point
    // would not show that.
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

test "q2_k is where the weighted-weight-error proxy disagrees with real output error" {
    // A pinned counterexample, kept because it is instructive rather than because
    // it drives any decision.
    //
    // Every other ggml type improves on the objective its own scale search
    // minimizes; q2_k does not, and gets worse the wider the importance spread.
    // On that evidence alone we withheld the imatrix from q2_k — and level 1 on
    // real krea2 activations immediately overturned it: measured on actual output
    // error, q2_k gains *more* than any other format. So the exclusion is gone and
    // `Imatrix.shipsWeighted` weights everything ggml will weight.
    //
    // What survives is the gap itself. The weighted weight error is a proxy; it
    // ignores channel covariance that a real GEMM sees, and here it pointed the
    // wrong way. If a future ggml makes q2_k agree with the other types, this test
    // fails and the note above should be revisited rather than silently kept.
    const allocator = std.testing.allocator;
    const rows = 16;
    const cols = 512;
    const n = rows * cols;

    const w = try allocator.alloc(f32, n);
    defer allocator.free(w);
    fillDeterministicWeights(w);

    const imat = try allocator.alloc(f32, cols);
    defer allocator.free(imat);
    fillLognormalImportance(imat, 2.0); // heavy tail: the realistic regime

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    const plain = try roundtripWeighted(allocator, w, .q2_k, &pool, null);
    defer allocator.free(plain);
    const weighted = try roundtripWeighted(allocator, w, .q2_k, &pool, imat);
    defer allocator.free(weighted);

    const ratio = weightedSqErr(w, weighted, imat) / weightedSqErr(w, plain, imat);
    if (ratio <= 1.0) {
        std.debug.print(
            "q2_k now improves on the weighted-weight-error proxy too (ratio {d:.4}); the\n" ++
                "documented proxy-vs-level-1 disagreement in Imatrix.shipsWeighted is stale\n",
            .{ratio},
        );
        return error.Q2KProxyNoteStale;
    }
}

test "an imatrix changes the bytes, and only for encoders that read it" {
    // Two claims at once: the weights are not being silently dropped for the
    // types that honour them, and they are not being wrongly applied to the ones
    // that document quant_weights as unused (q8_0, mxfp4).
    //
    // This is the ggml-level fact, independent of which formats we choose to send
    // weights to — that lives in `Imatrix.shipsWeighted`. This layer quantizes
    // with whatever it is handed.
    const allocator = std.testing.allocator;
    const rows = 4;
    const cols = 512;
    const n = rows * cols;

    const w = try allocator.alloc(f32, n);
    defer allocator.free(w);
    fillDeterministicWeights(w);

    const imat = try allocator.alloc(f32, cols);
    defer allocator.free(imat);
    for (imat, 0..) |*v, j| v.* = if (j % 32 < 4) 200.0 else 0.02;

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    const honours = [_]types.DataType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q4_1, .q5_0, .q5_1 };
    const ignores = [_]types.DataType{ .q8_0, .mxfp4 };

    inline for (honours ++ ignores) |dt| {
        const plain = try Quantizer.convertTensorData(allocator, std.mem.sliceAsBytes(w), .F32, dt, n, &pool);
        defer allocator.free(plain);
        const weighted = try Quantizer.convertTensorDataWeighted(
            allocator, std.mem.sliceAsBytes(w), .F32, dt, n, &pool, imat,
        );
        defer allocator.free(weighted);

        const same = std.mem.eql(u8, plain, weighted);
        const should_differ = comptime for (honours) |h| {
            if (h == dt) break true;
        } else false;

        if (should_differ and same) {
            std.debug.print("{s}: imatrix was accepted but changed nothing\n", .{@tagName(dt)});
            return error.ImatrixIgnored;
        }
        if (!should_differ and !same) {
            std.debug.print("{s}: imatrix changed bytes for a type that documents it as unused\n", .{@tagName(dt)});
            return error.ImatrixMisapplied;
        }
    }
}

test "weighted quantization is thread-count invariant" {
    // The row split hands each worker a different number of rows depending on the
    // pool size; every one of them gets the whole imatrix. If the weights were
    // ever sliced per worker instead, this is what would catch it.
    const allocator = std.testing.allocator;
    const rows = 12;
    const cols = 512;
    const n = rows * cols;

    const w = try allocator.alloc(f32, n);
    defer allocator.free(w);
    fillDeterministicWeights(w);

    const imat = try allocator.alloc(f32, cols);
    defer allocator.free(imat);
    for (imat, 0..) |*v, j| v.* = @as(f32, @floatFromInt(j % 17)) + 0.5;

    var first: ?[]u8 = null;
    defer if (first) |b| allocator.free(b);

    for ([_]usize{ 1, 3, 5, 8 }) |n_jobs| {
        var pool: thread_pool_mod.ThreadPool = undefined;
        try pool.init(.{ .allocator = allocator, .n_jobs = n_jobs });
        defer pool.deinit();

        const out = try Quantizer.convertTensorDataWeighted(
            allocator, std.mem.sliceAsBytes(w), .F32, .q4_k, n, &pool, imat,
        );
        if (first) |b| {
            defer allocator.free(out);
            try std.testing.expectEqualSlices(u8, b, out);
        } else {
            first = out;
        }
    }
}

test "an imatrix that does not fit the tensor is refused, not ignored" {
    // Reverting to unweighted quantization on a bad fit would make an
    // activation-aware run indistinguishable from a plain one — the failure mode
    // that is impossible to notice from the output file.
    const allocator = std.testing.allocator;
    const n = 4 * 512;
    const w = try allocator.alloc(f32, n);
    defer allocator.free(w);
    fillDeterministicWeights(w);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 2 });
    defer pool.deinit();

    const bad_align = try allocator.alloc(f32, 300); // not a whole number of q4_k blocks
    defer allocator.free(bad_align);
    @memset(bad_align, 1.0);
    try std.testing.expectError(error.ImatrixNotBlockAligned, Quantizer.convertTensorDataWeighted(
        allocator, std.mem.sliceAsBytes(w), .F32, .q4_k, n, &pool, bad_align,
    ));

    const bad_width = try allocator.alloc(f32, 768); // block-aligned, but 2048 % 768 != 0
    defer allocator.free(bad_width);
    @memset(bad_width, 1.0);
    try std.testing.expectError(error.ImatrixWidthMismatch, Quantizer.convertTensorDataWeighted(
        allocator, std.mem.sliceAsBytes(w), .F32, .q4_k, n, &pool, bad_width,
    ));
}

// ============================================================================
// Activation-weighted clipping search (plan §8A.2)
// ============================================================================

test "the clipping search never loses to amax on its own objective" {
    // Guaranteed by construction — α = 1.0 is the incumbent and improvement is
    // strict — but it is the property the whole feature rests on, so it is pinned
    // rather than assumed. If the grid ever stops including 1.0, this fails.
    const gpa = std.testing.allocator;
    const cols = 256;
    const vals = try gpa.alloc(f32, cols);
    defer gpa.free(vals);
    const w = try gpa.alloc(f32, cols);
    defer gpa.free(w);

    fillDeterministicWeights(vals);
    fillLognormalImportance(w, 1.5);

    inline for (.{ .{ 127.0, -128.0, 127.0 }, .{ 7.0, -7.0, 7.0 } }) |cfg| {
        var amax: f32 = 0;
        for (vals) |v| amax = @max(amax, @abs(v));
        const base: f32 = @max(amax / cfg[0], 1e-30);
        const s = Quantizer.searchScale(vals, w, cfg[0], cfg[1], cfg[2]);
        const e_base = Quantizer.clipError(vals, w, base, cfg[1], cfg[2]);
        const e_found = Quantizer.clipError(vals, w, s, cfg[1], cfg[2]);
        try std.testing.expect(e_found <= e_base);
    }
}

test "an outlier the network never reads gets clipped away" {
    // The case the search exists for: one column carries a value 50x everything
    // else and almost no activation energy. Keeping it costs every other column
    // resolution; clipping it costs one large error nobody reads.
    const gpa = std.testing.allocator;
    const cols = 256;
    const vals = try gpa.alloc(f32, cols);
    defer gpa.free(vals);
    const w = try gpa.alloc(f32, cols);
    defer gpa.free(w);

    fillDeterministicWeights(vals);
    for (vals) |*v| v.* = @mod(v.*, 0.05);
    @memset(w, 1.0);
    vals[7] = 1.0; // ~20x the bulk — a realistic outlier channel
    w[7] = 1e-3; // and effectively unread

    var amax: f32 = 0;
    for (vals) |v| amax = @max(amax, @abs(v));
    const base = amax / 7.0;
    const s = Quantizer.searchScale(vals, w, 7.0, -7.0, 7.0);

    // The search must actually clip, not just tie.
    try std.testing.expect(s < base);
    const e_base = Quantizer.clipError(vals, w, base, -7.0, 7.0);
    const e_found = Quantizer.clipError(vals, w, s, -7.0, 7.0);
    if (!(e_found < e_base * 0.5)) {
        std.debug.print("clip search barely helped: base {e:.4} found {e:.4}\n", .{ e_base, e_found });
        return error.ClipSearchIneffective;
    }
}

test "rotated importance is the group mean, because Hadamard entries are all ±1/√g" {
    // eᵀDe becomes εᵀ(HᵀDH)ε under the rotation, and every |H[j,k]| = 1/√g makes
    // diag(HᵀDH) constant within a group. Anything else here would mean the
    // weights are being applied in the wrong basis — which would look like a
    // working feature while steering on noise.
    const gpa = std.testing.allocator;
    const cols = 8;
    const g = 4;
    const w = [_]f32{ 1, 3, 0, 4, 10, 10, 10, 10 };

    const rot = (try Quantizer.rotatedWeights(gpa, &w, cols, true, g)).?;
    defer gpa.free(rot);
    // Group 0 mean = (1+3+0+4)/4 = 2; group 1 mean = 10.
    for (rot[0..4]) |v| try std.testing.expectApproxEqAbs(@as(f32, 2.0), v, 1e-6);
    for (rot[4..8]) |v| try std.testing.expectApproxEqAbs(@as(f32, 10.0), v, 1e-6);

    // Without rotation the weights pass through untouched.
    const plain = (try Quantizer.rotatedWeights(gpa, &w, cols, false, g)).?;
    defer gpa.free(plain);
    try std.testing.expectEqualSlices(f32, &w, plain);

    // No weights means no work, so the caller keeps the reference path.
    try std.testing.expect((try Quantizer.rotatedWeights(gpa, null, cols, true, g)) == null);
    // A width that disagrees is an error, never a silent reinterpretation.
    try std.testing.expectError(error.WeightsWidthMismatch, Quantizer.rotatedWeights(gpa, &w, cols + 1, false, g));
    try std.testing.expectError(error.ColsNotDivisibleByGroupSize, Quantizer.rotatedWeights(gpa, &w, cols, true, 3));
}

test "passing no weights reproduces the reference quantizers byte for byte" {
    // The ComfyUI-pinned fixtures elsewhere in this file are the real contract;
    // this asserts the weighted entry points cannot drift from them by taking a
    // different code path when handed null.
    const gpa = std.testing.allocator;
    const rows = 8;
    const cols = 256;
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fillDeterministicWeights(w);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 2 });
    defer pool.deinit();

    inline for (.{ true, false }) |convrot| {
        const a = try Quantizer.quantizeToInt8(gpa, w, rows, cols, convrot, 64, &pool);
        defer gpa.free(a.weight);
        defer gpa.free(a.scale);
        const b = try Quantizer.quantizeToInt8Weighted(gpa, w, rows, cols, convrot, 64, &pool, null);
        defer gpa.free(b.weight);
        defer gpa.free(b.scale);
        try std.testing.expectEqualSlices(u8, a.weight, b.weight);
        try std.testing.expectEqualSlices(f32, a.scale, b.scale);
    }

    const c = try Quantizer.quantizeToInt4(gpa, w, rows, cols, true, 64, 0, &pool);
    defer gpa.free(c.weight);
    defer gpa.free(c.scale);
    const d = try Quantizer.quantizeToInt4Weighted(gpa, w, rows, cols, true, 64, 0, &pool, null);
    defer gpa.free(d.weight);
    defer gpa.free(d.scale);
    try std.testing.expectEqualSlices(u8, c.weight, d.weight);
    try std.testing.expectEqualSlices(f32, c.scale, d.scale);

    const e = try Quantizer.quantizeToComfyFp8(gpa, w, &pool);
    defer gpa.free(e.weight);
    const f = try Quantizer.quantizeToComfyFp8Weighted(gpa, w, &pool, null, 0);
    defer gpa.free(f.weight);
    try std.testing.expectEqualSlices(u8, e.weight, f.weight);
    try std.testing.expectEqual(e.scale, f.scale);
}

test "weighted quantization moves the scales it is supposed to move" {
    // End to end through the real entry points: with a heavy-tailed importance
    // vector the per-row scales must actually change, and change downward (the
    // search only ever clips). A no-op here would mean the weights are reaching
    // the quantizer and being ignored.
    const gpa = std.testing.allocator;
    const rows = 8;
    const cols = 256;
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fillDeterministicWeights(w);

    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    fillLognormalImportance(imp, 2.0);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 2 });
    defer pool.deinit();

    const plain = try Quantizer.quantizeToInt4(gpa, w, rows, cols, false, 64, 0, &pool);
    defer gpa.free(plain.weight);
    defer gpa.free(plain.scale);
    const weighted = try Quantizer.quantizeToInt4Weighted(gpa, w, rows, cols, false, 64, 0, &pool, imp);
    defer gpa.free(weighted.weight);
    defer gpa.free(weighted.scale);

    var moved: usize = 0;
    for (plain.scale, weighted.scale) |p, q| {
        try std.testing.expect(q <= p * 1.0001); // clipping only ever shrinks
        if (q < p * 0.9999) moved += 1;
    }
    if (moved == 0) {
        std.debug.print("no row scale changed under a heavy-tailed imatrix\n", .{});
        return error.WeightsIgnored;
    }
}

test "the weighted fp8 global scale is thread-count invariant" {
    // The fp8 search reduces a global f64 sum, so its answer depends on how the
    // elements are partitioned. Partitioning by pool size would make the same
    // tensor quantize differently on machines with different core counts — a
    // reproducibility failure nobody would notice until two people compared
    // checksums. `fp8_search_chunks` is fixed for exactly this reason.
    const gpa = std.testing.allocator;
    const rows = 12;
    const cols = 256;
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fillDeterministicWeights(w);

    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    fillLognormalImportance(imp, 2.0);

    var first: ?f32 = null;
    for ([_]usize{ 1, 3, 8 }) |n_jobs| {
        var pool: thread_pool_mod.ThreadPool = undefined;
        try pool.init(.{ .allocator = gpa, .n_jobs = n_jobs });
        defer pool.deinit();

        const enc = try Quantizer.quantizeToComfyFp8Weighted(gpa, w, &pool, imp, cols);
        defer gpa.free(enc.weight);
        if (first) |f| {
            try std.testing.expectEqual(f, enc.scale);
        } else {
            first = enc.scale;
        }
    }
}

test "the vectorized clip error agrees with a scalar reference" {
    // `clipError` is SIMD and accumulates per lane, so it is not bit-identical to
    // a naive scalar loop. It must still agree to well within any margin that
    // could flip a candidate ranking, and it must handle the tail, the tie case
    // of round-half-to-even, and non-finite inputs identically.
    const gpa = std.testing.allocator;

    // 1003 is deliberately not a multiple of the lane count, so the scalar tail runs.
    const n = 1003;
    const vals = try gpa.alloc(f32, n);
    defer gpa.free(vals);
    const w = try gpa.alloc(f32, n);
    defer gpa.free(w);
    fillDeterministicWeights(vals);
    fillLognormalImportance(w, 1.5);

    // Exact .5 ties in v/s exercise the round-half-to-even branch, which is the
    // one place a vectorized rounding is easy to get subtly wrong.
    for (0..24) |k| vals[k] = @as(f32, @floatFromInt(@as(i32, @intCast(k)) - 12)) * 0.5 + 0.25;
    // ...and a NaN and an Inf, which must contribute nothing.
    vals[100] = std.math.nan(f32);
    vals[101] = std.math.inf(f32);
    vals[102] = -std.math.inf(f32);

    const Scalar = struct {
        fn err(v: []const f32, ws: []const f32, s: f32, qlo: f32, qhi: f32) f64 {
            var acc: f64 = 0;
            for (v, ws) |x, wi| {
                if (std.math.isNan(x) or std.math.isInf(x)) continue;
                const q = std.math.clamp(Quantizer.roundHalfToEven(x / s), qlo, qhi);
                const d: f64 = @as(f64, x) - @as(f64, s) * @as(f64, q);
                acc += @as(f64, wi) * d * d;
            }
            return acc;
        }
    };

    for ([_]f32{ 0.5, 0.125, 0.03, 0.007 }) |s| {
        inline for (.{ .{ -128.0, 127.0 }, .{ -7.0, 7.0 } }) |cl| {
            const got = Quantizer.clipError(vals, w, s, cl[0], cl[1]);
            const want = Scalar.err(vals, w, s, cl[0], cl[1]);
            try std.testing.expect(std.math.isFinite(got));
            // Relative agreement: the two differ only by summation order.
            const denom = @max(@abs(want), 1e-12);
            if (@abs(got - want) / denom > 1e-12) {
                std.debug.print("s={d}: vector {e:.17} scalar {e:.17}\n", .{ s, got, want });
                return error.VectorScalarMismatch;
            }
        }
    }
}

test "vectorized rounding matches the scalar rule element for element" {
    // Pinned separately from the error sum, because a rounding bug there would
    // show up only as a slightly different scale rather than as a wrong number,
    // and would be invisible in the aggregate.
    var x: [Quantizer.clip_lanes]f32 = undefined;
    const cases = [_]f32{ -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5 };
    @memcpy(&x, &cases);
    const v: @Vector(Quantizer.clip_lanes, f32) = x;
    const got: [Quantizer.clip_lanes]f32 = Quantizer.roundHalfToEvenVec(v);
    for (got, cases) |g, c| try std.testing.expectEqual(Quantizer.roundHalfToEven(c), g);

    // And away from ties, over a spread that crosses zero in both directions.
    var s: u64 = 0x1234_5678_9ABC_DEF0;
    for (0..64) |_| {
        for (&x) |*e| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            e.* = (@as(f32, @floatFromInt(s >> 40)) / 8388608.0 - 1.0) * 40.0;
        }
        const vv: @Vector(Quantizer.clip_lanes, f32) = x;
        const gg: [Quantizer.clip_lanes]f32 = Quantizer.roundHalfToEvenVec(vv);
        for (gg, x) |g, e| try std.testing.expectEqual(Quantizer.roundHalfToEven(e), g);
    }
}
