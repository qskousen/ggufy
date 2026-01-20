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
        dst_type: gguf.GgmlType,
        element_count: u64,
        threads: usize,
    ) ![]u8 {
        std.log.info("Converting tensor type {s} to {s}", .{@tagName(src_type), @tagName(dst_type)});
        // Optimization: Direct copy if types match
        if (src_type.formatType() == .gguf and std.mem.eql(u8, @tagName(src_type), @tagName(dst_type))) {
            const out = try allocator.alloc(u8, src_data.len);
            @memcpy(out, src_data);
            return out;
        }

        // 1. Dequantize to F32 (Intermediate Buffer)
        // We allocate this temporarily
        const f32_buffer = try allocator.alloc(f32, @intCast(element_count));
        defer allocator.free(f32_buffer);

        try dequantizeToF32(src_data, f32_buffer, src_type);

        // 2. Quantize from F32 to Target
        const out_size = dst_type.calcSizeInBytes(element_count);
        const out_buffer = try allocator.alloc(u8, out_size);
        errdefer allocator.free(out_buffer); // Free on error, otherwise return ownership

        try quantizeFromF32(f32_buffer, out_buffer, dst_type, allocator, threads);

        return out_buffer;
    }

    fn dequantizeToF32(input_bytes: []const u8, output_f32: []f32, src_type: types.DataType) !void {
        switch (src_type) {
            .F8_E4M3 => {
                if (input_bytes.len != output_f32.len)
                    return error.InputSizeMismatch;

                for (input_bytes, 0..) |b, i| {
                    output_f32[i] = fp8_e4m3_to_f32(b);
                }
            },
            .F8_E5M2 => {
                if (input_bytes.len != output_f32.len)
                    return error.InputSizeMismatch;

                for (input_bytes, 0..) |b, i| {
                    output_f32[i] = fp8_e5m2_to_f32(b);
                }
            },
            .BF16, .bf16 => {
                const count = input_bytes.len / 2;
                if (count != output_f32.len)
                    return error.InputSizeMismatch;

                const input_bf16 = std.mem.bytesAsSlice(u16, input_bytes);

                for (input_bf16, 0..) |val, i| {
                    output_f32[i] = bf16_to_f32(val);
                }
            },
            .F16, .f16 => {
                const f16_count = input_bytes.len / 2;
                if (f16_count != output_f32.len) return error.InputSizeMismatch;

                const input_f16 = std.mem.bytesAsSlice(f16, input_bytes);
                for (input_f16, 0..) |val, i| {
                    output_f32[i] = @floatCast(val);
                }
            },
            .F32, .f32 => {
                const input_vals = std.mem.bytesAsSlice(f32, input_bytes);
                @memcpy(output_f32, input_vals);
            },
            .F64, .f64 => {
                const f64_count = input_bytes.len / 8;
                if (f64_count != output_f32.len) return error.InputSizeMismatch;

                const input_f64 = std.mem.bytesAsSlice(f64, input_bytes);
                for (input_f64, 0..) |val, i| {
                    output_f32[i] = @floatCast(val);
                }
            },
            else => return error.UnsupportedSourceType,
        }
    }

    fn quantizeFromF32(input_f32: []const f32, output_bytes: []u8, dst_type: gguf.GgmlType, allocator: std.mem.Allocator, threads: usize) !void {
        switch (dst_type) {
            .f32 => {
                const out_slice = std.mem.bytesAsSlice(f32, output_bytes);
                @memcpy(out_slice, input_f32);
            },
            .f16 => {
                const out_slice = std.mem.bytesAsSlice(f16, output_bytes);
                for (input_f32, 0..) |val, i| {
                    out_slice[i] = @floatCast(val);
                }
            },
            .q8_0 => {
                // Q8_0: 32 values per block.
                // Block structure: delta (f16), followed by 32 int8 quants.
                // Total bytes: 2 + 32 = 34 bytes.
                const block_elements = 32;
                const block_size = 34;
                try convertTypeQX_0(
                    input_f32,
                    output_bytes,
                    allocator,
                    threads,
                    quantizeBlockQ8_0,
                    block_elements,
                    block_size,
                );
            },
            .q5_0 => {
                // Q5_0: 32 values per block.
                const block_elements = 32;
                const block_size = 22;
                try convertTypeQX_0(
                    input_f32,
                    output_bytes,
                    allocator,
                    threads,
                    quantizeBlockQ5_0,
                    block_elements,
                    block_size,
                );
            },
            .q4_0 => {
                // Q4_0: 32 values per block.
                const block_elements = 32;
                const block_size = 18;
                try convertTypeQX_0(
                    input_f32,
                    output_bytes,
                    allocator,
                    threads,
                    quantizeBlockQ4_0,
                    block_elements,
                    block_size,
                );
            },
            .q5_1 => {
                // Q5_1: 32 values per block.
                const block_elements = 32;
                const block_size = 24;
                try convertTypeQX_0(
                    input_f32,
                    output_bytes,
                    allocator,
                    threads,
                    quantizeBlockQ5_1,
                    block_elements,
                    block_size,
                );
            },
            .q4_1 => {
                // Q4_1: 32 values per block.
                const block_elements = 32;
                const block_size = 20;
                try convertTypeQX_0(
                    input_f32,
                    output_bytes,
                    allocator,
                    threads,
                    quantizeBlockQ4_1,
                    block_elements,
                    block_size,
                );
            },
            else => return error.UnsupportedDestinationType,
        }
    }

    fn convertTypeQX_0(
        input_f32: []const f32,
        output_bytes: []u8,
        allocator: std.mem.Allocator,
        threads: usize,
        comptime func: anytype,
        block_elements: usize,
        block_size: usize,
    ) !void {
        const element_count = input_f32.len;
        const block_count = element_count / block_elements;

        // Ensure output buffer is large enough
        if (output_bytes.len < block_count * block_size) return error.OutputBufferTooSmall;

        var pool: std.Thread.Pool = undefined;
        try pool.init(.{
            .allocator = allocator,
            .n_jobs = threads,
        });
        defer pool.deinit();

        var wg: std.Thread.WaitGroup = .{};

        // divide blocks up for threads
        const blocks_per_thread = block_count / threads;
        const leftover = block_count - (blocks_per_thread * threads);

        var i: usize = 0;
        while (i < threads) : (i += 1) {
            const start = i * blocks_per_thread;
            var end = start + blocks_per_thread;
            if (i == threads - 1) {
                end += leftover;
            }
            //std.log.debug("Spawning a task for blocks {} - {} of {}", .{ start, end, block_count });
            pool.spawnWg(&wg, processBlocks, .{ input_f32, output_bytes, start, end, block_elements, block_size, func });
        }
        wg.wait();
    }

    fn processBlocks(input_f32: []const f32, output_bytes: []u8, start: usize, end: usize, block_elements: usize, block_size: usize, comptime func: anytype) void {
        var i = start;
        while (i < end) : (i += 1) {
            const src_offset = i * block_elements;
            const dst_offset = i * block_size;

            const src_block = input_f32[src_offset .. src_offset + block_elements];
            const dst_block = output_bytes[dst_offset .. dst_offset + block_size];

            func(src_block, dst_block);
        }
    }

    fn quantizeBlockQ8_0(src: []const f32, dst: []u8) void {
        const block_size = 32; // Number of elements per block in Q8_0 format
        if (src.len != block_size) return; // Ensure the input length matches the expected block size

        _ = ggml.ggml_quantize_chunk(@intFromEnum(gguf.GgmlType.q8_0), src.ptr, dst.ptr, 0, 1, block_size, null);
    }

    fn quantizeBlockQ5_0(src: []const f32, dst: []u8) void {
        const block_size = 32; // QK5_0
        if (src.len != block_size) return;

        _ = ggml.ggml_quantize_chunk(@intFromEnum(gguf.GgmlType.q5_0), src.ptr, dst.ptr, 0, 1, block_size, null);
    }

    fn quantizeBlockQ4_0(src: []const f32, dst: []u8) void {
        const block_size = 32; // QK4_0
        if (src.len != block_size) return;

        _ = ggml.ggml_quantize_chunk(@intFromEnum(gguf.GgmlType.q4_0), src.ptr, dst.ptr, 0, 1, block_size, null);
    }

    fn quantizeBlockQ5_1(src: []const f32, dst: []u8) void {
        const block_size = 32; // QK5_1
        if (src.len != block_size) return;

        _ = ggml.ggml_quantize_chunk(@intFromEnum(gguf.GgmlType.q5_1), src.ptr, dst.ptr, 0, 1, block_size, null);
    }

    fn quantizeBlockQ4_1(src: []const f32, dst: []u8) void {
        const block_size = 32; // QK5_1
        if (src.len != block_size) return;

        _ = ggml.ggml_quantize_chunk(@intFromEnum(gguf.GgmlType.q4_1), src.ptr, dst.ptr, 0, 1, block_size, null);
    }

    fn fp8_e4m3_to_f32(x: u8) f32 {
        const sign = @as(f32, @floatFromInt((x >> 7) & 0x1));
        const exp = (x >> 3) & 0xF;
        const mant = x & 0x7;

        if (exp == 0) {
            // subnormal
            const m = @as(f32, @floatFromInt(mant)) / 8.0;
            return (1.0 - 2.0 * sign) * m * std.math.pow(f32, 2.0, -6.0);
        } else if (exp == 0xF) {
            // inf or nan
            if (mant == 0) return std.math.inf(f32) * (1.0 - 2.0 * sign);
            return std.math.nan(f32);
        } else {
            const e = @as(f32, @floatFromInt(exp)) - 7.0;
            const m = 1.0 + @as(f32, @floatFromInt(mant)) / 8.0;
            return (1.0 - 2.0 * sign) * m * std.math.pow(f32, 2.0, e);
        }
    }

    fn fp8_e5m2_to_f32(x: u8) f32 {
        const sign = @as(f32, @floatFromInt((x >> 7) & 0x1));
        const exp = (x >> 2) & 0x1F;
        const mant = x & 0x3;

        if (exp == 0) {
            // subnormal
            const m = @as(f32, @floatFromInt(mant)) / 4.0;
            return (1.0 - 2.0 * sign) * m * std.math.pow(f32, 2.0, -14.0);
        } else if (exp == 0x1F) {
            // inf or nan
            if (mant == 0) return std.math.inf(f32) * (1.0 - 2.0 * sign);
            return std.math.nan(f32);
        } else {
            const e = @as(f32, @floatFromInt(exp)) - 15.0;
            const m = 1.0 + @as(f32, @floatFromInt(mant)) / 4.0;
            return (1.0 - 2.0 * sign) * m * std.math.pow(f32, 2.0, e);
        }
    }

    fn bf16_to_f32(x: u16) f32 {
        const bits = (@as(u32, x) << 16);
        return @bitCast(bits);
    }
};

test "transform f16 to q8_0" {
    const allocator = std.testing.allocator;

    // Load the f16 source file
    const f16_file = try std.fs.cwd().openFile("test-artifact/output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight.f16", .{});
    defer f16_file.close();

    const f16_data = try f16_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(f16_data);

    // Calculate element count (f16 is 2 bytes per element)
    const element_count: u64 = @intCast(f16_data.len / 2);

    // Convert f16 to q8_0
    const q8_0_data = try Quantizer.convertTensorData(
        allocator,
        f16_data,
        types.DataType.f16,
        gguf.GgmlType.q8_0,
        element_count,
        1,
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

test "q5_0 quantization debug" {
    // Create a simple test block of 32 values
    var test_input: [32]f32 = undefined;
    // Fill with a simple pattern: values from -16 to +15
    for (0..32) |i| {
        test_input[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 16));
    }

    var output: [22]u8 = undefined;

    std.debug.print("\n=== Q5_0 Quantization Test ===\n", .{});
    std.debug.print("Input values:\n", .{});
    for (test_input, 0..) |val, i| {
        if (i % 8 == 0) std.debug.print("\n[{:2}]: ", .{i});
        std.debug.print("{d:6.2} ", .{val});
    }
    std.debug.print("\n\n", .{});

    // Find max like the function does
    var amax: f32 = 0.0;
    var max: f32 = 0.0;
    for (test_input) |value| {
        const abs_val = @abs(value);
        if (abs_val > amax) {
            amax = abs_val;
            max = value;
        }
    }

    const d = max / -16.0;
    const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

    std.debug.print("amax = {d}, max = {d}\n", .{amax, max});
    std.debug.print("d = {d}, id = {d}\n", .{d, id});

    // Call the function
    Quantizer.quantizeBlockQ5_0(&test_input, &output);

    // Print the output
    std.debug.print("\nOutput bytes:\n", .{});
    std.debug.print("Scale (f16): 0x{X:0>2}{X:0>2}\n", .{output[0], output[1]});

    std.debug.print("\nqs (packed 4-bit, 16 bytes):\n", .{});
    for (output[2..18], 0..) |byte, i| {
        if (i % 8 == 0) std.debug.print("\n", .{});
        const lo = byte & 0x0F;
        const hi = (byte >> 4) & 0x0F;
        std.debug.print("{X:1}{X:1} ", .{lo, hi});
    }

    std.debug.print("\n\nqh (high bits, 4 bytes): ", .{});
    const qh = std.mem.readInt(u32, output[18..22], .little);
    std.debug.print("0x{X:0>8}\n", .{qh});
    std.debug.print("Binary: {b:0>32}\n", .{qh});

    // Decode and verify
    std.debug.print("\nDecoded values:\n", .{});
    const d_f16 = std.mem.bytesAsValue(f16, output[0..2]).*;
    const d_decoded: f32 = @floatCast(d_f16);

    for (0..16) |j| {
        const byte = output[2 + j];
        const q0_low = byte & 0x0F;
        const q1_low = (byte >> 4) & 0x0F;

        const q0_high = (qh >> @intCast(j)) & 1;
        const q1_high = (qh >> @intCast(j + 16)) & 1;

        const q0_full = q0_low | (@as(u8, @intCast(q0_high)) << 4);
        const q1_full = q1_low | (@as(u8, @intCast(q1_high)) << 4);

        const v0 = (@as(f32, @floatFromInt(q0_full)) - 16.0) * d_decoded;
        const v1 = (@as(f32, @floatFromInt(q1_full)) - 16.0) * d_decoded;

        std.debug.print("[{:2}]: {d:6.2} (q={:2}) -> {d:6.2}  |  ", .{j, test_input[j], q0_full, v0});
        std.debug.print("[{:2}]: {d:6.2} (q={:2}) -> {d:6.2}\n", .{j+16, test_input[j+16], q1_full, v1});
    }
    std.debug.print("\n\n\n", .{});
}

test "q5_0 real world data" {
    const allocator = std.testing.allocator;

    // Load the f16 source file
    const f16_file = try std.fs.cwd().openFile("test-artifact/output_blocks.1.1.transformer_blocks.1.attn1.to_q.weight.f16", .{});
    defer f16_file.close();

    const f16_data = try f16_file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(f16_data);

    // Convert just the first block (32 values) to f32
    const input_f16 = std.mem.bytesAsSlice(f16, f16_data[0..64]); // 32 * 2 bytes
    var f32_block: [32]f32 = undefined;
    for (input_f16, 0..) |val, i| {
        f32_block[i] = @floatCast(val);
    }

    // Quantize using our function
    var our_output: [22]u8 = undefined;
    Quantizer.quantizeBlockQ5_0(&f32_block, &our_output);

    std.debug.print("\n=== First block from real data ===\n", .{});
    std.debug.print("First 32 f32 values: ", .{});
    for (f32_block[0..]) |v| {
        std.debug.print("{d:.6} ", .{v});
    }
    std.debug.print("\n", .{});

    // Find max like the function does
    var amax: f32 = 0.0;
    var max: f32 = 0.0;
    for (input_f16) |value| {
        const abs_val = @abs(value);
        if (abs_val > amax) {
            amax = abs_val;
            max = value;
        }
    }

    const d = max / -16.0;
    const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

    const d16: f16 = @floatCast(d);
    // Convert f16 to its raw bits (u16)
    const d16_bits: u16 = @bitCast(d16);

    std.debug.print("amax = {d}, max = {d}\n", .{amax, max});
    std.debug.print("d = {d}, id = {d}, d16 = {d}\n", .{d, id, d16});
    std.debug.print("d16 as hex: {X:0>4}\n", .{d16_bits});

    std.debug.print("Our output (first 22 bytes):\n", .{});
    for (our_output, 0..) |b, i| {
        std.debug.print("{X:0>2} ", .{b});
        if ((i + 1) % 16 == 0) std.debug.print("\n", .{});
    }
    std.debug.print("\n\n\n", .{});
}

test "q5_0 round trip accuracy" {
    // Create test data
    var test_input: [32]f32 = undefined;
    for (0..32) |i| {
        // Use a mix of values to test the full range
        test_input[i] = (@as(f32, @floatFromInt(i)) - 16.0) * 0.05;
    }

    // Quantize
    var quantized: [22]u8 = undefined;
    Quantizer.quantizeBlockQ5_0(&test_input, &quantized);

    // Dequantize manually
    const scale = std.mem.bytesAsValue(f16, quantized[0..2]).*;
    const scale_f32: f32 = @floatCast(scale);
    const qh = std.mem.readInt(u32, quantized[18..22], .little);

    var reconstructed: [32]f32 = undefined;
    for (0..16) |j| {
        const byte = quantized[2 + j];
        const q0_low = byte & 0x0F;
        const q1_low = (byte >> 4) & 0x0F;

        const q0_high = (qh >> @intCast(j)) & 1;
        const q1_high = (qh >> @intCast(j + 16)) & 1;

        const q0_full = q0_low | (@as(u8, @intCast(q0_high)) << 4);
        const q1_full = q1_low | (@as(u8, @intCast(q1_high)) << 4);

        reconstructed[j] = (@as(f32, @floatFromInt(q0_full)) - 16.0) * scale_f32;
        reconstructed[j + 16] = (@as(f32, @floatFromInt(q1_full)) - 16.0) * scale_f32;
    }

    // Check error
    var max_error: f32 = 0.0;
    for (test_input, reconstructed) |orig, recon| {
        const err = @abs(orig - recon);
        max_error = @max(max_error, err);
    }

    std.debug.print("Max reconstruction error: {d}\n", .{max_error});

    // For Q5_0, we expect some quantization error, but it should be reasonable
    try std.testing.expect(max_error < 0.05); // Adjust threshold as needed
}
