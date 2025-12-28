const std = @import("std");
const gguf = @import("Gguf.zig");

pub const Quantizer = struct {
    // Determine the size of the output buffer needed
    pub fn calcOutputSize(dst_type: gguf.GgmlType, element_count: u64) usize {
        const block_size = dst_type.getBlockSize();
        const type_size = dst_type.getBytesPerBlock();
        // Ensure we round up to full blocks if element_count isn't a perfect multiple
        const blocks = (element_count + block_size - 1) / block_size;
        return @intCast(blocks * type_size);
    }

    // Main entry point: Source -> F32 -> Dest
    pub fn convertTensorData(
        allocator: std.mem.Allocator,
        src_data: []const u8,
        src_type: gguf.GgmlType,
        dst_type: gguf.GgmlType,
        element_count: u64,
    ) ![]u8 {
        // Optimization: Direct copy if types match
        if (src_type == dst_type) {
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

        try quantizeFromF32(f32_buffer, out_buffer, dst_type);

        return out_buffer;
    }

    fn dequantizeToF32(input_bytes: []const u8, output_f32: []f32, src_type: gguf.GgmlType) !void {
        switch (src_type) {
            .f16 => {
                const f16_count = input_bytes.len / 2;
                if (f16_count != output_f32.len) return error.InputSizeMismatch;

                const input_f16 = std.mem.bytesAsSlice(f16, input_bytes);
                for (input_f16, 0..) |val, i| {
                    output_f32[i] = @floatCast(val);
                }
            },
            .f32 => {
                const input_vals = std.mem.bytesAsSlice(f32, input_bytes);
                @memcpy(output_f32, input_vals);
            },
            .f64 => {
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

    fn quantizeFromF32(input_f32: []const f32, output_bytes: []u8, dst_type: gguf.GgmlType) !void {
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
                const block_size = 32;
                const element_count = input_f32.len;
                const block_count = element_count / block_size;

                // Ensure output buffer is large enough
                if (output_bytes.len < block_count * 34) return error.OutputBufferTooSmall;

                var i: usize = 0;
                while (i < block_count) : (i += 1) {
                    const src_offset = i * block_size;
                    const dst_offset = i * 34; // 2 bytes delta + 32 bytes data

                    const src_block = input_f32[src_offset .. src_offset + block_size];
                    const dst_block = output_bytes[dst_offset .. dst_offset + 34];

                    quantizeBlockQ8_0(src_block, dst_block);
                }
            },
            else => return error.UnsupportedDestinationType,
        }
    }

    // Specific implementation for Q8_0
    // Reference: llama.cpp k_quantize_q8_0
    fn quantizeBlockQ8_0(src: []const f32, dst: []u8) void {
        const block_size = 32; // Number of elements per block in Q8_0 format
        if (src.len != block_size) return; // Ensure the input length matches the expected block size

        var amax: f32 = 0.0; // Absolute max value in the block

        for (src) |value| {
            amax = @max(@abs(value), amax);
        }

        const d = amax / ((1 << 7) - 1);
        // TODO: set this as f16 right off the bat rather than converting below?
        const id: f32 = if (d != 0.0) 1.0 / d else 0.0;

        // Store the scale factor in the first 2 bytes of the destination block
        var dst_slice = std.mem.bytesAsSlice(f16, dst[0..2]);
        const d_f16: f16 = @floatCast(d); // Cast f32 to f16 directly
        dst_slice[0] = d_f16;

        // start at 2 so we don't overwrite the scale factor
        var i: usize = 2;
        for (src) |value| {
            // Quantize the value
            const quantized_val = value * id;
            // round and cast to int
            const qv: i8 = @intFromFloat(@round(quantized_val));
            dst[i] = @bitCast(qv);
            i += 1;
        }
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
        gguf.GgmlType.f16,
        gguf.GgmlType.q8_0,
        element_count,
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
