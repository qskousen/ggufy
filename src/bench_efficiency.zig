const std = @import("std");
const DataTransform = @import("DataTransform.zig");
const ScaledQuant = @import("TensorClusters.zig");
const types = @import("types.zig");
const ThreadPool = @import("ThreadPool.zig");

const MODEL_PATH = "test-models/perfectdeliberate_v20.safetensors";
const TENSOR_NAME = "model.diffusion_model.output_blocks.5.2.conv.weight";

// F32 [1280, 1280, 3, 3] = 14,745,600 elements
// Divisible by 256 (k-quant block size), 128 (NVFP4 row req), 64 (NVFP4 col req)
const ROWS: usize = 1280;
const COLS: usize = 11520; // 1280 * 3 * 3
const N: usize = ROWS * COLS;

const Target = struct {
    group: []const u8,
    name: []const u8,
    dtype: types.DataType,
};

const targets = [_]Target{
    .{ .group = "ST",   .name = "F16",     .dtype = .F16 },
    .{ .group = "ST",   .name = "BF16",    .dtype = .BF16 },
    .{ .group = "ST",   .name = "F8_E4M3", .dtype = .F8_E4M3 },
    .{ .group = "ST",   .name = "F8_E5M2", .dtype = .F8_E5M2 },
    .{ .group = "ST",   .name = "F4_E2M1", .dtype = .F4_E2M1 },
    .{ .group = "GGUF", .name = "f16",     .dtype = .f16 },
    .{ .group = "GGUF", .name = "bf16",    .dtype = .bf16 },
    .{ .group = "GGUF", .name = "q8_0",    .dtype = .q8_0 },
    .{ .group = "GGUF", .name = "q6_k",    .dtype = .q6_k },
    .{ .group = "GGUF", .name = "q5_1",    .dtype = .q5_1 },
    .{ .group = "GGUF", .name = "q5_0",    .dtype = .q5_0 },
    .{ .group = "GGUF", .name = "q5_k",    .dtype = .q5_k },
    .{ .group = "GGUF", .name = "q4_1",    .dtype = .q4_1 },
    .{ .group = "GGUF", .name = "q4_0",    .dtype = .q4_0 },
    .{ .group = "GGUF", .name = "q4_k",    .dtype = .q4_k },
    .{ .group = "GGUF", .name = "mxfp4",   .dtype = .mxfp4 },
    .{ .group = "GGUF", .name = "q3_k",    .dtype = .q3_k },
    .{ .group = "GGUF", .name = "q2_k",    .dtype = .q2_k },
};

fn loadSafetensorsF32(io: std.Io, allocator: std.mem.Allocator, path: []const u8, name: []const u8) ![]f32 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var rdr = file.reader(io, &read_buf);

    const hdr_len_bytes = try rdr.interface.readAlloc(allocator, 8);
    defer allocator.free(hdr_len_bytes);
    const header_size = std.mem.readInt(u64, hdr_len_bytes[0..8], .little);

    const json_buf = try rdr.interface.readAlloc(allocator, header_size);
    defer allocator.free(json_buf);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_buf, .{});
    defer parsed.deinit();

    const tensor_val = parsed.value.object.get(name) orelse return error.TensorNotFound;
    const offsets_val = tensor_val.object.get("data_offsets") orelse return error.MissingDataOffsets;
    const off_start: u64 = @intCast(offsets_val.array.items[0].integer);
    const off_end: u64 = @intCast(offsets_val.array.items[1].integer);
    const byte_count = off_end - off_start;
    if (byte_count % @sizeOf(f32) != 0) return error.UnalignedF32Data;

    const data = try allocator.alloc(f32, byte_count / @sizeOf(f32));
    errdefer allocator.free(data);

    const n_read = try file.readPositionalAll(io, std.mem.sliceAsBytes(data), 8 + header_size + off_start);
    if (n_read < std.mem.sliceAsBytes(data).len) return error.UnexpectedEof;

    return data;
}

fn printMetrics(
    group: []const u8,
    name: []const u8,
    original: []const f32,
    decoded: []const f32,
    signal_power: f64,
    total_bytes: usize,
) void {
    const bits = @as(f64, @floatFromInt(total_bytes * 8)) / @as(f64, @floatFromInt(original.len));
    const compression = 32.0 / bits;
    var mse: f64 = 0;
    var max_err: f32 = 0;
    for (original, decoded) |o, d| {
        const err = o - d;
        mse += @as(f64, err) * @as(f64, err);
        const abs_err: f32 = if (err < 0) -err else err;
        if (abs_err > max_err) max_err = abs_err;
    }
    mse /= @as(f64, @floatFromInt(original.len));
    const psnr: f64 = if (mse > 0) 10.0 * std.math.log10(signal_power / mse) else std.math.inf(f64);
    std.debug.print("{s:<6} {s:<16} {d:>9.2}  {d:>6.2}x   {e:<12.3} {d:>10.1}   {d:>10.6}\n", .{
        group, name, bits, compression, mse, psnr, max_err,
    });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const n_threads = std.Thread.getCpuCount() catch 4;
    var pool: ThreadPool.ThreadPool = .{};
    try pool.init(.{ .allocator = allocator, .n_jobs = n_threads });
    defer pool.deinit();

    const print = std.debug.print;

    print("Loading {s}\n", .{MODEL_PATH});
    print("Tensor:  {s}\n", .{TENSOR_NAME});

    const original_all = try loadSafetensorsF32(io, allocator, MODEL_PATH, TENSOR_NAME);
    defer allocator.free(original_all);

    if (original_all.len < N) {
        print("Error: tensor has {d} elements, benchmark needs {d}\n", .{ original_all.len, N });
        return error.TensorTooSmall;
    }
    const original = original_all[0..N];

    // Signal power (mean square of original) used as PSNR reference
    var signal_power: f64 = 0;
    for (original) |v| signal_power += @as(f64, v) * @as(f64, v);
    signal_power /= @as(f64, N);

    const src_bytes = std.mem.sliceAsBytes(original);

    print("\nQuantization Round-Trip Efficiency  (N={d}, real F32 model weights)\n\n", .{N});
    print("{s:<6} {s:<16} {s:>9}  {s:>7}   {s:<12} {s:>10}   {s:>10}\n",
        .{ "Group", "Type", "Bits/elem", "vs F32", "MSE", "PSNR(dB)", "Max|err|" });
    print("{s}\n", .{"-" ** 78});

    // -------------------------------------------------------------------------
    // Simple single-tensor types via convertTensorData
    // -------------------------------------------------------------------------
    for (targets) |t| {
        const encoded = DataTransform.Quantizer.convertTensorData(
            allocator, src_bytes, .F32, t.dtype, N, &pool,
        ) catch |err| {
            print("{s:<6} {s:<16}  (encode failed: {})\n", .{ t.group, t.name, err });
            continue;
        };
        defer allocator.free(encoded);

        const decoded_bytes = DataTransform.Quantizer.convertTensorData(
            allocator, encoded, t.dtype, .F32, N, &pool,
        ) catch |err| {
            print("{s:<6} {s:<16}  (decode failed: {})\n", .{ t.group, t.name, err });
            continue;
        };
        defer allocator.free(decoded_bytes);

        const decoded: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(decoded_bytes)));
        printMetrics(t.group, t.name, original, decoded, signal_power, encoded.len);
    }

    // -------------------------------------------------------------------------
    // ST SCALED_F8_E4M3: F8_E4M3 weight + single F32 global scale
    // Storage: N bytes weight + 4 bytes scale
    // -------------------------------------------------------------------------
    {
        const enc = try DataTransform.Quantizer.quantizeToComfyFp8(allocator, original, &pool);
        defer allocator.free(enc.weight);

        const decoded = try allocator.alloc(f32, N);
        defer allocator.free(decoded);
        for (enc.weight, decoded) |byte, *out|
            out.* = DataTransform.Quantizer.lut_e4m3[byte] * enc.scale;

        printMetrics("ST", "SCALED_F8_E4M3", original, decoded, signal_power, enc.weight.len + 4);
    }

    // -------------------------------------------------------------------------
    // ST MXFP8_E4M3: F8_E4M3 weight + E8M0 per-block scale (block=32)
    // Storage: N bytes weight + N/32 bytes scale = 8.25 bits/elem
    // -------------------------------------------------------------------------
    {
        const enc = try DataTransform.Quantizer.quantizeToComfyMxfp8(allocator, original, &pool);
        defer allocator.free(enc.weight);
        defer allocator.free(enc.scale);

        const decoded = try allocator.alloc(f32, N);
        defer allocator.free(decoded);
        for (enc.weight, decoded, 0..) |byte, *out, i|
            out.* = DataTransform.Quantizer.lut_e4m3[byte] *
                DataTransform.Quantizer.e8m0_to_f32(enc.scale[i / 32]);

        printMetrics("ST", "MXFP8_E4M3", original, decoded, signal_power, enc.weight.len + enc.scale.len);
    }

    // -------------------------------------------------------------------------
    // ST NVFP4: FP4 nibbles + F8_E4M3 per-16-elem scale + F32 global scale
    // Storage: N/2 bytes weight + N/16 bytes scale + 4 bytes global = ~4.5 bits/elem
    // Shape constraint: rows must be a multiple of 128, cols a multiple of 64.
    // -------------------------------------------------------------------------
    {
        const enc = try ScaledQuant.quantizeToNvFp4Raw(original, ROWS, COLS, allocator, &pool);
        defer allocator.free(enc.weight);
        defer allocator.free(enc.scale);

        const decoded = try ScaledQuant.dequantizeFp4Raw(
            enc.weight, enc.scale, enc.global_scale, ROWS, COLS, allocator, &pool,
        );
        defer allocator.free(decoded);

        const total_bytes = enc.weight.len + enc.scale.len + 4; // +4 for global_scale f32
        printMetrics("ST", "NVFP4", original, decoded, signal_power, total_bytes);
    }

    print("\n", .{});
}
