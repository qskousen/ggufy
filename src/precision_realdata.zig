//! Real-model tensor loading for the precision report.
//!
//! Lets `zig build precision -- --model <path> --tensor <name>` run an actual
//! weight tensor through the same round-trip matrix as the synthetic data.
//! The tensor is read at its stored dtype and dequantized to F32; a contiguous
//! window of it is then handed to the harness. Keeps the "real data" path a
//! runtime option rather than a committed fixture, so any local model works.

const std = @import("std");
const types = @import("types.zig");
const Safetensors = @import("Safetensor.zig");
const Gguf = @import("Gguf.zig");
const DataTransform = @import("DataTransform.zig");
const ThreadPool = @import("ThreadPool.zig").ThreadPool;

pub const RealTensor = struct {
    /// Full dequantized tensor, F32. Caller owns.
    values: []f32,
    /// The dtype the tensor was stored as (e.g. "F16", "BF16", "q4_k").
    dtype: []const u8,

    pub fn deinit(self: *RealTensor, allocator: std.mem.Allocator) void {
        allocator.free(self.values);
        allocator.free(self.dtype);
    }
};

fn elemCount(t: types.Tensor) u64 {
    var c: u64 = 1;
    for (t.dims) |d| c *= d;
    return c;
}

/// Read `tensor`'s raw bytes from `source` (a `*Safetensors` or `*Gguf`) and
/// dequantize to an owned F32 slice.
fn readAndDequant(
    source: anytype,
    tensor: types.Tensor,
    allocator: std.mem.Allocator,
    pool: *ThreadPool,
) ![]f32 {
    const file = try source.openFileForTensor(tensor.name);
    const buf = try allocator.alloc(u8, tensor.size);
    defer allocator.free(buf);
    _ = try file.readPositionalAll(source.io, buf, tensor.offset + source.current_data_begin);

    const src_type = types.DataType.fromString(tensor.type) catch return error.UnsupportedTensorDtype;
    const count = elemCount(tensor);

    const f32_bytes = DataTransform.Quantizer.convertTensorData(allocator, buf, src_type, .F32, count, pool) catch
        return error.CannotDequantizeDtype;
    defer allocator.free(f32_bytes);

    const out = try allocator.alloc(f32, @intCast(count));
    for (0..out.len) |i| out[i] = @bitCast(std.mem.readInt(u32, f32_bytes[i * 4 ..][0..4], .little));
    return out;
}

fn openSourceAndFind(
    comptime Source: type,
    src: *Source,
    tensor_name: []const u8,
) !types.Tensor {
    for (src.tensors.items) |t| {
        if (std.mem.eql(u8, t.name, tensor_name)) return t;
    }
    return error.TensorNotFound;
}

/// Load one tensor from `path` as F32. Detects SafeTensors vs GGUF.
pub fn loadTensor(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    path: []const u8,
    tensor_name: []const u8,
    pool: *ThreadPool,
) !RealTensor {
    switch (try detectType(io, path)) {
        .safetensors => {
            var src = try Safetensors.init(path, io, allocator, arena, false, false);
            defer src.deinit();
            const t = try openSourceAndFind(Safetensors, &src, tensor_name);
            return .{ .values = try readAndDequant(&src, t, allocator, pool), .dtype = try allocator.dupe(u8, t.type) };
        },
        .gguf => {
            var src = try Gguf.init(path, io, allocator, arena, false);
            defer src.deinit();
            const t = try openSourceAndFind(Gguf, &src, tensor_name);
            return .{ .values = try readAndDequant(&src, t, allocator, pool), .dtype = try allocator.dupe(u8, t.type) };
        },
    }
}

/// Print every tensor with at least `min_count` elements as a candidate.
pub fn listTensors(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    path: []const u8,
    min_count: u64,
    w: *std.Io.Writer,
) !void {
    switch (try detectType(io, path)) {
        .safetensors => {
            var src = try Safetensors.init(path, io, allocator, arena, false, false);
            defer src.deinit();
            try printCandidates(src.tensors.items, min_count, w);
        },
        .gguf => {
            var src = try Gguf.init(path, io, allocator, arena, false);
            defer src.deinit();
            try printCandidates(src.tensors.items, min_count, w);
        },
    }
}

fn printCandidates(tensors: []const types.Tensor, min_count: u64, w: *std.Io.Writer) !void {
    try w.print("| Tensor | dtype | elements | dims |\n|--------|-------|----------|------|\n", .{});
    for (tensors) |t| {
        const c = elemCount(t);
        if (c < min_count) continue;
        try w.print("| `{s}` | {s} | {d} | ", .{ t.name, t.type, c });
        for (t.dims, 0..) |d, i| {
            if (i != 0) try w.print("×", .{});
            try w.print("{d}", .{d});
        }
        try w.print(" |\n", .{});
    }
    try w.flush();
}

fn detectType(io: std.Io, path: []const u8) !types.FileType {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(io, &read_buffer);
    return types.FileType.detect_from_file(&reader.interface, undefined);
}
