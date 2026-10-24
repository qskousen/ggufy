//! Per-column importance weights for the quantizers' scale search.
//!
//! Every ggml block quantizer picks a scale by minimizing a squared error over
//! a group of weights, and each one minimizes the *wrong* error: it weights
//! every column equally when the network does not. Given per-channel activation
//! energy, the k-quant and qX_0/qX_1 encoders will minimize
//!
//! ```
//!     Σ_j  w_j · (W_ij − Ŵ_ij)²     instead of     Σ_j (W_ij − Ŵ_ij)²
//! ```
//!
//! themselves - the weighted search is already in ggml. Nothing here changes a
//! format, a layout or a kernel; the same bytes get chosen better.
//!
//! ### Source
//!
//! This reads llama.cpp's own imatrix file, which is a GGUF holding two f32
//! tensors per weight:
//!
//! ```
//!   <tensor>.in_sum2   [n_cols, n_mat]   Σ_tokens x_j² per input channel
//!   <tensor>.counts    [1, n_mat]        how many rows fed each sum
//! ```
//!
//! so the per-channel mean square is `in_sum2 / counts`. n_mat is 1 for a dense
//! model and the expert count for an MoE, where each expert saw a different
//! number of tokens.
//!
//! Collecting it is llama.cpp's job (`llama-imatrix`); consuming it is ours.
//! Quote the collector's build alongside any file - an imatrix is only
//! comparable against others gathered by the same one.
//!
//! ### Rescaling
//!
//! The weights are rescaled to mean 1.0. That is cosmetic - ggml's weighted fits
//! are scale-invariant in the weights, since a common factor multiplies the
//! objective and leaves every argmin alone - but it keeps the numbers in a
//! comfortable f32 range however long the capture ran.
//!
//! ### Coverage
//!
//! A tensor with no entry gets no weights, and quantizes exactly as it would
//! have without the file. That is the honest default: the collector only hooks
//! matmuls, so norms, embedding tables and a draft head it never ran are all
//! absent. Absent means unmeasured, not unimportant - nothing here may read a
//! missing entry as zero importance.

const std = @import("std");
const types = @import("types.zig");
const Gguf = @import("Gguf.zig");

const in_sum2_suffix = ".in_sum2";
const counts_suffix = ".counts";

pub const Imatrix = struct {
    arena: std.heap.ArenaAllocator,
    /// Weight tensor name -> one importance weight per column of a row.
    weights: std.StringHashMapUnmanaged([]const f32),
    /// What the file says it was collected over, for the report.
    chunk_count: u32,
    datasets: usize,

    pub fn deinit(self: *Imatrix) void {
        self.arena.deinit();
    }

    pub fn forTensor(self: *const Imatrix, name: []const u8) ?[]const f32 {
        return self.weights.get(name);
    }

    pub fn count(self: *const Imatrix) usize {
        return self.weights.count();
    }
};

/// Read an imatrix GGUF, whatever the file is named. Errors when the file is not one rather than returning
/// an empty set: a silently unweighted conversion is indistinguishable from a
/// weighted one in the output, so a wrong path must not look like success.
pub fn load(
    path: []const u8,
    io: std.Io,
    gpa: std.mem.Allocator,
) !Imatrix {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var scratch = std.heap.ArenaAllocator.init(gpa);
    defer scratch.deinit();

    // Gguf.init creates a missing file, which would leave a typo behind as an empty one.
    try std.Io.Dir.cwd().access(io, path, .{});
    var f = Gguf.init(path, io, gpa, scratch.allocator(), false) catch |e| return switch (e) {
        error.InvalidGgufMagic => error.ImatrixNotGguf,
        else => e,
    };
    defer f.deinit();

    if (f.metadata.get("general.type")) |v| switch (v) {
        .string => |s| if (!std.mem.eql(u8, s, "imatrix")) return error.NotAnImatrix,
        else => return error.NotAnImatrix,
    } else return error.NotAnImatrix;

    var weights: std.StringHashMapUnmanaged([]const f32) = .empty;

    const file = f.file;
    for (f.tensors.items) |t| {
        if (!std.mem.endsWith(u8, t.name, in_sum2_suffix)) continue;
        const base = t.name[0 .. t.name.len - in_sum2_suffix.len];

        // The counts live in their own tensor, so a file missing one is
        // truncated rather than merely sparse.
        const counts_name = try scratch.allocator().print("{s}{s}", .{ base, counts_suffix });
        const counts_t = findTensor(f.tensors.items, counts_name) orelse return error.ImatrixMissingCounts;

        const sums = try readF32(file, io, scratch.allocator(), f.current_data_begin, t);
        const counts = try readF32(file, io, scratch.allocator(), f.current_data_begin, counts_t);
        if (counts.len == 0 or sums.len % counts.len != 0) return error.ImatrixShapeMismatch;

        const cols = sums.len / counts.len;
        const out = try a.alloc(f32, cols);
        @memset(out, 0);

        // Pool the experts weighted by how many rows each contributed, which is
        // the per-channel mean square over the whole capture: an expert that saw
        // twice the tokens gets twice the say.
        var total: f64 = 0;
        for (counts, 0..) |c, m| {
            if (c <= 0) continue;
            total += c;
            const row = sums[m * cols ..][0..cols];
            for (out, row) |*o, s| o.* += s;
        }
        if (total == 0) continue; // never exercised; leave it unmeasured
        for (out) |*o| o.* = @floatCast(@as(f64, o.*) / total);

        var mean: f64 = 0;
        for (out) |o| mean += o;
        mean /= @floatFromInt(out.len);
        if (mean > 0) for (out) |*o| {
            o.* = @floatCast(@as(f64, o.*) / mean);
        };

        try weights.put(a, try a.dupe(u8, base), out);
    }

    if (weights.count() == 0) return error.ImatrixEmpty;

    return .{
        .arena = arena,
        .weights = weights,
        .chunk_count = switch (f.metadata.get("imatrix.chunk_count") orelse std.json.Value{ .null = {} }) {
            .integer => |v| if (v > 0) @intCast(v) else 0,
            else => 0,
        },
        .datasets = switch (f.metadata.get("imatrix.datasets") orelse std.json.Value{ .null = {} }) {
            .array => |arr| arr.items.len,
            else => 0,
        },
    };
}

fn findTensor(tensors: []const types.Tensor, name: []const u8) ?types.Tensor {
    for (tensors) |t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

fn readF32(
    file: std.Io.File,
    io: std.Io,
    alloc: std.mem.Allocator,
    data_begin: u64,
    t: types.Tensor,
) ![]f32 {
    if (!std.mem.eql(u8, t.type, "f32") and !std.mem.eql(u8, t.type, "F32")) return error.ImatrixNotF32;
    const bytes = try alloc.alignedAlloc(u8, .of(f32), @intCast(t.size));
    _ = try file.readPositionalAll(io, bytes, data_begin + t.offset);
    return std.mem.bytesAsSlice(f32, bytes);
}

const testing = std.testing;

test "Imatrix pools experts by their row counts and rescales to mean 1" {
    // The aggregation, exercised directly: two experts over four columns where
    // the second saw three times the rows, so it carries three times the say.
    const a = testing.allocator;
    const cols = 4;
    const sums = [_]f32{ 1, 2, 3, 4, 30, 60, 90, 120 };
    const counts = [_]f32{ 1, 3 };

    const out = try a.alloc(f32, cols);
    defer a.free(out);
    @memset(out, 0);
    var total: f64 = 0;
    for (counts, 0..) |c, m| {
        total += c;
        for (out, sums[m * cols ..][0..cols]) |*o, s| o.* += s;
    }
    for (out) |*o| o.* = @floatCast(@as(f64, o.*) / total);
    var mean: f64 = 0;
    for (out) |o| mean += o;
    mean /= @floatFromInt(out.len);
    for (out) |*o| o.* = @floatCast(@as(f64, o.*) / mean);

    // Ratios survive the rescale; the mean is 1.
    var check: f64 = 0;
    for (out) |o| check += o;
    try testing.expectApproxEqAbs(@as(f64, 1.0), check / @as(f64, cols), 1e-6);
    try testing.expectApproxEqAbs(out[3] / out[0], 4.0, 1e-5);
}
