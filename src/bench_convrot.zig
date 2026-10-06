// Times the ConvRot-family weight quantizers on real BF16 checkpoint tensors and prints an
// FNV-1a hash of every output, so a speed change can be checked for byte-identical output.
//
//   zig build bench-convrot -Doptimize=ReleaseFast -- <model.safetensors> [tensor names...]
//
// Unlike the other benches this one follows -Doptimize, since the build mode is what it measures.

const std = @import("std");
const DataTransform = @import("DataTransform.zig");
const TensorClusters = @import("TensorClusters.zig");
const ThreadPool = @import("ThreadPool.zig");

const Q = DataTransform.Quantizer;

const default_names = [_][]const u8{
    "model.language_model.layers.10.mlp.down_proj.weight",
    "model.language_model.layers.10.mlp.gate_proj.weight",
    "model.language_model.layers.10.mlp.up_proj.weight",
    "model.language_model.layers.10.self_attn.q_proj.weight",
    "model.language_model.layers.10.self_attn.k_proj.weight",
    "model.language_model.layers.10.self_attn.v_proj.weight",
    "model.language_model.layers.10.self_attn.o_proj.weight",
};

const Tensor = struct { data: []f32, rows: usize, cols: usize };

fn loadBf16(io: std.Io, allocator: std.mem.Allocator, path: []const u8, name: []const u8) !Tensor {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);

    var len_buf: [8]u8 = undefined;
    _ = try file.readPositionalAll(io, &len_buf, 0);
    const header_size = std.mem.readInt(u64, &len_buf, .little);
    const json_buf = try allocator.alloc(u8, header_size);
    defer allocator.free(json_buf);
    _ = try file.readPositionalAll(io, json_buf, 8);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_buf, .{});
    defer parsed.deinit();
    const t = parsed.value.object.get(name) orelse return error.TensorNotFound;
    if (!std.mem.eql(u8, t.object.get("dtype").?.string, "BF16")) return error.NotBf16;
    const shape = t.object.get("shape").?.array.items;
    if (shape.len != 2) return error.NotRank2;
    const rows: usize = @intCast(shape[0].integer);
    const cols: usize = @intCast(shape[1].integer);
    const off: u64 = @intCast(t.object.get("data_offsets").?.array.items[0].integer);

    const raw = try allocator.alloc(u16, rows * cols);
    defer allocator.free(raw);
    _ = try file.readPositionalAll(io, std.mem.sliceAsBytes(raw), 8 + header_size + off);

    const data = try allocator.alloc(f32, rows * cols);
    for (raw, data) |b, *f| f.* = @bitCast(@as(u32, b) << 16);
    return .{ .data = data, .rows = rows, .cols = cols };
}

fn fnv(h: *u64, bytes: []const u8) void {
    for (bytes) |b| {
        h.* ^= b;
        h.* *%= 0x100000001b3;
    }
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

const Kind = enum { int8_convrot, int4_convrot, asym_w4a8, w6a8 };

/// Quantizes one tensor, folding every output buffer into `h`. Returns elapsed nanoseconds.
fn runOne(io: std.Io, allocator: std.mem.Allocator, kind: Kind, t: Tensor, pool: *ThreadPool.ThreadPool, h: *u64) !i96 {
    const t0 = nowNs(io);
    switch (kind) {
        .int8_convrot => {
            const r = try Q.quantizeToInt8(allocator, t.data, t.rows, t.cols, true, TensorClusters.int8_convrot_group_size, pool);
            const dt = nowNs(io) - t0;
            defer allocator.free(r.weight);
            defer allocator.free(r.scale);
            fnv(h, r.weight);
            fnv(h, std.mem.sliceAsBytes(r.scale));
            return dt;
        },
        .int4_convrot => {
            const r = try Q.quantizeToInt4(allocator, t.data, t.rows, t.cols, true, TensorClusters.int4_convrot_group_size, 0, pool);
            const dt = nowNs(io) - t0;
            defer allocator.free(r.weight);
            defer allocator.free(r.scale);
            fnv(h, r.weight);
            fnv(h, std.mem.sliceAsBytes(r.scale));
            return dt;
        },
        .asym_w4a8 => {
            const r = try Q.quantizeToAsymW4a8(allocator, t.data, t.rows, t.cols, TensorClusters.asym_w4a8_convrot_group_size, pool);
            const dt = nowNs(io) - t0;
            defer allocator.free(r.weight);
            defer allocator.free(r.s_rel);
            defer allocator.free(r.s_channel);
            fnv(h, r.weight);
            fnv(h, r.s_rel);
            fnv(h, std.mem.sliceAsBytes(r.s_channel));
            fnv(h, &.{@intFromBool(r.codebook_fit_recommended)});
            return dt;
        },
        .w6a8 => {
            const r = try Q.quantizeToW6a8(allocator, t.data, t.rows, t.cols, TensorClusters.asym_w4a8_convrot_group_size, pool);
            const dt = nowNs(io) - t0;
            defer allocator.free(r.weight);
            defer allocator.free(r.s_rel);
            defer allocator.free(r.s_channel);
            fnv(h, r.weight);
            fnv(h, r.s_rel);
            fnv(h, std.mem.sliceAsBytes(r.s_channel));
            return dt;
        },
    }
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.smp_allocator;
    const print = std.debug.print;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        print("usage: bench-convrot <model.safetensors> [tensor names...]\n", .{});
        return error.MissingArgument;
    }
    const names: []const []const u8 = if (args.len > 2) args[2..] else &default_names;

    var tensors: std.ArrayList(Tensor) = .empty;
    defer {
        for (tensors.items) |t| allocator.free(t.data);
        tensors.deinit(allocator);
    }
    var n_elems: usize = 0;
    for (names) |name| {
        const t = try loadBf16(io, allocator, args[1], name);
        try tensors.append(allocator, t);
        n_elems += t.data.len;
    }

    const n_threads = std.Thread.getCpuCount() catch 4;
    var pool: ThreadPool.ThreadPool = .{};
    try pool.init(.{ .allocator = allocator, .n_jobs = n_threads });
    defer pool.deinit();

    print("{d} tensors, {d} elements, {d} threads, {s}\n", .{ tensors.items.len, n_elems, n_threads, @tagName(@import("builtin").mode) });

    const reps = 3;
    for ([_]Kind{ .int8_convrot, .int4_convrot, .asym_w4a8, .w6a8 }) |kind| {
        var best: i96 = std.math.maxInt(i96);
        var hash: u64 = 0;
        for (0..reps) |_| {
            var h: u64 = 0xcbf29ce484222325;
            var total: i96 = 0;
            for (tensors.items) |t| total += try runOne(io, allocator, kind, t, &pool, &h);
            best = @min(best, total);
            hash = h;
        }
        const secs = @as(f64, @floatFromInt(best)) / std.time.ns_per_s;
        const ns_per = @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(n_elems));
        print("{s:<14} {d:>8.3} s  {d:>7.2} ns/elem  hash {x:0>16}\n", .{ @tagName(kind), secs, ns_per, hash });
    }
}
