//! Precision report runner — `zig build precision`.
//!
//! Drives the harness across (format × distribution × topology) and prints
//! Markdown tables characterizing how much numeric fidelity each quantization
//! format keeps or loses, and how error behaves under repeated and chained
//! conversions. Self-consistent: every measurement compares a quantize→
//! dequantize result against the original F32 input.
//!
//! Synthetic by default. Pass a real model to run an actual weight tensor
//! through the same matrix:
//!   zig build precision -- --model model.safetensors --tensor blk.0.ffn_down.weight
//!   zig build precision -- --model model.safetensors            (lists candidate tensors)

const std = @import("std");
const h = @import("precision_harness.zig");
const metrics = @import("PrecisionMetrics.zig");
const realdata = @import("precision_realdata.zig");
const ThreadPool = @import("ThreadPool.zig").ThreadPool;

const rows: usize = 128;
const cols: usize = 256;
const n: usize = rows * cols;
/// Fixed seed base so the report is byte-reproducible across runs.
const seed_base: u64 = 0xC0FFEE;

const W = *std.Io.Writer;

fn snrCell(w: W, snr_db: f64) !void {
    if (std.math.isPositiveInf(snr_db)) {
        try w.print(" {s: >8} |", .{"inf"});
    } else if (std.math.isNegativeInf(snr_db)) {
        try w.print(" {s: >8} |", .{"-inf"});
    } else {
        try w.print(" {d: >8.2} |", .{snr_db});
    }
}

fn metricHeader(w: W) !void {
    try w.print("| Format     |  bits |  SNR(dB) |      RMSE |   maxAbs |   cosine |  relFrob |       bias | relLarge |\n", .{});
    try w.print("|------------|-------|----------|-----------|----------|----------|----------|------------|----------|\n", .{});
}

fn metricRow(w: W, name: []const u8, bits: f32, m: metrics.Metrics) !void {
    try w.print("| {s: <10} | {d: >5.2} |", .{ name, bits });
    try snrCell(w, m.snr_db);
    try w.print(" {e: >9.3} | {e: >9.3} | {d: >8.6} | {e: >9.3} | {e: >10.2} | {e: >9.3} |\n", .{
        m.rmse, m.max_abs_err, m.cosine, m.rel_frob_err, m.mean_bias, m.max_rel_err_large,
    });
}

/// Print one full single-trip metric table for a given input tensor.
fn reportTable(w: W, allocator: std.mem.Allocator, input: []const f32, pool: *ThreadPool) !void {
    try metricHeader(w);
    for (h.formats) |spec| {
        const out = h.roundtrip(spec.fmt, allocator, input, rows, cols, pool) catch |err| {
            try w.print("| {s: <10} | {d: >5.2} | error: {s}\n", .{ spec.name, spec.bits, @errorName(err) });
            continue;
        };
        defer allocator.free(out);
        try metricRow(w, spec.name, spec.bits, metrics.compute(input, out));
    }
    try w.flush();
}

/// Cross-format chains: how error compounds and whether order matters.
fn reportChains(w: W, allocator: std.mem.Allocator, input: []const f32, pool: *ThreadPool) !void {
    const Chain = struct { name: []const u8, steps: []const h.Format };
    const chains = [_]Chain{
        .{ .name = "BF16→Q8_0→Q4_K", .steps = &.{ .bf16, .q8_0, .q4_k } },
        .{ .name = "MXFP4→Q4_K", .steps = &.{ .mxfp4, .q4_k } },
        .{ .name = "Q4_K→MXFP4", .steps = &.{ .q4_k, .mxfp4 } },
        .{ .name = "F8_E4M3→NVFP4", .steps = &.{ .f8_e4m3, .nvfp4 } },
        .{ .name = "F16→Q8_0→F16", .steps = &.{ .f16, .q8_0, .f16 } },
        .{ .name = "INT8_CR→INT4_CR", .steps = &.{ .int8_convrot, .int4_convrot } },
    };

    try w.print("| Chain               |  SNR(dB) |      RMSE |   cosine |  relFrob |\n", .{});
    try w.print("|---------------------|----------|-----------|----------|----------|\n", .{});
    for (chains) |c| {
        const out = try h.roundtripChain(c.steps, allocator, input, rows, cols, pool);
        defer allocator.free(out);
        const m = metrics.compute(input, out);
        try w.print("| {s: <19} |", .{c.name});
        try snrCell(w, m.snr_db);
        try w.print(" {e: >9.3} | {d: >8.6} | {e: >9.3} |\n", .{ m.rmse, m.cosine, m.rel_frob_err });
    }
    try w.flush();
}

/// Repeated same-format round-trips — does error stabilize? (synthetic only)
fn reportRepeated(w: W, allocator: std.mem.Allocator, pool: *ThreadPool) !void {
    const n_trips = 5;
    const watched = [_]h.Format{ .f8_e4m3, .q8_0, .q4_k, .q2_k, .mxfp4, .mxfp8, .nvfp4, .int8, .int4_convrot };

    try w.print("\n## Repeated round-trips (stability)\n", .{});
    try w.print("\n{d} successive round-trips of the same format on `gaussian_weight`, each measured vs. the original. A stable quantizer reaches a fixed point after trip 1.\n\n", .{n_trips});
    try w.print("| Format       | SNR trip1 | SNR trip{d} |  ΔRMSE(1→{d}) | stabilized |\n", .{ n_trips, n_trips });
    try w.print("|--------------|-----------|-----------|-------------|------------|\n", .{});

    const input = try h.generate(.gaussian_weight, allocator, n, seed_base);
    defer allocator.free(input);

    for (watched) |fmt| {
        var series: [n_trips]metrics.Metrics = undefined;
        try h.roundtripSeries(fmt, allocator, input, rows, cols, pool, &series);
        const drmse = series[n_trips - 1].rmse - series[0].rmse;
        const stable = @abs(drmse) <= series[0].rmse * 0.02;
        try w.print("| {s: <12} |", .{@tagName(fmt)});
        try snrCell(w, series[0].snr_db);
        try snrCell(w, series[n_trips - 1].snr_db);
        try w.print(" {e: >11.3} | {s: >10} |\n", .{ drmse, if (stable) "yes" else "NO" });
    }
    try w.flush();
}

fn reportSynthetic(w: W, allocator: std.mem.Allocator, pool: *ThreadPool) !void {
    try w.print("\n## Single round-trip fidelity\n", .{});
    try w.print("\nShape {d}×{d} ({d} elements). Higher SNR / cosine and lower error is better.\n", .{ rows, cols, n });
    for (h.distributions, 0..) |dist, di| {
        const input = try h.generate(dist, allocator, n, seed_base +% di);
        defer allocator.free(input);
        try w.print("\n### {s}\n\n", .{h.distributionName(dist)});
        try reportTable(w, allocator, input, pool);
    }

    try reportRepeated(w, allocator, pool);

    try w.print("\n## Cross-format chains\n", .{});
    try w.print("\nEach chain applied left-to-right on `gaussian_weight`; final output compared to the original.\n\n", .{});
    const chain_input = try h.generate(.gaussian_weight, allocator, n, seed_base);
    defer allocator.free(chain_input);
    try reportChains(w, allocator, chain_input, pool);
}

const Args = struct {
    model: ?[]const u8 = null,
    tensor: ?[]const u8 = null,
    offset: usize = 0,
};

fn parseArgs(raw: std.process.Args, allocator: std.mem.Allocator) !Args {
    var it = std.process.Args.Iterator.init(raw);
    _ = it.skip(); // program name
    var out: Args = .{};
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model")) {
            out.model = try allocator.dupe(u8, it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--tensor")) {
            out.tensor = try allocator.dupe(u8, it.next() orelse return error.MissingValue);
        } else if (std.mem.eql(u8, arg, "--offset")) {
            out.offset = std.fmt.parseInt(usize, it.next() orelse return error.MissingValue, 10) catch return error.BadOffset;
        }
    }
    return out;
}

fn reportRealData(w: W, allocator: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, args: Args, pool: *ThreadPool) !void {
    const model = args.model.?;

    if (args.tensor == null) {
        try w.print("\n## Tensors in `{s}`\n\n", .{model});
        try w.print("Candidates with ≥ {d} elements. Re-run with `--tensor <name>` to profile one.\n\n", .{n});
        try realdata.listTensors(io, allocator, arena, model, n, w);
        return;
    }

    const tensor_name = args.tensor.?;
    var rt = realdata.loadTensor(io, allocator, arena, model, tensor_name, pool) catch |err| {
        try w.print("\nFailed to load tensor `{s}` from `{s}`: {s}\n", .{ tensor_name, model, @errorName(err) });
        try w.flush();
        return;
    };
    defer rt.deinit(allocator);

    if (rt.values.len < n) {
        try w.print("\nTensor `{s}` has only {d} elements; need at least {d}. Try a larger tensor (run with `--model` alone to list candidates).\n", .{ tensor_name, rt.values.len, n });
        try w.flush();
        return;
    }

    const start = @min(args.offset, rt.values.len - n);
    const window = rt.values[start .. start + n];

    try w.print("\n## Real tensor: `{s}`\n\n", .{tensor_name});
    try w.print("Source dtype `{s}`, {d} elements; profiling a {d}-element window at offset {d} (reshaped {d}×{d}).\n\n", .{ rt.dtype, rt.values.len, n, start, rows, cols });
    try reportTable(w, allocator, window, pool);

    try w.print("\n### Cross-format chains on this tensor\n\n", .{});
    try reportChains(w, allocator, window, pool);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;
    const arena = init.arena.allocator();

    const args = try parseArgs(init.minimal.args, arena);

    var pool: ThreadPool = .{};
    try pool.init(.{ .allocator = allocator, .n_jobs = 8 });
    defer pool.deinit();

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const w = &stdout_writer.interface;

    try w.print("# ggufy precision report\n", .{});
    try w.print("\nQuantize→dequantize round-trip fidelity across formats, distributions, and conversion chains. Self-consistent (measured against the original F32 input).\n", .{});
    try w.flush();

    if (args.model != null) {
        try reportRealData(w, allocator, arena, io, args, &pool);
    } else {
        try reportSynthetic(w, allocator, &pool);
    }

    try w.flush();
}
