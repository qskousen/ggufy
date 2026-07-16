//! Precision test harness.
//!
//! Wraps every supported quantization format behind a single uniform
//! `roundtrip(fmt, input) -> f32` interface so that the byte-based GGML/FP8
//! paths and the scale-tensor cluster paths (NVFP4, MXFP8, INT4/INT8) look
//! identical to the test matrix. This makes multi-trip and cross-format
//! chaining trivial: every stage is just `[]f32 -> []f32`.
//!
//! It also provides reproducible synthetic data generators spanning the value
//! distributions that stress quantizers differently (Gaussian weights, heavy
//! tails, near-zero activations, and an exactly-representable sanity floor).
//!
//! The `runReport`-facing pieces live here as a pure library; the report
//! executable (`precision_main.zig`) formats the numbers, and the regression
//! `test` blocks at the bottom lock in behaviour for `zig build test`.

const std = @import("std");
const types = @import("types.zig");
const DataTransform = @import("DataTransform.zig");
const TC = @import("TensorClusters.zig");
const ThreadPool = @import("ThreadPool.zig").ThreadPool;
const metrics = @import("PrecisionMetrics.zig");

const Q = DataTransform.Quantizer;

/// Hadamard rotation group size for the ConvRot INT formats. Must be a power
/// of 4 that divides the column count; 64 divides the harness's 256 cols.
const convrot_group_size: usize = 64;

// ---------------------------------------------------------------------------
// Formats
// ---------------------------------------------------------------------------

pub const Format = enum {
    f16,
    bf16,
    f8_e4m3,
    f8_e5m2,
    scaled_f8_e4m3,
    q8_0,
    q6_k,
    q5_k,
    q4_k,
    q3_k,
    q2_k,
    q5_0,
    q4_0,
    mxfp4,
    mxfp8,
    nvfp4,
    int8,
    int8_convrot,
    int4,
    int4_convrot,
};

pub const FormatSpec = struct {
    fmt: Format,
    name: []const u8,
    /// Nominal on-disk bits per weight (including block scale overhead). Used
    /// only for the report's "bits" column — not for correctness.
    bits: f32,
};

pub const formats = [_]FormatSpec{
    .{ .fmt = .f16, .name = "F16", .bits = 16.0 },
    .{ .fmt = .bf16, .name = "BF16", .bits = 16.0 },
    .{ .fmt = .f8_e4m3, .name = "F8_E4M3", .bits = 8.0 },
    .{ .fmt = .f8_e5m2, .name = "F8_E5M2", .bits = 8.0 },
    .{ .fmt = .scaled_f8_e4m3, .name = "SCALED_F8", .bits = 8.0 },
    .{ .fmt = .q8_0, .name = "Q8_0", .bits = 8.5 },
    .{ .fmt = .q6_k, .name = "Q6_K", .bits = 6.5625 },
    .{ .fmt = .q5_k, .name = "Q5_K", .bits = 5.5 },
    .{ .fmt = .q4_k, .name = "Q4_K", .bits = 4.5 },
    .{ .fmt = .q3_k, .name = "Q3_K", .bits = 3.4375 },
    .{ .fmt = .q2_k, .name = "Q2_K", .bits = 2.625 },
    .{ .fmt = .q5_0, .name = "Q5_0", .bits = 5.5 },
    .{ .fmt = .q4_0, .name = "Q4_0", .bits = 4.5 },
    .{ .fmt = .mxfp4, .name = "MXFP4", .bits = 4.25 },
    .{ .fmt = .mxfp8, .name = "MXFP8", .bits = 8.25 },
    .{ .fmt = .nvfp4, .name = "NVFP4", .bits = 4.5 },
    .{ .fmt = .int8, .name = "INT8", .bits = 8.03 },
    .{ .fmt = .int8_convrot, .name = "INT8_CR", .bits = 8.03 },
    .{ .fmt = .int4, .name = "INT4", .bits = 4.03 },
    .{ .fmt = .int4_convrot, .name = "INT4_CR", .bits = 4.03 },
};

/// The GGUF datatype used for each byte-based format's round-trip.
/// Returns null for the scale-tensor cluster formats, which take a bespoke path.
fn ggufDstType(fmt: Format) ?types.DataType {
    return switch (fmt) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f8_e4m3 => .F8_E4M3,
        .f8_e5m2 => .F8_E5M2,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .q5_k => .q5_k,
        .q4_k => .q4_k,
        .q3_k => .q3_k,
        .q2_k => .q2_k,
        .q5_0 => .q5_0,
        .q4_0 => .q4_0,
        .mxfp4 => .mxfp4,
        else => null,
    };
}

// ---------------------------------------------------------------------------
// Round-trip: quantize then dequantize, returning an owned []f32.
// ---------------------------------------------------------------------------

/// Read a native little-endian f32 out of a byte buffer without alignment
/// assumptions (allocator byte buffers are not guaranteed 4-byte aligned).
inline fn readF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
}

/// Quantize `input` to `fmt` and dequantize back to F32. Caller owns the result.
/// `rows`×`cols` must equal `input.len`; the shape matters for the per-row and
/// block-tiled cluster formats. 128×256 satisfies every format's constraints.
pub fn roundtrip(
    fmt: Format,
    allocator: std.mem.Allocator,
    input: []const f32,
    rows: usize,
    cols: usize,
    pool: *ThreadPool,
) ![]f32 {
    std.debug.assert(input.len == rows * cols);

    if (ggufDstType(fmt)) |dst| return roundtripBytes(dst, allocator, input, pool);

    return switch (fmt) {
        .scaled_f8_e4m3 => blk: {
            const enc = try Q.quantizeToComfyFp8(allocator, input, pool);
            defer allocator.free(enc.weight);
            const deq = try Q.convertTensorData(allocator, enc.weight, .F8_E4M3, .F32, input.len, pool);
            defer allocator.free(deq);
            const out = try allocator.alloc(f32, input.len);
            for (0..input.len) |i| out[i] = readF32(deq, i) * enc.scale;
            break :blk out;
        },
        .mxfp8 => blk: {
            const enc = try Q.quantizeToComfyMxfp8(allocator, input, pool);
            defer allocator.free(enc.weight);
            defer allocator.free(enc.scale);
            break :blk try TC.dequantizeMxfp8Raw(enc.weight, enc.scale, rows, cols, allocator);
        },
        .nvfp4 => blk: {
            const enc = try TC.quantizeToNvFp4Raw(input, rows, cols, allocator, pool);
            defer allocator.free(enc.weight);
            defer allocator.free(enc.scale);
            break :blk try TC.dequantizeFp4Raw(enc.weight, enc.scale, enc.global_scale, rows, cols, allocator, pool);
        },
        .int8, .int8_convrot => blk: {
            const cr = fmt == .int8_convrot;
            const enc = try Q.quantizeToInt8(allocator, input, rows, cols, cr, convrot_group_size, pool);
            defer allocator.free(enc.weight);
            defer allocator.free(enc.scale);
            break :blk try TC.dequantizeInt8ConvrotRaw(enc.weight, enc.scale, rows, cols, cr, convrot_group_size, allocator, pool);
        },
        .int4, .int4_convrot => blk: {
            const cr = fmt == .int4_convrot;
            const enc = try Q.quantizeToInt4(allocator, input, rows, cols, cr, convrot_group_size, 0, pool);
            defer allocator.free(enc.weight);
            defer allocator.free(enc.scale);
            break :blk try TC.dequantizeInt4Raw(enc.weight, enc.scale, rows, cols, cr, convrot_group_size, allocator, pool);
        },
        else => unreachable, // byte-based formats handled above
    };
}

fn roundtripBytes(
    dst: types.DataType,
    allocator: std.mem.Allocator,
    input: []const f32,
    pool: *ThreadPool,
) ![]f32 {
    const n: u64 = @intCast(input.len);
    const in_bytes = std.mem.sliceAsBytes(input);

    const quantized = try Q.convertTensorData(allocator, in_bytes, .F32, dst, n, pool);
    defer allocator.free(quantized);

    const back = try Q.convertTensorData(allocator, quantized, dst, .F32, n, pool);
    defer allocator.free(back);

    const out = try allocator.alloc(f32, input.len);
    for (0..input.len) |i| out[i] = readF32(back, i);
    return out;
}

// ---------------------------------------------------------------------------
// Topologies
// ---------------------------------------------------------------------------

/// Run `input` through `n_trips` repeated round-trips of the same format,
/// filling `out_metrics[i]` with the error of trip i+1 measured against the
/// ORIGINAL input. A stable (idempotent) format flattens after trip 1.
/// `out_metrics.len` must equal `n_trips`.
pub fn roundtripSeries(
    fmt: Format,
    allocator: std.mem.Allocator,
    input: []const f32,
    rows: usize,
    cols: usize,
    pool: *ThreadPool,
    out_metrics: []metrics.Metrics,
) !void {
    var current = try allocator.dupe(f32, input);
    defer allocator.free(current);

    for (out_metrics) |*m| {
        const next = try roundtrip(fmt, allocator, current, rows, cols, pool);
        allocator.free(current);
        current = next;
        m.* = metrics.compute(input, current);
    }
}

/// Apply a sequence of formats to `input`, one round-trip each, returning the
/// final F32 output. Measures how error compounds across a conversion chain.
pub fn roundtripChain(
    chain: []const Format,
    allocator: std.mem.Allocator,
    input: []const f32,
    rows: usize,
    cols: usize,
    pool: *ThreadPool,
) ![]f32 {
    var current = try allocator.dupe(f32, input);
    errdefer allocator.free(current);

    for (chain) |fmt| {
        const next = try roundtrip(fmt, allocator, current, rows, cols, pool);
        allocator.free(current);
        current = next;
    }
    return current;
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

pub const Distribution = enum {
    /// Standard normal N(0,1).
    gaussian_std,
    /// N(0, 0.02) — realistic trained-weight scale.
    gaussian_weight,
    /// Uniform on [-1, 1).
    uniform,
    /// Gaussian body with ~1% of elements scaled 50× — attention-style outliers.
    heavy_tail,
    /// Mostly tiny values with a 5% tail of unit-scale spikes — activation-like.
    near_zero,
    /// Values drawn from the exact FP4 E2M1 representable set. A sanity floor:
    /// float formats (F16/BF16/FP8) must round-trip these near-perfectly.
    exact_fp4,
};

pub const distributions = [_]Distribution{
    .gaussian_std,
    .gaussian_weight,
    .uniform,
    .heavy_tail,
    .near_zero,
    .exact_fp4,
};

pub fn distributionName(d: Distribution) []const u8 {
    return switch (d) {
        .gaussian_std => "gaussian_std",
        .gaussian_weight => "gaussian_weight",
        .uniform => "uniform",
        .heavy_tail => "heavy_tail",
        .near_zero => "near_zero",
        .exact_fp4 => "exact_fp4",
    };
}

/// Generate `n` reproducible samples from `dist`. Same (dist, seed) → same data.
pub fn generate(dist: Distribution, allocator: std.mem.Allocator, n: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const out = try allocator.alloc(f32, n);
    errdefer allocator.free(out);

    switch (dist) {
        .gaussian_std => for (out) |*v| {
            v.* = rng.floatNorm(f32);
        },
        .gaussian_weight => for (out) |*v| {
            v.* = rng.floatNorm(f32) * 0.02;
        },
        .uniform => for (out) |*v| {
            v.* = rng.float(f32) * 2.0 - 1.0;
        },
        .heavy_tail => for (out) |*v| {
            var x = rng.floatNorm(f32) * 0.02;
            if (rng.float(f32) < 0.01) x *= 50.0;
            v.* = x;
        },
        .near_zero => for (out) |*v| {
            v.* = if (rng.float(f32) < 0.05) rng.floatNorm(f32) else rng.floatNorm(f32) * 1e-3;
        },
        .exact_fp4 => {
            const set = [_]f32{ 0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2, 3, -3, 4, -4, 6, -6 };
            for (out) |*v| v.* = set[rng.uintLessThan(usize, set.len)];
        },
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests — regression guards for `zig build test`.
//
// These assert distribution-independent INVARIANTS (more robust to benign
// numeric drift than magic-number thresholds) plus a few conservative floors
// seeded from a measured run. Shape is the harness-standard 128×256.
// ---------------------------------------------------------------------------

const testing = std.testing;
const test_rows: usize = 128;
const test_cols: usize = 256;
const test_n: usize = test_rows * test_cols;

fn testPool() ThreadPool {
    return .{}; // single-job pool; results are split-invariant anyway
}

test "every format round-trips without error and preserves length" {
    var pool = testPool();
    const input = try generate(.gaussian_std, testing.allocator, test_n, 0xABCDEF);
    defer testing.allocator.free(input);

    inline for (formats) |spec| {
        const out = try roundtrip(spec.fmt, testing.allocator, input, test_rows, test_cols, &pool);
        defer testing.allocator.free(out);
        try testing.expectEqual(test_n, out.len);
        // No format should produce NaN/Inf from finite input.
        for (out) |v| try testing.expect(std.math.isFinite(v));
    }
}

test "higher-precision formats keep more signal (SNR ordering)" {
    var pool = testPool();
    const input = try generate(.gaussian_std, testing.allocator, test_n, 0x1234);
    defer testing.allocator.free(input);

    const snr = struct {
        fn of(fmt: Format, in: []const f32, p: *ThreadPool) !f64 {
            const out = try roundtrip(fmt, testing.allocator, in, test_rows, test_cols, p);
            defer testing.allocator.free(out);
            return metrics.compute(in, out).snr_db;
        }
    }.of;

    const f16_snr = try snr(.f16, input, &pool);
    const q8_snr = try snr(.q8_0, input, &pool);
    const q4k_snr = try snr(.q4_k, input, &pool);
    const q2k_snr = try snr(.q2_k, input, &pool);

    // Monotone in bit budget. F16 >> Q8_0 > Q4_K > Q2_K.
    try testing.expect(f16_snr > q8_snr);
    try testing.expect(q8_snr > q4k_snr);
    try testing.expect(q4k_snr > q2k_snr);
    // Absolute floors (dB), conservative vs. a measured baseline.
    try testing.expect(q8_snr > 30.0);
    try testing.expect(q4k_snr > 12.0);
}

test "ConvRot rotation helps on heavy-tailed data" {
    var pool = testPool();
    const input = try generate(.heavy_tail, testing.allocator, test_n, 0x55);
    defer testing.allocator.free(input);

    const plain = try roundtrip(.int4, testing.allocator, input, test_rows, test_cols, &pool);
    defer testing.allocator.free(plain);
    const rotated = try roundtrip(.int4_convrot, testing.allocator, input, test_rows, test_cols, &pool);
    defer testing.allocator.free(rotated);

    const plain_snr = metrics.compute(input, plain).snr_db;
    const rotated_snr = metrics.compute(input, rotated).snr_db;
    // Hadamard rotation spreads outliers, so ConvRot should not be worse and is
    // typically better on heavy tails.
    try testing.expect(rotated_snr >= plain_snr - 0.5);
}

test "float formats round-trip the exact-representable set near-perfectly" {
    var pool = testPool();
    const input = try generate(.exact_fp4, testing.allocator, test_n, 0x99);
    defer testing.allocator.free(input);

    // FP4-exact values are also exactly representable in these wider float
    // formats, so error must be essentially zero (very high SNR).
    inline for ([_]Format{ .f16, .bf16, .f8_e4m3 }) |fmt| {
        const out = try roundtrip(fmt, testing.allocator, input, test_rows, test_cols, &pool);
        defer testing.allocator.free(out);
        const m = metrics.compute(input, out);
        try testing.expect(m.snr_db > 60.0);
    }
}

test "repeated round-trips stabilize (idempotent after first trip)" {
    var pool = testPool();
    const input = try generate(.gaussian_weight, testing.allocator, test_n, 0x77);
    defer testing.allocator.free(input);

    // A quantizer applied to its own output should reach a fixed point: trip 5
    // must be no worse than trip 2 by more than rounding noise.
    inline for ([_]Format{ .q8_0, .q4_k, .mxfp8, .int8 }) |fmt| {
        var series: [5]metrics.Metrics = undefined;
        try roundtripSeries(fmt, testing.allocator, input, test_rows, test_cols, &pool, &series);
        try testing.expect(series[4].rmse <= series[1].rmse * 1.05 + 1e-9);
    }
}
