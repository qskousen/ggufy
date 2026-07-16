//! Shared error metrics for comparing a reference F32 tensor against an
//! approximation (e.g. the result of a quantize→dequantize round-trip).
//!
//! All accumulation is done in f64 regardless of the input precision so that
//! large tensors do not lose the small-magnitude error terms to rounding.
//! Every metric operates on two equal-length `[]const f32` slices where
//! `ref` is ground truth and `approx` is the value under test.

const std = @import("std");

pub const Metrics = struct {
    /// Number of elements compared.
    count: usize,
    /// Mean squared error: mean((ref - approx)^2).
    mse: f64,
    /// Root mean squared error: sqrt(mse).
    rmse: f64,
    /// Largest absolute element error: max(|ref - approx|).
    max_abs_err: f64,
    /// Signal-to-noise ratio in decibels: 10·log10(Σref² / Σerr²).
    /// +inf when the approximation is exact (zero error), and this is a
    /// well-defined "perfect" sentinel rather than a divide-by-zero.
    snr_db: f64,
    /// Cosine similarity between the two flattened vectors: 1.0 == identical
    /// direction. Captures how well the *shape* of the tensor is preserved,
    /// which correlates with model quality better than raw magnitude error.
    cosine: f64,
    /// Relative Frobenius error: ‖ref - approx‖ / ‖ref‖.
    rel_frob_err: f64,
    /// Systematic bias: mean(approx - ref). Nonzero indicates the format
    /// pushes values consistently up or down (asymmetric rounding, clamping).
    mean_bias: f64,
    /// Max relative error restricted to "large" elements — those whose
    /// magnitude is ≥ `large_frac` of the reference amax. Small elements have
    /// huge relative errors that are irrelevant to model behaviour; this
    /// isolates how faithfully the dominant weights survive.
    max_rel_err_large: f64,

    pub fn format(self: Metrics, writer: *std.Io.Writer) !void {
        try writer.print(
            "n={d} rmse={e:.4} max_abs={e:.4} snr={d:.2}dB cos={d:.6} rel_frob={e:.4} bias={e:.4} max_rel_large={e:.4}",
            .{ self.count, self.rmse, self.max_abs_err, self.snr_db, self.cosine, self.rel_frob_err, self.mean_bias, self.max_rel_err_large },
        );
    }
};

/// Fraction of the reference amax above which an element counts as "large"
/// for `max_rel_err_large`.
pub const default_large_frac: f64 = 0.10;

/// Compute all metrics in a single pass-set over the data.
/// `ref` and `approx` must have equal, nonzero length.
pub fn compute(ref: []const f32, approx: []const f32) Metrics {
    return computeWithLargeFrac(ref, approx, default_large_frac);
}

pub fn computeWithLargeFrac(ref: []const f32, approx: []const f32, large_frac: f64) Metrics {
    std.debug.assert(ref.len == approx.len);
    std.debug.assert(ref.len > 0);
    const n = ref.len;

    var sum_sq_err: f64 = 0; // Σ (ref - approx)²
    var sum_sq_ref: f64 = 0; // Σ ref²
    var sum_sq_approx: f64 = 0; // Σ approx²
    var dot: f64 = 0; // Σ ref·approx
    var sum_err: f64 = 0; // Σ (approx - ref)   (signed, for bias)
    var max_abs_err: f64 = 0;
    var ref_amax: f64 = 0;

    for (ref, approx) |r64_f32, a64_f32| {
        const r: f64 = r64_f32;
        const a: f64 = a64_f32;
        const e = r - a;
        sum_sq_err += e * e;
        sum_sq_ref += r * r;
        sum_sq_approx += a * a;
        dot += r * a;
        sum_err += (a - r);
        const abs_e = @abs(e);
        if (abs_e > max_abs_err) max_abs_err = abs_e;
        const abs_r = @abs(r);
        if (abs_r > ref_amax) ref_amax = abs_r;
    }

    const nf: f64 = @floatFromInt(n);
    const mse = sum_sq_err / nf;

    const snr_db: f64 = if (sum_sq_err == 0)
        std.math.inf(f64)
    else if (sum_sq_ref == 0)
        // All-zero reference but nonzero error → undefined signal; report -inf.
        -std.math.inf(f64)
    else
        10.0 * std.math.log10(sum_sq_ref / sum_sq_err);

    const denom = @sqrt(sum_sq_ref) * @sqrt(sum_sq_approx);
    const cosine: f64 = if (denom == 0) (if (sum_sq_err == 0) 1.0 else 0.0) else dot / denom;

    const rel_frob_err: f64 = if (sum_sq_ref == 0)
        (if (sum_sq_err == 0) 0.0 else std.math.inf(f64))
    else
        @sqrt(sum_sq_err / sum_sq_ref);

    // Second pass for large-element relative error (needs ref_amax first).
    const large_threshold = ref_amax * large_frac;
    var max_rel_err_large: f64 = 0;
    if (large_threshold > 0) {
        for (ref, approx) |r64_f32, a64_f32| {
            const r: f64 = r64_f32;
            if (@abs(r) < large_threshold) continue;
            const rel = @abs(r - @as(f64, a64_f32)) / @abs(r);
            if (rel > max_rel_err_large) max_rel_err_large = rel;
        }
    }

    return .{
        .count = n,
        .mse = mse,
        .rmse = @sqrt(mse),
        .max_abs_err = max_abs_err,
        .snr_db = snr_db,
        .cosine = cosine,
        .rel_frob_err = rel_frob_err,
        .mean_bias = sum_err / nf,
        .max_rel_err_large = max_rel_err_large,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "identity round-trip is perfect" {
    const ref = [_]f32{ 1.0, -2.5, 3.25, 0.0, 100.0, -0.001 };
    const m = compute(&ref, &ref);
    try testing.expectEqual(@as(f64, 0), m.mse);
    try testing.expectEqual(@as(f64, 0), m.rmse);
    try testing.expectEqual(@as(f64, 0), m.max_abs_err);
    try testing.expect(std.math.isPositiveInf(m.snr_db));
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.cosine, 1e-12);
    try testing.expectEqual(@as(f64, 0), m.rel_frob_err);
    try testing.expectEqual(@as(f64, 0), m.mean_bias);
    try testing.expectEqual(@as(f64, 0), m.max_rel_err_large);
}

test "known constant error case" {
    // approx = ref + 1 everywhere → err=-1, mse=1, bias=+1.
    const ref = [_]f32{ 0, 0, 0, 0 };
    const approx = [_]f32{ 1, 1, 1, 1 };
    const m = compute(&ref, &approx);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.mse, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.rmse, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.max_abs_err, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.mean_bias, 1e-12);
    // Zero-signal reference with nonzero error → SNR -inf.
    try testing.expect(std.math.isNegativeInf(m.snr_db));
}

test "snr scales with error magnitude" {
    const ref = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    // Halving the error should add ~6.02 dB of SNR.
    const a_coarse = [_]f32{ 1.1, 2.1, 3.1, 4.1 };
    const a_fine = [_]f32{ 1.05, 2.05, 3.05, 4.05 };
    const coarse = compute(&ref, &a_coarse).snr_db;
    const fine = compute(&ref, &a_fine).snr_db;
    try testing.expectApproxEqAbs(@as(f64, 6.0206), fine - coarse, 1e-2);
}

test "cosine ignores uniform scaling" {
    const ref = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const scaled = [_]f32{ 2.0, 4.0, 6.0, 8.0 }; // 2× ref → same direction
    const m = compute(&ref, &scaled);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.cosine, 1e-9);
    // But magnitude error is large.
    try testing.expect(m.rel_frob_err > 0.5);
}

test "max_rel_err_large ignores tiny elements" {
    // One dominant element (100) and tiny noise elements. The tiny elements
    // get big relative errors that must NOT show up in max_rel_err_large.
    const ref = [_]f32{ 100.0, 0.001, 0.001, 0.001 };
    const approx = [_]f32{ 101.0, 0.5, 0.5, 0.5 }; // 1% error on the big one
    const m = compute(&ref, &approx);
    try testing.expectApproxEqAbs(@as(f64, 0.01), m.max_rel_err_large, 1e-6);
}
