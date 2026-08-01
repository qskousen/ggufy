//! Equalize.zig — plan §8B, activation equalization by per-channel scaling.
//!
//! §8A steers a quantizer's *scale search* with per-channel activation energy but
//! leaves the weights alone. §8B changes the weights themselves, using the
//! identity every AWQ-style method rests on:
//!
//! ```
//!     W · x  ==  (W · diag(s)) · (x / s)                for any s_j > 0
//! ```
//!
//! Scaling up the columns that matter gives them more of the quantizer's dynamic
//! range; the runtime divides the activations back down and the product is
//! unchanged. The quantization error, however, is not: a column scaled up by 4 is
//! quantized against a grid 4× finer relative to its own magnitude.
//!
//! ### The two mechanisms spend the same information, and α interpolates
//!
//! With `w_j` the per-channel energy `Imatrix` already builds (Σx², floored,
//! normalized to mean 1), this module takes
//!
//! ```
//!     s_j = w_j^(α/2)  (≙ rms(x_j)^α),  normalized to geometric mean 1
//! ```
//!
//! and the importance of the *equalized* problem is then `w'_j = w_j / s_j² =
//! w_j^(1−α)`. So α = 0 is pure §8A (no fold, all the information in the search),
//! α = 1 is pure §8B (the fold carries everything, `w'` is flat and the weighted
//! search becomes a no-op), and in between they share the work. Anything that
//! stacks the two **must** hand the search `w'`, not `w` — steering on the
//! pre-fold importance is the same class of mistake as passing unrotated weights
//! to a ConvRot format.
//!
//! ### Folding: exact on krea2's text tower, blocked on its DiT blocks
//!
//! `x / s` has to come from somewhere. Where a static per-channel producer feeds
//! the layer — an RMSNorm scale vector — dividing that vector by `s` is exact and
//! costs the runtime nothing. **That is not available for krea2's 28 main blocks**
//! [measured — TensorPencil `dit.zig` `blockForward`]. They are AdaLN-single:
//!
//! ```
//!     u = (1 + a) ⊙ (rmsNorm(x) ⊙ γ) + b        a, b = tvec_chunk + blocks.i.mod
//! ```
//!
//! `γ` absorbs `1/s` fine, and the multiplicative `(1 + a)` commutes with it, but
//! the additive `b` does not: `W·diag(s)` multiplies the shift by `s` too. `b`'s
//! `tvec` half is computed at runtime from `tproj.1`, which is **shared by all 28
//! blocks**, so no static tensor carries block *i*'s shift alone and there is
//! nothing to divide. Exactness there needs a runtime change (one per-channel
//! multiply at the modulate site, plus a vector in the file) — cheap, but not
//! free, and not this module's call to make. `ImageArch.Foldability` records which
//! layers are which so a measured gain can be labelled shippable or not.
//!
//! Level 1 is under no such constraint: it owns both sides of the GEMM, so it can
//! apply the exact transform and measure the headroom before anyone pays for the
//! runtime work. That is what this module is for first.
//!
//! ### Why `s` is clamped
//!
//! `Imatrix` floors an unobserved channel's energy at `min_relative_weight`
//! (1e-4) rather than zeroing it, because 16 prompts cannot prove a channel is
//! dead. A weight of 1e-4 only *deprioritizes* that column in a search — but a
//! fold physically shrinks its weights by `s_j`, and quantizing a column that has
//! been crushed to 1% of its true magnitude destroys it. The fold is destructive
//! where the search is merely biased, so `s` is clamped to `[1/max_ratio,
//! max_ratio]`: thin evidence can reallocate range, not delete a channel.

const std = @import("std");

/// Exponents measured by default. α = 0 is omitted deliberately: it reproduces
/// the base arm bit-for-bit, so measuring it would spend a GEMM to re-derive a
/// number already in the report.
pub const default_alphas = [_]f32{ 0.25, 0.5, 0.75 };

/// Widest per-channel scale swing allowed, either direction. See the header: this
/// bounds how much damage the fold can do to a channel the capture barely saw.
pub const default_max_ratio: f32 = 8.0;

pub const Params = struct {
    /// How much of the importance to move into the weights. 0 = no fold, 1 = full
    /// equalization (and a flat residual importance).
    alpha: f32,
    max_ratio: f32 = default_max_ratio,
};

/// Per-column fold factors from `Imatrix`'s per-column energy. Caller owns the
/// result.
///
/// `weights` is expected to be strictly positive (which `Imatrix.fromCache`
/// guarantees via its floor); a non-positive entry would make `ln` undefined, so
/// it is treated as the smallest representable positive weight rather than
/// silently producing a NaN that would propagate into the weights of a real
/// tensor.
pub fn scales(gpa: std.mem.Allocator, weights: []const f32, p: Params) ![]f32 {
    std.debug.assert(p.alpha >= 0 and p.alpha <= 1);
    std.debug.assert(p.max_ratio >= 1);

    const s = try gpa.alloc(f32, weights.len);
    errdefer gpa.free(s);
    if (p.alpha == 0 or weights.len == 0) {
        @memset(s, 1);
        return s;
    }

    // Geometric-mean normalization, done in the log domain: the mean of ln w is
    // subtracted before exponentiating, so no product of `cols` terms ever has to
    // be representable. A uniform factor in `s` is harmless to the algebra — it
    // cancels between W and x — but keeping it at 1 stops the fold from also
    // rescaling every weight in the tensor, which would interact with the
    // per-row and per-block scales for no reason.
    var mean_ln: f64 = 0;
    for (weights) |w| mean_ln += @log(@as(f64, @max(w, std.math.floatMin(f32))));
    mean_ln /= @floatFromInt(weights.len);

    const half_alpha: f64 = @as(f64, p.alpha) / 2;
    const lo: f64 = 1.0 / @as(f64, p.max_ratio);
    const hi: f64 = p.max_ratio;
    for (s, weights) |*o, w| {
        const ln_w = @log(@as(f64, @max(w, std.math.floatMin(f32))));
        o.* = @floatCast(std.math.clamp(@exp(half_alpha * (ln_w - mean_ln)), lo, hi));
    }
    return s;
}

/// The importance of the equalized problem, `w' = w / s²` — what a §8A search
/// must be given once the fold has been applied. Caller owns the result.
pub fn foldedWeights(gpa: std.mem.Allocator, weights: []const f32, s: []const f32) ![]f32 {
    std.debug.assert(weights.len == s.len);
    const out = try gpa.alloc(f32, weights.len);
    errdefer gpa.free(out);
    for (out, weights, s) |*o, w, sj| {
        const d: f64 = @as(f64, sj) * @as(f64, sj);
        o.* = @floatCast(@as(f64, w) / d);
    }
    return out;
}

/// `W · diag(s)`: column *j* of every row scaled by `s[j]`. Row-major
/// `[rows, cols]`, the layout both file formats and the harness use. Caller owns
/// the result.
pub fn foldIntoWeights(
    gpa: std.mem.Allocator,
    w: []const f32,
    rows: usize,
    cols: usize,
    s: []const f32,
) ![]f32 {
    std.debug.assert(w.len == rows * cols);
    std.debug.assert(s.len == cols);
    const out = try gpa.alloc(f32, w.len);
    errdefer gpa.free(out);
    for (0..rows) |r| {
        const src = w[r * cols ..][0..cols];
        const dst = out[r * cols ..][0..cols];
        for (dst, src, s) |*o, v, sj| o.* = v * sj;
    }
    return out;
}

/// `x / s` for a `[m, cols]` block of token rows — the counterpart the runtime
/// (or, at level 1, the harness) has to supply. Caller owns the result.
pub fn foldIntoActivations(
    gpa: std.mem.Allocator,
    x: []const f32,
    m: usize,
    cols: usize,
    s: []const f32,
) ![]f32 {
    std.debug.assert(x.len == m * cols);
    std.debug.assert(s.len == cols);
    const inv = try gpa.alloc(f32, cols);
    defer gpa.free(inv);
    for (inv, s) |*o, sj| o.* = 1.0 / sj;

    const out = try gpa.alloc(f32, x.len);
    errdefer gpa.free(out);
    for (0..m) |t| {
        const src = x[t * cols ..][0..cols];
        const dst = out[t * cols ..][0..cols];
        for (dst, src, inv) |*o, v, ij| o.* = v * ij;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const Imatrix = @import("Imatrix.zig");

test "alpha zero is the identity fold" {
    const gpa = testing.allocator;
    const w = [_]f32{ 0.01, 1, 100 };
    const s = try scales(gpa, &w, .{ .alpha = 0 });
    defer gpa.free(s);
    for (s) |v| try testing.expectEqual(@as(f32, 1), v);

    // ...and the residual importance is then the original importance, so an
    // eq-plus-search arm at alpha 0 is exactly the search arm.
    const wf = try foldedWeights(gpa, &w, s);
    defer gpa.free(wf);
    try testing.expectEqualSlices(f32, &w, wf);
}

test "scales are rms^alpha with geometric mean one" {
    const gpa = testing.allocator;
    // Energies 1/4, 1, 4: ln has mean 0 already, so the geomean normalization is
    // a no-op and the scales are exactly w^(alpha/2).
    const w = [_]f32{ 0.25, 1, 4 };
    const s = try scales(gpa, &w, .{ .alpha = 1 });
    defer gpa.free(s);
    try testing.expectApproxEqRel(@as(f32, 0.5), s[0], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 1.0), s[1], 1e-6);
    try testing.expectApproxEqRel(@as(f32, 2.0), s[2], 1e-6);

    // Half the exponent, half the swing in the log domain.
    const half = try scales(gpa, &w, .{ .alpha = 0.5 });
    defer gpa.free(half);
    try testing.expectApproxEqRel(@sqrt(@as(f32, 0.5)), half[0], 1e-6);
    try testing.expectApproxEqRel(@sqrt(@as(f32, 2.0)), half[2], 1e-6);

    // Geometric mean 1 regardless of where the energies sit: a common factor on
    // the input must not become a common factor on the weights.
    const scaled_up = [_]f32{ 2500, 10000, 40000 };
    const s2 = try scales(gpa, &scaled_up, .{ .alpha = 1 });
    defer gpa.free(s2);
    try testing.expectApproxEqRel(s[0], s2[0], 1e-5);
    try testing.expectApproxEqRel(s[2], s2[2], 1e-5);
}

test "alpha one flattens the residual importance" {
    // The property that makes alpha an interpolation between §8A and §8B rather
    // than two knobs: at alpha 1 the fold has absorbed all of the importance and
    // a weighted search has nothing left to steer with.
    const gpa = testing.allocator;
    const w = [_]f32{ 0.0625, 0.25, 1, 4, 16 };
    const s = try scales(gpa, &w, .{ .alpha = 1 });
    defer gpa.free(s);
    const wf = try foldedWeights(gpa, &w, s);
    defer gpa.free(wf);
    for (wf) |v| try testing.expectApproxEqRel(wf[0], v, 1e-5);

    // And at alpha 1/2 the residual is sqrt of the original, i.e. w^(1-alpha).
    const s_half = try scales(gpa, &w, .{ .alpha = 0.5 });
    defer gpa.free(s_half);
    const wf_half = try foldedWeights(gpa, &w, s_half);
    defer gpa.free(wf_half);
    for (wf_half, w) |got, orig| try testing.expectApproxEqRel(@sqrt(orig), got, 1e-4);
}

test "an unobserved channel is reallocated, not deleted" {
    // Imatrix floors a never-excited channel at 1e-4 of the mean. Left unclamped
    // at alpha 1 that is a 100x shrink of a real weight column on the evidence of
    // "no calibration prompt happened to excite it", and quantizing the result
    // destroys the channel. The clamp is what keeps thin evidence from being
    // load-bearing.
    const gpa = testing.allocator;
    const w = [_]f32{ Imatrix.min_relative_weight, 1, 1, 1 };
    const s = try scales(gpa, &w, .{ .alpha = 1 });
    defer gpa.free(s);
    try testing.expect(s[0] >= 1.0 / default_max_ratio);

    // Unclamped it lands below the clamp, so the clamp is doing real work here and
    // not just guarding a hypothetical.
    const loose = try scales(gpa, &w, .{ .alpha = 1, .max_ratio = 1e6 });
    defer gpa.free(loose);
    try testing.expect(loose[0] < 1.0 / default_max_ratio);
    try testing.expect(s[0] > loose[0]);
}

test "the fold is an identity on the product it factors" {
    // The whole method rests on (W·diag(s))·(x/s) == W·x. In f32 the two sides
    // differ by rounding, and this pins that difference as negligible next to any
    // quantization error it will be compared against — if it were not, the level-1
    // eq arm would be measuring its own arithmetic.
    const gpa = testing.allocator;
    const rows = 5;
    const cols = 8;
    const m = 3;

    var prng = std.Random.DefaultPrng.init(0x8B_5CA1E);
    const rnd = prng.random();
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    for (w) |*v| v.* = rnd.floatNorm(f32);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32) * 4;

    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    for (imp, 0..) |*v, j| v.* = @floatCast(std.math.pow(f64, 10, @as(f64, @floatFromInt(j)) - 4));

    const s = try scales(gpa, imp, .{ .alpha = 0.75 });
    defer gpa.free(s);
    const w_eq = try foldIntoWeights(gpa, w, rows, cols, s);
    defer gpa.free(w_eq);
    const x_eq = try foldIntoActivations(gpa, x, m, cols, s);
    defer gpa.free(x_eq);

    var num: f64 = 0;
    var den: f64 = 0;
    for (0..m) |t| {
        for (0..rows) |r| {
            var a: f64 = 0;
            var b: f64 = 0;
            for (0..cols) |c| {
                a += @as(f64, x[t * cols + c]) * @as(f64, w[r * cols + c]);
                b += @as(f64, x_eq[t * cols + c]) * @as(f64, w_eq[r * cols + c]);
            }
            num += (a - b) * (a - b);
            den += a * a;
        }
    }
    // Round-off only: ~1e-7 relative, five orders below the ~1e-2 rel-L2 that
    // int4 produces on real layers.
    try testing.expect(@sqrt(num / den) < 1e-5);
}
