//! The 1–100 sensitivity score `Convert.calculateQuantizationLevel` consumes, and
//! the **one place** its encoding is decided.
//!
//! Both measurement levels that can emit a `sensitivities/*.json` go through here
//! — `Sensitivity.writeSensitivitiesJson` (level 1, per-layer output error) and
//! `Divergence.writeSensitivitiesJson` (level 2, per-tensor whole-model damage) —
//! because the defect this module exists to prevent lived in neither the generator
//! nor the consumer but in the gap between them.
//!
//! ⚠️ **The defect, measured, so nobody re-derives it the expensive way.** Level 1
//! used to emit a **percentile rank** (`krea2.json` was a perfect 1..100 ramp,
//! median 50.5) while `calculateQuantizationLevel` reads the score as a *position on
//! the precision ladder*. Composing those two spreads the model evenly across the
//! levels **whatever the damage distribution is** — that is arithmetic, not a
//! property of the model. The resulting krea2 file routed to 6.91 bits/weight and
//! produced a 12.36 GiB model measuring **2.76× worse than the uniform
//! rate–distortion curve at its own size** — beaten on both axes by plain uniform
//! q6_k (11.76 GiB, 2.3× more accurate, 16/16 paired points). The hand-built
//! `sdxl.json` / `sd1.5.json` were brute-forced from per-layer image damage and
//! *kept magnitude* (sdxl's median score is 11.3), which is why they never hit this.
//!
//! ## The encoding
//!
//! ```
//! score = 1 + 99 · clamp( log2(d / d_median) / full_range_doublings , 0, 1 )
//! ```
//!
//! where `d_median` is the median damage of the tensors the converter can actually
//! route (see `isRoutable`). **One doubling of damage buys one bit of precision**,
//! and that is not a convention — it falls out of the allocation problem:
//!
//! Whole-model damage composes in quadrature (measured on krea2 over eight sets of
//! 65–263 tensors: `total ≈ 0.863·√Σdᵢ²`, sd 0.052), and one extra bit halves a
//! tensor's rel-L2 (measured: q6_k carries 0.26 of q4_k's error over 2 bits ≈ ½²).
//! So minimizing `Σ dᵢ²·4^(−Δbᵢ)` at fixed `Σ nᵢ·Δbᵢ` gives
//!
//! ```
//! Δbᵢ = log2(dᵢ) − ¼·log2(nᵢ) + c
//! ```
//!
//! — linear in **log damage**, which is what this encoding is. `full_range_doublings
//! = 4` maps score 100 onto 16× the median, i.e. onto the top of the ladder:
//! q4_k → q8_0 is 4 bits, and `Convert.sensitivityLadderTop` caps the ladder there
//! so that is the whole of it.
//!
//! ⚠️ **That cap is load-bearing and was added late (2026-08-02).** Before it the
//! ladder's top entry was **bf16**, 7.5 bits above q8_0 — so score 100 bought 11.5
//! bits where this derivation asks for 4, and at the default `-a 50` every tensor
//! past **13.4× the median damage** landed on 16-bit weights. The constant `4` was
//! justified in this very comment by "q4_k → q8_0", against a ladder that did not
//! end at q8_0. Two lessons kept because they are the same lesson twice: the
//! encoding is only as good as the consumer's interpretation of it (which is the
//! whole reason this module exists), and **"the ladder" has to mean one thing in
//! both modules** — hence `default_ladder_levels` here rather than a literal in
//! each report.
//!
//! ⚠️ **The score→bits map is still not linear, and this encoding cannot make it
//! so.** `calculateQuantizationLevel` interpolates over the level *index*; the
//! capped ladder's steps are 1.0 / 1.06 / 1.94 bits, so the top step is ~2× the
//! bottom one. Combined with the `-a` exponent below, a tensor 2 doublings above
//! the median gets +1 bit where the allocation wants +2. The encoding is therefore
//! *ordinally* right and *quantitatively* conservative through the middle of the
//! range. Fixing it properly means interpolating over bits/weight rather than over
//! an index, which changes what every existing arch file does and so needs its own
//! measured arm.
//!
//! ⚠️ **The `−¼·log2(nᵢ)` term is derived above and deliberately NOT applied.**
//! Damage is uncorrelated with parameter count on krea2 (rho −0.12), so it is a real
//! second effect — the largest tensors are the cheapest places to save — but the one
//! policy built on damage-per-byte predicted a 14% win and **measured a 20% loss**
//! (`ACTIVATION_AWARE_PLAN.md`, the frontier-policy arm). It goes in when an arm has
//! measured it end to end, not before.
//!
//! ### Why absolute, and not "normalize between the min and the max"
//!
//! Min–max normalization — in either linear or log space — has no notion of
//! *typical*, so its output distribution is driven by the **shape** of the input
//! rather than by how heterogeneous the model actually is. That is the same class of
//! defect as the percentile rank, just less obvious, and on real data it is worse
//! than it sounds. Replaying the converter's own formula over krea2's damage tables
//! at the default `-a 50`, mean bits/weight over the quantizable payload:
//!
//! | encoding | level 1, all 263 | level 1, 224 routable | level 2, 224 routable |
//! |---|---:|---:|---:|
//! | percentile rank (the defect) | 6.91 | 7.12 | 6.77 |
//! | linear magnitude, min–max | 4.77 | 7.33 | 5.28 |
//! | log magnitude, min–max | 6.09 | **9.48** | 6.48 |
//! | **this module (absolute)** | **4.50** | **4.50** | **4.50** |
//!
//! Row 3 is the trap: once the hiprec outliers are excluded the routable population
//! is *tight* in log space, so min–max places its median high and upgrades nearly
//! everything — spending 40% more bits than the encoding it was meant to fix.
//!
//! The absolute encoding instead reports what is true of krea2: among the 224
//! tensors routing can touch, damage spans **2.3× around the median** with no long
//! tail, so **nothing is heterogeneous enough to deserve extra bits** and the file
//! comes out as the uniform target type. That is not the encoding failing to fire —
//! uniform allocation is optimal when sensitivities are equal, and measured, uniform
//! is exactly what every scoring of krea2 ties with (0.99–1.03 of the uniform
//! rate–distortion curve) while spending 10–20% more bytes to get there. On an
//! architecture whose routable layers genuinely span decades, the same formula fires
//! hard; krea2's heterogeneity simply lives in tensors `keys_hiprec` already
//! protects.
//!
//! ⚠️ **A file that routes nothing is a result, not a bug, and it must be visible.**
//! `Sensitivity.writeMarkdown` reports how many routable layers a file it just wrote
//! would upgrade (via `upgradeThreshold`), precisely so "the feature did nothing"
//! cannot be mistaken for "the feature is broken" — or vice versa.
//!
//! ### One knob interaction worth knowing
//!
//! `calculateQuantizationLevel` raises the normalized score to `0.5 + 3·(a/100)`, so
//! at the default `-a 50` (exponent 2) the mapping from score to level is quadratic:
//! a tensor 2 doublings above the median lands 1 level up where the allocation above
//! wants 2. The default is therefore **more conservative than the rate–distortion
//! optimum**, and `-a ≈ 33` (exponent 1.5) tracks it most closely. That is what the
//! knob is for; the encoding does not try to pre-compensate for it, because it
//! cannot see either the aggressiveness or the ladder's length.

const std = @import("std");
const imagearch = @import("ImageArch.zig");

/// How many doublings of damage above the median span the full 1–100 range.
///
/// 4, because one doubling of rel-L2 is worth one bit and the ladder above a k-quant
/// target has ~4 useful bits in it (q4_k → q8_0). Raising it makes the encoding
/// stingier, lowering it more generous; it is the one free parameter here and it is
/// deliberately a single named constant rather than a flag, so that two files
/// generated by this repo are always on the same scale.
pub const full_range_doublings: f64 = 4;

/// How many levels the sensitivity ladder offers in the case every generated file
/// is implicitly calibrated against: a k-family target on a >f16 source, i.e.
/// **q4_k → q5_k → q6_k → q8_0**.
///
/// Four, not five. `Convert.sensitivityLadderTop` caps the sensitivity mechanism at
/// the highest *quantized* level, so bf16 is no longer the ladder's top entry — see
/// that function for why, and for the 7.5-bit step that made it necessary. The
/// number lives here because both the converter and every report that predicts what
/// a file will do have to agree on it, and they are in different modules; a literal
/// `5` in either place is how the encoding and its interpretation drifted apart the
/// first time. `Convert.zig` has the test pinning this to the real candidate list.
pub const default_ladder_levels: usize = 4;

/// The score every tensor gets when the population carries no usable scale.
///
/// **1, not 50.** 1 means "the type the user asked for"; 50 would upgrade half the
/// ladder on no evidence at all. Under-spending on ambiguity is the conservative
/// direction, and over-spending is the exact failure this module was written for.
pub const homogeneous_score: f64 = 1;

/// The population's median damage, in log2 — the anchor the encoding measures
/// doublings from. Built from one population and then applied to any damage
/// (including tensors outside it, whose scores clamp rather than move the scale).
pub const Ladder = struct {
    log2_med: f64,

    /// Where `damage` sits on the 1–100 ladder.
    pub fn score(self: Ladder, damage: f64) f64 {
        // A non-positive damage has no logarithm. Zero really does mean "this format
        // encodes this tensor exactly", which is the bottom of the ladder; an
        // *unmeasured* tensor is a different thing entirely and must be omitted from
        // the file by the caller, never scored.
        if (!(damage > 0)) return homogeneous_score;
        const doublings = @log2(damage) - self.log2_med;
        const t = std.math.clamp(doublings / full_range_doublings, 0, 1);
        return 1 + 99 * t;
    }
};

/// Build the ladder from the damages of the tensors the converter can route.
///
/// Returns null only when none of them is positive — there is then no anchor, and
/// the caller emits `homogeneous_score`. A single measured tensor is *not* a
/// degenerate case here: it is its own median, so it scores 1 and keeps the target
/// type, which is the correct answer when one layer is all that was measured.
/// `damages` is not modified.
pub fn fromDamages(gpa: std.mem.Allocator, damages: []const f64) !?Ladder {
    var pos: std.ArrayList(f64) = .empty;
    defer pos.deinit(gpa);
    for (damages) |d| if (d > 0) try pos.append(gpa, d);
    if (pos.items.len == 0) return null;

    std.mem.sort(f64, pos.items, {}, std.sort.asc(f64));
    const n = pos.items.len;
    const med = if (n % 2 == 1) pos.items[n / 2] else (pos.items[n / 2 - 1] + pos.items[n / 2]) / 2;
    return .{ .log2_med = @log2(med) };
}

/// The smallest score that moves a tensor **off** the target type, under
/// `Convert.calculateQuantizationLevel`'s own rule:
///
/// ```
/// level_index = round( norm^(0.5 + 3·aggressiveness/100) · (levels − 1) )
/// ```
///
/// so an upgrade needs that product to reach 0.5. Null when the ladder offers fewer
/// than two levels, i.e. when nothing can be upgraded at all.
///
/// Note the direction that surprises: a **shorter** ladder has a **higher**
/// threshold, because each of its levels covers more of the score range.
///
/// This exists so a report can say how many layers a file it just wrote would
/// actually upgrade, *before* anyone spends an hour converting and measuring it. The
/// generator's encoding and the converter's interpretation were designed separately
/// once; this is the seam where that is checked, and `Convert.zig` has the test that
/// pins it to the real rule.
pub fn upgradeThreshold(aggressiveness: f64, levels: usize) ?f64 {
    if (levels < 2) return null;
    const hard = std.math.clamp(aggressiveness, 1, 100);
    const exponent = 0.5 + (hard / 100) * 3;
    const norm = std.math.pow(f64, 0.5 / @as(f64, @floatFromInt(levels - 1)), 1 / exponent);
    return 1 + 99 * norm;
}

/// The reserved key every generated sensitivities file carries its provenance under.
///
/// It is not a tensor name, so `Convert`'s `sensitivities.value.object.get(t.name)`
/// can never collide with it — the lookup is by checkpoint tensor name and no
/// checkpoint has a tensor called `__meta__`.
pub const meta_key = "__meta__";

/// Identifies *this* encoding, so a consumer can tell a file generated now from one
/// generated by the percentile-rank emitter that produced a model 2.76× off the
/// rate–distortion curve.
///
/// ⚠️ **The absence of this record is the whole reason it exists.** Every score file
/// this repo has ever emitted was a bare `{"tensor.name": score}` map: no encoding,
/// no generator, no reference format, no layer count. The programme's most expensive
/// defect was the generator's encoding and the converter's interpretation drifting
/// apart — and the artifact that crosses that seam carried nothing that could
/// identify which encoding produced it. A percentile-era file and a LadderScore-era
/// file are byte-indistinguishable apart from their score *distribution*, which is
/// exactly what nobody looks at before converting.
pub const encoding_id = "ladderscore-v1-absolute-median";

/// What a generated sensitivities file records about how it was made.
pub const Provenance = struct {
    /// Which measurement level produced the damages: "level1-output-error" or
    /// "level2-per-tensor-damage".
    generator: []const u8,
    /// Detected architecture, or "" when unknown — it decides `isRoutable`, and so
    /// the anchor.
    arch: []const u8,
    /// The format whose damage the scores are of (e.g. "Q4_K"). A score file is
    /// implicitly about one format; saying which is the difference between a
    /// reproducible artifact and a number.
    reference_format: []const u8,
    /// How many tensors carry a score in this file.
    scored: usize,
    /// How many of those set the median anchor (the routable population).
    anchor_population: usize,
    /// False when the run measured only part of the model. **Load-bearing**: the
    /// anchor is the population's median, so a truncated sweep does not produce a
    /// partial file, it produces a *miscalibrated* one — every score shifts with the
    /// subset's median. Under the old percentile encoding a partial run merely gave a
    /// sparse ranking; under this one it silently rescales the whole ladder.
    complete: bool,

    pub fn write(self: Provenance, w: *std.Io.Writer) !void {
        try w.print(
            \\  "{s}": {{
            \\    "encoding": "{s}",
            \\    "full_range_doublings": {d},
            \\    "generator": "{s}",
            \\    "arch": "{s}",
            \\    "reference_format": "{s}",
            \\    "scored": {d},
            \\    "anchor_population": {d},
            \\    "complete": {s}
            \\  }}
        , .{
            meta_key,          encoding_id,          full_range_doublings,
            self.generator,    self.arch,            self.reference_format,
            self.scored,       self.anchor_population,
            if (self.complete) "true" else "false",
        });
    }
};

/// Read the provenance record of a sensitivities file the converter was handed, and
/// say what is worth warning about. Returns null when there is nothing to say.
///
/// Deliberately advisory rather than fatal: the hand-built `sdxl.json` / `sd1.5.json`
/// have no `__meta__` at all and are legitimate (they were brute-forced from per-layer
/// image damage, which is the *right* measurement), so refusing an unlabelled file
/// would reject the best data in the repo. What must not happen is a file silently
/// asserting an encoding the converter does not implement.
pub fn metaWarning(root: std.json.Value) ?[]const u8 {
    const obj = switch (root) {
        .object => |o| o,
        else => return null,
    };
    const meta = obj.get(meta_key) orelse return "no __meta__ record: cannot tell which encoding produced it (hand-built files are like this and are fine; a file generated by `sensitivity`/`divergence` before 2026-08-02 may be a percentile rank, which measured 2.76x off the rate-distortion curve)";
    const mo = switch (meta) {
        .object => |o| o,
        else => return "__meta__ is not an object",
    };
    if (mo.get("complete")) |c| {
        if (c == .bool and !c.bool) return "__meta__.complete is false: the sweep that wrote this file was truncated, so its median anchor is a subset's and every score is rescaled. Regenerate it over the whole model.";
    }
    const enc = mo.get("encoding") orelse return "__meta__ has no encoding field";
    if (enc != .string or !std.mem.eql(u8, enc.string, encoding_id)) {
        return "__meta__.encoding is not the encoding this build interprets; the scores may mean something else entirely";
    }
    return null;
}

/// Whether a tensor is one the sensitivity score can affect at all.
///
/// ⚠️ **Normalize over the population the consumer can act on.** Measured, and it
/// cost an arm to find: the first regenerated krea2 file normalized over all 263
/// measured tensors, and because the seven most damaging are all `keys_hiprec`
/// (never routed — `assignTensorType` applies its structural protections *before*
/// the sensitivity branch) that compressed the 224 routable ones into scores 1..23,
/// all below the upgrade threshold. The "routed" model came out as uniform q4_k plus
/// a single q5_k tensor.
///
/// A median anchor is far more robust to that than min–max was, but excluding the
/// tensors whose score is ignored anyway is free and exactly right.
///
/// Not covered here, deliberately: `Convert`'s element-count threshold and its 1-D /
/// embedding rules. Both measurement levels only ever score 2-D matmul weights, and
/// the threshold needs conversion options this side does not have.
pub fn isRoutable(arch: ?*const imagearch.Arch, name: []const u8) bool {
    const a = arch orelse return true; // unknown architecture: no grounds to exclude anything
    const key = imagearch.stripPrefix(name);
    return !a.isHighPrecision(key) and !a.shouldUpcast(key);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "a homogeneous population routes uniformly, at the size the user asked for" {
    const gpa = testing.allocator;
    // krea2's 224 routable tensors in miniature: a tight spread, no tail. Measured,
    // they span 2.3x around the median, and uniform allocation is optimal when
    // sensitivities are equal — so nothing here should climb the ladder.
    const d = [_]f64{ 1.0e-3, 1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3 };
    const lad = (try fromDamages(gpa, &d)).?;
    const thr = upgradeThreshold(50, 5).?;

    for (d) |x| try testing.expect(lad.score(x) < thr);
    try testing.expectApproxEqAbs(@as(f64, 1), lad.score(1.2e-3), 1e-12); // the median
    try testing.expectApproxEqAbs(@as(f64, 1), lad.score(1.0e-3), 1e-12); // below it

    // This is the property min–max normalization cannot have: it would have put the
    // 1.4e-3 layer at score 100 and bought it 4 extra bits for being 17% worse than
    // typical. Measured on krea2's real table, that costs 40% more bits than the
    // percentile encoding it was meant to fix.
    try testing.expect(lad.score(1.4e-3) < 20);
}

test "one doubling of damage is one bit, which is the allocation the RD optimum wants" {
    const gpa = testing.allocator;
    const d = [_]f64{ 1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3, 1.6e-2 };
    const lad = (try fromDamages(gpa, &d)).?;
    // Median is 4.0e-3. Each doubling above it is one quarter of the range.
    try testing.expectApproxEqAbs(@as(f64, 1), lad.score(4.0e-3), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1 + 99 * 0.25), lad.score(8.0e-3), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1 + 99 * 0.50), lad.score(1.6e-2), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 100), lad.score(6.4e-2), 1e-9); // 4 doublings
    try testing.expectApproxEqAbs(@as(f64, 100), lad.score(1.0), 1e-9); // and it saturates
}

test "a genuine outlier climbs; the layers below it do not move" {
    const gpa = testing.allocator;
    // One layer 8x the median, the rest clustered — the shape the hand-built arch
    // files have, and the shape percentile ranking destroyed.
    const d = [_]f64{ 2.4e-2, 3.0e-3, 2.9e-3, 3.1e-3, 3.0e-3, 2.8e-3, 3.2e-3 };
    const lad = (try fromDamages(gpa, &d)).?;
    const thr = upgradeThreshold(50, 5).?;

    var upgraded: usize = 0;
    for (d) |x| if (lad.score(x) >= thr) {
        upgraded += 1;
    };
    try testing.expectEqual(@as(usize, 1), upgraded);

    // And crucially, the outlier's presence does not change anyone else's score:
    // the anchor is the median, not the max, so a tail cannot drag the body of the
    // distribution up OR down the ladder.
    const without = [_]f64{ 3.0e-3, 2.9e-3, 3.1e-3, 3.0e-3, 2.8e-3, 3.2e-3 };
    const lad2 = (try fromDamages(gpa, &without)).?;
    for (without) |x| try testing.expectApproxEqAbs(lad.score(x), lad2.score(x), 1e-9);
}

test "scores are invariant to the scale of the error metric" {
    const gpa = testing.allocator;
    const d = [_]f64{ 1.0e-3, 4.0e-3, 2.0e-3, 9.0e-3, 1.5e-3 };
    var scaled: [d.len]f64 = undefined;
    for (d, 0..) |x, i| scaled[i] = x * 137.0;

    const a = (try fromDamages(gpa, &d)).?;
    const b = (try fromDamages(gpa, &scaled)).?;
    for (d, scaled) |x, y| try testing.expectApproxEqAbs(a.score(x), b.score(y), 1e-9);
}

test "no positive damage means no anchor, and the no-anchor score is the target type" {
    const gpa = testing.allocator;
    const none = [_]f64{ 0, 0, -1 };
    try testing.expect(try fromDamages(gpa, &none) == null);
    // 1 = "use the type the user asked for". 50 would upgrade half the ladder on no
    // evidence, which is the direction that produced a 2.76x-off-the-curve model.
    try testing.expectEqual(@as(f64, 1), homogeneous_score);

    // A single measured tensor is its own median: it keeps the target type rather
    // than being declared extreme in either direction.
    const one = [_]f64{5.0e-3};
    const lad = (try fromDamages(gpa, &one)).?;
    try testing.expectApproxEqAbs(@as(f64, 1), lad.score(5.0e-3), 1e-12);
}

test "zero damage is the bottom of the ladder; unmeasured is the caller's problem" {
    const gpa = testing.allocator;
    const d = [_]f64{ 0, 1.0e-3, 4.0e-3 };
    const lad = (try fromDamages(gpa, &d)).?;
    // A zero does not participate in the anchor (it has no log) and scores 1.
    try testing.expectApproxEqAbs(@as(f64, 1), lad.score(0), 1e-12);
    // ...and it must not have moved the median: the population is {1e-3, 4e-3}, so
    // the anchor is their mean, 2.5e-3.
    try testing.expectApproxEqAbs(@log2(2.5e-3), lad.log2_med, 1e-12);
}

test "the upgrade threshold moves the right way with aggressiveness and ladder length" {
    // Higher aggressiveness = stay nearer the target = a higher bar to upgrade.
    const lo = upgradeThreshold(1, 5).?;
    const mid = upgradeThreshold(50, 5).?;
    const hi = upgradeThreshold(100, 5).?;
    try testing.expect(lo < mid);
    try testing.expect(mid < hi);
    // A SHORTER ladder has a HIGHER threshold — each of its levels covers more of
    // the score range. Easy to get backwards, so it is pinned.
    try testing.expect(upgradeThreshold(50, 3).? > mid);
    // A one-level ladder can upgrade nothing.
    try testing.expect(upgradeThreshold(50, 1) == null);
    // The documented number for the default: q4_k→bf16 is 5 levels at -a 50.
    try testing.expectApproxEqAbs(@as(f64, 36.0), mid, 0.5);
}
