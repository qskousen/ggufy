//! Level 1 — per-layer output error with real captured activations.
//!
//! The workhorse of ACTIVATION_AWARE_PLAN.md §7. Level 0 (`zig build precision`)
//! measures how far a quantized weight is from the original; that says the
//! dequantizer is correct and nothing about which layers matter. Level 1 asks
//! the question the network actually poses:
//!
//!     Y   = X · Wᵀ        the layer's real output on captured activations
//!     Ŷ_f = X · Ŵᵀ        FORMAT arm — dequantize, then the f32 reference GEMM
//!     Ŷ_k = X ⊛ q(W)      KERNEL arm — the native quantized GEMM, where one exists
//!     Ŷ_w = X · Ŵ_wᵀ      WEIGHTED arm — as the format arm, but with the layer's
//!                         own activation energy steering whichever scale search
//!                         the format has: ggml's `quant_weights` for the
//!                         block-quant types, the clipping search for scaled-fp8
//!                         and the int8/int4 clusters (plan §8A)
//!     Ŷ_e = (X/s) · Ŵ_eᵀ  EQUALIZED arm — the shipped arm again, but on the
//!                         equalized problem: quantize `W·diag(s)` and feed it
//!                         `X/s`, which is the same function before quantization
//!                         and a different one after (plan §8B)
//!     Ŷ_g = X_ho · Ŵ_gᵀ   GPTQ arm — the shipped arm's grid, with the rounding
//!                         chosen by error compensation against the *full* Gram
//!                         `XᵗX` rather than its diagonal (plan §8C)
//!
//! **The equalized arm must be evaluated on the folded activations**, which is why
//! it is the harness's business and not a post-processing step on the weight
//! error: `Ŷ_e` compared against `X · Ŵ_eᵀ` would be measuring a layer that has
//! been handed inputs it will never see, and would report a fake win exactly in
//! proportion to how aggressive the fold was.
//!
//! **The GPTQ arm must be evaluated on rows it did not fit.** Every other arm here
//! derives its quantization from the *weights*, using the activations only to score
//! it; §8C derives it from the activations themselves, so scoring it on the same
//! rows measures memorization. It is the one arm that splits the row sample, and it
//! reports both halves — the held-out pair is the finding, and the gap to the
//! in-sample pair is how much of the apparent win was the sample rather than the
//! layer.
//!
//! and reports how far each Ŷ is from Y. `X` comes from a calibration cache
//! (`ggufy calibrate`), so this is cheap: one small GEMM per (layer, format,
//! arm) once the capture is paid for.
//!
//! **Format loss and kernel loss are two numbers, never one** (hygiene rule #3).
//! The format arm isolates the quantization; the kernel arm isolates the
//! implementation. A format whose two arms disagree has a kernel bug, and
//! collapsing them would hide it. The kernel arm is only reported for formats
//! TensorPencil can actually execute — a format nothing can run has no kernel
//! number, and inventing one would be a guess about a GEMM that does not exist.
//!
//! The primary output is `src/sensitivities/<arch>.json`, emitted under exact
//! checkpoint tensor names so the existing converter routing picks it up with no
//! changes at all.

const std = @import("std");
const tp = @import("TensorPencil");
const ph = @import("precision_harness.zig");
const metrics = @import("PrecisionMetrics.zig");
const types = @import("types.zig");
const DataTransform = @import("DataTransform.zig");
const CalibrationCache = @import("CalibrationCache.zig");
const ThreadPool = @import("ThreadPool.zig").ThreadPool;
const cb = @import("callbacks.zig");
const imagearch = @import("ImageArch.zig");
const gguf = @import("Gguf.zig");
const Imatrix = @import("Imatrix.zig");
const Equalize = @import("Equalize.zig");
const Gptq = @import("Gptq.zig");
const TensorClusters = @import("TensorClusters.zig");
const LadderScore = @import("LadderScore.zig");

pub const Format = ph.Format;

/// The formats measured when a caller does not name any. Everything ggufy can
/// emit: the ranking is only useful if it covers the choices the converter can
/// actually make.
pub const default_formats = blk: {
    var out: [ph.formats.len]Format = undefined;
    for (ph.formats, 0..) |f, i| out[i] = f.fmt;
    break :blk out;
};

/// The format the sensitivity score is derived from. Q4_K is the interesting
/// operating point — aggressive enough that layers separate, common enough that
/// the ranking transfers to what people actually convert to.
pub const default_reference_format: Format = .q4_k;

/// Default ConvRot rotation group for the rotating arms — the harness's small one,
/// which every measurement before 2026-07-31 used. `TensorClusters.int4_convrot_group_size`
/// is what `convert` ships.
pub const default_convrot_group: usize = ph.convrot_group_size;

// ---------------------------------------------------------------------------
// ggufy DataType → TensorPencil DType
// ---------------------------------------------------------------------------

/// The single conversion between ggufy's file-format type vocabulary and
/// TensorPencil's "a format we can compute with" vocabulary (§2.2). Null means
/// TensorPencil has no kernel for it, which is exactly the condition under which
/// the kernel arm is skipped rather than faked.
pub fn toTp(dt: types.DataType) ?tp.DType {
    return switch (dt) {
        .F32, .f32 => .f32,
        .F16, .f16 => .f16,
        .BF16 => .bf16,
        .F8_E4M3 => .f8_e4m3,
        .q4_0 => .q4_0,
        .q8_0 => .q8_0,
        .q4_k => .q4_k,
        .q5_k => .q5_k,
        .q6_k => .q6_k,
        else => null,
    };
}

/// At or above this many tokens TensorPencil's CPU path packs and runs an f32
/// microkernel, so a block-quant GEMM has quantized weights and f32 activations.
/// Below it, block-quant weights take ggml's int8 vec_dot GEMV, which
/// **quantizes the activations too**.
///
/// That distinction is hygiene rule #4. The harness does not dodge it by
/// restricting the token count — it sets `ops.matmul.exact_activations` for the
/// kernel arm, which pins the f32-activation path at every m. This constant is
/// kept only to test that the pinning still means what we think it means.
///
/// Read from TensorPencil rather than copied: a duplicated 16 here would drift
/// silently the day the dispatch changed.
const tp_small_m_max: usize = tp.ops.matmul.small_m_max;

/// The ggufy datatype a format quantizes to, when it has a plain byte-array
/// representation that a kernel could consume. The cluster formats (NVFP4,
/// INT4/INT8-convrot, …) carry sidecar scale tensors and are excluded here; they
/// still get a format arm.
fn kernelDataType(fmt: Format) ?types.DataType {
    return switch (fmt) {
        .f16 => .f16,
        .bf16 => .BF16,
        .f8_e4m3 => .F8_E4M3,
        .q8_0 => .q8_0,
        .q6_k => .q6_k,
        .q5_k => .q5_k,
        .q4_k => .q4_k,
        .q4_0 => .q4_0,
        else => null,
    };
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// How far one arm's output is from the reference output. Three numbers because
/// they fail differently: `rel_l2` is the aggregate the ranking uses,
/// `mean_token_cos` catches a direction change that a magnitude error would
/// hide, and `max_token_rel` catches a single wrecked token that an average
/// would bury.
pub const ArmMetrics = struct {
    /// ‖Ŷ − Y‖ / ‖Y‖ over the whole output block.
    rel_l2: f64,
    /// Mean over tokens of cosine(Ŷ_token, Y_token).
    mean_token_cos: f64,
    /// Worst single token's ‖Ŷ_t − Y_t‖ / ‖Y_t‖.
    max_token_rel: f64,
};

fn armMetrics(y_ref: []const f32, y_apx: []const f32, m: usize, n: usize) ArmMetrics {
    const overall = metrics.compute(y_ref, y_apx);

    var cos_sum: f64 = 0;
    var counted: usize = 0;
    var max_rel: f64 = 0;
    for (0..m) |t| {
        const a = y_ref[t * n ..][0..n];
        const b = y_apx[t * n ..][0..n];
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        var derr: f64 = 0;
        for (a, b) |va, vb| {
            const fa: f64 = va;
            const fb: f64 = vb;
            dot += fa * fb;
            na += fa * fa;
            nb += fb * fb;
            const d = fa - fb;
            derr += d * d;
        }
        // A token whose reference output is exactly zero has no direction and no
        // scale, so it contributes to neither statistic rather than producing a
        // 0/0 that would silently read as "perfect".
        if (na > 0) {
            max_rel = @max(max_rel, @sqrt(derr / na));
            if (nb > 0) {
                cos_sum += dot / (@sqrt(na) * @sqrt(nb));
                counted += 1;
            }
        }
    }

    return .{
        .rel_l2 = overall.rel_frob_err,
        .mean_token_cos = if (counted > 0) cos_sum / @as(f64, @floatFromInt(counted)) else 1.0,
        .max_token_rel = max_rel,
    };
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// One §8B arm: the configuration `convert` ships for this format, re-run on the
/// equalized problem `(W·diag(s), X/s)` at this α.
pub const EqArm = struct {
    alpha: f32,
    /// Whether the base this stacked on was the weighted (§8A) arm or the plain
    /// format arm. The decision-relevant question is always "does equalization add
    /// anything on top of what we already ship", so the base follows
    /// `Imatrix.shipsWeighted` rather than being fixed.
    on_weighted: bool,
    metrics: ArmMetrics,
};

/// One §8C arm: the format's shipped configuration, re-rounded with GPTQ error
/// compensation against the full Gram of the *training* rows.
///
/// Four metrics, not one, because the arm's credibility is the point:
/// `held_out` vs `held_out_base` is the finding, and `in_sample` vs
/// `in_sample_base` is the same comparison on the rows the Hessian was fitted to.
/// A held-out ratio near 1.0 while the in-sample ratio is far below it means the
/// compensation learned this row sample and not the layer — the exact failure the
/// learned-rounding post-mortem was about, made visible instead of assumed away.
pub const GptqArm = struct {
    /// Token rows the Hessian was built from.
    train_tokens: usize,
    /// Token rows every `held_out` number is measured on. Disjoint from the above.
    eval_tokens: usize,
    /// Ridge as a fraction of the mean diagonal of the Gram.
    damp: f32,
    /// Compensation granularity: 0 when GPTQ rounds column by column, else the
    /// encoder's block width. This is the ceiling on what the arm can find — a
    /// 256-wide block over 6144 columns gets 24 compensation steps where the
    /// per-column path gets 6144, and no amount of calibration data changes that.
    block: usize = 0,
    /// Whether the grid both arms share came from the §8A weighted search. Follows
    /// `Imatrix.shipsWeighted`, since the question is always "on top of what we
    /// ship".
    on_weighted: bool,
    /// The shipped configuration, on the held-out rows.
    held_out_base: ArmMetrics,
    /// GPTQ, on the held-out rows. **This is the decision number.**
    held_out: ArmMetrics,
    /// The shipped configuration, on the training rows.
    in_sample_base: ArmMetrics,
    /// GPTQ, on the training rows.
    in_sample: ArmMetrics,
};

pub const FormatResult = struct {
    fmt: Format,
    /// Nominal bits/weight, for the rate-distortion view.
    bits: f32,
    format_arm: ArmMetrics,
    /// Null when TensorPencil has no kernel for this format — see `toTp`.
    kernel_arm: ?ArmMetrics,
    /// The format arm again, but quantized with this layer's activation energy as
    /// ggml's `imatrix` (plan §8A). Null when the encoder ignores an imatrix, when
    /// the cache has no usable statistics for the layer, or when the arm is off.
    ///
    /// This is the measurement that decides whether activation-aware quantization
    /// is worth turning on: it is the same quantity as `format_arm` — real output
    /// error on real activations — so the two are directly comparable, unlike the
    /// weighted *weight* error the encoder optimizes internally.
    weighted_arm: ?ArmMetrics = null,
    /// §8B equalization arms, one per requested α, in the order requested. Empty
    /// when the arm is off or the cache has nothing usable for the layer.
    ///
    /// Unlike `weighted_arm` this is available for *every* format: equalization
    /// changes the weights rather than steering a search, so it also reaches the
    /// formats §8A cannot (q8_0, mxfp4, nvfp4, mxfp8).
    eq_arms: []const EqArm = &.{},
    /// §8C GPTQ arm. Null when the arm is off, when the format has no grid GPTQ can
    /// own (everything but the int8/int4 clusters — see `Gptq`'s header), or when
    /// the row sample is too small to split.
    gptq_arm: ?GptqArm = null,

    /// The arm an `EqArm` is to be compared against — the shipped configuration it
    /// stacked on.
    pub fn baseArm(self: FormatResult, on_weighted: bool) ArmMetrics {
        if (on_weighted) if (self.weighted_arm) |wm| return wm;
        return self.format_arm;
    }
};

pub const LayerResult = struct {
    name: []const u8,
    rows: usize,
    cols: usize,
    /// Token rows of X the measurement used.
    tokens: usize,
    per_format: []FormatResult,
    /// Percentile-ranked 1–100 sensitivity at the reference format, filled by
    /// `scoreLayers` once every layer is measured.
    ///
    /// Null when the reference format could not be produced for this layer at
    /// all — typically a shape that violates the format's block constraint
    /// (`first.weight` is 6144x64, and Q4_K needs a multiple of 256). Such a
    /// layer is **unmeasured, not insensitive**, and conflating the two would
    /// route it to the most aggressive quantization available. It is left out of
    /// the emitted JSON, where the converter's own fallback handles it.
    score: ?f64 = null,

    /// This layer's metrics at `fmt`, or null if it was not measured.
    pub fn find(self: LayerResult, fmt: Format) ?FormatResult {
        for (self.per_format) |f| if (f.fmt == fmt) return f;
        return null;
    }
};

pub const Report = struct {
    arena: std.heap.ArenaAllocator,
    layers: []LayerResult,
    reference_format: Format,
    /// Where the numbers came from, so a report is self-describing.
    model_path: []const u8,
    calib_path: []const u8,
    arch: []const u8,
    prompt_set: []const u8,
    /// True when the source checkpoint is not f32/f16 — i.e. the "reference" is
    /// itself already quantized, and every absolute number is relative to it.
    /// Plan open question 4; never let this go unlabelled.
    reference_is_quantized: bool,
    reference_dtype: []const u8,
    /// How many (layer, format) pairs got a kernel arm.
    kernel_arms: usize,
    /// How many (layer, format) pairs got an activation-weighted arm.
    weighted_arms: usize,
    /// How many (layer, format, α) triples got a §8B equalization arm.
    eq_arms: usize = 0,
    /// The α values measured, in report order. Empty when the arm was off.
    eq_alphas: []const f32 = &.{},
    /// How many (layer, format) pairs got a §8C GPTQ arm.
    gptq_arms: usize = 0,
    /// The ridge and the held-out stride the GPTQ arm ran with, so its numbers are
    /// reproducible from the report alone. `gptq_holdout` is 0 when the evaluation
    /// came from a second cache instead of a row split.
    gptq_damp: f32 = 0,
    gptq_holdout: usize = 0,
    /// The second cache the GPTQ arm scored against, and its prompt set. Empty when
    /// the arm used a row split. This is the difference between "held-out rows" and
    /// "held-out prompts", and the report must never blur the two.
    gptq_eval_calib: []const u8 = "",
    gptq_eval_prompt_set: []const u8 = "",
    /// The ConvRot group every rotating arm used.
    convrot_group: usize = default_convrot_group,
    /// Per-layer foldability of the architecture's graph (§8B). Counted so the
    /// report can say how much of a measured gain is actually shippable without a
    /// runtime change.
    fold_exact: usize = 0,
    fold_runtime_shift: usize = 0,
    fold_none: usize = 0,
    /// False when no architecture in `ImageArch` matched the cache's `arch` string,
    /// in which case every layer counts as `.none` for lack of a graph to read —
    /// absence of a relation, not absence of foldable layers.
    fold_known: bool = false,
    /// Layers that got a score — i.e. that had a reference-format measurement.
    scored: usize,
    /// How many scored layers set the median anchor (the routable population).
    anchor_population: usize = 0,
    /// False when `--max-layers` cut the sweep short.
    ///
    /// ⚠️ **This is not cosmetic.** The anchor is the routable population's *median*,
    /// so a truncated sweep is not a partial file — it is a **miscalibrated** one,
    /// with every score shifted by the subset's median. `cache.layers()` is in cache
    /// order, so a truncated run also takes a structurally biased slice (the early
    /// blocks), not a random one. The JSON records this and the converter warns on it.
    complete: bool = true,

    pub fn deinit(self: *Report) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Turn measured error into the 1–100 scale `Convert.zig` consumes, via the one
/// module that owns that encoding (`LadderScore`) — see its header for why it is one
/// bit per doubling of damage above the routable median, rather than the percentile
/// rank this function used to emit, and what the percentile version measured.
///
/// `arch` supplies the routable/hiprec split, and it matters: the tensors the
/// converter protects structurally are also the most damaging ones, so letting them
/// set the scale compresses everything the score can actually steer. Null (an
/// architecture `ImageArch` does not know) normalizes over every measured layer,
/// which is the only option available and is recorded in the report.
const Scored = struct { scored: usize, anchor_population: usize };

fn scoreLayers(
    gpa: std.mem.Allocator,
    layers: []LayerResult,
    reference: Format,
    arch: ?*const imagearch.Arch,
) !Scored {
    // Only layers that actually have a reference-format measurement take part.
    var measured: std.ArrayList(usize) = .empty;
    defer measured.deinit(gpa);
    for (layers, 0..) |*l, i| {
        l.score = null;
        if (l.find(reference) != null) try measured.append(gpa, i);
    }
    if (measured.items.len == 0) return .{ .scored = 0, .anchor_population = 0 };

    const err = struct {
        fn f(ls: []const LayerResult, ref: Format, i: usize) f64 {
            return ls[i].find(ref).?.format_arm.rel_l2;
        }
    }.f;

    // The anchor comes from the routable population only.
    var pop: std.ArrayList(f64) = .empty;
    defer pop.deinit(gpa);
    for (measured.items) |i| {
        if (LadderScore.isRoutable(arch, layers[i].name)) try pop.append(gpa, err(layers, reference, i));
    }
    // Nothing routable measured — a sweep that covered only structurally-protected
    // tensors. Normalize over everything rather than emit a file of flat scores.
    if (pop.items.len == 0) {
        for (measured.items) |i| try pop.append(gpa, err(layers, reference, i));
    }

    const ladder = try LadderScore.fromDamages(gpa, pop.items);
    for (measured.items) |i| {
        // Ties get the same score by construction: the score is a function of the
        // error alone, so routing cannot depend on sort order.
        layers[i].score = if (ladder) |l| l.score(err(layers, reference, i)) else LadderScore.homogeneous_score;
    }
    return .{ .scored = measured.items.len, .anchor_population = pop.items.len };
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub const Options = struct {
    /// Checkpoint the cache was captured from.
    model_path: []const u8,
    /// Calibration cache written by `ggufy calibrate`.
    calib_path: []const u8,
    formats: []const Format = &default_formats,
    reference_format: Format = default_reference_format,
    /// Measure only this schedule bucket. Null concatenates every bucket's row
    /// sample, which is what a whole-model ranking wants; a single bucket
    /// answers "which layers matter early vs late".
    bucket: ?usize = null,
    /// Skip the native-kernel arm even where one exists (it roughly doubles the
    /// GEMM cost and is not needed for the ranking).
    kernel_arm: bool = true,
    /// Measure the activation-weighted (`imatrix`) arm alongside the plain format
    /// arm, for every ggml encoder that reads one. On by default: it is the only
    /// evidence that says whether `convert --calib` helps this model, and the
    /// cache it needs is already open.
    weighted_arm: bool = true,
    /// §8B: measure an equalization arm at each of these α exponents, stacked on
    /// whatever `convert` ships for the format. Empty (the default) is off —
    /// unlike the weighted arm this is not a free ride on an open cache, it is a
    /// full extra quantization and GEMM per (layer, format, α), and nothing ships
    /// it yet.
    eq_alphas: []const f32 = &.{},
    /// Widest per-channel fold factor the equalization arm may use, either
    /// direction. See `Equalize.default_max_ratio` for why a bound exists at all.
    eq_max_ratio: f32 = Equalize.default_max_ratio,
    /// ConvRot rotation group for every arm that rotates. Defaults to the harness's
    /// small group; pass `TensorClusters.int4_convrot_group_size` to measure the
    /// configuration `convert` actually writes.
    ///
    /// It matters far more to §8C than to §8A. A diagonal importance vector only
    /// gets *averaged* differently by a different group, but GPTQ computes its whole
    /// Hessian in the rotated basis, so the group size selects which cross terms
    /// exist at all — measuring at 64 and shipping at 256 measures a different
    /// algorithm, not the same one at a different resolution.
    convrot_group: usize = default_convrot_group,
    /// §8C: measure a GPTQ arm for the int8/int4 formats. Off by default — it is
    /// the most expensive arm here by a wide margin (an `m×m` sweep over every
    /// column, plus a full extra quantization and four GEMMs per layer and format)
    /// and nothing ships it yet.
    gptq: bool = false,
    gptq_damp: f32 = Gptq.default_damp,
    /// One row in `gptq_holdout` is held back from the Hessian and every held-out
    /// number is measured on those rows alone. 3 keeps two thirds of a thin sample
    /// for the fit while still leaving a real evaluation set; 2 splits it evenly.
    /// A layer with fewer rows than this gets no arm rather than an unsplit one.
    /// Ignored when `gptq_eval_calib` is set, which supersedes it.
    gptq_holdout: usize = 3,
    /// §8C: score the GPTQ arm on the rows of a **second cache** — same checkpoint,
    /// disjoint prompt set — instead of on held-out rows of this one. A row split
    /// can only answer "does the compensation transfer to other patches of the same
    /// images"; this answers the question that decides shippability.
    gptq_eval_calib: ?[]const u8 = null,
    /// §8C: fit the Hessian on at most this many token rows, subsampled evenly from
    /// whatever the cache holds. Only useful for one thing, but an important one —
    /// sweeping it traces how the win scales with calibration data, which says
    /// whether a bigger capture is worth paying for without having to take one.
    gptq_train_rows: ?usize = null,
    /// Stop after this many layers. For a quick look at a long run, not for a
    /// report anyone should trust — a partial ranking is a partial ranking.
    max_layers: ?usize = null,
    threads: usize = 0,
    /// Skip the model-hash check. Only for deliberately measuring a cache
    /// against a *different* checkpoint (e.g. the unquantized sibling).
    allow_hash_mismatch: bool = false,
    log: ?*std.Io.Writer = null,
    callbacks: cb.CaptureCallbacks = .{},
};

/// Concatenate the row samples the options select into one `[m, cols]` block.
fn gatherX(
    gpa: std.mem.Allocator,
    cache: *const CalibrationCache.Cache,
    name: []const u8,
    which: ?usize,
    cols: *usize,
) ![]f32 {
    var out: std.ArrayList(f32) = .empty;
    errdefer out.deinit(gpa);

    const first: usize = which orelse 0;
    const last: usize = if (which) |w| w + 1 else cache.prov.buckets;
    for (first..last) |k| {
        const b = try cache.bucket(name, k);
        if (b.kept == 0) continue;
        if (cols.* == 0) cols.* = b.cols else if (b.cols != cols.*) return error.ShapeMismatch;
        const rows = try b.rowsAlloc(gpa);
        defer gpa.free(rows);
        try out.appendSlice(gpa, rows);
    }
    return out.toOwnedSlice(gpa);
}

/// Run level 1. The caller owns the returned `Report`.
pub fn run(gpa: std.mem.Allocator, io: std.Io, opts: Options) !Report {
    var cache = try CalibrationCache.Cache.open(gpa, io, opts.calib_path);
    defer cache.deinit();

    var ck = try tp.safetensors.SafeTensors.open(gpa, io, opts.model_path);
    defer ck.deinit();

    // Refuse to measure a cache that does not belong to this checkpoint before
    // spending the compute — and before producing a ranking that would look
    // perfectly plausible.
    var diag: CalibrationCache.Diagnostic = .{};
    CalibrationCache.validate(&cache, .{
        .checkpoint = &ck,
        // Rows were already scanned when the cache was written; re-scanning
        // gigabytes here buys nothing.
        .scan_rows = false,
    }, &diag) catch |err| {
        std.log.err("calibration cache '{s}' does not validate: {s}", .{ opts.calib_path, diag.msg });
        return err;
    };
    if (!opts.allow_hash_mismatch) {
        var hash: [16]u8 = undefined;
        try hashCheckpoint(io, opts.model_path, &hash);
        if (!std.mem.eql(u8, &hash, cache.prov.model_hash)) {
            std.log.err(
                "cache was captured from a different checkpoint (cache {s}, this model {s}). " ++
                    "Re-capture, or pass the flag to measure across checkpoints deliberately.",
                .{ cache.prov.model_hash, hash },
            );
            return error.ModelHashMismatch;
        }
    }

    // §8C's evaluation cache, when one was given. Validated against the same
    // checkpoint as the primary — a cache from a different model would make the
    // "held-out prompts" number meaningless in a way nothing downstream could see.
    var eval_cache: ?CalibrationCache.Cache = if (opts.gptq and opts.gptq_eval_calib != null)
        try CalibrationCache.Cache.open(gpa, io, opts.gptq_eval_calib.?)
    else
        null;
    defer if (eval_cache) |*c| c.deinit();
    if (eval_cache) |*c| {
        var ediag: CalibrationCache.Diagnostic = .{};
        CalibrationCache.validate(c, .{ .checkpoint = &ck, .scan_rows = false }, &ediag) catch |err| {
            std.log.err("GPTQ evaluation cache '{s}' does not validate: {s}", .{ opts.gptq_eval_calib.?, ediag.msg });
            return err;
        };
        if (!opts.allow_hash_mismatch and !std.mem.eql(u8, c.prov.model_hash, cache.prov.model_hash)) {
            std.log.err(
                "GPTQ evaluation cache was captured from a different checkpoint than the primary cache " ++
                    "({s} vs {s}); the two must differ only in prompts",
                .{ c.prov.model_hash, cache.prov.model_hash },
            );
            return error.ModelHashMismatch;
        }
        if (std.mem.eql(u8, c.prov.prompt_set, cache.prov.prompt_set)) {
            // Not fatal: fitting and scoring on two captures of the *same* prompts
            // is a legitimate control (it prices capture-to-capture noise, which is
            // the floor any prompt-generalization number sits on). But it is not the
            // experiment the flag is named for, and the report must not imply it is.
            std.log.warn(
                "both caches use prompt set '{s}', so the GPTQ arm measures capture noise, " ++
                    "not prompt generalization",
                .{c.prov.prompt_set},
            );
        }
    }

    var pool: ThreadPool = undefined;
    const n_jobs = if (opts.threads == 0) @max(1, std.Thread.getCpuCount() catch 1) else opts.threads;
    try pool.init(.{ .allocator = gpa, .n_jobs = n_jobs });
    defer pool.deinit();

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();

    const names = cache.layers();
    const limit = @min(names.len, opts.max_layers orelse names.len);
    var results: std.ArrayList(LayerResult) = .empty;
    try results.ensureTotalCapacity(arena, limit);

    var ref_dtype: ?tp.DType = null;
    var kernel_arms: usize = 0;
    var weighted_arms: usize = 0;
    var eq_arm_count: usize = 0;
    var gptq_arm_count: usize = 0;

    // The architecture's foldability relation (§8B), looked up from the cache's own
    // provenance. Null for an architecture `ImageArch` does not know, which the
    // report distinguishes from "known, and nothing folds".
    const arch = imagearch.byName(cache.prov.arch);
    var folds: struct { exact: usize = 0, shift: usize = 0, none: usize = 0 } = .{};

    // Per-column activation energy, keyed as the converter keys it. Built once;
    // every layer's vector is a view into it.
    //
    // Every activation-aware arm reads the same per-channel energy, so one build
    // serves them all — and each has to be listed, because §8B and §8C need it even
    // when the §8A *arm* is switched off. For §8C that is not merely a missing
    // measurement: it picks the grid both of its arms share, so building it only for
    // `weighted_arm` would leave `--gptq --no-weighted-arm` quietly comparing on the
    // plain quantizer instead of the shipped one, and reporting a ratio against the
    // wrong baseline.
    var imat: ?Imatrix.Imatrix = if (opts.weighted_arm or opts.eq_alphas.len > 0 or opts.gptq)
        try Imatrix.fromCache(gpa, &cache)
    else
        null;
    defer if (imat) |*im| im.deinit();

    for (names[0..limit], 0..) |name, li| {
        if (opts.callbacks.isCancelled()) return error.Canceled;

        const view = ck.get(name) orelse return error.LayerNotInCheckpoint;
        if (view.info.shape.rank != 2) {
            // Level 1 is defined on a GEMM. A non-matrix tensor was tagged by
            // mistake; skipping it silently would leave a hole in the ranking,
            // so say so.
            std.log.warn("{s}: rank {d} tensor is not a GEMM weight; skipped", .{ name, view.info.shape.rank });
            continue;
        }
        if (ref_dtype == null) ref_dtype = view.info.dtype;

        const rows = view.info.shape.dims[0];
        const cols_ck = view.info.shape.dims[1];

        var cols: usize = 0;
        const x = try gatherX(gpa, &cache, name, opts.bucket, &cols);
        defer gpa.free(x);
        if (x.len == 0) continue;
        if (cols != cols_ck) return error.ShapeMismatch;
        const m = x.len / cols;

        // W as the model sees it, in f32.
        const w = view.toF32Alloc(gpa) catch |err| {
            std.log.err("{s}: cannot dequantize a {t} checkpoint tensor to f32: {t}", .{ name, view.info.dtype, err });
            return err;
        };
        defer gpa.free(w);

        const y_ref = try gpa.alloc(f32, m * rows);
        defer gpa.free(y_ref);
        try tp.ops.matmul.matmul(io, gpa, y_ref, x, m, tp.ops.matmul.Weight.fromF32(w, rows, cols), null);

        const y_apx = try gpa.alloc(f32, m * rows);
        defer gpa.free(y_apx);

        // §8C's split, its two reference outputs and its per-basis Hessians, all
        // shared across formats. Null when the arm is off or this layer's sample is
        // too thin to hold anything back.
        var x_eval: []f32 = &.{};
        defer gpa.free(x_eval);
        if (eval_cache) |*c| {
            var ecols: usize = 0;
            x_eval = gatherX(gpa, c, name, opts.bucket, &ecols) catch &.{};
            if (x_eval.len > 0 and ecols != cols) {
                // Same checkpoint, so a width disagreement means the two captures
                // saw different graphs. Refusing the layer is right; silently
                // scoring against the wrong activations is not.
                std.log.warn("{s}: evaluation cache has {d} columns, primary has {d}; GPTQ arm omitted", .{ name, ecols, cols });
                gpa.free(x_eval);
                x_eval = &.{};
            }
        }

        var gctx: ?GptqCtx = if (opts.gptq) blk: {
            const made = if (eval_cache != null)
                if (x_eval.len == 0) null else GptqCtx.initPair(gpa, io, x, x_eval, rows, cols, w, opts.gptq_damp, opts.gptq_train_rows, &pool) catch |err| {
                    if (err != error.SampleTooSmall) return err;
                    break :blk null;
                }
            else
                GptqCtx.initSplit(gpa, io, x, m, rows, cols, w, opts.gptq_holdout, opts.gptq_damp, opts.gptq_train_rows, &pool) catch |err| {
                    if (err != error.SampleTooSmall) return err;
                    std.log.warn(
                        "{s}: {d} token rows cannot be split 1-in-{d} for the GPTQ arm; omitted",
                        .{ name, m, opts.gptq_holdout },
                    );
                    break :blk null;
                };
            break :blk made;
        } else null;
        defer if (gctx) |*g| g.deinit();

        const per_format = try arena.alloc(FormatResult, opts.formats.len);
        var kept: usize = 0;

        for (opts.formats) |fmt| {
            const w_hat = ph.roundtripGroup(fmt, gpa, w, rows, cols, &pool, opts.convrot_group) catch |err| {
                // A format whose block/group constraints this shape violates is a
                // fact about the shape, not a failure of the run.
                std.log.warn("{s} / {s}: round-trip failed ({t}); omitted", .{ name, formatName(fmt), err });
                continue;
            };
            defer gpa.free(w_hat);

            try tp.ops.matmul.matmul(io, gpa, y_apx, x, m, tp.ops.matmul.Weight.fromF32(w_hat, rows, cols), null);
            const format_arm = armMetrics(y_ref, y_apx, m, rows);

            const kernel_arm: ?ArmMetrics = if (opts.kernel_arm)
                kernelArm(gpa, io, w, x, y_apx, y_ref, m, rows, cols, fmt, &pool) catch |err| blk: {
                    std.log.warn("{s} / {s}: kernel arm failed ({t}); format arm only", .{ name, formatName(fmt), err });
                    break :blk null;
                }
            else
                null;
            if (kernel_arm != null) kernel_arms += 1;

            const weighted_arm: ?ArmMetrics = if (opts.weighted_arm) if (imat) |*im|
                weightedArm(gpa, io, im, name, x, y_apx, y_ref, w, m, rows, cols, fmt, &pool, opts.convrot_group) catch |err| blk: {
                    std.log.warn("{s} / {s}: weighted arm failed ({t}); omitted", .{ name, formatName(fmt), err });
                    break :blk null;
                }
            else
                null else null;
            if (weighted_arm != null) weighted_arms += 1;

            // §8B. Stacks on the shipped configuration: weighted where `convert`
            // applies weighting, plain where it does not.
            const eq: []const EqArm = if (opts.eq_alphas.len > 0) if (imat) |*im| blk: {
                const dst = ph.ggufDstType(fmt) orelse ph.clusterDstType(fmt);
                const on_weighted = weighted_arm != null and
                    if (dst) |d| Imatrix.shipsWeighted(d) else false;
                break :blk eqArms(gpa, arena, io, im, name, x, y_apx, y_ref, w, m, rows, cols, fmt, &pool, .{
                    .alphas = opts.eq_alphas,
                    .max_ratio = opts.eq_max_ratio,
                    .on_weighted = on_weighted,
                }) catch |err| bad: {
                    std.log.warn("{s} / {s}: equalization arm failed ({t}); omitted", .{ name, formatName(fmt), err });
                    break :bad &.{};
                };
            } else &.{} else &.{};
            eq_arm_count += eq.len;

            // §8C. Like §8B it stacks on the shipped configuration, but it owns the
            // rounding rather than the input, so it needs its own reference outputs
            // on the split — which is what `gctx` carries.
            const gptq_arm: ?GptqArm = if (gctx) |*g|
                gptqArm(gpa, g, if (imat) |*im| im else null, name, w, rows, cols, fmt, &pool, opts.convrot_group) catch |err| blk: {
                    std.log.warn("{s} / {s}: GPTQ arm failed ({t}); omitted", .{ name, formatName(fmt), err });
                    break :blk null;
                }
            else
                null;
            if (gptq_arm != null) gptq_arm_count += 1;

            per_format[kept] = .{
                .fmt = fmt,
                .bits = bitsFor(fmt),
                .format_arm = format_arm,
                .kernel_arm = kernel_arm,
                .weighted_arm = weighted_arm,
                .eq_arms = eq,
                .gptq_arm = gptq_arm,
            };
            kept += 1;
        }

        switch (if (arch) |a| a.foldability(name) else .none) {
            .exact => folds.exact += 1,
            .runtime_shift => folds.shift += 1,
            .none => folds.none += 1,
        }

        results.appendAssumeCapacity(.{
            .name = try arena.dupe(u8, name),
            .rows = rows,
            .cols = cols,
            .tokens = m,
            .per_format = per_format[0..kept],
        });

        opts.callbacks.reportProgress(@intCast(li + 1), @intCast(limit), 0, 0);
        if (opts.log) |w_log| {
            const r = results.items[results.items.len - 1];
            if (r.find(opts.reference_format)) |ref| {
                try w_log.print("[{d}/{d}] {s}  {s} rel-L2 {d:.5}\n", .{
                    li + 1, limit, name, formatName(opts.reference_format), ref.format_arm.rel_l2,
                });
            } else {
                try w_log.print("[{d}/{d}] {s}\n", .{ li + 1, limit, name });
            }
            try w_log.flush();
        }
    }

    const sc = try scoreLayers(gpa, results.items, opts.reference_format, arch);
    const scored = sc.scored;
    if (scored < results.items.len) {
        std.log.warn(
            "{d} of {d} layers could not be measured at the reference format {s} (shape constraints); " ++
                "they are left out of the sensitivities JSON rather than scored as insensitive",
            .{ results.items.len - scored, results.items.len, formatName(opts.reference_format) },
        );
    }
    // A truncated sweep rescales the whole ladder (the anchor is the measured
    // population's median), so it is recorded rather than left to be inferred.
    const complete = opts.max_layers == null or opts.max_layers.? >= names.len;
    if (!complete) {
        std.log.warn(
            "--max-layers stopped this sweep at {d} of {d} layers. The median anchor is that " ++
                "subset's, so EVERY score is rescaled — the JSON is marked incomplete and the " ++
                "converter will warn. Do not route on it.",
            .{ limit, names.len },
        );
    }

    const dt = ref_dtype orelse tp.DType.f32;
    return .{
        .arena = arena_state,
        .layers = results.items,
        .reference_format = opts.reference_format,
        .model_path = try arena.dupe(u8, opts.model_path),
        .calib_path = try arena.dupe(u8, opts.calib_path),
        .arch = try arena.dupe(u8, cache.prov.arch),
        .prompt_set = try arena.dupe(u8, cache.prov.prompt_set),
        .reference_is_quantized = dt != .f32 and dt != .f16 and dt != .bf16,
        .reference_dtype = @tagName(dt),
        .kernel_arms = kernel_arms,
        .weighted_arms = weighted_arms,
        .eq_arms = eq_arm_count,
        .eq_alphas = try arena.dupe(f32, opts.eq_alphas),
        .convrot_group = opts.convrot_group,
        .gptq_arms = gptq_arm_count,
        .gptq_damp = if (opts.gptq) opts.gptq_damp else 0,
        .gptq_holdout = if (opts.gptq and eval_cache == null) opts.gptq_holdout else 0,
        .gptq_eval_calib = if (eval_cache != null) try arena.dupe(u8, opts.gptq_eval_calib.?) else "",
        .gptq_eval_prompt_set = if (eval_cache) |c| try arena.dupe(u8, c.prov.prompt_set) else "",
        .fold_exact = folds.exact,
        .fold_runtime_shift = folds.shift,
        .fold_none = folds.none,
        .fold_known = arch != null,
        .scored = scored,
        .anchor_population = sc.anchor_population,
        .complete = complete,
    };
}

/// What one call to `eqArms` should measure.
const EqSpec = struct {
    alphas: []const f32,
    max_ratio: f32,
    /// Quantize with the §8A weighted search (on the *equalized* importance), as
    /// opposed to the plain encoder.
    on_weighted: bool,
};

/// Plan §8B: quantize `W·diag(s)` and evaluate it on `X/s`.
///
/// Two things here are easy to get wrong and would both manufacture a win:
///
/// 1. **The activations must be folded too.** `X · (W·diag(s))ᵀ` is a different
///    layer, not an approximation of this one. The reference `Y` stays `X·Wᵀ`,
///    because that is the function the fold leaves invariant.
/// 2. **A stacked weighted search must get the equalized importance** `w/s²`. The
///    fold moves importance out of the objective and into the weights; handing the
///    search the original `w` would double-count it, the same error as passing
///    unrotated weights to a ConvRot format.
///
/// Returns an arena-allocated slice with one entry per α that could be measured,
/// which is all of them or none — a shape the format cannot take fails on the
/// first α and there is no reason to retry it.
fn eqArms(
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    io: std.Io,
    im: *const Imatrix.Imatrix,
    layer: []const u8,
    x: []const f32,
    y_scratch: []f32,
    y_ref: []const f32,
    w: []const f32,
    m: usize,
    rows: usize,
    cols: usize,
    fmt: Format,
    pool: *ThreadPool,
    spec: EqSpec,
) ![]const EqArm {
    const weights = im.get(layer) orelse return &.{};
    if (weights.len != cols) return &.{};

    // No grouping checks here on purpose: the caller only sets `on_weighted` when
    // the weighted arm already ran for this (layer, format), which is exactly the
    // condition `weightedArm` gates on, and the plain path already round-tripped
    // this shape for the format arm.
    const out = try arena.alloc(EqArm, spec.alphas.len);
    var kept: usize = 0;
    for (spec.alphas) |alpha| {
        const s = try Equalize.scales(gpa, weights, .{ .alpha = alpha, .max_ratio = spec.max_ratio });
        defer gpa.free(s);

        const w_eq = try Equalize.foldIntoWeights(gpa, w, rows, cols, s);
        defer gpa.free(w_eq);
        const x_eq = try Equalize.foldIntoActivations(gpa, x, m, cols, s);
        defer gpa.free(x_eq);

        const w_hat = if (spec.on_weighted) blk: {
            const imp = try Equalize.foldedWeights(gpa, weights, s);
            defer gpa.free(imp);
            break :blk try ph.roundtripWeighted(fmt, gpa, w_eq, rows, cols, pool, imp);
        } else try ph.roundtrip(fmt, gpa, w_eq, rows, cols, pool);
        defer gpa.free(w_hat);

        try tp.ops.matmul.matmul(io, gpa, y_scratch, x_eq, m, tp.ops.matmul.Weight.fromF32(w_hat, rows, cols), null);
        out[kept] = .{
            .alpha = alpha,
            .on_weighted = spec.on_weighted,
            .metrics = armMetrics(y_ref, y_scratch, m, rows),
        };
        kept += 1;
    }
    return out[0..kept];
}

/// The format arm re-run with this layer's activation energy steering whichever
/// scale search the format has — ggml's `imatrix` for the block-quant types, the
/// clipping search for scaled-fp8 and the int8/int4 clusters (plan §8A).
///
/// Returns null when there is nothing to measure: a format with no searchable
/// scale, a layer the cache has no usable statistics for, or a row width the
/// format's grouping cannot accommodate.
///
/// Gated on `Imatrix.weightKind` (what mechanism *exists*), deliberately not on
/// `Imatrix.shipsWeighted` (what the converter applies). Measuring a format the
/// converter withholds is the only way a policy call stays checkable — that is
/// how q2_k's exclusion was overturned and how SCALED_F8's was established.
fn weightedArm(
    gpa: std.mem.Allocator,
    io: std.Io,
    im: *const Imatrix.Imatrix,
    layer: []const u8,
    x: []const f32,
    y_scratch: []f32,
    y_ref: []const f32,
    w: []const f32,
    m: usize,
    rows: usize,
    cols: usize,
    fmt: Format,
    pool: *ThreadPool,
    group: usize,
) !?ArmMetrics {
    const dst = ph.ggufDstType(fmt) orelse ph.clusterDstType(fmt) orelse return null;
    const kind = Imatrix.weightKind(dst);
    if (kind == .none) return null;

    // Cache keys carry the checkpoint's container prefix; `get` strips it.
    const weights = im.get(layer) orelse return null;
    if (weights.len != cols) return null;

    switch (kind) {
        .none => unreachable,
        .ggml_block => {
            const gt = gguf.GgmlType.fromString(@tagName(dst)) catch return null;
            const block = gt.getBlockSize();
            if (block == 0 or cols % block != 0) return null;
        },
        .rotated_int => if (group == 0 or cols % group != 0) return null,
        .plain_int => if (cols % 2 != 0) return null,
        .global_fp8 => {},
    }

    const w_hat = try ph.roundtripWeightedGroup(fmt, gpa, w, rows, cols, pool, weights, group);
    defer gpa.free(w_hat);

    try tp.ops.matmul.matmul(io, gpa, y_scratch, x, m, tp.ops.matmul.Weight.fromF32(w_hat, rows, cols), null);
    return armMetrics(y_ref, y_scratch, m, rows);
}

// ---------------------------------------------------------------------------
// §8C — GPTQ
// ---------------------------------------------------------------------------

/// How §8C reaches a given format — the two mechanisms in `Gptq`.
pub const GptqKind = union(enum) {
    /// GPTQ owns the rounding, against a symmetric per-row grid, one column at a
    /// time. The group size is the harness's, matching the weighted arm, so a §8A
    /// and a §8C row of the same format are measured in the same basis.
    grid: struct { grid: Gptq.Grid, basis: Gptq.Basis },
    /// ggml owns the encoder; GPTQ only chooses what values it sees, one block at a
    /// time. The GGUF path.
    ggml: types.DataType,
};

/// What §8C can be measured for, and how.
///
/// The legacy `q4_0`/`q5_0`/`q4_1`/`q5_1` types are absent on purpose even though
/// they take an imatrix: their weighted encoder normalizes over the whole row
/// (`Gptq.ggmlBlockLocalWeighted`), so quantizing block-at-a-time would change the
/// encoder as well as the values, and the ratio would no longer isolate the
/// compensation. Everything else absent has no scale search to steer at all.
fn gptqSpec(fmt: Format, group: usize) ?GptqKind {
    const rotated: Gptq.Basis = .{ .convrot = true, .group_size = group };
    return switch (fmt) {
        .int4 => .{ .grid = .{ .grid = Gptq.int4_grid, .basis = .{} } },
        .int8 => .{ .grid = .{ .grid = Gptq.int8_grid, .basis = .{} } },
        .int4_convrot => .{ .grid = .{ .grid = Gptq.int4_grid, .basis = rotated } },
        .int8_convrot => .{ .grid = .{ .grid = Gptq.int8_grid, .basis = rotated } },
        .q2_k => .{ .ggml = .q2_k },
        .q3_k => .{ .ggml = .q3_k },
        .q4_k => .{ .ggml = .q4_k },
        .q5_k => .{ .ggml = .q5_k },
        .q6_k => .{ .ggml = .q6_k },
        else => null,
    };
}

/// Everything the GPTQ arm needs that does not depend on the format: the row split,
/// a reference output for each half, and the Hessians, which depend on the basis
/// and so are shared between the rotated formats and between the plain ones.
const GptqCtx = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    pool: *ThreadPool,
    damp: f32,
    cols: usize,
    out_rows: usize,
    /// `[train_tokens, cols]` and `[eval_tokens, cols]`, disjoint by construction.
    x_train: []f32,
    x_eval: []f32,
    train_tokens: usize,
    eval_tokens: usize,
    y_train: []f32,
    y_eval: []f32,
    /// Scratch for one arm's output, sized for the larger half.
    y_apx: []f32,
    plain: ?Gptq.Hessian = null,
    rotated: ?Gptq.Hessian = null,

    /// Fit on one cache's rows, score on **another cache's** — captured from the
    /// same checkpoint with a disjoint prompt set. The strictly harder question,
    /// and the one that decides whether §8C is shippable: a row split can only ever
    /// say "does this transfer to other patches of the same images".
    ///
    /// Note the asymmetry with `initSplit`, which is deliberate and makes the
    /// comparison conservative rather than confounded: here the Hessian gets *all*
    /// of the primary cache's rows, not two thirds of them. More training data can
    /// only help the fit, so if this regime still scores worse than the row split
    /// did, the difference is attributable to the prompts.
    fn initPair(
        gpa: std.mem.Allocator,
        io: std.Io,
        x_train: []const f32,
        x_eval: []const f32,
        out_rows: usize,
        cols: usize,
        w: []const f32,
        damp: f32,
        cap: ?usize,
        pool: *ThreadPool,
    ) !GptqCtx {
        if (x_train.len == 0 or x_eval.len == 0) return error.SampleTooSmall;
        if (x_train.len % cols != 0 or x_eval.len % cols != 0) return error.ShapeMismatch;
        return initFrom(gpa, io, x_train, x_eval, out_rows, cols, w, damp, cap, pool);
    }

    /// Evenly-spaced subsample of `n` rows down to `cap`, `idx = i·n/cap`. Spread
    /// rather than truncated, for the same reason the held-out split strides: the
    /// row block concatenates schedule buckets in order, and taking a prefix would
    /// silently fit the Hessian on early-sigma activations alone.
    fn capRows(gpa: std.mem.Allocator, x: []const f32, cols: usize, cap: usize) ![]f32 {
        const n = x.len / cols;
        const out = try gpa.alloc(f32, cap * cols);
        errdefer gpa.free(out);
        for (0..cap) |i| {
            const src = (i * n) / cap;
            @memcpy(out[i * cols ..][0..cols], x[src * cols ..][0..cols]);
        }
        return out;
    }

    /// Held-out rows are taken by a fixed stride rather than by a shuffle. Two
    /// reasons: it is reproducible without carrying a seed into the report, and
    /// `gatherX` concatenates the schedule buckets in order, so a stride keeps the
    /// split proportional across them — a random subset of a 96-row sample could
    /// easily take most of its evaluation rows from one end of the schedule.
    fn initSplit(
        gpa: std.mem.Allocator,
        io: std.Io,
        x: []const f32,
        m: usize,
        out_rows: usize,
        cols: usize,
        w: []const f32,
        holdout: usize,
        damp: f32,
        cap: ?usize,
        pool: *ThreadPool,
    ) !GptqCtx {
        if (holdout < 2) return error.InvalidHoldout;
        var eval_n: usize = 0;
        for (0..m) |i| {
            if (i % holdout == holdout - 1) eval_n += 1;
        }
        const train_n = m - eval_n;
        if (eval_n == 0 or train_n == 0) return error.SampleTooSmall;

        const train = try gpa.alloc(f32, train_n * cols);
        defer gpa.free(train);
        const eval = try gpa.alloc(f32, eval_n * cols);
        defer gpa.free(eval);

        var ti: usize = 0;
        var ei: usize = 0;
        for (0..m) |i| {
            const src = x[i * cols ..][0..cols];
            if (i % holdout == holdout - 1) {
                @memcpy(eval[ei * cols ..][0..cols], src);
                ei += 1;
            } else {
                @memcpy(train[ti * cols ..][0..cols], src);
                ti += 1;
            }
        }
        return initFrom(gpa, io, train, eval, out_rows, cols, w, damp, cap, pool);
    }

    fn initFrom(
        gpa: std.mem.Allocator,
        io: std.Io,
        x_train_full: []const f32,
        x_eval: []const f32,
        out_rows: usize,
        cols: usize,
        w: []const f32,
        damp: f32,
        cap: ?usize,
        pool: *ThreadPool,
    ) !GptqCtx {
        // Only the *training* block is capped. Capping the evaluation block too
        // would change what the number means as well as how it was fitted.
        var capped: ?[]f32 = null;
        defer if (capped) |c| gpa.free(c);
        if (cap) |k| {
            if (k == 0) return error.SampleTooSmall;
            if (k < x_train_full.len / cols) capped = try capRows(gpa, x_train_full, cols, k);
        }
        const x_train: []const f32 = capped orelse x_train_full;

        const train_n = x_train.len / cols;
        const eval_n = x_eval.len / cols;

        var self: GptqCtx = .{
            .gpa = gpa,
            .io = io,
            .pool = pool,
            .damp = damp,
            .cols = cols,
            .out_rows = out_rows,
            .x_train = try gpa.dupe(f32, x_train),
            .x_eval = undefined,
            .train_tokens = train_n,
            .eval_tokens = eval_n,
            .y_train = undefined,
            .y_eval = undefined,
            .y_apx = undefined,
        };
        errdefer gpa.free(self.x_train);
        self.x_eval = try gpa.dupe(f32, x_eval);
        errdefer gpa.free(self.x_eval);
        self.y_train = try gpa.alloc(f32, train_n * out_rows);
        errdefer gpa.free(self.y_train);
        self.y_eval = try gpa.alloc(f32, eval_n * out_rows);
        errdefer gpa.free(self.y_eval);
        self.y_apx = try gpa.alloc(f32, @max(train_n, eval_n) * out_rows);
        errdefer gpa.free(self.y_apx);

        const ref = tp.ops.matmul.Weight.fromF32(w, out_rows, cols);
        try tp.ops.matmul.matmul(io, gpa, self.y_train, self.x_train, train_n, ref, null);
        try tp.ops.matmul.matmul(io, gpa, self.y_eval, self.x_eval, eval_n, ref, null);
        return self;
    }

    fn deinit(self: *GptqCtx) void {
        if (self.plain) |*h| h.deinit();
        if (self.rotated) |*h| h.deinit();
        self.gpa.free(self.x_train);
        self.gpa.free(self.x_eval);
        self.gpa.free(self.y_train);
        self.gpa.free(self.y_eval);
        self.gpa.free(self.y_apx);
        self.* = undefined;
    }

    /// The Hessian for a basis, built on first use. Built from the *training* rows
    /// only — the whole point of the split.
    fn hessian(self: *GptqCtx, basis: Gptq.Basis) !*const Gptq.Hessian {
        const slot = if (basis.convrot) &self.rotated else &self.plain;
        if (slot.* == null) {
            slot.* = try Gptq.Hessian.init(
                self.gpa,
                self.x_train,
                self.train_tokens,
                self.cols,
                basis,
                self.damp,
                self.pool,
            );
        }
        return &slot.*.?;
    }

    /// One candidate Ŵ, scored on both halves of the split.
    fn score(self: *GptqCtx, w_hat: []const f32) !struct { held_out: ArmMetrics, in_sample: ArmMetrics } {
        const weight = tp.ops.matmul.Weight.fromF32(w_hat, self.out_rows, self.cols);
        const ho = self.y_apx[0 .. self.eval_tokens * self.out_rows];
        try tp.ops.matmul.matmul(self.io, self.gpa, ho, self.x_eval, self.eval_tokens, weight, null);
        const held_out = armMetrics(self.y_eval, ho, self.eval_tokens, self.out_rows);

        const is = self.y_apx[0 .. self.train_tokens * self.out_rows];
        try tp.ops.matmul.matmul(self.io, self.gpa, is, self.x_train, self.train_tokens, weight, null);
        return .{
            .held_out = held_out,
            .in_sample = armMetrics(self.y_train, is, self.train_tokens, self.out_rows),
        };
    }
};

/// Plan §8C: keep the Gram's off-diagonal terms.
///
/// The comparison is against the *shipped* configuration — the §8A weighted search
/// where `convert --calib` applies one, the plain quantizer otherwise — and both
/// arms are handed the same importance vector, so they end up on the same grid and
/// the only difference measured is which level each weight rounds to. Isolating it
/// that way is what makes the ratio attributable to the compensation rather than to
/// a scale change.
///
/// Returns null for a format with no grid GPTQ can own.
fn gptqArm(
    gpa: std.mem.Allocator,
    ctx: *GptqCtx,
    im: ?*const Imatrix.Imatrix,
    layer: []const u8,
    w: []const f32,
    rows: usize,
    cols: usize,
    fmt: Format,
    pool: *ThreadPool,
    group: usize,
) !?GptqArm {
    const spec = gptqSpec(fmt, group) orelse return null;

    // The grid follows what `convert` would ship for this format, weighting
    // included; `null` means the plain quantizer, exactly as in `roundtripWeighted`.
    const dst = ph.ggufDstType(fmt) orelse ph.clusterDstType(fmt);
    const ships_weighted = if (dst) |d| Imatrix.shipsWeighted(d) else false;
    const weights: ?[]const f32 = if (ships_weighted) if (im) |m| blk: {
        const got = m.get(layer) orelse break :blk null;
        break :blk if (got.len == cols) got else null;
    } else null else null;

    // Shape constraints are a fact about the tensor, not a failure: report no arm
    // rather than an error, the same way `weightedArm` does.
    var block: usize = 0;
    switch (spec) {
        .grid => |g| if (g.basis.convrot and cols % g.basis.group_size != 0) return null,
        .ggml => |d| {
            const gt = gguf.GgmlType.fromString(@tagName(d)) catch return null;
            block = @intCast(gt.getBlockSize());
            // Per-*row* block alignment, which is stricter than the flat blocking
            // the plain format arm gets away with (`first` is 6144×64: 393216
            // elements divide by 256, its rows do not).
            if (block == 0 or cols % block != 0) return null;
        },
    }

    const h = try ctx.hessian(switch (spec) {
        .grid => |g| g.basis,
        .ggml => .{},
    });

    const w_base = if (weights) |ws|
        try ph.roundtripWeightedGroup(fmt, gpa, w, rows, cols, pool, ws, group)
    else
        try ph.roundtripGroup(fmt, gpa, w, rows, cols, pool, group);
    defer gpa.free(w_base);

    const w_gptq = switch (spec) {
        .grid => |g| try Gptq.roundtrip(gpa, h, w, rows, cols, g.grid, weights, pool, .{ .damp = ctx.damp }),
        .ggml => |d| try Gptq.roundtripGgml(gpa, h, w, rows, cols, d, weights, pool, .{ .damp = ctx.damp }),
    };
    defer gpa.free(w_gptq);

    const base = try ctx.score(w_base);
    const comp = try ctx.score(w_gptq);

    return .{
        .train_tokens = ctx.train_tokens,
        .eval_tokens = ctx.eval_tokens,
        .damp = ctx.damp,
        .block = block,
        .on_weighted = weights != null,
        .held_out_base = base.held_out,
        .held_out = comp.held_out,
        .in_sample_base = base.in_sample,
        .in_sample = comp.in_sample,
    };
}

/// The native quantized GEMM, on the same inputs as the format arm. Returns null
/// when the format has no byte-array form or TensorPencil has no kernel for it.
fn kernelArm(
    gpa: std.mem.Allocator,
    io: std.Io,
    w: []const f32,
    x: []const f32,
    y_scratch: []f32,
    y_ref: []const f32,
    m: usize,
    rows: usize,
    cols: usize,
    fmt: Format,
    pool: *ThreadPool,
) !?ArmMetrics {
    const dst = kernelDataType(fmt) orelse return null;
    const dtype = toTp(dst) orelse return null;
    // ggml rows are whole blocks; a shape that does not divide has no kernel.
    if (dtype.blockElems() > 1 and cols % dtype.blockElems() != 0) return null;
    // Below `tp_small_m_max` TensorPencil's block-quant GEMM would also quantize
    // the activations (hygiene rule #4), which would silently turn this into a
    // W8A8 measurement in a column labelled "weight format loss". Ask for the
    // exact-activation path instead of refusing to measure: the quantity we want
    // is well-defined at every m, it is only the default dispatch that is not.
    const prev_exact = tp.ops.matmul.exact_activations;
    tp.ops.matmul.exact_activations = true;
    defer tp.ops.matmul.exact_activations = prev_exact;

    const bytes = try DataTransform.Quantizer.convertTensorData(
        gpa,
        std.mem.sliceAsBytes(w),
        .f32,
        dst,
        @intCast(rows * cols),
        pool,
    );
    defer gpa.free(bytes);
    if (bytes.len != dtype.storageBytes(rows * cols)) return error.QuantizedSizeMismatch;

    try tp.ops.matmul.matmul(io, gpa, y_scratch, x, m, tp.ops.matmul.Weight.init(bytes, dtype, rows, cols), null);
    return armMetrics(y_ref, y_scratch, m, rows);
}

fn hashCheckpoint(io: std.Io, path: []const u8, out: *[16]u8) !void {
    // Same construction as Calibrate.hashModel; kept here rather than imported so
    // this module does not depend on the capture driver.
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    var buf: [1 << 20]u8 = undefined;
    var reader = file.reader(io, &buf);
    var hasher = std.hash.Wyhash.init(0);
    var len: u64 = 0;
    while (true) {
        const chunk = reader.interface.peekGreedy(1) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        hasher.update(chunk);
        len += chunk.len;
        reader.interface.toss(chunk.len);
    }
    hasher.update(std.mem.asBytes(&len));
    _ = std.fmt.bufPrint(out, "{x:0>16}", .{hasher.final()}) catch unreachable;
}

pub fn formatName(fmt: Format) []const u8 {
    for (ph.formats) |f| if (f.fmt == fmt) return f.name;
    return @tagName(fmt);
}

fn bitsFor(fmt: Format) f32 {
    for (ph.formats) |f| if (f.fmt == fmt) return f.bits;
    return 0;
}

/// Parse a comma-separated format list ("q4_k,nvfp4,int4_convrot").
pub fn parseFormats(gpa: std.mem.Allocator, spec: []const u8) ![]Format {
    var out: std.ArrayList(Format) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const tok = std.mem.trim(u8, raw, " \t");
        if (tok.len == 0) continue;
        const fmt = blk: {
            for (ph.formats) |f| {
                if (std.ascii.eqlIgnoreCase(tok, f.name) or std.ascii.eqlIgnoreCase(tok, @tagName(f.fmt)))
                    break :blk f.fmt;
            }
            return error.UnknownFormat;
        };
        try out.append(gpa, fmt);
    }
    if (out.items.len == 0) return error.NoFormats;
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// `{"tensor.name": score}` — exactly the shape `Convert.zig` consumes, under
/// exact checkpoint tensor names, so it drops into the existing `--sensitivities`
/// path with zero converter changes.
pub fn writeSensitivitiesJson(report: *const Report, w: *std.Io.Writer) !void {
    try w.writeAll("{\n");
    // Provenance first, under a reserved key no checkpoint tensor can collide with.
    // Every score file before 2026-08-02 was a bare name→score map, so a percentile-era
    // file and a LadderScore-era one were indistinguishable without inspecting the
    // score *distribution* — which is exactly what nobody does before converting.
    try (LadderScore.Provenance{
        .generator = "level1-output-error",
        .arch = report.arch,
        .reference_format = formatName(report.reference_format),
        .scored = report.scored,
        .anchor_population = report.anchor_population,
        .complete = report.complete,
    }).write(w);
    var written: usize = 1;
    for (report.layers) |l| {
        const score = l.score orelse continue;
        if (written > 0) try w.writeAll(",\n");
        // Emit the *stripped* name, because that is what the converter looks up:
        // `Convert.zig`'s `filterAndStripTensors` removes the
        // `model.diffusion_model.` container prefix from every tensor before
        // quantization. A file keyed on the prefixed names the probe saw would
        // match nothing at all — and silently, since a miss is only a warning.
        //
        // `std.json.fmt` emits the surrounding quotes itself — adding our own
        // produced `""name""`, which is not JSON.
        try w.print("  {f}: {d:.6}", .{ std.json.fmt(imagearch.stripPrefix(l.name), .{}), score });
        written += 1;
    }
    if (written > 0) try w.writeByte('\n');
    try w.writeAll("}\n");
}

pub fn writeCsv(report: *const Report, w: *std.Io.Writer) !void {
    try w.writeAll("layer,rows,cols,tokens,score,format,bits,arm,rel_l2,mean_token_cos,max_token_rel\n");
    for (report.layers) |l| {
        for (l.per_format) |f| {
            try writeCsvRow(w, l, f, "format", f.format_arm);
            if (f.kernel_arm) |k| try writeCsvRow(w, l, f, "kernel", k);
            if (f.weighted_arm) |i| try writeCsvRow(w, l, f, "weighted", i);
            for (f.eq_arms) |e| {
                // The α and the base are both part of the arm's identity: an
                // `eq@0.50` row that stacked on the weighted search is not
                // comparable to one that stacked on the plain encoder.
                var buf: [32]u8 = undefined;
                const tag = try std.fmt.bufPrint(&buf, "eq@{d:.2}{s}", .{ e.alpha, if (e.on_weighted) "+w" else "" });
                try writeCsvRow(w, l, f, tag, e.metrics);
            }
            if (f.gptq_arm) |g| {
                // Four rows, and the `_ho` / `_is` suffix is load-bearing: these are
                // measured on subsets of the layer's `tokens`, not on all of them,
                // so they are not comparable to the arms above — only to each other.
                const suffix: []const u8 = if (g.on_weighted) "+w" else "";
                var buf: [40]u8 = undefined;
                try writeCsvRow(w, l, f, try std.fmt.bufPrint(&buf, "base_ho{s}", .{suffix}), g.held_out_base);
                try writeCsvRow(w, l, f, try std.fmt.bufPrint(&buf, "gptq_ho{s}", .{suffix}), g.held_out);
                try writeCsvRow(w, l, f, try std.fmt.bufPrint(&buf, "base_is{s}", .{suffix}), g.in_sample_base);
                try writeCsvRow(w, l, f, try std.fmt.bufPrint(&buf, "gptq_is{s}", .{suffix}), g.in_sample);
            }
        }
    }
}

fn writeCsvRow(w: *std.Io.Writer, l: LayerResult, f: FormatResult, arm: []const u8, mm: ArmMetrics) !void {
    try w.print("{s},{d},{d},{d},", .{ l.name, l.rows, l.cols, l.tokens });
    // Blank rather than a stand-in number: an unscored layer is unmeasured at the
    // reference format, and any placeholder would be read as data.
    if (l.score) |sc| try w.print("{d:.4}", .{sc});
    try w.print(",{s},{d},{s},{e:.6},{d:.8},{e:.6}\n", .{
        formatName(f.fmt), f.bits, arm, mm.rel_l2, mm.mean_token_cos, mm.max_token_rel,
    });
}

/// A human-readable summary: the provenance, the per-format aggregate (the
/// rate-distortion view), and the ranked layers.
pub fn writeMarkdown(report: *const Report, w: *std.Io.Writer, top_n: usize) !void {
    try w.print("# Level 1 — per-layer output error\n\n", .{});
    try w.print("- model: `{s}`\n", .{report.model_path});
    try w.print("- calibration: `{s}` (prompt set `{s}`)\n", .{ report.calib_path, report.prompt_set });
    try w.print("- arch: `{s}`\n", .{report.arch});
    try w.print("- layers measured: {d} ({d} scored at the reference format)\n", .{ report.layers.len, report.scored });
    try w.print(
        "- sensitivity score: `1 + 99·clamp(log2(d/d_median)/{d:.0}, 0, 1)` where `d` is {s} format-arm\n" ++
            "  rel-L2 and the median is over the layers the converter can route — one extra bit of\n" ++
            "  precision per doubling of damage (`LadderScore`). **Not** a percentile rank: see that\n" ++
            "  module for the 2.76×-off-the-curve model the rank encoding produced\n",
        .{ LadderScore.full_range_doublings, formatName(report.reference_format) },
    );
    // What this file would actually *do*, said before anyone spends an hour
    // converting and measuring it. The generator's encoding and the converter's
    // interpretation were designed separately once, and this is the seam.
    {
        const arch = imagearch.byName(report.arch);
        var routable: usize = 0;
        var upgraded: usize = 0;
        const thr = LadderScore.upgradeThreshold(50, LadderScore.default_ladder_levels).?;
        for (report.layers) |l| {
            const sc = l.score orelse continue;
            if (!LadderScore.isRoutable(arch, l.name)) continue;
            routable += 1;
            if (sc >= thr) upgraded += 1;
        }
        try w.print(
            "- at the default `-a 50` on a {d}-level ladder (q4_k→q8_0, the k-family case), **{d} of {d}**\n" ++
                "  scored routable layers would be upgraded off the target type (score ≥ {d:.1})\n",
            .{ LadderScore.default_ladder_levels, upgraded, routable, thr },
        );
    }
    if (report.scored < report.layers.len) {
        try w.print(
            "- {d} layers have no {s} measurement (their shape violates its block constraint); they are\n" ++
                "  omitted from the JSON rather than scored, so the converter falls back to its own rule\n",
            .{ report.layers.len - report.scored, formatName(report.reference_format) },
        );
    }
    if (report.reference_is_quantized) {
        // Plan open question 4. Absolute numbers against an already-quantized
        // baseline are still comparable to each other, but they are not "error
        // versus the true weights", and reporting them as such would be wrong.
        try w.print(
            "\n> **The reference is itself quantized.** The checkpoint stores `{s}` weights, so every\n" ++
                "> number here is error relative to an already-quantized baseline, not to full precision.\n" ++
                "> The ranking is unaffected; the absolute magnitudes are floors, not truths.\n" ++
                ">\n" ++
                "> One consequence is sharp enough to state separately: **any format that can represent\n" ++
                "> `{s}` exactly will score near zero here, and that is an artifact, not a result.** On an\n" ++
                "> fp8 checkpoint that means F8_E4M3, SCALED_F8 and MXFP8 (whose elements are e4m3) are\n" ++
                "> re-encoding values that already fit, so their rows say nothing about how those formats\n" ++
                "> would treat real full-precision weights. Compare them only on an unquantized checkpoint.\n",
            .{ report.reference_dtype, report.reference_dtype },
        );
    }

    if (report.kernel_arms > 0) {
        // Measured, and worth stating plainly: on CPU this column is expected to
        // read exactly zero, and a nonzero value is the interesting event.
        try w.writeAll(
            "\n> **What the kernel column means here.** TensorPencil's CPU block-quant GEMM dequantizes\n" ++
                "> each k-slice and runs the same f32 microkernel as the format arm, so on CPU the two arms\n" ++
                "> are expected to agree *exactly*. The column is a conformance check, not a second\n" ++
                "> measurement: a nonzero `kernel−format` on CPU means a kernel bug. It becomes a real\n" ++
                "> measurement once the GPU quant GEMMs exist. The arm runs with\n" ++
                "> `ops.matmul.exact_activations`, so it is weight-quantization loss only at every token\n" ++
                "> count — the small-m GEMV that would also quantize activations is never used here.\n",
        );
    }

    try w.writeAll("\n## Formats, averaged over layers\n\n");
    try w.writeAll("| format | bits | format rel-L2 | format cos | kernel rel-L2 | kernel−format |\n");
    try w.writeAll("|---|---:|---:|---:|---:|---:|\n");

    // Iterate the canonical order rather than whatever the first layer had, so
    // the table reads the same across runs.
    for (ph.formats) |spec| {
        var n: usize = 0;
        var sum_rel: f64 = 0;
        var sum_cos: f64 = 0;
        // The kernel comparison is accumulated over only the layers that have
        // BOTH arms, and the format side of that comparison is re-summed over the
        // same subset. Averaging the two arms over different populations makes a
        // missing kernel arm on one layer look like a kernel discrepancy on all of
        // them — which, given the note above tells the reader that a nonzero delta
        // means a kernel bug, would send them hunting one that does not exist.
        var kn: usize = 0;
        var sum_krel: f64 = 0;
        var sum_rel_paired: f64 = 0;
        for (report.layers) |l| {
            const f = l.find(spec.fmt) orelse continue;
            n += 1;
            sum_rel += f.format_arm.rel_l2;
            sum_cos += f.format_arm.mean_token_cos;
            if (f.kernel_arm) |k| {
                kn += 1;
                sum_krel += k.rel_l2;
                sum_rel_paired += f.format_arm.rel_l2;
            }
        }
        if (n == 0) continue;
        const fn_: f64 = @floatFromInt(n);
        const mean_rel = sum_rel / fn_;
        if (kn > 0) {
            const kfn: f64 = @floatFromInt(kn);
            const mean_krel = sum_krel / kfn;
            const delta = mean_krel - sum_rel_paired / kfn;
            try w.print("| {s} | {d} | {e:.4} | {d:.6} | {e:.4} | {s}{e:.2} |", .{
                spec.name, spec.bits, mean_rel, sum_cos / fn_, mean_krel,
                if (delta >= 0) "+" else "-", @abs(delta),
            });
            // Say when the kernel column covers fewer layers than the format
            // column, so its absolute value is not read as like-for-like either.
            if (kn != n) try w.print(" {d}/{d} layers", .{ kn, n });
            try w.writeByte('\n');
        } else {
            try w.print("| {s} | {d} | {e:.4} | {d:.6} | — | — |\n", .{
                spec.name, spec.bits, mean_rel, sum_cos / fn_,
            });
        }
    }

    if (report.weighted_arms > 0) try writeWeightedSection(w, report);
    if (report.eq_arms > 0) try writeEqSection(w, report);
    if (report.gptq_arms > 0) try writeGptqSection(w, report);

    try w.print("\n## Most sensitive layers ({s})\n\n", .{formatName(report.reference_format)});
    try w.writeAll("| layer | shape | score | rel-L2 | mean token cos | worst token |\n");
    try w.writeAll("|---|---|---:|---:|---:|---:|\n");

    const order = try report.arena.child_allocator.alloc(usize, report.layers.len);
    defer report.arena.child_allocator.free(order);
    for (order, 0..) |*o, i| o.* = i;
    const Ctx = struct {
        layers: []const LayerResult,
        fn lessThan(self: @This(), a: usize, b: usize) bool {
            return (self.layers[a].score orelse -1) > (self.layers[b].score orelse -1);
        }
    };
    std.mem.sort(usize, order, Ctx{ .layers = report.layers }, Ctx.lessThan);

    var shown: usize = 0;
    for (order) |i| {
        if (shown >= top_n) break;
        const l = report.layers[i];
        const score = l.score orelse continue;
        const f = l.find(report.reference_format) orelse continue;
        try w.print("| `{s}` | {d}×{d} | {d:.1} | {e:.4} | {d:.6} | {e:.4} |\n", .{
            l.name, l.rows, l.cols, score, f.format_arm.rel_l2, f.format_arm.mean_token_cos, f.format_arm.max_token_rel,
        });
        shown += 1;
    }
}

/// The activation-aware verdict: does steering the scale search with captured
/// per-channel energy actually reduce real output error, per format?
///
/// Paired over the layers that have both arms, and reporting how many layers went
/// each way — a mean that improves while most layers get worse is a different
/// finding from one that improves everywhere, and only the win/loss counts
/// separate them.
fn writeWeightedSection(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("\n## Activation-aware quantization, averaged over layers\n\n");
    try w.writeAll(
        "Same quantity as the format arm — real output error on captured activations — with the\n" ++
            "layer's per-channel energy steering the scale search: ggml's `quant_weights` for the\n" ++
            "block-quant types, the clipping search for scaled-fp8 and the int8/int4 clusters.\n" ++
            "Ratio below 1.0 means it helped.\n\n" ++
            "**shipped** is what `convert --calib` actually applies. A measured-but-not-shipped row is\n" ++
            "one this table's own numbers argued against; see `Imatrix.shipsWeighted` for the reasoning.\n\n",
    );
    try w.writeAll("| format | mechanism | shipped | layers | format rel-L2 | weighted rel-L2 | ratio | layers improved |\n");
    try w.writeAll("|---|---|:--:|---:|---:|---:|---:|---:|\n");

    for (ph.formats) |spec| {
        var n: usize = 0;
        var sum_f: f64 = 0;
        var sum_i: f64 = 0;
        var better: usize = 0;
        for (report.layers) |l| {
            const f = l.find(spec.fmt) orelse continue;
            const im = f.weighted_arm orelse continue;
            n += 1;
            sum_f += f.format_arm.rel_l2;
            sum_i += im.rel_l2;
            if (im.rel_l2 < f.format_arm.rel_l2) better += 1;
        }
        if (n == 0) continue;
        const fn_: f64 = @floatFromInt(n);
        const mf = sum_f / fn_;
        const mi = sum_i / fn_;
        const dst = ph.ggufDstType(spec.fmt) orelse ph.clusterDstType(spec.fmt);
        const mech = if (dst) |d| switch (Imatrix.weightKind(d)) {
            .ggml_block => "imatrix",
            .rotated_int => "clip (rotated)",
            .plain_int => "clip",
            .global_fp8 => "clip (global)",
            .none => "—",
        } else "—";
        const shipped = if (dst) |d| Imatrix.shipsWeighted(d) else false;
        try w.print("| {s} | {s} | {s} | {d} | {e:.4} | {e:.4} | {d:.4} | {d}/{d} |\n", .{
            spec.name, mech, if (shipped) "yes" else "no", n, mf, mi, if (mf > 0) mi / mf else 1.0, better, n,
        });
    }

    try w.writeAll(
        "\n> Formats absent from this table have no searchable scale to steer: f16/bf16 and plain fp8\n" ++
            "> encode elements directly, Q8_0 and MXFP4 discard `quant_weights`, and MXFP8/NVFP4 carry\n" ++
            "> block scales in a constrained encoding (power-of-two E8M0, per-block fp8) that a clipping\n" ++
            "> ratio is the wrong search for. Those are unimplemented, not measured-and-useless.\n",
    );
    if (report.convrot_group == TensorClusters.int4_convrot_group_size) {
        try w.print(
            "\n> **The ConvRot rows were measured at the shipped group size ({d}).** No transfer caveat\n" ++
                "> applies to them: this is the rotation `convert` writes.\n",
            .{report.convrot_group},
        );
    } else {
        try w.print(
            "\n> **Caveat on the ConvRot rows.** This run rotates in groups of {d}; the converter uses\n" ++
                "> {d}. Both arms here use the same group size, so the *ratio* is a clean comparison — but it\n" ++
                "> is measured at a different group size than the one `convert` ships, and the rotated\n" ++
                "> importance is a per-group mean, so the benefit is not guaranteed to transfer unchanged.\n" ++
                "> Pass `--convrot-group {d}` to measure the shipped configuration; for §8C that is not a\n" ++
                "> caveat but a different algorithm, since the group size selects the basis it compensates in.\n",
            .{ report.convrot_group, TensorClusters.int4_convrot_group_size, TensorClusters.int4_convrot_group_size },
        );
    }
}

/// §8B: does folding per-channel importance into the weights buy anything on top
/// of the configuration we already ship?
///
/// Paired per (format, α) against that format's shipped arm, with win/loss counts
/// for the same reason the §8A table has them. The foldability tally is printed
/// alongside because a gain is only collectable where the architecture's graph has
/// somewhere to put `1/s`.
fn writeEqSection(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("\n## Activation equalization (§8B), averaged over layers\n\n");
    try w.writeAll(
        "Quantize `W·diag(s)` and evaluate on `X/s`, with `s = w^(α/2)` from the same per-channel\n" ++
            "energy §8A steers with. Stacked on each format's **shipped** configuration (the weighted\n" ++
            "search where `convert --calib` applies one, the plain encoder otherwise), so the ratio\n" ++
            "answers \"does this add anything to what we ship today\". Below 1.0 means it helped.\n\n" ++
            "α interpolates between the two mechanisms: the residual importance handed to the search is\n" ++
            "`w^(1−α)`, so α = 0 is pure §8A and α = 1 leaves the search nothing to steer.\n\n" ++
            "> Each α here is a fixed, un-fitted choice, so its ratio is an honest out-of-sample number.\n" ++
            "> **Reading a per-layer best α off this table would not be** — that is selection on the same\n" ++
            "> activations the error is measured on, and it needs a held-out split of the row sample\n" ++
            "> before anyone trusts it. One α per format is what these numbers support.\n\n",
    );
    try w.writeAll("| format | base | α | layers | base rel-L2 | equalized rel-L2 | ratio | layers improved |\n");
    try w.writeAll("|---|---|---:|---:|---:|---:|---:|---:|\n");

    for (ph.formats) |spec| {
        for (report.eq_alphas) |alpha| {
            var n: usize = 0;
            var sum_b: f64 = 0;
            var sum_e: f64 = 0;
            var better: usize = 0;
            var on_weighted: usize = 0;
            for (report.layers) |l| {
                const f = l.find(spec.fmt) orelse continue;
                for (f.eq_arms) |e| {
                    if (e.alpha != alpha) continue;
                    const base = f.baseArm(e.on_weighted);
                    n += 1;
                    if (e.on_weighted) on_weighted += 1;
                    sum_b += base.rel_l2;
                    sum_e += e.metrics.rel_l2;
                    if (e.metrics.rel_l2 < base.rel_l2) better += 1;
                }
            }
            if (n == 0) continue;
            const fn_: f64 = @floatFromInt(n);
            const mb = sum_b / fn_;
            const me = sum_e / fn_;
            // Every ratio is paired against that layer's own base, so a format
            // whose weighted arm was unavailable on a few shapes is still compared
            // correctly — but the column has to say so, or the two means look like
            // they were taken over one baseline when they were not.
            var base_buf: [24]u8 = undefined;
            const base_label = if (on_weighted == n)
                "weighted"
            else if (on_weighted == 0)
                "format"
            else
                try std.fmt.bufPrint(&base_buf, "mixed {d}/{d} w", .{ on_weighted, n });
            try w.print("| {s} | {s} | {d:.2} | {d} | {e:.4} | {e:.4} | {d:.4} | {d}/{d} |\n", .{
                spec.name,  base_label,
                alpha,      n,
                mb,         me,
                if (mb > 0) me / mb else 1.0,
                better,     n,
            });
        }
    }

    // Where the gain can actually be collected. This is the half of §8B that is a
    // fact about the architecture rather than about the numbers above, and it is
    // the reason the section exists separately from §8A's.
    try w.writeAll("\n### Where the fold can go\n\n");
    if (!report.fold_known) {
        try w.print(
            "`ImageArch` has no foldability relation for arch `{s}`, so every layer is counted as\n" ++
                "unfoldable. That is a missing analysis, not a measurement: the ratios above are the\n" ++
                "headroom, and nothing says yet how much of it is reachable.\n",
            .{report.arch},
        );
        return;
    }
    const total = report.fold_exact + report.fold_runtime_shift + report.fold_none;
    try w.print(
        "- **{d}/{d} exact** — a static per-channel producer feeds the layer, so `1/s` folds into it\n" ++
            "  with no runtime change at all\n" ++
            "- **{d}/{d} need a runtime change** — a producer exists but the activations are additively\n" ++
            "  modulated after it (AdaLN) by a vector no per-layer static tensor carries, so\n" ++
            "  `W·diag(s)` would also scale the shift. Exactness needs the runtime to divide the\n" ++
            "  modulated activations: one per-channel multiply, plus a vector per site in the file\n" ++
            "- **{d}/{d} have no producer** — the input is a computed activation (attention output,\n" ++
            "  SwiGLU product, raw patches), where equalization means a format *and* kernel change\n",
        .{ report.fold_exact, total, report.fold_runtime_shift, total, report.fold_none, total },
    );
    try w.writeAll(
        "\n> A ratio above is the *headroom*, measured with the exact transform. Only the `exact` rows\n" ++
            "> of that tally are collectable as things stand; the `runtime_shift` ones are a bet on a\n" ++
            "> TensorPencil change, and the `none` ones are out of scope for §8B by construction.\n",
    );
}

/// §8C: does keeping the Gram's off-diagonal terms buy anything the diagonal
/// (§8A) does not — and does it survive being scored on rows it never saw?
///
/// Two ratios per format, and the second one is not decoration. `held-out` is the
/// claim; `in-sample` is the same comparison on the rows the Hessian was fitted to,
/// which is an upper bound on what any amount of calibration could deliver. The
/// spread between them prices the row sample: wide means the compensation is
/// fitting this sample, narrow means the covariance it found is a property of the
/// layer.
fn writeGptqSection(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("\n## GPTQ error compensation (§8C), averaged over layers\n\n");
    try w.print(
        "Same grid as the format's **shipped** configuration (the §8A weighted search where\n" ++
            "`convert --calib` applies one, the plain quantizer otherwise) — only the choice of level\n" ++
            "changes, made by compensating each column's rounding error into the columns after it\n" ++
            "against the full Gram `XᵗX` rather than its diagonal. Ridge λ = {d:.3}·mean(diag).\n\n",
        .{report.gptq_damp},
    );
    // Which generalization question this run answers. These are different claims and
    // the table's column headings are identical in both cases, so the distinction has
    // to be stated here or a reader will carry the stronger one away from the weaker run.
    if (report.gptq_eval_calib.len > 0) {
        try w.print(
            "**Scored on a disjoint prompt set**: the Hessian is fitted on every row of the primary\n" ++
                "cache and both arms are then measured on `{s}` (prompt set `{s}` vs `{s}`), captured\n" ++
                "from the same checkpoint. This is the question that decides shippability — a row split\n" ++
                "can only say whether the compensation transfers to other patches of the *same* images.\n" ++
                "Note the Hessian here gets *more* training rows than a row split would give it, so a\n" ++
                "worse number than the split's is attributable to the prompts and not to less data.\n" ++
                "Ratio below 1.0 means it helped.\n\n",
            .{ report.gptq_eval_calib, report.prompt_set, report.gptq_eval_prompt_set },
        );
    } else {
        try w.print(
            "**Both arms are re-measured on a held-out row split**, 1 row in {d} withheld from the\n" ++
                "Hessian, because this is the only arm here whose quantization is derived from the\n" ++
                "activations it is then scored on. Ratio below 1.0 means it helped.\n\n" ++
                "> These rows are held out but **not independent**: both halves are token rows from the\n" ++
                "> same prompts and the same forward passes, i.e. neighbouring patches of the same\n" ++
                "> images. Treat the held-out ratio as an upper bound and use `--calib-eval` with a\n" ++
                "> disjoint-prompt capture for the number that decides anything.\n\n",
            .{report.gptq_holdout},
        );
    }
    try w.writeAll("| format | mechanism | grid | layers | tokens fit/held | base rel-L2 | gptq rel-L2 | held-out ratio | improved | in-sample ratio |\n");
    try w.writeAll("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n");

    for (ph.formats) |spec| {
        var n: usize = 0;
        var sum_b: f64 = 0;
        var sum_g: f64 = 0;
        var sum_bi: f64 = 0;
        var sum_gi: f64 = 0;
        var better: usize = 0;
        var on_weighted: usize = 0;
        var fit: usize = 0;
        var held: usize = 0;
        var block: usize = 0;
        for (report.layers) |l| {
            const f = l.find(spec.fmt) orelse continue;
            const g = f.gptq_arm orelse continue;
            n += 1;
            if (g.on_weighted) on_weighted += 1;
            block = g.block;
            fit += g.train_tokens;
            held += g.eval_tokens;
            sum_b += g.held_out_base.rel_l2;
            sum_g += g.held_out.rel_l2;
            sum_bi += g.in_sample_base.rel_l2;
            sum_gi += g.in_sample.rel_l2;
            if (g.held_out.rel_l2 < g.held_out_base.rel_l2) better += 1;
        }
        if (n == 0) continue;
        const fn_: f64 = @floatFromInt(n);
        const mb = sum_b / fn_;
        const mg = sum_g / fn_;
        const mbi = sum_bi / fn_;
        const mgi = sum_gi / fn_;
        const grid = if (on_weighted == n)
            "weighted"
        else if (on_weighted == 0)
            "plain"
        else
            "mixed";
        var mech_buf: [24]u8 = undefined;
        const mech = if (block == 0)
            "per column"
        else
            try std.fmt.bufPrint(&mech_buf, "ggml, blk {d}", .{block});
        try w.print("| {s} | {s} | {s} | {d} | {d}/{d} | {e:.4} | {e:.4} | {d:.4} | {d}/{d} | {d:.4} |\n", .{
            spec.name,           mech,
            grid,                n,
            fit / n,             held / n,
            mb,                  mg,
            if (mb > 0) mg / mb else 1.0,
            better,              n,
            if (mbi > 0) mgi / mbi else 1.0,
        });
    }

    try w.print(
        "\n> **The two mechanisms are not equally powerful, and the `mechanism` column says which ran.**\n" ++
            "> For the int formats GPTQ owns the rounding and compensates after *every column*. For the\n" ++
            "> k-quants it cannot — their grid is chosen inside ggml's own search — so it instead hands\n" ++
            "> ggml one block of already-compensated weights at a time and compensates on the error that\n" ++
            "> comes back. ggml keeps the encoder; GPTQ only chooses what it sees. The cost is that\n" ++
            "> correlation *within* a block is unreachable: 24 compensation steps across a 6144-column\n" ++
            "> layer instead of 6144. Expect the k-quant rows to gain less, structurally.\n" ++
            "\n> **Q4_0/Q5_0/Q4_1/Q5_1 are absent although they take an imatrix.** Their weighted encoder\n" ++
            "> normalizes `sigma2` over the whole row rather than per block, so block-at-a-time would\n" ++
            "> change the encoder as well as the values and the ratio would stop isolating the\n" ++
            "> compensation. Pinned by `Gptq.ggmlBlockLocalWeighted`.\n" ++
            "\n> **The ConvRot rows are compensated in the rotated basis**, where the rounding actually\n" ++
            "> happens: the sampled activations are rotated and the Gram follows as `RGR`. This is the\n" ++
            "> one place where §8C is strictly better informed than §8A rather than merely better\n" ++
            "> informed on paper — a *diagonal* importance collapses under the Hadamard to a single\n" ++
            "> value per group of {d} columns, while the Gram keeps every cross term.\n",
        .{report.convrot_group},
    );
    try w.writeAll(
        "\n> **Reading the two ratios together.** The in-sample column is what the compensation could\n" ++
            "> do if the calibration sample were the whole distribution, so it is a ceiling and not a\n" ++
            "> result. A held-out ratio that lands close to it means the sample is large enough to\n" ++
            "> estimate the covariance that matters; one that sits far above it means the Hessian is\n" ++
            "> being fitted to noise, and the fix is more rows (`calibrate --rows`) or more prompts,\n" ++
            "> not more damping.\n",
    );
}

/// One layer present in both the measured report and a hand-authored file.
const DiffRow = struct { name: []const u8, measured: f64, heur: f64 };

/// Compare the measured scores against a hand-authored sensitivities JSON — the
/// first real check on scores nobody ever measured (§7 output 3).
pub fn writeHeuristicDiff(
    report: *const Report,
    heuristic: *const std.json.Value,
    w: *std.Io.Writer,
    top_n: usize,
) !void {
    if (heuristic.* != .object) return error.InvalidSensitivities;
    const obj = heuristic.object;

    const gpa = report.arena.child_allocator;
    var rows: std.ArrayList(DiffRow) = .empty;
    defer rows.deinit(gpa);

    var matched: usize = 0;
    for (report.layers) |l| {
        const score = l.score orelse continue;
        const v = obj.get(l.name) orelse continue;
        const h: f64 = switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => continue,
        };
        matched += 1;
        try rows.append(gpa, .{ .name = l.name, .measured = score, .heur = h });
    }

    try w.print("\n## Measured vs hand-authored\n\n", .{});
    try w.print("- layers in both: {d} of {d} measured, {d} in the heuristic file\n", .{ matched, report.layers.len, obj.count() });
    if (matched < 2) {
        try w.writeAll("\nToo few layers in common to compare.\n");
        return;
    }

    // Spearman: both sides are already 1–100 relative scales, so rank
    // correlation is the honest comparison — an absolute difference between two
    // arbitrarily-calibrated scales would mean nothing.
    const rho = spearman(gpa, rows.items) catch std.math.nan(f64);
    try w.print("- Spearman rank correlation: {d:.4}\n", .{rho});

    const Ctx = struct {
        fn lessThan(_: void, a: DiffRow, b: DiffRow) bool {
            return @abs(a.measured - a.heur) > @abs(b.measured - b.heur);
        }
    };
    std.mem.sort(DiffRow, rows.items, {}, Ctx.lessThan);

    try w.writeAll("\n| layer | measured | hand-authored | delta |\n|---|---:|---:|---:|\n");
    for (rows.items[0..@min(top_n, rows.items.len)]) |r| {
        const d = r.measured - r.heur;
        try w.print("| `{s}` | {d:.1} | {d:.1} | {s}{d:.1} |\n", .{ r.name, r.measured, r.heur, if (d >= 0) "+" else "-", @abs(d) });
    }
}

fn spearman(gpa: std.mem.Allocator, rows: []const DiffRow) !f64 {
    const n = rows.len;
    const ra = try gpa.alloc(f64, n);
    defer gpa.free(ra);
    const rb = try gpa.alloc(f64, n);
    defer gpa.free(rb);

    try rankOf(gpa, rows, ra, true);
    try rankOf(gpa, rows, rb, false);

    var mean: f64 = 0;
    for (ra) |v| mean += v;
    mean /= @floatFromInt(n);
    var num: f64 = 0;
    var da: f64 = 0;
    var db: f64 = 0;
    for (ra, rb) |a, b| {
        num += (a - mean) * (b - mean);
        da += (a - mean) * (a - mean);
        db += (b - mean) * (b - mean);
    }
    if (da == 0 or db == 0) return std.math.nan(f64);
    return num / (@sqrt(da) * @sqrt(db));
}

/// Mid-ranks (ties share the average rank), which is what makes Spearman
/// well-defined on scores that repeat.
fn rankOf(gpa: std.mem.Allocator, rows: []const DiffRow, out: []f64, measured: bool) !void {
    const n = rows.len;
    const order = try gpa.alloc(usize, n);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;

    const Ctx = struct {
        rows: []const DiffRow,
        measured: bool,
        fn val(self: @This(), i: usize) f64 {
            return if (self.measured) self.rows[i].measured else self.rows[i].heur;
        }
        fn lessThan(self: @This(), a: usize, b: usize) bool {
            return self.val(a) < self.val(b);
        }
    };
    const ctx = Ctx{ .rows = rows, .measured = measured };
    std.mem.sort(usize, order, ctx, Ctx.lessThan);

    var i: usize = 0;
    while (i < n) {
        var j = i + 1;
        while (j < n and ctx.val(order[j]) == ctx.val(order[i])) j += 1;
        const mid = (@as(f64, @floatFromInt(i)) + @as(f64, @floatFromInt(j - 1))) / 2 + 1;
        for (order[i..j]) |idx| out[idx] = mid;
        i = j;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "arm metrics are exact on a known perturbation" {
    // 2 tokens, 3 outputs. Token 0 is scaled (direction preserved, cosine 1);
    // token 1 is rotated within the plane, so its cosine drops.
    const y = [_]f32{ 3, 4, 0, 1, 0, 0 };
    const yh = [_]f32{ 3.3, 4.4, 0, 0, 1, 0 };
    const m = armMetrics(&y, &yh, 2, 3);

    // token 0: |y|=5, |dy|=0.5 -> rel 0.1, cos 1
    // token 1: |y|=1, |dy|=sqrt(2) -> rel sqrt(2), cos 0
    try testing.expectApproxEqAbs(@as(f64, 0.5), m.mean_token_cos, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, @sqrt(2.0)), m.max_token_rel, 1e-6);
    // overall: ‖dy‖ = sqrt(0.25 + 2) , ‖y‖ = sqrt(25 + 1)
    try testing.expectApproxEqAbs(@sqrt(2.25) / @sqrt(26.0), m.rel_l2, 1e-6);
}

test "a zero reference token is excluded rather than counted as perfect" {
    // A layer with a dead token would otherwise contribute cosine 1.0 and pull
    // the mean toward "no damage".
    const y = [_]f32{ 0, 0, 1, 0 };
    const yh = [_]f32{ 0.5, 0.5, 1, 0 };
    const m = armMetrics(&y, &yh, 2, 2);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m.mean_token_cos, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m.max_token_rel, 1e-9);
}

test "scores keep magnitude, and ties share a score" {
    // Was "scores are percentile ranks", which is the encoding this replaced: it
    // spread the model evenly over the precision ladder whatever the damage
    // distribution was, and measured 2.76x off the uniform rate–distortion curve on
    // krea2. See `LadderScore` for the numbers.
    const gpa = testing.allocator;

    var f0 = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.1, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var f1 = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.5, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var f2 = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.5, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var f3 = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.9, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var layers = [_]LayerResult{
        .{ .name = "a", .rows = 1, .cols = 1, .tokens = 1, .per_format = &f0 },
        .{ .name = "b", .rows = 1, .cols = 1, .tokens = 1, .per_format = &f1 },
        .{ .name = "c", .rows = 1, .cols = 1, .tokens = 1, .per_format = &f2 },
        .{ .name = "d", .rows = 1, .cols = 1, .tokens = 1, .per_format = &f3 },
    };
    const scored = (try scoreLayers(gpa, &layers, .q4_k, null)).scored;

    try testing.expectEqual(@as(usize, 4), scored);
    // The median of {0.1, 0.5, 0.5, 0.9} is 0.5, and the median layer keeps the
    // target type — so both tied layers score 1, as does the one below them.
    try testing.expectApproxEqAbs(@as(f64, 1), layers[0].score.?, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1), layers[1].score.?, 1e-9);
    // Ties share a score by construction: the score is a function of the error
    // alone, so routing cannot depend on sort order.
    try testing.expectApproxEqAbs(layers[1].score.?, layers[2].score.?, 1e-12);
    // The worst layer is 0.848 doublings above the median, i.e. a fifth of the way
    // up the ladder — NOT at the top. Being the worst of a homogeneous population
    // buys nothing; that is the whole difference from a rank encoding.
    try testing.expectApproxEqAbs(@as(f64, 1 + 99 * @log2(1.8) / 4.0), layers[3].score.?, 1e-6);
    try testing.expect(layers[3].score.? < LadderScore.upgradeThreshold(50, 5).?);

    // And the magnitude property, which ranking did not have: multiplying every
    // error by a constant must not change a single score.
    var g0 = f0;
    var g1 = f1;
    var g2 = f2;
    var g3 = f3;
    for ([_]*[1]FormatResult{ &g0, &g1, &g2, &g3 }) |g| g[0].format_arm.rel_l2 *= 1000;
    var scaled = [_]LayerResult{
        .{ .name = "a", .rows = 1, .cols = 1, .tokens = 1, .per_format = &g0 },
        .{ .name = "b", .rows = 1, .cols = 1, .tokens = 1, .per_format = &g1 },
        .{ .name = "c", .rows = 1, .cols = 1, .tokens = 1, .per_format = &g2 },
        .{ .name = "d", .rows = 1, .cols = 1, .tokens = 1, .per_format = &g3 },
    };
    _ = try scoreLayers(gpa, &scaled, .q4_k, null);
    for (layers, scaled) |a, b| try testing.expectApproxEqAbs(a.score.?, b.score.?, 1e-9);
}

test "the scale is normalized over the layers the converter can actually route" {
    // The trap, measured: krea2's seven most damaging tensors are all `keys_hiprec`,
    // which `assignTensorType` protects *before* it ever looks at a score. The first
    // regenerated file normalized over all 263 measured tensors, which compressed the
    // 224 routable ones into scores 1..23 — all below the upgrade threshold — and the
    // "routed" model came out as uniform q4_k plus a single q5_k tensor.
    const gpa = testing.allocator;
    var hi = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 1.0, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var a = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.010, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var b = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.012, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var c = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.040, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var layers = [_]LayerResult{
        // krea2's hiprec list matches `txtfusion` — never routed, whatever its score.
        .{ .name = "model.diffusion_model.txtfusion.lw.0.attn.wo.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &hi },
        .{ .name = "model.diffusion_model.blocks.0.attn.wq.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &a },
        .{ .name = "model.diffusion_model.blocks.1.attn.wq.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &b },
        .{ .name = "model.diffusion_model.blocks.2.attn.wq.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &c },
    };

    const thr = LadderScore.upgradeThreshold(50, 5).?;

    _ = try scoreLayers(gpa, &layers, .q4_k, imagearch.byName("krea2").?);
    // The routable population is {0.010, 0.012, 0.040}, median 0.012 — so the 0.040
    // layer is 1.74 doublings above typical and earns an upgrade, while the hiprec
    // outlier saturates at 100 without ever having set the scale.
    try testing.expect(layers[3].score.? > thr);
    try testing.expectApproxEqAbs(@as(f64, 100), layers[0].score.?, 1e-9);

    // With the arch unknown there is no split to make, so the hiprec layer joins the
    // population and drags the median up to 0.026 — and the one routable layer that
    // deserved protection no longer gets it. That is exactly the trap, as a number.
    _ = try scoreLayers(gpa, &layers, .q4_k, null);
    try testing.expect(layers[3].score.? < thr);
}

test "a layer with no reference-format measurement is unscored, not ranked lowest" {
    // `first.weight` is 6144×64 and Q4_K needs a multiple of 256 columns, so this
    // is the real case, not a hypothetical. Scoring it 1 ("least sensitive")
    // would route the model's input projection to the most aggressive format
    // available — the exact opposite of correct.
    const gpa = testing.allocator;
    var with = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.2, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var other = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.4, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    // Measured at q8_0 only — nothing at the reference format.
    var without = [_]FormatResult{.{ .fmt = .q8_0, .bits = 8.5, .format_arm = .{ .rel_l2 = 0.01, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};

    var layers = [_]LayerResult{
        .{ .name = "a", .rows = 1, .cols = 1, .tokens = 1, .per_format = &with },
        .{ .name = "first.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &without },
        .{ .name = "c", .rows = 1, .cols = 1, .tokens = 1, .per_format = &other },
    };
    const scored = (try scoreLayers(gpa, &layers, .q4_k, null)).scored;

    try testing.expectEqual(@as(usize, 2), scored);
    try testing.expect(layers[1].score == null);
    // With two values the median is their mean, so the lower one sits at the bottom
    // and the higher one climbs by however many doublings it actually is above it —
    // here 0.4 against a 0.3 median, which is a third of a doubling and nothing like
    // the top of the ladder.
    try testing.expectApproxEqAbs(@as(f64, 1), layers[0].score.?, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1 + 99 * @log2(4.0 / 3.0) / 4.0), layers[2].score.?, 1e-6);

    // And the unscored layer must not appear in the JSON at all: absent means
    // "converter, use your own rule", which is the correct fallback.
    var report: Report = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .layers = &layers,
        .reference_format = .q4_k,
        .model_path = "m",
        .calib_path = "c",
        .arch = "krea2",
        .prompt_set = "p",
        .reference_is_quantized = false,
        .reference_dtype = "f32",
        .kernel_arms = 0,
        .weighted_arms = 0,
        .scored = scored,
    };
    defer report.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try writeSensitivitiesJson(&report, &aw.writer);
    const json = aw.written();
    try testing.expect(std.mem.indexOf(u8, json, "first.weight") == null);
    try testing.expect(std.mem.indexOf(u8, json, "\"a\"") != null);

    // It must still be valid JSON of the shape Convert.zig consumes: two scores plus
    // the reserved provenance key, which is not a tensor name and so can never
    // collide with the converter's lookup.
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 3), parsed.value.object.count());
    const meta = parsed.value.object.get(LadderScore.meta_key).?.object;
    try testing.expectEqualStrings(LadderScore.encoding_id, meta.get("encoding").?.string);
    try testing.expectEqualStrings("level1-output-error", meta.get("generator").?.string);
    try testing.expectEqualStrings("Q4_K", meta.get("reference_format").?.string);
    try testing.expect(meta.get("complete").?.bool);
    // A file this build understands must raise no warning; that is the whole point of
    // recording the encoding.
    try testing.expect(LadderScore.metaWarning(parsed.value) == null);
}

test "an incomplete sweep's file is marked, and the converter warns about it" {
    // ⚠️ The anchor is the measured population's MEDIAN, so `--max-layers` does not
    // write a partial file — it writes a rescaled one, with every score shifted. That
    // is strictly worse than the percentile encoding it replaced, where a partial run
    // merely gave a sparse ranking, so it has to be visible on the artifact itself.
    const gpa = testing.allocator;
    var f = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.2, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var layers = [_]LayerResult{.{ .name = "a", .rows = 1, .cols = 1, .tokens = 1, .per_format = &f, .score = 1 }};
    var report: Report = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .layers = &layers,
        .reference_format = .q4_k,
        .model_path = "m",
        .calib_path = "c",
        .arch = "krea2",
        .prompt_set = "p",
        .reference_is_quantized = false,
        .reference_dtype = "f32",
        .kernel_arms = 0,
        .weighted_arms = 0,
        .scored = 1,
        .anchor_population = 1,
        .complete = false,
    };
    defer report.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try writeSensitivitiesJson(&report, &aw.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, aw.written(), .{});
    defer parsed.deinit();
    try testing.expect(!parsed.value.object.get(LadderScore.meta_key).?.object.get("complete").?.bool);
    // And it must be the *first* thing the converter complains about, ahead of any
    // encoding check, because a rescaled ladder is the more serious defect.
    const why = LadderScore.metaWarning(parsed.value).?;
    try testing.expect(std.mem.indexOf(u8, why, "truncated") != null);
}

test "scores are emitted under the canonical unprefixed tensor name" {
    // The probe tags weights with the checkpoint's own names, which for the krea2
    // DiT are container-prefixed (`model.diffusion_model.blocks.0...`). The
    // converter strips that prefix before looking a tensor up, so emitting the
    // prefixed form would produce a file matching nothing — silently, since a
    // miss is only a warning.
    const gpa = testing.allocator;
    var f = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.2, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var g = [_]FormatResult{.{ .fmt = .q4_k, .bits = 4.5, .format_arm = .{ .rel_l2 = 0.4, .mean_token_cos = 1, .max_token_rel = 0 }, .kernel_arm = null }};
    var layers = [_]LayerResult{
        .{ .name = "model.diffusion_model.blocks.0.attn.wq.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &f },
        .{ .name = "model.diffusion_model.txtfusion.projector.weight", .rows = 1, .cols = 1, .tokens = 1, .per_format = &g },
    };
    const scored = (try scoreLayers(gpa, &layers, .q4_k, null)).scored;

    var report: Report = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .layers = &layers,
        .reference_format = .q4_k,
        .model_path = "m",
        .calib_path = "c",
        .arch = "krea2",
        .prompt_set = "p",
        .reference_is_quantized = false,
        .reference_dtype = "bf16",
        .kernel_arms = 0,
        .weighted_arms = 0,
        .scored = scored,
    };
    defer report.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try writeSensitivitiesJson(&report, &aw.writer);

    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, aw.written(), .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    // Two scores plus the reserved provenance key.
    try testing.expectEqual(@as(usize, 3), obj.count());
    try testing.expect(obj.get("blocks.0.attn.wq.weight") != null);
    try testing.expect(obj.get("txtfusion.projector.weight") != null);
    // The prefixed form must NOT be what got written, or the file only ever
    // matches the packaging it was measured on.
    try testing.expect(obj.get("model.diffusion_model.blocks.0.attn.wq.weight") == null);
}

test "the format list parses names and tags, and rejects nonsense" {
    const gpa = testing.allocator;
    const got = try parseFormats(gpa, "q4_k, NVFP4 ,int4_convrot");
    defer gpa.free(got);
    try testing.expectEqualSlices(Format, &.{ .q4_k, .nvfp4, .int4_convrot }, got);

    // The report's display names work too, since that is what a user copies.
    const by_name = try parseFormats(gpa, "Q4_K,INT4_CR");
    defer gpa.free(by_name);
    try testing.expectEqualSlices(Format, &.{ .q4_k, .int4_convrot }, by_name);

    try testing.expectError(error.UnknownFormat, parseFormats(gpa, "q4_k,nope"));
    try testing.expectError(error.NoFormats, parseFormats(gpa, " , "));
}

test "the kernel-arm token threshold still matches TensorPencil's dispatch" {
    // The threshold itself now comes from TensorPencil, so it cannot drift. What
    // this test pins is the *meaning* we attach to it: that at `small_m_max`
    // tokens the CPU block-quant GEMM is dequant-then-f32-GEMM, and therefore
    // bit-identical to the format arm.
    //
    // If TP ever gives block-quant a genuinely fused kernel, or moves activation
    // quantization above this line, the two arms stop agreeing — and this fails,
    // which is right: the harness would then be reporting an activation-quantized
    // result in a column labelled "weight format loss".
    const gpa = testing.allocator;
    const io = testing.io;

    const rows = 4;
    const cols = 256; // one q4_k super-block
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    var prng = std.Random.DefaultPrng.init(0xA5A5_1234);
    const rnd = prng.random();
    for (w) |*v| v.* = rnd.floatNorm(f32);

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();

    const bytes = try DataTransform.Quantizer.convertTensorData(
        gpa,
        std.mem.sliceAsBytes(w),
        .f32,
        .q4_k,
        rows * cols,
        &pool,
    );
    defer gpa.free(bytes);
    const w_hat = try ph.roundtrip(.q4_k, gpa, w, rows, cols, &pool);
    defer gpa.free(w_hat);

    const m = tp_small_m_max;
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (x) |*v| v.* = rnd.floatNorm(f32);

    const y_kernel = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_kernel);
    const y_format = try gpa.alloc(f32, m * rows);
    defer gpa.free(y_format);

    try tp.ops.matmul.matmul(io, gpa, y_kernel, x, m, tp.ops.matmul.Weight.init(bytes, .q4_k, rows, cols), null);
    try tp.ops.matmul.matmul(io, gpa, y_format, x, m, tp.ops.matmul.Weight.fromF32(w_hat, rows, cols), null);

    if (!std.mem.eql(f32, y_kernel, y_format)) {
        std.debug.print(
            "kernel and format arms differ at m={d}: TensorPencil's block-quant dispatch threshold " ++
                "has moved, or its CPU block-quant GEMM is no longer dequant+GEMM. Re-check " ++
                "`small_m_max` in tp_ops/matmul.zig against `tp_small_m_max` here.\n  y_kernel[0..4]={any}\n  y_format[0..4]={any}\n",
            .{ m, y_kernel[0..@min(4, y_kernel.len)], y_format[0..@min(4, y_format.len)] },
        );
        return error.TestExpectedEqual;
    }
}

test "the kernel arm covers exactly the formats TensorPencil can execute" {
    // If this drifts, either a kernel landed (good — extend the map) or a format
    // is claiming a kernel arm it cannot run.
    for (ph.formats) |spec| {
        const dt = kernelDataType(spec.fmt) orelse continue;
        try testing.expect(toTp(dt) != null);
    }
    try testing.expect(kernelDataType(.q4_k) != null);
    try testing.expect(kernelDataType(.nvfp4) == null); // cluster format, no kernel anywhere
    try testing.expect(kernelDataType(.q2_k) == null); // ggufy emits it; nothing runs it
}

test "the GPTQ split is disjoint, exhaustive and evenly spread" {
    // The one property that makes the §8C numbers mean anything: no row is in both
    // halves. Checked by content rather than by index arithmetic, since a bug in
    // the copy would leave the counts right and the data wrong.
    const gpa = testing.allocator;
    const io = testing.io;
    const m = 12;
    const cols = 4;
    const rows = 2;
    const holdout = 3;

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    // Row i is the constant i, so a row's identity survives the copy.
    for (0..m) |i| {
        for (0..cols) |c| x[i * cols + c] = @floatFromInt(i);
    }
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    @memset(w, 0.25);

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();

    var ctx = try GptqCtx.initSplit(gpa, io, x, m, rows, cols, w, holdout, Gptq.default_damp, null, &pool);
    defer ctx.deinit();

    try testing.expectEqual(@as(usize, 4), ctx.eval_tokens);
    try testing.expectEqual(@as(usize, 8), ctx.train_tokens);

    var seen = [_]u8{0} ** m;
    for (0..ctx.train_tokens) |i| seen[@intFromFloat(ctx.x_train[i * cols])] += 1;
    for (0..ctx.eval_tokens) |i| seen[@intFromFloat(ctx.x_eval[i * cols])] += 2;
    // 1 = train only, 2 = eval only; 3 would be both and 0 neither.
    for (seen, 0..) |s, i| try testing.expectEqual(@as(u8, if (i % holdout == holdout - 1) 2 else 1), s);

    // Stride, not shuffle: the evaluation rows are spread across the sample, so a
    // sample that concatenates schedule buckets in order splits proportionally.
    for (0..ctx.eval_tokens) |i| {
        const idx: usize = @intFromFloat(ctx.x_eval[i * cols]);
        try testing.expectEqual(i * holdout + holdout - 1, idx);
    }
}

test "a sample too thin to split gets no GPTQ arm rather than an unsplit one" {
    const gpa = testing.allocator;
    const io = testing.io;
    const cols = 4;
    const x = try gpa.alloc(f32, 2 * cols);
    defer gpa.free(x);
    @memset(x, 1);
    const w = try gpa.alloc(f32, cols);
    defer gpa.free(w);
    @memset(w, 0.25);

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();

    // 2 rows at 1-in-3 leaves nothing held out.
    try testing.expectError(
        error.SampleTooSmall,
        GptqCtx.initSplit(gpa, io, x, 2, 1, cols, w, 3, Gptq.default_damp, null, &pool),
    );
    // ...and a holdout of 1 would hold out everything, which is not a split.
    try testing.expectError(
        error.InvalidHoldout,
        GptqCtx.initSplit(gpa, io, x, 2, 1, cols, w, 1, Gptq.default_damp, null, &pool),
    );
}

test "a second cache supersedes the split and keeps every training row" {
    // The `--calib-eval` regime answers a harder question than the row split, and
    // the argument that it answers it *cleanly* rests on the Hessian getting more
    // data here, not less — so a worse ratio cannot be blamed on a thinner fit.
    // This pins that: all of the primary rows train, all of the eval cache scores,
    // and the two blocks stay distinct.
    const gpa = testing.allocator;
    const io = testing.io;
    const cols = 4;
    const rows = 2;

    const xa = try gpa.alloc(f32, 6 * cols);
    defer gpa.free(xa);
    for (0..6) |i| {
        for (0..cols) |c| xa[i * cols + c] = @floatFromInt(i + 1);
    }
    const xb = try gpa.alloc(f32, 3 * cols);
    defer gpa.free(xb);
    for (0..3) |i| {
        for (0..cols) |c| xb[i * cols + c] = @floatFromInt(100 + i);
    }
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    @memset(w, 0.25);

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();

    var ctx = try GptqCtx.initPair(gpa, io, xa, xb, rows, cols, w, Gptq.default_damp, null, &pool);
    defer ctx.deinit();

    try testing.expectEqual(@as(usize, 6), ctx.train_tokens);
    try testing.expectEqual(@as(usize, 3), ctx.eval_tokens);
    try testing.expectEqualSlices(f32, xa, ctx.x_train);
    try testing.expectEqualSlices(f32, xb, ctx.x_eval);

    // A split of the same primary cache would have trained on fewer rows — the
    // asymmetry the conclusion depends on.
    var split = try GptqCtx.initSplit(gpa, io, xa, 6, rows, cols, w, 3, Gptq.default_damp, null, &pool);
    defer split.deinit();
    try testing.expect(split.train_tokens < ctx.train_tokens);
}

test "capping the training rows subsamples evenly and leaves the evaluation alone" {
    // The row-count sweep is only interpretable if each point differs from the next
    // in exactly one way. Two things have to hold: the cap must spread across the
    // block rather than take a prefix (the block concatenates schedule buckets in
    // order, so a prefix would quietly fit on early-sigma activations only), and it
    // must not touch the evaluation rows.
    const gpa = testing.allocator;
    const io = testing.io;
    const cols = 4;
    const rows = 2;

    const xa = try gpa.alloc(f32, 12 * cols);
    defer gpa.free(xa);
    for (0..12) |i| {
        for (0..cols) |c| xa[i * cols + c] = @floatFromInt(i);
    }
    const xb = try gpa.alloc(f32, 5 * cols);
    defer gpa.free(xb);
    @memset(xb, 7);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    @memset(w, 0.25);

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();

    var ctx = try GptqCtx.initPair(gpa, io, xa, xb, rows, cols, w, Gptq.default_damp, 4, &pool);
    defer ctx.deinit();

    try testing.expectEqual(@as(usize, 4), ctx.train_tokens);
    try testing.expectEqual(@as(usize, 5), ctx.eval_tokens);
    // i·12/4 → rows 0, 3, 6, 9: spread over the whole block, not the first four.
    for ([_]f32{ 0, 3, 6, 9 }, 0..) |want, i| try testing.expectEqual(want, ctx.x_train[i * cols]);
    try testing.expectEqualSlices(f32, xb, ctx.x_eval);

    // A cap at or above what the cache holds is a no-op, not a resample.
    var full = try GptqCtx.initPair(gpa, io, xa, xb, rows, cols, w, Gptq.default_damp, 99, &pool);
    defer full.deinit();
    try testing.expectEqualSlices(f32, xa, full.x_train);
}

test "every arm that consumes the imatrix keeps it alive" {
    // `imat` is built from one condition shared by three consumers, and §8C's
    // dependence on it is the non-obvious one: it does not merely *measure* the
    // weighted arm, it takes the grid both of its own arms stand on. If this
    // condition ever drops `gptq`, `--gptq --no-weighted-arm` silently compares
    // against the plain quantizer and reports a ratio against a baseline the
    // converter never ships — no crash, no warning, just a wrong number.
    //
    // Expressed as the predicate itself rather than through a run, since the
    // failure is a missing term and not a behaviour a small fixture would expose.
    const needs = struct {
        fn imatrix(o: Options) bool {
            return o.weighted_arm or o.eq_alphas.len > 0 or o.gptq;
        }
    }.imatrix;
    const base: Options = .{ .model_path = "m", .calib_path = "c", .weighted_arm = false };

    try testing.expect(!needs(base));
    try testing.expect(needs(.{ .model_path = "m", .calib_path = "c" })); // §8A on by default
    var eq = base;
    eq.eq_alphas = &.{0.25};
    try testing.expect(needs(eq)); // §8B
    var gq = base;
    gq.gptq = true;
    try testing.expect(needs(gq)); // §8C — the one that was missing
}

test "the GPTQ arm covers exactly the formats whose grid it can own" {
    // §8C needs one symmetric per-row scale to round against. If this drifts,
    // either a format gained such a grid or something is claiming an arm whose
    // rounding it does not control — and a k-quant appearing here would mean the
    // arm is silently rounding against a grid it reconstructed rather than the one
    // ggml chose.
    const group = default_convrot_group;
    for ([_]Format{ .int4, .int8, .int4_convrot, .int8_convrot }) |fmt| {
        const spec = gptqSpec(fmt, group) orelse return error.TestUnexpectedResult;
        // The rotated formats must be compensated in the rotated basis, at the same
        // group size the weighted arm uses, or the two are not comparable.
        const rotated = fmt == .int4_convrot or fmt == .int8_convrot;
        try testing.expectEqual(rotated, spec.grid.basis.convrot);
        if (rotated) try testing.expectEqual(group, spec.grid.basis.group_size);
    }
    // The GGUF path: ggml owns the encoder, so these go through the block mechanism.
    for ([_]Format{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k }) |fmt| {
        const spec = gptqSpec(fmt, group) orelse return error.TestUnexpectedResult;
        try testing.expect(spec == .ggml);
        const gt = try gguf.GgmlType.fromString(@tagName(spec.ggml));
        // Only safe because their weighted encoder is block-local — the guard that
        // keeps this arm measuring compensation rather than an encoder change.
        try testing.expect(Gptq.ggmlBlockLocalWeighted(gt));
    }
    // Takes an imatrix, but its weighted encoder is row-coupled: excluded.
    try testing.expect(gptqSpec(.q4_0, group) == null);
    try testing.expect(gptqSpec(.q5_0, group) == null);
    for ([_]Format{ .q8_0, .nvfp4, .mxfp4, .mxfp8, .scaled_f8_e4m3, .f16 }) |fmt| {
        try testing.expect(gptqSpec(fmt, group) == null);
    }

    // ...and the basis follows the requested group, which is what makes measuring
    // the shipped configuration (256) possible at all.
    const shipped = gptqSpec(.int4_convrot, TensorClusters.int4_convrot_group_size).?;
    try testing.expectEqual(TensorClusters.int4_convrot_group_size, shipped.grid.basis.group_size);

    // The grids are the ComfyUI ones, restated nowhere: int4 never emits −8, int8
    // clamps asymmetrically against a 127 divisor.
    try testing.expectEqual(@as(f32, 7), Gptq.int4_grid.qdiv);
    try testing.expectEqual(@as(f32, -7), Gptq.int4_grid.qlo);
    try testing.expectEqual(@as(f32, 127), Gptq.int8_grid.qdiv);
    try testing.expectEqual(@as(f32, -128), Gptq.int8_grid.qlo);
}

test "spearman is 1 on a matching order and -1 on a reversed one" {
    const gpa = testing.allocator;
    var same = [_]DiffRow{
        .{ .name = "a", .measured = 1, .heur = 10 },
        .{ .name = "b", .measured = 2, .heur = 20 },
        .{ .name = "c", .measured = 3, .heur = 30 },
    };
    try testing.expectApproxEqAbs(@as(f64, 1), try spearman(gpa, same[0..]), 1e-9);

    var rev = [_]DiffRow{
        .{ .name = "a", .measured = 1, .heur = 30 },
        .{ .name = "b", .measured = 2, .heur = 20 },
        .{ .name = "c", .measured = 3, .heur = 10 },
    };
    try testing.expectApproxEqAbs(@as(f64, -1), try spearman(gpa, rev[0..]), 1e-9);
}
