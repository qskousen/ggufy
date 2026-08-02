//! Level 2 — one-pass ε/v divergence (`ggufy divergence`).
//!
//! The measurement ACTIVATION_AWARE_PLAN.md §7 calls level 2, and the only one
//! here that is **free of trajectory drift**.
//!
//! Level 1 asks how much one layer's output moves. Level 3 asks how different the
//! final picture looks — and that question turned out to be nearly unanswerable at
//! any affordable sample size, because quantization perturbs the *denoising
//! trajectory*: the arms draw different pictures rather than the same picture
//! worse, and every reference-matched image metric measures mostly that. Measured
//! (2026-07-31): resolving §8A at |t| = 2 needs ~109 paired renders with PSNR and
//! ~715 with LPIPS.
//!
//! Level 2 removes the trajectory from the question by **teacher-forcing**:
//!
//!   1. run the reference model's sampling loop, snapshotting `(x_i, σ_i)` and its
//!      own predicted velocity `v_i` at every step;
//!   2. for each candidate, predict at those *same* `(x_i, σ_i)` — never advancing
//!      the candidate's own latent — and compare `v'_i` against `v_i`.
//!
//! Both arms therefore see identical inputs at every point measured, and the
//! difference is attributable to the weights alone. There is no composition to
//! drift, and one prompt gives `steps` independent data points instead of one
//! image.
//!
//! It is also the cross-check on level 1 the programme has owed itself since the
//! start (open question 2): level 1's ranking is a per-layer proxy, and this is the
//! whole-model quantity it is a proxy *for*.
//!
//! ⚠️ **The conditioning is encoded once and reused across arms.** The text
//! encoder is the same checkpoint for every arm, so re-encoding per arm would
//! introduce a difference in the *inputs* — GPU reduction order alone would do it —
//! and this measurement's whole claim is that the inputs are identical. Encode
//! once, carry the `Cond` across sessions.
//!
//! **Determinism is measured, not assumed** (ACTIVATION_AWARE.md hygiene rule 2).
//! Two independent runs of this harness — separate processes, different schedules
//! either side of a fixed sigma — produced **bit-identical** figures at the shared
//! point: `rel_l2 = 0.11029768`, `cos = 0.99391809`, `max_pos_rel = 0.62827505`, all
//! eight digits. That is what licenses comparing arms measured in *different* runs,
//! which is how the safetensors and GGUF output paths get put on one scale.
//!
//! ⚠️ **Which CPU kernel the block quants take depends on the token count.**
//! `ops.matmul.small_m_max` (16) is the line: at or above it a block-quantized
//! weight is dequantized and multiplied in f32 (weight-only, W4A16), below it
//! ggml's `vec_dot` GEMV also quantizes the activations. A DiT forward at any
//! realistic resolution is hundreds of tokens, so this measures the weight-only
//! path — the same regime a user's render is in, and comparable to the int4
//! cluster formats, which dequantize in the GEMM too. Deliberately *not* pinned
//! with `exact_activations`: the point is to measure what a render actually
//! computes. A measurement at a toy resolution would silently change arms.
//!
//! ⚠️ **What this does not measure.** Per-tensor attribution (quantize exactly one
//! tensor, measure the whole-model divergence) is the arm that would rank layers
//! the way level 1 does, and it needs TensorPencil's weight-overlay store (plan
//! item 14) to swap a single tensor without rewriting a 26 GB checkpoint. This
//! module measures whole-checkpoint arms, which is what `convert` actually
//! produces.

const std = @import("std");
const tp = @import("TensorPencil");
const cb = @import("callbacks.zig");
const ph = @import("precision_harness.zig");
const ThreadPool = @import("ThreadPool.zig").ThreadPool;
const TensorClusters = @import("TensorClusters.zig");
const imagearch = @import("ImageArch.zig");
const LadderScore = @import("LadderScore.zig");

pub const Backend = tp.pipeline.Backend;
pub const Format = ph.Format;

/// How far one arm's velocity prediction sits from the reference's, at one step.
///
/// Deliberately the same three figures `Sensitivity.ArmMetrics` reports, so a
/// level-1 number and a level-2 number can be read side by side.
pub const Point = struct {
    step: usize,
    sigma: f32,
    /// Relative L2 over the whole velocity tensor: ‖v' − v‖ / ‖v‖.
    rel_l2: f64,
    /// Cosine between the two velocity tensors, flattened.
    cos: f64,
    /// Mean over latent positions of the per-position cosine (16 channels each) —
    /// a direction error that a whole-tensor cosine can average away.
    mean_pos_cos: f64,
    /// Worst per-position relative error, i.e. where the arm is most wrong.
    max_pos_rel: f64,
};

// The four aggregates, as free functions over a point set: the whole-checkpoint
// arms and the per-tensor arms both report them, and one implementation is what
// keeps the two tables literally comparable.

fn meanRelL2Of(points: []const Point) f64 {
    if (points.len == 0) return 0;
    var sum: f64 = 0;
    for (points) |p| sum += p.rel_l2;
    return sum / @as(f64, @floatFromInt(points.len));
}

fn meanCosOf(points: []const Point) f64 {
    if (points.len == 0) return 1.0;
    var sum: f64 = 0;
    for (points) |p| sum += p.cos;
    return sum / @as(f64, @floatFromInt(points.len));
}

fn worstRelL2Of(points: []const Point) f64 {
    var worst: f64 = 0;
    for (points) |p| worst = @max(worst, p.rel_l2);
    return worst;
}

fn meanRelL2AtStepOf(points: []const Point, step: usize) f64 {
    var sum: f64 = 0;
    var n: usize = 0;
    for (points) |p| if (p.step == step) {
        sum += p.rel_l2;
        n += 1;
    };
    return if (n == 0) 0 else sum / @as(f64, @floatFromInt(n));
}

/// One candidate checkpoint, measured over every (prompt, step).
pub const ArmResult = struct {
    /// Display label — the checkpoint's basename.
    name: []const u8,
    path: []const u8,
    /// `prompts × steps` points, prompt-major.
    points: []Point,
    steps: usize,

    /// Mean relative L2 over every point.
    pub fn meanRelL2(self: ArmResult) f64 {
        return meanRelL2Of(self.points);
    }

    pub fn meanCos(self: ArmResult) f64 {
        return meanCosOf(self.points);
    }

    pub fn worstRelL2(self: ArmResult) f64 {
        return worstRelL2Of(self.points);
    }

    /// Mean relative L2 at one step index, across prompts — the per-σ curve.
    pub fn meanRelL2AtStep(self: ArmResult, step: usize) f64 {
        return meanRelL2AtStepOf(self.points, step);
    }
};

pub const Report = struct {
    arena: std.heap.ArenaAllocator,
    reference: []const u8,
    prompts: []const []const u8,
    /// The reference schedule, `steps + 1` long.
    sigmas: []const f32,
    resolution: [2]usize,
    seed: u64,
    backend: []const u8,
    arms: []ArmResult,

    pub fn steps(self: *const Report) usize {
        return self.sigmas.len - 1;
    }

    pub fn deinit(self: *Report) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const Options = struct {
    io: std.Io,
    /// The full-precision reference DiT checkpoint.
    reference: []const u8,
    /// Candidate DiT checkpoints (converted models), each measured against the
    /// reference on the reference's own trajectory.
    candidates: []const []const u8,
    text_encoder_path: []const u8,
    vae_path: []const u8,
    /// Prompts to sample trajectories from. Each contributes `steps` points.
    prompts: []const []const u8,
    width: usize = 512,
    height: usize = 512,
    steps: usize = 8,
    /// Schedule shift. Defaults to TensorPencil's own, so the sigmas measured are
    /// the sigmas a render actually visits — a different shift measures a
    /// different trajectory and is not comparable to a level-3 verdict.
    shift: f32 = tp.sampler.default_shift,
    seed: u64 = 0,
    /// Classifier-free guidance. 1.0 (the default) keeps it to one forward per
    /// point; anything else doubles the cost and folds guidance into the measured
    /// velocity, which is what a user running CFG actually gets.
    cfg: f32 = 1.0,
    negative: []const u8 = "",
    backend: Backend = .cpu,
    vram_budget: u64 = 0,
    callbacks: cb.CaptureCallbacks = .{},
    /// Progress notes from TensorPencil's loader.
    log: ?*std.Io.Writer = null,
};

/// One snapshot of the reference trajectory: the sampler state a candidate is
/// asked to predict from, and what the reference predicted there.
const Snapshot = struct {
    step: usize,
    sigma: f32,
    /// The latent the reference fed to its own forward at this step.
    x: []f32,
    /// The reference's velocity at `(x, sigma)`.
    v: []f32,
};

fn basename(path: []const u8) []const u8 {
    const base = std.fs.path.basename(path);
    if (std.mem.lastIndexOfScalar(u8, base, '.')) |dot| if (dot > 0) return base[0..dot];
    return base;
}

fn baseOpts(opts: Options, dit_path: []const u8, cancel: *std.atomic.Value(bool)) tp.pipeline.Options {
    return .{
        .prompt = opts.prompts[0],
        .negative = opts.negative,
        .width = opts.width,
        .height = opts.height,
        .steps = opts.steps,
        .cfg = opts.cfg,
        .seed = opts.seed,
        .shift = opts.shift,
        .backend = opts.backend,
        .vram_budget = opts.vram_budget,
        .dit_path = dit_path,
        .text_encoder_path = opts.text_encoder_path,
        .vae_path = opts.vae_path,
        .cancel = cancel,
    };
}

/// Compare `v_apx` against `v_ref`, both `[16][lat_h][lat_w]` planar.
fn point(step: usize, sigma: f32, v_ref: []const f32, v_apx: []const f32, positions: usize, channels: usize) Point {
    var dot: f64 = 0;
    var nr: f64 = 0;
    var na: f64 = 0;
    var derr: f64 = 0;
    for (v_ref, v_apx) |a, b| {
        const fa: f64 = a;
        const fb: f64 = b;
        dot += fa * fb;
        nr += fa * fa;
        na += fb * fb;
        const d = fa - fb;
        derr += d * d;
    }

    // Per-position statistics. The latent is channel-planar, so a position's 16
    // values are strided by `positions` rather than contiguous.
    var cos_sum: f64 = 0;
    var counted: usize = 0;
    var max_rel: f64 = 0;
    for (0..positions) |p| {
        var pdot: f64 = 0;
        var pnr: f64 = 0;
        var pna: f64 = 0;
        var pderr: f64 = 0;
        for (0..channels) |c| {
            const fa: f64 = v_ref[c * positions + p];
            const fb: f64 = v_apx[c * positions + p];
            pdot += fa * fb;
            pnr += fa * fa;
            pna += fb * fb;
            const d = fa - fb;
            pderr += d * d;
        }
        // A position whose reference velocity is exactly zero has no direction and
        // no scale; counting it would be a 0/0 that reads as "perfect".
        if (pnr > 0) {
            max_rel = @max(max_rel, @sqrt(pderr / pnr));
            if (pna > 0) {
                cos_sum += pdot / (@sqrt(pnr) * @sqrt(pna));
                counted += 1;
            }
        }
    }

    return .{
        .step = step,
        .sigma = sigma,
        .rel_l2 = if (nr > 0) @sqrt(derr / nr) else 0,
        .cos = if (nr > 0 and na > 0) dot / (@sqrt(nr) * @sqrt(na)) else 1.0,
        .mean_pos_cos = if (counted > 0) cos_sum / @as(f64, @floatFromInt(counted)) else 1.0,
        .max_pos_rel = max_rel,
    };
}

/// The reference pass: what every arm is asked to reproduce.
///
/// Shared by the whole-checkpoint arms and the per-tensor arms so the two are
/// measured against a bit-identical reference — they are compared against each
/// other in the report, and a second copy of this loop is exactly how they would
/// silently stop being comparable.
const Ref = struct {
    /// One per prompt, encoded ONCE on the reference session (see the module note).
    conds: []tp.pipeline.Cond,
    neg: ?tp.pipeline.Cond,
    /// `prompts × steps`, prompt-major.
    snaps: []Snapshot,
    steps: usize,

    fn deinit(self: *Ref, gpa: std.mem.Allocator) void {
        for (self.conds) |*c| c.deinit(gpa);
        gpa.free(self.conds);
        if (self.neg) |*c| c.deinit(gpa);
        for (self.snaps) |s| {
            gpa.free(s.x);
            gpa.free(s.v);
        }
        gpa.free(self.snaps);
        self.* = undefined;
    }
};

/// Run the reference model's own sampling loop, snapshotting `(x_i, σ_i, v_i)` at
/// every step of every prompt. `sess` must be the *reference* session.
fn captureReference(
    gpa: std.mem.Allocator,
    sess: *tp.pipeline.Session,
    opts: Options,
    sigmas: []const f32,
    lat_h: usize,
    lat_w: usize,
    cancel: *std.atomic.Value(bool),
) !Ref {
    // Family-aware: krea2's Wan VAE has 16 latent channels, SD's AutoencoderKL 4.
    const lat_len = sess.latentChannels() * lat_h * lat_w;

    var ref: Ref = .{
        .conds = try gpa.alloc(tp.pipeline.Cond, opts.prompts.len),
        .neg = null,
        .snaps = try gpa.alloc(Snapshot, opts.prompts.len * opts.steps),
        .steps = opts.steps,
    };
    // Partial construction has to be unwound by hand: a Cond and a Snapshot both
    // own heap buffers, and this loop can fail (or be cancelled) halfway.
    var conds_ready: usize = 0;
    var snaps_ready: usize = 0;
    errdefer {
        for (ref.conds[0..conds_ready]) |*c| c.deinit(gpa);
        gpa.free(ref.conds);
        if (ref.neg) |*c| c.deinit(gpa);
        for (ref.snaps[0..snaps_ready]) |s| {
            gpa.free(s.x);
            gpa.free(s.v);
        }
        gpa.free(ref.snaps);
    }

    if (opts.cfg != 1.0) ref.neg = try sess.encode(gpa, opts.negative, .{});

    const x = try gpa.alloc(f32, lat_len);
    defer gpa.free(x);
    const v = try gpa.alloc(f32, lat_len);
    defer gpa.free(v);

    for (opts.prompts, 0..) |prompt, pi| {
        if (opts.callbacks.isCancelled()) {
            cancel.store(true, .release);
            return error.Canceled;
        }
        var t_enc = std.Io.Clock.real.now(opts.io).nanoseconds;
        ref.conds[pi] = try sess.encode(gpa, prompt, .{});
        conds_ready = pi + 1;
        const enc_ns = std.Io.Clock.real.now(opts.io).nanoseconds - t_enc;

        // Every prompt starts from the same seed, exactly as `generate` does, so a
        // trajectory is reproducible from the report's header alone — including the
        // scaling to the schedule's first sigma, which is 1.0 (a no-op) for krea2 and
        // ~14.6 for SD. Without it an SD arm denoises a latent 15x too small, i.e.
        // measures the model well outside the distribution it was trained on.
        //
        // ⚠️ Via the **session**, not `tp.sampler` directly: the two parameterizations
        // scale differently (krea2 by `sigma0`, SD by `sqrt(1 + sigma0²)`), and calling
        // the flow-matching form on an SD arm starts every trajectory 0.23% off — which
        // is enough to move a render visibly, so it would move a divergence figure too.
        tp.sampler.fillNoise(x, opts.seed);
        sess.scaleInitialNoise(x, sigmas[0]);

        t_enc = std.Io.Clock.real.now(opts.io).nanoseconds;
        var den = try sess.denoiser(gpa, ref.conds[pi], ref.neg, opts.cfg, lat_h, lat_w, sigmas);
        defer den.deinit(gpa);
        const den_ns = std.Io.Clock.real.now(opts.io).nanoseconds - t_enc;
        if (opts.log) |lw| {
            try lw.print("ref prompt {d}: encode {d:.1}s, denoiser {d:.1}s\n", .{
                pi, @as(f64, @floatFromInt(enc_ns)) / 1e9, @as(f64, @floatFromInt(den_ns)) / 1e9,
            });
            try lw.flush();
        }

        for (0..opts.steps) |i| {
            const t_step = std.Io.Clock.real.now(opts.io).nanoseconds;
            try den.predict(gpa, v, x, sigmas[i], cancel);
            if (opts.log) |lw| {
                try lw.print("  ref step {d}: predict {d:.1}s\n", .{
                    i, @as(f64, @floatFromInt(std.Io.Clock.real.now(opts.io).nanoseconds - t_step)) / 1e9,
                });
                try lw.flush();
            }
            // Snapshot the INPUT to this step and the velocity from it. Every arm
            // will be asked for exactly this pair.
            ref.snaps[pi * opts.steps + i] = .{
                .step = i,
                .sigma = sigmas[i],
                .x = try gpa.dupe(f32, x),
                .v = try gpa.dupe(f32, v),
            };
            snaps_ready = pi * opts.steps + i + 1;
            tp.sampler.eulerStep(x, v, sigmas[i], sigmas[i + 1]);
            opts.callbacks.reportProgress(@intCast(pi), @intCast(opts.prompts.len), @intCast(i + 1), @intCast(opts.steps));
        }
    }

    return ref;
}

/// Measure one arm: predict at every snapshot with whatever DiT `sess` currently
/// holds, and score it against the reference's velocity there.
///
/// ⚠️ **Teacher-forced**: the prediction always starts from the REFERENCE's latent,
/// never from one this arm produced. That single line is what makes the measurement
/// drift-free, and the only thing that distinguishes it from a drift curve.
fn measureArm(
    gpa: std.mem.Allocator,
    sess: *tp.pipeline.Session,
    opts: Options,
    ref: Ref,
    sigmas: []const f32,
    lat_h: usize,
    lat_w: usize,
    cancel: *std.atomic.Value(bool),
    out: []Point,
) !void {
    const channels = sess.latentChannels();
    const positions = lat_h * lat_w;

    const v_apx = try gpa.alloc(f32, channels * positions);
    defer gpa.free(v_apx);

    for (0..ref.conds.len) |pi| {
        var den = try sess.denoiser(gpa, ref.conds[pi], ref.neg, opts.cfg, lat_h, lat_w, sigmas);
        defer den.deinit(gpa);

        for (0..ref.steps) |i| {
            const s = ref.snaps[pi * ref.steps + i];
            try den.predict(gpa, v_apx, s.x, s.sigma, cancel);
            out[pi * ref.steps + i] = point(i, s.sigma, s.v, v_apx, positions, channels);
        }
    }
}

pub fn run(gpa: std.mem.Allocator, opts: Options) !Report {
    if (opts.candidates.len == 0) return error.NoCandidates;
    if (opts.prompts.len == 0) return error.NoPrompts;
    if (opts.steps < 1) return error.NoSteps;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();

    const lat_h = opts.height / 8;
    const lat_w = opts.width / 8;

    // TensorPencil polls an atomic; ggufy's callbacks are a predicate. Mirror one
    // into the other at the points a measurement can act on.
    var cancel: std.atomic.Value(bool) = .init(false);

    // --- reference trajectory -------------------------------------------------
    // The schedule comes from the *session*, because it is family-dependent: krea2's
    // flow-matching sigmas read `--shift`, SD's discrete ladder ignores it. So the
    // reference session is built first and the sigmas outlive it in the arena — every
    // arm must sample the identical ladder or the comparison means nothing.
    var ref: Ref = undefined;
    var sigmas: []const f32 = undefined;
    {
        const ref_opts = baseOpts(opts, opts.reference, &cancel);
        var sess = try tp.pipeline.Session.init(opts.io, gpa, ref_opts, opts.log);
        defer sess.deinit();
        const sched = try sess.schedule(gpa, opts.steps, opts.shift);
        defer gpa.free(sched);
        sigmas = try arena.dupe(f32, sched);
        ref = try captureReference(gpa, sess, opts, sigmas, lat_h, lat_w, &cancel);
    }
    defer ref.deinit(gpa);
    // The reference session (and its ~26 GB mapping) is gone by here; only the
    // snapshots and conditionings survive, a few MB in total.

    // --- candidate arms -------------------------------------------------------
    const arms = try arena.alloc(ArmResult, opts.candidates.len);
    for (opts.candidates, 0..) |cand_path, ai| {
        if (opts.callbacks.isCancelled()) {
            cancel.store(true, .release);
            return error.Canceled;
        }
        const cand_opts = baseOpts(opts, cand_path, &cancel);
        var cand = try tp.pipeline.Session.init(opts.io, gpa, cand_opts, opts.log);
        defer cand.deinit();

        const points = try arena.alloc(Point, ref.snaps.len);
        try measureArm(gpa, cand, opts, ref, sigmas, lat_h, lat_w, &cancel, points);

        arms[ai] = .{
            .name = try arena.dupe(u8, basename(cand_path)),
            .path = try arena.dupe(u8, cand_path),
            .points = points,
            .steps = opts.steps,
        };
    }

    const prompts = try arena.alloc([]const u8, opts.prompts.len);
    for (opts.prompts, prompts) |src, *dst| dst.* = try arena.dupe(u8, src);

    return .{
        .arena = arena_state,
        .reference = try arena.dupe(u8, opts.reference),
        .prompts = prompts,
        .sigmas = sigmas,
        .resolution = .{ opts.width, opts.height },
        .seed = opts.seed,
        .backend = @tagName(opts.backend),
        .arms = arms,
    };
}

// ---------------------------------------------------------------------------
// Per-tensor attribution
// ---------------------------------------------------------------------------
//
// Level 1 ranks layers by how much *that layer's own output* moves when it is
// quantized. That ranking is what `convert` routes precision on, and until now
// nothing had checked it against the quantity it stands in for: how much the
// *model's* prediction moves when that layer — and only that layer — is
// quantized. This measures exactly that, one tensor at a time, on the same
// teacher-forced trajectory the whole-checkpoint arms use.
//
// It is affordable because of TensorPencil's `weights.Overlay` (plan item 14): the
// reference checkpoint stays mapped and one tensor is substituted from memory, so
// a data point costs `prompts × steps` forwards instead of rewriting 26 GB.
//
// ⚠️ **The substituted tensor is a dequantized f32 round-trip, not the packed
// quantized bytes.** That isolates *format* loss from *kernel* loss (hygiene rule
// 3): the arm answers "what does this format's rounding do to the model", not
// "what does this format's GEMM do". It also changes which dequant path the layer
// takes inside the GEMM, which is why `control:exact-f32` exists — it patches the
// same tensor with its own exact values, so the two arms differ *only* by the
// rounding. Read a tensor's number against that control, not against zero.

pub const PerTensor = struct {
    /// The format each tensor is round-tripped through, one tensor at a time.
    format: Format,
    /// Checkpoint tensor names to measure. Empty derives the list from the
    /// checkpoint (every rank-2 `*.weight`), which is rarely what you want: an
    /// explicit list stratified over level 1's ranking answers the correlation
    /// question at a fraction of the cost.
    tensors: []const []const u8 = &.{},
    /// Keep at most this many of the selected tensors, sampled at an even stride so
    /// a partial run still spans the checkpoint rather than stopping at block 3.
    max_tensors: ?usize = null,
    /// ConvRot rotation group. Defaults to the size `convert` ships, not the
    /// harness's small one — this arm exists to describe files people run.
    convrot_group: usize = TensorClusters.int4_convrot_group_size,
    /// Measure the two control arms (see `TensorArm.Kind`). On by default: they
    /// cost one arm each and they are the only thing standing between a plausible
    /// table and a table that means what it says.
    controls: bool = true,
    threads: usize = 0,
    /// What dtype the substituted tensor is written back as.
    ///
    /// `.f32` is the faithful choice — a dequantized round-trip is exactly
    /// representable, so the arm measures the format's rounding and nothing else —
    /// and it is the only one the CPU path needs.
    ///
    /// ⚠️ **`.bf16` exists to make this arm runnable on a GPU.** Both GPU DiT
    /// forwards pick one weight class for a whole model (CUDA) or a whole block
    /// (Vulkan), so a single f32 tensor among bf16 is not representable there and is
    /// refused outright. Writing the patch back as bf16 keeps the model uniform, at
    /// the cost of bf16's 8 mantissa bits rounding the dequantized values (~0.2%
    /// relative) on top of the format's own error. Against a bf16 or fp8 base the
    /// *control* arms stay exact either way, since both dtypes embed losslessly in
    /// bf16 — so the penalty lands only on the quantized arms, and it is measurable:
    /// run the same tensor both ways on CPU.
    patch_dtype: PatchDtype = .f32,
    /// Measure *sets* of tensors — one arm per set, every member quantized at once —
    /// instead of one arm per tensor. Empty (the default) keeps the per-tensor mode.
    ///
    /// The question this exists for is **additivity**: a per-tensor table says what
    /// each layer costs alone, and every use of it (routing, ranking, "the error is
    /// dominated by the weak set") quietly assumes those costs compose. Quantizing a
    /// whole set at once measures the composition directly, and comparing it against
    /// the quadrature sum of its members' solo damages says whether the assumption
    /// holds.
    ///
    /// ⚠️ **Memory scales with the set.** Every member's patch is resident at once,
    /// so a 263-tensor set is the whole model again (~13 GB at bf16) on top of the
    /// base mapping. Prefer quartile-sized sets to whole-model ones.
    sets: []const Set = &.{},
    /// CSV sink written **as each arm completes**, flushed every time.
    ///
    /// A full 263-tensor sweep is hours of forwards, and without this a crash, an
    /// OOM or a stray Ctrl-C in hour three leaves nothing at all — the report is
    /// only assembled at the end. With it the run is resumable by hand: the CSV
    /// names the tensors already measured.
    stream_csv: ?*std.Io.Writer = null,

    pub const PatchDtype = enum { f32, bf16 };

    /// A named group of checkpoint tensors, measured as one arm.
    pub const Set = struct {
        label: []const u8,
        tensors: []const []const u8,
    };
};

/// The substituted tensor's bytes at `dt`.
///
/// `owned` is null for `.f32`, which borrows the round-trip buffer directly rather
/// than copying it — worth the extra field because a wide krea2 tensor is 400 MB and
/// a full sweep is 263 of them, on a box where page-cache pressure is the documented
/// failure mode.
const Patch = struct {
    bytes: []const u8,
    owned: ?[]u8,

    fn deinit(self: Patch, gpa: std.mem.Allocator) void {
        if (self.owned) |o| gpa.free(o);
    }
};

fn patchBytes(gpa: std.mem.Allocator, dt: PerTensor.PatchDtype, values: []const f32) !Patch {
    switch (dt) {
        // ⚠️ **Copies, and must.** This used to hand back `values` itself with
        // `owned = null`, which made the patch's lifetime the caller's local
        // buffer rather than the `Patch` — and the quantized arm frees its
        // round-trip buffer at the end of the inner member loop, BEFORE
        // `replaceDenoiser` and the forward. A model that keeps zero-copy views
        // into its store (`sd_unet.mat` does; the DiT materializes, which is why
        // krea2 never showed it) then reads freed memory.
        //
        // It did not fail loudly. Small tensors read freed-but-intact heap and
        // reported plausible numbers; a 26 MB one is mmap-backed, so the free
        // unmapped it and the arm reported rel L2 ~1.0 — *identically for q4_k and
        // q8_0*, which is the tell: an 8-bit round trip cannot be as damaging as a
        // 4-bit one, so the number could not have been about quantization at all.
        // Then it segfaulted. The copy costs one allocation per arm against a
        // whole-model forward.
        .f32 => {
            const out = try gpa.alloc(u8, values.len * 4);
            errdefer gpa.free(out);
            @memcpy(out, std.mem.sliceAsBytes(values));
            return .{ .bytes = out, .owned = out };
        },
        .bf16 => {
            const out = try gpa.alloc(u8, values.len * 2);
            errdefer gpa.free(out);
            tp.dtype.f32ToBf16Row(values, out);
            return .{ .bytes = out, .owned = out };
        },
    }
}

fn patchTpDtype(dt: PerTensor.PatchDtype) tp.DType {
    return switch (dt) {
        .f32 => .f32,
        .bf16 => .bf16,
    };
}


/// One tensor's whole-model divergence, or a control.
pub const TensorArm = struct {
    /// The checkpoint tensor name, or a `control:*` label.
    name: []const u8,
    rows: usize,
    cols: usize,
    kind: Kind,
    /// `prompts × steps` points, prompt-major. Empty when `skipped` is set.
    points: []Point,
    steps: usize,
    /// How many tensors this arm quantized: 1 in per-tensor mode, the set size in
    /// set mode. Reported so a set's number is never mistaken for a layer's.
    members: usize = 1,
    /// Why this tensor produced no measurement, if it did not. A layer the format
    /// cannot encode is unmeasured, never zero — the level-1 harness learned that
    /// the hard way (a zero would rank the model's input projection as its least
    /// sensitive layer).
    skipped: ?[]const u8 = null,

    pub const Kind = enum {
        /// One tensor replaced by its dequantized round-trip through the format.
        quantized,
        /// Nothing substituted, but the DiT reloaded through the overlay anyway.
        /// Expected to be **exactly zero**: it proves the overlay + DiT-swap
        /// machinery is inert, so every other number is about quantization.
        control_reload,
        /// One tensor replaced by its own values, exactly, converted to f32. Not
        /// expected to be zero — it is the cost of the f32 dequant path replacing
        /// the checkpoint's native one, i.e. the floor below which a per-tensor
        /// number is not resolvable.
        control_exact,
    };

    pub fn meanRelL2(self: TensorArm) f64 {
        return meanRelL2Of(self.points);
    }
    pub fn meanCos(self: TensorArm) f64 {
        return meanCosOf(self.points);
    }
    pub fn worstRelL2(self: TensorArm) f64 {
        return worstRelL2Of(self.points);
    }
};

pub const TensorReport = struct {
    arena: std.heap.ArenaAllocator,
    reference: []const u8,
    format: Format,
    convrot_group: usize,
    patch_dtype: PerTensor.PatchDtype,
    prompts: []const []const u8,
    sigmas: []const f32,
    resolution: [2]usize,
    seed: u64,
    backend: []const u8,
    /// 0–2 controls, then one arm per tensor.
    controls: []TensorArm,
    arms: []TensorArm,
    /// The architecture detected from the reference checkpoint's tensor names, or
    /// null for one `ImageArch` does not know. Only `writeSensitivitiesJson` needs
    /// it — for the routable/hiprec split its scale is normalized over — but it is
    /// recorded on the report so a run says which relation it used.
    arch: ?*const imagearch.Arch = null,
    /// False when `--max-tensors` (or an explicit `--tensors` list) covered only part
    /// of the routable population.
    ///
    /// ⚠️ The median anchor is the measured population's, so a partial sweep does not
    /// write a partial file — it writes a **rescaled** one. `--max-tensors` thins by
    /// even stride precisely so a partial ranking stays representative, but the
    /// *anchor* is still a subset's median, which shifts every score.
    complete: bool = true,

    pub fn steps(self: *const TensorReport) usize {
        return self.sigmas.len - 1;
    }

    pub fn deinit(self: *TensorReport) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// Columns of a weight seen as a matrix: everything past the output dimension.
/// For a rank-2 tensor that is `dims[1]`; for a rank-4 convolution it is
/// `ci*kh*kw`, the width its im2col GEMM actually has.
fn flatCols(dims: []const usize) usize {
    var c: usize = 1;
    for (dims[1..]) |d| c *= d;
    return c;
}

/// Which tensors a per-tensor run will measure.
///
/// An explicit list is taken as given (a name the checkpoint lacks is an error, not
/// a skip — a typo must not quietly shrink the sample). Otherwise: every rank-2
/// tensor named `*.weight`, which on the krea2 DiT is exactly the matmul weights,
/// with the `_scale` sidecars of an already-quantized checkpoint excluded.
/// `missing` receives the offending name on `error.MissingTensor` — reported by the
/// caller rather than logged here, so this stays a pure selection function (and its
/// test stays silent, per the repo's no-output-on-pass rule).
///
/// ⚠️ **`prefix` restricts the sweep to the denoiser's own tensors**, and a bundled
/// checkpoint makes that mandatory rather than tidy. An LDM single-file SD
/// checkpoint's container also holds `cond_stage_model.*` (CLIP) and
/// `first_stage_model.*` (the VAE); patching one of those through a **denoiser**
/// overlay changes nothing, so it measures as *zero damage* — which in a
/// sensitivities file is not a missing row but a confident wrong one. Comes from
/// `Session.denoiserPrefix()`, and is `""` for krea2 and for ggufy's model-only
/// GGUFs, where the container holds the denoiser alone.
fn selectTensors(
    arena: std.mem.Allocator,
    base: tp.weights.WeightStore,
    prefix: []const u8,
    pt: PerTensor,
    missing: *[]const u8,
) ![]const []const u8 {
    var picked: std.ArrayList([]const u8) = .empty;

    if (pt.tensors.len > 0) {
        for (pt.tensors) |n| {
            if (base.get(n) == null) {
                missing.* = n;
                return error.MissingTensor;
            }
            if (!std.mem.startsWith(u8, n, prefix)) {
                // Explicit, so this is a user error rather than something to filter:
                // the name exists but is not the denoiser's, and patching it would
                // report zero damage instead of failing.
                missing.* = n;
                return error.TensorNotInDenoiser;
            }
            try picked.append(arena, try arena.dupe(u8, n));
        }
    } else {
        for (base.names()) |n| {
            if (!std.mem.startsWith(u8, n, prefix)) continue;
            if (!std.mem.endsWith(u8, n, ".weight")) continue;
            const v = base.get(n).?;
            // Rank >= 2, not rank == 2: a **convolution** is rank 4, and on a UNet
            // that is most of the model. Measuring only the rank-2 tensors would
            // cover 31% of SD1.5's UNet parameters and silently omit the other
            // 68.5% — and it is the convolutions that make this architecture worth
            // measuring at all. Same partial-coverage trap as the activation probe
            // that recorded a UNet's linears and none of its convs.
            if (v.info.shape.rank < 2) continue;
            const dims = v.info.shape.slice();
            if (dims[0] < 2 or flatCols(dims) < 2) continue;
            try picked.append(arena, try arena.dupe(u8, n));
        }
    }

    if (pt.max_tensors) |cap| if (picked.items.len > cap and cap > 0) {
        // Even stride, not the first `cap`: a prefix of the name order is a prefix
        // of the *network*, and would describe the early blocks only.
        const stride = @as(f64, @floatFromInt(picked.items.len)) / @as(f64, @floatFromInt(cap));
        var thinned: std.ArrayList([]const u8) = .empty;
        for (0..cap) |i| {
            const idx: usize = @intFromFloat(@as(f64, @floatFromInt(i)) * stride);
            try thinned.append(arena, picked.items[@min(idx, picked.items.len - 1)]);
        }
        picked = thinned;
    };

    return picked.items;
}

/// Level 2, per tensor: quantize exactly one tensor of the reference checkpoint and
/// measure the whole model's velocity divergence, for every selected tensor.
///
/// `opts.reference` is both the reference *and* the base for every arm, so
/// `opts.candidates` has no meaning here and is refused rather than ignored.
pub fn runPerTensor(gpa: std.mem.Allocator, opts: Options, pt: PerTensor) !TensorReport {
    if (opts.prompts.len == 0) return error.NoPrompts;
    if (opts.steps < 1) return error.NoSteps;
    // Refused, not ignored: `--candidates` here would look like it was measured.
    if (opts.candidates.len != 0) return error.CandidatesNotUsed;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();

    const lat_h = opts.height / 8;
    const lat_w = opts.width / 8;
    var cancel: std.atomic.Value(bool) = .init(false);

    var pool: ThreadPool = undefined;
    const n_jobs = if (pt.threads == 0) @max(1, std.Thread.getCpuCount() catch 1) else pt.threads;
    try pool.init(.{ .allocator = gpa, .n_jobs = n_jobs });
    defer pool.deinit();

    // ONE session for the whole run: the reference trajectory, then every arm, with
    // only the DiT swapped between them. The text encoder and VAE are loaded once
    // and — the part that matters — the conditioning is encoded once, so no arm can
    // differ from another in its inputs.
    const sess_opts = baseOpts(opts, opts.reference, &cancel);
    var sess = try tp.pipeline.Session.init(opts.io, gpa, sess_opts, opts.log);
    defer sess.deinit();

    // Family-dependent, so it comes from the session (see the whole-checkpoint arm).
    const sched = try sess.schedule(gpa, opts.steps, opts.shift);
    defer gpa.free(sched);
    const sigmas = try arena.dupe(f32, sched);

    var ref = try captureReference(gpa, sess, opts, sigmas, lat_h, lat_w, &cancel);
    defer ref.deinit(gpa);

    const base = sess.denoiserStore();
    var ov: tp.weights.Overlay = .{ .base = base };
    defer ov.deinit(gpa);

    var missing: []const u8 = "";
    // A bundled SD checkpoint's container also holds CLIP and the VAE; only the
    // denoiser's own tensors can be measured through a denoiser overlay.
    const den_prefix = sess.denoiserPrefix();
    const names = selectTensors(arena, base, den_prefix, pt, &missing) catch |err| switch (err) {
        error.MissingTensor => {
            std.log.err("per-tensor: '{s}' is not in the reference checkpoint '{s}'", .{ missing, opts.reference });
            return err;
        },
        error.TensorNotInDenoiser => {
            std.log.err(
                "per-tensor: '{s}' is in '{s}' but is not part of the denoiser (prefix '{s}') — " ++
                    "patching it would measure as zero damage rather than fail",
                .{ missing, opts.reference, den_prefix },
            );
            return err;
        },
        else => return err,
    };
    if (names.len == 0) return error.NoTensors;

    // --- controls -------------------------------------------------------------
    var controls: std.ArrayList(TensorArm) = .empty;
    if (pt.controls) {
        {
            // Empty overlay: same bytes, reached through one more indirection.
            try sess.replaceDenoiser(ov.store());
            const points = try arena.alloc(Point, ref.snaps.len);
            try measureArm(gpa, sess, opts, ref, sigmas, lat_h, lat_w, &cancel, points);
            try controls.append(arena, .{
                .name = "control:reload",
                .rows = 0,
                .cols = 0,
                .kind = .control_reload,
                .points = points,
                .steps = opts.steps,
            });
            try streamArm(pt.stream_csv, controls.items[controls.items.len - 1]);
        }
        {
            // The same tensor the first arm will quantize, substituted by its own
            // exact values at f32. Anything this arm reports is the dequant-path
            // change, not rounding.
            const v = base.get(names[0]).?;
            const exact = try v.toF32Alloc(gpa);
            defer gpa.free(exact);
            // Written back in the run's own patch dtype, so this control carries
            // whatever the patch representation costs and the quantized arms can be
            // read against it. Against a bf16 or fp8 base both dtypes are lossless
            // here, so it stays an exactness check either way.
            const exact_patch = try patchBytes(gpa, pt.patch_dtype, exact);
            defer exact_patch.deinit(gpa);
            ov.clear();
            try ov.put(gpa, names[0], patchTpDtype(pt.patch_dtype), exact_patch.bytes);
            try sess.replaceDenoiser(ov.store());
            const points = try arena.alloc(Point, ref.snaps.len);
            try measureArm(gpa, sess, opts, ref, sigmas, lat_h, lat_w, &cancel, points);
            try controls.append(arena, .{
                .name = try std.fmt.allocPrint(arena, "control:exact-f32 ({s})", .{names[0]}),
                .rows = v.info.shape.dims[0],
                .cols = v.info.shape.dims[1],
                .kind = .control_exact,
                .points = points,
                .steps = opts.steps,
            });
            try streamArm(pt.stream_csv, controls.items[controls.items.len - 1]);
            // Restore before `exact` goes out of scope: the DiT holds views into it.
            ov.clear();
            try sess.replaceDenoiser(base);
        }
    }

    // --- one arm per unit -------------------------------------------------------
    // A "unit" is one tensor in the default mode, or a whole named set when
    // `pt.sets` is given. Sharing the loop keeps the two modes measured identically —
    // an additivity check is only meaningful if the parts and the whole come off the
    // same code path.
    const Unit = struct { label: []const u8, names: []const []const u8 };
    var units: std.ArrayList(Unit) = .empty;
    if (pt.sets.len > 0) {
        for (pt.sets) |set| try units.append(arena, .{ .label = set.label, .names = set.tensors });
    } else {
        for (names) |n| try units.append(arena, .{ .label = n, .names = (try arena.dupe([]const u8, &.{n})) });
    }

    const arms = try arena.alloc(TensorArm, units.items.len);
    for (units.items, 0..) |unit, ai| {
        if (opts.callbacks.isCancelled()) {
            cancel.store(true, .release);
            return error.Canceled;
        }

        // Patches for every member of the unit, all resident until the arm is done.
        var patches: std.ArrayList(Patch) = .empty;
        defer {
            // Drop the overlay's references FIRST: the DiT is still holding this
            // store, so freeing the patch bytes while they are reachable would leave
            // the next forward reading freed memory rather than failing loudly.
            ov.clear();
            for (patches.items) |pp| pp.deinit(gpa);
            patches.deinit(gpa);
        }

        var rows: usize = 0;
        var cols: usize = 0;
        var skip: ?[]const u8 = null;
        var members: usize = 0;

        for (unit.names) |name| {
            const v = base.get(name) orelse {
                skip = try std.fmt.allocPrint(arena, "'{s}' is not in the checkpoint", .{name});
                break;
            };
            const shape = v.info.shape.slice();
            if (shape.len < 2) {
                if (unit.names.len == 1) {
                    skip = try std.fmt.allocPrint(arena, "rank {d}, not a matrix", .{shape.len});
                    break;
                }
                continue; // a set just skips members it cannot use
            }
            // A rank-4 convolution weight [co][ci][kh][kw] is row-major exactly
            // [co][ci*kh*kw], which is how `convert` quantizes it and how the
            // im2col GEMM reads it — so the flattened view needs no reshaping, and
            // the patch keeps the base tensor's rank (`Overlay.put`), so the
            // loader's rank-4 check still passes.
            rows = shape[0];
            cols = flatCols(shape);

            const w = try v.toF32Alloc(gpa);
            defer gpa.free(w);

            // A format that cannot encode this shape (q4_k wants cols % 256 == 0, and
            // krea2's patch embed is 6144x64) leaves the tensor UNMEASURED. Not zero:
            // zero would read as "quantizing this layer is free".
            const w_hat = ph.roundtripGroup(pt.format, gpa, w, rows, cols, &pool, pt.convrot_group) catch |err| {
                if (unit.names.len == 1) {
                    skip = try std.fmt.allocPrint(arena, "{s} cannot encode {d}x{d}: {t}", .{ ph.formats[@intFromEnum(pt.format)].name, rows, cols, err });
                    break;
                }
                continue;
            };
            defer gpa.free(w_hat);

            const patch = try patchBytes(gpa, pt.patch_dtype, w_hat);
            errdefer patch.deinit(gpa);
            try patches.append(gpa, patch);
            try ov.put(gpa, name, patchTpDtype(pt.patch_dtype), patch.bytes);
            members += 1;
        }

        if (skip == null and members == 0) skip = try arena.dupe(u8, "no member could be encoded");
        if (skip) |why| {
            arms[ai] = .{
                .name = unit.label,
                .rows = rows,
                .cols = cols,
                .kind = .quantized,
                .points = &.{},
                .steps = opts.steps,
                .members = members,
                .skipped = why,
            };
            try streamArm(pt.stream_csv, arms[ai]);
            continue;
        }

        try sess.replaceDenoiser(ov.store());

        const points = try arena.alloc(Point, ref.snaps.len);
        try measureArm(gpa, sess, opts, ref, sigmas, lat_h, lat_w, &cancel, points);
        arms[ai] = .{
            .name = unit.label,
            .rows = if (unit.names.len == 1) rows else 0,
            .cols = if (unit.names.len == 1) cols else 0,
            .kind = .quantized,
            .points = points,
            .steps = opts.steps,
            .members = members,
        };
        try streamArm(pt.stream_csv, arms[ai]);
        opts.callbacks.reportProgress(@intCast(ai + 1), @intCast(units.items.len), @intCast(ref.snaps.len), @intCast(ref.snaps.len));
        // One line per arm as it lands: a 24-tensor sweep is over an hour of
        // forwards, and a silent log is indistinguishable from a hang.
        if (opts.log) |lw| {
            if (unit.names.len == 1) {
                try lw.print("[{d}/{d}] {s} ({d}x{d}) rel L2 {e:.4}\n", .{ ai + 1, units.items.len, unit.label, rows, cols, arms[ai].meanRelL2() });
            } else {
                try lw.print("[{d}/{d}] set {s} ({d} tensors) rel L2 {e:.4}\n", .{ ai + 1, units.items.len, unit.label, members, arms[ai].meanRelL2() });
            }
            try lw.flush();
        }

        // Put the base weights back before `w_hat` is freed — the DiT holds views
        // into it, and a session whose weights dangle is a session nobody may use.
        ov.clear();
        try sess.replaceDenoiser(base);
    }

    const prompts = try arena.alloc([]const u8, opts.prompts.len);
    for (opts.prompts, prompts) |src, *dst| dst.* = try arena.dupe(u8, src);

    return .{
        .arena = arena_state,
        .reference = try arena.dupe(u8, opts.reference),
        .format = pt.format,
        .convrot_group = pt.convrot_group,
        .patch_dtype = pt.patch_dtype,
        .prompts = prompts,
        .sigmas = sigmas,
        .resolution = .{ opts.width, opts.height },
        .seed = opts.seed,
        .backend = @tagName(opts.backend),
        .controls = controls.items,
        .arms = arms,
        // Detected from the DiT's own names rather than taken as a flag: this
        // report's only consumer of the arch is the sensitivities emitter, and a
        // mistyped flag there would silently change which tensors set the scale.
        .arch = imagearch.detectArch(base.names()),
        // An explicit tensor list or a `--max-tensors` cap means the anchor is a
        // subset's median, which rescales every score in an emitted file.
        .complete = pt.tensors.len == 0 and pt.max_tensors == null,
    };
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

pub fn writeMarkdown(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("# Level 2 — one-pass velocity divergence\n\n");
    try w.print("- reference: `{s}`\n", .{report.reference});
    try w.print("- {d}x{d}, {d} steps, seed {d}, backend {s}, {d} prompt{s} — **{d} points per arm**\n", .{
        report.resolution[0],  report.resolution[1],
        report.steps(),        report.seed,
        report.backend,        report.prompts.len,
        if (report.prompts.len == 1) "" else "s",
        report.prompts.len * report.steps(),
    });
    try w.writeAll(
        "- **teacher-forced**: every arm predicts from the reference's own latent at each step, so\n" ++
            "  the two models see identical inputs and no trajectory drift enters the numbers.\n",
    );

    try w.writeAll("\n## Arms\n\n");
    try w.writeAll("| arm | mean rel L2 | mean cos | worst rel L2 |\n|---|---:|---:|---:|\n");
    for (report.arms) |a| {
        try w.print("| {s} | {d:.5} | {d:.6} | {d:.5} |\n", .{ a.name, a.meanRelL2(), a.meanCos(), a.worstRelL2() });
    }

    try w.writeAll("\n## Per step (mean rel L2 across prompts)\n\n");
    try w.writeAll("| step | sigma |");
    for (report.arms) |a| try w.print(" {s} |", .{a.name});
    try w.writeAll("\n|---|---:|");
    for (report.arms) |_| try w.writeAll("---:|");
    try w.writeByte('\n');
    for (0..report.steps()) |i| {
        try w.print("| {d} | {d:.3} |", .{ i + 1, report.sigmas[i] });
        for (report.arms) |a| try w.print(" {d:.5} |", .{a.meanRelL2AtStep(i)});
        try w.writeByte('\n');
    }

    try w.writeAll(
        "\n> A rising column with the step index means the arm's error grows as the latent gets\n" ++
            "> cleaner (late steps carry the detail); a flat one means the damage is uniform along the\n" ++
            "> schedule. Both are invisible at level 1, which sees one layer at one distribution.\n",
    );

    try w.writeAll("\n## Prompts\n\n");
    for (report.prompts, 0..) |p, i| try w.print("{d}. {s}\n", .{ i + 1, p });
}

pub fn writeCsv(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("arm,prompt,step,sigma,rel_l2,cos,mean_pos_cos,max_pos_rel\n");
    for (report.arms) |a| {
        for (a.points, 0..) |p, idx| {
            try w.print("{s},{d},{d},{d:.6},{d:.8},{d:.8},{d:.8},{d:.8}\n", .{
                a.name, idx / a.steps, p.step, p.sigma, p.rel_l2, p.cos, p.mean_pos_cos, p.max_pos_rel,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Per-tensor report, and the level-1 correlation
// ---------------------------------------------------------------------------

/// Rank the arms by mean rel L2, descending — most damaging tensor first. Returns
/// indices into `report.arms`, skipping the unmeasured ones.
fn rankArms(arena: std.mem.Allocator, arms: []const TensorArm) ![]usize {
    var idx: std.ArrayList(usize) = .empty;
    for (arms, 0..) |a, i| if (a.skipped == null and a.points.len > 0) try idx.append(arena, i);
    const items = idx.items;
    std.mem.sort(usize, items, arms, struct {
        fn lt(ctx: []const TensorArm, a: usize, b: usize) bool {
            return ctx[a].meanRelL2() > ctx[b].meanRelL2();
        }
    }.lt);
    return items;
}

pub fn writeTensorMarkdown(w: *std.Io.Writer, report: *const TensorReport, l1: ?*const Level1) !void {
    var scratch = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer scratch.deinit();

    try w.writeAll("# Level 2 — per-tensor attribution\n\n");
    try w.print("- reference: `{s}`\n", .{report.reference});
    try w.print("- format: **{s}** (ConvRot group {d}), patch written back as **{t}**\n", .{ ph.formats[@intFromEnum(report.format)].name, report.convrot_group, report.patch_dtype });
    try w.print("- {d}x{d}, {d} steps, seed {d}, backend {s}, {d} prompt{s} — **{d} points per arm**, {d} arms\n", .{
        report.resolution[0], report.resolution[1],
        report.steps(),       report.seed,
        report.backend,       report.prompts.len,
        if (report.prompts.len == 1) "" else "s",
        report.prompts.len * report.steps(),
        report.arms.len,
    });
    try w.writeAll(
        "- each arm quantizes **exactly one tensor** of the reference checkpoint (dequantized back\n" ++
            "  to f32 and substituted through a weight overlay) and measures the whole model's\n" ++
            "  velocity error, teacher-forced on the reference's own trajectory.\n",
    );

    if (report.controls.len > 0) {
        try w.writeAll("\n## Controls\n\n");
        try w.writeAll("| control | mean rel L2 | worst rel L2 | expected |\n|---|---:|---:|---|\n");
        for (report.controls) |c| {
            const expect: []const u8 = switch (c.kind) {
                .control_reload => "exactly 0 — the overlay and DiT swap must be inert",
                .control_exact => "the f32 dequant path's own cost: the resolution floor",
                .quantized => "",
            };
            try w.print("| {s} | {e:.4} | {e:.4} | {s} |\n", .{ c.name, c.meanRelL2(), c.worstRelL2(), expect });
        }
        for (report.controls) |c| if (c.kind == .control_reload and c.meanRelL2() != 0) {
            try w.print(
                "\n> ⚠️ **`control:reload` is not zero ({e:.4}).** The overlay path is supposed to be\n" ++
                    "> byte-for-byte the base checkpoint, so every number below carries this as an\n" ++
                    "> unexplained offset. Do not quote the table until this is understood.\n",
                .{c.meanRelL2()},
            );
        };
    }

    const floor: f64 = blk: {
        for (report.controls) |c| if (c.kind == .control_exact) break :blk c.meanRelL2();
        break :blk 0;
    };

    const order = try rankArms(scratch.allocator(), report.arms);
    try w.writeAll("\n## Tensors, most damaging first\n\n");
    try w.writeAll("| # | tensor | shape | mean rel L2 | mean cos | worst rel L2 |");
    if (floor > 0) try w.writeAll(" x floor |");
    if (l1) |_| try w.writeAll(" level 1 rel L2 |");
    try w.writeAll("\n|---|---|---|---:|---:|---:|");
    if (floor > 0) try w.writeAll("---:|");
    if (l1) |_| try w.writeAll("---:|");
    try w.writeByte('\n');
    for (order, 1..) |i, n| {
        const a = report.arms[i];
        try w.print("| {d} | `{s}` | {d}x{d} | {e:.4} | {d:.6} | {e:.4} |", .{ n, a.name, a.rows, a.cols, a.meanRelL2(), a.meanCos(), a.worstRelL2() });
        if (floor > 0) try w.print(" {d:.1} |", .{a.meanRelL2() / floor});
        if (l1) |m| {
            if (m.get(a.name)) |e| try w.print(" {e:.4} |", .{e}) else try w.writeAll(" — |");
        }
        try w.writeByte('\n');
    }

    var skipped: usize = 0;
    for (report.arms) |a| if (a.skipped != null) {
        skipped += 1;
    };
    if (skipped > 0) {
        try w.print("\n## Unmeasured ({d})\n\n", .{skipped});
        for (report.arms) |a| if (a.skipped) |why| try w.print("- `{s}`: {s}\n", .{ a.name, why });
        try w.writeAll(
            "\n> Unmeasured, not zero. A layer the format cannot encode has no number here, and\n" ++
                "> scoring it as 0 would rank it as the safest layer in the model.\n",
        );
    }

    if (l1) |m| {
        try w.writeAll("\n## Against level 1\n\n");
        try writeCorrelation(w, report, m, scratch.allocator());
    }

    try w.writeAll("\n## Prompts\n\n");
    for (report.prompts, 0..) |p, i| try w.print("{d}. {s}\n", .{ i + 1, p });
}

/// The per-tensor CSV header. Public so a caller that streams (`PerTensor.stream_csv`)
/// writes the same schema the final report does.
pub const tensor_csv_header = "tensor,rows,cols,kind,prompt,step,sigma,rel_l2,cos,mean_pos_cos,max_pos_rel\n";

fn writeArmRows(w: *std.Io.Writer, a: TensorArm) !void {
    for (a.points, 0..) |p, idx| {
        try w.print("{s},{d},{d},{t},{d},{d},{d:.6},{d:.8},{d:.8},{d:.8},{d:.8}\n", .{
            a.name, a.rows,   a.cols,        @as(TensorArm.Kind, a.kind), idx / a.steps,
            p.step, p.sigma,  p.rel_l2,      p.cos,                       p.mean_pos_cos,
            p.max_pos_rel,
        });
    }
}

/// Append one finished arm to the streaming sink and flush, so an interrupted sweep
/// keeps everything measured up to that point.
fn streamArm(sink: ?*std.Io.Writer, a: TensorArm) !void {
    const w = sink orelse return;
    try writeArmRows(w, a);
    try w.flush();
}

/// Emit a `sensitivities/<arch>.json` from **measured per-tensor whole-model
/// damage** — the file `Convert.zig` routes precision on, generated from the one
/// arm that measures the quantity routing actually needs.
///
/// This is the cheap form of the brute-force method the hand-built `sdxl.json` /
/// `sd1.5.json` came from (~100 renders per layer): 16 forwards per layer, no
/// sampler, no VAE, and drift-free. It is the recommended generator — level 1's
/// per-layer proxy ranks layers at only Spearman 0.36 against this.
///
/// The 1–100 encoding is `LadderScore`'s, shared with level 1 so that the two
/// generators and the converter's interpretation cannot drift apart again. Read that
/// module before touching the numbers: the percentile encoding this replaced
/// produced a krea2 file measuring 2.76× off the uniform rate–distortion curve.
///
/// `arch` supplies the routable/hiprec split the scale is normalized over; pass null
/// for an unknown architecture and every measured tensor takes part. On krea2 that
/// matters a lot — the seven most damaging tensors are all `keys_hiprec`, so letting
/// them set the scale compresses the 224 the score can steer.
///
/// Unmeasured layers are **omitted**, never scored 0: `Convert.zig`'s "no entry →
/// use the target type" fallback is the correct handling, whereas a 0 would assert
/// that the layer is the safest in the model.
pub fn writeSensitivitiesJson(
    gpa: std.mem.Allocator,
    w: *std.Io.Writer,
    report: *const TensorReport,
    strip_prefix: []const u8,
    arch: ?*const imagearch.Arch,
) !void {
    var pop: std.ArrayList(f64) = .empty;
    defer pop.deinit(gpa);
    var n: usize = 0;
    for (report.arms) |a| {
        if (a.skipped != null or a.points.len == 0) continue;
        n += 1;
        if (LadderScore.isRoutable(arch, a.name)) try pop.append(gpa, a.meanRelL2());
    }
    if (n == 0) return error.NoMeasuredLayers;
    // Nothing routable measured: normalize over everything rather than emit a file
    // of flat scores. Not a case to optimize for — it means the sweep covered only
    // structurally-protected tensors — but it must not silently produce a uniform file.
    if (pop.items.len == 0) {
        for (report.arms) |a| {
            if (a.skipped != null or a.points.len == 0) continue;
            try pop.append(gpa, a.meanRelL2());
        }
    }
    const ladder = try LadderScore.fromDamages(gpa, pop.items);

    try w.writeAll("{\n");
    try (LadderScore.Provenance{
        .generator = "level2-per-tensor-damage",
        .arch = if (arch) |a| a.name else "",
        .reference_format = ph.formats[@intFromEnum(report.format)].name,
        .scored = n,
        .anchor_population = pop.items.len,
        .complete = report.complete,
    }).write(w);
    var written: usize = 1;
    for (report.arms) |a| {
        if (a.skipped != null or a.points.len == 0) continue;
        const d = a.meanRelL2();
        const score: f64 = if (ladder) |l| l.score(d) else LadderScore.homogeneous_score;
        const name = if (strip_prefix.len > 0 and std.mem.startsWith(u8, a.name, strip_prefix))
            a.name[strip_prefix.len..]
        else
            a.name;
        if (written > 0) try w.writeAll(",\n");
        try w.print("  {f}: {d:.6}", .{ std.json.fmt(name, .{}), score });
        written += 1;
    }
    if (written > 0) try w.writeByte('\n');
    try w.writeAll("}\n");
}

pub fn writeTensorCsv(w: *std.Io.Writer, report: *const TensorReport) !void {
    try w.writeAll(tensor_csv_header);
    const all = [_][]const TensorArm{ report.controls, report.arms };
    for (all) |group| {
        for (group) |a| try writeArmRows(w, a);
    }
}

/// One format's per-layer rel-L2 from a level-1 CSV, for the correlation.
///
/// ⚠️ **The format and arm must be named explicitly.** A level-1 CSV holds every
/// format and every arm; picking rows by name is the difference between comparing
/// this run's format against itself and comparing it against a different mechanism's
/// numbers. That mistake has already been made once in this programme (`l2_analyze.py`
/// printed INT4_CONVROT's level-1 ratio against a k-quant run), so there is no
/// default here.
pub const Level1 = struct {
    arena: std.heap.ArenaAllocator,
    map: std.StringHashMapUnmanaged(f64) = .empty,
    format_name: []const u8,
    arm: []const u8,

    pub fn get(self: *const Level1, name: []const u8) ?f64 {
        return self.map.get(name);
    }

    pub fn deinit(self: *Level1) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Parse a `Sensitivity.writeCsv` file, keeping the rows for one (format, arm).
    pub fn parseCsv(gpa: std.mem.Allocator, text: []const u8, format_name: []const u8, arm: []const u8) !Level1 {
        var self: Level1 = .{
            .arena = std.heap.ArenaAllocator.init(gpa),
            .format_name = "",
            .arm = "",
        };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.format_name = try a.dupe(u8, format_name);
        self.arm = try a.dupe(u8, arm);

        // layer,rows,cols,tokens,score,format,bits,arm,rel_l2,mean_token_cos,max_token_rel
        var lines = std.mem.splitScalar(u8, text, '\n');
        const header = lines.next() orelse return error.EmptyCsv;
        if (!std.mem.startsWith(u8, header, "layer,rows,cols,tokens,score,format,bits,arm,rel_l2")) return error.NotALevel1Csv;

        while (lines.next()) |line| {
            if (line.len == 0) continue;
            var f = std.mem.splitScalar(u8, std.mem.trim(u8, line, "\r"), ',');
            const layer = f.next() orelse continue;
            _ = f.next() orelse continue; // rows
            _ = f.next() orelse continue; // cols
            _ = f.next() orelse continue; // tokens
            _ = f.next() orelse continue; // score
            const fmt_name = f.next() orelse continue;
            _ = f.next() orelse continue; // bits
            const arm_name = f.next() orelse continue;
            const rel = f.next() orelse continue;
            if (!std.mem.eql(u8, fmt_name, format_name)) continue;
            if (!std.mem.eql(u8, arm_name, arm)) continue;
            const v = std.fmt.parseFloat(f64, rel) catch continue;
            try self.map.put(a, try a.dupe(u8, layer), v);
        }
        if (self.map.count() == 0) return error.NoMatchingRows;
        return self;
    }
};

/// Pearson correlation. Returns 0 for a degenerate input rather than a NaN.
pub fn pearson(a: []const f64, b: []const f64) f64 {
    std.debug.assert(a.len == b.len);
    if (a.len < 2) return 0;
    const n: f64 = @floatFromInt(a.len);
    var ma: f64 = 0;
    var mb: f64 = 0;
    for (a, b) |x, y| {
        ma += x;
        mb += y;
    }
    ma /= n;
    mb /= n;
    var sab: f64 = 0;
    var saa: f64 = 0;
    var sbb: f64 = 0;
    for (a, b) |x, y| {
        const dx = x - ma;
        const dy = y - mb;
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    if (saa <= 0 or sbb <= 0) return 0;
    return sab / (@sqrt(saa) * @sqrt(sbb));
}

/// Fractional ranks with ties averaged, written into `out`.
fn rankInto(arena: std.mem.Allocator, v: []const f64, out: []f64) !void {
    const idx = try arena.alloc(usize, v.len);
    for (idx, 0..) |*p, i| p.* = i;
    std.mem.sort(usize, idx, v, struct {
        fn lt(ctx: []const f64, x: usize, y: usize) bool {
            return ctx[x] < ctx[y];
        }
    }.lt);
    var i: usize = 0;
    while (i < idx.len) {
        var j = i + 1;
        while (j < idx.len and v[idx[j]] == v[idx[i]]) j += 1;
        // Ties share the mean of the ranks they span, or an arbitrary order inside a
        // tie would become a correlation.
        const mean_rank = (@as(f64, @floatFromInt(i)) + @as(f64, @floatFromInt(j - 1))) / 2 + 1;
        for (idx[i..j]) |k| out[k] = mean_rank;
        i = j;
    }
}

/// Spearman's rank correlation — Pearson on tie-averaged ranks.
pub fn spearman(arena: std.mem.Allocator, a: []const f64, b: []const f64) !f64 {
    std.debug.assert(a.len == b.len);
    if (a.len < 2) return 0;
    const ra = try arena.alloc(f64, a.len);
    const rb = try arena.alloc(f64, b.len);
    try rankInto(arena, a, ra);
    try rankInto(arena, b, rb);
    return pearson(ra, rb);
}

/// The comparison this whole arm exists for: does level 1's per-layer ranking
/// predict the per-layer effect on the model's own prediction?
fn writeCorrelation(w: *std.Io.Writer, report: *const TensorReport, l1: *const Level1, arena: std.mem.Allocator) !void {
    var lvl1: std.ArrayList(f64) = .empty;
    var lvl2: std.ArrayList(f64) = .empty;
    var unmatched: usize = 0;
    for (report.arms) |a| {
        if (a.skipped != null or a.points.len == 0) continue;
        const e1 = l1.get(a.name) orelse {
            unmatched += 1;
            continue;
        };
        const e2 = a.meanRelL2();
        if (e1 <= 0 or e2 <= 0) continue;
        try lvl1.append(arena, e1);
        try lvl2.append(arena, e2);
    }

    try w.print("- level 1 source: format `{s}`, arm `{s}`\n", .{ l1.format_name, l1.arm });
    if (lvl1.items.len < 3) {
        try w.print("- only {d} tensors matched — not enough to correlate.\n", .{lvl1.items.len});
        return;
    }

    // Logs, because both quantities span orders of magnitude across layers: a raw
    // Pearson would be a statement about the two or three widest layers.
    const la = try arena.alloc(f64, lvl1.items.len);
    const lb = try arena.alloc(f64, lvl2.items.len);
    for (lvl1.items, la) |v, *p| p.* = @log10(v);
    for (lvl2.items, lb) |v, *p| p.* = @log10(v);

    const rho = try spearman(arena, lvl1.items, lvl2.items);
    const r_log = pearson(la, lb);
    try w.print("- **n = {d}** matched tensors{s}\n", .{
        lvl1.items.len,
        if (unmatched > 0) " (some arms had no level-1 row; see below)" else "",
    });
    try w.print("- **Spearman rho = {d:.4}** — does level 1 rank layers the way the model does?\n", .{rho});
    try w.print("- Pearson r on log10 = {d:.4}\n", .{r_log});
    if (unmatched > 0) try w.print("- {d} measured tensor(s) had no matching level-1 row and were left out.\n", .{unmatched});
    try w.writeAll(
        "\n> Level 1 measures one layer's own output error; this measures the whole model's\n" ++
            "> velocity error with only that layer quantized. They are different quantities, so\n" ++
            "> rho is the useful figure: it is exactly the question `convert`'s sensitivity routing\n" ++
            "> asks of level 1 — *rank these layers* — and nothing above level 1 had ever checked it.\n",
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "point: identical velocities are a perfect match, and a scaled one keeps its direction" {
    const positions = 4;
    const channels = 2;
    var a: [positions * channels]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (&a) |*v| v.* = prng.random().float(f32) * 2 - 1;

    const same = point(0, 1.0, &a, &a, positions, channels);
    try testing.expectApproxEqAbs(@as(f64, 0), same.rel_l2, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), same.cos, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), same.mean_pos_cos, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), same.max_pos_rel, 1e-12);

    // A uniformly scaled velocity is the same direction with the wrong magnitude:
    // cosine must not notice, rel L2 must. That is why both are reported — a
    // quantization that only loses scale and one that loses direction are
    // different failures.
    var scaled: [positions * channels]f32 = undefined;
    for (&scaled, a) |*d, s| d.* = s * 1.1;
    const sc = point(0, 1.0, &a, &scaled, positions, channels);
    try testing.expectApproxEqAbs(@as(f64, 1), sc.cos, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1), sc.mean_pos_cos, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.1), sc.rel_l2, 1e-6);
}

test "point: a zero-velocity position is skipped rather than scoring as perfect" {
    // Early in the schedule whole channels can be zero; counting 0/0 as cos = 1
    // would dilute every real number in the average.
    const positions = 2;
    const channels = 2;
    // Position 0 is all zero in the reference; position 1 is real and exactly wrong
    // by 90 degrees.
    const ref = [_]f32{ 0, 1, 0, 0 }; // channel-planar: c0 = {0, 1}, c1 = {0, 0}
    const apx = [_]f32{ 5, 0, 5, 1 };
    const p = point(0, 1.0, &ref, &apx, positions, channels);
    // Only position 1 counted: ref (1, 0) vs apx (0, 1) -> cos 0.
    try testing.expectApproxEqAbs(@as(f64, 0), p.mean_pos_cos, 1e-12);
    try testing.expect(p.max_pos_rel > 1.0);
}

test "ArmResult aggregates and the per-step curve picks the right points" {
    const pts = [_]Point{
        .{ .step = 0, .sigma = 1.0, .rel_l2 = 0.10, .cos = 0.99, .mean_pos_cos = 0.99, .max_pos_rel = 0.5 },
        .{ .step = 1, .sigma = 0.5, .rel_l2 = 0.30, .cos = 0.97, .mean_pos_cos = 0.97, .max_pos_rel = 0.9 },
        .{ .step = 0, .sigma = 1.0, .rel_l2 = 0.20, .cos = 0.98, .mean_pos_cos = 0.98, .max_pos_rel = 0.6 },
        .{ .step = 1, .sigma = 0.5, .rel_l2 = 0.40, .cos = 0.96, .mean_pos_cos = 0.96, .max_pos_rel = 1.0 },
    };
    const arm: ArmResult = .{ .name = "x", .path = "x.safetensors", .points = @constCast(&pts), .steps = 2 };
    try testing.expectApproxEqAbs(@as(f64, 0.25), arm.meanRelL2(), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.40), arm.worstRelL2(), 1e-12);
    // Two prompts, so each step index averages two points.
    try testing.expectApproxEqAbs(@as(f64, 0.15), arm.meanRelL2AtStep(0), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.35), arm.meanRelL2AtStep(1), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0), arm.meanRelL2AtStep(9), 1e-12);
}

test "basename strips the directory and one extension" {
    try testing.expectEqualStrings("anim-int4-calib", basename("out/anim-int4-calib.safetensors"));
    try testing.expectEqualStrings("model", basename("model"));
    try testing.expectEqualStrings(".hidden", basename(".hidden"));
}

test "run rejects a request it cannot measure" {
    const gpa = testing.allocator;
    const io = testing.io;
    const base: Options = .{
        .io = io,
        .reference = "ref.safetensors",
        .candidates = &.{"cand.safetensors"},
        .text_encoder_path = "te.safetensors",
        .vae_path = "vae.safetensors",
        .prompts = &.{"a prompt"},
    };
    {
        var o = base;
        o.candidates = &.{};
        try testing.expectError(error.NoCandidates, run(gpa, o));
    }
    {
        var o = base;
        o.prompts = &.{};
        try testing.expectError(error.NoPrompts, run(gpa, o));
    }
    {
        var o = base;
        o.steps = 0;
        try testing.expectError(error.NoSteps, run(gpa, o));
    }
}

test "runPerTensor refuses the two ways to ask for something it does not measure" {
    const gpa = testing.allocator;
    const base: Options = .{
        .io = testing.io,
        .reference = "ref.safetensors",
        .candidates = &.{},
        .text_encoder_path = "te.safetensors",
        .vae_path = "vae.safetensors",
        .prompts = &.{"a prompt"},
    };
    const pt: PerTensor = .{ .format = .q4_k };
    {
        // `--candidates` has no meaning in this mode; ignoring it would leave a
        // report that looks as if those checkpoints were measured.
        var o = base;
        o.candidates = &.{"cand.safetensors"};
        try testing.expectError(error.CandidatesNotUsed, runPerTensor(gpa, o, pt));
    }
    {
        var o = base;
        o.prompts = &.{};
        try testing.expectError(error.NoPrompts, runPerTensor(gpa, o, pt));
    }
    {
        var o = base;
        o.steps = 0;
        try testing.expectError(error.NoSteps, runPerTensor(gpa, o, pt));
    }
}

test "selectTensors: explicit names are taken as given, derived names are the matrices" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    // A three-tensor safetensors: one matrix weight, one vector weight, one scale.
    const header =
        \\{"blocks.0.attn.wq.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"blocks.0.prenorm.weight":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"blocks.0.attn.wq.weight_scale":{"dtype":"F32","shape":[2,2],"data_offsets":[24,40]}}
    ;
    const bytes = try gpa.alloc(u8, 8 + header.len + 40);
    defer gpa.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);

    var st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes);
    defer st.deinit();
    const store: tp.weights.WeightStore = .{ .safetensors = &st };

    // Derived: the rank-2 `.weight`. The rank-1 norm is not a matmul, and the
    // `_scale` sidecar is not a weight at all.
    var missing: []const u8 = "";
    const derived = try selectTensors(arena.allocator(), store, "", .{ .format = .q4_k }, &missing);
    try testing.expectEqual(@as(usize, 1), derived.len);
    try testing.expectEqualStrings("blocks.0.attn.wq.weight", derived[0]);

    // Explicit: taken as given, in the order given.
    const given = try selectTensors(arena.allocator(), store, "", .{
        .format = .q4_k,
        .tensors = &.{ "blocks.0.prenorm.weight", "blocks.0.attn.wq.weight" },
    }, &missing);
    try testing.expectEqual(@as(usize, 2), given.len);
    try testing.expectEqualStrings("blocks.0.prenorm.weight", given[0]);

    // A name the checkpoint lacks is an error: a typo must not silently shrink the
    // sample a correlation is computed over.
    try testing.expectError(error.MissingTensor, selectTensors(arena.allocator(), store, "", .{
        .format = .q4_k,
        .tensors = &.{"blocks.99.nope.weight"},
    }, &missing));
    try testing.expectEqualStrings("blocks.99.nope.weight", missing);

    // A prefix restricts the derived sweep to the denoiser's own tensors — the
    // bundled-SD case, where the container also holds CLIP and the VAE.
    const scoped = try selectTensors(arena.allocator(), store, "blocks.", .{ .format = .q4_k }, &missing);
    try testing.expectEqual(@as(usize, 1), scoped.len);
    const none = try selectTensors(arena.allocator(), store, "model.diffusion_model.", .{ .format = .q4_k }, &missing);
    try testing.expectEqual(@as(usize, 0), none.len);

    // An EXPLICIT name outside the denoiser fails rather than being filtered: it
    // would otherwise report zero damage, which is a confident wrong row.
    try testing.expectError(error.TensorNotInDenoiser, selectTensors(arena.allocator(), store, "model.diffusion_model.", .{
        .format = .q4_k,
        .tensors = &.{"blocks.0.attn.wq.weight"},
    }, &missing));
}

test "max_tensors thins by even stride, not by prefix" {
    // A prefix of the name order is a prefix of the network: capping at 2 must not
    // reduce a 28-block model to blocks 0 and 1.
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    var buf: [4096]u8 = undefined;
    var names: [10][]const u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var head: std.ArrayList(u8) = .empty;
    defer head.deinit(gpa);
    try head.append(gpa, '{');
    for (0..10) |i| {
        const start = w.buffered().len;
        try w.print("blocks.{d}.attn.wq.weight", .{i});
        names[i] = w.buffered()[start..];
        if (i > 0) try head.append(gpa, ',');
        try head.print(gpa, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[{d},{d}]}}", .{ names[i], i * 16, (i + 1) * 16 });
    }
    try head.append(gpa, '}');

    const file = try gpa.alloc(u8, 8 + head.items.len + 160);
    defer gpa.free(file);
    std.mem.writeInt(u64, file[0..8], head.items.len, .little);
    @memcpy(file[8..][0..head.items.len], head.items);
    @memset(file[8 + head.items.len ..], 0);

    var st = try tp.safetensors.SafeTensors.initFromSlice(gpa, file);
    defer st.deinit();
    const store: tp.weights.WeightStore = .{ .safetensors = &st };

    var missing: []const u8 = "";
    const picked = try selectTensors(arena.allocator(), store, "", .{ .format = .q4_k, .max_tensors = 3 }, &missing);
    try testing.expectEqual(@as(usize, 3), picked.len);
    try testing.expectEqualStrings("blocks.0.attn.wq.weight", picked[0]);
    try testing.expectEqualStrings("blocks.3.attn.wq.weight", picked[1]);
    try testing.expectEqualStrings("blocks.6.attn.wq.weight", picked[2]);
}

test "pearson and spearman against hand-computed values" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const x = [_]f64{ 1, 2, 3, 4, 5 };
    const y = [_]f64{ 2, 4, 6, 8, 10 }; // perfectly linear
    try testing.expectApproxEqAbs(@as(f64, 1), pearson(&x, &y), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1), try spearman(a, &x, &y), 1e-12);

    const rev = [_]f64{ 10, 8, 6, 4, 2 };
    try testing.expectApproxEqAbs(@as(f64, -1), pearson(&x, &rev), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -1), try spearman(a, &x, &rev), 1e-12);

    // Monotone but strongly non-linear: Spearman sees a perfect ranking, Pearson
    // does not. This is why rho is the headline figure — level 1 and level 2 are
    // different quantities and only their ordering has to agree.
    const exp = [_]f64{ 1, 2, 4, 8, 1024 };
    try testing.expectApproxEqAbs(@as(f64, 1), try spearman(a, &x, &exp), 1e-12);
    try testing.expect(pearson(&x, &exp) < 0.85);

    // Ties share the mean rank, so the tie's internal order cannot create signal.
    const t1 = [_]f64{ 1, 1, 2, 3 };
    const t2 = [_]f64{ 5, 5, 6, 7 };
    try testing.expectApproxEqAbs(@as(f64, 1), try spearman(a, &t1, &t2), 1e-12);
    const t3 = [_]f64{ 5, 5, 7, 6 }; // the tie swapped, the rest inverted at the top
    try testing.expect((try spearman(a, &t1, &t3)) < 1.0);

    // Degenerate inputs are 0, not NaN: a constant column would otherwise poison a
    // whole report with a NaN nobody can interpret.
    const flat = [_]f64{ 3, 3, 3, 3, 3 };
    try testing.expectEqual(@as(f64, 0), pearson(&x, &flat));
    try testing.expectEqual(@as(f64, 0), pearson(x[0..1], y[0..1]));
}

test "Level1 CSV parse keeps exactly the requested format and arm" {
    const gpa = testing.allocator;
    const csv =
        "layer,rows,cols,tokens,score,format,bits,arm,rel_l2,mean_token_cos,max_token_rel\n" ++
        "blocks.0.attn.wq.weight,6144,6144,96,50.0000,Q4_K,4.5,format,1.5e-2,0.9,3e-2\n" ++
        "blocks.0.attn.wq.weight,6144,6144,96,50.0000,Q4_K,4.5,weighted,1.2e-2,0.9,3e-2\n" ++
        "blocks.0.attn.wq.weight,6144,6144,96,50.0000,INT4_CR,4.03,format,9.9e-2,0.8,2e-1\n" ++
        "blocks.1.mlp.down.weight,6144,16384,96,80.0000,Q4_K,4.5,format,2.5e-2,0.9,4e-2\n";

    var l1 = try Level1.parseCsv(gpa, csv, "Q4_K", "format");
    defer l1.deinit();

    // Two rows kept: the same layer's `weighted` arm and the other format's row are
    // different mechanisms and must not be mixed into one correlation.
    try testing.expectEqual(@as(usize, 2), l1.map.count());
    try testing.expectApproxEqAbs(@as(f64, 1.5e-2), l1.get("blocks.0.attn.wq.weight").?, 1e-15);
    try testing.expectApproxEqAbs(@as(f64, 2.5e-2), l1.get("blocks.1.mlp.down.weight").?, 1e-15);

    var weighted = try Level1.parseCsv(gpa, csv, "Q4_K", "weighted");
    defer weighted.deinit();
    try testing.expectApproxEqAbs(@as(f64, 1.2e-2), weighted.get("blocks.0.attn.wq.weight").?, 1e-15);

    try testing.expectError(error.NoMatchingRows, Level1.parseCsv(gpa, csv, "Q2_K", "format"));
    try testing.expectError(error.NotALevel1Csv, Level1.parseCsv(gpa, "a,b,c\n1,2,3\n", "Q4_K", "format"));
}

test "the sensitivities emitter keeps magnitude, so a long tail routes as a long tail" {
    // The bug this guards against is not in either component but in their
    // composition: level 1 emitted percentile RANKS while `Convert.zig` reads the
    // score as a linear position on the precision ladder, so the model spread evenly
    // across levels regardless of the damage distribution — measured on krea2 as a
    // 2.3x worse model at a larger size than plain uniform q6_k.
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    // A long-tailed damage distribution: one bad layer, the rest clustered low.
    const dmg = [_]f64{ 0.020, 0.0030, 0.0025, 0.0022, 0.0020 };
    var arms: [6]TensorArm = undefined;
    var pts: [6][1]Point = undefined;
    for (dmg, 0..) |d, i| {
        pts[i] = .{.{ .step = 0, .sigma = 1.0, .rel_l2 = d, .cos = 1, .mean_pos_cos = 1, .max_pos_rel = d }};
        arms[i] = .{
            .name = try std.fmt.allocPrint(arena.allocator(), "pfx.blocks.{d}.attn.wq.weight", .{i}),
            .rows = 8, .cols = 8, .kind = .quantized, .points = &pts[i], .steps = 1,
        };
    }
    // An unmeasurable layer must be OMITTED, not scored 0 — a 0 would tell the
    // converter this is the safest layer in the model.
    arms[5] = .{ .name = "pfx.first.weight", .rows = 6144, .cols = 64, .kind = .quantized, .points = &.{}, .steps = 1, .skipped = "cannot encode" };

    var report: TensorReport = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .reference = "r", .format = .q4_k, .convrot_group = 256, .patch_dtype = .f32,
        .prompts = &.{}, .sigmas = &.{ 1.0, 0.0 }, .resolution = .{ 256, 256 },
        .seed = 0, .backend = "cpu", .controls = &.{}, .arms = arms[0..],
    };
    defer report.arena.deinit();

    var buf: [2048]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeSensitivitiesJson(gpa, &w, &report, "pfx.", null);
    const out = w.buffered();

    // The prefix is stripped, because that is the namespace the converter looks up.
    try testing.expect(std.mem.indexOf(u8, out, "\"blocks.0.attn.wq.weight\"") != null);
    try testing.expect(std.mem.indexOf(u8, out, "pfx.") == null);
    // The unmeasured layer is absent entirely.
    try testing.expect(std.mem.indexOf(u8, out, "first.weight") == null);

    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, out, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    // Five scores plus the reserved provenance key, which is not a tensor name.
    try testing.expectEqual(@as(usize, 6), obj.count());
    const meta = obj.get(LadderScore.meta_key).?.object;
    try testing.expectEqualStrings("level2-per-tensor-damage", meta.get("generator").?.string);
    try testing.expectEqualStrings(LadderScore.encoding_id, meta.get("encoding").?.string);

    // 0.020 against a 0.0025 median is exactly 3 doublings, i.e. three quarters of
    // the way up the ladder — the encoding reports how much worse the layer is, not
    // merely that it is the worst.
    const worst = obj.get("blocks.0.attn.wq.weight").?.float;
    const best = obj.get("blocks.4.attn.wq.weight").?.float;
    try testing.expectApproxEqAbs(@as(f64, 1 + 99 * 0.75), worst, 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1), best, 1e-9);

    // The property that matters: with a long tail, the MEDIAN layer must stay low
    // enough that the converter leaves it at the target type. Convert.zig maps
    // norm = (score-1)/99 through norm^(0.5+3*hard/100) and rounds onto the ladder,
    // so at the default aggressiveness 50 (exponent 2) a layer needs
    // norm² · (levels−1) ≥ 0.5 to be upgraded at all. A percentile encoding puts the
    // median at 50.5 and upgrades half the model; magnitude must not.
    const median = obj.get("blocks.2.attn.wq.weight").?.float;
    try testing.expect(median < LadderScore.upgradeThreshold(50, LadderScore.default_ladder_levels).?);
    var upgraded: usize = 0;
    var it = obj.iterator();
    const max_idx: f64 = @floatFromInt(LadderScore.default_ladder_levels - 1);
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, LadderScore.meta_key)) continue;
        const norm = (e.value_ptr.float - 1) / 99;
        if (std.math.pow(f64, norm, 2.0) * max_idx >= 0.5) upgraded += 1;
    }
    try testing.expectEqual(@as(usize, 1), upgraded); // only the genuine outlier
}

test "the emitted scale ignores tensors the converter protects structurally" {
    // Measured trap: krea2's most damaging tensors are all `keys_hiprec`, and
    // `assignTensorType` protects them *before* it ever reads a score. Letting them
    // set the scale therefore steers on tensors the score cannot move, and the first
    // regenerated krea2 file came out as uniform-plus-one-tensor because of it.
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    // txtfusion (hiprec) is 8x the routable median; among the routable three, one is
    // 4x the median and should be the only one to earn an upgrade.
    const names = [_][]const u8{
        "pfx.txtfusion.lw.0.attn.wo.weight",
        "pfx.blocks.0.attn.wq.weight",
        "pfx.blocks.1.attn.wq.weight",
        "pfx.blocks.2.attn.wq.weight",
    };
    const dmg = [_]f64{ 2.4e-2, 3.0e-3, 3.0e-3, 1.2e-2 };
    var arms: [4]TensorArm = undefined;
    var pts: [4][1]Point = undefined;
    for (dmg, 0..) |d, i| {
        pts[i] = .{.{ .step = 0, .sigma = 1.0, .rel_l2 = d, .cos = 1, .mean_pos_cos = 1, .max_pos_rel = d }};
        arms[i] = .{ .name = names[i], .rows = 8, .cols = 8, .kind = .quantized, .points = &pts[i], .steps = 1 };
    }
    var report: TensorReport = .{
        .arena = std.heap.ArenaAllocator.init(gpa),
        .reference = "r", .format = .q4_k, .convrot_group = 256, .patch_dtype = .f32,
        .prompts = &.{}, .sigmas = &.{ 1.0, 0.0 }, .resolution = .{ 256, 256 },
        .seed = 0, .backend = "cpu", .controls = &.{}, .arms = arms[0..],
    };
    defer report.arena.deinit();
    const thr = LadderScore.upgradeThreshold(50, LadderScore.default_ladder_levels).?;

    // With the arch known, the routable median is 3.0e-3 and the 1.2e-2 tensor is two
    // doublings above it — enough to be upgraded.
    var buf: [2048]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeSensitivitiesJson(gpa, &w, &report, "pfx.", imagearch.byName("krea2").?);
    var with = try std.json.parseFromSlice(std.json.Value, gpa, w.buffered(), .{});
    defer with.deinit();
    try testing.expect(with.value.object.get("blocks.2.attn.wq.weight").?.float >= thr);

    // Without it the hiprec tensor joins the population, the median moves to 7.5e-3,
    // and the tensor that deserved protection no longer gets it.
    var w2 = std.Io.Writer.fixed(&buf);
    try writeSensitivitiesJson(gpa, &w2, &report, "pfx.", null);
    var without = try std.json.parseFromSlice(std.json.Value, gpa, w2.buffered(), .{});
    defer without.deinit();
    try testing.expect(without.value.object.get("blocks.2.attn.wq.weight").?.float < thr);
}

test "patchBytes: f32 is exact, bf16 is bf16-rounded and nothing worse" {
    // The bf16 patch dtype exists only so this arm can run on a GPU, so its cost has
    // to be exactly bf16 rounding — not, say, a truncation or a byte-order slip that
    // would show up as a plausible extra error in every arm.
    const gpa = testing.allocator;
    const vals = [_]f32{ 1.0, -2.5, 0.125, 42.0, 1e-8, -3.7e5, 0.0 };

    const as_f32 = try patchBytes(gpa, .f32, &vals);
    defer as_f32.deinit(gpa);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&vals), as_f32.bytes);
    // ⚠️ **The patch must OWN its bytes, for both dtypes.** This test asserted the
    // opposite until 2026-08-02 — `owned == null` and pointer identity with the
    // caller's buffer — which was the contract of the borrowed-slice bug documented
    // in `patchBytes`: the quantized arm frees its round-trip buffer before the
    // forward, so a borrowing patch pointed at freed memory and reported rel L2 ~1.0
    // for q4_k and q8_0 alike. The implementation was fixed and the test was not, so
    // the suite has been red ever since — which is its own lesson, since a red test
    // nobody reads is a test that cannot pin anything.
    try testing.expect(as_f32.owned != null);
    try testing.expect(as_f32.bytes.ptr != std.mem.sliceAsBytes(&vals).ptr);

    const bf = try patchBytes(gpa, .bf16, &vals);
    defer bf.deinit(gpa);
    const as_bf16 = bf.bytes;
    try testing.expect(bf.owned != null);
    try testing.expectEqual(vals.len * 2, as_bf16.len);
    const back = try gpa.alloc(f32, vals.len);
    defer gpa.free(back);
    tp.dtype.bf16ToF32Row(as_bf16, back, 1.0);
    for (vals, back) |want, got| {
        try testing.expectEqual(tp.dtype.bf16ToF32(tp.dtype.f32ToBf16(want)), got);
    }
    // And the relative cost is small enough that it cannot be confused with a
    // format's error (bf16 keeps 8 mantissa bits).
    for (vals, back) |want, got| {
        if (want == 0) continue;
        try testing.expect(@abs((got - want) / want) < 0.01);
    }
}

test "an unmeasurable tensor is left out of the ranking, never ranked safest" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    var pts = [_]Point{.{ .step = 0, .sigma = 1.0, .rel_l2 = 0.02, .cos = 0.999, .mean_pos_cos = 0.999, .max_pos_rel = 0.1 }};
    var big = [_]Point{.{ .step = 0, .sigma = 1.0, .rel_l2 = 0.05, .cos = 0.99, .mean_pos_cos = 0.99, .max_pos_rel = 0.2 }};
    const arms = [_]TensorArm{
        .{ .name = "small", .rows = 2, .cols = 2, .kind = .quantized, .points = &pts, .steps = 1 },
        .{ .name = "unencodable", .rows = 6144, .cols = 64, .kind = .quantized, .points = &.{}, .steps = 1, .skipped = "Q4_K cannot encode 6144x64" },
        .{ .name = "big", .rows = 2, .cols = 2, .kind = .quantized, .points = &big, .steps = 1 },
    };
    const order = try rankArms(arena.allocator(), &arms);
    try testing.expectEqual(@as(usize, 2), order.len);
    try testing.expectEqualStrings("big", arms[order[0]].name);
    try testing.expectEqualStrings("small", arms[order[1]].name);
}

test "the default schedule is the one renders use" {
    // A different shift is a different trajectory, so a level-2 number taken on one
    // is not comparable to a level-3 verdict taken on the other. This caught a
    // hardcoded 3.0 here against TensorPencil's 1.15 — the first run of this
    // harness measured sigmas no render ever visits.
    const gpa = testing.allocator;
    const o: Options = .{
        .io = testing.io,
        .reference = "r",
        .candidates = &.{"c"},
        .text_encoder_path = "te",
        .vae_path = "vae",
        .prompts = &.{"p"},
        .steps = 8,
    };
    const ours = try tp.pipeline.schedule(gpa, o.steps, o.shift);
    defer gpa.free(ours);
    const theirs = try tp.pipeline.schedule(gpa, o.steps, tp.sampler.default_shift);
    defer gpa.free(theirs);
    try testing.expectEqualSlices(f32, theirs, ours);
    // ...and it must actually descend into the clean regime, or every point sits in
    // the noisy end and the per-sigma curve says nothing about where image detail
    // is decided.
    try testing.expect(ours[o.steps - 1] < 0.4);
}
