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
//!     Ŷ_i = X · Ŵ_iᵀ      IMATRIX arm — as the format arm, but quantized with the
//!                         layer's own activation energy steering ggml's scale
//!                         search (plan §8A)
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
    imatrix_arm: ?ArmMetrics = null,
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
    imatrix_arms: usize,
    /// Layers that got a score — i.e. that had a reference-format measurement.
    scored: usize,

    pub fn deinit(self: *Report) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

/// Turn measured error into the 1–100 scale `Convert.zig` consumes.
///
/// Percentile rank, not a normalized magnitude: the converter treats the score
/// as a *relative* position, and rel-L2 across layers has a long tail, so a
/// linear map would compress almost everything into a narrow band and let two
/// outliers set the scale. Ranking is invariant to that.
fn scoreLayers(gpa: std.mem.Allocator, layers: []LayerResult, reference: Format) !usize {
    // Only layers that actually have a reference-format measurement take part.
    var order: std.ArrayList(usize) = .empty;
    defer order.deinit(gpa);
    for (layers, 0..) |*l, i| {
        l.score = null;
        if (l.find(reference) != null) try order.append(gpa, i);
    }
    if (order.items.len == 0) return 0;
    if (order.items.len == 1) {
        layers[order.items[0]].score = 50;
        return 1;
    }

    const Ctx = struct {
        layers: []const LayerResult,
        reference: Format,
        fn err(self: @This(), i: usize) f64 {
            return self.layers[i].find(self.reference).?.format_arm.rel_l2;
        }
        fn lessThan(self: @This(), a: usize, b: usize) bool {
            return self.err(a) < self.err(b);
        }
    };
    const ctx = Ctx{ .layers = layers, .reference = reference };
    std.mem.sort(usize, order.items, ctx, Ctx.lessThan);

    // Ties must not get different scores just because the sort put one first:
    // equal error means equal sensitivity, and a converter routing them
    // differently on sort order would be nondeterministic in effect.
    const ranked = order.items;
    const denom: f64 = @floatFromInt(ranked.len - 1);
    var i: usize = 0;
    while (i < ranked.len) {
        var j = i + 1;
        while (j < ranked.len and ctx.err(ranked[j]) == ctx.err(ranked[i])) j += 1;
        const mid = (@as(f64, @floatFromInt(i)) + @as(f64, @floatFromInt(j - 1))) / 2;
        const score = 1 + 99 * (mid / denom);
        for (ranked[i..j]) |idx| layers[idx].score = score;
        i = j;
    }
    return ranked.len;
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
    imatrix_arm: bool = true,
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
    var imatrix_arms: usize = 0;

    // Per-column activation energy, keyed as the converter keys it. Built once;
    // every layer's vector is a view into it.
    var imat: ?Imatrix.Imatrix = if (opts.imatrix_arm) try Imatrix.fromCache(gpa, &cache) else null;
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

        const per_format = try arena.alloc(FormatResult, opts.formats.len);
        var kept: usize = 0;

        for (opts.formats) |fmt| {
            const w_hat = ph.roundtrip(fmt, gpa, w, rows, cols, &pool) catch |err| {
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

            const imatrix_arm: ?ArmMetrics = if (imat) |*im|
                imatrixArm(gpa, io, im, name, w, x, y_apx, y_ref, m, rows, cols, fmt, &pool) catch |err| blk: {
                    std.log.warn("{s} / {s}: imatrix arm failed ({t}); omitted", .{ name, formatName(fmt), err });
                    break :blk null;
                }
            else
                null;
            if (imatrix_arm != null) imatrix_arms += 1;

            per_format[kept] = .{
                .fmt = fmt,
                .bits = bitsFor(fmt),
                .format_arm = format_arm,
                .kernel_arm = kernel_arm,
                .imatrix_arm = imatrix_arm,
            };
            kept += 1;
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

    const scored = try scoreLayers(gpa, results.items, opts.reference_format);
    if (scored < results.items.len) {
        std.log.warn(
            "{d} of {d} layers could not be measured at the reference format {s} (shape constraints); " ++
                "they are left out of the sensitivities JSON rather than scored as insensitive",
            .{ results.items.len - scored, results.items.len, formatName(opts.reference_format) },
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
        .imatrix_arms = imatrix_arms,
        .scored = scored,
    };
}

/// The format arm re-run with this layer's activation energy as ggml's
/// `imatrix`. Returns null when there is nothing to measure: an encoder that
/// ignores the weights, a layer the cache has no usable statistics for, or a row
/// width that is not a whole number of blocks.
///
/// Deliberately gated on `Imatrix.readsImatrix` (does ggml consume it) rather
/// than `usesImatrix` (do we ship it): the point of a measurement arm is to be
/// able to check the policy, and q2_k is currently excluded by that policy on
/// synthetic evidence. Measuring it here is how that gets confirmed or overturned
/// on real data.
fn imatrixArm(
    gpa: std.mem.Allocator,
    io: std.Io,
    im: *const Imatrix.Imatrix,
    layer: []const u8,
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
    const dst = ph.ggufDstType(fmt) orelse return null;
    const gt = gguf.GgmlType.fromString(@tagName(dst)) catch return null;
    if (!Imatrix.readsImatrix(gt)) return null;

    // Cache keys carry the checkpoint's container prefix; the map is keyed the
    // way the converter looks tensors up.
    const weights = im.get(imagearch.stripPrefix(layer)) orelse return null;
    if (weights.len != cols) return null;
    const block = gt.getBlockSize();
    if (block == 0 or cols % block != 0) return null;

    const bytes = try DataTransform.Quantizer.convertTensorDataWeighted(
        gpa,
        std.mem.sliceAsBytes(w),
        .f32,
        dst,
        @intCast(rows * cols),
        pool,
        weights,
    );
    defer gpa.free(bytes);

    const back = try DataTransform.Quantizer.convertTensorData(
        gpa,
        bytes,
        dst,
        .f32,
        @intCast(rows * cols),
        pool,
    );
    defer gpa.free(back);
    const w_hat: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, back));

    try tp.ops.matmul.matmul(io, gpa, y_scratch, x, m, tp.ops.matmul.Weight.fromF32(w_hat, rows, cols), null);
    return armMetrics(y_ref, y_scratch, m, rows);
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
    var written: usize = 0;
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
            if (f.imatrix_arm) |i| try writeCsvRow(w, l, f, "imatrix", i);
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
    try w.print("- sensitivity score: percentile rank of {s} format-arm rel-L2\n", .{formatName(report.reference_format)});
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

    if (report.imatrix_arms > 0) try writeImatrixSection(w, report);

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

/// The activation-aware verdict: does feeding ggml the captured per-channel
/// energy actually reduce real output error, per format?
///
/// Paired over the layers that have both arms, and reporting how many layers went
/// each way — a mean that improves while most layers get worse is a different
/// finding from one that improves everywhere, and only the win/loss counts
/// separate them.
fn writeImatrixSection(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("\n## Activation-aware quantization (imatrix), averaged over layers\n\n");
    try w.writeAll(
        "Same quantity as the format arm — real output error on captured activations — with the\n" ++
            "layer's per-channel energy passed to ggml as `quant_weights`. Ratio below 1.0 means the\n" ++
            "imatrix helped. `convert --calib` applies this to every format marked **shipped**.\n\n",
    );
    try w.writeAll("| format | layers | format rel-L2 | imatrix rel-L2 | ratio | layers improved | shipped |\n");
    try w.writeAll("|---|---:|---:|---:|---:|---:|:--:|\n");

    for (ph.formats) |spec| {
        var n: usize = 0;
        var sum_f: f64 = 0;
        var sum_i: f64 = 0;
        var better: usize = 0;
        for (report.layers) |l| {
            const f = l.find(spec.fmt) orelse continue;
            const im = f.imatrix_arm orelse continue;
            n += 1;
            sum_f += f.format_arm.rel_l2;
            sum_i += im.rel_l2;
            if (im.rel_l2 < f.format_arm.rel_l2) better += 1;
        }
        if (n == 0) continue;
        const fn_: f64 = @floatFromInt(n);
        const mf = sum_f / fn_;
        const mi = sum_i / fn_;
        const shipped = if (ph.ggufDstType(spec.fmt)) |dst| blk: {
            const gt = gguf.GgmlType.fromString(@tagName(dst)) catch break :blk false;
            break :blk Imatrix.usesImatrix(gt);
        } else false;
        try w.print("| {s} | {d} | {e:.4} | {e:.4} | {d:.4} | {d}/{d} | {s} |\n", .{
            spec.name, n, mf, mi, if (mf > 0) mi / mf else 1.0, better, n, if (shipped) "yes" else "no",
        });
    }

    try w.writeAll(
        "\n> A format measured here but marked **not shipped** is one ggml would accept an imatrix for\n" ++
            "> while `Imatrix.usesImatrix` withholds it. Today that is Q2_K only, excluded on synthetic\n" ++
            "> weighted-error evidence; this table is the real-data measurement that should confirm or\n" ++
            "> overturn that call.\n",
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

test "scores are percentile ranks, and ties share a score" {
    const gpa = testing.allocator;

    fn_scope: {
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
        const scored = try scoreLayers(gpa, &layers, .q4_k);

        try testing.expectEqual(@as(usize, 4), scored);
        try testing.expectApproxEqAbs(@as(f64, 1), layers[0].score.?, 1e-9); // lowest error
        try testing.expectApproxEqAbs(@as(f64, 100), layers[3].score.?, 1e-9); // highest
        // The tied pair sits at the average of ranks 1 and 2 — and crucially gets
        // the SAME score, so routing does not depend on sort order.
        try testing.expectApproxEqAbs(layers[1].score.?, layers[2].score.?, 1e-12);
        try testing.expectApproxEqAbs(@as(f64, 1 + 99 * 1.5 / 3.0), layers[1].score.?, 1e-9);
        break :fn_scope;
    }
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
    const scored = try scoreLayers(gpa, &layers, .q4_k);

    try testing.expectEqual(@as(usize, 2), scored);
    try testing.expect(layers[1].score == null);
    // The two that were measured still span the full range between them.
    try testing.expectApproxEqAbs(@as(f64, 1), layers[0].score.?, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 100), layers[2].score.?, 1e-9);

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
        .imatrix_arms = 0,
        .scored = scored,
    };
    defer report.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try writeSensitivitiesJson(&report, &aw.writer);
    const json = aw.written();
    try testing.expect(std.mem.indexOf(u8, json, "first.weight") == null);
    try testing.expect(std.mem.indexOf(u8, json, "\"a\"") != null);

    // It must still be valid JSON of the shape Convert.zig consumes.
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 2), parsed.value.object.count());
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
    const scored = try scoreLayers(gpa, &layers, .q4_k);

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
        .imatrix_arms = 0,
        .scored = scored,
    };
    defer report.deinit();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try writeSensitivitiesJson(&report, &aw.writer);

    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, aw.written(), .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expectEqual(@as(usize, 2), obj.count());
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
