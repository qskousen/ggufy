//! **The free half of the routing question: how much does per-tensor quantization
//! error VARY within one checkpoint?** Weight space only, no inference, no
//! calibration cache — so it runs on every architecture on disk today, including the
//! ones TensorPencil cannot execute.
//!
//! The krea2 verdict turned on one number. Among the 224 tensors `convert` can route,
//! per-tensor *damage* spans **2.3× around the median with no long tail**, and
//! uniform allocation is provably optimal when sensitivities are equal — which is why
//! three independent scorings all landed exactly on the uniform rate–distortion curve
//! and none beat it. Whether that is a fact about **krea2** or about **diffusion
//! transformers** is the open question, and answering it properly needs a working
//! forward pass per architecture (a calibration cache for level 1, teacher-forced
//! trajectories for level 2 — TensorPencil item 22, an XL each). This module is what
//! is available *before* paying that: the spread of the one factor a checkpoint alone
//! can reveal.
//!
//! ⚠️ **It can promote an architecture; it cannot rule one out.** A layer's
//! whole-model damage is roughly its weight error times how much the network
//! amplifies that error downstream, and this sees only the first factor. So a **wide**
//! spread proves heterogeneity exists and makes an architecture worth the expensive
//! measurement; a **narrow** one proves nothing, because all of the variation could
//! live in the amplification. Reporting a narrow spread as "routing has no room here"
//! would be exactly the mistake this programme keeps catching — a proxy quoted past
//! what it measures. The one architecture where both numbers are known (krea2)
//! calibrates how much the screen is worth; that comparison belongs in
//! `ACTIVATION_AWARE_PLAN.md`, not in this file's docstring, because it is data.
//!
//! ⚠️ **It emits no JSON, deliberately.** A weight-space error ranking is precisely
//! the kind of plausible-looking file that would get copied into
//! `src/sensitivities/` and shipped. Level 1's *activation*-based ranking already
//! agrees with per-tensor whole-model damage at only Spearman 0.36; a weight-only one
//! has strictly less information. This module prints a report and nothing else.
//!
//! What it measures, precisely: for every tensor in the **routable population** — the
//! ones whose sensitivity score `Convert.assignTensorType` would actually read, i.e.
//! rank-2, above the element threshold, not `keys_hiprec`, not upcast-forced — the
//! relative Frobenius error of a quantize→dequantize round trip at one format. Then
//! the *dispersion* of that population: its median, its ratio to the median at each
//! decile, and the tensors at the top.

const std = @import("std");
const tp = @import("TensorPencil");
const ph = @import("precision_harness.zig");
const metrics = @import("PrecisionMetrics.zig");
const imagearch = @import("ImageArch.zig");
const LadderScore = @import("LadderScore.zig");
const ThreadPool = @import("ThreadPool.zig").ThreadPool;
const cb = @import("callbacks.zig");

pub const Format = ph.Format;

/// `Convert.QUANTIZATION_THRESHOLD`'s default, duplicated rather than imported to
/// keep this module off `Convert.zig`'s dependency web (it needs no build options and
/// no write path). An architecture's own `threshhold` overrides it, same as there.
pub const default_threshold: u64 = 256 * 256;

pub const Options = struct {
    model_path: []const u8,
    format: Format = .q4_k,
    /// ConvRot group for the rotating formats, matching what `convert` ships.
    convrot_group: usize = 256,
    threads: usize = 0,
    /// Progress line per tensor, for a 26 GB checkpoint where silence is
    /// indistinguishable from a hang.
    log: ?*std.Io.Writer = null,
    callbacks: cb.ConvertCallbacks = .{},
};

pub const TensorError = struct {
    name: []const u8,
    rows: usize,
    cols: usize,
    params: u64,
    /// Relative Frobenius error of the round trip, ‖W − Ŵ‖ / ‖W‖.
    rel_frob: f64,
};

pub const Report = struct {
    arena: std.heap.ArenaAllocator,
    model_path: []const u8,
    arch: []const u8,
    format: Format,
    /// Every routable tensor that the format could encode, ascending by error.
    routable: []TensorError,
    /// Tensors the format cannot encode at their shape. Unmeasured, never zero —
    /// krea2's patch embed (6144×64) is the real case, and it is also the single most
    /// damaging tensor in that model.
    unencodable: usize,
    /// Tensors excluded because `Convert` would not read their score anyway.
    protected: usize,
    /// Tensors excluded as too small to quantize.
    subthreshold: usize,

    pub fn deinit(self: *Report) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Ratio of the `q`-quantile error to the median. 1.0 means "the same as
    /// typical"; this is the dispersion measure the routing question turns on.
    pub fn quantileOverMedian(self: *const Report, q: f64) f64 {
        if (self.routable.len == 0) return 1;
        const med = self.quantile(0.5);
        if (!(med > 0)) return 1;
        return self.quantile(q) / med;
    }

    pub fn quantile(self: *const Report, q: f64) f64 {
        if (self.routable.len == 0) return 0;
        if (self.routable.len == 1) return self.routable[0].rel_frob;
        const pos = q * @as(f64, @floatFromInt(self.routable.len - 1));
        const lo: usize = @intFromFloat(@floor(pos));
        const hi = @min(lo + 1, self.routable.len - 1);
        const frac = pos - @floor(pos);
        return self.routable[lo].rel_frob * (1 - frac) + self.routable[hi].rel_frob * frac;
    }

    /// How many routable tensors `LadderScore`'s encoding would upgrade off the target
    /// type at `aggressiveness`, **if** weight-space error were the damage — which it
    /// is not. Reported because it is the concrete form of "is there anything here to
    /// route", and it is the number that would be 0 on a homogeneous architecture.
    pub fn wouldUpgrade(self: *const Report, gpa: std.mem.Allocator, aggressiveness: f64, levels: usize) !usize {
        const thr = LadderScore.upgradeThreshold(aggressiveness, levels) orelse return 0;
        const errs = try gpa.alloc(f64, self.routable.len);
        defer gpa.free(errs);
        for (self.routable, 0..) |t, i| errs[i] = t.rel_frob;
        const ladder = (try LadderScore.fromDamages(gpa, errs)) orelse return 0;
        var n: usize = 0;
        for (errs) |e| if (ladder.score(e) >= thr) {
            n += 1;
        };
        return n;
    }
};

/// Screen one checkpoint. Safetensors only: a GGUF input is already quantized, so a
/// round trip through it would measure the *second* quantization and not the first.
pub fn run(gpa: std.mem.Allocator, io: std.Io, opts: Options) !Report {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();

    var ck = try tp.safetensors.SafeTensors.open(gpa, io, opts.model_path);
    defer ck.deinit();

    const arch = imagearch.detectArch(ck.names());
    const threshold: u64 = if (arch) |a| (a.threshhold orelse default_threshold) else default_threshold;

    var pool: ThreadPool = undefined;
    const n_jobs = if (opts.threads == 0) @max(1, std.Thread.getCpuCount() catch 1) else opts.threads;
    try pool.init(.{ .allocator = gpa, .n_jobs = n_jobs });
    defer pool.deinit();

    var out: std.ArrayList(TensorError) = .empty;
    var unencodable: usize = 0;
    var protected: usize = 0;
    var subthreshold: usize = 0;

    for (ck.names()) |name| {
        if (opts.callbacks.isCancelled()) return error.Canceled;

        const view = ck.get(name) orelse continue;
        const shape = view.info.shape.slice();
        // ⚠️ **Rank 3 and 4 count.** `Convert.assignTensorType` keeps only rank ≤ 1
        // float structurally; a UNet's conv weights are rank-4 `[out, in, kh, kw]` and
        // are quantized over their flat element count, exactly like a matrix. Screening
        // rank-2 only would have described SD1.5 by its 144 attention/FF linears while
        // ignoring the convolutions that hold most of its parameters — a report about
        // the wrong model. Rows are the outermost dim, matching how ggml tiles a row.
        if (shape.len < 2) continue;
        if (isEmbeddingWeight(name)) continue;

        const rows = shape[0];
        var cols: usize = 1;
        for (shape[1..]) |d| cols *= d;
        const params: u64 = @as(u64, rows) * @as(u64, cols);
        if (params < threshold) {
            subthreshold += 1;
            continue;
        }
        if (!LadderScore.isRoutable(arch, name)) {
            protected += 1;
            continue;
        }

        // A tensor whose dtype is not float at all (SD checkpoints carry
        // `position_ids` as i64) is not a quantization candidate and must not abort
        // the screen. Counted as unencodable so the report's tensor accounting stays
        // complete rather than silently short.
        const w = view.toF32Alloc(gpa) catch |err| {
            if (err == error.UnsupportedDType) {
                unencodable += 1;
                continue;
            }
            return err;
        };
        defer gpa.free(w);
        const w_hat = ph.roundtripGroup(opts.format, gpa, w, rows, cols, &pool, opts.convrot_group) catch {
            unencodable += 1;
            continue;
        };
        defer gpa.free(w_hat);

        const m = metrics.compute(w, w_hat);
        try out.append(arena, .{
            .name = try arena.dupe(u8, name),
            .rows = rows,
            .cols = cols,
            .params = params,
            .rel_frob = m.rel_frob_err,
        });
        if (opts.log) |lw| {
            try lw.print("[{d}] {s} ({d}x{d}) rel_frob {e:.4}\n", .{ out.items.len, name, rows, cols, m.rel_frob_err });
            try lw.flush();
        }
    }

    const items = out.items;
    std.mem.sort(TensorError, items, {}, struct {
        fn lt(_: void, a: TensorError, b: TensorError) bool {
            return a.rel_frob < b.rel_frob;
        }
    }.lt);

    return .{
        .arena = arena_state,
        .model_path = try arena.dupe(u8, opts.model_path),
        .arch = try arena.dupe(u8, if (arch) |a| a.name else "unknown"),
        .format = opts.format,
        .routable = items,
        .unencodable = unencodable,
        .protected = protected,
        .subthreshold = subthreshold,
    };
}

/// Mirrors `Convert.isEmbeddingWeight`'s intent: an `nn.Embedding` table is a lookup
/// table, not a matmul weight, and is never quantized whatever its score. Kept local
/// (and narrow) rather than exported from `Convert.zig`, because this module has no
/// business growing a dependency on the write path.
fn isEmbeddingWeight(name: []const u8) bool {
    const needles = [_][]const u8{
        "token_embd", ".embed.weight", "embed_tokens", "token_embedding", ".wte.", "position_embedding",
    };
    for (needles) |n| if (std.mem.indexOf(u8, name, n) != null) return true;
    return false;
}

pub fn writeMarkdown(gpa: std.mem.Allocator, report: *const Report, w: *std.Io.Writer, top_n: usize) !void {
    try w.print("# Heterogeneity screen — weight-space {s} error\n\n", .{ph.formats[@intFromEnum(report.format)].name});
    try w.print("- model: `{s}`\n", .{report.model_path});
    try w.print("- arch: `{s}`\n", .{report.arch});
    try w.print(
        "- routable tensors measured: **{d}** ({d} structurally protected, {d} sub-threshold, {d} unencodable at this format)\n",
        .{ report.routable.len, report.protected, report.subthreshold, report.unencodable },
    );
    if (report.routable.len == 0) {
        try w.writeAll("\n**Nothing routable was measurable, so there is nothing to say about dispersion.**\n");
        return;
    }

    const med = report.quantile(0.5);
    try w.print("- median error: {e:.4}\n\n", .{med});

    try w.writeAll("## Dispersion of the routable population\n\n");
    try w.writeAll("| quantile | error | × median |\n|---|---:|---:|\n");
    for ([_]f64{ 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 }) |q| {
        try w.print("| p{d:.0} | {e:.4} | {d:.2}× |\n", .{ q * 100, report.quantile(q), report.quantileOverMedian(q) });
    }
    // The two numbers the krea2 comparison is made on: max/median (what the encoding
    // normalizes against) and p90/p10 (robust to a single outlier at either end).
    const p10 = report.quantile(0.1);
    try w.print(
        "\n**max/median {d:.2}×**, p90/p10 {d:.2}×.\n",
        .{ report.quantileOverMedian(1.0), if (p10 > 0) report.quantile(0.9) / p10 else 0 },
    );
    const up50 = try report.wouldUpgrade(gpa, 50, LadderScore.default_ladder_levels);
    const up33 = try report.wouldUpgrade(gpa, 33, LadderScore.default_ladder_levels);
    try w.print(
        "\nIf weight-space error *were* the damage (it is not — see this module's header), `LadderScore`\n" ++
            "would upgrade **{d} of {d}** routable tensors at `-a 50`, and {d} at `-a 33`.\n",
        .{ up50, report.routable.len, up33 },
    );

    try w.print("\n## Highest-error routable tensors\n\n", .{});
    try w.writeAll("| tensor | shape | params | error | × median |\n|---|---|---:|---:|---:|\n");
    const n = @min(top_n, report.routable.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const t = report.routable[report.routable.len - 1 - i];
        try w.print("| `{s}` | {d}×{d} | {d:.1}M | {e:.4} | {d:.2}× |\n", .{
            t.name, t.rows, t.cols, @as(f64, @floatFromInt(t.params)) / 1e6, t.rel_frob, if (med > 0) t.rel_frob / med else 0,
        });
    }
}

pub fn writeCsv(report: *const Report, w: *std.Io.Writer) !void {
    try w.writeAll("tensor,rows,cols,params,rel_frob\n");
    for (report.routable) |t| {
        try w.print("{s},{d},{d},{d},{e:.8}\n", .{ t.name, t.rows, t.cols, t.params, t.rel_frob });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn fakeReport(gpa: std.mem.Allocator, errs: []const f64) !Report {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    const arena = arena_state.allocator();
    const items = try arena.alloc(TensorError, errs.len);
    for (errs, 0..) |e, i| items[i] = .{
        .name = try std.fmt.allocPrint(arena, "blocks.{d}.attn.wq.weight", .{i}),
        .rows = 8,
        .cols = 8,
        .params = 64,
        .rel_frob = e,
    };
    std.mem.sort(TensorError, items, {}, struct {
        fn lt(_: void, a: TensorError, b: TensorError) bool {
            return a.rel_frob < b.rel_frob;
        }
    }.lt);
    return .{
        .arena = arena_state,
        .model_path = "m",
        .arch = "krea2",
        .format = .q4_k,
        .routable = items,
        .unencodable = 0,
        .protected = 0,
        .subthreshold = 0,
    };
}

test "dispersion is reported relative to the median, not to the extremes" {
    const gpa = testing.allocator;
    // A tight population plus one outlier: the median must not move much, which is
    // the whole reason the ratio is taken against it.
    var r = try fakeReport(gpa, &.{ 1.0e-2, 1.1e-2, 1.2e-2, 1.3e-2, 8.0e-2 });
    defer r.deinit();

    try testing.expectApproxEqAbs(@as(f64, 1.2e-2), r.quantile(0.5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), r.quantileOverMedian(0.5), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 8.0e-2 / 1.2e-2), r.quantileOverMedian(1.0), 1e-9);
    // p0 sits below the median, so its ratio is < 1 — the table is two-sided.
    try testing.expect(r.quantileOverMedian(0.0) < 1);
}

test "a homogeneous population would route nothing, a heterogeneous one would" {
    const gpa = testing.allocator;
    // krea2-shaped: 2.3x around the median, no tail.
    var tight = try fakeReport(gpa, &.{ 2.0e-2, 2.4e-2, 2.8e-2, 3.2e-2, 4.6e-2 });
    defer tight.deinit();
    try testing.expectEqual(@as(usize, 0), try tight.wouldUpgrade(gpa, 50, LadderScore.default_ladder_levels));

    // Two decades of spread: the encoding fires, and hard.
    var wide = try fakeReport(gpa, &.{ 1.0e-3, 4.0e-3, 1.0e-2, 4.0e-2, 2.0e-1 });
    defer wide.deinit();
    try testing.expect(try wide.wouldUpgrade(gpa, 50, LadderScore.default_ladder_levels) >= 2);
}

test "an empty population says so instead of dividing by zero" {
    const gpa = testing.allocator;
    var r = try fakeReport(gpa, &.{});
    defer r.deinit();
    try testing.expectEqual(@as(f64, 0), r.quantile(0.5));
    try testing.expectEqual(@as(f64, 1), r.quantileOverMedian(1.0));
    try testing.expectEqual(@as(usize, 0), try r.wouldUpgrade(gpa, 50, LadderScore.default_ladder_levels));

    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeMarkdown(gpa, &r, &w, 10);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "nothing to say about dispersion") != null);
}

test "a rank-4 conv weight is screened as rows x (in*kh*kw), not skipped" {
    // The regression this pins: screening rank-2 only reported SD1.5 through its 144
    // attention/FF linears and ignored every convolution — most of the model. Rank is
    // not what `Convert` gates on (it gates on rank <= 1), so it must not gate here.
    const shape = [_]usize{ 320, 320, 3, 3 };
    var cols: usize = 1;
    for (shape[1..]) |d| cols *= d;
    try testing.expectEqual(@as(usize, 2880), cols);
    try testing.expectEqual(@as(usize, 320 * 2880), shape[0] * cols);
}

test "embedding tables and rank-1 tensors are out of the population by construction" {
    // Not a score decision: `Convert.assignTensorType` keeps these float before it
    // ever reads a sensitivity, so including them would report dispersion over
    // tensors no routing can touch — the normalization trap in another costume.
    try testing.expect(isEmbeddingWeight("model.diffusion_model.llm_adapter.embed.weight"));
    try testing.expect(isEmbeddingWeight("text_model.embed_tokens.weight"));
    try testing.expect(isEmbeddingWeight("cond_stage_model.transformer.text_model.embeddings.position_embedding.weight"));
    try testing.expect(!isEmbeddingWeight("blocks.0.attn.wq.weight"));
    try testing.expect(!isEmbeddingWeight("txtfusion.projector.weight"));
}
