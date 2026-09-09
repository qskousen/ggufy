//! AdaRound.zig — adaptive rounding, the gradient-based way of choosing a level.
//!
//! Same job as `Gptq.zig` and the same objective, `‖(Ŵ − W)X̃ᵗ‖²_F`: the grid is
//! already fixed by §8A, and all that is left is whether each weight goes up or
//! down. GPTQ answers that greedily — one column at a time, error pushed forward,
//! each choice final. This answers it by optimizing every weight jointly, so an
//! early decision can be revisited once a later one is known.
//!
//! That difference is the whole reason this file exists. "Both minimize the same
//! objective" does not make them the same algorithm: one is a single-pass
//! approximation and the other is not, and the literature routinely has adaptive
//! rounding beating GPTQ at equal width. Whether that survives to pixels on a
//! diffusion model is what the arm measures.
//!
//! ### The relaxation
//!
//! Per weight, with `t = w/s` the continuous level and `fl = ⌊t⌋`:
//!
//!     ŵ = s · (fl + h(V)),   h(V) = clip(σ(V)(ζ−γ) + γ, 0, 1),  ζ = 1.1, γ = −0.1
//!
//! `h` is a *rectified* sigmoid: the stretch past [0,1] is what keeps a gradient
//! alive at the extremes, where a plain sigmoid saturates and the weight can never
//! change its mind again. `V` is initialized so `h(V) = t − fl`, i.e. the arm
//! starts at the **unrounded** weight with zero reconstruction error, and the
//! regularizer
//!
//!     f_reg = Σ_j (1 − |2h_j − 1|^β),   β annealed 20 → 2
//!
//! is what drives it to a corner. Annealing β *down* widens the basin late, so
//! weights commit gradually instead of snapping in the first few steps.
//!
//! ⚠️ **`fl` is clamped to `[qlo, qhi−1]`, not `[qlo, qhi]`.** Then `fl + h` can
//! never leave the grid and the final hard round needs no second clamp — which
//! matters because a clamp there would silently move a weight the optimizer had
//! already priced.
//!
//! ### Why this is affordable without autograd
//!
//! The gradient is closed form. With `e_r` row *r*'s error,
//! `∂/∂e_r ‖e_r X̃ᵗ‖² = 2 (e_r X̃ᵗ) X̃`, which is two `m·cols` passes over the
//! sample — never the `cols × cols` Gram. So a step costs `O(m·cols)` per row,
//! the same order as one GPTQ column sweep, and `Gptq.Hessian.xt` already holds
//! `X̃ᵗ` in exactly the `[cols][m]` layout both passes want.
//!
//! The objective is **row-separable** (`‖E X̃ᵗ‖²_F = Σ_r ‖e_r X̃ᵗ‖²`), so rows are
//! optimized in parallel with no shared writes and no ordering effect.
//!
//! ### Calibration
//!
//! The reconstruction term is divided by the loss round-to-nearest achieves on the
//! same row, so it reads as "fraction of RTN's error" and is O(1) whatever the
//! layer's scale. Without that, `lambda_reg` would have to be retuned per layer
//! and a single global default would be meaningless.

const std = @import("std");
const DataTransform = @import("DataTransform.zig");
const thread_pool_mod = @import("ThreadPool.zig");
const Gptq = @import("Gptq.zig");
pub const tp = @import("TensorPencil");
const DeviceBuffer = tp.gpu.cuda.backend.DeviceBuffer;

const Q = DataTransform.Quantizer;
const ThreadPool = thread_pool_mod.ThreadPool;

/// Rectified-sigmoid stretch, from the paper. The interval must strictly contain
/// [0,1] or `h` saturates at the corners and a committed weight can never move.
pub const zeta: f32 = 1.1;
pub const gamma: f32 = -0.1;

pub const Params = struct {
    iters: usize = 200,
    /// Weight on the rounding regularizer, against a reconstruction term already
    /// normalized to RTN = 1.0.
    ///
    /// ⚠️ The regularizer is **per weight**, not averaged over the row. Averaging
    /// it divides each weight's rounding gradient by `cols`, which both pushes the
    /// useful range of this constant into the hundreds and makes it depend on layer
    /// width — fatal here, where widths run 1280 to 23040 in one model.
    lambda_reg: f32 = 1.0,
    /// ⚠️ Adam moves `V` by about `lr` per step regardless of gradient size, so
    /// `lr · iters` is the total distance a weight can travel. `V` spans roughly
    /// ±2.5 over the useful range of `h`, so the paper's 1e-3 needs its 10k
    /// iterations to get anywhere; at a few hundred it cannot flip a single
    /// weight and the arm silently degenerates to round-to-nearest.
    lr: f32 = 1.0,
    beta_start: f32 = 20.0,
    beta_end: f32 = 2.0,
    /// Fraction of the run with the regularizer off, so the reconstruction term
    /// picks a direction before anything is pushed to a corner.
    warmup: f32 = 0.2,
    stats: ?*Stats = null,
};

pub const Stats = struct {
    /// Rows where the optimized levels beat plain RTN on the row's own objective.
    rows_improved: usize = 0,
    rows_total: usize = 0,
    /// Σ over rows of the final objective, and of RTN's, both in the same units.
    /// Their ratio is the layer's in-sample gain — the number that says whether
    /// the optimizer did its job at all, separately from whether that helps.
    obj_final: f64 = 0,
    obj_rtn: f64 = 0,
};

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// `h(V)`, the rectified sigmoid.
fn hOf(v: f32) f32 {
    return std.math.clamp(sigmoid(v) * (zeta - gamma) + gamma, 0.0, 1.0);
}

/// `V` such that `h(V) = rest`. Inverse of `hOf` on the open interval; `rest` is
/// nudged off {0,1} because those map to ±inf.
fn vOf(rest: f32) f32 {
    const r = std.math.clamp(rest, 1e-4, 1.0 - 1e-4);
    const s = (r - gamma) / (zeta - gamma);
    return @log(s / (1.0 - s));
}

/// Choose levels for `w` by adaptive rounding. Returns the same `Gptq.Levels` the
/// GPTQ sweep does, so every consumer downstream is identical and the two can be
/// swapped with nothing else changing.
///
/// `weights` is §8A's per-column importance and, exactly as in `Gptq.sweep`, is
/// used **only** to pick the scale. Keeping it out of the objective is what makes
/// this arm "same grid, different level" rather than a second mechanism.
pub fn sweep(
    gpa: std.mem.Allocator,
    h: *const Gptq.Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    grid: Gptq.Grid,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) !Gptq.Levels {
    if (cols != h.cols) return error.ShapeMismatch;
    if (w.len != rows * cols) return error.InputSizeMismatch;
    if (rows == 0) return error.EmptyInput;
    if (p.iters == 0) return error.InvalidParams;
    if (h.basis.convrot and cols % 2 != 0) return error.ColsNotEven;

    // Rotate into the basis the rounding happens in, exactly as the GPTQ sweep
    // and the §8A search do.
    const work = try gpa.dupe(f32, w);
    defer gpa.free(work);
    if (h.basis.convrot) try Q.rotateGroupwiseInPlace(work, rows, cols, h.basis.group_size, pool);

    const rot_w = try Q.rotatedWeights(gpa, weights, cols, h.basis.convrot, h.basis.group_size);
    defer if (rot_w) |rw| gpa.free(rw);

    const scales = try gpa.alloc(f32, rows);
    errdefer gpa.free(scales);
    for (0..rows) |r| {
        scales[r] = Q.searchScale(work[r * cols ..][0..cols], rot_w, grid.qdiv, grid.qlo, grid.qhi);
    }

    const q = try gpa.alloc(f32, rows * cols);
    errdefer gpa.free(q);

    var acc = Accum{};
    if (gpu_backend) |be| {
        try sweepGpu(gpa, be, work, q, scales, rows, cols, h, grid, p, pool, &acc);
    } else {
        try forEachRowChunk(pool, rows, rowsJob, .{ work, q, scales, cols, h, grid, p, &acc });
    }

    if (p.stats) |o| o.* = .{
        .rows_improved = acc.rows_improved.load(.monotonic),
        .rows_total = rows,
        .obj_final = @bitCast(acc.obj_final.load(.monotonic)),
        .obj_rtn = @bitCast(acc.obj_rtn.load(.monotonic)),
    };
    return .{ .q = q, .scale = scales, .stats = .{} };
}

/// Device backend for the two GEMMs, or null for the CPU path. A global in the
/// house style of `ops.matmul.probe`, and under the same single-threaded rule.
///
/// ⚠️ **Only the GEMMs move.** They are 511/512 of the arithmetic here (two
/// `m·cols` passes per weight against one elementwise update), so the split buys
/// almost all of the device's advantage without a single new kernel. Moving the
/// update too would mean an AdaRound-shaped PTX/SPIR-V kernel in TP for the
/// remaining ~1/512 — worth it only if a capture-scale run ever becomes
/// PCIe-bound, which the numbers in `sweepGpu` say it is not.
pub var gpu_backend: ?*tp.gpu.cuda.Backend = null;

/// `y[m][rows] = x[m][cols] · Wᵗ` is exactly TP's `opMatmul`, and both of
/// AdaRound's GEMMs are that shape with the **activation sample as W**:
///
///     S = E · X̃ᵗ    W = X̃  `[m, cols]`     x = E `[rows, cols]` -> y `[rows, m]`
///     G = S · X̃     W = X̃ᵗ `[cols, m]`     x = S `[rows, m]`    -> y `[rows, cols]`
///
/// The sample is constant for the layer, so it is what the backend's weight cache
/// is *for* — uploaded once and reused across all `iters` steps, while only the
/// per-iteration `E`/`S` cross the bus. Reversing the roles (treating the changing
/// side as the weight) would re-upload it every step and be slower than the CPU.
fn sweepGpu(
    gpa: std.mem.Allocator,
    be: *tp.gpu.cuda.Backend,
    work: []const f32,
    q_out: []f32,
    scales: []const f32,
    rows: usize,
    cols: usize,
    h: *const Gptq.Hessian,
    grid: Gptq.Grid,
    p: Params,
    pool: *ThreadPool,
    acc: *Accum,
) !void {
    const m = h.m;

    // The sample in both orientations, f32 for the device GEMM. `xt` is already
    // `[cols][m]`, which is the second GEMM's weight verbatim.
    const w_ct = try gpa.alloc(f32, cols * m); // X̃ᵗ [cols, m]
    defer gpa.free(w_ct);
    const w_mc = try gpa.alloc(f32, m * cols); // X̃  [m, cols]
    defer gpa.free(w_mc);
    for (0..cols) |j| {
        const src = h.xt[j * m ..][0..m];
        for (0..m) |k| {
            const v: f32 = @floatCast(src[k]);
            w_ct[j * m + k] = v;
            w_mc[k * cols + j] = v;
        }
    }

    var d_e = try be.tensorCreate(rows * cols * @sizeOf(f32));
    defer be.tensorDestroy(&d_e);
    var d_s = try be.tensorCreate(rows * m * @sizeOf(f32));
    defer be.tensorDestroy(&d_s);

    const t = try gpa.alloc(f64, rows * cols);
    defer gpa.free(t);
    const fl = try gpa.alloc(f64, rows * cols);
    defer gpa.free(fl);
    const vv = try gpa.alloc(f64, rows * cols);
    defer gpa.free(vv);
    const am = try gpa.alloc(f64, rows * cols);
    defer gpa.free(am);
    const av = try gpa.alloc(f64, rows * cols);
    defer gpa.free(av);
    const host = try gpa.alloc(f32, rows * cols);
    defer gpa.free(host);
    const inv = try gpa.alloc(f64, rows);
    defer gpa.free(inv);
    const orn = try gpa.alloc(f64, rows);
    defer gpa.free(orn);
    const sbuf = try gpa.alloc(f32, rows * m);
    defer gpa.free(sbuf);

    for (0..rows) |r| {
        const sc: f64 = scales[r];
        for (0..cols) |j| {
            const i = r * cols + j;
            if (!(sc > 0)) {
                t[i] = 0;
                fl[i] = 0;
                vv[i] = 0;
                host[i] = 0;
                continue;
            }
            const tv = @as(f64, work[i]) / sc;
            t[i] = tv;
            fl[i] = std.math.clamp(@floor(tv), grid.qlo, grid.qhi - 1);
            vv[i] = vOf(@floatCast(tv - fl[i]));
            host[i] = @floatCast((std.math.clamp(@round(tv), grid.qlo, grid.qhi) - tv) * sc);
        }
    }
    @memset(am, 0);
    @memset(av, 0);

    // RTN's objective per row, through the same two GEMMs the loop uses, so the
    // yardstick and the thing measured against it share every rounding decision.
    try gemmPair(be, d_e, d_s, host, sbuf, rows, cols, m, w_mc, w_ct, false);
    for (0..rows) |r| {
        var t2: f64 = 0;
        for (sbuf[r * m ..][0..m]) |v| t2 += @as(f64, v) * v;
        orn[r] = t2 / @as(f64, @floatFromInt(m));
        inv[r] = if (orn[r] > 0) 1.0 / orn[r] else 1.0;
    }

    const warm: usize = @intFromFloat(@as(f32, @floatFromInt(p.iters)) * p.warmup);
    var lut: [pow_bins + 1]f64 = undefined;

    // ⚠️ Both host loops run across the pool. Left serial they cost more than the
    // GEMMs they feed, and moving the GEMMs to the device buys nothing at all:
    // measured, a single-threaded host step ran the whole conversion at the same
    // 0.5 GB/h as the 8-thread CPU path.
    var st = StepState{
        .host = host,
        .t = t,
        .fl = fl,
        .vv = vv,
        .am = am,
        .av = av,
        .scales = scales,
        .inv = inv,
        .cols = cols,
        .lut = &lut,
        .lr = p.lr,
    };

    var it: usize = 0;
    while (it < p.iters) : (it += 1) {
        try forEachRowChunk(pool, rows, errJob, .{&st});
        try gemmPair(be, d_e, d_s, host, sbuf, rows, cols, m, w_mc, w_ct, true);

        const beta: f64 = if (it < warm) 0 else blk: {
            const frac = @as(f64, @floatFromInt(it - warm)) /
                @as(f64, @floatFromInt(@max(1, p.iters - warm)));
            break :blk p.beta_start + (p.beta_end - p.beta_start) * frac;
        };
        st.beta = beta;
        st.lam = if (it < warm) 0 else p.lambda_reg;
        if (st.lam > 0) buildPowLut(&lut, beta - 1.0);
        st.bc1 = 1.0 - std.math.pow(f64, 0.9, @floatFromInt(it + 1));
        st.bc2 = 1.0 - std.math.pow(f64, 0.999, @floatFromInt(it + 1));

        try forEachRowChunk(pool, rows, stepJob, .{&st});
    }

    for (0..rows) |r| {
        const sc: f64 = scales[r];
        const dst = q_out[r * cols ..][0..cols];
        if (!(sc > 0)) {
            @memset(dst, 0);
            continue;
        }
        for (0..cols) |j| {
            const i = r * cols + j;
            dst[j] = @floatCast(fl[i] + @as(f64, if (hOf(@floatCast(vv[i])) >= 0.5) 1.0 else 0.0));
            host[i] = @floatCast((@as(f64, dst[j]) - t[i]) * sc);
        }
    }
    try gemmPair(be, d_e, d_s, host, sbuf, rows, cols, m, w_mc, w_ct, false);
    for (0..rows) |r| {
        if (!(scales[r] > 0)) continue;
        var t2: f64 = 0;
        for (sbuf[r * m ..][0..m]) |v| t2 += @as(f64, v) * v;
        const fin = t2 / @as(f64, @floatFromInt(m));
        if (fin < orn[r]) _ = acc.rows_improved.fetchAdd(1, .monotonic);
        addF64(&acc.obj_final, fin);
        addF64(&acc.obj_rtn, orn[r]);
    }
}

/// Per-iteration state shared by the two host passes. Bundled rather than passed
/// as fifteen arguments, and mutated only between passes — each pass owns a
/// disjoint row range, so nothing here is written by two threads.
const StepState = struct {
    host: []f32,
    t: []f64,
    fl: []f64,
    vv: []f64,
    am: []f64,
    av: []f64,
    scales: []const f32,
    inv: []const f64,
    cols: usize,
    lut: *const [pow_bins + 1]f64,
    lr: f32,
    beta: f64 = 0,
    lam: f64 = 0,
    bc1: f64 = 1,
    bc2: f64 = 1,
};

/// `e = (fl + h(V) − t)·s`, the input to the first GEMM.
fn errJob(s: *StepState, start: usize, end: usize) void {
    for (start..end) |r| {
        const sc: f64 = s.scales[r];
        if (!(sc > 0)) continue;
        for (0..s.cols) |j| {
            const i = r * s.cols + j;
            s.host[i] = @floatCast((s.fl[i] + hOf(@floatCast(s.vv[i])) - s.t[i]) * sc);
        }
    }
}

/// Chain the downloaded gradient through `h'`, add the rounding regularizer, and
/// take one Adam step on `V`.
fn stepJob(s: *StepState, start: usize, end: usize) void {
    for (start..end) |r| {
        const sc: f64 = s.scales[r];
        if (!(sc > 0)) continue;
        for (0..s.cols) |j| {
            const i = r * s.cols + j;
            var g: f64 = 2.0 * @as(f64, s.host[i]) * sc * s.inv[r];

            const sig = 1.0 / (1.0 + @exp(-s.vv[i]));
            const stretched = sig * (zeta - gamma) + gamma;
            const dh: f64 = if (stretched <= 0.0 or stretched >= 1.0)
                0.0
            else
                sig * (1.0 - sig) * (zeta - gamma);

            g *= dh;
            if (s.lam > 0) {
                const hv = std.math.clamp(stretched, 0.0, 1.0);
                const u = 2.0 * hv - 1.0;
                const au = @abs(u);
                if (au > 1e-6 and au < 1.0) {
                    const d = -2.0 * s.beta * powLut(s.lut, au) * std.math.sign(u);
                    g += s.lam * d * dh;
                }
            }

            s.am[i] = 0.9 * s.am[i] + 0.1 * g;
            s.av[i] = 0.999 * s.av[i] + 0.001 * g * g;
            s.vv[i] -= @as(f64, s.lr) * (s.am[i] / s.bc1) / (@sqrt(s.av[i] / s.bc2) + 1e-8);
        }
    }
}

/// Upload `E`, run both GEMMs, and read back either `S` (for an objective) or the
/// gradient `G` in place of `E` (for a step).
fn gemmPair(
    be: *tp.gpu.cuda.Backend,
    d_e: DeviceBuffer,
    d_s: DeviceBuffer,
    host: []f32,
    sbuf: []f32,
    rows: usize,
    cols: usize,
    m: usize,
    w_mc: []const f32,
    w_ct: []const f32,
    want_grad: bool,
) !void {
    try be.tensorUpload(d_e, std.mem.sliceAsBytes(host));
    // S[rows, m] = E[rows, cols] @ X̃ᵗ, with W = X̃ [m, cols].
    try be.opMatmul(d_s, 0, d_e, 0, rows, std.mem.sliceAsBytes(w_mc), false, m, cols, 1.0, null);
    if (!want_grad) {
        try be.tensorDownload(d_s, std.mem.sliceAsBytes(sbuf));
        return;
    }
    // G[rows, cols] = S[rows, m] @ X̃, with W = X̃ᵗ [cols, m]. Written over E,
    // which the next step overwrites anyway.
    try be.opMatmul(d_e, 0, d_s, 0, rows, std.mem.sliceAsBytes(w_ct), false, cols, m, 1.0, null);
    try be.tensorDownload(d_e, std.mem.sliceAsBytes(host));
}

/// `|2h−1|^(β−1)` is evaluated once per weight per iteration, which is millions of
/// `pow` calls a layer and measurably more than the Adam update around it. β is
/// fixed within an iteration, so one table serves the whole pass.
const pow_bins: usize = 4096;

fn buildPowLut(lut: *[pow_bins + 1]f64, e: f64) void {
    for (0..pow_bins + 1) |i| {
        const x = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(pow_bins));
        lut[i] = std.math.pow(f64, x, e);
    }
}

fn powLut(lut: *const [pow_bins + 1]f64, x: f64) f64 {
    const pos = x * @as(f64, @floatFromInt(pow_bins));
    const i: usize = @intFromFloat(@min(pos, @as(f64, @floatFromInt(pow_bins - 1))));
    const frac = pos - @as(f64, @floatFromInt(i));
    return lut[i] + (lut[i + 1] - lut[i]) * frac;
}

const Accum = struct {
    rows_improved: std.atomic.Value(usize) = .init(0),
    // f64 accumulators as raw bits, CAS-added: contended only once per row.
    obj_final: std.atomic.Value(u64) = .init(0),
    obj_rtn: std.atomic.Value(u64) = .init(0),
};

fn addF64(a: *std.atomic.Value(u64), v: f64) void {
    var cur = a.load(.monotonic);
    while (true) {
        const next: u64 = @bitCast(@as(f64, @bitCast(cur)) + v);
        cur = a.cmpxchgWeak(cur, next, .monotonic, .monotonic) orelse return;
    }
}

/// Rows processed together per pass over `X̃ᵗ`.
///
/// The optimizer is memory-bound, not flop-bound: one row's gradient touches the
/// whole `[cols][m]` sample twice, so doing rows one at a time re-reads it `rows`
/// times per iteration. Blocking amortizes each read over `block_rows` rows and is
/// worth ~5x at SDXL's shapes. Rows stay mathematically independent — this only
/// changes the traversal order.
const block_rows: usize = 8;

fn rowsJob(
    work: []const f32,
    q_out: []f32,
    scales: []const f32,
    cols: usize,
    h: *const Gptq.Hessian,
    grid: Gptq.Grid,
    p: Params,
    acc: *Accum,
    start: usize,
    end: usize,
) void {
    const m = h.m;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const B = block_rows;
    const t = a.alloc(f64, B * cols) catch return;
    const fl = a.alloc(f64, B * cols) catch return;
    const vv = a.alloc(f64, B * cols) catch return;
    const am = a.alloc(f64, B * cols) catch return;
    const av = a.alloc(f64, B * cols) catch return;
    const e = a.alloc(f64, B * cols) catch return;
    const sblk = a.alloc(f64, B * m) catch return;
    const sc = a.alloc(f64, B) catch return;
    const inv = a.alloc(f64, B) catch return;
    const orn = a.alloc(f64, B) catch return;

    const warm: usize = @intFromFloat(@as(f32, @floatFromInt(p.iters)) * p.warmup);

    var base = start;
    while (base < end) : (base += B) {
        const nb = @min(B, end - base);

        for (0..nb) |b| {
            const row = work[(base + b) * cols ..][0..cols];
            sc[b] = scales[base + b];
            if (!(sc[b] > 0)) {
                @memset(q_out[(base + b) * cols ..][0..cols], 0);
                sc[b] = 0;
                continue;
            }
            for (0..cols) |j| {
                const tv = @as(f64, row[j]) / sc[b];
                t[b * cols + j] = tv;
                fl[b * cols + j] = std.math.clamp(@floor(tv), grid.qlo, grid.qhi - 1);
                vv[b * cols + j] = vOf(@floatCast(tv - fl[b * cols + j]));
                am[b * cols + j] = 0;
                av[b * cols + j] = 0;
                const qr = std.math.clamp(@round(tv), grid.qlo, grid.qhi);
                e[b * cols + j] = (qr - tv) * sc[b];
            }
        }

        // RTN's objective per row: the yardstick the reconstruction term is
        // normalized by, and what the result must beat to mean anything.
        blockS(e, sblk, h, m, cols, nb);
        for (0..nb) |b| {
            var t2: f64 = 0;
            for (sblk[b * m ..][0..m]) |v| t2 += v * v;
            orn[b] = t2 / @as(f64, @floatFromInt(m));
            inv[b] = if (orn[b] > 0) 1.0 / orn[b] else 1.0;
        }

        var it: usize = 0;
        while (it < p.iters) : (it += 1) {
            for (0..nb) |b| {
                if (sc[b] == 0) continue;
                for (0..cols) |j| {
                    const hv: f64 = hOf(@floatCast(vv[b * cols + j]));
                    e[b * cols + j] = (fl[b * cols + j] + hv - t[b * cols + j]) * sc[b];
                }
            }
            blockS(e, sblk, h, m, cols, nb);

            const beta: f64 = if (it < warm) 0 else blk: {
                const frac = @as(f64, @floatFromInt(it - warm)) /
                    @as(f64, @floatFromInt(@max(1, p.iters - warm)));
                break :blk p.beta_start + (p.beta_end - p.beta_start) * frac;
            };
            const lam: f64 = if (it < warm) 0 else p.lambda_reg;
            const bc1 = 1.0 - std.math.pow(f64, 0.9, @floatFromInt(it + 1));
            const bc2 = 1.0 - std.math.pow(f64, 0.999, @floatFromInt(it + 1));

            // Second pass over the sample: grad_e = 2 (e X̃ᵗ) X̃, all rows at once.
            for (0..cols) |j| {
                const xj = h.xt[j * m ..][0..m];
                for (0..nb) |b| {
                    if (sc[b] == 0) continue;
                    const sb = sblk[b * m ..][0..m];
                    var g: f64 = 0;
                    for (0..m) |k| g += xj[k] * sb[k];
                    g = 2.0 * g * sc[b] * inv[b];

                    const idx = b * cols + j;
                    const sig = 1.0 / (1.0 + @exp(-vv[idx]));
                    const stretched = sig * (zeta - gamma) + gamma;
                    // Outside [0,1] the clip kills the gradient — the point of the
                    // rectification, and h stays pinned until the other term moves it.
                    const dh: f64 = if (stretched <= 0.0 or stretched >= 1.0)
                        0.0
                    else
                        sig * (1.0 - sig) * (zeta - gamma);

                    var gg = g * dh;
                    if (lam > 0) {
                        const hv = std.math.clamp(stretched, 0.0, 1.0);
                        const u = 2.0 * hv - 1.0;
                        const au = @abs(u);
                        if (au > 1e-6 and au < 1.0) {
                            // d/dh [1 - |2h-1|^β] = -2β|2h-1|^(β-1)·sign(2h-1)
                            const d = -2.0 * beta * std.math.pow(f64, au, beta - 1.0) *
                                std.math.sign(u);
                            gg += lam * d * dh;
                        }
                    }

                    am[idx] = 0.9 * am[idx] + 0.1 * gg;
                    av[idx] = 0.999 * av[idx] + 0.001 * gg * gg;
                    vv[idx] -= @as(f64, p.lr) * (am[idx] / bc1) / (@sqrt(av[idx] / bc2) + 1e-8);
                }
            }
        }

        // Hard round, then price it against RTN on each row's own objective.
        for (0..nb) |b| {
            if (sc[b] == 0) continue;
            const dst = q_out[(base + b) * cols ..][0..cols];
            for (0..cols) |j| {
                const hv: f64 = hOf(@floatCast(vv[b * cols + j]));
                dst[j] = @floatCast(fl[b * cols + j] + @as(f64, if (hv >= 0.5) 1.0 else 0.0));
                e[b * cols + j] = (@as(f64, dst[j]) - t[b * cols + j]) * sc[b];
            }
        }
        blockS(e, sblk, h, m, cols, nb);
        for (0..nb) |b| {
            if (sc[b] == 0) continue;
            var t2: f64 = 0;
            for (sblk[b * m ..][0..m]) |v| t2 += v * v;
            const fin = t2 / @as(f64, @floatFromInt(m));
            if (fin < orn[b]) _ = acc.rows_improved.fetchAdd(1, .monotonic);
            addF64(&acc.obj_final, fin);
            addF64(&acc.obj_rtn, orn[b]);
        }
    }
}

/// `S[b] = e[b] · X̃ᵗ` for a block of rows, one pass over the sample.
fn blockS(e: []const f64, sblk: []f64, h: *const Gptq.Hessian, m: usize, cols: usize, nb: usize) void {
    @memset(sblk[0 .. nb * m], 0);
    for (0..cols) |j| {
        const xj = h.xt[j * m ..][0..m];
        for (0..nb) |b| {
            const ej = e[b * cols + j];
            if (ej == 0) continue;
            const sb = sblk[b * m ..][0..m];
            for (0..m) |k| sb[k] += xj[k] * ej;
        }
    }
}

/// `‖e X̃ᵗ‖² / m`. `s` is scratch of length `m`.
fn objective(e: []const f64, s: []f64, h: *const Gptq.Hessian, m: usize, cols: usize) f64 {
    @memset(s, 0);
    for (0..cols) |j| {
        const ej = e[j];
        if (ej == 0) continue;
        const xj = h.xt[j * m ..][0..m];
        for (0..m) |k| s[k] += xj[k] * ej;
    }
    var acc: f64 = 0;
    for (s) |v| acc += v * v;
    return acc / @as(f64, @floatFromInt(m));
}

fn forEachRowChunk(pool: *ThreadPool, rows: usize, comptime func: anytype, args: anytype) !void {
    if (rows == 0) return;
    const n = @max(1, @min(pool.threads.len, rows));
    const per = rows / n;
    const leftover = rows - per * n;
    var wg: thread_pool_mod.WaitGroup = .{};
    var i: usize = 0;
    var at: usize = 0;
    while (i < n) : (i += 1) {
        const take = per + @intFromBool(i < leftover);
        const start = at;
        at += take;
        if (take == 0) continue;
        pool.spawnWg(&wg, func, args ++ .{ start, at });
    }
    wg.wait();
}

/// Dequantized result in the **original** basis, matching `Gptq.roundtrip` so the
/// level-1 harness can drop either in.
pub fn roundtrip(
    gpa: std.mem.Allocator,
    h: *const Gptq.Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    grid: Gptq.Grid,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) ![]f32 {
    const lv = try sweep(gpa, h, w, rows, cols, grid, weights, pool, p);
    defer lv.deinit(gpa);
    const out = try gpa.alloc(f32, rows * cols);
    errdefer gpa.free(out);
    for (0..rows) |r| {
        const sc = lv.scale[r];
        for (0..cols) |j| out[r * cols + j] = lv.q[r * cols + j] * sc;
    }
    if (h.basis.convrot) try Q.rotateGroupwiseInPlace(out, rows, cols, h.basis.group_size, pool);
    return out;
}

pub fn quantizeInt4(
    gpa: std.mem.Allocator,
    h: *const Gptq.Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) !Q.Int4Data {
    if (cols % 2 != 0) return error.ColsNotEven;
    const lv = try sweep(gpa, h, w, rows, cols, Gptq.int4_grid, weights, pool, p);
    defer gpa.free(lv.q);
    errdefer gpa.free(lv.scale);

    const packed_cols = cols / 2;
    const out = try gpa.alloc(u8, rows * packed_cols);
    errdefer gpa.free(out);
    for (0..rows) |r| {
        const q = lv.q[r * cols ..][0..cols];
        const dst = out[r * packed_cols ..][0..packed_cols];
        for (0..packed_cols) |pc| {
            const lo: i8 = @intFromFloat(q[2 * pc]);
            const hi: i8 = @intFromFloat(q[2 * pc + 1]);
            dst[pc] = (@as(u8, @bitCast(lo)) & 0x0F) | (@as(u8, @bitCast(hi)) << 4);
        }
    }
    return .{ .weight = out, .scale = lv.scale };
}

pub fn quantizeInt8(
    gpa: std.mem.Allocator,
    h: *const Gptq.Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) !Q.ConvrotInt8Data {
    const lv = try sweep(gpa, h, w, rows, cols, Gptq.int8_grid, weights, pool, p);
    defer gpa.free(lv.q);
    errdefer gpa.free(lv.scale);
    const out = try gpa.alloc(u8, rows * cols);
    errdefer gpa.free(out);
    for (out, lv.q) |*o, qq| o.* = @bitCast(@as(i8, @intFromFloat(qq)));
    return .{ .weight = out, .scale = lv.scale };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Deterministic pseudo-random block, so a failure is reproducible without
/// depending on std's PRNG staying put.
fn fill(buf: []f32, seed: u64) void {
    var st = seed;
    for (buf) |*v| {
        st = st *% 6364136223846793005 +% 1442695040888963407;
        const u: f32 = @floatFromInt((st >> 40) & 0xFFFF);
        v.* = (u / 32768.0) - 1.0;
    }
}

test "h and its inverse round-trip on the open interval" {
    for ([_]f32{ 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99 }) |rest| {
        try testing.expectApproxEqAbs(rest, hOf(vOf(rest)), 1e-4);
    }
    // The rectification: h saturates exactly at the corners.
    try testing.expectEqual(@as(f32, 0.0), hOf(-50));
    try testing.expectEqual(@as(f32, 1.0), hOf(50));
}

test "levels stay on the grid" {
    const gpa = testing.allocator;
    const rows = 6;
    const cols = 32;
    const m = 12;
    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 2 });
    defer pool.deinit();

    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(w, 1);
    fill(x, 2);

    var h = try Gptq.Hessian.init(gpa, x, m, cols, .{}, 0.01, &pool);
    defer h.deinit();

    const lv = try sweep(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, .{ .iters = 40 });
    defer lv.deinit(gpa);
    for (lv.q) |v| {
        try testing.expect(v >= Gptq.int4_grid.qlo and v <= Gptq.int4_grid.qhi);
        try testing.expectEqual(v, @round(v));
    }
}

test "adaptive rounding beats round-to-nearest on its own objective" {
    const gpa = testing.allocator;
    const rows = 8;
    const cols = 64;
    const m = 24;
    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 2 });
    defer pool.deinit();

    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(w, 3);
    fill(x, 4);

    var h = try Gptq.Hessian.init(gpa, x, m, cols, .{}, 0.01, &pool);
    defer h.deinit();

    var st: Stats = .{};
    const lv = try sweep(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, .{
        .iters = 300,
        .stats = &st,
    });
    defer lv.deinit(gpa);

    // The whole claim of the mechanism, in-sample: if this fails the optimizer is
    // not working and no downstream number means anything.
    try testing.expect(st.obj_final < st.obj_rtn);
    try testing.expect(st.rows_improved * 2 >= st.rows_total);
}

test "adaptive rounding beats GPTQ on the objective they share" {
    const gpa = testing.allocator;
    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 4 });
    defer pool.deinit();

    // Two aspect ratios, so the defaults are not fitted to one shape.
    for ([_][3]usize{ .{ 16, 256, 64 }, .{ 32, 512, 128 } }) |dims| {
        const rows = dims[0];
        const cols = dims[1];
        const m = dims[2];
        const w = try gpa.alloc(f32, rows * cols);
        defer gpa.free(w);
        const x = try gpa.alloc(f32, m * cols);
        defer gpa.free(x);
        fill(w, 11);
        fill(x, 12);

        var h = try Gptq.Hessian.init(gpa, x, m, cols, .{}, 0.01, &pool);
        defer h.deinit();
        const s = try gpa.alloc(f64, m);
        defer gpa.free(s);
        const e = try gpa.alloc(f64, cols);
        defer gpa.free(e);

        const glv = try Gptq.sweep(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, .{});
        defer glv.deinit(gpa);
        var gobj: f64 = 0;
        for (0..rows) |r| {
            const sc: f64 = glv.scale[r];
            for (0..cols) |j| {
                const t = @as(f64, w[r * cols + j]) / sc;
                e[j] = (@as(f64, glv.q[r * cols + j]) - t) * sc;
            }
            gobj += objective(e, s, &h, m, cols);
        }

        var st: Stats = .{};
        const lv = try sweep(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, .{ .stats = &st });
        defer lv.deinit(gpa);

        // The reason this file exists: joint optimization should reach a lower
        // point on the shared objective than the greedy single pass. If it stops
        // doing so, the defaults have drifted and every arm built on it is
        // measuring a degenerate optimizer rather than the mechanism.
        if (!(st.obj_final < gobj)) {
            std.debug.print("{d}x{d} m{d}: adaround {d:.5} vs gptq {d:.5} (rtn {d:.5})\n", .{
                rows, cols, m, st.obj_final, gobj, st.obj_rtn,
            });
        }
        try testing.expect(st.obj_final < gobj);
    }
}

test "the device path agrees with the CPU reference" {
    const gpa = testing.allocator;
    // Self-skipping rather than `-Dintegration`-gated: this is the only check that
    // the GPU arm computes what the CPU one does, so it should run wherever a GPU
    // exists rather than only in a mode nobody remembers to pass.
    const be = tp.gpu.cuda.Backend.initLibs(gpa) catch return;
    defer be.deinit();
    be.bindThread();

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 4 });
    defer pool.deinit();

    const rows = 64;
    const cols = 512;
    const m = 128;
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(w, 21);
    fill(x, 22);

    var h = try Gptq.Hessian.init(gpa, x, m, cols, .{}, 0.01, &pool);
    defer h.deinit();

    const p: Params = .{ .iters = 60 };
    gpu_backend = null;
    var st_cpu: Stats = .{};
    var pc = p;
    pc.stats = &st_cpu;
    const cpu = try sweep(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, pc);
    defer cpu.deinit(gpa);

    gpu_backend = be;
    defer gpu_backend = null;
    var st_gpu: Stats = .{};
    var pg = p;
    pg.stats = &st_gpu;
    const gpu = try sweep(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, pg);
    defer gpu.deinit(gpa);

    // The scale search is identical arithmetic on both paths, so the grid must match
    // exactly; if it does not, the two are not the same experiment at all.
    try testing.expectEqualSlices(f32, cpu.scale, gpu.scale);

    // The levels themselves can differ on a few weights: the device GEMM is f32
    // against the reference's f64, so a weight sitting near h = 0.5 can land on
    // either side. What must hold is that the optimizer got to the same place.
    var diff: usize = 0;
    for (cpu.q, gpu.q) |a, b| {
        if (a != b) diff += 1;
    }
    const frac = @as(f64, @floatFromInt(diff)) / @as(f64, @floatFromInt(cpu.q.len));
    const ratio = (st_gpu.obj_final / st_gpu.obj_rtn) / (st_cpu.obj_final / st_cpu.obj_rtn);
    if (frac > 0.02 or ratio > 1.10 or ratio < 0.90) {
        std.debug.print("gpu/cpu: {d:.4} of levels differ, objective ratio {d:.4}\n", .{ frac, ratio });
    }
    try testing.expect(frac <= 0.02);
    try testing.expect(ratio > 0.90 and ratio < 1.10);
}

test "convrot rounds in the rotated basis and returns to the original one" {
    const gpa = testing.allocator;
    const rows = 4;
    const cols = 32;
    const m = 16;
    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 2 });
    defer pool.deinit();

    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(w, 5);
    fill(x, 6);

    var h = try Gptq.Hessian.init(gpa, x, m, cols, .{ .convrot = true, .group_size = 16 }, 0.01, &pool);
    defer h.deinit();

    const out = try roundtrip(gpa, &h, w, rows, cols, Gptq.int4_grid, null, &pool, .{ .iters = 40 });
    defer gpa.free(out);
    try testing.expectEqual(w.len, out.len);
    for (out) |v| try testing.expect(std.math.isFinite(v));
}
