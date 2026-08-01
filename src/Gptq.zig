//! Gptq.zig — plan §8C, GPTQ / OBQ error compensation.
//!
//! §8A minimizes a *diagonal* approximation of the output error:
//!
//! ```
//!     Σ_j  w_j · (W_ij − Ŵ_ij)²          w_j = Σ_tokens x_j²
//! ```
//!
//! which is `eᵀ diag(G) e` for `G = XᵗX`. That is the whole objective only if
//! the input channels are uncorrelated, and in a transformer they emphatically
//! are not. §8C keeps the off-diagonals: the quantity a GEMM actually pays is
//!
//! ```
//!     ‖(W − Ŵ) Xᵗ‖²_F  =  Σ_rows  eᵀ G e            G = XᵗX  (cols × cols)
//! ```
//!
//! and `G`'s off-diagonal terms mean a rounding error made in column *i* can be
//! **partly cancelled** by deliberately mis-rounding the columns that follow it.
//! That is GPTQ: sweep the columns left to right, round each one to the format's
//! own grid, then push the residual into the not-yet-rounded columns. Per output
//! row, with `R = {i+1 … cols−1}` still free and `e_i` the error just committed,
//! the minimizer of `eᵀ H e` is
//!
//! ```
//!     Δw_R = −e_i · c_i,          c_i = H_RR⁻¹ H_R,i,      H = λI + G
//! ```
//!
//! The result is an ordinary tensor in the ordinary format — no runtime change,
//! no new kernel, nothing for ComfyUI to know about. That is why §8B's parking
//! (which needed a runtime change for 2%) strengthened the case for this.
//!
//! ### Why there is no `cols × cols` matrix here
//!
//! Textbook GPTQ forms `H`, Choleskys it, and inverts — `O(cols³)` time and
//! `cols²` memory. On krea2's `mlp.down` (`cols = 16384`) that is a 2.1 GB f64
//! allocation and ~1.5e12 flops **per layer**, which would make a level-1 sweep
//! unaffordable and the arm therefore unmeasurable.
//!
//! It is also unnecessary. The calibration cache holds `m` sampled token rows,
//! and `m ≪ cols` (96 at rung 1 against 6144–16384 columns), so `G = XᵗX` has
//! rank `m` and `H = λI + XᵗX` is a ridge plus a low-rank term. Woodbury on the
//! *suffix* block collapses the whole thing to `m × m` algebra:
//!
//! ```
//!     H_RR⁻¹ = (1/λ)(I − X_Rᵗ M_R⁻¹ X_R),      M_R = λI_m + X_R X_Rᵗ
//!     H_R,i  = X_Rᵗ x_i                         (i ∉ R, so no λ term)
//!
//!     c_i = (1/λ) X_Rᵗ (I − M_R⁻¹ (M_R − λI)) x_i
//!         = X_Rᵗ M_R⁻¹ x_i
//! ```
//!
//! — exact, not an approximation, and `M_R` is `m × m`. Sweeping `i` left to
//! right shrinks `R` by one column at a time, so `M_R` is maintained by a rank-1
//! **downdate** and its inverse by Sherman–Morrison. `z_i = M_R⁻¹ x_i` then falls
//! out of the same intermediate: with `u = M⁻¹x_i` and `t = x_iᵗu` before the
//! downdate, `z_i = u / (1 − t)`.
//!
//! ### …and no `rows × cols` update pass either
//!
//! The same rank argument kills the other half of the cost. Textbook GPTQ applies
//! each column's correction to every column after it — `rows·cols²/2` work, which
//! looks unavoidable. But the correction is *linear* in the errors, so for one
//! output row the total adjustment at column `j` telescopes:
//!
//! ```
//!     Σ_{i<j} e_i · c_i[j]  =  Σ_{i<j} e_i · (x_jᵗ z_i)  =  x_jᵗ · ( Σ_{i<j} e_i z_i )
//! ```
//!
//! — a dot product against a **running `m`-vector accumulator**. So a row is swept
//! in `O(m·cols)` rather than `O(cols²)`, the `[block, cols]` coefficient matrix
//! never has to exist, and every output row carries its own accumulator, which
//! makes the sweep embarrassingly parallel with no barriers and no shared writes.
//!
//! Per layer, `cols = 6144`, `m = 96`, `rows = 6144`: `1.2e11 → 7.2e9` flops, and
//! the `cols³` Cholesky (`7.7e10`) is gone as well. **Measured on three krea2
//! `blocks.0` layers: 15.5 s of GPTQ → 2.4 s**, i.e. 6.5×, against a blocked
//! coefficient-matrix implementation of the identical arithmetic (which the dense
//! oracle test pins both of them to). This is the one place where the reference
//! implementation's shape is actively wrong for us — GPTQ is normally run with
//! `m ≫ cols`, where none of this structure exists.
//!
//! ### The rotated basis, for free
//!
//! ConvRot rounds `H_g·w` group-wise, so GPTQ has to run in that basis:
//! `G̃ = R G R` with `R` block-diagonal Hadamard. Rotating a `cols × cols` Gram
//! would cost `cols²·log g`; here the activations *are* the state, so rotating
//! the `m` sampled rows (`m·cols·log g`) is the whole job. `X̃ = X R`, and
//! `G̃ = X̃ᵗX̃ = R G R` follows.
//!
//! This is also why the §8A caveat about `rotatedWeights` does not recur: that
//! function exists because a *diagonal* importance vector has to be mapped into
//! the rotated basis, where it collapses to one value per group. §8C never forms
//! a diagonal — it carries `X̃` and keeps every cross term.
//!
//! ### Scope: the integer formats, deliberately
//!
//! GPTQ needs to own the rounding decision, which means it needs the format's
//! grid up front. ggufy's int8/int4 (plain and ConvRot) are exactly that shape —
//! one symmetric per-row scale, chosen by `Quantizer.searchScale`, levels
//! `[qlo, qhi]`. Both are taken from `DataTransform` rather than restated, so
//! the only difference between this and the shipped quantizer is *which level*
//! each weight lands on.
//!
//! ### The ggml block quants — the GGUF path, at block granularity
//!
//! The k-quants have no grid to round against: a per-256 super-block scale plus
//! quantized per-32 sub-scales and mins, all chosen inside ggml's own search.
//! Extracting that grid would mean reimplementing `ggml-quants.c`, which is not
//! worth doing and would put the golden-byte fixtures at risk.
//!
//! `roundtripGgml` avoids needing it. Instead of rounding, it hands ggml one
//! **block** of already-compensated weights at a time and compensates on the error
//! that comes back. ggml keeps complete ownership of the grid, the search and the
//! bytes; GPTQ only decides what values it is shown. Since a super-block is
//! self-contained, quantizing one block at a time is bit-identical to quantizing
//! the whole row — which is what makes the degenerate case (zero compensation)
//! exactly the shipped quantizer, and it is tested that way.
//!
//! The Woodbury form generalizes to a simultaneously-quantized column set `S`
//! without changing shape:
//!
//! ```
//!     d_R = −H_RR⁻¹ H_RS e_S = −X_Rᵗ · ( M_R⁻¹ X_S e_S )
//! ```
//!
//! so the accumulator gains one `m`-vector per *block* rather than per column, and
//! `M_R⁻¹x_j` for the block's columns is shared across every output row exactly as
//! `z_j` is. Cost stays `O(m·rows·cols)`. At block granularity the Sherman–Morrison
//! machinery is not worth it either — one `m × m` Cholesky per block is cheaper
//! than 256 rank-1 updates and cannot drift.
//!
//! ⚠️ **What this cannot reach is *within*-block correlation.** ggml must see all
//! 256 values at once to pick its scales, so compensation happens only across
//! blocks: 24 steps for a 6144-column layer against 6144 for the per-column int
//! path. Expect a smaller gain than the int formats show, for that reason and
//! because the k-quants' per-32 sub-scales already spend bits on what compensation
//! would otherwise recover.

const std = @import("std");
const DataTransform = @import("DataTransform.zig");
const thread_pool_mod = @import("ThreadPool.zig");
const types = @import("types.zig");
const gguf = @import("Gguf.zig");

const Q = DataTransform.Quantizer;
const ThreadPool = thread_pool_mod.ThreadPool;

/// Ridge as a fraction of the mean diagonal of `G`, i.e. GPTQ's `percdamp`.
///
/// It is not optional: `G` has rank `m ≪ cols`, so `H_RR` is singular without it
/// and `c_i` would be meaningless. 1% is the reference implementation's value and
/// also the honest one — the ridge is what stops the compensation from chasing
/// directions the calibration sample says nothing about.
pub const default_damp: f32 = 0.01;

/// The format's rounding grid. One symmetric scale per output row, taken from
/// `DataTransform` so a level change here cannot silently diverge from what
/// `convert` writes.
pub const Grid = struct {
    /// Divisor the format's default scale uses: `amax / qdiv`.
    qdiv: f32,
    qlo: f32,
    qhi: f32,
};

/// `clamp(round(w/s), -7, 7)` — note ComfyUI's convrot_w4a4 never emits −8.
pub const int4_grid: Grid = .{ .qdiv = 7.0, .qlo = -7.0, .qhi = 7.0 };
/// `clamp(round(w/s), -128, 127)` — asymmetric bounds against a 127 divisor.
pub const int8_grid: Grid = .{ .qdiv = 127.0, .qlo = -128.0, .qhi = 127.0 };

/// The basis the quantizer rounds in.
pub const Basis = struct {
    convrot: bool = false,
    /// Hadamard group size; ignored unless `convrot`.
    group_size: usize = 0,
};

pub const Params = struct {
    damp: f32 = default_damp,
    /// Columns between refactorizations of `M_R⁻¹`. Sherman–Morrison is exact in
    /// exact arithmetic; in f64 over thousands of steps it drifts, so the inverse
    /// is periodically recomputed from the exactly-downdated `M_R`.
    refactor_every: usize = 256,
    /// Filled in if provided. Reported rather than logged: a ridge bump is
    /// interesting but not an error, and this runs inside a per-layer loop.
    stats: ?*Stats = null,
};

pub const Stats = struct {
    /// Times `M_R⁻¹` was recomputed from scratch, scheduled or forced.
    refactors: usize = 0,
    /// Times a Cholesky of `M_R` failed and the ridge had to be raised. Expected
    /// to be 0; a nonzero count means the sample is more degenerate than `damp`
    /// assumed and the compensation for those columns is more heavily damped.
    ridge_bumps: usize = 0,
    /// Times the Sherman–Morrison denominator came out non-positive, which cannot
    /// happen in exact arithmetic (`xᵗM_R⁻¹x < 1` strictly, for λ > 0) and so is a
    /// direct measure of accumulated drift. Each one forces a refactorization.
    forced_refactors: usize = 0,
};

// ---------------------------------------------------------------------------
// Per-layer state
// ---------------------------------------------------------------------------

/// The activation second moment for one layer, in the basis the quantizer rounds
/// in — held in the low-rank form `H = λI + X̃ᵗX̃` rather than as a Gram matrix.
///
/// Built once per (layer, basis) and shared by every format that rounds in that
/// basis, since `H` depends only on the activations.
pub const Hessian = struct {
    gpa: std.mem.Allocator,
    /// Token rows the moment was accumulated from.
    m: usize,
    cols: usize,
    /// The ridge, `damp · mean_j G_jj`.
    lambda: f64,
    /// `X̃ᵗ`, `[cols][m]`: column *j* of the rotated activation block, contiguous.
    /// The sweep touches one column at a time, so this is the layout that makes
    /// `c_i` a sequence of contiguous dot products.
    xt: []f64,
    /// `λI_m + X̃X̃ᵗ`, the `m × m` seed the downdate starts from.
    m0: []f64,
    basis: Basis,

    /// `x` is `[m, cols]` row-major token rows, as `Sensitivity.gatherX` produces.
    pub fn init(
        gpa: std.mem.Allocator,
        x: []const f32,
        m: usize,
        cols: usize,
        basis: Basis,
        damp: f32,
        pool: *ThreadPool,
    ) !Hessian {
        if (m == 0 or cols == 0) return error.EmptyInput;
        if (x.len != m * cols) return error.InputSizeMismatch;
        if (!(damp > 0)) return error.InvalidDamp;

        // Rotate the sample, not the Gram — see the header. `rotateGroupwiseInPlace`
        // validates the group size against `cols` for us.
        const xr = try gpa.dupe(f32, x);
        defer gpa.free(xr);
        if (basis.convrot) try Q.rotateGroupwiseInPlace(xr, m, cols, basis.group_size, pool);

        const xt = try gpa.alloc(f64, cols * m);
        errdefer gpa.free(xt);
        var trace: f64 = 0;
        for (0..cols) |j| {
            const dst = xt[j * m ..][0..m];
            for (0..m) |t| {
                const v: f64 = xr[t * cols + j];
                dst[t] = v;
                trace += v * v;
            }
        }
        // A layer the capture never excited has no covariance to exploit, and a
        // zero ridge would make every solve singular. Refusing is the same call
        // `Imatrix.fromCache` makes for the same reason: no information is not
        // the same as uniform information.
        if (!(trace > 0)) return error.NoActivationEnergy;

        const lambda = @as(f64, damp) * trace / @as(f64, @floatFromInt(cols));

        const m0 = try gpa.alloc(f64, m * m);
        errdefer gpa.free(m0);
        @memset(m0, 0);
        for (0..cols) |j| {
            const xj = xt[j * m ..][0..m];
            for (0..m) |a| {
                const va = xj[a];
                if (va == 0) continue;
                const row = m0[a * m ..][0..m];
                for (0..m) |b| row[b] += va * xj[b];
            }
        }
        for (0..m) |a| m0[a * m + a] += lambda;

        return .{
            .gpa = gpa,
            .m = m,
            .cols = cols,
            .lambda = lambda,
            .xt = xt,
            .m0 = m0,
            .basis = basis,
        };
    }

    pub fn deinit(self: *Hessian) void {
        self.gpa.free(self.xt);
        self.gpa.free(self.m0);
        self.* = undefined;
    }

    /// `G_jj` — the per-column energy of the sampled rows, in the rotated basis.
    /// This is exactly what §8A steers on, exposed so the two mechanisms can be
    /// shown to be reading the same statistic.
    pub fn diag(self: *const Hessian, j: usize) f64 {
        var acc: f64 = 0;
        for (self.xt[j * self.m ..][0..self.m]) |v| acc += v * v;
        return acc;
    }
};

// ---------------------------------------------------------------------------
// The sweep
// ---------------------------------------------------------------------------

/// The sweep's raw output: the integer level chosen for every weight, in the basis
/// the rounding happened in, plus the per-row scale it was chosen against.
///
/// Levels rather than dequantized values because the two consumers want different
/// things from them — the level-1 arm wants `s·q` un-rotated back, the converter
/// wants the codes themselves. Dequantizing and re-quantizing to recover the codes
/// would not be safe: compensation pushes weights past the original `amax`, so a
/// second `searchScale` on the dequantized result can land on a different scale.
pub const Levels = struct {
    /// `[rows, cols]`, each entry an integer in `[grid.qlo, grid.qhi]` held as f32.
    q: []f32,
    /// `[rows]`.
    scale: []f32,
    stats: Stats,

    pub fn deinit(self: Levels, gpa: std.mem.Allocator) void {
        gpa.free(self.q);
        gpa.free(self.scale);
    }
};

/// Quantize `w` (`[rows, cols]`, row-major) with GPTQ compensation and return the
/// dequantized result in the **original** basis — the same `[]f32` shape
/// `precision_harness.roundtrip` returns, so the level-1 arm can drop it in.
///
/// `weights` is the per-input-column importance §8A uses, and is used *only* to
/// pick the scale, via the same `searchScale` the converter calls. Pass it when
/// the format ships weighted, null otherwise: the point of the arm is to isolate
/// what the compensation adds on top of the configuration that already ships.
pub fn roundtrip(
    gpa: std.mem.Allocator,
    h: *const Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    grid: Grid,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) ![]f32 {
    const lv = try sweep(gpa, h, w, rows, cols, grid, weights, pool, p);
    defer lv.deinit(gpa);

    const out = try gpa.alloc(f32, rows * cols);
    errdefer gpa.free(out);
    for (0..rows) |r| {
        const s = lv.scale[r];
        for (out[r * cols ..][0..cols], lv.q[r * cols ..][0..cols]) |*o, q| o.* = s * q;
    }
    if (h.basis.convrot) try Q.rotateGroupwiseInPlace(out, rows, cols, h.basis.group_size, pool);
    if (p.stats) |o| o.* = lv.stats;
    return out;
}

/// The compensated sweep itself. Caller owns the result.
pub fn sweep(
    gpa: std.mem.Allocator,
    h: *const Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    grid: Grid,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) !Levels {
    if (cols != h.cols) return error.ShapeMismatch;
    if (w.len != rows * cols) return error.InputSizeMismatch;
    if (rows == 0) return error.EmptyInput;
    if (p.refactor_every == 0) return error.InvalidParams;
    if (h.basis.convrot and cols % 2 != 0) return error.ColsNotEven;

    const m = h.m;

    // Holds the rotated weights on the way in and the chosen levels on the way out.
    const work = try gpa.dupe(f32, w);
    errdefer gpa.free(work);
    if (h.basis.convrot) try Q.rotateGroupwiseInPlace(work, rows, cols, h.basis.group_size, pool);

    // Importance in the basis the rounding happens in — the §8A mapping, reused
    // rather than restated. Only the scale search consumes it.
    const rot_w = try Q.rotatedWeights(gpa, weights, cols, h.basis.convrot, h.basis.group_size);
    defer if (rot_w) |rw| gpa.free(rw);

    // Every scale is fixed before the first column is touched. GPTQ's residuals
    // change the weights as it goes, and letting a later row's scale respond to
    // an earlier row's compensation would make the result depend on row order.
    const scales = try gpa.alloc(f32, rows);
    errdefer gpa.free(scales);
    try forEachRowChunk(pool, rows, scaleRowsJob, .{ work, scales, cols, rot_w, grid });

    // m × m state: `mr` is λI + Σ_{j>i} x_j x_jᵗ, downdated exactly; `pinv` is its
    // inverse, maintained by Sherman–Morrison and refactored periodically.
    const mr = try gpa.alloc(f64, m * m);
    defer gpa.free(mr);
    @memcpy(mr, h.m0);
    const pinv = try gpa.alloc(f64, m * m);
    defer gpa.free(pinv);
    const chol = try gpa.alloc(f64, m * m);
    defer gpa.free(chol);
    const scratch = try gpa.alloc(f64, m * m);
    defer gpa.free(scratch);
    const u = try gpa.alloc(f64, m);
    defer gpa.free(u);

    var st: Stats = .{};
    try refactor(mr, pinv, chol, scratch, m, h.lambda, &st);

    // `z[j]` is `M_R(j)⁻¹ x_j`: everything the sweep needs to know about the
    // activations, one `m`-vector per column. Building it is the only sequential
    // phase, since shrinking `R` is inherently ordered — `O(m²)` per column, so
    // ~0.1 s for a 6144-column layer at m = 96.
    const z = try gpa.alloc(f64, cols * m);
    defer gpa.free(z);

    var since_refactor: usize = 0;
    for (0..cols) |j| {
        const xj = h.xt[j * m ..][0..m];
        downdate(mr, xj, m);
        const zj = z[j * m ..][0..m];

        var done = false;
        if (since_refactor < p.refactor_every) {
            matVec(pinv, xj, u, m);
            var t: f64 = 0;
            for (xj, u) |a, b| t += a * b;
            const denom = 1.0 - t;
            // Strictly positive in exact arithmetic for λ > 0, so a violation is
            // drift and nothing else — counted, and handled by refactoring.
            if (denom > 1e-9) {
                const inv_d = 1.0 / denom;
                for (0..m) |a| {
                    const ua = u[a] * inv_d;
                    if (ua == 0) continue;
                    const row = pinv[a * m ..][0..m];
                    for (0..m) |b| row[b] += ua * u[b];
                }
                for (zj, u) |*o, ua| o.* = ua * inv_d;
                since_refactor += 1;
                done = true;
            } else {
                st.forced_refactors += 1;
            }
        }
        if (!done) {
            try refactor(mr, pinv, chol, scratch, m, h.lambda, &st);
            since_refactor = 0;
            matVec(pinv, xj, zj, m);
        }
    }

    // Then every output row, independently: sweep left to right carrying its own
    // `m`-vector of committed error. No shared writes, no barriers, so this is
    // exactly thread-count invariant — each row's arithmetic is identical however
    // the rows are split.
    const acc = try gpa.alloc(f64, @max(1, pool.threads.len) * m);
    defer gpa.free(acc);
    try sweepRows(pool, work, h.xt, z, scales, acc, m, rows, cols, grid);

    return .{ .q = work, .scale = scales, .stats = st };
}

/// GPTQ-compensated INT4, in the on-disk cluster layout — the `convert` path for
/// `INT4_CONVROT`. Signature mirrors `Quantizer.quantizeToInt4Weighted` so the
/// cluster writer can take either.
///
/// Packing is nibble-per-element, low nibble first, exactly as
/// `quantizeInt4Rows` does it; the levels differ, the container does not. There is
/// no stochastic-rounding variant: SR dithers *within* a grid, and GPTQ is already
/// choosing levels deliberately against a measured objective, so the two are
/// answering the same question in contradictory ways.
pub fn quantizeInt4(
    gpa: std.mem.Allocator,
    h: *const Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) !Q.Int4Data {
    if (cols % 2 != 0) return error.ColsNotEven;
    const lv = try sweep(gpa, h, w, rows, cols, int4_grid, weights, pool, p);
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
    if (p.stats) |o| o.* = lv.stats;
    return .{ .weight = out, .scale = lv.scale };
}

/// GPTQ-compensated INT8, plain or ConvRot. Counterpart of `quantizeInt8Weighted`.
pub fn quantizeInt8(
    gpa: std.mem.Allocator,
    h: *const Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) !Q.ConvrotInt8Data {
    const lv = try sweep(gpa, h, w, rows, cols, int8_grid, weights, pool, p);
    defer gpa.free(lv.q);
    errdefer gpa.free(lv.scale);

    const out = try gpa.alloc(u8, rows * cols);
    errdefer gpa.free(out);
    for (out, lv.q) |*o, q| o.* = @bitCast(@as(i8, @intFromFloat(q)));
    if (p.stats) |o| o.* = lv.stats;
    return .{ .weight = out, .scale = lv.scale };
}

/// Does this encoder's *weighted* path depend only on the block it is handed?
///
/// A statement of fact about ggml, read out of `ggml-quants.c` and pinned by a
/// test. The k-quants normalize per super-block —
/// `sigma2 = 2·Σx²/QK_K`, then `weight[l] = qw[l]·√(sigma2 + x[l]²)` — so handing
/// them one super-block at a time is bit-identical to handing them the whole row.
/// The legacy `qX_0`/`qX_1` types compute `sigma2 = Σx²/n_per_row` over the entire
/// row, which couples every block to every other one.
///
/// §8C must refuse the coupled types outright rather than quantize them
/// block-at-a-time anyway. Its entire claim is that the only thing it changes is
/// *which values ggml is shown*; if the block size also changed the weights ggml
/// derives, the measured difference would be part compensation and part encoder
/// change, with no way to separate them. (Their unweighted paths are block-local —
/// `quantize_row_qX_0_ref` has no cross-block term — so only the weighted arm is
/// affected, which is unfortunately the arm that matters.)
pub fn ggmlBlockLocalWeighted(t: gguf.GgmlType) bool {
    return switch (t) {
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

/// GPTQ against a ggml block encoder — plan §8C for the **GGUF** path.
///
/// Unlike `roundtrip`, nothing here decides a rounding: ggml quantizes each block
/// of already-compensated weights with its own scale search, and the compensation
/// is computed from the error it returns. See the header for why that is the only
/// workable shape for the k-quants, and for the one thing it gives up (correlation
/// *within* a block).
///
/// `weights` is the same per-column importance the §8A `imatrix` path uses; it is
/// forwarded to ggml one block-slice at a time, which lines up with how ggml
/// indexes `quant_weights` inside the row it is handed.
pub fn roundtripGgml(
    gpa: std.mem.Allocator,
    h: *const Hessian,
    w: []const f32,
    rows: usize,
    cols: usize,
    dst: types.DataType,
    weights: ?[]const f32,
    pool: *ThreadPool,
    p: Params,
) ![]f32 {
    if (cols != h.cols) return error.ShapeMismatch;
    if (w.len != rows * cols) return error.InputSizeMismatch;
    if (rows == 0) return error.EmptyInput;
    // No ggml type is ConvRot; a rotated Hessian here would mean the caller paired
    // a basis with a format that does not round in it.
    if (h.basis.convrot) return error.RotatedBasisUnsupported;
    if (weights) |ws| if (ws.len != cols) return error.WeightsWidthMismatch;

    const gt = gguf.GgmlType.fromString(@tagName(dst)) catch return error.UnsupportedDestinationType;
    const blk: usize = @intCast(gt.getBlockSize());
    if (blk == 0 or cols % blk != 0) return error.BlockMisaligned;
    if (weights != null and !ggmlBlockLocalWeighted(gt)) return error.RowCoupledWeightedEncoder;

    const m = h.m;

    const out = try gpa.dupe(f32, w);
    errdefer gpa.free(out);

    // One accumulator per output row, live for the whole sweep — the block loop is
    // outermost here (ggml wants a panel, not a row), so unlike the per-column path
    // these cannot be thread-local.
    const acc = try gpa.alloc(f64, rows * m);
    defer gpa.free(acc);
    @memset(acc, 0);

    // The compensated block, [rows][blk], laid out as `rows` rows of `blk` — which
    // is exactly the unit ggml quantizes independently.
    const panel = try gpa.alloc(f32, rows * blk);
    defer gpa.free(panel);

    const mr = try gpa.alloc(f64, m * m);
    defer gpa.free(mr);
    @memcpy(mr, h.m0);
    const pinv = try gpa.alloc(f64, m * m);
    defer gpa.free(pinv);
    const chol = try gpa.alloc(f64, m * m);
    defer gpa.free(chol);
    const scratch = try gpa.alloc(f64, m * m);
    defer gpa.free(scratch);
    // `M_R⁻¹x_j` for the block's columns, all against the same post-block `M_R`.
    const zc = try gpa.alloc(f64, blk * m);
    defer gpa.free(zc);

    var st: Stats = .{};
    var b0: usize = 0;
    while (b0 < cols) : (b0 += blk) {
        const b1 = b0 + blk;

        // R = everything after this block, so the whole block leaves M_R at once.
        for (b0..b1) |j| downdate(mr, h.xt[j * m ..][0..m], m);
        try refactor(mr, pinv, chol, scratch, m, h.lambda, &st);
        for (0..blk) |k| matVec(pinv, h.xt[(b0 + k) * m ..][0..m], zc[k * m ..][0..m], m);

        try forEachRowChunk(pool, rows, gatherBlockJob, .{ panel, out, h.xt, acc, m, cols, b0, blk });

        const n_elems: u64 = @intCast(rows * blk);
        const in_bytes = std.mem.sliceAsBytes(panel);
        const quantized = if (weights) |ws|
            try Q.convertTensorDataWeighted(gpa, in_bytes, .F32, dst, n_elems, pool, ws[b0..b1])
        else
            try Q.convertTensorData(gpa, in_bytes, .F32, dst, n_elems, pool);
        defer gpa.free(quantized);
        const back = try Q.convertTensorData(gpa, quantized, dst, .F32, n_elems, pool);
        defer gpa.free(back);

        try forEachRowChunk(pool, rows, commitBlockJob, .{ panel, back, out, zc, acc, m, cols, b0, blk });
    }

    if (p.stats) |o| o.* = st;
    return out;
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/// Spawn `func(args ++ .{start, end})` over `[0, rows)` split across the pool.
fn forEachRowChunk(pool: *ThreadPool, rows: usize, comptime func: anytype, args: anytype) !void {
    return forEachChunk(pool, 0, rows, func, args);
}

fn forEachChunk(
    pool: *ThreadPool,
    from: usize,
    to: usize,
    comptime func: anytype,
    args: anytype,
) !void {
    if (to <= from) return;
    const total = to - from;
    const n = @max(1, @min(pool.threads.len, total));
    const per = total / n;
    const leftover = total - per * n;

    var wg: thread_pool_mod.WaitGroup = .{};
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const start = from + i * per;
        var end = start + per;
        if (i == n - 1) end += leftover;
        if (start == end) continue;
        pool.spawnWg(&wg, func, args ++ .{ start, end });
    }
    wg.wait();
}

fn scaleRowsJob(
    work: []const f32,
    scales: []f32,
    cols: usize,
    rot_w: ?[]const f32,
    grid: Grid,
    start: usize,
    end: usize,
) void {
    for (start..end) |r| {
        const row = work[r * cols ..][0..cols];
        scales[r] = Q.searchScale(row, rot_w, grid.qdiv, grid.qlo, grid.qhi);
    }
}

/// Spawn `sweepRowsJob` over row ranges, giving each thread its own accumulator.
/// Not `forEachRowChunk` because that per-thread scratch is the whole reason the
/// sweep needs no synchronization.
fn sweepRows(
    pool: *ThreadPool,
    work: []f32,
    xt: []const f64,
    z: []const f64,
    scales: []const f32,
    acc: []f64,
    m: usize,
    rows: usize,
    cols: usize,
    grid: Grid,
) !void {
    const n = @max(1, @min(pool.threads.len, rows));
    const per = rows / n;
    const leftover = rows - per * n;

    var wg: thread_pool_mod.WaitGroup = .{};
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const start = i * per;
        var end = start + per;
        if (i == n - 1) end += leftover;
        if (start == end) continue;
        pool.spawnWg(&wg, sweepRowsJob, .{
            work, xt, z, scales, acc[i * m ..][0..m], m, cols, grid, start, end,
        });
    }
    wg.wait();
}

/// `panel[r][k] = W[r][b0+k] − x_{b0+k}ᵗ·acc_r` — the block as GPTQ wants ggml to
/// see it. Reads the *original* weights: `out` still holds them ahead of `b0`,
/// because the compensation lives in the accumulator rather than in the array.
fn gatherBlockJob(
    panel: []f32,
    out: []const f32,
    xt: []const f64,
    acc: []const f64,
    m: usize,
    cols: usize,
    b0: usize,
    blk: usize,
    start: usize,
    end: usize,
) void {
    for (start..end) |r| {
        const a = acc[r * m ..][0..m];
        const src = out[r * cols + b0 ..][0..blk];
        const dst = panel[r * blk ..][0..blk];
        for (0..blk) |k| {
            const xj = xt[(b0 + k) * m ..][0..m];
            var comp: f64 = 0;
            for (xj, a) |xv, av| comp += xv * av;
            dst[k] = @floatCast(@as(f64, src[k]) - comp);
        }
    }
}

/// Take ggml's answer for the block, record it, and fold the block's error into the
/// accumulator: `acc_r += Σ_k e_r[k]·(M_R⁻¹x_{S_k})`.
fn commitBlockJob(
    panel: []const f32,
    back: []const u8,
    out: []f32,
    zc: []const f64,
    acc: []f64,
    m: usize,
    cols: usize,
    b0: usize,
    blk: usize,
    start: usize,
    end: usize,
) void {
    for (start..end) |r| {
        const a = acc[r * m ..][0..m];
        const src = panel[r * blk ..][0..blk];
        const dst = out[r * cols + b0 ..][0..blk];
        for (0..blk) |k| {
            const dq = readF32(back, r * blk + k);
            dst[k] = dq;
            const err: f64 = @as(f64, dq) - @as(f64, src[k]);
            if (err == 0) continue;
            const zk = zc[k * m ..][0..m];
            for (a, zk) |*o, zv| o.* += err * zv;
        }
    }
}

/// Native-endian f32 out of an unaligned byte buffer — allocator buffers carry no
/// 4-byte alignment guarantee. Same helper `precision_harness` needs, for the same
/// reason.
inline fn readF32(bytes: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
}

fn sweepRowsJob(
    work: []f32,
    xt: []const f64,
    z: []const f64,
    scales: []const f32,
    acc: []f64,
    m: usize,
    cols: usize,
    grid: Grid,
    start: usize,
    end: usize,
) void {
    for (start..end) |r| {
        const row = work[r * cols ..][0..cols];
        const s = scales[r];
        // Σ_{i<j} e_i z_i — the entire history of this row's committed error,
        // compressed into m numbers because that is the rank of the Gram.
        @memset(acc, 0);

        for (0..cols) |j| {
            const xj = xt[j * m ..][0..m];
            var comp: f64 = 0;
            for (xj, acc) |a, b| comp += a * b;

            const v: f32 = @floatCast(@as(f64, row[j]) - comp);
            // The same rounding rule and the same clamp as the shipped quantizer;
            // only the value being rounded has moved.
            const q = std.math.clamp(Q.roundHalfToEven(v / s), grid.qlo, grid.qhi);
            const dq = s * q;
            row[j] = q;

            const err: f64 = @as(f64, dq) - @as(f64, v);
            if (err == 0) continue;
            const zj = z[j * m ..][0..m];
            for (acc, zj) |*o, zv| o.* += err * zv;
        }
    }
}

// ---------------------------------------------------------------------------
// m × m linear algebra
// ---------------------------------------------------------------------------

/// `a -= x xᵗ`, keeping `a` symmetric.
fn downdate(a: []f64, x: []const f64, m: usize) void {
    for (0..m) |i| {
        const xi = x[i];
        if (xi == 0) continue;
        const row = a[i * m ..][0..m];
        for (0..m) |j| row[j] -= xi * x[j];
    }
}

fn matVec(a: []const f64, x: []const f64, out: []f64, m: usize) void {
    for (0..m) |i| {
        const row = a[i * m ..][0..m];
        var acc: f64 = 0;
        for (row, x) |va, vx| acc += va * vx;
        out[i] = acc;
    }
}

/// `pinv = mr⁻¹`, via Cholesky. `mr` is symmetric positive definite by
/// construction (a ridge plus a Gram), so a failure means the ridge has been lost
/// to rounding — in which case it is raised rather than the whole layer abandoned.
fn refactor(
    mr: []const f64,
    pinv: []f64,
    chol: []f64,
    scratch: []f64,
    m: usize,
    lambda: f64,
    st: *Stats,
) !void {
    st.refactors += 1;
    var bump: f64 = 0;
    var attempt: usize = 0;
    while (true) : (attempt += 1) {
        @memcpy(chol, mr);
        if (bump > 0) for (0..m) |i| {
            chol[i * m + i] += bump;
        };
        if (cholesky(chol, m)) {
            invertFromChol(chol, pinv, scratch, m);
            return;
        }
        if (attempt >= 4) return error.NotPositiveDefinite;
        st.ridge_bumps += 1;
        bump = if (bump == 0) lambda else bump * 16;
    }
}

/// In-place lower Cholesky of a symmetric `m × m` row-major matrix. Returns false
/// on a non-positive pivot; the upper triangle is left as scratch.
fn cholesky(a: []f64, m: usize) bool {
    for (0..m) |j| {
        const rj = a[j * m ..][0..m];
        var d = rj[j];
        for (0..j) |k| d -= rj[k] * rj[k];
        if (!(d > 0)) return false;
        const l = @sqrt(d);
        rj[j] = l;
        const inv_l = 1.0 / l;
        for (j + 1..m) |i| {
            const ri = a[i * m ..][0..m];
            var s = ri[j];
            for (0..j) |k| s -= ri[k] * rj[k];
            ri[j] = s * inv_l;
        }
    }
    return true;
}

/// `out = (L Lᵗ)⁻¹` given `L` lower in `l`. `scratch` holds `L⁻¹`.
fn invertFromChol(l: []const f64, out: []f64, scratch: []f64, m: usize) void {
    // L⁻¹ by forward substitution, one column at a time. Lower triangular, so
    // entries above the diagonal are never touched or read.
    for (0..m) |c| {
        for (c..m) |i| {
            const ri = l[i * m ..][0..m];
            var s: f64 = if (i == c) 1.0 else 0.0;
            for (c..i) |k| s -= ri[k] * scratch[k * m + c];
            scratch[i * m + c] = s / ri[i];
        }
    }
    // out = (L⁻¹)ᵗ (L⁻¹), summing only over k ≥ max(a, b) where L⁻¹ is nonzero.
    for (0..m) |a| {
        for (a..m) |b| {
            var acc: f64 = 0;
            for (b..m) |k| acc += scratch[k * m + a] * scratch[k * m + b];
            out[a * m + b] = acc;
            out[b * m + a] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const ph = @import("precision_harness.zig");

fn testPool() ThreadPool {
    return .{}; // single job; every result here is split-invariant
}

fn fill(buf: []f32, rnd: std.Random, scale: f32) void {
    for (buf) |*v| v.* = rnd.floatNorm(f32) * scale;
}

/// `‖(W − Ŵ)Xᵗ‖²_F` — the quantity GPTQ minimizes and the only honest way to
/// score it, since it is defined on the activations rather than on the weights.
fn outputSq(w: []const f32, w_hat: []const f32, x: []const f32, m: usize, rows: usize, cols: usize) f64 {
    var acc: f64 = 0;
    for (0..rows) |r| {
        for (0..m) |t| {
            var d: f64 = 0;
            for (0..cols) |c| {
                d += (@as(f64, w[r * cols + c]) - @as(f64, w_hat[r * cols + c])) * @as(f64, x[t * cols + c]);
            }
            acc += d * d;
        }
    }
    return acc;
}

/// A deliberately naive oracle: build the dense `H = λI + XᵗX`, and at each
/// column solve `H_RR c = H_R,i` outright. This is the *definition* of the update
/// rather than a second copy of the Woodbury derivation, which is the only kind
/// of reference worth pinning the fast path against.
fn denseReference(
    gpa: std.mem.Allocator,
    x: []const f32,
    m: usize,
    w: []const f32,
    rows: usize,
    cols: usize,
    grid: Grid,
    damp: f32,
) ![]f32 {
    const hm = try gpa.alloc(f64, cols * cols);
    defer gpa.free(hm);
    var trace: f64 = 0;
    for (0..cols) |i| {
        for (0..cols) |j| {
            var acc: f64 = 0;
            for (0..m) |t| acc += @as(f64, x[t * cols + i]) * @as(f64, x[t * cols + j]);
            hm[i * cols + j] = acc;
        }
        trace += hm[i * cols + i];
    }
    const lambda = @as(f64, damp) * trace / @as(f64, @floatFromInt(cols));
    for (0..cols) |i| hm[i * cols + i] += lambda;

    const out = try gpa.dupe(f32, w);
    errdefer gpa.free(out);

    const a = try gpa.alloc(f64, cols * cols);
    defer gpa.free(a);
    const sol = try gpa.alloc(f64, cols);
    defer gpa.free(sol);
    const c = try gpa.alloc(f32, cols);
    defer gpa.free(c);

    const scales = try gpa.alloc(f32, rows);
    defer gpa.free(scales);
    for (0..rows) |r| {
        scales[r] = Q.searchScale(w[r * cols ..][0..cols], null, grid.qdiv, grid.qlo, grid.qhi);
    }

    for (0..cols) |i| {
        const n = cols - i - 1;
        if (n > 0) {
            // Gaussian elimination on H_RR c = H_R,i, R = {i+1 … cols−1}.
            for (0..n) |a_i| {
                for (0..n) |a_j| a[a_i * n + a_j] = hm[(i + 1 + a_i) * cols + (i + 1 + a_j)];
                sol[a_i] = hm[(i + 1 + a_i) * cols + i];
            }
            for (0..n) |k| {
                var piv = k;
                for (k + 1..n) |r2| {
                    if (@abs(a[r2 * n + k]) > @abs(a[piv * n + k])) piv = r2;
                }
                if (piv != k) {
                    for (0..n) |cc| std.mem.swap(f64, &a[k * n + cc], &a[piv * n + cc]);
                    std.mem.swap(f64, &sol[k], &sol[piv]);
                }
                const d = a[k * n + k];
                for (k + 1..n) |r2| {
                    const f = a[r2 * n + k] / d;
                    if (f == 0) continue;
                    for (k..n) |cc| a[r2 * n + cc] -= f * a[k * n + cc];
                    sol[r2] -= f * sol[k];
                }
            }
            var k = n;
            while (k > 0) {
                k -= 1;
                var s = sol[k];
                for (k + 1..n) |cc| s -= a[k * n + cc] * sol[cc];
                sol[k] = s / a[k * n + k];
            }
            for (0..n) |a_i| c[i + 1 + a_i] = @floatCast(sol[a_i]);
        }

        for (0..rows) |r| {
            const row = out[r * cols ..][0..cols];
            const s = scales[r];
            const v = row[i];
            const q = std.math.clamp(Q.roundHalfToEven(v / s), grid.qlo, grid.qhi);
            row[i] = s * q;
            const err = row[i] - v;
            if (err == 0) continue;
            for (i + 1..cols) |j| row[j] -= err * c[j];
        }
    }
    return out;
}

test "the low-rank sweep matches a dense H_RR solve" {
    // The whole module rests on c_i = X_Rᵗ M_R⁻¹ x_i being the same vector as
    // H_RR⁻¹ H_R,i. This is that claim, checked against an oracle that solves the
    // system directly at every column.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 6;
    const rows = 5;
    const cols = 24;

    var prng = std.Random.DefaultPrng.init(0x8C_0DE);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(x, rnd, 1.0);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    var h = try Hessian.init(gpa, x, m, cols, .{}, default_damp, &pool);
    defer h.deinit();

    inline for ([_]Grid{ int4_grid, int8_grid }) |grid| {
        const fast = try roundtrip(gpa, &h, w, rows, cols, grid, null, &pool, .{});
        defer gpa.free(fast);
        const ref = try denseReference(gpa, x, m, w, rows, cols, grid, default_damp);
        defer gpa.free(ref);

        var num: f64 = 0;
        var den: f64 = 0;
        for (fast, ref) |a, b| {
            const d = @as(f64, a) - @as(f64, b);
            num += d * d;
            den += @as(f64, b) * @as(f64, b);
        }
        try testing.expect(@sqrt(num / @max(den, 1e-30)) < 1e-5);
    }
}

test "the sweep is bit-identical however the rows are split across threads" {
    // Plan §5 hygiene rule 2: "deterministic" has to be a test, not a hope. Each
    // output row carries its own accumulator and writes only its own weights, so
    // the thread count cannot reach the arithmetic — and this is what pins that,
    // since a shared accumulator or a shared coefficient buffer would break it in a
    // way that only shows up as an unreproducible model.
    //
    // `expectEqualSlices`, not an approximate compare: the claim is exactness.
    const gpa = testing.allocator;
    const m = 5;
    const rows = 37; // prime, so no thread count divides it evenly
    const cols = 64;

    var prng = std.Random.DefaultPrng.init(0xB10C);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(x, rnd, 1.0);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    var one = testPool();
    var h = try Hessian.init(gpa, x, m, cols, .{}, default_damp, &one);
    defer h.deinit();
    const ref = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &one, .{});
    defer gpa.free(ref);

    for ([_]usize{ 2, 3, 8, 64 }) |jobs| {
        var pool: ThreadPool = undefined;
        try pool.init(.{ .allocator = gpa, .n_jobs = jobs });
        defer pool.deinit();
        const got = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &pool, .{});
        defer gpa.free(got);
        try testing.expectEqualSlices(f32, ref, got);
    }
}

test "an uncorrelated sample leaves the shipped quantizer alone" {
    // With orthogonal activation columns G is diagonal, there is nothing off the
    // diagonal to exploit, and c_i is zero: GPTQ must degenerate to RTN exactly.
    // This is the boundary where §8C's information advantage over §8A vanishes.
    const gpa = testing.allocator;
    var pool = testPool();
    const cols = 16;
    const rows = 4;

    const x = try gpa.alloc(f32, cols * cols);
    defer gpa.free(x);
    @memset(x, 0);
    for (0..cols) |j| x[j * cols + j] = @floatFromInt(j + 1); // orthogonal, unequal energies

    var prng = std.Random.DefaultPrng.init(0xD1A6);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, prng.random(), 0.02);

    var h = try Hessian.init(gpa, x, cols, cols, .{}, default_damp, &pool);
    defer h.deinit();

    const got = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &pool, .{});
    defer gpa.free(got);
    const rtn = try ph.roundtrip(.int4, gpa, w, rows, cols, &pool);
    defer gpa.free(rtn);
    try testing.expectEqualSlices(f32, rtn, got);
}

const TC = @import("TensorClusters.zig");

/// The shipped quantizer's round-trip at the group size these tests use, with the
/// §8A weighting if `weights` is given. The thing §8C must reduce to when it has
/// no covariance to spend.
fn shippedRoundtrip(
    gpa: std.mem.Allocator,
    fmt: ph.Format,
    w: []const f32,
    rows: usize,
    cols: usize,
    group: usize,
    weights: ?[]const f32,
    pool: *ThreadPool,
) ![]f32 {
    return switch (fmt) {
        .int4, .int4_convrot => blk: {
            const cr = fmt == .int4_convrot;
            const enc = try Q.quantizeToInt4Weighted(gpa, w, rows, cols, cr, group, 0, pool, weights);
            defer gpa.free(enc.weight);
            defer gpa.free(enc.scale);
            break :blk try TC.dequantizeInt4Raw(enc.weight, enc.scale, rows, cols, cr, group, gpa, pool);
        },
        .int8, .int8_convrot => blk: {
            const cr = fmt == .int8_convrot;
            const enc = try Q.quantizeToInt8Weighted(gpa, w, rows, cols, cr, group, pool, weights);
            defer gpa.free(enc.weight);
            defer gpa.free(enc.scale);
            break :blk try TC.dequantizeInt8ConvrotRaw(enc.weight, enc.scale, rows, cols, cr, group, gpa, pool);
        },
        else => unreachable,
    };
}

test "with no cross terms to spend, GPTQ is the shipped quantizer bit-for-bit" {
    // A sample that excites exactly one column **in the basis the rounding happens
    // in** makes every compensation coefficient identically zero, so this is an
    // exactness check rather than a tolerance one: it pins the grid, the rounding
    // rule and both scale searches (plain and §8A-weighted) against ComfyUI's
    // int4/int8 paths, in the plain and the rotated basis alike.
    //
    // The two bases need different samples, and that asymmetry is the point: a
    // single excited column is *spread across its whole group* by the Hadamard, so
    // the rotated case has to be excited along the direction the rotation collapses
    // — a row of H itself, since H is symmetric and involutive, giving `H·H[0] =
    // e_0`. (Not the all-ones vector: ggufy's H4 is the *regular* Hadamard, whose
    // rows do not sum to ±g, so a constant group excites the whole group.)
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 3;
    const rows = 6;
    const cols = 64;
    const group = 16;

    var prng = std.Random.DefaultPrng.init(0xE7AC7);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, prng.random(), 0.02);

    // A skewed importance vector, so the weighted arm's clip search actually moves
    // the scale off `amax` and the comparison has teeth.
    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    for (imp, 0..) |*v, j| v.* = if (j % 8 == 0) 1e3 else 1e-2;

    const cases = [_]struct { convrot: bool, grid: Grid, fmt: ph.Format }{
        .{ .convrot = false, .grid = int4_grid, .fmt = .int4 },
        .{ .convrot = false, .grid = int8_grid, .fmt = .int8 },
        .{ .convrot = true, .grid = int4_grid, .fmt = .int4_convrot },
        .{ .convrot = true, .grid = int8_grid, .fmt = .int8_convrot },
    };

    const hmat = try Q.buildHadamard(gpa, group);
    defer gpa.free(hmat);

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (cases) |case| {
        @memset(x, 0);
        for (0..m) |t| {
            const amp: f32 = @floatFromInt(t + 1);
            if (case.convrot) {
                for (0..group) |j| x[t * cols + j] = amp * hmat[j];
            } else {
                x[t * cols + 0] = amp;
            }
        }

        const basis: Basis = .{ .convrot = case.convrot, .group_size = group };
        var h = try Hessian.init(gpa, x, m, cols, basis, default_damp, &pool);
        defer h.deinit();

        for ([_]?[]const f32{ null, imp }) |weights| {
            const got = try roundtrip(gpa, &h, w, rows, cols, case.grid, weights, &pool, .{});
            defer gpa.free(got);
            const ref = try shippedRoundtrip(gpa, case.fmt, w, rows, cols, group, weights, &pool);
            defer gpa.free(ref);
            try testing.expectEqualSlices(f32, ref, got);
        }
    }
}

test "compensation lowers the output error it is defined on" {
    // The reason to build any of this. On a low-rank correlated sample — which is
    // what a 96-row calibration block of a real layer is — GPTQ must beat RTN on
    // ‖(W−Ŵ)Xᵗ‖², and by a margin, not by rounding noise.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 8;
    const rows = 16;
    const cols = 64;

    var prng = std.Random.DefaultPrng.init(0x0B7EC);
    const rnd = prng.random();

    // Correlated columns: three latent factors mixed into 64 channels, which is
    // the structure that gives the off-diagonals something to say.
    const factors = 3;
    const load = try gpa.alloc(f32, factors * cols);
    defer gpa.free(load);
    fill(load, rnd, 1.0);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (0..m) |t| {
        var f_amp: [factors]f32 = undefined;
        for (&f_amp) |*v| v.* = rnd.floatNorm(f32);
        for (0..cols) |c| {
            var acc: f32 = 0;
            for (0..factors) |f| acc += f_amp[f] * load[f * cols + c];
            x[t * cols + c] = acc + rnd.floatNorm(f32) * 0.05;
        }
    }

    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    inline for ([_]struct { basis: Basis, grid: Grid, fmt: ph.Format }{
        .{ .basis = .{}, .grid = int4_grid, .fmt = .int4 },
        .{ .basis = .{ .convrot = true, .group_size = 16 }, .grid = int4_grid, .fmt = .int4_convrot },
    }) |case| {
        var h = try Hessian.init(gpa, x, m, cols, case.basis, default_damp, &pool);
        defer h.deinit();
        const gptq = try roundtrip(gpa, &h, w, rows, cols, case.grid, null, &pool, .{});
        defer gpa.free(gptq);
        const rtn = try ph.roundtrip(case.fmt, gpa, w, rows, cols, &pool);
        defer gpa.free(rtn);

        const e_gptq = outputSq(w, gptq, x, m, rows, cols);
        const e_rtn = outputSq(w, rtn, x, m, rows, cols);
        try testing.expect(e_gptq < 0.5 * e_rtn);
    }
}

test "the ridge is what bounds the compensation" {
    // λ → large drives c_i → 0, so a heavily damped sweep converges back onto the
    // plain quantizer. Stated as a limit rather than an equality because the
    // approach is asymptotic: this pins the direction and that it is monotone in
    // damp, which is what makes `damp` a meaningful knob rather than a magic one.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 8;
    const rows = 8;
    const cols = 32;

    var prng = std.Random.DefaultPrng.init(0x21D6E);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(x, rnd, 1.0);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    const rtn = try ph.roundtrip(.int4, gpa, w, rows, cols, &pool);
    defer gpa.free(rtn);

    var prev: f64 = std.math.inf(f64);
    for ([_]f32{ 0.01, 1.0, 100.0, 1e5 }) |damp| {
        var h = try Hessian.init(gpa, x, m, cols, .{}, damp, &pool);
        defer h.deinit();
        const got = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &pool, .{});
        defer gpa.free(got);

        var dist: f64 = 0;
        for (got, rtn) |a, b| {
            const d = @as(f64, a) - @as(f64, b);
            dist += d * d;
        }
        try testing.expect(dist <= prev);
        prev = dist;
    }
    // ...and by the far end of that sweep it has effectively arrived.
    try testing.expect(prev < 1e-12);
}

test "the two mechanisms stack without either being a no-op" {
    // §8A steers where the grid sits, §8C which level each weight lands on, and
    // `weights` reaches `searchScale` and nothing else. So all four combinations
    // have to be distinct: if weighting the GPTQ arm changed nothing, the stacked
    // measurement would silently be the unstacked one.
    //
    // Note what is deliberately *not* asserted: that flat weights reproduce the
    // unweighted arm. They do not, and should not — the clip search can beat
    // `amax` on plain unweighted squared error too, so `null` (no search) and a
    // flat weight vector (search, uniform importance) are different experiments.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 6;
    const rows = 4;
    const cols = 32;

    var prng = std.Random.DefaultPrng.init(0x8A_8C);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(x, rnd, 1.0);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);
    // One channel carrying a big outlier weight, so clipping it is a real choice.
    for (0..rows) |r| w[r * cols + 7] *= 40;

    var h = try Hessian.init(gpa, x, m, cols, .{}, default_damp, &pool);
    defer h.deinit();

    const skewed = try gpa.alloc(f32, cols);
    defer gpa.free(skewed);
    for (skewed, 0..) |*v, j| v.* = if (j == 7) 1e-3 else 1.0;

    const rtn = try shippedRoundtrip(gpa, .int4, w, rows, cols, 16, null, &pool);
    defer gpa.free(rtn);
    const rtn_w = try shippedRoundtrip(gpa, .int4, w, rows, cols, 16, skewed, &pool);
    defer gpa.free(rtn_w);
    const gptq = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &pool, .{});
    defer gpa.free(gptq);
    const gptq_w = try roundtrip(gpa, &h, w, rows, cols, int4_grid, skewed, &pool, .{});
    defer gpa.free(gptq_w);

    // §8A moves the grid on both the plain and the compensated arm...
    try testing.expect(!std.mem.eql(f32, rtn, rtn_w));
    try testing.expect(!std.mem.eql(f32, gptq, gptq_w));
    // ...and §8C changes the rounding under both grids.
    try testing.expect(!std.mem.eql(f32, rtn, gptq));
    try testing.expect(!std.mem.eql(f32, rtn_w, gptq_w));
}

test "every compensated weight is still representable in the format" {
    // The arm reports dequantized f32, so nothing else here would notice if the
    // compensation walked a weight off the grid — and a level-1 win that no
    // writable file could reproduce is the worst possible outcome for §8C. GPTQ
    // *does* push values past the original amax (that is what compensation is), so
    // the clamp is load-bearing rather than defensive.
    //
    // Checked in the basis the rounding happens in: after un-rotation a ConvRot
    // weight is a sum of grid points, not a grid point.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 8;
    const rows = 12;
    const cols = 64;
    const group = 16;

    var prng = std.Random.DefaultPrng.init(0x9E11);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (0..m) |t| {
        const shared = rnd.floatNorm(f32) * 4;
        for (0..cols) |c| x[t * cols + c] = shared + rnd.floatNorm(f32) * 0.1;
    }
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    for ([_]bool{ false, true }) |convrot| {
        const basis: Basis = .{ .convrot = convrot, .group_size = group };
        var h = try Hessian.init(gpa, x, m, cols, basis, default_damp, &pool);
        defer h.deinit();
        const got = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &pool, .{});
        defer gpa.free(got);

        // Back into the rounding basis. The Hadamard is its own inverse.
        const grid_view = try gpa.dupe(f32, got);
        defer gpa.free(grid_view);
        if (convrot) try Q.rotateGroupwiseInPlace(grid_view, rows, cols, group, &pool);

        // Recover each row's scale the way the quantizer chose it, from the original
        // weights in the same basis.
        const orig = try gpa.dupe(f32, w);
        defer gpa.free(orig);
        if (convrot) try Q.rotateGroupwiseInPlace(orig, rows, cols, group, &pool);

        var clamped: usize = 0;
        for (0..rows) |r| {
            const s = Q.searchScale(orig[r * cols ..][0..cols], null, int4_grid.qdiv, int4_grid.qlo, int4_grid.qhi);
            for (grid_view[r * cols ..][0..cols]) |v| {
                const level = v / s;
                const nearest = @round(level);
                try testing.expect(@abs(level - nearest) < 1e-3);
                try testing.expect(nearest >= int4_grid.qlo and nearest <= int4_grid.qhi);
                if (@abs(nearest) == int4_grid.qhi) clamped += 1;
            }
        }
        // And the clamp is doing real work on this data, so the bound above is not
        // vacuously satisfied by a sweep that never reached the rails.
        try testing.expect(clamped > 0);
    }
}

test "the codes the converter writes dequantize to what the arm measured" {
    // The load-bearing link between §8C's measurement and §8C's product. The arm
    // scores `roundtrip`'s dequantized f32; `convert` writes `quantizeInt4`'s
    // nibbles. If those two ever disagree, every number in the plan is about a file
    // nobody ships. Checked through `TensorClusters`' real dequantizer, so the
    // packing and the scale placement are exercised, not just the levels.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 8;
    const rows = 16;
    const cols = 128;
    const group = 64; // Hadamard orders are powers of four, not two

    var prng = std.Random.DefaultPrng.init(0xC0DE5);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (0..m) |t| {
        const shared = rnd.floatNorm(f32) * 3;
        for (0..cols) |c| x[t * cols + c] = shared + rnd.floatNorm(f32) * 0.2;
    }
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);
    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    for (imp, 0..) |*v, j| v.* = if (j % 3 == 0) 9.0 else 0.3;

    for ([_]bool{ false, true }) |convrot| {
        const basis: Basis = .{ .convrot = convrot, .group_size = group };
        var h = try Hessian.init(gpa, x, m, cols, basis, default_damp, &pool);
        defer h.deinit();

        for ([_]?[]const f32{ null, imp }) |weights| {
            const measured = try roundtrip(gpa, &h, w, rows, cols, int4_grid, weights, &pool, .{});
            defer gpa.free(measured);

            const codes = try quantizeInt4(gpa, &h, w, rows, cols, weights, &pool, .{});
            defer gpa.free(codes.weight);
            defer gpa.free(codes.scale);
            const written = try TC.dequantizeInt4Raw(codes.weight, codes.scale, rows, cols, convrot, group, gpa, &pool);
            defer gpa.free(written);

            try testing.expectEqualSlices(f32, measured, written);
        }

        // Same for int8, whose packing is a plain i8 cast rather than nibbles.
        const measured8 = try roundtrip(gpa, &h, w, rows, cols, int8_grid, imp, &pool, .{});
        defer gpa.free(measured8);
        const codes8 = try quantizeInt8(gpa, &h, w, rows, cols, imp, &pool, .{});
        defer gpa.free(codes8.weight);
        defer gpa.free(codes8.scale);
        const written8 = try TC.dequantizeInt8ConvrotRaw(codes8.weight, codes8.scale, rows, cols, convrot, group, gpa, &pool);
        defer gpa.free(written8);
        try testing.expectEqualSlices(f32, measured8, written8);
    }
}

test "with nothing to compensate the emitted codes are the shipped quantizer's" {
    // The converter-side counterpart of the roundtrip degeneracy test: not just the
    // same values, the same *bytes*, so `--gptq` on a layer GPTQ cannot help writes
    // a file byte-identical to `--calib` alone.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 3;
    const rows = 6;
    const cols = 64;
    const group = 16;

    var prng = std.Random.DefaultPrng.init(0x5A11E);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, prng.random(), 0.02);
    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    for (imp, 0..) |*v, j| v.* = if (j % 8 == 0) 1e3 else 1e-2;

    const hmat = try Q.buildHadamard(gpa, group);
    defer gpa.free(hmat);
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);

    for ([_]bool{ false, true }) |convrot| {
        @memset(x, 0);
        for (0..m) |t| {
            const amp: f32 = @floatFromInt(t + 1);
            if (convrot) {
                for (0..group) |j| x[t * cols + j] = amp * hmat[j];
            } else x[t * cols + 0] = amp;
        }
        var h = try Hessian.init(gpa, x, m, cols, .{ .convrot = convrot, .group_size = group }, default_damp, &pool);
        defer h.deinit();

        const got = try quantizeInt4(gpa, &h, w, rows, cols, imp, &pool, .{});
        defer gpa.free(got.weight);
        defer gpa.free(got.scale);
        const ref = try Q.quantizeToInt4Weighted(gpa, w, rows, cols, convrot, group, 0, &pool, imp);
        defer gpa.free(ref.weight);
        defer gpa.free(ref.scale);

        try testing.expectEqualSlices(u8, ref.weight, got.weight);
        try testing.expectEqualSlices(f32, ref.scale, got.scale);
    }
}

test "the ggml path degenerates to the shipped encoder bit-for-bit" {
    // The whole reason `roundtripGgml` hands ggml one block at a time instead of
    // reconstructing its grid: a super-block is self-contained, so block-at-a-time
    // must be byte-identical to whole-row. With a sample that excites one column,
    // the compensation is exactly zero and the two have to agree exactly — which
    // pins both that property and the per-block `quant_weights` slicing, where
    // getting the row width wrong is silent (CLAUDE.md's §8A warning).
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 3;
    const rows = 5;
    const cols = 512; // two q4_k super-blocks, sixteen q4_0 blocks

    var prng = std.Random.DefaultPrng.init(0x66_1E);
    const rnd = prng.random();
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    @memset(x, 0);
    for (0..m) |t| x[t * cols + 0] = @floatFromInt(t + 1);

    const imp = try gpa.alloc(f32, cols);
    defer gpa.free(imp);
    for (imp, 0..) |*v, j| v.* = if (j % 5 == 0) 8.0 else 0.2;

    var h = try Hessian.init(gpa, x, m, cols, .{}, default_damp, &pool);
    defer h.deinit();

    inline for ([_]struct { dt: types.DataType, fmt: ph.Format }{
        .{ .dt = .q4_k, .fmt = .q4_k },
        .{ .dt = .q5_k, .fmt = .q5_k },
        .{ .dt = .q6_k, .fmt = .q6_k },
        .{ .dt = .q3_k, .fmt = .q3_k },
        .{ .dt = .q2_k, .fmt = .q2_k },
    }) |case| {
        for ([_]?[]const f32{ null, imp }) |weights| {
            const got = try roundtripGgml(gpa, &h, w, rows, cols, case.dt, weights, &pool, .{});
            defer gpa.free(got);
            const ref = if (weights) |ws|
                try ph.roundtripWeighted(case.fmt, gpa, w, rows, cols, &pool, ws)
            else
                try ph.roundtrip(case.fmt, gpa, w, rows, cols, &pool);
            defer gpa.free(ref);
            try testing.expectEqualSlices(f32, ref, got);
        }
    }

    // q4_0's *unweighted* encoder is block-local too, so it degenerates exactly...
    const plain = try roundtripGgml(gpa, &h, w, rows, cols, .q4_0, null, &pool, .{});
    defer gpa.free(plain);
    const plain_ref = try ph.roundtrip(.q4_0, gpa, w, rows, cols, &pool);
    defer gpa.free(plain_ref);
    try testing.expectEqualSlices(f32, plain_ref, plain);

    // ...but its weighted one normalizes over `n_per_row`, so block-at-a-time would
    // be a different encoder and the arm refuses rather than quietly confounding a
    // compensation measurement with an encoder change.
    try testing.expectError(
        error.RowCoupledWeightedEncoder,
        roundtripGgml(gpa, &h, w, rows, cols, .q4_0, imp, &pool, .{}),
    );
}

test "block-locality matches what ggml-quants.c actually does" {
    // Pinned deliberately, the same way `Imatrix.readsImatrix` is: if a ggml bump
    // moves a k-quant's sigma2 normalization to the row, §8C's GGUF arm silently
    // starts measuring an encoder change on top of its compensation.
    for ([_]gguf.GgmlType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k }) |t|
        try testing.expect(ggmlBlockLocalWeighted(t));
    for ([_]gguf.GgmlType{ .q4_0, .q4_1, .q5_0, .q5_1 }) |t|
        try testing.expect(!ggmlBlockLocalWeighted(t));

    // Every block-local type must also be one ggml actually reads an imatrix for,
    // or the guard is protecting something that never happens.
    const Imatrix = @import("Imatrix.zig");
    for ([_]gguf.GgmlType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k }) |t|
        try testing.expect(Imatrix.readsImatrix(t));
}

test "the block-set compensation matches a dense H_RR solve" {
    // `roundtrip`'s oracle checks the single-column update. This checks the other
    // one: for a set S quantized at once, d_R = −H_RR⁻¹H_RS e_S, with R everything
    // *after the block* rather than after each column. Getting that wrong — using
    // the per-column residual matrix inside a block, say — would still produce a
    // plausible improvement, just not the optimal one.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 5;
    const rows = 3;
    const cols = 64; // two q4_0 blocks of 32
    const blk = 32;

    var prng = std.Random.DefaultPrng.init(0xB10CE7);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(x, rnd, 1.0);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    var h = try Hessian.init(gpa, x, m, cols, .{}, default_damp, &pool);
    defer h.deinit();
    const got = try roundtripGgml(gpa, &h, w, rows, cols, .q4_0, null, &pool, .{});
    defer gpa.free(got);

    // Dense reference: build H, and for each block solve H_RR C = H_RS outright,
    // driving the same ggml encoder on the same compensated values.
    const hm = try gpa.alloc(f64, cols * cols);
    defer gpa.free(hm);
    var trace: f64 = 0;
    for (0..cols) |i| {
        for (0..cols) |j| {
            var a: f64 = 0;
            for (0..m) |t| a += @as(f64, x[t * cols + i]) * @as(f64, x[t * cols + j]);
            hm[i * cols + j] = a;
        }
        trace += hm[i * cols + i];
    }
    const lambda = @as(f64, default_damp) * trace / @as(f64, @floatFromInt(cols));
    for (0..cols) |i| hm[i * cols + i] += lambda;

    const ref = try gpa.dupe(f32, w);
    defer gpa.free(ref);
    const panel = try gpa.alloc(f32, rows * blk);
    defer gpa.free(panel);

    var b0: usize = 0;
    while (b0 < cols) : (b0 += blk) {
        const b1 = b0 + blk;
        for (0..rows) |r| @memcpy(panel[r * blk ..][0..blk], ref[r * cols + b0 ..][0..blk]);

        const qb = try Q.convertTensorData(gpa, std.mem.sliceAsBytes(panel), .F32, .q4_0, rows * blk, &pool);
        defer gpa.free(qb);
        const back = try Q.convertTensorData(gpa, qb, .q4_0, .F32, rows * blk, &pool);
        defer gpa.free(back);

        const n = cols - b1;
        for (0..rows) |r| {
            for (0..blk) |k| {
                const dq = readF32(back, r * blk + k);
                const e = @as(f64, dq) - @as(f64, panel[r * blk + k]);
                ref[r * cols + b0 + k] = dq;
                if (n == 0 or e == 0) continue;
                // d_R += −e · (H_RR⁻¹ H_R,(b0+k)), solved directly.
                const a = try gpa.alloc(f64, n * n);
                defer gpa.free(a);
                const sol = try gpa.alloc(f64, n);
                defer gpa.free(sol);
                for (0..n) |i| {
                    for (0..n) |j| a[i * n + j] = hm[(b1 + i) * cols + (b1 + j)];
                    sol[i] = hm[(b1 + i) * cols + (b0 + k)];
                }
                solveInPlace(a, sol, n);
                for (0..n) |i| ref[r * cols + b1 + i] -= @floatCast(e * sol[i]);
            }
        }
    }

    var num: f64 = 0;
    var den: f64 = 0;
    for (got, ref) |a, b| {
        const d = @as(f64, a) - @as(f64, b);
        num += d * d;
        den += @as(f64, b) * @as(f64, b);
    }
    try testing.expect(@sqrt(num / @max(den, 1e-30)) < 1e-5);
}

/// Gaussian elimination with partial pivoting, `a` and `b` clobbered. Test-only —
/// the production path never forms a matrix this size.
fn solveInPlace(a: []f64, b: []f64, n: usize) void {
    for (0..n) |k| {
        var piv = k;
        for (k + 1..n) |r| {
            if (@abs(a[r * n + k]) > @abs(a[piv * n + k])) piv = r;
        }
        if (piv != k) {
            for (0..n) |c| std.mem.swap(f64, &a[k * n + c], &a[piv * n + c]);
            std.mem.swap(f64, &b[k], &b[piv]);
        }
        const d = a[k * n + k];
        for (k + 1..n) |r| {
            const f = a[r * n + k] / d;
            if (f == 0) continue;
            for (k..n) |c| a[r * n + c] -= f * a[k * n + c];
            b[r] -= f * b[k];
        }
    }
    var k = n;
    while (k > 0) {
        k -= 1;
        var s = b[k];
        for (k + 1..n) |c| s -= a[k * n + c] * b[c];
        b[k] = s / a[k * n + k];
    }
}

test "the rotated Gram carries information the rotated diagonal cannot" {
    // Both CLAUDE.md and this module's header claim that §8A's rotated importance
    // is *strictly* weaker than §8C's in the ConvRot basis, because
    // `diag(Hᵀ·diag(G)·H)` is constant within a group by construction (every
    // Hadamard entry has magnitude 1/√g) while `diag(H·G·H)` is not — the
    // difference being exactly the off-diagonal mass. Left untested that is a
    // plausible-sounding story, so here it is as an inequality on real numbers.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 8;
    const cols = 32;
    const group = 16;

    var prng = std.Random.DefaultPrng.init(0x6A6A);
    const rnd = prng.random();
    // Strongly correlated columns — with independent columns G is near-diagonal and
    // there would be nothing for the two to disagree about.
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (0..m) |t| {
        const shared = rnd.floatNorm(f32);
        for (0..cols) |c| x[t * cols + c] = shared * @as(f32, @floatFromInt(c % 5 + 1)) + rnd.floatNorm(f32) * 0.01;
    }

    var h = try Hessian.init(gpa, x, m, cols, .{ .convrot = true, .group_size = group }, default_damp, &pool);
    defer h.deinit();

    // §8A's view of the same layer: per-column energy, mapped into the rotated basis.
    const energy = try gpa.alloc(f32, cols);
    defer gpa.free(energy);
    for (0..cols) |c| {
        var acc: f64 = 0;
        for (0..m) |t| acc += @as(f64, x[t * cols + c]) * @as(f64, x[t * cols + c]);
        energy[c] = @floatCast(acc);
    }
    const rot = (try Q.rotatedWeights(gpa, energy, cols, true, group)).?;
    defer gpa.free(rot);

    // §8A: flat within each group, by construction.
    for (0..cols) |c| try testing.expectApproxEqRel(rot[c - c % group], rot[c], 1e-5);

    // §8C: not flat — and not by a rounding margin. The spread within a group is
    // the off-diagonal mass §8A discards.
    var lo: f64 = std.math.inf(f64);
    var hi: f64 = 0;
    for (0..group) |c| {
        lo = @min(lo, h.diag(c));
        hi = @max(hi, h.diag(c));
    }
    try testing.expect(hi > 10 * lo);

    // Both still describe the same total energy: a rotation is orthogonal, so it
    // moves energy between channels without creating or destroying any.
    var sum_gram: f64 = 0;
    var sum_flat: f64 = 0;
    for (0..cols) |c| {
        sum_gram += h.diag(c);
        sum_flat += rot[c];
    }
    try testing.expectApproxEqRel(sum_flat, sum_gram, 1e-4);
}

test "a layer the capture never excited is refused, not silently damped" {
    const gpa = testing.allocator;
    var pool = testPool();
    const x = try gpa.alloc(f32, 4 * 16);
    defer gpa.free(x);
    @memset(x, 0);
    try testing.expectError(error.NoActivationEnergy, Hessian.init(gpa, x, 4, 16, .{}, default_damp, &pool));
}

test "the sweep stays finite and refactors cleanly on a rank-deficient sample" {
    // cols ≫ m is the real operating point (96 rows against 6144 columns), so the
    // Hessian is always rank-deficient and the ridge is always load-bearing. This
    // also exercises the scheduled refactorization path more than once.
    const gpa = testing.allocator;
    var pool = testPool();
    const m = 4;
    const rows = 3;
    const cols = 256;

    var prng = std.Random.DefaultPrng.init(0x4A11);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    fill(x, rnd, 1.0);
    const w = try gpa.alloc(f32, rows * cols);
    defer gpa.free(w);
    fill(w, rnd, 0.02);

    var h = try Hessian.init(gpa, x, m, cols, .{}, default_damp, &pool);
    defer h.deinit();

    var st: Stats = .{};
    const got = try roundtrip(gpa, &h, w, rows, cols, int4_grid, null, &pool, .{
        .refactor_every = 64,
        .stats = &st,
    });
    defer gpa.free(got);
    for (got) |v| try testing.expect(std.math.isFinite(v));
    try testing.expect(st.refactors >= 4);
    // Drift and degeneracy are both supposed to stay theoretical here.
    try testing.expectEqual(@as(usize, 0), st.ridge_bumps);
    try testing.expectEqual(@as(usize, 0), st.forced_refactors);
}
