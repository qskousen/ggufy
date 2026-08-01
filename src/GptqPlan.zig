//! GptqPlan.zig — the `convert --gptq` side of plan §8C.
//!
//! `Gptq.zig` is the algorithm; this is the policy that decides which tensors it
//! is applied to and feeds it the activations, the same division `Imatrix.zig`
//! has against the calibration cache. Keeping them apart is what lets the level-1
//! harness measure formats and layers the converter declines to touch.
//!
//! Unlike §8A, §8C needs the sampled activation **rows**, not the per-channel
//! energy — so a plan holds the cache open for the whole conversion and builds a
//! Hessian per tensor on demand (a few MB each, discarded immediately). The
//! per-column energy is still needed: it picks the grid, via the same §8A weighted
//! search `convert --calib` already ships, so that `--gptq` changes only which
//! level each weight lands on.
//!
//! ### What it declines, and why
//!
//! **Only the int4/int8 cluster formats**, which are safetensors-only. GGUF gets
//! nothing: §8C reaches the k-quants at block granularity (`Gptq.roundtripGgml`)
//! and level 1 measured that at ~5% with 9% of layers regressing, against 9–19%
//! for `--calib` alone on the same path. Measured and declined, not overlooked.
//!
//! ⚠️ **`cols >= min_cols_per_row_ratio × m`.** GPTQ's compensation is
//! unconstrained once the remaining free columns number fewer than the sampled
//! rows — for the last `m` columns of any layer that is always true, which is
//! harmless when they are a sliver of the row and fatal when they are the row.
//! Measured on krea2: `first` (6144×64) with 63 training rows sat exactly at
//! `m ≈ cols` and **regressed to 1.52**, while the same layer with 95 rows was
//! over-determined and helped (0.835). Everything from `cols = 256` up improved.
//! The guard keeps at least half of every row in the well-posed regime; the layers
//! it excludes fall back to the §8A path, which is what they would have got anyway.
//!
//! Cost: a full INT4_CONVROT conversion of the 26 GB krea2 checkpoint goes from
//! 88 s with `--calib` to roughly 7 minutes with `--gptq` **[projected from the
//! level-1 sweep: ~2·m·rows·cols per layer]**.

const std = @import("std");
const types = @import("types.zig");
const Gptq = @import("Gptq.zig");
const Imatrix = @import("Imatrix.zig");
const CalibrationCache = @import("CalibrationCache.zig");
const TensorClusters = @import("TensorClusters.zig");
const imagearch = @import("ImageArch.zig");
const thread_pool_mod = @import("ThreadPool.zig");

const ThreadPool = thread_pool_mod.ThreadPool;

/// A row needs at least this many columns per sampled token row before the
/// compensation is well posed over most of it. See the header.
pub const min_cols_per_row_ratio: usize = 2;

/// Why a tensor did or did not get compensated. Counted rather than logged per
/// tensor: on a 430-tensor model every arm but `use` is a normal outcome.
pub const Summary = struct {
    use: usize = 0,
    /// Not an int cluster — every ggml type, the MX/NVFP4 clusters, plain fp8.
    unsupported_type: usize = 0,
    /// The cache never saw this tensor, or saw it with a different width.
    no_data: usize = 0,
    /// `cols < min_cols_per_row_ratio × m`, or the format's group constraint fails.
    too_narrow: usize = 0,
    /// The sweep itself failed. Should be zero; counted so it cannot hide.
    failed: usize = 0,
};

pub const Plan = struct {
    gpa: std.mem.Allocator,
    cache: *const CalibrationCache.Cache,
    imatrix: *const Imatrix.Imatrix,
    damp: f32 = Gptq.default_damp,
    summary: Summary = .{},

    /// Compensated cluster codes for `t`, or null when policy or shape declines it
    /// — in which case the caller quantizes as it would have without `--gptq`.
    ///
    /// `f32_data` is the dequantized source weight the writer already holds, so
    /// this adds no extra read of the model.
    pub fn quantize(
        self: *Plan,
        gpa: std.mem.Allocator,
        t: types.Tensor,
        f32_data: []const f32,
        pool: *ThreadPool,
    ) !?types.ClusterCodes {
        const dt = types.DataType.fromString(t.type) catch {
            self.summary.unsupported_type += 1;
            return null;
        };
        const kind: enum { int4, int8 } = switch (dt) {
            .INT4_CONVROT, .INT4_CONVROT_SR => .int4,
            .INT8, .INT8_CONVROT => .int8,
            else => {
                self.summary.unsupported_type += 1;
                return null;
            },
        };
        if (t.dims.len != 2) {
            self.summary.unsupported_type += 1;
            return null;
        }
        const rows = t.dims[0];
        const cols = t.dims[1];
        if (f32_data.len != rows * cols) {
            self.summary.no_data += 1;
            return null;
        }

        // The grid comes from §8A, exactly as `--calib` would have chosen it, so
        // the only thing `--gptq` changes is the level each weight rounds to.
        const weights = self.imatrix.forTensor(t);

        var x_cols: usize = 0;
        const x = self.gatherRows(gpa, t.name, &x_cols) catch null orelse {
            self.summary.no_data += 1;
            return null;
        };
        defer gpa.free(x);
        if (x_cols != cols or x.len == 0) {
            self.summary.no_data += 1;
            return null;
        }
        const m = x.len / cols;

        const convrot = dt != .INT8;
        const group: usize = @intCast(if (dt == .INT8_CONVROT)
            TensorClusters.int8_convrot_group_size
        else
            TensorClusters.int4_convrot_group_size);

        if (cols < min_cols_per_row_ratio * m or
            (convrot and (group == 0 or cols % group != 0)) or
            (kind == .int4 and cols % 2 != 0))
        {
            self.summary.too_narrow += 1;
            return null;
        }

        var h = Gptq.Hessian.init(gpa, x, m, cols, .{
            .convrot = convrot,
            .group_size = group,
        }, self.damp, pool) catch {
            // A layer the capture never excited has no covariance to spend.
            self.summary.no_data += 1;
            return null;
        };
        defer h.deinit();

        const codes: types.ClusterCodes = switch (kind) {
            .int4 => blk: {
                const c = Gptq.quantizeInt4(gpa, &h, f32_data, rows, cols, weights, pool, .{ .damp = self.damp }) catch {
                    self.summary.failed += 1;
                    return null;
                };
                break :blk .{ .weight = c.weight, .scale = c.scale };
            },
            .int8 => blk: {
                const c = Gptq.quantizeInt8(gpa, &h, f32_data, rows, cols, weights, pool, .{ .damp = self.damp }) catch {
                    self.summary.failed += 1;
                    return null;
                };
                break :blk .{ .weight = c.weight, .scale = c.scale };
            },
        };
        self.summary.use += 1;
        return codes;
    }

    /// Every sampled row the cache holds for this tensor, buckets concatenated —
    /// the same block `Sensitivity.gatherX` builds, keyed the way the converter
    /// keys tensors.
    fn gatherRows(self: *const Plan, gpa: std.mem.Allocator, name: []const u8, cols: *usize) !?[]f32 {
        const key = self.cacheKey(name) orelse return null;
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(gpa);
        for (0..self.cache.prov.buckets) |k| {
            const b = self.cache.bucket(key, k) catch return null;
            if (b.kept == 0) continue;
            if (cols.* == 0) cols.* = b.cols else if (b.cols != cols.*) return null;
            const rows = try b.rowsAlloc(gpa);
            defer gpa.free(rows);
            try out.appendSlice(gpa, rows);
        }
        if (out.items.len == 0) return null;
        return try out.toOwnedSlice(gpa);
    }

    /// Cache keys are checkpoint names, which may carry a container prefix the
    /// converter has already stripped. Try the name as given, then every cache
    /// layer whose stripped form matches — the same namespace problem
    /// `Imatrix.fromCache` solves by stripping on the way in.
    fn cacheKey(self: *const Plan, name: []const u8) ?[]const u8 {
        if (self.cache.bucket(name, 0)) |_| return name else |_| {}
        const want = imagearch.stripPrefix(name);
        for (self.cache.layers()) |l| {
            if (std.mem.eql(u8, imagearch.stripPrefix(l), want)) return l;
        }
        return null;
    }
};

fn planQuantize(
    ctx: *anyopaque,
    gpa: std.mem.Allocator,
    t: types.Tensor,
    f32_data: []const f32,
    pool: *anyopaque,
) anyerror!?types.ClusterCodes {
    const self: *Plan = @alignCast(@ptrCast(ctx));
    return self.quantize(gpa, t, f32_data, @alignCast(@ptrCast(pool)));
}

pub fn lookup(p: *Plan) types.GptqLookup {
    return .{ .ctx = p, .quantize = planQuantize };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const Activations = @import("Activations.zig");
const tp = @import("TensorPencil");

/// Drive one tagged GEMM through TensorPencil so the collector sees `x` the way a
/// real forward delivers it — the same helper `Imatrix`'s tests use.
fn observe(collector: *Activations.Collector, tag: []const u8, x: []const f32, m: usize, cols: usize) !void {
    const gpa = testing.allocator;
    const out_rows = 2;
    const wdata = try gpa.alloc(f32, out_rows * cols);
    defer gpa.free(wdata);
    @memset(wdata, 0.25);
    const y = try gpa.alloc(f32, m * out_rows);
    defer gpa.free(y);

    var w = tp.ops.matmul.Weight.fromF32(wdata, out_rows, cols);
    w.tag = tag;
    const prev = tp.ops.matmul.probe;
    tp.ops.matmul.probe = collector.probe();
    defer tp.ops.matmul.probe = prev;
    try tp.ops.matmul.matmul(testing.io, gpa, y, x, m, w, null);
}

const test_prov: CalibrationCache.Provenance = .{
    .model_path = "test.safetensors",
    .model_hash = "deadbeef",
    .arch = "krea2",
    .prompt_set = "test",
    .backend = "cpu",
    .producer = "ggufy-test",
    .resolution = 512,
    .steps = 1,
    .seed = 1234,
};

test "the plan compensates a wide layer and declines a narrow one" {
    // Both halves of the policy in one fixture: a layer with plenty of columns per
    // sampled row is compensated (and its codes differ from the §8A-only ones, or
    // `--gptq` would be a no-op), while one at `cols ≈ m` is declined and left to
    // the path it would have taken anyway.
    const gpa = testing.allocator;
    const m = 8;
    const cols = 256;
    const narrow_cols = 8; // 8 < 2 × 8 rows

    var c = Activations.Collector.init(gpa, .{ .sample_rows = 16, .buckets = 1 });
    defer c.deinit();

    var prng = std.Random.DefaultPrng.init(0x9A1);
    const rnd = prng.random();
    const x = try gpa.alloc(f32, m * cols);
    defer gpa.free(x);
    for (0..m) |t| {
        const shared = rnd.floatNorm(f32) * 3;
        for (0..cols) |j| x[t * cols + j] = shared + rnd.floatNorm(f32) * 0.2;
    }
    try observe(&c, "blocks.0.attn.wq.weight", x, m, cols);

    const xn = try gpa.alloc(f32, m * narrow_cols);
    defer gpa.free(xn);
    for (xn) |*v| v.* = rnd.floatNorm(f32);
    try observe(&c, "blocks.0.attn.wk.weight", xn, m, narrow_cols);
    try c.checkOk();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try CalibrationCache.write(gpa, &aw.writer, &c, test_prov);
    const bytes = try gpa.dupe(u8, aw.written());
    defer gpa.free(bytes);
    const st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes);
    var cache = try CalibrationCache.Cache.init(gpa, st);
    defer cache.deinit();

    var im = try Imatrix.fromCache(gpa, &cache);
    defer im.deinit();

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();

    var plan: Plan = .{ .gpa = gpa, .cache = &cache, .imatrix = &im };

    const out_rows = 4;
    const w = try gpa.alloc(f32, out_rows * cols);
    defer gpa.free(w);
    for (w) |*v| v.* = rnd.floatNorm(f32) * 0.02;

    var dims = [_]usize{ out_rows, cols };
    const t: types.Tensor = .{ .name = "blocks.0.attn.wq.weight", .type = "INT4_CONVROT", .dims = &dims, .size = 0, .offset = 0 };

    // 256 columns against 8 rows clears the guard.
    const codes = (try plan.quantize(gpa, t, w, &pool)).?;
    defer gpa.free(codes.weight);
    defer gpa.free(codes.scale);
    try testing.expectEqual(@as(usize, 1), plan.summary.use);

    // ...and it is doing something: the §8A-only codes for the same tensor differ.
    const Q = @import("DataTransform.zig").Quantizer;
    const group: usize = @intCast(TensorClusters.int4_convrot_group_size);
    const plain = try Q.quantizeToInt4Weighted(gpa, w, out_rows, cols, true, group, 0, &pool, im.forTensor(t));
    defer gpa.free(plain.weight);
    defer gpa.free(plain.scale);
    try testing.expect(!std.mem.eql(u8, plain.weight, codes.weight));

    // The narrow layer is declined, not compensated badly.
    var ndims = [_]usize{ out_rows, narrow_cols };
    const nt: types.Tensor = .{ .name = "blocks.0.attn.wk.weight", .type = "INT4_CONVROT", .dims = &ndims, .size = 0, .offset = 0 };
    const wn = try gpa.alloc(f32, out_rows * narrow_cols);
    defer gpa.free(wn);
    @memset(wn, 0.01);
    try testing.expect((try plan.quantize(gpa, nt, wn, &pool)) == null);
    try testing.expectEqual(@as(usize, 1), plan.summary.too_narrow);
}

test "formats and tensors outside the policy are declined by the right arm" {
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 4, .buckets = 1 });
    defer c.deinit();
    var x: [4 * 256]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    for (&x) |*v| v.* = prng.random().floatNorm(f32);
    try observe(&c, "blocks.0.attn.wq.weight", &x, 4, 256);
    try c.checkOk();

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try CalibrationCache.write(gpa, &aw.writer, &c, test_prov);
    const bytes = try gpa.dupe(u8, aw.written());
    defer gpa.free(bytes);
    const st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes);
    var cache = try CalibrationCache.Cache.init(gpa, st);
    defer cache.deinit();
    var im = try Imatrix.fromCache(gpa, &cache);
    defer im.deinit();

    var pool: ThreadPool = undefined;
    try pool.init(.{ .allocator = gpa, .n_jobs = 1 });
    defer pool.deinit();
    var plan: Plan = .{ .gpa = gpa, .cache = &cache, .imatrix = &im };

    const w = try gpa.alloc(f32, 4 * 256);
    defer gpa.free(w);
    @memset(w, 0.01);
    var dims = [_]usize{ 4, 256 };

    // GGUF block quants: measured at ~5% with regressions, so the converter does
    // not reach them at all — this is the arm that records that decision.
    for ([_][]const u8{ "q4_k", "q6_k", "q8_0", "f16", "NVFP4", "MXFP4" }) |ty| {
        const t: types.Tensor = .{ .name = "blocks.0.attn.wq.weight", .type = ty, .dims = &dims, .size = 0, .offset = 0 };
        try testing.expect((try plan.quantize(gpa, t, w, &pool)) == null);
    }
    try testing.expectEqual(@as(usize, 6), plan.summary.unsupported_type);

    // Captured nothing for this name.
    const t: types.Tensor = .{ .name = "blocks.0.norm.weight", .type = "INT4_CONVROT", .dims = &dims, .size = 0, .offset = 0 };
    try testing.expect((try plan.quantize(gpa, t, w, &pool)) == null);
    try testing.expectEqual(@as(usize, 1), plan.summary.no_data);
    try testing.expectEqual(@as(usize, 0), plan.summary.failed);
}
