//! Activation statistics collection — the consumer side of TensorPencil's
//! `ops.matmul.probe`.
//!
//! Quantization decisions want to know what each layer's inputs actually look
//! like, not just what its weights look like (see ACTIVATION_AWARE.md: weight
//! reconstruction error says nothing about which layers matter). This module runs
//! a model forward with a probe installed and accumulates, per layer:
//!
//!   - `diag`  — Σ x_j² over every token, i.e. per-input-channel activation
//!               energy. This is the weighting ggml's k-quant scale search takes
//!               as an importance matrix, and the cheap sensitivity signal.
//!   - `amax`  — max |x_j| per channel: which channels carry outliers.
//!   - `rows`  — a bounded uniform sample of actual token rows, so downstream can
//!               compute the real quantity of interest, `WX` vs `ŴX`, instead of a
//!               proxy. Storing all of X is not an option: one 6144-column layer at
//!               1024² is ~100 MB per step, times hundreds of layers.
//!
//! Statistics are kept per *schedule bucket* (early/mid/late denoising) because a
//! diffusion model's activation distribution moves along the schedule; the driver
//! calls `setBucket` as it steps.
//!
//! This module is deliberately inference-free: it observes GEMMs and owns no model.
//! Whoever drives the forward pass installs `probe()` and sets the bucket.

const std = @import("std");
const tp = @import("TensorPencil");

const Weight = tp.ops.matmul.Weight;

pub const Options = struct {
    /// Token rows to retain per layer per bucket (the `r` of the row sample).
    /// Uniform over every token seen, via reservoir sampling — so this is a memory
    /// bound, not a "first r tokens" truncation.
    sample_rows: usize = 64,
    /// Number of schedule buckets to keep separate statistics for.
    buckets: usize = 3,
    /// Base seed for the row sampling. Fixed by default: the same run must select
    /// the same tokens, or two formats cannot be compared on equal footing.
    seed: u64 = 0x9E3779B97F4A7C15,
};

/// Per-layer, per-bucket accumulator.
pub const Bucket = struct {
    /// Σ x_j² per column, f64 to keep small terms against a long accumulation.
    diag: []f64,
    /// max |x_j| per column.
    amax: []f32,
    /// Tokens accumulated into `diag`/`amax`.
    count: u64,
    /// Retained token rows, `kept * cols` values, row-major.
    rows: []f32,
    /// How many rows of `rows` are populated (≤ `Options.sample_rows`).
    kept: usize,
    /// Which token index each retained row came from — lets a consumer line up
    /// the sample across arms, and makes the sampling auditable.
    row_index: []u64,
    /// Tokens offered to the reservoir so far (== count; kept separate because the
    /// sampling probability is defined on it).
    seen: u64,
    prng: std.Random.DefaultPrng,

    fn init(gpa: std.mem.Allocator, cols: usize, opts: Options, seed: u64) !Bucket {
        const diag = try gpa.alloc(f64, cols);
        errdefer gpa.free(diag);
        const amax = try gpa.alloc(f32, cols);
        errdefer gpa.free(amax);
        const rows = try gpa.alloc(f32, opts.sample_rows * cols);
        errdefer gpa.free(rows);
        const row_index = try gpa.alloc(u64, opts.sample_rows);
        @memset(diag, 0);
        @memset(amax, 0);
        return .{
            .diag = diag,
            .amax = amax,
            .count = 0,
            .rows = rows,
            .kept = 0,
            .row_index = row_index,
            .seen = 0,
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    fn deinit(self: *Bucket, gpa: std.mem.Allocator) void {
        gpa.free(self.diag);
        gpa.free(self.amax);
        gpa.free(self.rows);
        gpa.free(self.row_index);
    }

    /// The retained sample as a `[kept][cols]` view.
    pub fn sample(self: *const Bucket, cols: usize) []const f32 {
        return self.rows[0 .. self.kept * cols];
    }
};

pub const Layer = struct {
    /// Input width of this layer's GEMM (`Weight.cols`).
    cols: usize,
    /// Output width — carried so a consumer can size `WX` without the checkpoint.
    rows: usize,
    buckets: []Bucket,

    fn deinit(self: *Layer, gpa: std.mem.Allocator) void {
        for (self.buckets) |*b| b.deinit(gpa);
        gpa.free(self.buckets);
    }
};

pub const Collector = struct {
    gpa: std.mem.Allocator,
    opts: Options,
    /// Keyed by `Weight.tag` (the checkpoint tensor name). Keys are owned copies:
    /// the probe's `Weight` borrows from the model, which may outlive or predecease
    /// us, and a cache keyed on a dangling name would be silent corruption.
    layers: std.StringHashMapUnmanaged(Layer) = .empty,
    bucket: usize = 0,

    /// GEMMs observed with no `tag`. Not an error — TensorPencil tags model weights
    /// but a caller may run its own untagged GEMMs — but a *large* count means the
    /// loader is not tagging and the capture would be near-empty, so the driver
    /// should check it.
    untagged: u64 = 0,
    /// Set if an allocation failed inside the probe. The hook signature cannot
    /// return an error, so failure is recorded and `checkOk` reports it.
    oom: bool = false,
    /// Set if a tag was seen with two different shapes — the tag would then be
    /// aggregating unrelated tensors.
    shape_conflict: bool = false,

    pub fn init(gpa: std.mem.Allocator, opts: Options) Collector {
        std.debug.assert(opts.buckets > 0);
        std.debug.assert(opts.sample_rows > 0);
        return .{ .gpa = gpa, .opts = opts };
    }

    pub fn deinit(self: *Collector) void {
        var it = self.layers.iterator();
        while (it.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            e.value_ptr.deinit(self.gpa);
        }
        self.layers.deinit(self.gpa);
        self.* = undefined;
    }

    /// Install the result as `tp.ops.matmul.probe` for the duration of a forward.
    /// The collector must outlive the probe, and (like the probe itself) is not
    /// internally synchronized: serialize the GEMMs being observed.
    pub fn probe(self: *Collector) tp.ops.matmul.Probe {
        return .{ .ctx = self, .input = onInput };
    }

    /// Select which schedule bucket subsequent observations accumulate into.
    pub fn setBucket(self: *Collector, bucket: usize) void {
        std.debug.assert(bucket < self.opts.buckets);
        self.bucket = bucket;
    }

    /// Map a sigma (or any monotone schedule position) in [0, 1] onto a bucket,
    /// where 0 is the first step and 1 the last.
    pub fn bucketForProgress(self: *const Collector, progress: f32) usize {
        const p = std.math.clamp(progress, 0.0, 1.0);
        const idx: usize = @intFromFloat(p * @as(f32, @floatFromInt(self.opts.buckets)));
        return @min(idx, self.opts.buckets - 1);
    }

    pub fn layerCount(self: *const Collector) usize {
        return self.layers.count();
    }

    pub fn get(self: *const Collector, tag: []const u8) ?*const Layer {
        return self.layers.getPtr(tag);
    }

    pub fn iterator(self: *const Collector) std.StringHashMapUnmanaged(Layer).Iterator {
        return self.layers.iterator();
    }

    /// Report any problem recorded during observation. Call after the forward
    /// passes and before trusting the numbers — silent partial capture is the
    /// failure mode that would quietly produce a plausible-but-wrong report.
    pub fn checkOk(self: *const Collector) !void {
        if (self.oom) return error.OutOfMemory;
        if (self.shape_conflict) return error.TagShapeConflict;
        if (self.layers.count() == 0) return error.NoLayersObserved;
    }

    fn layerFor(self: *Collector, tag: []const u8, cols: usize, rows: usize) !*Layer {
        if (self.layers.getPtr(tag)) |l| {
            if (l.cols != cols or l.rows != rows) return error.TagShapeConflict;
            return l;
        }
        const key = try self.gpa.dupe(u8, tag);
        errdefer self.gpa.free(key);
        const buckets = try self.gpa.alloc(Bucket, self.opts.buckets);
        errdefer self.gpa.free(buckets);
        var made: usize = 0;
        errdefer for (buckets[0..made]) |*b| b.deinit(self.gpa);
        for (buckets, 0..) |*b, i| {
            // Seed per (layer, bucket) so the retained tokens do not depend on the
            // order layers happen to be visited in.
            b.* = try Bucket.init(self.gpa, cols, self.opts, self.opts.seed ^ std.hash.Wyhash.hash(i, tag));
            made += 1;
        }
        try self.layers.put(self.gpa, key, .{ .cols = cols, .rows = rows, .buckets = buckets });
        return self.layers.getPtr(tag).?;
    }

    fn onInput(ctx: *anyopaque, w: Weight, x: []const f32, m: usize) void {
        const self: *Collector = @ptrCast(@alignCast(ctx));
        const tag = w.tag orelse {
            self.untagged += 1;
            return;
        };
        const layer = self.layerFor(tag, w.cols, w.rows) catch |err| {
            switch (err) {
                error.TagShapeConflict => self.shape_conflict = true,
                else => self.oom = true,
            }
            return;
        };
        const b = &layer.buckets[self.bucket];
        const cols = w.cols;

        for (0..m) |i| {
            const row = x[i * cols ..][0..cols];
            for (row, b.diag, b.amax) |v, *d, *a| {
                const fv: f64 = v;
                d.* += fv * fv;
                const av = @abs(v);
                if (av > a.*) a.* = av;
            }
            b.count += 1;

            // Reservoir sampling (algorithm R): every token seen has an equal chance
            // of being retained, so the sample is not biased toward whichever tokens
            // happen to come first.
            const idx = b.seen;
            b.seen += 1;
            if (b.kept < self.opts.sample_rows) {
                @memcpy(b.rows[b.kept * cols ..][0..cols], row);
                b.row_index[b.kept] = idx;
                b.kept += 1;
            } else {
                const j = b.prng.random().uintLessThan(u64, b.seen);
                if (j < self.opts.sample_rows) {
                    const slot: usize = @intCast(j);
                    @memcpy(b.rows[slot * cols ..][0..cols], row);
                    b.row_index[slot] = idx;
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Tests — these run real GEMMs through tp.ops.matmul with the probe installed,
// so they exercise the actual TensorPencil hook rather than calling onInput
// directly.
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Run one tagged GEMM of `m` tokens, so the collector sees `x` exactly as a model
/// forward would deliver it.
fn observeGemm(collector: *Collector, tag: []const u8, x: []const f32, m: usize, cols: usize) !void {
    const gpa = testing.allocator;
    const rows = 2;
    const wdata = try gpa.alloc(f32, rows * cols);
    defer gpa.free(wdata);
    @memset(wdata, 0.5);
    const y = try gpa.alloc(f32, m * rows);
    defer gpa.free(y);

    var w = Weight.fromF32(wdata, rows, cols);
    w.tag = tag;

    const prev = tp.ops.matmul.probe;
    tp.ops.matmul.probe = collector.probe();
    defer tp.ops.matmul.probe = prev;
    try tp.ops.matmul.matmul(testing.io, gpa, y, x, m, w, null);
}

test "diag and amax accumulate the per-channel energy and peak" {
    var c = Collector.init(testing.allocator, .{ .sample_rows = 8, .buckets = 1 });
    defer c.deinit();

    // 3 tokens, 4 channels. Channel 3 is the outlier channel.
    const x = [_]f32{
        1, 0,  2, 10,
        2, -1, 0, -20,
        0, 1,  1, 5,
    };
    try observeGemm(&c, "blocks.0.attn.wq.weight", &x, 3, 4);
    try c.checkOk();

    const layer = c.get("blocks.0.attn.wq.weight").?;
    try testing.expectEqual(@as(usize, 4), layer.cols);
    const b = &layer.buckets[0];
    try testing.expectEqual(@as(u64, 3), b.count);

    // Σ x² down each column.
    const want_diag = [_]f64{ 1 + 4 + 0, 0 + 1 + 1, 4 + 0 + 1, 100 + 400 + 25 };
    for (want_diag, b.diag, 0..) |want, got, j| {
        errdefer std.debug.print("diag[{d}]: want {d}, got {d}\n", .{ j, want, got });
        try testing.expectApproxEqAbs(want, got, 1e-9);
    }
    // max |x| down each column — the outlier channel must show its magnitude.
    try testing.expectEqualSlices(f32, &.{ 2, 1, 2, 20 }, b.amax);
}

test "rows below the sample bound are all retained, in order" {
    var c = Collector.init(testing.allocator, .{ .sample_rows = 8, .buckets = 1 });
    defer c.deinit();

    const x = [_]f32{ 1, 2, 3, 4, 5, 6 }; // 3 tokens x 2 channels
    try observeGemm(&c, "layer", &x, 3, 2);

    const b = &c.get("layer").?.buckets[0];
    try testing.expectEqual(@as(usize, 3), b.kept);
    try testing.expectEqualSlices(f32, &x, b.sample(2));
    try testing.expectEqualSlices(u64, &.{ 0, 1, 2 }, b.row_index[0..3]);
}

test "the reservoir stays bounded, samples real rows, and is reproducible" {
    const gpa = testing.allocator;
    // 200 tokens, each row filled with its own index so a retained row identifies
    // itself: row i is [i, i].
    const n = 200;
    const x = try gpa.alloc(f32, n * 2);
    defer gpa.free(x);
    for (0..n) |i| {
        x[i * 2] = @floatFromInt(i);
        x[i * 2 + 1] = @floatFromInt(i);
    }

    var first_run_idx: [8]u64 = undefined;
    for (0..2) |run| {
        var c = Collector.init(gpa, .{ .sample_rows = 8, .buckets = 1 });
        defer c.deinit();
        try observeGemm(&c, "layer", x, n, 2);
        const b = &c.get("layer").?.buckets[0];

        // Bounded, and every retained row is a row that actually occurred, matching
        // the index recorded for it.
        try testing.expectEqual(@as(usize, 8), b.kept);
        try testing.expectEqual(@as(u64, n), b.count);
        for (0..b.kept) |k| {
            const v = b.rows[k * 2];
            try testing.expectEqual(v, b.rows[k * 2 + 1]);
            try testing.expectEqual(@as(f32, @floatFromInt(b.row_index[k])), v);
            try testing.expect(v >= 0 and v < n);
        }
        // Not merely the first 8 tokens — the sample must reach later tokens.
        var max_idx: u64 = 0;
        for (b.row_index[0..b.kept]) |ri| max_idx = @max(max_idx, ri);
        try testing.expect(max_idx >= 8);

        // Reproducible: a fixed seed must retain the SAME tokens every run, or two
        // formats measured in separate runs would be compared on different samples.
        if (run == 0) {
            @memcpy(&first_run_idx, b.row_index[0..8]);
        } else {
            try testing.expectEqualSlices(u64, &first_run_idx, b.row_index[0..8]);
        }
    }
}

test "buckets keep separate statistics" {
    var c = Collector.init(testing.allocator, .{ .sample_rows = 4, .buckets = 3 });
    defer c.deinit();

    const early = [_]f32{ 1, 1 };
    const late = [_]f32{ 10, 10 };
    c.setBucket(0);
    try observeGemm(&c, "layer", &early, 1, 2);
    c.setBucket(2);
    try observeGemm(&c, "layer", &late, 1, 2);
    try observeGemm(&c, "layer", &late, 1, 2);

    const l = c.get("layer").?;
    try testing.expectEqual(@as(u64, 1), l.buckets[0].count);
    try testing.expectEqual(@as(u64, 0), l.buckets[1].count);
    try testing.expectEqual(@as(u64, 2), l.buckets[2].count);
    try testing.expectApproxEqAbs(@as(f64, 1), l.buckets[0].diag[0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 200), l.buckets[2].diag[0], 1e-9);
    try testing.expectEqual(@as(f32, 1), l.buckets[0].amax[0]);
    try testing.expectEqual(@as(f32, 10), l.buckets[2].amax[0]);
}

test "bucketForProgress spreads the schedule across buckets" {
    var c = Collector.init(testing.allocator, .{ .buckets = 3 });
    defer c.deinit();
    try testing.expectEqual(@as(usize, 0), c.bucketForProgress(0.0));
    try testing.expectEqual(@as(usize, 0), c.bucketForProgress(0.32));
    try testing.expectEqual(@as(usize, 1), c.bucketForProgress(0.34));
    try testing.expectEqual(@as(usize, 2), c.bucketForProgress(0.9));
    try testing.expectEqual(@as(usize, 2), c.bucketForProgress(1.0)); // must not overflow
}

test "untagged GEMMs are counted, not silently merged" {
    const gpa = testing.allocator;
    var c = Collector.init(gpa, .{ .buckets = 1 });
    defer c.deinit();

    const wdata = [_]f32{ 1, 0, 0, 1 };
    const x = [_]f32{ 1, 2 };
    var y: [2]f32 = undefined;
    const untagged = Weight.fromF32(&wdata, 2, 2); // no tag

    const prev = tp.ops.matmul.probe;
    tp.ops.matmul.probe = c.probe();
    defer tp.ops.matmul.probe = prev;
    try tp.ops.matmul.matmul(testing.io, gpa, &y, &x, 1, untagged, null);

    try testing.expectEqual(@as(u64, 1), c.untagged);
    try testing.expectEqual(@as(usize, 0), c.layerCount());
    // Nothing observed at all is an error, not an empty-but-fine result.
    try testing.expectError(error.NoLayersObserved, c.checkOk());
}

test "a tag reused with a different shape is reported, not aggregated" {
    var c = Collector.init(testing.allocator, .{ .buckets = 1 });
    defer c.deinit();
    const x2 = [_]f32{ 1, 2 };
    const x3 = [_]f32{ 1, 2, 3 };
    try observeGemm(&c, "layer", &x2, 1, 2);
    try observeGemm(&c, "layer", &x3, 1, 3); // same name, different width
    try testing.expectError(error.TagShapeConflict, c.checkOk());
}
