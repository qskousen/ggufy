//! Imatrix.zig — per-column importance weights for the quantizers' scale search.
//!
//! Plan §8A. Every quantizer here picks a scale by minimizing some squared error
//! over a group of weights; each one of them is minimizing the *wrong* error,
//! because it weights every column equally when the network does not. Given
//! per-channel activation energy from a calibration cache, they can minimize
//!
//! ```
//!     Σ_j  w_j · (W_ij − Ŵ_ij)²        instead of        Σ_j (W_ij − Ŵ_ij)²
//! ```
//!
//! instead. No format change, no runtime change, no new kernel: the same bytes in
//! the same layout, chosen better.
//!
//! Two mechanisms, because the formats divide in two:
//!
//! - **§8A.1, ggml's `imatrix`** — the k-quant and qX_0/qX_1 encoders already
//!   accept a `quant_weights` vector and do the weighted search themselves. We
//!   only have to hand it over (and hand it over *per row*, see
//!   `DataTransform.convertTypeGguf`).
//! - **§8A.2, the clipping search** — the formats ggml knows nothing about
//!   (int8, int8-convrot, int4-convrot) pick `scale = amax/qmax`, which is
//!   optimal only under equal weighting. `DataTransform.searchScale` searches a
//!   clipping ratio against the weighted objective instead.
//!
//! Which formats each mechanism *could* serve is `weightKind`; which ones we
//! actually ship it to is `shipsWeighted`. They differ (scaled-fp8 is measurable
//! but withheld), and keeping them separate is what lets the level-1 harness
//! audit the policy instead of agreeing with it by construction.
//!
//! Both are **opt-in** (`convert --calib`). The unweighted paths stay bit-exact
//! against their reference implementations, which is what keeps the ComfyUI and
//! ggml golden fixtures the correctness contract for this repo.
//!
//! ### Aggregation
//!
//! A cache holds `diag = Σ_tokens x_j²` per (layer, sigma bucket). Summing the
//! raw `diag` across buckets pools them **weighted by how many tokens each
//! contributed**, which is the correct per-channel mean-square over the whole
//! capture; a bucket that saw twice the tokens gets twice the say. The result is
//! then rescaled to mean 1.0. That rescale is cosmetic — ggml's weighted fits are
//! scale-invariant in the weights (a common factor multiplies the objective and
//! leaves every argmin alone) — but it keeps the numbers in a comfortable f32
//! range regardless of how long the capture ran, and it makes `min_relative_weight`
//! mean the same thing for every layer.
//!
//! ### Namespace
//!
//! Cache keys are the **checkpoint** tensor names, which for krea2 carry a
//! `model.diffusion_model.` container prefix. `Convert.filterAndStripTensors`
//! strips that prefix from every tensor before quantization, so the map is keyed
//! on the stripped form — the same lesson `Sensitivity.zig` learned when its
//! emitted JSON matched nothing.

const std = @import("std");
const types = @import("types.zig");
const gguf = @import("Gguf.zig");
const imagearch = @import("ImageArch.zig");
const CalibrationCache = @import("CalibrationCache.zig");
const TensorClusters = @import("TensorClusters.zig");

/// Channels whose calibrated energy falls below this fraction of their layer's
/// mean are raised to it.
///
/// A channel that no calibration prompt excited gets `diag = 0`, and a weight of
/// exactly zero tells ggml that column's error is free — a claim the 16-prompt
/// rung-1 set is in no position to make. Four orders of magnitude below the mean
/// is far enough down to leave the ranking of everything that *was* observed
/// untouched, and far enough up that a whole block of unobserved channels still
/// produces a conditioned weighted fit. (ggml itself guards its degenerate cases
/// — `make_qkx3_quants` falls back on `D > 0` — so this is about not throwing
/// away a channel on thin evidence, not about avoiding a NaN.)
pub const min_relative_weight: f32 = 1e-4;

/// What happened when the weights were built. Reported once per conversion,
/// because "activation-aware quantization ran" and "activation-aware
/// quantization ran on the layers you think" are different claims.
pub const Stats = struct {
    /// Layers with usable weights.
    layers: usize = 0,
    /// Layers in the cache that were never observed (zero tokens, or zero energy
    /// on every channel) and so carry no information.
    empty: usize = 0,
    /// Layers dropped because two prefixed cache names strip to one converter
    /// name, making the choice between them arbitrary.
    ambiguous: usize = 0,
    /// Channels raised to `min_relative_weight`, over all layers.
    floored: u64 = 0,
    /// Channels examined, so `floored` has a denominator. Counts a layer that was
    /// later dropped as ambiguous, since it was still examined.
    channels: u64 = 0,
};

/// Why a tensor did or did not get weighted. Every arm except `use` is a normal
/// outcome for *some* tensor in a real model, which is why the writer counts them
/// rather than warning per tensor.
pub const Decision = union(enum) {
    /// Weighted, with these per-column weights (length == the tensor's row width).
    use: []const f32,
    /// The destination format is not weighted: it has no scale search to steer
    /// (f16/bf16/f32, plain fp8), discards the weights (q8_0, mxfp4, nvfp4),
    /// carries block scales in a constrained encoding a clipping ratio is wrong
    /// for (MXFP4/MXFP8/NVFP4 clusters), or is deliberately withheld on measured
    /// evidence (SCALED_F8). `weightKind` says which; `shipsWeighted` decides.
    unweighted_type,
    /// No calibration entry for this tensor. Expected for everything the probe
    /// does not see — norms, biases, embeddings, and (until TP tags them) the
    /// text encoder and VAE.
    no_data,
    /// The cache's input width disagrees with the tensor's row width. This is the
    /// arm that catches shape-fixed tensors, whose dims have been rewritten to
    /// `(N/256, 256)` and no longer describe the GEMM the probe measured.
    width_mismatch,
    /// The row width does not satisfy the format's grouping constraint, so the
    /// weights cannot be lined up with the data: not a whole number of ggml
    /// blocks, not a whole number of ConvRot groups, or an odd column count for a
    /// nibble-packed format.
    block_misaligned,
};

/// Per-conversion tally of `Decision`s.
pub const Summary = struct {
    use: usize = 0,
    unweighted_type: usize = 0,
    no_data: usize = 0,
    width_mismatch: usize = 0,
    block_misaligned: usize = 0,
};

/// Does this ggml encoder read `quant_weights` at all? A statement of fact about
/// ggml, checked against its own `quantize_*` entry points: the k-quants and the
/// legacy qX_0/qX_1 types consume it; q8_0, mxfp4 and nvfp4 take the argument and
/// explicitly discard it.
pub fn readsImatrix(t: gguf.GgmlType) bool {
    return switch (t) {
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q4_1, .q5_0, .q5_1 => true,
        else => false,
    };
}

/// How a destination format consumes per-column importance, which decides what
/// shape constraint applies to it.
pub const WeightKind = enum {
    /// Nothing to steer: f16/bf16/f32 have no scale search, q8_0/mxfp4/nvfp4
    /// discard the argument, and plain F8_E4M3/F8_E5M2 are direct element
    /// encodings with no scale at all.
    none,
    /// A ggml block-quant encoder taking `quant_weights` (§8A.1).
    ggml_block,
    /// Symmetric per-row integer with a Hadamard rotation over column groups —
    /// the clipping search runs in the rotated basis (§8A.2).
    rotated_int,
    /// Symmetric per-row integer, no rotation.
    plain_int,
    /// One global scale over the whole tensor (the ComfyUI scaled-fp8 cluster).
    /// Implemented and measurable, but **not shipped** — see `shipsWeighted`.
    global_fp8,
};

pub fn weightKind(dt: types.DataType) WeightKind {
    return switch (dt) {
        .SCALED_F8_E4M3 => .global_fp8,
        .INT8_CONVROT, .INT4_CONVROT, .INT4_CONVROT_SR => .rotated_int,
        .INT8 => .plain_int,
        // MXFP4/MXFP8/NVFP4 are absent deliberately: their block scales are a
        // constrained encoding (power-of-two E8M0, per-block fp8) rather than a
        // free scalar, so a clipping ratio is not the right search for them and
        // pretending otherwise would report coverage we do not have.
        else => blk: {
            const gt = gguf.GgmlType.fromString(@tagName(dt)) catch break :blk .none;
            break :blk if (readsImatrix(gt)) .ggml_block else .none;
        },
    };
}

/// Policy: do we actually apply the weighting this format *could* take?
///
/// `weightKind` states what mechanism exists; this states what we ship. The two
/// are separate so the level-1 harness can measure formats the converter
/// withholds — which is the only way a policy call like this stays checkable.
/// Both times it mattered, the check paid for itself:
///
/// **q2_k — measured, and NOT excluded.** On synthetic weights an imatrix makes
/// ggml's *own internal* objective (the weighted weight error Σ_j w_j (W−Ŵ)²)
/// worse for q2_k alone, and worse the wider the importance spread — ratio 1.21
/// and 1.44 at lognormal σ = 1 and 2, while every other type improved. That was
/// taken as a reason to withhold it, and it was wrong: level 1 on real krea2
/// activations, measured on **output** error over 263 layers, has q2_k gaining
/// more than any other format (0.810). The weighted weight error ignores the
/// channel covariance a real GEMM sees. `DataTransform`'s test still pins that
/// disagreement so the gap stays visible.
///
/// **SCALED_F8 — measured, and excluded.** Level 1 over 263 krea2 layers: mean
/// ratio **0.9978** with only **102/263 layers improving**, i.e. the majority
/// regress for a 0.2% aggregate gain, at the cost of the most expensive search we
/// have. The mechanism is clear in hindsight and generalizes: **a clipping search
/// is for fixed-point grids.** Integer formats waste absolute resolution on an
/// outlier that stretches the range, and clipping reclaims it. fp8 is
/// floating-point — its precision is already relative to magnitude — so there is
/// no wasted range to reclaim and clipping only destroys the largest values. Any
/// future float-scaled format should be assumed to behave the same way until
/// measured.
pub fn shipsWeighted(dt: types.DataType) bool {
    return switch (weightKind(dt)) {
        .none, .global_fp8 => false,
        .ggml_block, .rotated_int, .plain_int => true,
    };
}

pub const Imatrix = struct {
    arena: std.heap.ArenaAllocator,
    /// Stripped tensor name → per-column weights. An entry of length 0 marks a
    /// name that two cache layers collided on; `get` reports it as absent.
    map: std.StringHashMapUnmanaged([]const f32) = .empty,
    stats: Stats = .{},

    pub fn deinit(self: *Imatrix) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Per-column weights for a checkpoint tensor name, or null when the cache
    /// has nothing usable for it.
    ///
    /// The name is stripped before lookup, not merely assumed stripped: GGUF
    /// output hands us already-stripped names, but safetensors output preserves
    /// the original prefixed ones so bundled checkpoints round-trip intact
    /// (`Convert.filterAndStripTensors`). Stripping here makes one map serve both.
    pub fn get(self: *const Imatrix, name: []const u8) ?[]const f32 {
        const w = self.map.get(imagearch.stripPrefix(name)) orelse return null;
        return if (w.len == 0) null else w;
    }

    /// The full applicability guard. Ordered so the least alarming explanation
    /// wins: a q8_0 tensor is `unweighted_type` whether or not it was captured,
    /// because the format is the reason nothing happens.
    pub fn decide(self: *const Imatrix, t: types.Tensor) Decision {
        const dt = types.DataType.fromString(t.type) catch return .unweighted_type;
        const kind = weightKind(dt);
        if (!shipsWeighted(dt)) return .unweighted_type;

        const w = self.get(t.name) orelse return .no_data;

        if (t.dims.len == 0) return .width_mismatch;
        // `dims` is outermost-first (PyTorch order) on both input paths, so the
        // last entry is the row width — the GEMM input width the probe measured.
        const cols = t.dims[t.dims.len - 1];
        if (cols != w.len) return .width_mismatch;

        switch (kind) {
            // Excluded by `shipsWeighted` above, so reaching here would mean the
            // two functions had drifted apart.
            .none, .global_fp8 => unreachable,
            .ggml_block => {
                const gt = gguf.GgmlType.fromString(t.type) catch return .unweighted_type;
                const block = gt.getBlockSize();
                if (block == 0 or cols % block != 0) return .block_misaligned;
            },
            // ConvRot rounds in a Hadamard basis applied over fixed groups of
            // columns, so a row that is not a whole number of groups cannot be
            // rotated at all — the quantizer would reject it too. Read the group
            // size from the format rather than assuming the two are equal.
            .rotated_int => {
                const g: usize = @intCast(if (dt == .INT8_CONVROT)
                    TensorClusters.int8_convrot_group_size
                else
                    TensorClusters.int4_convrot_group_size);
                if (g == 0 or cols % g != 0) return .block_misaligned;
            },
            // Nibble packing pairs adjacent columns.
            .plain_int => if (cols % 2 != 0) return .block_misaligned,
        }

        return .{ .use = w };
    }

    /// Convenience over `decide` for the write path.
    pub fn forTensor(self: *const Imatrix, t: types.Tensor) ?[]const f32 {
        return switch (self.decide(t)) {
            .use => |w| w,
            else => null,
        };
    }

    /// Tally what `decide` would say about a whole tensor list, so the converter
    /// can report coverage before it starts writing.
    pub fn summarize(self: *const Imatrix, tensors: []const types.Tensor) Summary {
        var s: Summary = .{};
        for (tensors) |t| switch (self.decide(t)) {
            .use => s.use += 1,
            .unweighted_type => s.unweighted_type += 1,
            .no_data => s.no_data += 1,
            .width_mismatch => s.width_mismatch += 1,
            .block_misaligned => s.block_misaligned += 1,
        };
        return s;
    }
};

/// Build the weights from an opened calibration cache. The caller is expected to
/// have run `CalibrationCache.validate` first — this reads the cache as trusted
/// data and only rechecks what it must to stay self-consistent.
pub fn fromCache(gpa: std.mem.Allocator, cache: *const CalibrationCache.Cache) !Imatrix {
    const buckets = cache.prov.buckets;
    if (buckets == 0) return error.NoBuckets;

    var self: Imatrix = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
    errdefer self.arena.deinit();
    const arena = self.arena.allocator();

    // Pooled Σx² across buckets, in f64: the per-bucket sums are already large
    // and adding hundreds of thousands of them in f32 loses the small channels.
    var acc: std.ArrayList(f64) = .empty;
    defer acc.deinit(gpa);

    for (cache.layers()) |layer| {
        acc.clearRetainingCapacity();
        var cols: usize = 0;
        var tokens: u64 = 0;

        for (0..buckets) |k| {
            const b = try cache.bucket(layer, k);
            if (k == 0) {
                cols = b.cols;
                try acc.appendNTimes(gpa, 0, cols);
            } else if (b.cols != cols) {
                return error.ShapeMismatch;
            }

            const d = try b.diagAlloc(gpa);
            defer gpa.free(d);
            for (acc.items, d) |*a, v| a.* += v;
            tokens += b.count;
        }

        const key = imagearch.stripPrefix(layer);
        const gop = try self.map.getOrPut(arena, key);
        if (gop.found_existing) {
            // Two checkpoint names, one converter name: there is no principled
            // way to pick, and picking wrongly weights a layer by another
            // layer's activations. Drop both.
            //
            // Only un-count the first one if it was actually counted — a name
            // that collided with an *empty* layer never incremented `layers`, and
            // decrementing unconditionally would under-report coverage.
            if (gop.value_ptr.*.len > 0) self.stats.layers -= 1;
            gop.value_ptr.* = &.{};
            self.stats.ambiguous += 1;
            continue;
        }
        gop.key_ptr.* = try arena.dupe(u8, key);
        gop.value_ptr.* = &.{};

        var sum: f64 = 0;
        for (acc.items) |a| sum += a;
        if (tokens == 0 or sum <= 0) {
            // Never observed, or observed as identically zero. Either way there
            // is nothing here to steer a scale search with, and a flat weight
            // vector is not the same experiment as no weights at all.
            self.stats.empty += 1;
            continue;
        }

        const mean = sum / @as(f64, @floatFromInt(cols));
        const floor = mean * min_relative_weight;
        const w = try arena.alloc(f32, cols);
        for (w, acc.items) |*o, a| {
            if (a < floor) self.stats.floored += 1;
            o.* = @floatCast(@max(a, floor) / mean);
        }
        self.stats.channels += cols;
        gop.value_ptr.* = w;
        self.stats.layers += 1;
    }

    return self;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const Activations = @import("Activations.zig");
const tp = @import("TensorPencil");

/// Drive one tagged GEMM through TensorPencil so the collector observes `x`
/// exactly as a model forward would deliver it. Same helper as
/// CalibrationCache.zig's — the point is that these tests exercise the real
/// probe path, not a hand-filled struct.
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

fn cacheFromCollector(gpa: std.mem.Allocator, c: *const Activations.Collector, bytes_out: *[]u8) !CalibrationCache.Cache {
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try CalibrationCache.write(gpa, &aw.writer, c, test_prov);
    bytes_out.* = try gpa.dupe(u8, aw.written());
    errdefer gpa.free(bytes_out.*);
    const st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes_out.*);
    return CalibrationCache.Cache.init(gpa, st);
}

fn tensor(name: []const u8, ty: []const u8, dims: []usize) types.Tensor {
    return .{ .name = name, .type = ty, .dims = dims, .size = 0, .offset = 0 };
}

test "weights pool across buckets by token count and normalize to mean one" {
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 4, .buckets = 2 });
    defer c.deinit();

    // Two columns, two buckets. Bucket 0 sees one token, bucket 1 sees three, so
    // pooling must give bucket 1 three times the influence.
    c.setBucket(0);
    try observe(&c, "model.diffusion_model.blocks.0.attn.wq.weight", &.{ 1, 0 }, 1, 2);
    c.setBucket(1);
    try observe(&c, "model.diffusion_model.blocks.0.attn.wq.weight", &.{ 0, 1, 0, 1, 0, 1 }, 3, 2);
    try c.checkOk();

    var bytes: []u8 = undefined;
    var cache = try cacheFromCollector(gpa, &c, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    var im = try fromCache(gpa, &cache);
    defer im.deinit();

    // Keyed on the stripped name, and reachable by either spelling: GGUF output
    // asks with the stripped name, safetensors output with the original prefixed
    // one, and one map has to serve both.
    const w = im.get("blocks.0.attn.wq.weight").?;
    try testing.expectEqualSlices(f32, w, im.get("model.diffusion_model.blocks.0.attn.wq.weight").?);
    try testing.expect(im.get("blocks.0.attn.wk.weight") == null);

    // Σx² is 1 for column 0 and 3 for column 1; mean 2 → 0.5 and 1.5.
    try testing.expectEqual(@as(usize, 2), w.len);
    try testing.expectApproxEqAbs(@as(f32, 0.5), w[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.5), w[1], 1e-6);
    try testing.expectEqual(@as(usize, 1), im.stats.layers);
    try testing.expectEqual(@as(u64, 0), im.stats.floored);
}

test "an unobserved channel is floored rather than zeroed" {
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();
    // Column 1 is never excited.
    try observe(&c, "blocks.0.mlp.down.weight", &.{ 2, 0, 2, 0 }, 2, 2);
    try c.checkOk();

    var bytes: []u8 = undefined;
    var cache = try cacheFromCollector(gpa, &c, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    var im = try fromCache(gpa, &cache);
    defer im.deinit();

    const w = im.get("blocks.0.mlp.down.weight").?;
    try testing.expectApproxEqAbs(@as(f32, 2.0), w[0], 1e-6);
    try testing.expectApproxEqAbs(min_relative_weight, w[1], 1e-9);
    try testing.expect(w[1] > 0);
    try testing.expectEqual(@as(u64, 1), im.stats.floored);
    try testing.expectEqual(@as(u64, 2), im.stats.channels);
}

test "a layer observed as identically zero yields no weights at all" {
    // A flat weight vector is a different experiment from no weights: ggml takes
    // a wholly different branch when quant_weights is null. Refusing to invent
    // one keeps "no information" from masquerading as "uniform importance".
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();
    try observe(&c, "blocks.0.attn.wo.weight", &.{ 0, 0, 0, 0 }, 2, 2);
    try c.checkOk();

    var bytes: []u8 = undefined;
    var cache = try cacheFromCollector(gpa, &c, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    var im = try fromCache(gpa, &cache);
    defer im.deinit();

    try testing.expect(im.get("blocks.0.attn.wo.weight") == null);
    try testing.expectEqual(@as(usize, 0), im.stats.layers);
    try testing.expectEqual(@as(usize, 1), im.stats.empty);
}

test "two cache names stripping to one converter name drop both" {
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();
    try observe(&c, "model.blocks.0.attn.wq.weight", &.{ 1, 2 }, 1, 2);
    try observe(&c, "blocks.0.attn.wq.weight", &.{ 3, 4 }, 1, 2);
    try c.checkOk();

    var bytes: []u8 = undefined;
    var cache = try cacheFromCollector(gpa, &c, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    var im = try fromCache(gpa, &cache);
    defer im.deinit();

    try testing.expect(im.get("blocks.0.attn.wq.weight") == null);
    try testing.expectEqual(@as(usize, 1), im.stats.ambiguous);
    try testing.expectEqual(@as(usize, 0), im.stats.layers);
}

test "the applicability guard classifies every reason a tensor goes unweighted" {
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();

    var x256: [256]f32 = undefined;
    for (&x256, 0..) |*v, i| v.* = @floatFromInt(i % 7 + 1);
    try observe(&c, "blocks.0.attn.wq.weight", &x256, 1, 256);
    try observe(&c, "blocks.0.mlp.gate.weight", &.{ 1, 2, 3, 4 }, 1, 4);
    try c.checkOk();

    var bytes: []u8 = undefined;
    var cache = try cacheFromCollector(gpa, &c, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    var im = try fromCache(gpa, &cache);
    defer im.deinit();

    var d_ok = [_]usize{ 8, 256 };
    var d_bad_width = [_]usize{ 8, 128 };
    var d_small = [_]usize{ 4, 4 };

    // The happy path: a captured layer, a k-quant target, 256 | cols.
    try testing.expect(im.decide(tensor("blocks.0.attn.wq.weight", "q4_k", &d_ok)) == .use);

    // A format that discards quant_weights, however good the data is.
    try testing.expect(im.decide(tensor("blocks.0.attn.wq.weight", "q8_0", &d_ok)) == .unweighted_type);
    try testing.expect(im.decide(tensor("blocks.0.attn.wq.weight", "f16", &d_ok)) == .unweighted_type);

    // Nothing captured for this tensor.
    try testing.expect(im.decide(tensor("blocks.0.norm.weight", "q4_k", &d_ok)) == .no_data);

    // Shape-fixed / reshaped: the dims no longer describe the measured GEMM.
    try testing.expect(im.decide(tensor("blocks.0.attn.wq.weight", "q4_k", &d_bad_width)) == .width_mismatch);

    // Captured, but the row is not a whole number of q4_k blocks.
    try testing.expect(im.decide(tensor("blocks.0.mlp.gate.weight", "q4_k", &d_small)) == .block_misaligned);
    // ...while a 32-element-block type takes the same tensor happily only if it
    // divides — 4 does not divide 32 either, so this stays misaligned.
    try testing.expect(im.decide(tensor("blocks.0.mlp.gate.weight", "q4_0", &d_small)) == .block_misaligned);

    const s = im.summarize(&.{
        tensor("blocks.0.attn.wq.weight", "q4_k", &d_ok),
        tensor("blocks.0.attn.wq.weight", "q8_0", &d_ok),
        tensor("blocks.0.norm.weight", "q4_k", &d_ok),
        tensor("blocks.0.attn.wq.weight", "q4_k", &d_bad_width),
        tensor("blocks.0.mlp.gate.weight", "q4_k", &d_small),
    });
    try testing.expectEqual(@as(usize, 1), s.use);
    try testing.expectEqual(@as(usize, 1), s.unweighted_type);
    try testing.expectEqual(@as(usize, 1), s.no_data);
    try testing.expectEqual(@as(usize, 1), s.width_mismatch);
    try testing.expectEqual(@as(usize, 1), s.block_misaligned);
}

test "readsImatrix matches the ggml encoders that consume quant_weights" {
    // Pinned deliberately: if a ggml bump starts honouring quant_weights in a
    // type we list as unweighted, we silently stop applying it there.
    for ([_]gguf.GgmlType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q4_1, .q5_0, .q5_1 }) |t|
        try testing.expect(readsImatrix(t));
    for ([_]gguf.GgmlType{ .q8_0, .mxfp4, .nvfp4, .f16, .bf16, .f32 }) |t|
        try testing.expect(!readsImatrix(t));
}

test "policy ships every mechanism level 1 says helps, and only those" {
    // The seam between fact (`weightKind`) and choice (`shipsWeighted`) exists so
    // a withheld format stays measurable. This pins both live decisions: q2_k IS
    // shipped (a synthetic-evidence exclusion that level 1 overturned), and
    // SCALED_F8 is NOT (a real-data exclusion level 1 established).
    for ([_]types.DataType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q4_1, .q5_0, .q5_1 }) |dt| {
        try testing.expectEqual(WeightKind.ggml_block, weightKind(dt));
        try testing.expect(shipsWeighted(dt));
    }
    for ([_]types.DataType{ .INT8_CONVROT, .INT4_CONVROT, .INT4_CONVROT_SR }) |dt| {
        try testing.expectEqual(WeightKind.rotated_int, weightKind(dt));
        try testing.expect(shipsWeighted(dt));
    }
    try testing.expectEqual(WeightKind.plain_int, weightKind(.INT8));
    try testing.expect(shipsWeighted(.INT8));

    // Measurable, deliberately not shipped.
    try testing.expectEqual(WeightKind.global_fp8, weightKind(.SCALED_F8_E4M3));
    try testing.expect(!shipsWeighted(.SCALED_F8_E4M3));

    // No mechanism at all.
    for ([_]types.DataType{ .q8_0, .mxfp4, .NVFP4, .MXFP8_E4M3, .MXFP4, .f16, .BF16, .F8_E4M3 }) |dt| {
        try testing.expectEqual(WeightKind.none, weightKind(dt));
        try testing.expect(!shipsWeighted(dt));
    }
}

test "a collision with an unobserved layer does not under-count coverage" {
    // The ambiguous path un-counts the entry it collides with; doing that
    // unconditionally would subtract a layer that was never added, and silently
    // under-report how much of the model the weighting reached.
    const gpa = testing.allocator;
    var c = Activations.Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();
    // Sorted layer order puts the unprefixed (all-zero) name first, so the
    // prefixed one collides with an entry that was counted as `empty`.
    try observe(&c, "blocks.9.attn.wq.weight", &.{ 0, 0 }, 1, 2);
    try observe(&c, "model.blocks.9.attn.wq.weight", &.{ 1, 2 }, 1, 2);
    try c.checkOk();

    var bytes: []u8 = undefined;
    var cache = try cacheFromCollector(gpa, &c, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    var im = try fromCache(gpa, &cache);
    defer im.deinit();

    try testing.expect(im.get("blocks.9.attn.wq.weight") == null);
    try testing.expectEqual(@as(usize, 1), im.stats.ambiguous);
    try testing.expectEqual(@as(usize, 1), im.stats.empty);
    try testing.expectEqual(@as(usize, 0), im.stats.layers);
}
