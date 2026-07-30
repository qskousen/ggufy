//! On-disk form of `Activations.Collector` — the calibration cache.
//!
//! Capture is expensive (minutes on a GPU, hours on CPU) and its output is
//! consumed repeatedly: by the level-1 sensitivity harness, by the `imatrix`
//! hook in the k-quant scale search, by the clip search. So it is written once
//! to a file and read back many times, and the file is a **safetensors
//! container** — ggufy already parses safetensors, TensorPencil already parses
//! safetensors, and no third tool has to learn a bespoke format.
//!
//! Layout, per probed linear and per schedule bucket `k` (see
//! ACTIVATION_AWARE_PLAN.md §4):
//!
//! | tensor                  | shape         | dtype | meaning                         |
//! |-------------------------|---------------|-------|---------------------------------|
//! | `<name>/b<k>/diag`      | `[cols]`      | F32   | Σ x_j² — per-channel energy     |
//! | `<name>/b<k>/amax`      | `[cols]`      | F32   | max \|x_j\| — outlier channels  |
//! | `<name>/b<k>/rows`      | `[kept, cols]`| F32   | retained token rows of X        |
//! | `<name>/b<k>/rows_idx`  | `[kept]`      | I64   | which token each row came from  |
//! | `<name>/b<k>/count`     | `[1]`         | I64   | tokens accumulated              |
//!
//! `<name>` is the **checkpoint tensor name**, the same namespace `Convert.zig`
//! and `src/sensitivities/*.json` use — so a measured sensitivity file drops
//! into the existing routing with no converter change.
//!
//! Provenance lives in the header's `__metadata__` map (which the format defines
//! as string→string, so every field is stringified). It exists to make
//! `validate` possible: a stale cache that silently produces plausible-looking
//! sensitivity numbers is the worst failure mode available here, so a consumer
//! is expected to gate on `Cache.validate` before trusting anything.
//!
//! The writer reproduces the reference implementation's container conventions
//! exactly — tensors ordered by dtype size descending then name ascending,
//! `__metadata__` first with sorted keys, header space-padded so the data
//! section starts 8-byte aligned — which is what lets
//! `src/test_fixtures/calib_container.safetensors` (written by the Python
//! `safetensors` package) pin our bytes.

const std = @import("std");
const builtin = @import("builtin");
const tp = @import("TensorPencil");
const Activations = @import("Activations.zig");

const Collector = Activations.Collector;

/// Bumped whenever the tensor layout or metadata contract changes. `validate`
/// refuses a cache written by a different schema rather than misreading it.
pub const schema_version: u32 = 1;

const native_le = builtin.cpu.arch.endian() == .little;

// ---------------------------------------------------------------------------
// Provenance
// ---------------------------------------------------------------------------

/// What produced this cache. Everything here is either needed to reproduce the
/// capture or needed to reject a cache that does not belong to the model being
/// quantized.
pub const Provenance = struct {
    /// Path the checkpoint was captured from — a human breadcrumb, not an identity.
    model_path: []const u8 = "",
    /// Content hash of the checkpoint, hex. This *is* the identity: `validate`
    /// compares it, because a cache captured from a different model is
    /// indistinguishable from a good one by shape alone once two checkpoints
    /// share an architecture.
    model_hash: []const u8 = "",
    arch: []const u8 = "",
    /// Identifier of the prompt set used, so a run is reproducible.
    prompt_set: []const u8 = "",
    /// TensorPencil backend spelling (`cpu`/`vulkan`/`zig-cuda`/`cuda`).
    backend: []const u8 = "",
    /// Tool + version that wrote the file.
    producer: []const u8 = "",
    /// Square image side the capture ran at, in pixels.
    resolution: u32 = 0,
    steps: u32 = 0,
    /// Sampler seed.
    seed: u64 = 0,
    /// `Activations.Options.seed` — the row-sampling seed.
    sample_seed: u64 = 0,
    buckets: usize = 0,
    sample_rows: usize = 0,
    schema: u32 = schema_version,

    const Field = struct {
        key: []const u8,
        kind: enum { str, uint },
        offset: usize,
    };

    /// One row per metadata key, so writing and reading cannot drift apart.
    const fields = [_]Field{
        .{ .key = "arch", .kind = .str, .offset = @offsetOf(Provenance, "arch") },
        .{ .key = "backend", .kind = .str, .offset = @offsetOf(Provenance, "backend") },
        .{ .key = "buckets", .kind = .uint, .offset = @offsetOf(Provenance, "buckets") },
        .{ .key = "model_hash", .kind = .str, .offset = @offsetOf(Provenance, "model_hash") },
        .{ .key = "model_path", .kind = .str, .offset = @offsetOf(Provenance, "model_path") },
        .{ .key = "producer", .kind = .str, .offset = @offsetOf(Provenance, "producer") },
        .{ .key = "prompt_set", .kind = .str, .offset = @offsetOf(Provenance, "prompt_set") },
        .{ .key = "resolution", .kind = .uint, .offset = @offsetOf(Provenance, "resolution") },
        .{ .key = "sample_rows", .kind = .uint, .offset = @offsetOf(Provenance, "sample_rows") },
        .{ .key = "sample_seed", .kind = .uint, .offset = @offsetOf(Provenance, "sample_seed") },
        .{ .key = "schema", .kind = .uint, .offset = @offsetOf(Provenance, "schema") },
        .{ .key = "seed", .kind = .uint, .offset = @offsetOf(Provenance, "seed") },
        .{ .key = "steps", .kind = .uint, .offset = @offsetOf(Provenance, "steps") },
    };

    fn readUint(self: *const Provenance, f: Field) u64 {
        const base: [*]const u8 = @ptrCast(self);
        return switch (f.offset) {
            @offsetOf(Provenance, "resolution"), @offsetOf(Provenance, "steps"), @offsetOf(Provenance, "schema") =>
                @as(*const u32, @alignCast(@ptrCast(base + f.offset))).*,
            @offsetOf(Provenance, "seed"), @offsetOf(Provenance, "sample_seed") =>
                @as(*const u64, @alignCast(@ptrCast(base + f.offset))).*,
            else => @as(*const usize, @alignCast(@ptrCast(base + f.offset))).*,
        };
    }

    fn writeUint(self: *Provenance, f: Field, v: u64) !void {
        const base: [*]u8 = @ptrCast(self);
        switch (f.offset) {
            @offsetOf(Provenance, "resolution"), @offsetOf(Provenance, "steps"), @offsetOf(Provenance, "schema") => {
                @as(*u32, @alignCast(@ptrCast(base + f.offset))).* =
                    std.math.cast(u32, v) orelse return error.InvalidProvenance;
            },
            @offsetOf(Provenance, "seed"), @offsetOf(Provenance, "sample_seed") => {
                @as(*u64, @alignCast(@ptrCast(base + f.offset))).* = v;
            },
            else => {
                @as(*usize, @alignCast(@ptrCast(base + f.offset))).* =
                    std.math.cast(usize, v) orelse return error.InvalidProvenance;
            },
        }
    }

    fn readStr(self: *const Provenance, f: Field) []const u8 {
        const base: [*]const u8 = @ptrCast(self);
        return @as(*const []const u8, @alignCast(@ptrCast(base + f.offset))).*;
    }

    fn writeStr(self: *Provenance, f: Field, v: []const u8) void {
        const base: [*]u8 = @ptrCast(self);
        @as(*[]const u8, @alignCast(@ptrCast(base + f.offset))).* = v;
    }

    /// Render as the sorted string→string pairs the header carries. Strings are
    /// allocated in `arena`.
    pub fn toMetadata(self: *const Provenance, arena: std.mem.Allocator) ![]const [2][]const u8 {
        const out = try arena.alloc([2][]const u8, fields.len);
        for (fields, out) |f, *kv| {
            kv[0] = f.key;
            kv[1] = switch (f.kind) {
                .str => self.readStr(f),
                .uint => try std.fmt.allocPrint(arena, "{d}", .{self.readUint(f)}),
            };
        }
        return out;
    }

    /// Parse back from a header `__metadata__` map. Strings borrow from `map`
    /// (i.e. from the owning `SafeTensors`), which outlives the `Cache`.
    pub fn fromMetadata(map: ?std.json.ObjectMap) !Provenance {
        const m = map orelse return error.MissingProvenance;
        var p: Provenance = .{};
        inline for (fields) |f| {
            const v = m.get(f.key) orelse return error.MissingProvenance;
            if (v != .string) return error.InvalidProvenance;
            switch (f.kind) {
                .str => p.writeStr(f, v.string),
                .uint => try p.writeUint(f, std.fmt.parseInt(u64, v.string, 10) catch
                    return error.InvalidProvenance),
            }
        }
        return p;
    }
};

// ---------------------------------------------------------------------------
// Container writer
// ---------------------------------------------------------------------------

/// Where an entry's bytes come from, and how they are narrowed on the way out.
/// The narrowing arms exist so the writer never has to materialize a converted
/// copy of a multi-gigabyte row sample: `diag` is accumulated in f64 but stored
/// f32, and token indices are counted in u64 but stored I64.
pub const Source = union(enum) {
    f32s: []const f32,
    f64_as_f32: []const f64,
    u64_as_i64: []const u64,
    i64s: []const i64,

    fn len(self: Source) usize {
        return switch (self) {
            inline else => |s| s.len,
        };
    }

    fn elemBytes(self: Source) usize {
        return switch (self) {
            .f32s, .f64_as_f32 => 4,
            .u64_as_i64, .i64s => 8,
        };
    }

    fn dtypeName(self: Source) []const u8 {
        return switch (self) {
            .f32s, .f64_as_f32 => "F32",
            .u64_as_i64, .i64s => "I64",
        };
    }
};

pub const Entry = struct {
    name: []const u8,
    dims: []const usize,
    src: Source,
};

fn entryLessThan(_: void, a: Entry, b: Entry) bool {
    // The reference implementation orders by dtype size descending, then name.
    // Reproducing it is what makes our output byte-comparable with a
    // Python-written file; it also happens to keep every tensor's offset a
    // multiple of its element size, so a reader may borrow the mapping directly.
    const sa = a.src.elemBytes();
    const sb = b.src.elemBytes();
    if (sa != sb) return sa > sb;
    return std.mem.order(u8, a.name, b.name) == .lt;
}

fn metaLessThan(_: void, a: [2][]const u8, b: [2][]const u8) bool {
    return std.mem.order(u8, a[0], b[0]) == .lt;
}

fn writeSource(w: *std.Io.Writer, src: Source) !void {
    switch (src) {
        .f32s => |s| {
            if (native_le) {
                try w.writeAll(std.mem.sliceAsBytes(s));
            } else {
                for (s) |v| try w.writeInt(u32, @bitCast(v), .little);
            }
        },
        .f64_as_f32 => |s| for (s) |v| {
            try w.writeInt(u32, @bitCast(@as(f32, @floatCast(v))), .little);
        },
        .u64_as_i64 => |s| for (s) |v| {
            const iv = std.math.cast(i64, v) orelse return error.ValueOutOfRange;
            try w.writeInt(i64, iv, .little);
        },
        .i64s => |s| for (s) |v| try w.writeInt(i64, v, .little),
    }
}

/// Write a safetensors container. `entries` is sorted in place. `metadata` is
/// key/value pairs; it is sorted internally, so callers need not.
pub fn writeContainer(
    gpa: std.mem.Allocator,
    w: *std.Io.Writer,
    entries: []Entry,
    metadata: []const [2][]const u8,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    std.mem.sort(Entry, entries, {}, entryLessThan);

    var header: std.json.ObjectMap = .empty;

    if (metadata.len > 0) {
        const sorted = try arena.dupe([2][]const u8, metadata);
        std.mem.sort([2][]const u8, sorted, {}, metaLessThan);
        var meta: std.json.ObjectMap = .empty;
        for (sorted) |kv| try meta.put(arena, kv[0], .{ .string = kv[1] });
        try header.put(arena, "__metadata__", .{ .object = meta });
    }

    var off: u64 = 0;
    for (entries, 0..) |e, i| {
        if (i > 0 and std.mem.eql(u8, e.name, entries[i - 1].name)) return error.DuplicateTensor;

        var n: usize = 1;
        for (e.dims) |d| n *= d;
        if (n != e.src.len()) return error.ShapeMismatch;

        const bytes: u64 = @as(u64, e.src.len()) * e.src.elemBytes();

        var obj: std.json.ObjectMap = .empty;
        try obj.put(arena, "dtype", .{ .string = e.src.dtypeName() });
        var shape: std.json.Array = .init(arena);
        for (e.dims) |d| try shape.append(.{ .integer = @intCast(d) });
        try obj.put(arena, "shape", .{ .array = shape });
        var offsets: std.json.Array = .init(arena);
        try offsets.append(.{ .integer = @intCast(off) });
        try offsets.append(.{ .integer = @intCast(off + bytes) });
        try obj.put(arena, "data_offsets", .{ .array = offsets });
        try header.put(arena, e.name, .{ .object = obj });

        off += bytes;
    }

    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try std.json.Stringify.value(std.json.Value{ .object = header }, .{}, &aw.writer);
    const hdr = aw.written();

    // Pad the header with spaces so the data section starts 8-byte aligned.
    const pad = (8 - (hdr.len % 8)) % 8;
    try w.writeInt(u64, hdr.len + pad, .little);
    try w.writeAll(hdr);
    for (0..pad) |_| try w.writeByte(' ');

    for (entries) |e| try writeSource(w, e.src);
}

// ---------------------------------------------------------------------------
// Cache writer
// ---------------------------------------------------------------------------

/// Field suffixes, in one place so the writer and the reader agree by construction.
pub const field_diag = "diag";
pub const field_amax = "amax";
pub const field_rows = "rows";
pub const field_rows_idx = "rows_idx";
pub const field_count = "count";

fn keyFor(arena: std.mem.Allocator, layer: []const u8, bucket: usize, field: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{s}/b{d}/{s}", .{ layer, bucket, field });
}

/// Serialize a collector into `w`. `prov.buckets` / `prov.sample_rows` are
/// overwritten from the collector's own options — they describe the capture, and
/// letting a caller mis-state them would poison `validate`.
pub fn write(
    gpa: std.mem.Allocator,
    w: *std.Io.Writer,
    collector: *const Collector,
    prov: Provenance,
) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var p = prov;
    p.buckets = collector.opts.buckets;
    p.sample_rows = collector.opts.sample_rows;
    p.sample_seed = collector.opts.seed;
    p.schema = schema_version;

    var entries: std.ArrayList(Entry) = .empty;
    try entries.ensureTotalCapacity(arena, collector.layerCount() * collector.opts.buckets * 5);

    var it = collector.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const layer = e.value_ptr.*;
        for (layer.buckets, 0..) |*b, k| {
            const count = try arena.alloc(i64, 1);
            count[0] = std.math.cast(i64, b.count) orelse return error.ValueOutOfRange;

            const dims_cols = try arena.dupe(usize, &.{layer.cols});
            const dims_kept = try arena.dupe(usize, &.{b.kept});
            const dims_rows = try arena.dupe(usize, &.{ b.kept, layer.cols });
            const dims_one = try arena.dupe(usize, &.{1});

            try entries.appendSlice(arena, &.{
                .{ .name = try keyFor(arena, name, k, field_diag), .dims = dims_cols, .src = .{ .f64_as_f32 = b.diag } },
                .{ .name = try keyFor(arena, name, k, field_amax), .dims = dims_cols, .src = .{ .f32s = b.amax } },
                .{ .name = try keyFor(arena, name, k, field_rows), .dims = dims_rows, .src = .{ .f32s = b.sample(layer.cols) } },
                .{ .name = try keyFor(arena, name, k, field_rows_idx), .dims = dims_kept, .src = .{ .u64_as_i64 = b.row_index[0..b.kept] } },
                .{ .name = try keyFor(arena, name, k, field_count), .dims = dims_one, .src = .{ .i64s = count } },
            });
        }
    }

    try writeContainer(gpa, w, entries.items, try p.toMetadata(arena));
}

/// Serialize a collector to `path` under `dir`, truncating any existing file.
pub fn writeFileIn(
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: std.Io.Dir,
    path: []const u8,
    collector: *const Collector,
    prov: Provenance,
) !void {
    const file = try dir.createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var buf: [1 << 20]u8 = undefined;
    var fw = file.writer(io, &buf);
    try write(gpa, &fw.interface, collector, prov);
    try fw.interface.flush();
}

/// Serialize a collector to `path`, truncating any existing file.
pub fn writeFile(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    collector: *const Collector,
    prov: Provenance,
) !void {
    return writeFileIn(gpa, io, std.Io.Dir.cwd(), path, collector, prov);
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// One layer's statistics for one schedule bucket.
///
/// The accessors allocate rather than borrow the mapping: safetensors makes no
/// alignment promise about the data section, and a `[]const f32` cast over a
/// misaligned mapping is undefined behaviour that would only show up on some
/// targets. The level-1 harness reads each layer once, so a copy is not the
/// cost that matters.
pub const BucketView = struct {
    cache: *const Cache,
    layer: []const u8,
    bucket: usize,
    /// Input width of the layer's GEMM.
    cols: usize,
    /// Token rows actually retained.
    kept: usize,
    /// Tokens accumulated into `diag`/`amax`.
    count: u64,

    fn f32Field(self: BucketView, gpa: std.mem.Allocator, field: []const u8, want: usize) ![]f32 {
        const v = try self.cache.require(self.layer, self.bucket, field);
        if (v.info.elemCount() != want) return error.ShapeMismatch;
        if (v.info.dtype != .f32) return error.DTypeMismatch;
        return v.toF32Alloc(gpa);
    }

    /// Σ x_j² per input channel — `[cols]`.
    pub fn diagAlloc(self: BucketView, gpa: std.mem.Allocator) ![]f32 {
        return self.f32Field(gpa, field_diag, self.cols);
    }

    /// max |x_j| per input channel — `[cols]`.
    pub fn amaxAlloc(self: BucketView, gpa: std.mem.Allocator) ![]f32 {
        return self.f32Field(gpa, field_amax, self.cols);
    }

    /// The retained token rows, row-major `[kept, cols]`.
    pub fn rowsAlloc(self: BucketView, gpa: std.mem.Allocator) ![]f32 {
        return self.f32Field(gpa, field_rows, self.kept * self.cols);
    }

    /// Which token index each retained row came from — `[kept]`.
    pub fn rowsIdxAlloc(self: BucketView, gpa: std.mem.Allocator) ![]u64 {
        const v = try self.cache.require(self.layer, self.bucket, field_rows_idx);
        if (v.info.dtype != .i64) return error.DTypeMismatch;
        if (v.info.elemCount() != self.kept) return error.ShapeMismatch;
        const out = try gpa.alloc(u64, self.kept);
        errdefer gpa.free(out);
        for (out, 0..) |*o, i| {
            const raw = std.mem.readInt(i64, v.bytes[i * 8 ..][0..8], .little);
            if (raw < 0) return error.InvalidValue;
            o.* = @intCast(raw);
        }
        return out;
    }
};

pub const Cache = struct {
    gpa: std.mem.Allocator,
    st: tp.safetensors.SafeTensors,
    arena: std.heap.ArenaAllocator,
    prov: Provenance,
    /// Sorted, deduplicated layer names.
    layer_names: []const []const u8,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Cache {
        return openIn(gpa, io, std.Io.Dir.cwd(), path);
    }

    pub fn openIn(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Cache {
        var st = try tp.safetensors.SafeTensors.openIn(gpa, io, dir, path);
        errdefer st.deinit();
        return init(gpa, st);
    }

    /// Take ownership of an already-parsed container (used by the in-memory tests
    /// and by any caller that has the bytes already).
    pub fn init(gpa: std.mem.Allocator, st: tp.safetensors.SafeTensors) !Cache {
        var self: Cache = .{
            .gpa = gpa,
            .st = st,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .prov = undefined,
            .layer_names = &.{},
        };
        errdefer self.arena.deinit();
        const arena = self.arena.allocator();

        self.prov = try Provenance.fromMetadata(self.st.metadata);

        // Layer names come from the `count` entries: exactly one per (layer,
        // bucket), so deriving from them cannot invent a layer out of a stray
        // tensor and cannot miss one that was written.
        var names: std.ArrayList([]const u8) = .empty;
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(gpa);
        for (self.st.names()) |n| {
            const parsed = parseKey(n) orelse continue;
            if (!std.mem.eql(u8, parsed.field, field_count)) continue;
            if ((try seen.getOrPut(gpa, parsed.layer)).found_existing) continue;
            try names.append(arena, try arena.dupe(u8, parsed.layer));
        }
        std.mem.sort([]const u8, names.items, {}, strLessThan);
        self.layer_names = names.items;
        return self;
    }

    pub fn deinit(self: *Cache) void {
        self.st.deinit();
        self.arena.deinit();
        self.* = undefined;
    }

    pub fn layers(self: *const Cache) []const []const u8 {
        return self.layer_names;
    }

    fn require(self: *const Cache, layer: []const u8, k: usize, field: []const u8) !tp.safetensors.TensorView {
        var buf: [512]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "{s}/b{d}/{s}", .{ layer, k, field }) catch
            return error.NameTooLong;
        return self.st.get(key) orelse error.MissingTensor;
    }

    /// A layer's statistics for one bucket. Reads only the shape/count headers;
    /// the arrays are fetched by the `BucketView` accessors on demand.
    pub fn bucket(self: *const Cache, layer: []const u8, k: usize) !BucketView {
        const diag = try self.require(layer, k, field_diag);
        if (diag.info.shape.rank != 1) return error.ShapeMismatch;
        const rows = try self.require(layer, k, field_rows);
        if (rows.info.shape.rank != 2) return error.ShapeMismatch;
        const cnt = try self.require(layer, k, field_count);
        if (cnt.info.dtype != .i64 or cnt.info.elemCount() != 1) return error.ShapeMismatch;

        const cols = diag.info.shape.dims[0];
        if (rows.info.shape.dims[1] != cols) return error.ShapeMismatch;
        const raw = std.mem.readInt(i64, cnt.bytes[0..8], .little);
        if (raw < 0) return error.InvalidValue;

        return .{
            .cache = self,
            .layer = layer,
            .bucket = k,
            .cols = cols,
            .kept = rows.info.shape.dims[0],
            .count = @intCast(raw),
        };
    }
};

fn strLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

const ParsedKey = struct { layer: []const u8, bucket: usize, field: []const u8 };

/// Split `<layer>/b<k>/<field>`. Layer names contain dots but never slashes, so
/// the last two separators are unambiguous.
fn parseKey(name: []const u8) ?ParsedKey {
    const last = std.mem.lastIndexOfScalar(u8, name, '/') orelse return null;
    const head = name[0..last];
    const prev = std.mem.lastIndexOfScalar(u8, head, '/') orelse return null;
    const btok = head[prev + 1 ..];
    if (btok.len < 2 or btok[0] != 'b') return null;
    const k = std.fmt.parseInt(usize, btok[1..], 10) catch return null;
    if (prev == 0) return null;
    return .{ .layer = name[0..prev], .bucket = k, .field = name[last + 1 ..] };
}

// ---------------------------------------------------------------------------
// Sanity gate
// ---------------------------------------------------------------------------

pub const ValidateOptions = struct {
    /// If set, the cache's `model_hash` must equal this. Leave null only when the
    /// caller has no hash to check against — a mismatched cache is otherwise
    /// undetectable.
    model_hash: ?[]const u8 = null,
    /// If set, every layer must exist in this checkpoint with a matching input
    /// width. This is the check that catches a cache captured from a differently
    /// shaped variant of the same architecture.
    checkpoint: ?*const tp.safetensors.SafeTensors = null,
    /// Scan every element of `rows` for non-finite values. On by default: a
    /// single NaN in the sample poisons every metric computed from it, and the
    /// scan is linear in data we are about to read anyway.
    scan_rows: bool = true,
};

/// Why validation failed, in words. Filled only on error; `msg` points into `buf`.
pub const Diagnostic = struct {
    buf: [512]u8 = undefined,
    msg: []const u8 = "",

    fn set(self: *Diagnostic, comptime fmt: []const u8, args: anytype) void {
        self.msg = std.fmt.bufPrint(&self.buf, fmt, args) catch self.buf[0..];
    }
};

fn fail(diag: ?*Diagnostic, err: anyerror, comptime fmt: []const u8, args: anytype) anyerror {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// Refuse a cache that cannot be trusted. Call this before consuming one.
///
/// Checks: schema version, non-empty, every (layer, bucket) present and
/// self-consistent, `kept` within the declared sample bound, no non-finite or
/// negative energies, and — when the caller supplies them — the model hash and
/// the checkpoint's per-layer input width.
pub fn validate(cache: *const Cache, opts: ValidateOptions, diag: ?*Diagnostic) !void {
    if (cache.prov.schema != schema_version) {
        return fail(diag, error.SchemaMismatch, "cache schema {d}, expected {d}", .{ cache.prov.schema, schema_version });
    }
    if (cache.prov.buckets == 0 or cache.prov.sample_rows == 0) {
        return fail(diag, error.InvalidProvenance, "buckets={d} sample_rows={d}", .{ cache.prov.buckets, cache.prov.sample_rows });
    }
    if (cache.layer_names.len == 0) {
        return fail(diag, error.NoLayers, "cache contains no layers", .{});
    }
    if (opts.model_hash) |want| {
        if (!std.mem.eql(u8, want, cache.prov.model_hash)) {
            return fail(diag, error.ModelHashMismatch, "cache model_hash '{s}', expected '{s}'", .{ cache.prov.model_hash, want });
        }
    }

    const gpa = cache.gpa;
    var total_tokens: u64 = 0;

    for (cache.layer_names) |name| {
        var layer_tokens: u64 = 0;
        var cols: ?usize = null;

        for (0..cache.prov.buckets) |k| {
            const b = cache.bucket(name, k) catch |err| {
                return fail(diag, err, "{s} bucket {d}: {t}", .{ name, k, err });
            };

            if (cols) |c| {
                if (b.cols != c) return fail(diag, error.ShapeMismatch, "{s}: cols {d} in bucket {d}, {d} elsewhere", .{ name, b.cols, k, c });
            } else cols = b.cols;

            if (b.cols == 0) return fail(diag, error.ShapeMismatch, "{s} bucket {d}: zero columns", .{ name, k });
            if (b.kept > cache.prov.sample_rows) {
                return fail(diag, error.ShapeMismatch, "{s} bucket {d}: kept {d} > sample_rows {d}", .{ name, k, b.kept, cache.prov.sample_rows });
            }
            // Rows can only have been retained from tokens that were counted, and
            // a bucket that counted tokens must have retained at least one.
            if (b.kept > b.count or (b.count > 0 and b.kept == 0)) {
                return fail(diag, error.InconsistentCounts, "{s} bucket {d}: kept {d}, count {d}", .{ name, k, b.kept, b.count });
            }
            layer_tokens += b.count;

            const d = b.diagAlloc(gpa) catch |err| return fail(diag, err, "{s} bucket {d} diag: {t}", .{ name, k, err });
            defer gpa.free(d);
            for (d, 0..) |v, j| {
                if (!std.math.isFinite(v)) return fail(diag, error.NonFinite, "{s} bucket {d} diag[{d}] = {d}", .{ name, k, j, v });
                if (v < 0) return fail(diag, error.NegativeEnergy, "{s} bucket {d} diag[{d}] = {d}", .{ name, k, j, v });
            }

            const a = b.amaxAlloc(gpa) catch |err| return fail(diag, err, "{s} bucket {d} amax: {t}", .{ name, k, err });
            defer gpa.free(a);
            for (a, 0..) |v, j| {
                if (!std.math.isFinite(v)) return fail(diag, error.NonFinite, "{s} bucket {d} amax[{d}] = {d}", .{ name, k, j, v });
                if (v < 0) return fail(diag, error.NegativeEnergy, "{s} bucket {d} amax[{d}] = {d}", .{ name, k, j, v });
            }

            const idx = b.rowsIdxAlloc(gpa) catch |err| return fail(diag, err, "{s} bucket {d} rows_idx: {t}", .{ name, k, err });
            gpa.free(idx);

            if (opts.scan_rows) {
                const r = b.rowsAlloc(gpa) catch |err| return fail(diag, err, "{s} bucket {d} rows: {t}", .{ name, k, err });
                defer gpa.free(r);
                for (r, 0..) |v, j| {
                    if (!std.math.isFinite(v)) return fail(diag, error.NonFinite, "{s} bucket {d} rows[{d}] = {d}", .{ name, k, j, v });
                }
            }
        }

        if (layer_tokens == 0) {
            return fail(diag, error.EmptyLayer, "{s}: no tokens in any bucket", .{name});
        }
        total_tokens += layer_tokens;

        if (opts.checkpoint) |ck| {
            const v = ck.get(name) orelse
                return fail(diag, error.LayerNotInCheckpoint, "{s} is not in the checkpoint", .{name});
            const rank = v.info.shape.rank;
            if (rank == 0) return fail(diag, error.ShapeMismatch, "{s}: checkpoint tensor is rank 0", .{name});
            // A linear's weight is [out, in]; the GEMM input width is the last dim.
            const want = v.info.shape.dims[rank - 1];
            if (want != cols.?) {
                return fail(diag, error.ShapeMismatch, "{s}: cache cols {d}, checkpoint {d}", .{ name, cols.?, want });
            }
        }
    }

    if (total_tokens == 0) return fail(diag, error.EmptyCache, "no tokens accumulated anywhere", .{});
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn writeToBuf(gpa: std.mem.Allocator, collector: *const Collector, prov: Provenance) ![]u8 {
    var aw: std.Io.Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    try write(gpa, &aw.writer, collector, prov);
    return aw.toOwnedSlice();
}

/// Drive one tagged GEMM through TensorPencil so the collector observes `x`
/// exactly as a model forward would deliver it.
fn observe(collector: *Collector, tag: []const u8, x: []const f32, m: usize, cols: usize) !void {
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

const test_prov: Provenance = .{
    .model_path = "/models/krea2.safetensors",
    .model_hash = "deadbeef",
    .arch = "krea2",
    .prompt_set = "default-16",
    .backend = "cpu",
    .producer = "ggufy-test",
    .resolution = 512,
    .steps = 4,
    .seed = 1234,
};

/// The (layer, bucket) group both container fixtures hold, as the writer's inputs.
fn fixtureEntries() [5]Entry {
    const S = struct {
        const diag = [_]f64{ 1.5, 2.5, 3.5, 4.5 };
        const amax = [_]f32{ 0.5, 9.25, 1.0, 0.125 };
        const rows = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
        const idx = [_]u64{ 3, 11 };
        const count = [_]i64{42};
    };
    const n = "blocks.0.attn.wq.weight/b0/";
    return .{
        .{ .name = n ++ field_diag, .dims = &.{4}, .src = .{ .f64_as_f32 = &S.diag } },
        .{ .name = n ++ field_amax, .dims = &.{4}, .src = .{ .f32s = &S.amax } },
        .{ .name = n ++ field_rows, .dims = &.{ 2, 4 }, .src = .{ .f32s = &S.rows } },
        .{ .name = n ++ field_rows_idx, .dims = &.{2}, .src = .{ .u64_as_i64 = &S.idx } },
        .{ .name = n ++ field_count, .dims = &.{1}, .src = .{ .i64s = &S.count } },
    };
}

test "the container writer reproduces the reference implementation's bytes" {
    // Pinned against a file written by the Python `safetensors` package
    // (gen_calib_fixtures.py). This is the external check on the container
    // format: dtype spellings, entry ordering (dtype size descending, then
    // name), offset arithmetic, and the space-padded 8-byte-aligned header.
    //
    // The fixture carries no `__metadata__`: the reference implementation keeps
    // it in a Rust HashMap and emits its keys in a per-process random order, so
    // a file with metadata is not byte-comparable against anything. Metadata is
    // covered read-side by the next test.
    const gpa = testing.allocator;
    const golden = @embedFile("test_fixtures/calib_container.safetensors");

    var entries = fixtureEntries();
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try writeContainer(gpa, &aw.writer, &entries, &.{});
    const got = aw.written();

    if (!std.mem.eql(u8, golden, got)) {
        const n = @min(golden.len, got.len);
        var first: usize = 0;
        while (first < n and golden[first] == got[first]) first += 1;
        std.debug.print(
            "container bytes differ: golden {d} B, got {d} B, first diff at {d}\n  golden: '{s}'\n  got:    '{s}'\n",
            .{ golden.len, got.len, first, golden[first..@min(golden.len, first + 48)], got[first..@min(got.len, first + 48)] },
        );
        return error.TestExpectedEqual;
    }
}

test "the reader parses a reference-written container, metadata included" {
    // The read side of the same pin: values, shapes and the `__metadata__` map
    // as the reference implementation emits them (in whatever key order it
    // chose this run).
    const gpa = testing.allocator;
    const bytes = @embedFile("test_fixtures/calib_container_meta.safetensors");

    var st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes);
    defer st.deinit();

    const meta = st.metadata.?;
    try testing.expectEqualStrings("krea2", meta.get("arch").?.string);
    try testing.expectEqualStrings("cpu", meta.get("backend").?.string);
    try testing.expectEqualStrings("1", meta.get("schema").?.string);

    const rows = try st.require("blocks.0.attn.wq.weight/b0/rows");
    try testing.expectEqual(@as(usize, 2), rows.info.shape.rank);
    try testing.expectEqual(@as(usize, 2), rows.info.shape.dims[0]);
    try testing.expectEqual(@as(usize, 4), rows.info.shape.dims[1]);
    const rv = try rows.toF32Alloc(gpa);
    defer gpa.free(rv);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, rv);

    const diag = try (try st.require("blocks.0.attn.wq.weight/b0/diag")).toF32Alloc(gpa);
    defer gpa.free(diag);
    try testing.expectEqualSlices(f32, &.{ 1.5, 2.5, 3.5, 4.5 }, diag);

    const idx = try st.require("blocks.0.attn.wq.weight/b0/rows_idx");
    try testing.expectEqual(tp.DType.i64, idx.info.dtype);
    try testing.expectEqual(@as(i64, 3), std.mem.readInt(i64, idx.bytes[0..8], .little));
    try testing.expectEqual(@as(i64, 11), std.mem.readInt(i64, idx.bytes[8..16], .little));

    const cnt = try st.require("blocks.0.attn.wq.weight/b0/count");
    try testing.expectEqual(@as(i64, 42), std.mem.readInt(i64, cnt.bytes[0..8], .little));
}

test "a captured collector round-trips through the cache unchanged" {
    const gpa = testing.allocator;
    var c = Collector.init(gpa, .{ .sample_rows = 4, .buckets = 2 });
    defer c.deinit();

    // Two layers of different widths, tokens split across two buckets, and one
    // bucket left empty so the zero-row case is covered.
    const x4 = [_]f32{
        1, 0,  2,  10,
        2, -1, 0,  -20,
        0, 1,  1,  5,
        3, 2,  -4, 1,
    };
    const x2 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    c.setBucket(0);
    try observe(&c, "blocks.0.attn.wq.weight", &x4, 4, 4);
    try observe(&c, "blocks.0.mlp.down.weight", &x2, 3, 2);
    c.setBucket(1);
    try observe(&c, "blocks.0.attn.wq.weight", &x4, 4, 4);
    try c.checkOk();

    const bytes = try writeToBuf(gpa, &c, test_prov);
    defer gpa.free(bytes);

    var st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes);
    var cache = try Cache.init(gpa, st);
    errdefer st.deinit();
    defer cache.deinit();

    try testing.expectEqualStrings("krea2", cache.prov.arch);
    try testing.expectEqualStrings("deadbeef", cache.prov.model_hash);
    try testing.expectEqual(@as(u32, 512), cache.prov.resolution);
    try testing.expectEqual(@as(u64, 1234), cache.prov.seed);
    // Capture-derived fields come from the collector, not the caller's struct.
    try testing.expectEqual(@as(usize, 2), cache.prov.buckets);
    try testing.expectEqual(@as(usize, 4), cache.prov.sample_rows);

    try testing.expectEqual(@as(usize, 2), cache.layers().len);
    try testing.expectEqualStrings("blocks.0.attn.wq.weight", cache.layers()[0]);
    try testing.expectEqualStrings("blocks.0.mlp.down.weight", cache.layers()[1]);

    var it = c.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        const layer = e.value_ptr.*;
        for (layer.buckets, 0..) |*want, k| {
            const b = try cache.bucket(name, k);
            try testing.expectEqual(layer.cols, b.cols);
            try testing.expectEqual(want.kept, b.kept);
            try testing.expectEqual(want.count, b.count);

            const d = try b.diagAlloc(gpa);
            defer gpa.free(d);
            for (want.diag, d) |w, g| try testing.expectEqual(@as(f32, @floatCast(w)), g);

            const a = try b.amaxAlloc(gpa);
            defer gpa.free(a);
            try testing.expectEqualSlices(f32, want.amax, a);

            const r = try b.rowsAlloc(gpa);
            defer gpa.free(r);
            try testing.expectEqualSlices(f32, want.sample(layer.cols), r);

            const idx = try b.rowsIdxAlloc(gpa);
            defer gpa.free(idx);
            try testing.expectEqualSlices(u64, want.row_index[0..want.kept], idx);
        }
    }
}

test "a written cache is also readable by ggufy's own safetensors parser" {
    // The cache is meant to be a plain safetensors file, not a private format:
    // TensorPencil reads it above, and the repo's own parser must read the same
    // bytes. Two independent parsers agreeing is the check that we did not
    // encode something only one of them tolerates.
    const gpa = testing.allocator;
    var c = Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();
    const x = [_]f32{ 1, 2, 3, 4 };
    try observe(&c, "first.weight", &x, 2, 2);

    const bytes = try writeToBuf(gpa, &c, test_prov);
    defer gpa.free(bytes);

    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes[8 .. 8 + header_len], .{});
    defer parsed.deinit();

    const root = parsed.value.object;
    const meta = root.get("__metadata__").?.object;
    try testing.expectEqualStrings("krea2", meta.get("arch").?.string);

    const rows = root.get("first.weight/b0/rows").?.object;
    try testing.expectEqualStrings("F32", rows.get("dtype").?.string);
    const shape = rows.get("shape").?.array.items;
    try testing.expectEqual(@as(i64, 2), shape[0].integer);
    try testing.expectEqual(@as(i64, 2), shape[1].integer);
    const offs = rows.get("data_offsets").?.array.items;
    try testing.expectEqual(@as(i64, 16), offs[1].integer - offs[0].integer);
}

/// Build a valid single-layer cache in memory for the gate tests.
fn buildValidCache(gpa: std.mem.Allocator, bytes_out: *[]u8) !Cache {
    var c = Collector.init(gpa, .{ .sample_rows = 4, .buckets = 2 });
    defer c.deinit();
    const x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    c.setBucket(0);
    try observe(&c, "blocks.0.attn.wq.weight", &x, 3, 2);
    c.setBucket(1);
    try observe(&c, "blocks.0.attn.wq.weight", &x, 3, 2);

    bytes_out.* = try writeToBuf(gpa, &c, test_prov);
    errdefer gpa.free(bytes_out.*);
    const st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes_out.*);
    return Cache.init(gpa, st);
}

test "the sanity gate accepts a good cache and reports what is wrong with a bad one" {
    const gpa = testing.allocator;

    var bytes: []u8 = undefined;
    var cache = try buildValidCache(gpa, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    try validate(&cache, .{ .model_hash = "deadbeef" }, null);

    // Wrong model: the single check that catches a cache from another checkpoint.
    var diag: Diagnostic = .{};
    try testing.expectError(error.ModelHashMismatch, validate(&cache, .{ .model_hash = "cafe" }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.msg, "deadbeef") != null);

    // A layer the checkpoint does not have.
    var ck_bytes: std.ArrayList(u8) = .empty;
    defer ck_bytes.deinit(gpa);
    {
        const header =
            \\{"other.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}
        ;
        try ck_bytes.appendSlice(gpa, &std.mem.toBytes(@as(u64, header.len)));
        try ck_bytes.appendSlice(gpa, header);
        try ck_bytes.appendNTimes(gpa, 0, 16);
    }
    var ck = try tp.safetensors.SafeTensors.initFromSlice(gpa, ck_bytes.items);
    defer ck.deinit();
    try testing.expectError(error.LayerNotInCheckpoint, validate(&cache, .{ .checkpoint = &ck }, &diag));

    // A checkpoint whose layer exists but is the wrong width.
    var wide_bytes: std.ArrayList(u8) = .empty;
    defer wide_bytes.deinit(gpa);
    {
        const header =
            \\{"blocks.0.attn.wq.weight":{"dtype":"F32","shape":[2,8],"data_offsets":[0,64]}}
        ;
        try wide_bytes.appendSlice(gpa, &std.mem.toBytes(@as(u64, header.len)));
        try wide_bytes.appendSlice(gpa, header);
        try wide_bytes.appendNTimes(gpa, 0, 64);
    }
    var wide = try tp.safetensors.SafeTensors.initFromSlice(gpa, wide_bytes.items);
    defer wide.deinit();
    try testing.expectError(error.ShapeMismatch, validate(&cache, .{ .checkpoint = &wide }, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.msg, "cache cols 2") != null);
}

test "the sanity gate rejects a NaN that a shape check would miss" {
    const gpa = testing.allocator;
    var c = Collector.init(gpa, .{ .sample_rows = 4, .buckets = 1 });
    defer c.deinit();
    // A finite-looking capture with one poisoned token: shapes, counts and
    // metadata are all correct, so only the value scan can catch it.
    const x = [_]f32{ 1, 2, std.math.nan(f32), 4 };
    try observe(&c, "blocks.0.attn.wq.weight", &x, 2, 2);

    const bytes = try writeToBuf(gpa, &c, test_prov);
    defer gpa.free(bytes);
    const st = try tp.safetensors.SafeTensors.initFromSlice(gpa, bytes);
    var cache = try Cache.init(gpa, st);
    defer cache.deinit();

    var diag: Diagnostic = .{};
    try testing.expectError(error.NonFinite, validate(&cache, .{}, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.msg, "blocks.0.attn.wq.weight") != null);

    // With the row scan off, the NaN still shows up in `diag` (Σx² propagates
    // it), which is why the cheap path is not blind to this.
    try testing.expectError(error.NonFinite, validate(&cache, .{ .scan_rows = false }, &diag));
}

test "the sanity gate rejects a schema it does not understand" {
    const gpa = testing.allocator;
    var bytes: []u8 = undefined;
    var cache = try buildValidCache(gpa, &bytes);
    defer gpa.free(bytes);
    defer cache.deinit();

    cache.prov.schema = schema_version + 1;
    var diag: Diagnostic = .{};
    try testing.expectError(error.SchemaMismatch, validate(&cache, .{}, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.msg, "expected") != null);
}

test "a cache with no provenance is refused rather than read with defaults" {
    const gpa = testing.allocator;
    const header =
        \\{"a.weight/b0/count":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}
    ;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    try buf.appendSlice(gpa, &std.mem.toBytes(@as(u64, header.len)));
    try buf.appendSlice(gpa, header);
    try buf.appendNTimes(gpa, 0, 8);

    var st = try tp.safetensors.SafeTensors.initFromSlice(gpa, buf.items);
    errdefer st.deinit();
    try testing.expectError(error.MissingProvenance, Cache.init(gpa, st));
    st.deinit();
}

test "writeFile produces the same bytes as the in-memory writer" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var c = Collector.init(gpa, .{ .sample_rows = 2, .buckets = 1 });
    defer c.deinit();
    const x = [_]f32{ 1, 2, 3, 4 };
    try observe(&c, "first.weight", &x, 2, 2);

    const want = try writeToBuf(gpa, &c, test_prov);
    defer gpa.free(want);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    try writeFileIn(gpa, io, tmp.dir, "cache.safetensors", &c, test_prov);

    const got = try tmp.dir.readFileAlloc(io, "cache.safetensors", gpa, .unlimited);
    defer gpa.free(got);
    try testing.expectEqualSlices(u8, want, got);

    // And it opens from disk, which the in-memory path does not exercise.
    var cache = try Cache.openIn(gpa, io, tmp.dir, "cache.safetensors");
    defer cache.deinit();
    try testing.expectEqual(@as(usize, 1), cache.layers().len);
}

test "malformed keys are ignored rather than inventing layers" {
    try testing.expect(parseKey("no-slashes") == null);
    try testing.expect(parseKey("only/one") == null);
    try testing.expect(parseKey("a/x0/diag") == null); // bucket token must start with 'b'
    try testing.expect(parseKey("a/bxx/diag") == null); // ...and parse as a number
    try testing.expect(parseKey("/b0/diag") == null); // ...and leave a non-empty layer

    const p = parseKey("blocks.0.attn.wq.weight/b2/rows_idx").?;
    try testing.expectEqualStrings("blocks.0.attn.wq.weight", p.layer);
    try testing.expectEqual(@as(usize, 2), p.bucket);
    try testing.expectEqualStrings("rows_idx", p.field);
}

test "duplicate and mis-shaped entries are rejected by the container writer" {
    const gpa = testing.allocator;
    const v = [_]f32{ 1, 2 };

    var dup = [_]Entry{
        .{ .name = "a/b0/diag", .dims = &.{2}, .src = .{ .f32s = &v } },
        .{ .name = "a/b0/diag", .dims = &.{2}, .src = .{ .f32s = &v } },
    };
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try testing.expectError(error.DuplicateTensor, writeContainer(gpa, &aw.writer, &dup, &.{}));

    var bad = [_]Entry{
        .{ .name = "a/b0/diag", .dims = &.{3}, .src = .{ .f32s = &v } },
    };
    var aw2: std.Io.Writer.Allocating = .init(gpa);
    defer aw2.deinit();
    try testing.expectError(error.ShapeMismatch, writeContainer(gpa, &aw2.writer, &bad, &.{}));
}
