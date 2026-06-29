const std = @import("std");
const types = @import("types.zig");
const ScaledQuant = @import("TensorClusters.zig");
const Safetensor = @import("Safetensor.zig");

pub const ComponentInfo = struct {
    name: []const u8,
    dir_path: []const u8,
    class_name: []const u8,
};

pub const RepoInfo = struct {
    pipeline_class: []const u8,
    components: []ComponentInfo,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *RepoInfo) void {
        for (self.components) |c| {
            self.allocator.free(c.name);
            self.allocator.free(c.dir_path);
            self.allocator.free(c.class_name);
        }
        self.allocator.free(self.components);
        self.allocator.free(self.pipeline_class);
    }
};

pub const MergedSource = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    tensors: std.ArrayList(types.Tensor),
    qkv_fusions: std.ArrayList(ScaledQuant.QkvFusionCluster),

    current_file_handle: ?std.Io.File = null,
    current_open_path: []const u8 = "",
    current_data_begin: u64 = 0,

    pub fn openFileForTensor(self: *MergedSource, name: []const u8) !std.Io.File {
        for (self.tensors.items) |t| {
            if (std.mem.eql(u8, t.name, name)) {
                const tensor_path = t.source_path orelse return error.TensorNotFound;
                if (!std.mem.eql(u8, self.current_open_path, tensor_path)) {
                    if (self.current_file_handle) |h| h.close(self.io);
                    const new_file = try std.Io.Dir.cwd().openFile(self.io, tensor_path, .{});
                    self.current_file_handle = new_file;
                    self.current_open_path = tensor_path;
                    var len_bytes: [8]u8 = undefined;
                    _ = try new_file.readPositionalAll(self.io, len_bytes[0..], 0);
                    const st_len = std.mem.readInt(u64, len_bytes[0..8], .little);
                    self.current_data_begin = 8 + st_len;
                }
                return self.current_file_handle.?;
            }
        }
        return error.TensorNotFound;
    }

    pub fn getSourceMetadata(_: *MergedSource) ?std.json.ObjectMap {
        return null;
    }

    pub fn deinit(self: *MergedSource) void {
        if (self.current_file_handle) |h| h.close(self.io);
    }
};

/// Parse model_index.json from a HuggingFace diffusers repo directory.
/// Returns error.ModelIndexNotFound if absent.
pub fn parseModelIndex(
    dir_path: []const u8,
    io: std.Io,
    allocator: std.mem.Allocator,
) !RepoInfo {
    const index_path = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, "model_index.json" });
    defer allocator.free(index_path);

    const file = std.Io.Dir.cwd().openFile(io, index_path, .{}) catch return error.ModelIndexNotFound;
    defer file.close(io);

    var buf: [65536]u8 = undefined;
    var reader = file.reader(io, &buf);
    const content = try reader.interface.allocRemaining(allocator, .unlimited);
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const obj = parsed.value.object;
    const pipeline_class = try allocator.dupe(u8, obj.get("_class_name").?.string);

    var components: std.ArrayList(ComponentInfo) = .empty;
    errdefer {
        for (components.items) |c| {
            allocator.free(c.name);
            allocator.free(c.dir_path);
            allocator.free(c.class_name);
        }
        components.deinit(allocator);
    }

    var it = obj.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.startsWith(u8, key, "_")) continue;
        const val = entry.value_ptr.*;
        if (val != .array) continue;
        const arr = val.array;
        if (arr.items.len < 2) continue;
        const class_str = switch (arr.items[1]) {
            .string => |s| s,
            else => continue,
        };
        const comp_dir = try std.fs.path.join(allocator, &[_][]const u8{ dir_path, key });
        errdefer allocator.free(comp_dir);
        try components.append(allocator, .{
            .name = try allocator.dupe(u8, key),
            .dir_path = comp_dir,
            .class_name = try allocator.dupe(u8, class_str),
        });
    }

    return RepoInfo{
        .pipeline_class = pipeline_class,
        .components = try components.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// Find the "transformer" or "unet" component from a parsed repo.
pub fn getDiffusionComponent(repo: *const RepoInfo) ?*const ComponentInfo {
    for (repo.components) |*c| {
        if (std.mem.eql(u8, c.name, "transformer") or
            std.mem.eql(u8, c.name, "unet"))
        {
            return c;
        }
    }
    return null;
}

/// Return the tensor name prefix used when building a merged checkpoint.
pub fn defaultPrefix(component_name: []const u8) []const u8 {
    if (std.mem.eql(u8, component_name, "transformer") or
        std.mem.eql(u8, component_name, "unet"))
    {
        return "model.diffusion_model.";
    }
    if (std.mem.eql(u8, component_name, "vae")) return "first_stage_model.";
    if (std.mem.eql(u8, component_name, "text_encoder")) return "cond_stage_model.";
    return "";
}

// ============================================================================
// ZImagePipeline transform helpers
// ============================================================================

const QkvComponent = enum { q, k, v };

const QkvMatch = struct {
    prefix: []const u8,
    component: QkvComponent,
};

fn matchQkv(name: []const u8) ?QkvMatch {
    const suffixes = [_]struct {
        suffix: []const u8,
        component: QkvComponent,
    }{
        .{ .suffix = ".attention.to_q.weight", .component = .q },
        .{ .suffix = ".attention.to_k.weight", .component = .k },
        .{ .suffix = ".attention.to_v.weight", .component = .v },
    };
    for (suffixes) |p| {
        if (std.mem.endsWith(u8, name, p.suffix)) {
            return .{
                .prefix = name[0 .. name.len - p.suffix.len],
                .component = p.component,
            };
        }
    }
    return null;
}

/// Apply ZImagePipeline tensor renaming rules.
/// Returns the new name (arena-allocated), or null if this tensor is a QKV
/// component that should be handled via a fusion cluster.
fn zImageRename(name: []const u8, arena_alloc: std.mem.Allocator) !?[]const u8 {
    // Top-level prefix renames
    const working: []const u8 = blk: {
        if (std.mem.startsWith(u8, name, "all_final_layer.2-1.")) {
            break :blk try std.fmt.allocPrint(arena_alloc, "final_layer.{s}", .{name["all_final_layer.2-1.".len..]});
        } else if (std.mem.startsWith(u8, name, "all_x_embedder.2-1.")) {
            break :blk try std.fmt.allocPrint(arena_alloc, "x_embedder.{s}", .{name["all_x_embedder.2-1.".len..]});
        } else {
            break :blk name;
        }
    };

    // QKV components → fusion cluster (signal with null)
    if (matchQkv(working) != null) return null;

    // Attention sub-key renames
    if (std.mem.endsWith(u8, working, ".attention.norm_k.weight")) {
        const p = working[0 .. working.len - ".attention.norm_k.weight".len];
        return try std.fmt.allocPrint(arena_alloc, "{s}.attention.k_norm.weight", .{p});
    }
    if (std.mem.endsWith(u8, working, ".attention.norm_q.weight")) {
        const p = working[0 .. working.len - ".attention.norm_q.weight".len];
        return try std.fmt.allocPrint(arena_alloc, "{s}.attention.q_norm.weight", .{p});
    }
    if (std.mem.endsWith(u8, working, ".attention.to_out.0.weight")) {
        const p = working[0 .. working.len - ".attention.to_out.0.weight".len];
        return try std.fmt.allocPrint(arena_alloc, "{s}.attention.out.weight", .{p});
    }

    return try arena_alloc.dupe(u8, working);
}

// ============================================================================
// Krea2Pipeline transform helpers
// ============================================================================

const Krea2Renamed = struct {
    name: []const u8,
    /// Possibly reshaped (flattened) dims; arena-allocated.
    dims: []usize,
};

const Krea2Sub = struct {
    suffix: []const u8,
    /// When true, the source tensor's dims are flattened to 1-D in the output.
    flatten: bool,
};

/// Map the per-block sub-key of a Krea2 diffusers tensor to its native name.
/// Shared by transformer_blocks / text_fusion.layerwise_blocks / refiner_blocks.
/// Returns null for an unrecognised sub-key.
fn krea2BlockSub(sub: []const u8) ?Krea2Sub {
    const table = [_]struct { src: []const u8, dst: []const u8, flatten: bool }{
        .{ .src = "attn.norm_k.weight", .dst = "attn.qknorm.knorm.scale", .flatten = false },
        .{ .src = "attn.norm_q.weight", .dst = "attn.qknorm.qnorm.scale", .flatten = false },
        .{ .src = "attn.to_gate.weight", .dst = "attn.gate.weight", .flatten = false },
        .{ .src = "attn.to_k.weight", .dst = "attn.wk.weight", .flatten = false },
        .{ .src = "attn.to_out.0.weight", .dst = "attn.wo.weight", .flatten = false },
        .{ .src = "attn.to_q.weight", .dst = "attn.wq.weight", .flatten = false },
        .{ .src = "attn.to_v.weight", .dst = "attn.wv.weight", .flatten = false },
        .{ .src = "ff.down.weight", .dst = "mlp.down.weight", .flatten = false },
        .{ .src = "ff.gate.weight", .dst = "mlp.gate.weight", .flatten = false },
        .{ .src = "ff.up.weight", .dst = "mlp.up.weight", .flatten = false },
        .{ .src = "norm1.weight", .dst = "prenorm.scale", .flatten = false },
        .{ .src = "norm2.weight", .dst = "postnorm.scale", .flatten = false },
        // scale_shift_table is stored [6, hidden] in diffusers but flattened to
        // [6*hidden] under the native name.
        .{ .src = "scale_shift_table", .dst = "mod.lin", .flatten = true },
    };
    for (table) |e| {
        if (std.mem.eql(u8, sub, e.src)) return .{ .suffix = e.dst, .flatten = e.flatten };
    }
    return null;
}

fn flattenDims(dims: []const usize, arena_alloc: std.mem.Allocator) ![]usize {
    var total: usize = 1;
    for (dims) |d| total *= d;
    const out = try arena_alloc.alloc(usize, 1);
    out[0] = total;
    return out;
}

/// Map a Krea2 diffusers transformer tensor name to the native single-file
/// (ComfyUI) naming, reshaping block scale_shift_table tensors to 1-D.
/// All returned slices are arena-allocated.  Unrecognised names pass through
/// unchanged (with a warning) so a partial model is still produced.
fn krea2Rename(name: []const u8, dims: []const usize, arena_alloc: std.mem.Allocator) !Krea2Renamed {
    // Block-structured groups: "<src>{idx}.<sub>" -> "<dst>{idx}.<mapped-sub>"
    const block_groups = [_]struct { src: []const u8, dst: []const u8 }{
        .{ .src = "transformer_blocks.", .dst = "blocks." },
        .{ .src = "text_fusion.layerwise_blocks.", .dst = "txtfusion.layerwise_blocks." },
        .{ .src = "text_fusion.refiner_blocks.", .dst = "txtfusion.refiner_blocks." },
    };
    for (block_groups) |g| {
        if (!std.mem.startsWith(u8, name, g.src)) continue;
        const rest = name[g.src.len..]; // "{idx}.{sub}"
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse break;
        const idx = rest[0..dot];
        const sub = rest[dot + 1 ..];
        if (krea2BlockSub(sub)) |mapped| {
            return .{
                .name = try std.fmt.allocPrint(arena_alloc, "{s}{s}.{s}", .{ g.dst, idx, mapped.suffix }),
                .dims = if (mapped.flatten) try flattenDims(dims, arena_alloc) else try arena_alloc.dupe(usize, dims),
            };
        }
        std.log.warn("krea2Rename: unmapped block sub-key '{s}' in '{s}'", .{ sub, name });
        return .{
            .name = try std.fmt.allocPrint(arena_alloc, "{s}{s}.{s}", .{ g.dst, idx, sub }),
            .dims = try arena_alloc.dupe(usize, dims),
        };
    }

    // Singletons (exact matches).
    const singletons = [_]struct { src: []const u8, dst: []const u8 }{
        .{ .src = "img_in.weight", .dst = "first.weight" },
        .{ .src = "img_in.bias", .dst = "first.bias" },
        .{ .src = "txt_in.norm.weight", .dst = "txtmlp.0.scale" },
        .{ .src = "txt_in.linear_1.weight", .dst = "txtmlp.1.weight" },
        .{ .src = "txt_in.linear_1.bias", .dst = "txtmlp.1.bias" },
        .{ .src = "txt_in.linear_2.weight", .dst = "txtmlp.3.weight" },
        .{ .src = "txt_in.linear_2.bias", .dst = "txtmlp.3.bias" },
        .{ .src = "time_embed.linear_1.weight", .dst = "tmlp.0.weight" },
        .{ .src = "time_embed.linear_1.bias", .dst = "tmlp.0.bias" },
        .{ .src = "time_embed.linear_2.weight", .dst = "tmlp.2.weight" },
        .{ .src = "time_embed.linear_2.bias", .dst = "tmlp.2.bias" },
        .{ .src = "time_mod_proj.weight", .dst = "tproj.1.weight" },
        .{ .src = "time_mod_proj.bias", .dst = "tproj.1.bias" },
        .{ .src = "text_fusion.projector.weight", .dst = "txtfusion.projector.weight" },
        .{ .src = "final_layer.linear.weight", .dst = "last.linear.weight" },
        .{ .src = "final_layer.linear.bias", .dst = "last.linear.bias" },
        .{ .src = "final_layer.norm.weight", .dst = "last.norm.scale" },
        // final_layer.scale_shift_table keeps its 2-D shape (unlike block tables).
        .{ .src = "final_layer.scale_shift_table", .dst = "last.modulation.lin" },
    };
    for (singletons) |s| {
        if (std.mem.eql(u8, name, s.src)) {
            return .{ .name = try arena_alloc.dupe(u8, s.dst), .dims = try arena_alloc.dupe(usize, dims) };
        }
    }

    std.log.warn("krea2Rename: unmapped tensor '{s}' passed through unchanged", .{name});
    return .{ .name = try arena_alloc.dupe(u8, name), .dims = try arena_alloc.dupe(usize, dims) };
}

// ============================================================================
// loadMergedSource
// ============================================================================

const Pipeline = enum { zimage, krea2, generic };

fn detectPipeline(pipeline_class: []const u8) Pipeline {
    if (std.mem.eql(u8, pipeline_class, "ZImagePipeline")) return .zimage;
    if (std.mem.eql(u8, pipeline_class, "Krea2Pipeline")) return .krea2;
    return .generic;
}

const PendingQkv = struct {
    q: ?types.Tensor = null,
    k: ?types.Tensor = null,
    v: ?types.Tensor = null,
};

/// Load a component directory, apply architecture-specific transforms, and
/// return a MergedSource ready to pass to conv.convert().
pub fn loadMergedSource(
    component_dir: []const u8,
    pipeline_class: []const u8,
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !MergedSource {
    // Load all shard headers via the existing Safetensors logic.
    var st_source = try Safetensor.init(component_dir, io, allocator, arena_alloc, false, false);
    defer st_source.deinit();

    const pipeline = detectPipeline(pipeline_class);
    if (pipeline == .generic) {
        std.log.warn(
            "No tensor-renaming rules for pipeline '{s}'; tensors will be written with their original diffusers names.",
            .{pipeline_class},
        );
    }

    var out_tensors: std.ArrayList(types.Tensor) = .empty;
    var qkv_fusions: std.ArrayList(ScaledQuant.QkvFusionCluster) = .empty;

    // Pending QKV components, keyed by attention prefix.
    // Hash map internals use the regular allocator; keys and tensor data go to arena.
    var pending_qkv = std.StringHashMap(PendingQkv).init(allocator);
    defer pending_qkv.deinit();

    for (st_source.tensors.items) |raw_t| {
        switch (pipeline) {
            .zimage => {
                const renamed = try zImageRename(raw_t.name, arena_alloc);
                if (renamed == null) {
                    // QKV component — store for later fusion
                    const m = matchQkv(raw_t.name).?;
                    const prefix_key = try arena_alloc.dupe(u8, m.prefix);
                    const entry = try pending_qkv.getOrPut(prefix_key);
                    if (!entry.found_existing) entry.value_ptr.* = .{};
                    const t_copy = types.Tensor{
                        .name = try arena_alloc.dupe(u8, raw_t.name),
                        .type = try arena_alloc.dupe(u8, raw_t.type),
                        .dims = try arena_alloc.dupe(usize, raw_t.dims),
                        .size = raw_t.size,
                        .offset = raw_t.offset,
                        .source_path = if (raw_t.source_path) |sp| try arena_alloc.dupe(u8, sp) else null,
                    };
                    switch (m.component) {
                        .q => entry.value_ptr.q = t_copy,
                        .k => entry.value_ptr.k = t_copy,
                        .v => entry.value_ptr.v = t_copy,
                    }
                } else {
                    // Normal (possibly renamed) tensor — preserve source_path and offset
                    var out_t = raw_t;
                    out_t.name = renamed.?;
                    if (raw_t.source_path) |sp| {
                        out_t.source_path = try arena_alloc.dupe(u8, sp);
                    }
                    try out_tensors.append(arena_alloc, out_t);
                }
            },
            .krea2 => {
                // Krea2 keeps Q/K/V separate (no fusion): rename only, with a
                // reshape for the block scale_shift_table tensors.
                const r = try krea2Rename(raw_t.name, raw_t.dims, arena_alloc);
                var out_t = raw_t;
                out_t.name = r.name;
                out_t.dims = r.dims;
                if (raw_t.source_path) |sp| {
                    out_t.source_path = try arena_alloc.dupe(u8, sp);
                }
                try out_tensors.append(arena_alloc, out_t);
            },
            .generic => {
                // No arch-specific transforms — pass through as-is
                try out_tensors.append(arena_alloc, raw_t);
            },
        }
    }

    // Build QKV fusion clusters and insert placeholder tensors.
    // Collect and sort keys for deterministic output.
    var sorted_keys: std.ArrayList([]const u8) = .empty;
    defer sorted_keys.deinit(allocator);
    var kit = pending_qkv.keyIterator();
    while (kit.next()) |k| try sorted_keys.append(allocator, k.*);
    std.sort.block([]const u8, sorted_keys.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);

    for (sorted_keys.items) |prefix| {
        const pqkv = pending_qkv.get(prefix).?;
        const q = pqkv.q orelse return error.MissingQkvQ;
        const k = pqkv.k orelse return error.MissingQkvK;
        const v = pqkv.v orelse return error.MissingQkvV;

        const output_name = try std.fmt.allocPrint(arena_alloc, "{s}.attention.qkv.weight", .{prefix});

        // Fused tensor has 3× the first dimension of Q.
        var fused_dims = try arena_alloc.dupe(usize, q.dims);
        fused_dims[0] = q.dims[0] * 3;

        const placeholder = types.Tensor{
            .name = output_name,
            .type = try arena_alloc.dupe(u8, q.type),
            .dims = fused_dims,
            .size = q.size + k.size + v.size,
            .offset = 0,
            .source_path = null, // virtual — resolved via QkvFusionCluster
        };
        try out_tensors.append(arena_alloc, placeholder);

        try qkv_fusions.append(arena_alloc, .{
            .output_name = output_name,
            .q_tensor = q,
            .k_tensor = k,
            .v_tensor = v,
        });
    }

    return MergedSource{
        .io = io,
        .allocator = allocator,
        .tensors = out_tensors,
        .qkv_fusions = qkv_fusions,
    };
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "matchQkv: identifies q/k/v components and strips the suffix" {
    const mq = matchQkv("blocks.0.attention.to_q.weight").?;
    try testing.expectEqual(QkvComponent.q, mq.component);
    try testing.expectEqualStrings("blocks.0", mq.prefix);

    try testing.expectEqual(QkvComponent.k, matchQkv("a.b.attention.to_k.weight").?.component);
    try testing.expectEqual(QkvComponent.v, matchQkv("a.b.attention.to_v.weight").?.component);

    // to_out / norms / unrelated names are not QKV components
    try testing.expect(matchQkv("blocks.0.attention.to_out.0.weight") == null);
    try testing.expect(matchQkv("blocks.0.attention.norm_q.weight") == null);
    try testing.expect(matchQkv("img_in.weight") == null);
    // bias variants are not matched (only ".weight" suffixes)
    try testing.expect(matchQkv("blocks.0.attention.to_q.bias") == null);
}

test "zImageRename: top-level prefix renames" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try testing.expectEqualStrings(
        "final_layer.linear.weight",
        (try zImageRename("all_final_layer.2-1.linear.weight", a)).?,
    );
    try testing.expectEqualStrings(
        "x_embedder.proj.weight",
        (try zImageRename("all_x_embedder.2-1.proj.weight", a)).?,
    );
}

test "zImageRename: attention sub-key renames" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try testing.expectEqualStrings(
        "blocks.0.attention.k_norm.weight",
        (try zImageRename("blocks.0.attention.norm_k.weight", a)).?,
    );
    try testing.expectEqualStrings(
        "blocks.0.attention.q_norm.weight",
        (try zImageRename("blocks.0.attention.norm_q.weight", a)).?,
    );
    try testing.expectEqualStrings(
        "blocks.0.attention.out.weight",
        (try zImageRename("blocks.0.attention.to_out.0.weight", a)).?,
    );
}

test "zImageRename: QKV components signal null, others pass through" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // QKV weights are deferred to a fusion cluster, signalled with null
    try testing.expect((try zImageRename("blocks.0.attention.to_q.weight", a)) == null);
    try testing.expect((try zImageRename("blocks.0.attention.to_k.weight", a)) == null);
    try testing.expect((try zImageRename("blocks.0.attention.to_v.weight", a)) == null);

    // Unrecognised names are passed through unchanged
    try testing.expectEqualStrings("img_in.weight", (try zImageRename("img_in.weight", a)).?);
    try testing.expectEqualStrings("blocks.0.mlp.fc1.weight", (try zImageRename("blocks.0.mlp.fc1.weight", a)).?);
}

test "defaultPrefix: component name mapping" {
    try testing.expectEqualStrings("model.diffusion_model.", defaultPrefix("transformer"));
    try testing.expectEqualStrings("model.diffusion_model.", defaultPrefix("unet"));
    try testing.expectEqualStrings("first_stage_model.", defaultPrefix("vae"));
    try testing.expectEqualStrings("cond_stage_model.", defaultPrefix("text_encoder"));
    try testing.expectEqualStrings("", defaultPrefix("scheduler"));
}

test "parseModelIndex + getDiffusionComponent: parse a diffusers model_index.json" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    // Mirrors the structure of a real HF diffusers repo (e.g. Krea2/Z-Image).
    const json =
        \\{
        \\  "_class_name": "ZImagePipeline",
        \\  "_diffusers_version": "0.39.0.dev0",
        \\  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        \\  "transformer": ["diffusers", "ZImageTransformer2DModel"],
        \\  "vae": ["diffusers", "AutoencoderKL"],
        \\  "patch_size": 2
        \\}
    ;
    {
        const f = try tmp.dir.createFile(io, "model_index.json", .{});
        defer f.close(io);
        try f.writePositionalAll(io, json, 0);
    }

    var path_buf: [256]u8 = undefined;
    const dir_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});

    var repo = try parseModelIndex(dir_path, io, testing.allocator);
    defer repo.deinit();

    try testing.expectEqualStrings("ZImagePipeline", repo.pipeline_class);
    // Keys starting with "_" and scalar values (patch_size) are excluded;
    // scheduler/transformer/vae remain.
    try testing.expectEqual(@as(usize, 3), repo.components.len);

    const comp = getDiffusionComponent(&repo).?;
    try testing.expectEqualStrings("transformer", comp.name);
    try testing.expectEqualStrings("ZImageTransformer2DModel", comp.class_name);
}

test "parseModelIndex: missing file returns ModelIndexNotFound" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var path_buf: [256]u8 = undefined;
    const dir_path = try std.fmt.bufPrint(&path_buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});

    try testing.expectError(error.ModelIndexNotFound, parseModelIndex(dir_path, io, testing.allocator));
}

test "detectPipeline: maps class names to handlers" {
    try testing.expectEqual(Pipeline.zimage, detectPipeline("ZImagePipeline"));
    try testing.expectEqual(Pipeline.krea2, detectPipeline("Krea2Pipeline"));
    try testing.expectEqual(Pipeline.generic, detectPipeline("FluxPipeline"));
}

// Ground-truth pairs below were derived by content-hashing every tensor in the
// distributed Krea2 `transformer/` shards against the native `turbo.safetensors`
// single-file model (all 430 matched 1:1, zero ambiguity).
fn expectKrea2(
    a: std.mem.Allocator,
    src: []const u8,
    src_dims: []const usize,
    exp_name: []const u8,
    exp_dims: []const usize,
) !void {
    const r = try krea2Rename(src, src_dims, a);
    try testing.expectEqualStrings(exp_name, r.name);
    try testing.expectEqualSlices(usize, exp_dims, r.dims);
}

test "krea2Rename: block sub-keys across all three block groups" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const d2 = [_]usize{ 6144, 6144 };
    // transformer_blocks
    try expectKrea2(a, "transformer_blocks.0.attn.to_q.weight", &d2, "blocks.0.attn.wq.weight", &d2);
    try expectKrea2(a, "transformer_blocks.0.attn.to_k.weight", &d2, "blocks.0.attn.wk.weight", &d2);
    try expectKrea2(a, "transformer_blocks.0.attn.to_v.weight", &d2, "blocks.0.attn.wv.weight", &d2);
    try expectKrea2(a, "transformer_blocks.0.attn.to_gate.weight", &d2, "blocks.0.attn.gate.weight", &d2);
    try expectKrea2(a, "transformer_blocks.0.attn.to_out.0.weight", &d2, "blocks.0.attn.wo.weight", &d2);
    try expectKrea2(a, "transformer_blocks.27.attn.norm_k.weight", &.{128}, "blocks.27.attn.qknorm.knorm.scale", &.{128});
    try expectKrea2(a, "transformer_blocks.27.attn.norm_q.weight", &.{128}, "blocks.27.attn.qknorm.qnorm.scale", &.{128});
    try expectKrea2(a, "transformer_blocks.3.ff.down.weight", &d2, "blocks.3.mlp.down.weight", &d2);
    try expectKrea2(a, "transformer_blocks.3.ff.gate.weight", &d2, "blocks.3.mlp.gate.weight", &d2);
    try expectKrea2(a, "transformer_blocks.3.ff.up.weight", &d2, "blocks.3.mlp.up.weight", &d2);
    try expectKrea2(a, "transformer_blocks.0.norm1.weight", &.{6144}, "blocks.0.prenorm.scale", &.{6144});
    try expectKrea2(a, "transformer_blocks.0.norm2.weight", &.{6144}, "blocks.0.postnorm.scale", &.{6144});

    // text_fusion sub-groups reuse the same sub-key rules
    try expectKrea2(a, "text_fusion.layerwise_blocks.1.ff.up.weight", &d2, "txtfusion.layerwise_blocks.1.mlp.up.weight", &d2);
    try expectKrea2(a, "text_fusion.refiner_blocks.0.attn.to_v.weight", &d2, "txtfusion.refiner_blocks.0.attn.wv.weight", &d2);
    try expectKrea2(a, "text_fusion.layerwise_blocks.0.norm1.weight", &.{2560}, "txtfusion.layerwise_blocks.0.prenorm.scale", &.{2560});
}

test "krea2Rename: scale_shift_table flattens only inside blocks" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Block scale_shift_table [6, 6144] -> [36864]
    try expectKrea2(a, "transformer_blocks.0.scale_shift_table", &.{ 6, 6144 }, "blocks.0.mod.lin", &.{36864});
    // final_layer.scale_shift_table keeps its 2-D shape
    try expectKrea2(a, "final_layer.scale_shift_table", &.{ 2, 6144 }, "last.modulation.lin", &.{ 2, 6144 });
}

test "krea2Rename: singletons with inserted indices" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    try expectKrea2(a, "img_in.weight", &.{ 6144, 64 }, "first.weight", &.{ 6144, 64 });
    try expectKrea2(a, "img_in.bias", &.{6144}, "first.bias", &.{6144});
    try expectKrea2(a, "txt_in.norm.weight", &.{2560}, "txtmlp.0.scale", &.{2560});
    try expectKrea2(a, "txt_in.linear_1.weight", &.{ 6144, 2560 }, "txtmlp.1.weight", &.{ 6144, 2560 });
    try expectKrea2(a, "txt_in.linear_2.bias", &.{6144}, "txtmlp.3.bias", &.{6144});
    try expectKrea2(a, "time_embed.linear_1.weight", &.{ 6144, 256 }, "tmlp.0.weight", &.{ 6144, 256 });
    try expectKrea2(a, "time_embed.linear_2.weight", &.{ 6144, 6144 }, "tmlp.2.weight", &.{ 6144, 6144 });
    try expectKrea2(a, "time_mod_proj.weight", &.{ 36864, 6144 }, "tproj.1.weight", &.{ 36864, 6144 });
    try expectKrea2(a, "text_fusion.projector.weight", &.{ 1, 12 }, "txtfusion.projector.weight", &.{ 1, 12 });
    try expectKrea2(a, "final_layer.norm.weight", &.{6144}, "last.norm.scale", &.{6144});
    try expectKrea2(a, "final_layer.linear.weight", &.{ 64, 6144 }, "last.linear.weight", &.{ 64, 6144 });
}

test "krea2Rename: unknown tensor passes through unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectKrea2(a, "some.unexpected.tensor", &.{4}, "some.unexpected.tensor", &.{4});
}
