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
// loadMergedSource
// ============================================================================

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

    const is_zimage = std.mem.eql(u8, pipeline_class, "ZImagePipeline");

    var out_tensors: std.ArrayList(types.Tensor) = .empty;
    var qkv_fusions: std.ArrayList(ScaledQuant.QkvFusionCluster) = .empty;

    // Pending QKV components, keyed by attention prefix.
    // Hash map internals use the regular allocator; keys and tensor data go to arena.
    var pending_qkv = std.StringHashMap(PendingQkv).init(allocator);
    defer pending_qkv.deinit();

    for (st_source.tensors.items) |raw_t| {
        if (is_zimage) {
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
        } else {
            // No arch-specific transforms — pass through as-is
            try out_tensors.append(arena_alloc, raw_t);
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
