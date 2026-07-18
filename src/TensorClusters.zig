const std = @import("std");
const types = @import("types.zig");
const DataTransform = @import("DataTransform.zig");
const thread_pool_mod = @import("ThreadPool.zig");

pub const ComfyQuantScheme = enum { nvfp4, float8_e4m3fn, mxfp4, mxfp8_e4m3fn, int8_convrot, convrot_w4a4, unknown };

pub const Fp4Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,
    weight_scale: types.Tensor,
    weight_scale_2: types.Tensor,
    // Per-tensor `.comfy_quant` marker, or null when the identity came from the file-level
    // `_quantization_metadata` header (comfy-kitchen layout, no per-tensor markers).
    comfy_quant: ?types.Tensor,
    rows: usize,
    cols: usize,
};

pub const Float8Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,
    weight_scale: types.Tensor,
    input_scale: ?types.Tensor,  // optional activation scale (scalar, dropped on conversion)
    // Per-tensor `.comfy_quant` marker, or null when sourced from the `_quantization_metadata` header.
    comfy_quant: ?types.Tensor,
    rows: usize,
    cols: usize,
};

/// MXFP4 cluster (OCP MX spec): E2M1 packed nibbles + E8M0 per-block scale, block size 32.
/// Scale layout is linear row-major, distinct from NVFP4's cuBLAS-tiled layout.
pub const Mxfp4Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // U8 packed nibbles, [rows, cols/2]
    weight_scale: types.Tensor,  // U8 E8M0, [rows, cols/32]
    // Per-tensor `.comfy_quant` marker, or null when sourced from the `_quantization_metadata` header.
    comfy_quant: ?types.Tensor,
    rows: usize,
    cols: usize,
};

/// MXFP8 cluster (OCP MX spec): E4M3 elements + E8M0 per-block scale, block size 32.
pub const Mxfp8Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // F8_E4M3, [rows, cols]
    weight_scale: types.Tensor,  // U8 E8M0, [rows, cols/32]
    // Per-tensor `.comfy_quant` marker, or null when sourced from the `_quantization_metadata` header.
    comfy_quant: ?types.Tensor,
    rows: usize,
    cols: usize,
};

/// ConvRot INT8 cluster (ComfyUI "int8_tensorwise" with convrot + per_row):
/// I8 weight rotated by a group-wise Hadamard, F32 per-row scale.
pub const Int8ConvrotCluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // I8, [rows, cols]
    weight_scale: types.Tensor,  // F32, [rows, 1]
    // The per-tensor `.comfy_quant` marker, or null when the quant identity was sourced
    // from the file-level `_quantization_metadata` header (comfy-kitchen layout).
    comfy_quant: ?types.Tensor,
    rows: usize,
    cols: usize,
    convrot: bool,
    group_size: usize,
};

/// int4 cluster (ComfyUI "convrot_w4a4", always Hadamard-rotated):
/// nibble-packed signed 4-bit weight, F32 per-row scale.
pub const Int4Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // I8, [rows, cols/2] (nibble-packed, two signed 4-bit values per byte)
    weight_scale: types.Tensor,  // F32, [rows]
    // Per-tensor `.comfy_quant` marker, or null when sourced from the `_quantization_metadata` header.
    comfy_quant: ?types.Tensor,
    rows: usize,
    cols: usize,
    convrot: bool,
    group_size: usize,
};

pub const GroupResult = struct {
    fp4_clusters: []Fp4Cluster,
    float8_clusters: []Float8Cluster,
    mxfp4_clusters: []Mxfp4Cluster,
    mxfp8_clusters: []Mxfp8Cluster,
    int8_convrot_clusters: []Int8ConvrotCluster,
    int4_clusters: []Int4Cluster,
};

/// Parse the JSON payload of a comfy_quant blob to identify the quantization scheme.
/// Reads the "format" key; returns .unknown on any parse error or unrecognised value.
pub fn parseComfyQuantScheme(data: []const u8) ComfyQuantScheme {
    var fba_buf: [16384]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), data, .{}) catch return .unknown;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return .unknown,
    };
    const fmt_val = obj.get("format") orelse return .unknown;
    const fmt_str = switch (fmt_val) {
        .string => |s| s,
        else => return .unknown,
    };
    if (std.mem.eql(u8, fmt_str, "nvfp4")) return .nvfp4;
    if (std.mem.eql(u8, fmt_str, "mxfp8_e4m3fn") or std.mem.eql(u8, fmt_str, "mxfp8")) return .mxfp8_e4m3fn;
    if (std.mem.eql(u8, fmt_str, "mxfp4")) return .mxfp4;
    if (std.mem.eql(u8, fmt_str, "float8_e4m3fn")) return .float8_e4m3fn;
    // ConvRot ships under the "int8_tensorwise" layout, distinguished by a `convrot`
    // boolean and per-row scaling rather than a dedicated format string.
    if (std.mem.eql(u8, fmt_str, "int8_tensorwise")) return .int8_convrot;
    // ComfyUI's 4-bit ConvRot layout; the format name itself implies group-wise
    // Hadamard rotation (there is no separate `convrot` boolean).
    if (std.mem.eql(u8, fmt_str, "convrot_w4a4")) return .convrot_w4a4;
    return .unknown;
}

/// Read the boolean `key` from a comfy_quant JSON blob; returns `default` on any miss.
fn comfyQuantBool(data: []const u8, key: []const u8, default: bool) bool {
    var fba_buf: [16384]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), data, .{}) catch return default;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return default,
    };
    return switch (obj.get(key) orelse return default) {
        .bool => |b| b,
        else => default,
    };
}

/// Read the integer `key` from a comfy_quant JSON blob; returns `default` on any miss.
fn comfyQuantInt(data: []const u8, key: []const u8, default: usize) usize {
    var fba_buf: [16384]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), data, .{}) catch return default;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return default,
    };
    return switch (obj.get(key) orelse return default) {
        .integer => |n| if (n > 0) @intCast(n) else default,
        else => default,
    };
}

/// Returns true if `stripped` matches `full_name` exactly or as a dot-separated suffix.
pub fn nameSuffixMatch(full_name: []const u8, stripped: []const u8) bool {
    if (std.mem.eql(u8, full_name, stripped)) return true;
    return full_name.len > stripped.len and
        full_name[full_name.len - stripped.len - 1] == '.' and
        std.mem.endsWith(u8, full_name, stripped);
}

/// Read a source tensor's raw bytes into a freshly-allocated buffer (caller owns).
fn readTensorBytes(source: anytype, tensor: types.Tensor, allocator: std.mem.Allocator) ![]u8 {
    const file = try source.openFileForTensor(tensor.name);
    const buf = try allocator.alloc(u8, tensor.size);
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(source.io, buf, tensor.offset + source.current_data_begin);
    return buf;
}

/// Build an int8_convrot cluster from a layer `base_name` and its (already-parsed) quant
/// identity, appending to `list`. `comfy_quant` is the per-tensor marker tensor when the
/// identity came from a `.comfy_quant` blob, or null when it came from the file-level
/// `_quantization_metadata` header. Returns `false` without appending (logging a warning)
/// on any structural mismatch, `true` when a cluster was appended. `base_name` must
/// outlive the cluster (arena- or source-owned).
fn appendInt8ConvrotCluster(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    base_name: []const u8,
    convrot: bool,
    group_size: usize,
    comfy_quant: ?types.Tensor,
    list: *std.ArrayList(Int8ConvrotCluster),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !bool {
    const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
    defer allocator.free(wname);
    const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
    defer allocator.free(wsname);

    const wi = name_map.get(wname) orelse {
        std.log.warn("TensorClusters: missing .weight for int8_convrot cluster {s}", .{base_name});
        return false;
    };
    const wsi = name_map.get(wsname) orelse {
        std.log.warn("TensorClusters: missing .weight_scale for int8_convrot cluster {s}", .{base_name});
        return false;
    };

    const weight = source.tensors.items[wi];
    const weight_scale = source.tensors.items[wsi];

    if (weight.dims.len != 2) {
        std.log.warn("TensorClusters: int8_convrot cluster {s} weight is not 2-D; skipping", .{base_name});
        return false;
    }
    const rows = weight.dims[0];
    const cols = weight.dims[1];
    // Per-row (a.k.a. per-channel) scaling only: expect one F32 scale per output row.
    if (weight_scale.size != rows * 4) {
        std.log.warn("TensorClusters: int8_convrot cluster {s} has non-per-row scale ({} bytes, expected {}); skipping", .{ base_name, weight_scale.size, rows * 4 });
        return false;
    }

    if (convrot and (!DataTransform.Quantizer.isValidHadamardSize(group_size) or cols % group_size != 0)) {
        std.log.warn("TensorClusters: int8_convrot cluster {s} has incompatible group_size {} for cols {}; skipping", .{ base_name, group_size, cols });
        return false;
    }

    try list.append(arena_alloc, .{
        .base_name = base_name,
        .weight = weight,
        .weight_scale = weight_scale,
        .comfy_quant = comfy_quant,
        .rows = rows,
        .cols = cols,
        .convrot = convrot,
        .group_size = group_size,
    });
    std.log.debug("TensorClusters: grouped int8_convrot cluster {s} [{}, {}] convrot={} gs={}", .{
        base_name, rows, cols, convrot, group_size,
    });
    return true;
}

/// Build an int4 cluster (ComfyUI "convrot_w4a4") from a layer `base_name`. The weight is
/// stored nibble-packed with dims [rows, cols/2], so the logical column count is twice
/// the packed second dim. See `appendInt8ConvrotCluster` for the argument conventions.
fn appendInt4Cluster(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    base_name: []const u8,
    convrot: bool,
    group_size: usize,
    comfy_quant: ?types.Tensor,
    list: *std.ArrayList(Int4Cluster),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !bool {
    const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
    defer allocator.free(wname);
    const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
    defer allocator.free(wsname);

    const wi = name_map.get(wname) orelse {
        std.log.warn("TensorClusters: missing .weight for int4 cluster {s}", .{base_name});
        return false;
    };
    const wsi = name_map.get(wsname) orelse {
        std.log.warn("TensorClusters: missing .weight_scale for int4 cluster {s}", .{base_name});
        return false;
    };

    const weight = source.tensors.items[wi];
    const weight_scale = source.tensors.items[wsi];

    if (weight.dims.len != 2) {
        std.log.warn("TensorClusters: int4 cluster {s} weight is not 2-D; skipping", .{base_name});
        return false;
    }
    const rows = weight.dims[0];
    const cols = weight.dims[1] * 2; // nibble-packed: two logical columns per stored byte
    // Per-row (a.k.a. per-channel) scaling only: expect one F32 scale per output row.
    if (weight_scale.size != rows * 4) {
        std.log.warn("TensorClusters: int4 cluster {s} has non-per-row scale ({} bytes, expected {}); skipping", .{ base_name, weight_scale.size, rows * 4 });
        return false;
    }

    if (convrot and (!DataTransform.Quantizer.isValidHadamardSize(group_size) or cols % group_size != 0)) {
        std.log.warn("TensorClusters: int4 cluster {s} has incompatible group_size {} for cols {}; skipping", .{ base_name, group_size, cols });
        return false;
    }

    try list.append(arena_alloc, .{
        .base_name = base_name,
        .weight = weight,
        .weight_scale = weight_scale,
        .comfy_quant = comfy_quant,
        .rows = rows,
        .cols = cols,
        .convrot = convrot,
        .group_size = group_size,
    });
    std.log.debug("TensorClusters: grouped int4 cluster {s} [{}, {}] convrot={} gs={}", .{
        base_name, rows, cols, convrot, group_size,
    });
    return true;
}

/// Build an NVFP4 cluster (E2M1 packed nibbles + FP8 block scale + FP32 global scale) from a
/// layer `base_name` and append to `list`. The `.weight`, `.weight_scale` and `.weight_scale_2`
/// companions are resolved by name; a missing one is warned and skipped. `comfy_quant` is the
/// per-tensor marker, or null when the identity came from the `_quantization_metadata` header.
/// Returns true if a cluster was appended, false on any structural miss. `base_name` must outlive
/// the cluster (arena- or source-owned).
fn appendNvfp4Cluster(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    base_name: []const u8,
    comfy_quant: ?types.Tensor,
    list: *std.ArrayList(Fp4Cluster),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !bool {
    const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
    defer allocator.free(wname);
    const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
    defer allocator.free(wsname);
    const ws2name = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{base_name});
    defer allocator.free(ws2name);

    const wi = name_map.get(wname) orelse {
        std.log.warn("TensorClusters: missing .weight for cluster {s}", .{base_name});
        return false;
    };
    const wsi = name_map.get(wsname) orelse {
        std.log.warn("TensorClusters: missing .weight_scale for cluster {s}", .{base_name});
        return false;
    };
    const ws2i = name_map.get(ws2name) orelse {
        std.log.warn("TensorClusters: missing .weight_scale_2 for cluster {s}", .{base_name});
        return false;
    };

    const weight = source.tensors.items[wi];
    const weight_scale = source.tensors.items[wsi];
    const weight_scale_2 = source.tensors.items[ws2i];

    try list.append(arena_alloc, .{
        .base_name = base_name,
        .weight = weight,
        .weight_scale = weight_scale,
        .weight_scale_2 = weight_scale_2,
        .comfy_quant = comfy_quant,
        .rows = weight.dims[0],
        .cols = weight.dims[1] * 2,
    });
    std.log.debug("TensorClusters: grouped nvfp4 cluster {s} [{}, {}]", .{
        base_name, weight.dims[0], weight.dims[1] * 2,
    });
    return true;
}

/// Build an FP8 (float8_e4m3fn) cluster from a layer `base_name` and append to `list`. The
/// `.weight` and `.weight_scale` companions are required; `.input_scale` (an activation scale)
/// is optional and dropped on conversion. A missing required companion is warned and skipped.
/// `comfy_quant` is the per-tensor marker, or null when sourced from the header. Returns true
/// if a cluster was appended.
fn appendFloat8Cluster(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    base_name: []const u8,
    comfy_quant: ?types.Tensor,
    list: *std.ArrayList(Float8Cluster),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !bool {
    const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
    defer allocator.free(wname);
    const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
    defer allocator.free(wsname);
    const isname = try std.fmt.allocPrint(allocator, "{s}.input_scale", .{base_name});
    defer allocator.free(isname);

    const wi = name_map.get(wname) orelse {
        std.log.warn("TensorClusters: missing .weight for fp8 cluster {s}", .{base_name});
        return false;
    };
    const wsi = name_map.get(wsname) orelse {
        std.log.warn("TensorClusters: missing .weight_scale for fp8 cluster {s}", .{base_name});
        return false;
    };

    const weight = source.tensors.items[wi];
    const weight_scale = source.tensors.items[wsi];
    const input_scale: ?types.Tensor = if (name_map.get(isname)) |isi|
        source.tensors.items[isi]
    else
        null;

    try list.append(arena_alloc, .{
        .base_name = base_name,
        .weight = weight,
        .weight_scale = weight_scale,
        .input_scale = input_scale,
        .comfy_quant = comfy_quant,
        .rows = weight.dims[0],
        .cols = weight.dims[1],
    });
    std.log.debug("TensorClusters: grouped fp8 cluster {s} [{}, {}]", .{
        base_name, weight.dims[0], weight.dims[1],
    });
    return true;
}

/// Build an MXFP4 cluster (E2M1 packed nibbles + E8M0 per-block scale) from a layer `base_name`
/// and append to `list`. The `.weight` and `.weight_scale` companions are resolved by name; a
/// missing one is warned and skipped. `comfy_quant` is the per-tensor marker, or null when sourced
/// from the header. Returns true if a cluster was appended.
fn appendMxfp4Cluster(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    base_name: []const u8,
    comfy_quant: ?types.Tensor,
    list: *std.ArrayList(Mxfp4Cluster),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !bool {
    const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
    defer allocator.free(wname);
    const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
    defer allocator.free(wsname);

    const wi = name_map.get(wname) orelse {
        std.log.warn("TensorClusters: missing .weight for mxfp4 cluster {s}", .{base_name});
        return false;
    };
    const wsi = name_map.get(wsname) orelse {
        std.log.warn("TensorClusters: missing .weight_scale for mxfp4 cluster {s}", .{base_name});
        return false;
    };

    const weight = source.tensors.items[wi];
    const weight_scale = source.tensors.items[wsi];

    try list.append(arena_alloc, .{
        .base_name = base_name,
        .weight = weight,
        .weight_scale = weight_scale,
        .comfy_quant = comfy_quant,
        .rows = weight.dims[0],
        .cols = weight.dims[1] * 2,
    });
    std.log.debug("TensorClusters: grouped mxfp4 cluster {s} [{}, {}]", .{
        base_name, weight.dims[0], weight.dims[1] * 2,
    });
    return true;
}

/// Build an MXFP8 cluster (E4M3 elements + E8M0 per-block scale) from a layer `base_name` and
/// append to `list`. The `.weight` and `.weight_scale` companions are resolved by name; a missing
/// one is warned and skipped. `comfy_quant` is the per-tensor marker, or null when sourced from the
/// header. Returns true if a cluster was appended.
fn appendMxfp8Cluster(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    base_name: []const u8,
    comfy_quant: ?types.Tensor,
    list: *std.ArrayList(Mxfp8Cluster),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !bool {
    const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
    defer allocator.free(wname);
    const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
    defer allocator.free(wsname);

    const wi = name_map.get(wname) orelse {
        std.log.warn("TensorClusters: missing .weight for mxfp8 cluster {s}", .{base_name});
        return false;
    };
    const wsi = name_map.get(wsname) orelse {
        std.log.warn("TensorClusters: missing .weight_scale for mxfp8 cluster {s}", .{base_name});
        return false;
    };

    const weight = source.tensors.items[wi];
    const weight_scale = source.tensors.items[wsi];

    try list.append(arena_alloc, .{
        .base_name = base_name,
        .weight = weight,
        .weight_scale = weight_scale,
        .comfy_quant = comfy_quant,
        .rows = weight.dims[0],
        .cols = weight.dims[1],
    });
    std.log.debug("TensorClusters: grouped mxfp8 cluster {s} [{}, {}]", .{
        base_name, weight.dims[0], weight.dims[1],
    });
    return true;
}

/// Resolve a `_quantization_metadata` layer key to the base name of its `.weight` tensor in
/// the source. comfy-kitchen stores logical layer names that may omit the tensor-name prefix
/// (e.g. "model.diffusion_model."), so a fully-qualified key hits `name_map` directly (O(1))
/// and a prefix-less key falls back to a dot-separated suffix scan — the same way layer keys
/// are matched elsewhere. The suffix scan requires a *unique* match: a key that suffix-matches
/// more than one `.weight` tensor is ambiguous and resolves to null (warned) rather than being
/// silently bound to whichever tensor happens to come first. Returns a slice into a source
/// tensor name (stable for the whole conversion), or null if no unambiguous `<key>.weight` exists.
fn resolveClusterBase(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    layer_key: []const u8,
    allocator: std.mem.Allocator,
) !?[]const u8 {
    const wsuffix = try std.fmt.allocPrint(allocator, "{s}.weight", .{layer_key});
    defer allocator.free(wsuffix);

    // Fast path: fully-qualified keys are already exact tensor names. Return the
    // source-owned slice, not the parsed-JSON key (which does not outlive the conversion).
    if (name_map.get(wsuffix)) |wi| {
        const name = source.tensors.items[wi].name;
        return name[0 .. name.len - ".weight".len];
    }

    // Slow path (prefix-less keys only): match by dot-separated suffix, requiring uniqueness.
    var match: ?[]const u8 = null;
    for (source.tensors.items) |t| {
        if (nameSuffixMatch(t.name, wsuffix)) {
            if (match != null) {
                std.log.warn("TensorClusters: _quantization_metadata layer {s} suffix-matches multiple .weight tensors; skipping (ambiguous)", .{layer_key});
                return null;
            }
            match = t.name[0 .. t.name.len - ".weight".len];
        }
    }
    return match;
}

/// Group clusters from a file-level `_quantization_metadata` JSON header — the marker-less layout
/// ComfyUI ingests via `convert_old_quants` (comfy/utils.py). The header maps each layer base name
/// to its quant identity `{format, ...}`; ComfyUI expands each entry verbatim into a per-tensor
/// `.comfy_quant` blob, so we route every entry through the *same* per-scheme dispatch and helpers
/// used for markers, with a null `comfy_quant` (there is no marker tensor to drop). Bases already
/// grouped via markers are skipped. Unknown/unsupported schemes are warned and left ungrouped.
fn groupFromQuantMetadata(
    source: anytype,
    name_map: *std.StringHashMap(usize),
    qm_json: []const u8,
    fp4_list: *std.ArrayList(Fp4Cluster),
    float8_list: *std.ArrayList(Float8Cluster),
    mxfp4_list: *std.ArrayList(Mxfp4Cluster),
    mxfp8_list: *std.ArrayList(Mxfp8Cluster),
    int8_convrot_list: *std.ArrayList(Int8ConvrotCluster),
    int4_list: *std.ArrayList(Int4Cluster),
    seen: *std.StringHashMap(void),
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !void {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, qm_json, .{}) catch |err| {
        std.log.warn("TensorClusters: failed to parse _quantization_metadata: {}", .{err});
        return;
    };
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return,
    };
    const layers = switch (root.get("layers") orelse return) {
        .object => |o| o,
        else => return,
    };

    var it = layers.iterator();
    while (it.next()) |entry| {
        const layer_key = entry.key_ptr.*;
        if (entry.value_ptr.* != .object) continue;

        // Re-serialize the per-layer identity to JSON so we can route it through the
        // same byte-level scheme/field parsers used for `.comfy_quant` marker blobs — this is
        // exactly the payload ComfyUI writes into the expanded `.comfy_quant` tensor.
        const layer_json = try std.json.Stringify.valueAlloc(allocator, entry.value_ptr.*, .{});
        defer allocator.free(layer_json);

        const scheme = parseComfyQuantScheme(layer_json);
        if (scheme == .unknown) {
            std.log.warn("TensorClusters: _quantization_metadata layer {s} has an unsupported scheme; skipping", .{layer_key});
            continue;
        }

        // Resolve the (possibly prefix-less) layer key against the real tensor namespace;
        // `base_name` is a source-owned slice that outlives the conversion.
        const base_name = (try resolveClusterBase(source, name_map, layer_key, allocator)) orelse {
            std.log.warn("TensorClusters: _quantization_metadata layer {s} has no matching .weight tensor; skipping", .{layer_key});
            continue;
        };
        // Skip bases already grouped via a per-tensor marker (compared on the resolved full
        // name, since `seen` is keyed by full tensor base names). Header-sourced clusters carry
        // a null `comfy_quant` — there is no per-tensor marker tensor in this layout.
        if (seen.contains(base_name)) continue;

        _ = switch (scheme) {
            .nvfp4 => try appendNvfp4Cluster(source, name_map, base_name, null, fp4_list, arena_alloc, allocator),
            .float8_e4m3fn => try appendFloat8Cluster(source, name_map, base_name, null, float8_list, arena_alloc, allocator),
            .mxfp4 => try appendMxfp4Cluster(source, name_map, base_name, null, mxfp4_list, arena_alloc, allocator),
            .mxfp8_e4m3fn => try appendMxfp8Cluster(source, name_map, base_name, null, mxfp8_list, arena_alloc, allocator),
            .int8_convrot => blk: {
                const convrot = comfyQuantBool(layer_json, "convrot", false);
                const group_size = comfyQuantInt(layer_json, "convrot_groupsize", 256);
                break :blk try appendInt8ConvrotCluster(source, name_map, base_name, convrot, group_size, null, int8_convrot_list, arena_alloc, allocator);
            },
            .convrot_w4a4 => blk: {
                // The convrot_w4a4 format is always Hadamard-rotated; only the group size varies.
                const group_size = comfyQuantInt(layer_json, "convrot_groupsize", 256);
                break :blk try appendInt4Cluster(source, name_map, base_name, true, group_size, null, int4_list, arena_alloc, allocator);
            },
            .unknown => unreachable, // handled above
        };
    }
}

/// Scan source tensors and group them into NVFP4/FP8/MXFP4/MXFP8/INT8-ConvRot clusters. Two
/// encodings are supported and treated equivalently (as ComfyUI's convert_old_quants does): per-
/// tensor `.comfy_quant` marker blobs, and a single file-level `_quantization_metadata` header that
/// maps each layer to the same identity. Result slices are arena-allocated; `allocator` is used for
/// temporary work only.
pub fn groupClusters(
    source: anytype,
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !GroupResult {
    var name_map = std.StringHashMap(usize).init(allocator);
    defer name_map.deinit();
    for (source.tensors.items, 0..) |t, i| try name_map.put(t.name, i);

    var fp4_list: std.ArrayList(Fp4Cluster) = .empty;
    var float8_list: std.ArrayList(Float8Cluster) = .empty;
    var mxfp4_list: std.ArrayList(Mxfp4Cluster) = .empty;
    var mxfp8_list: std.ArrayList(Mxfp8Cluster) = .empty;
    var int8_convrot_list: std.ArrayList(Int8ConvrotCluster) = .empty;
    var int4_list: std.ArrayList(Int4Cluster) = .empty;

    // Track bases grouped via per-tensor markers so the file-level `_quantization_metadata`
    // fallback below does not double-group them. Keyed by full tensor base name, across all schemes.
    var seen_bases = std.StringHashMap(void).init(allocator);
    defer seen_bases.deinit();

    const comfy_suffix = ".comfy_quant";

    for (source.tensors.items) |t| {
        if (!std.mem.endsWith(u8, t.name, comfy_suffix)) continue;

        const tensor_file = try source.openFileForTensor(t.name);
        const data = try allocator.alloc(u8, t.size);
        defer allocator.free(data);
        _ = try tensor_file.readPositionalAll(source.io, data, t.offset + source.current_data_begin);

        const scheme = parseComfyQuantScheme(data);
        const base_name = t.name[0 .. t.name.len - comfy_suffix.len];

        // Only mark the base "seen" when it was actually grouped, so a marker that fails its
        // structural checks does not suppress the `_quantization_metadata` fallback that might
        // still be able to group the same layer.
        const grouped = switch (scheme) {
            .nvfp4 => try appendNvfp4Cluster(source, &name_map, base_name, t, &fp4_list, arena_alloc, allocator),
            .float8_e4m3fn => try appendFloat8Cluster(source, &name_map, base_name, t, &float8_list, arena_alloc, allocator),
            .mxfp4 => try appendMxfp4Cluster(source, &name_map, base_name, t, &mxfp4_list, arena_alloc, allocator),
            .mxfp8_e4m3fn => try appendMxfp8Cluster(source, &name_map, base_name, t, &mxfp8_list, arena_alloc, allocator),
            .int8_convrot => blk: {
                const convrot = comfyQuantBool(data, "convrot", false);
                const group_size = comfyQuantInt(data, "convrot_groupsize", 256);
                break :blk try appendInt8ConvrotCluster(source, &name_map, base_name, convrot, group_size, t, &int8_convrot_list, arena_alloc, allocator);
            },
            .convrot_w4a4 => blk: {
                // The convrot_w4a4 format is always Hadamard-rotated; only the group size varies.
                const group_size = comfyQuantInt(data, "convrot_groupsize", 256);
                break :blk try appendInt4Cluster(source, &name_map, base_name, true, group_size, t, &int4_list, arena_alloc, allocator);
            },
            .unknown => false,
        };
        if (grouped) try seen_bases.put(base_name, {});
    }

    // Fallback for the marker-less layout ComfyUI ingests via convert_old_quants: a single
    // file-level `_quantization_metadata` header maps each layer base name to its quant identity.
    // Group every supported scheme from it (skipping bases already handled by a marker above).
    if (source.getSourceMetadata()) |meta| {
        if (meta.get("_quantization_metadata")) |qm_val| {
            if (qm_val == .string) {
                try groupFromQuantMetadata(source, &name_map, qm_val.string, &fp4_list, &float8_list, &mxfp4_list, &mxfp8_list, &int8_convrot_list, &int4_list, &seen_bases, arena_alloc, allocator);
            }
        }
    }

    std.log.info("TensorClusters: found {} nvfp4, {} fp8, {} mxfp4, {} mxfp8, {} int8_convrot, {} int4 clusters", .{
        fp4_list.items.len, float8_list.items.len, mxfp4_list.items.len, mxfp8_list.items.len, int8_convrot_list.items.len, int4_list.items.len,
    });

    return GroupResult{
        .fp4_clusters = fp4_list.items,
        .float8_clusters = float8_list.items,
        .mxfp4_clusters = mxfp4_list.items,
        .mxfp8_clusters = mxfp8_list.items,
        .int8_convrot_clusters = int8_convrot_list.items,
        .int4_clusters = int4_list.items,
    };
}

fn processNvFp4DequantRows(
    weight_bytes: []const u8,
    scale_bytes: []const u8,
    global_scale: f32,
    cols: usize,
    n_col_blocks: usize,
    out: []f32,
    start_row: usize,
    end_row: usize,
) void {
    for (start_row..end_row) |row| {
        for (0..cols) |col| {
            const nibble: u4 = if (col % 2 == 0)
                @intCast(weight_bytes[row * (cols / 2) + col / 2] >> 4)
            else
                @intCast(weight_bytes[row * (cols / 2) + col / 2] & 0xF);
            const fp4_val = DataTransform.Quantizer.lut_fp4_e2m1[nibble];
            const scale_col = col / 16;
            const r0 = row / 128;
            const r1 = row % 128;
            const c0 = scale_col / 4;
            const c1 = scale_col % 4;
            const scale_idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 / 32) * 4 + c1;
            const block_scale = DataTransform.Quantizer.lut_e4m3[scale_bytes[scale_idx]];
            out[row * cols + col] = fp4_val * block_scale * global_scale;
        }
    }
}

/// Core NVFP4 dequantization over raw byte slices. Caller owns the returned slice.
/// weight_bytes: packed nibbles [rows × (cols/2)], even cols in HIGH nibble, odd in LOW.
/// scale_bytes: F8_E4M3 values in cuBLAS tiled order.
/// global_scale: scalar multiplier applied to every element.
pub fn dequantizeFp4Raw(
    weight_bytes: []const u8,
    scale_bytes: []const u8,
    global_scale: f32,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) ![]f32 {
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);

    const n_col_blocks = (cols / 16 + 3) / 4;
    const threads_u64: u64 = @intCast(pool.threads.len);
    const rows_per_thread = @divTrunc(rows, threads_u64);
    const leftover = rows - (rows_per_thread * threads_u64);

    var wg: thread_pool_mod.WaitGroup = .{};
    var i: u64 = 0;
    while (i < threads_u64) : (i += 1) {
        const start = i * rows_per_thread;
        var end = start + rows_per_thread;
        if (i == threads_u64 - 1) end += leftover;
        pool.spawnWg(&wg, processNvFp4DequantRows, .{ weight_bytes, scale_bytes, global_scale, cols, n_col_blocks, out, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
    }
    wg.wait();

    return out;
}

/// Dequantize an NVFP4 cluster to a flat F32 slice of [rows * cols] elements.
/// Caller owns the returned slice.
pub fn dequantizeFp4Cluster(
    cluster: Fp4Cluster,
    source: anytype,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) ![]f32 {
    if (cluster.cols % 64 != 0) return error.InvalidClusterShape;
    if (cluster.rows % 128 != 0) return error.InvalidClusterShape;

    const weight_bytes = try readTensorBytes(source, cluster.weight, allocator);
    defer allocator.free(weight_bytes);
    const scale_bytes = try readTensorBytes(source, cluster.weight_scale, allocator);
    defer allocator.free(scale_bytes);
    const gs_bytes = try readTensorBytes(source, cluster.weight_scale_2, allocator);
    defer allocator.free(gs_bytes);
    const global_scale: f32 = @bitCast(std.mem.readInt(u32, gs_bytes[0..4], .little));

    return dequantizeFp4Raw(weight_bytes, scale_bytes, global_scale, cluster.rows, cluster.cols, allocator, pool);
}

pub const NvFp4Encoded = struct {
    weight: []u8,       // packed nibbles, rows × (cols/2); even col → HIGH nibble, odd → LOW
    scale: []u8,        // F8_E4M3 bytes in cuBLAS tiled order, rows × (cols/16)
    global_scale: f32,
};

fn processNvFp4QuantRows(
    data: []const f32,
    weight: []u8,
    scale: []u8,
    inv_global: f32,
    cols: usize,
    num_scale_cols: usize,
    n_col_blocks: usize,
    start_row: usize,
    end_row: usize,
) void {
    for (start_row..end_row) |row| {
        const r0 = row / 128;
        const r1 = row % 128;
        for (0..num_scale_cols) |sc| {
            var block_max: f32 = 0.0;
            for (data[row * cols + sc * 16 ..][0..16]) |v|
                block_max = @max(block_max, @abs(v * inv_global));
            const scale_byte: u8 = if (block_max > 0.0)
                DataTransform.Quantizer.f32_to_fp8_e4m3(block_max / 6.0)
            else
                0;
            const c0 = sc / 4;
            const c1 = sc % 4;
            const idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 / 32) * 4 + c1;
            scale[idx] = scale_byte;
        }
        for (0..cols) |col| {
            const sc = col / 16;
            const c0 = sc / 4;
            const c1 = sc % 4;
            const scale_idx = (r0 * n_col_blocks + c0) * 512 + (r1 % 32) * 16 + (r1 / 32) * 4 + c1;
            const block_scale = DataTransform.Quantizer.lut_e4m3[scale[scale_idx]];
            const nibble: u4 = if (block_scale == 0.0) 0 else
                DataTransform.Quantizer.f32_to_fp4_e2m1(data[row * cols + col] * inv_global / block_scale);
            const byte_idx = row * (cols / 2) + col / 2;
            if (col % 2 == 0) {
                weight[byte_idx] |= @as(u8, nibble) << 4;
            } else {
                weight[byte_idx] |= @as(u8, nibble);
            }
        }
    }
}

/// Quantize a flat F32 matrix [rows*cols] to NVFP4.
/// rows must be a multiple of 128; cols must be a multiple of 64.
/// Caller owns the returned slices.
pub fn quantizeToNvFp4Raw(
    data: []const f32,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) !NvFp4Encoded {
    if (cols % 64 != 0) return error.InvalidClusterShape;
    if (rows % 128 != 0) return error.InvalidClusterShape;

    const num_scale_cols = cols / 16;
    const n_col_blocks = (num_scale_cols + 3) / 4;

    // Step 1: global scale = max_abs / (fp4_max × fp8_max) — sequential reduction
    var max_abs: f32 = 0.0;
    for (data) |v| {
        if (!std.math.isNan(v) and !std.math.isInf(v))
            max_abs = @max(max_abs, @abs(v));
    }
    const global_scale: f32 = if (max_abs > 0.0) max_abs / (6.0 * 448.0) else 1.0;
    const inv_global = 1.0 / global_scale;

    const scale = try allocator.alloc(u8, rows * num_scale_cols);
    errdefer allocator.free(scale);
    const weight = try allocator.alloc(u8, rows * (cols / 2));
    errdefer allocator.free(weight);
    @memset(weight, 0);

    // Steps 2+3 combined per row chunk: compute block scales then pack nibbles
    const threads_u64: u64 = @intCast(pool.threads.len);
    const rows_per_thread = @divTrunc(rows, threads_u64);
    const leftover = rows - (rows_per_thread * threads_u64);

    var wg: thread_pool_mod.WaitGroup = .{};
    var i: u64 = 0;
    while (i < threads_u64) : (i += 1) {
        const start = i * rows_per_thread;
        var end = start + rows_per_thread;
        if (i == threads_u64 - 1) end += leftover;
        pool.spawnWg(&wg, processNvFp4QuantRows, .{ data, weight, scale, inv_global, cols, num_scale_cols, n_col_blocks, @as(usize, @intCast(start)), @as(usize, @intCast(end)) });
    }
    wg.wait();

    return .{ .weight = weight, .scale = scale, .global_scale = global_scale };
}

/// Dequantize an FP8 (ComfyUI) cluster to F32: f8_weight × scalar_scale.
/// Caller owns the returned slice.
pub fn dequantizeFloat8Cluster(
    cluster: Float8Cluster,
    source: anytype,
    allocator: std.mem.Allocator,
) ![]f32 {
    const weight_bytes = try readTensorBytes(source, cluster.weight, allocator);
    defer allocator.free(weight_bytes);
    const scale_buf = try readTensorBytes(source, cluster.weight_scale, allocator);
    defer allocator.free(scale_buf);
    const scalar_scale: f32 = @bitCast(std.mem.readInt(u32, scale_buf[0..4], .little));

    const out = try allocator.alloc(f32, weight_bytes.len);
    errdefer allocator.free(out);

    for (weight_bytes, 0..) |byte, i| {
        out[i] = DataTransform.Quantizer.lut_e4m3[byte] * scalar_scale;
    }

    return out;
}

/// Dequantize an MXFP4 (OCP MX) cluster to F32.
/// Scale layout is linear row-major; block size is 32. Nibble packing follows OCP:
/// element[2i] in the low nibble, element[2i+1] in the high nibble.
/// Caller owns the returned slice.
pub fn dequantizeMxfp4Cluster(
    cluster: Mxfp4Cluster,
    source: anytype,
    allocator: std.mem.Allocator,
) ![]f32 {
    if (cluster.cols % 32 != 0) return error.InvalidClusterShape;

    const weight_bytes = try readTensorBytes(source, cluster.weight, allocator);
    defer allocator.free(weight_bytes);
    const scale_bytes = try readTensorBytes(source, cluster.weight_scale, allocator);
    defer allocator.free(scale_bytes);

    const rows = cluster.rows;
    const cols = cluster.cols;
    const num_scale_cols = cols / 32;
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);

    for (0..rows) |row| {
        for (0..cols) |col| {
            const byte_idx = row * (cols / 2) + col / 2;
            const nibble: u4 = if (col % 2 == 0)
                @intCast(weight_bytes[byte_idx] & 0xF)
            else
                @intCast(weight_bytes[byte_idx] >> 4);
            const scale_idx = row * num_scale_cols + col / 32;
            const scale = DataTransform.Quantizer.e8m0_to_f32(scale_bytes[scale_idx]);
            out[row * cols + col] = DataTransform.Quantizer.lut_fp4_e2m1[nibble] * scale;
        }
    }

    return out;
}

/// Core MXFP8 dequantization over raw byte slices. Caller owns the returned slice.
/// weight_bytes: F8_E4M3 encoded values [rows × cols].
/// scale_bytes: E8M0 values in linear row-major order, one per 32 elements.
pub fn dequantizeMxfp8Raw(
    weight_bytes: []const u8,
    scale_bytes: []const u8,
    rows: usize,
    cols: usize,
    allocator: std.mem.Allocator,
) ![]f32 {
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);

    const num_scale_cols = cols / 32;
    for (weight_bytes, 0..) |byte, flat_idx| {
        const row = flat_idx / cols;
        const col = flat_idx % cols;
        const scale_idx = row * num_scale_cols + col / 32;
        const scale = DataTransform.Quantizer.e8m0_to_f32(scale_bytes[scale_idx]);
        out[flat_idx] = DataTransform.Quantizer.lut_e4m3[byte] * scale;
    }

    return out;
}

/// Dequantize an MXFP8 E4M3 (OCP MX) cluster to F32.
/// Scale layout is linear row-major; block size is 32.
/// Caller owns the returned slice.
pub fn dequantizeMxfp8Cluster(
    cluster: Mxfp8Cluster,
    source: anytype,
    allocator: std.mem.Allocator,
) ![]f32 {
    if (cluster.cols % 32 != 0) return error.InvalidClusterShape;

    const weight_bytes = try readTensorBytes(source, cluster.weight, allocator);
    defer allocator.free(weight_bytes);
    const scale_bytes = try readTensorBytes(source, cluster.weight_scale, allocator);
    defer allocator.free(scale_bytes);

    return dequantizeMxfp8Raw(weight_bytes, scale_bytes, cluster.rows, cluster.cols, allocator);
}

/// Core ConvRot INT8 dequantization over raw byte slices. Caller owns the returned slice.
/// weight_bytes: I8 values [rows × cols] (row-major). scale_f32: per-row scale [rows].
/// If `convrot`, un-rotates each row group-wise with the Hadamard transform.
pub fn dequantizeInt8ConvrotRaw(
    weight_bytes: []const u8,
    scale_f32: []const f32,
    rows: usize,
    cols: usize,
    convrot: bool,
    group_size: usize,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) ![]f32 {
    if (weight_bytes.len != rows * cols) return error.InvalidClusterShape;
    if (scale_f32.len != rows) return error.InvalidClusterShape;

    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);

    for (0..rows) |row| {
        const s = scale_f32[row];
        for (0..cols) |col| {
            const q: i8 = @bitCast(weight_bytes[row * cols + col]);
            out[row * cols + col] = @as(f32, @floatFromInt(q)) * s;
        }
    }

    if (convrot) try DataTransform.Quantizer.rotateGroupwiseInPlace(out, rows, cols, group_size, pool);

    return out;
}

/// Dequantize a ConvRot INT8 cluster to F32. Caller owns the returned slice.
pub fn dequantizeInt8ConvrotCluster(
    cluster: Int8ConvrotCluster,
    source: anytype,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) ![]f32 {
    const weight_bytes = try readTensorBytes(source, cluster.weight, allocator);
    defer allocator.free(weight_bytes);
    const scale_raw = try readTensorBytes(source, cluster.weight_scale, allocator);
    defer allocator.free(scale_raw);
    const scale_f32: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(scale_raw)));

    return dequantizeInt8ConvrotRaw(weight_bytes, scale_f32, cluster.rows, cluster.cols, cluster.convrot, cluster.group_size, allocator, pool);
}

/// Core int4 dequantization over raw byte slices. Caller owns the returned slice.
/// weight_bytes: nibble-packed signed 4-bit values [rows × cols/2] (row-major); element 2k
/// is the low nibble of byte k, element 2k+1 the high nibble. scale_f32: per-row [rows].
/// If `convrot`, un-rotates each row group-wise with the Hadamard transform.
pub fn dequantizeInt4Raw(
    weight_bytes: []const u8,
    scale_f32: []const f32,
    rows: usize,
    cols: usize,
    convrot: bool,
    group_size: usize,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) ![]f32 {
    if (cols % 2 != 0) return error.InvalidClusterShape;
    const packed_cols = cols / 2;
    if (weight_bytes.len != rows * packed_cols) return error.InvalidClusterShape;
    if (scale_f32.len != rows) return error.InvalidClusterShape;

    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);

    for (0..rows) |row| {
        const s = scale_f32[row];
        for (0..packed_cols) |pc| {
            const byte = weight_bytes[row * packed_cols + pc];
            const lo = signExtendNibble(byte & 0x0F);
            const hi = signExtendNibble(byte >> 4);
            out[row * cols + 2 * pc] = @as(f32, @floatFromInt(lo)) * s;
            out[row * cols + 2 * pc + 1] = @as(f32, @floatFromInt(hi)) * s;
        }
    }

    if (convrot) try DataTransform.Quantizer.rotateGroupwiseInPlace(out, rows, cols, group_size, pool);

    return out;
}

/// Sign-extend a two's-complement 4-bit nibble (0..15) to a signed i8 in [-8, 7].
fn signExtendNibble(nibble: u8) i8 {
    return if (nibble >= 8) @as(i8, @intCast(nibble)) - 16 else @intCast(nibble);
}

/// Dequantize an int4 cluster to F32. Caller owns the returned slice.
pub fn dequantizeInt4Cluster(
    cluster: Int4Cluster,
    source: anytype,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) ![]f32 {
    const weight_bytes = try readTensorBytes(source, cluster.weight, allocator);
    defer allocator.free(weight_bytes);
    const scale_raw = try readTensorBytes(source, cluster.weight_scale, allocator);
    defer allocator.free(scale_raw);
    const scale_f32: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(scale_raw)));

    return dequantizeInt4Raw(weight_bytes, scale_f32, cluster.rows, cluster.cols, cluster.convrot, cluster.group_size, allocator, pool);
}

/// Check whether `dest_tensor` belongs to any cluster in `groups`.
/// If so, dequantize it and return the F32 buffer. Returns null if not cluster-sourced.
/// Caller owns the returned slice.
pub fn tryDequantCluster(
    dest_tensor: types.Tensor,
    source: anytype,
    groups: *const GroupResult,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) !?[]f32 {
    for (groups.fp4_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeFp4Cluster(cluster, source, allocator, pool);
        }
    }
    for (groups.float8_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeFloat8Cluster(cluster, source, allocator);
        }
    }
    for (groups.mxfp4_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeMxfp4Cluster(cluster, source, allocator);
        }
    }
    for (groups.mxfp8_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeMxfp8Cluster(cluster, source, allocator);
        }
    }
    for (groups.int8_convrot_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeInt8ConvrotCluster(cluster, source, allocator, pool);
        }
    }
    for (groups.int4_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeInt4Cluster(cluster, source, allocator, pool);
        }
    }
    return null;
}

/// Single dequant entry point for the safetensors writer: reconstruct output tensor `t`'s source
/// data as F32 whether the source is a cluster (int8_convrot/nvfp4/fp8/mx*, via the per-cluster
/// dequantizers in `tryDequantCluster`) or a plain tensor (via a generic name-matched dequant).
/// Returns null if no source matches. Caller owns the returned slice.
pub fn dequantSourceToF32(
    t: types.Tensor,
    source: anytype,
    groups: *const GroupResult,
    allocator: std.mem.Allocator,
    pool: *thread_pool_mod.ThreadPool,
) !?[]f32 {
    if (try tryDequantCluster(t, source, groups, allocator, pool)) |cluster_f32| return cluster_f32;

    var n_elements: u64 = 1;
    for (t.dims) |d| n_elements *= d;
    if (try loadMatchingSourceAsF32(source, allocator, t.name, n_elements, pool)) |plain|
        return plain.data;
    return null;
}

/// How `collapseModelTensors` types the single logical tensor that replaces each cluster.
pub const CollapseMode = enum {
    /// Collapse to a BF16 weight — the dequantized target used by the conversion pipeline,
    /// which re-quantizes from this float representation.
    dequant,
    /// Collapse to a weight carrying the cluster's *logical* quant type (INT8_CONVROT,
    /// NVFP4, ...). Used by template export so a template generated from a ComfyUI
    /// cluster model round-trips back through `convert` instead of emitting the raw
    /// on-disk sub-tensors (I8 weight + weight_scale + comfy_quant) that `convert`
    /// cannot re-ingest.
    preserve_quant,
};

/// Replace cluster physical tensors in `model_tensors` with a single logical tensor.
/// Companion tensors (weight_scale, weight_scale_2, comfy_quant) are removed from the list.
/// `mode` selects the collapsed weight's type: `.dequant` → BF16 (conversion input),
/// `.preserve_quant` → the cluster's logical quant type (template export).
///
/// Note: in `.preserve_quant` mode the collapsed `size` field is left at the BF16 estimate.
/// It is intentionally not the true cluster byte size — the only consumer (template export)
/// serializes shape + type only and never reads `size`.
pub fn collapseModelTensors(
    model_tensors: *std.ArrayList(types.Tensor),
    groups: *const GroupResult,
    mode: CollapseMode,
    arena_alloc: std.mem.Allocator,
) !void {
    if (groups.fp4_clusters.len == 0 and groups.float8_clusters.len == 0 and
        groups.mxfp4_clusters.len == 0 and groups.mxfp8_clusters.len == 0 and
        groups.int8_convrot_clusters.len == 0 and groups.int4_clusters.len == 0) return;

    var new_tensors: std.ArrayList(types.Tensor) = .empty;

    for (model_tensors.items) |t| {
        var handled = false;

        for (groups.fp4_clusters) |cluster| {
            if (nameSuffixMatch(cluster.weight.name, t.name)) {
                var new_t = t;
                new_t.dims = try arena_alloc.dupe(usize, &[_]usize{ cluster.rows, cluster.cols });
                new_t.type = if (mode == .preserve_quant) "NVFP4" else "BF16";
                new_t.size = cluster.rows * cluster.cols * 2;
                try new_tensors.append(arena_alloc,new_t);
                handled = true;
                break;
            }
            if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                nameSuffixMatch(cluster.weight_scale_2.name, t.name) or
                (if (cluster.comfy_quant) |cq| nameSuffixMatch(cq.name, t.name) else false))
            {
                handled = true; // drop companion tensor
                break;
            }
        }

        if (!handled) {
            for (groups.float8_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.type = if (mode == .preserve_quant) "SCALED_F8_E4M3" else "BF16";
                    var n: usize = 1;
                    for (t.dims) |d| n *= d;
                    new_t.size = n * 2;
                    try new_tensors.append(arena_alloc,new_t);
                    handled = true;
                    break;
                }
                const input_scale_match = if (cluster.input_scale) |is|
                    nameSuffixMatch(is.name, t.name)
                else
                    false;
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    (if (cluster.comfy_quant) |cq| nameSuffixMatch(cq.name, t.name) else false) or
                    input_scale_match)
                {
                    handled = true;
                    break;
                }
            }
        }

        if (!handled) {
            for (groups.mxfp4_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.dims = try arena_alloc.dupe(usize, &[_]usize{ cluster.rows, cluster.cols });
                    new_t.type = if (mode == .preserve_quant) "MXFP4" else "BF16";
                    new_t.size = cluster.rows * cluster.cols * 2;
                    try new_tensors.append(arena_alloc,new_t);
                    handled = true;
                    break;
                }
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    (if (cluster.comfy_quant) |cq| nameSuffixMatch(cq.name, t.name) else false))
                {
                    handled = true;
                    break;
                }
            }
        }

        if (!handled) {
            for (groups.mxfp8_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.type = if (mode == .preserve_quant) "MXFP8_E4M3" else "BF16";
                    var n: usize = 1;
                    for (t.dims) |d| n *= d;
                    new_t.size = n * 2;
                    try new_tensors.append(arena_alloc,new_t);
                    handled = true;
                    break;
                }
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    (if (cluster.comfy_quant) |cq| nameSuffixMatch(cq.name, t.name) else false))
                {
                    handled = true;
                    break;
                }
            }
        }

        if (!handled) {
            for (groups.int8_convrot_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.dims = try arena_alloc.dupe(usize, &[_]usize{ cluster.rows, cluster.cols });
                    new_t.type = if (mode == .preserve_quant)
                        (if (cluster.convrot) "INT8_CONVROT" else "INT8")
                    else
                        "BF16";
                    new_t.size = cluster.rows * cluster.cols * 2;
                    try new_tensors.append(arena_alloc, new_t);
                    handled = true;
                    break;
                }
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    (if (cluster.comfy_quant) |cq| nameSuffixMatch(cq.name, t.name) else false))
                {
                    handled = true;
                    break;
                }
            }
        }

        if (!handled) {
            for (groups.int4_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.dims = try arena_alloc.dupe(usize, &[_]usize{ cluster.rows, cluster.cols });
                    new_t.type = if (mode == .preserve_quant) "INT4_CONVROT" else "BF16";
                    new_t.size = cluster.rows * cluster.cols * 2;
                    try new_tensors.append(arena_alloc, new_t);
                    handled = true;
                    break;
                }
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    (if (cluster.comfy_quant) |cq| nameSuffixMatch(cq.name, t.name) else false))
                {
                    handled = true;
                    break;
                }
            }
        }

        if (!handled) try new_tensors.append(arena_alloc,t);
    }

    model_tensors.* = new_tensors;
}

// =============================================================================
// Cluster write path: single source of truth for on-disk layout + encoding.
//
// A ComfyUI cluster dest type expands into several physical sub-tensors laid out
// contiguously: [weight][scale...][comfy_quant JSON]. `clusterWriteLayout` describes
// that layout (shapes, dtypes, byte sizes) and is consumed by THREE places that must
// agree: Convert.assignTensorType (total size), the safetensors header writer (offsets),
// and `writeClusterData` (the bytes). Keeping them derived from one descriptor removes
// the prior triplicated, drift-prone size/shape math.
// =============================================================================

// ComfyUI quantization-identity blobs (the `.comfy_quant` payload). Byte-exact strings
// that ComfyUI matches on load — do not reformat.
pub const fp8_comfy_json = "{\"format\": \"float8_e4m3fn\"}";
pub const mxfp4_comfy_json = "{\"format\":\"mxfp4\"}";
pub const mxfp8_comfy_json = "{\"format\":\"mxfp8\"}";
pub const nvfp4_comfy_json = "{\"format\": \"nvfp4\"}";
pub const int8_comfy_json = "{\"per_row\": true, \"format\": \"int8_tensorwise\"}";
pub const int8_convrot_comfy_json = "{\"convrot\": true, \"convrot_groupsize\": 256, \"per_row\": true, \"format\": \"int8_tensorwise\"}";
pub const int8_convrot_group_size: u64 = 256;
// ComfyUI "convrot_w4a4": nibble-packed signed 4-bit weight (Hadamard-rotated group-wise
// at convrot_groupsize=256) with per-row F32 scales. quant_group_size=64 is a kernel-tiling
// contract value (storage is still per-row); linear_dtype selects the int4 MMA path.
pub const int4_convrot_comfy_json = "{\"format\": \"convrot_w4a4\", \"convrot_groupsize\": 256, \"quant_group_size\": 64, \"linear_dtype\": \"int4\"}";
pub const int4_convrot_group_size: u64 = 256;
/// Seed used for INT4_CONVROT_SR stochastic rounding when no `--stochastic-rounding`
/// override is supplied. Any nonzero value works; 0 would disable stochastic rounding.
pub const default_stochastic_seed: u64 = 0xC0FFEE;

/// True if `dtype` is a ComfyUI safetensors cluster type (expands into weight + scale(s) +
/// comfy_quant sub-tensors on disk).
pub fn isClusterType(dtype: types.DataType) bool {
    return switch (dtype) {
        .SCALED_F8_E4M3, .MXFP4, .MXFP8_E4M3, .NVFP4, .INT8, .INT8_CONVROT, .INT4_CONVROT, .INT4_CONVROT_SR => true,
        else => false,
    };
}

/// One physical sub-tensor of a cluster. `suffix` is appended to the cluster base name
/// (the weight's own name minus ".weight"); e.g. ".weight", ".weight_scale", ".comfy_quant".
pub const SubTensorSpec = struct {
    suffix: []const u8,
    dtype: []const u8,
    dims: []const usize,
    bytes: u64,
};

fn rowsCols(dims: []const usize) struct { rows: u64, cols: u64 } {
    const cols: u64 = if (dims.len >= 1) dims[dims.len - 1] else 0;
    var rows: u64 = 1;
    if (dims.len >= 2) for (dims[0 .. dims.len - 1]) |d| {
        rows *= d;
    };
    return .{ .rows = rows, .cols = cols };
}

/// Describe the physical sub-tensor layout for a safetensors cluster dest `dtype`.
/// Returns null if `dtype` is not a cluster type. Slices are arena-allocated.
pub fn clusterWriteLayout(arena: std.mem.Allocator, dtype: types.DataType, dims: []const usize) !?[]const SubTensorSpec {
    const rc = rowsCols(dims);
    const rows = rc.rows;
    const cols = rc.cols;
    const n_elements = rows * cols;

    var list: std.ArrayList(SubTensorSpec) = .empty;
    const comfy = struct {
        fn spec(a: std.mem.Allocator, json: []const u8) !SubTensorSpec {
            return .{ .suffix = ".comfy_quant", .dtype = "U8", .dims = try a.dupe(usize, &.{json.len}), .bytes = json.len };
        }
    };

    switch (dtype) {
        .SCALED_F8_E4M3 => {
            try list.append(arena, .{ .suffix = ".weight", .dtype = "F8_E4M3", .dims = dims, .bytes = n_elements });
            try list.append(arena, .{ .suffix = ".weight_scale", .dtype = "F32", .dims = &.{}, .bytes = 4 });
            try list.append(arena, try comfy.spec(arena, fp8_comfy_json));
        },
        .MXFP4 => {
            try list.append(arena, .{ .suffix = ".weight", .dtype = "U32", .dims = try arena.dupe(usize, &.{ rows, cols / 8 }), .bytes = rows * cols / 2 });
            try list.append(arena, .{ .suffix = ".weight_scale", .dtype = "U8", .dims = try arena.dupe(usize, &.{ rows, (cols + 31) / 32 }), .bytes = rows * ((cols + 31) / 32) });
            try list.append(arena, try comfy.spec(arena, mxfp4_comfy_json));
        },
        .MXFP8_E4M3 => {
            const nsc = (cols + 31) / 32;
            const nrb = (rows + 127) / 128;
            const ncb = (nsc + 3) / 4;
            try list.append(arena, .{ .suffix = ".weight", .dtype = "F8_E4M3", .dims = dims, .bytes = n_elements });
            try list.append(arena, .{ .suffix = ".weight_scale", .dtype = "U8", .dims = try arena.dupe(usize, &.{ nrb * 128, ncb * 4 }), .bytes = nrb * 128 * ncb * 4 });
            try list.append(arena, try comfy.spec(arena, mxfp8_comfy_json));
        },
        .NVFP4 => {
            try list.append(arena, .{ .suffix = ".weight", .dtype = "U8", .dims = try arena.dupe(usize, &.{ rows, cols / 2 }), .bytes = rows * (cols / 2) });
            try list.append(arena, .{ .suffix = ".weight_scale", .dtype = "F8_E4M3", .dims = try arena.dupe(usize, &.{ rows, cols / 16 }), .bytes = rows * (cols / 16) });
            try list.append(arena, .{ .suffix = ".weight_scale_2", .dtype = "F32", .dims = &.{}, .bytes = 4 });
            try list.append(arena, try comfy.spec(arena, nvfp4_comfy_json));
        },
        .INT8, .INT8_CONVROT => {
            const json = if (dtype == .INT8_CONVROT) int8_convrot_comfy_json else int8_comfy_json;
            try list.append(arena, .{ .suffix = ".weight", .dtype = "I8", .dims = dims, .bytes = n_elements });
            try list.append(arena, .{ .suffix = ".weight_scale", .dtype = "F32", .dims = try arena.dupe(usize, &.{ rows, 1 }), .bytes = rows * 4 });
            try list.append(arena, try comfy.spec(arena, json));
        },
        .INT4_CONVROT, .INT4_CONVROT_SR => {
            // convrot_w4a4: signed int8 weight, two 4-bit nibbles per byte along the column dim,
            // with 1-D per-row F32 scales (matching comfy_kitchen's on-disk layout). SR shares
            // this exact layout — only the nibble values differ.
            try list.append(arena, .{ .suffix = ".weight", .dtype = "I8", .dims = try arena.dupe(usize, &.{ rows, cols / 2 }), .bytes = rows * (cols / 2) });
            try list.append(arena, .{ .suffix = ".weight_scale", .dtype = "F32", .dims = try arena.dupe(usize, &.{rows}), .bytes = rows * 4 });
            try list.append(arena, try comfy.spec(arena, int4_convrot_comfy_json));
        },
        else => return null,
    }
    return list.items;
}

/// Total on-disk byte size of a cluster dest `dtype`, or null if not a cluster type.
pub fn clusterWriteSize(arena: std.mem.Allocator, dtype: types.DataType, dims: []const usize) !?u64 {
    const specs = (try clusterWriteLayout(arena, dtype, dims)) orelse return null;
    var total: u64 = 0;
    for (specs) |s| total += s.bytes;
    return total;
}

/// Find the source tensor matching `name` (exact or dot-suffix), read it, and convert to
/// F32. Returns the F32 slice (caller owns) and the matched source's dtype name, or null
/// if no source tensor matches.
pub fn loadMatchingSourceAsF32(
    source: anytype,
    allocator: std.mem.Allocator,
    name: []const u8,
    n_elements: u64,
    pool: *thread_pool_mod.ThreadPool,
) !?struct { data: []f32, source_type: []const u8 } {
    for (source.tensors.items) |source_tensor| {
        if (!nameSuffixMatch(source_tensor.name, name)) continue;

        const source_dtype = try types.DataType.fromString(source_tensor.type);
        const source_size: usize = @intCast(source_dtype.calcSizeInBytes(n_elements));
        const src_bytes = try readTensorBytesSized(source, source_tensor, allocator, source_size);
        defer allocator.free(src_bytes);

        const f32_bytes = try DataTransform.Quantizer.convertTensorData(
            allocator, src_bytes, source_dtype, .F32, n_elements, pool,
        );
        const f32_slice: []f32 = @alignCast(std.mem.bytesAsSlice(f32, f32_bytes));
        return .{ .data = f32_slice, .source_type = source_tensor.type };
    }
    return null;
}

/// Like readTensorBytes but with an explicit size (source tensor `.size` may reflect a
/// logical/collapsed view, so callers that recompute the physical size pass it here).
fn readTensorBytesSized(source: anytype, tensor: types.Tensor, allocator: std.mem.Allocator, size: usize) ![]u8 {
    const file = try source.openFileForTensor(tensor.name);
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(source.io, buf, tensor.offset + source.current_data_begin);
    return buf;
}

/// Quantize `f32_data` to the cluster `dtype` and write its physical bytes
/// ([weight][scale...][comfy_quant]) to `writer`, in the order given by clusterWriteLayout.
/// `stochastic_rounding` is the resolved SR seed; it is applied only to INT4_CONVROT_SR
/// (all other dest types quantize deterministically).
pub fn writeClusterData(
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    dtype: types.DataType,
    f32_data: []const f32,
    dims: []const usize,
    stochastic_rounding: u64,
    pool: *thread_pool_mod.ThreadPool,
) !void {
    const rc = rowsCols(dims);
    const rows = rc.rows;
    const cols = rc.cols;

    const Q = DataTransform.Quantizer;
    switch (dtype) {
        .SCALED_F8_E4M3 => {
            const cluster = try Q.quantizeToComfyFp8(allocator, f32_data, pool);
            defer allocator.free(cluster.weight);
            var scale_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &scale_buf, @bitCast(cluster.scale), .little);
            try writer.writeAll(cluster.weight);
            try writer.writeAll(&scale_buf);
            try writer.writeAll(fp8_comfy_json);
        },
        .MXFP4 => {
            const cluster = try Q.quantizeToComfyMxfp4(allocator, f32_data, pool);
            defer allocator.free(cluster.weight);
            defer allocator.free(cluster.scale);
            try writer.writeAll(cluster.weight);
            try writer.writeAll(cluster.scale);
            try writer.writeAll(mxfp4_comfy_json);
        },
        .MXFP8_E4M3 => {
            const cluster = try Q.quantizeToComfyMxfp8(allocator, f32_data, pool);
            defer allocator.free(cluster.weight);
            defer allocator.free(cluster.scale);
            const n_scale_cols: usize = @intCast((cols + 31) / 32);
            const blocked_scale = try Q.toBlockedMxfp8(allocator, cluster.scale, @intCast(rows), n_scale_cols);
            defer allocator.free(blocked_scale);
            try writer.writeAll(cluster.weight);
            try writer.writeAll(blocked_scale);
            try writer.writeAll(mxfp8_comfy_json);
        },
        .NVFP4 => {
            const cluster = try quantizeToNvFp4Raw(f32_data, rows, cols, allocator, pool);
            defer allocator.free(cluster.weight);
            defer allocator.free(cluster.scale);
            var gs_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &gs_buf, @bitCast(cluster.global_scale), .little);
            try writer.writeAll(cluster.weight);
            try writer.writeAll(cluster.scale);
            try writer.writeAll(&gs_buf);
            try writer.writeAll(nvfp4_comfy_json);
        },
        .INT8, .INT8_CONVROT => {
            const is_convrot = dtype == .INT8_CONVROT;
            const cluster = try Q.quantizeToInt8(allocator, f32_data, @intCast(rows), @intCast(cols), is_convrot, @intCast(int8_convrot_group_size), pool);
            defer allocator.free(cluster.weight);
            defer allocator.free(cluster.scale);
            try writer.writeAll(cluster.weight);
            try writer.writeAll(std.mem.sliceAsBytes(cluster.scale));
            try writer.writeAll(if (is_convrot) int8_convrot_comfy_json else int8_comfy_json);
        },
        .INT4_CONVROT, .INT4_CONVROT_SR => {
            // Both write the same convrot_w4a4 layout; SR differs only by rounding the weights
            // stochastically (nonzero seed). Non-SR always quantizes deterministically (seed 0).
            const seed: u64 = if (dtype == .INT4_CONVROT_SR) stochastic_rounding else 0;
            const cluster = try Q.quantizeToInt4(allocator, f32_data, @intCast(rows), @intCast(cols), true, @intCast(int4_convrot_group_size), seed, pool);
            defer allocator.free(cluster.weight);
            defer allocator.free(cluster.scale);
            try writer.writeAll(cluster.weight);
            try writer.writeAll(std.mem.sliceAsBytes(cluster.scale));
            try writer.writeAll(int4_convrot_comfy_json);
        },
        else => return error.NotAClusterType,
    }
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

const fixture_dir = "src/test_fixtures";

fn readFixtureFile(allocator: std.mem.Allocator, path: []const u8, max_size: usize) ![]u8 {
    const io = std.testing.io;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const file_len = try file.length(io);
    if (file_len > max_size) return error.FileTooLarge;
    const buf = try allocator.alloc(u8, @intCast(file_len));
    errdefer allocator.free(buf);
    _ = try file.readPositionalAll(io, buf, 0);
    return buf;
}

fn loadFixture(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    var path_buf: [256]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ fixture_dir, name });
    return readFixtureFile(allocator, path, 64 * 1024 * 1024) catch |err| {
        if (err == error.FileNotFound) return null;
        return err;
    };
}

test "parseComfyQuantScheme: identifies all known formats" {
    try testing.expectEqual(.nvfp4,         parseComfyQuantScheme("{\"format\": \"nvfp4\"}"));
    try testing.expectEqual(.float8_e4m3fn, parseComfyQuantScheme("{\"format\": \"float8_e4m3fn\"}"));
    try testing.expectEqual(.mxfp4,         parseComfyQuantScheme("{\"format\": \"mxfp4\"}"));
    try testing.expectEqual(.mxfp8_e4m3fn,  parseComfyQuantScheme("{\"format\": \"mxfp8_e4m3fn\"}"));
    try testing.expectEqual(.mxfp8_e4m3fn,  parseComfyQuantScheme("{\"format\": \"mxfp8\"}"));
    try testing.expectEqual(.unknown,       parseComfyQuantScheme("{\"format\": \"bf16\"}"));
    try testing.expectEqual(.unknown,       parseComfyQuantScheme("{}"));
    try testing.expectEqual(.unknown,       parseComfyQuantScheme("not json"));
}

test "parseComfyQuantScheme: ignores non-format keys containing scheme names" {
    // Substring search would have returned .float8_e4m3fn here; key-based parsing returns the
    // correct scheme from the "format" field regardless of other fields.
    try testing.expectEqual(
        .mxfp8_e4m3fn,
        parseComfyQuantScheme("{\"format\": \"mxfp8_e4m3fn\", \"note\": \"uses float8_e4m3fn elements\"}"),
    );
    // A "format" key takes precedence; stray mentions of other scheme names elsewhere are ignored.
    try testing.expectEqual(
        .nvfp4,
        parseComfyQuantScheme("{\"format\": \"nvfp4\", \"base\": \"mxfp4\"}"),
    );
}

test "parseComfyQuantScheme: nvfp4 not confused with mxfp4" {
    try testing.expectEqual(.nvfp4, parseComfyQuantScheme("{\"format\": \"nvfp4\"}"));
    try testing.expectEqual(.mxfp4, parseComfyQuantScheme("{\"format\": \"mxfp4\"}"));
}

test "parseComfyQuantScheme: int8 convrot and its boolean/int fields" {
    const blob = "{\"convrot\": true, \"convrot_groupsize\": 256, \"per_row\": true, \"format\": \"int8_tensorwise\"}";
    try testing.expectEqual(.int8_convrot, parseComfyQuantScheme(blob));
    try testing.expectEqual(true, comfyQuantBool(blob, "convrot", false));
    try testing.expectEqual(true, comfyQuantBool(blob, "per_row", false));
    try testing.expectEqual(false, comfyQuantBool(blob, "missing", false));
    try testing.expectEqual(@as(usize, 256), comfyQuantInt(blob, "convrot_groupsize", 999));
    try testing.expectEqual(@as(usize, 64), comfyQuantInt("{\"format\":\"int8_tensorwise\"}", "convrot_groupsize", 64));
}

test "_quantization_metadata: comfy-kitchen layer identity round-trips through the shared parsers" {
    // comfy-kitchen (PR #37) ships no per-tensor `.comfy_quant` markers; it stores a single
    // file-level `_quantization_metadata` header mapping each layer to its identity, using
    // `per_channel` where the marker layout used `per_row`. groupFromQuantMetadata re-serializes
    // each layer Value and routes it through parseComfyQuantScheme/comfyQuantBool/comfyQuantInt;
    // this exercises that round-trip on the exact field layout produced by the converter.
    const allocator = std.testing.allocator;
    const header =
        \\{"format_version": "1.0", "layers": {
        \\  "blocks.0.attn.gate": {"format": "int8_tensorwise", "per_channel": true, "convrot": true, "convrot_groupsize": 256},
        \\  "blocks.0.norm": {"format": "bf16"}
        \\}}
    ;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{});
    defer parsed.deinit();
    const layers = parsed.value.object.get("layers").?.object;

    const gate = layers.get("blocks.0.attn.gate").?;
    const gate_json = try std.json.Stringify.valueAlloc(allocator, gate, .{});
    defer allocator.free(gate_json);
    try testing.expectEqual(.int8_convrot, parseComfyQuantScheme(gate_json));
    try testing.expectEqual(true, comfyQuantBool(gate_json, "convrot", false));
    try testing.expectEqual(@as(usize, 256), comfyQuantInt(gate_json, "convrot_groupsize", 999));

    // A non-int8 layer resolves to .unknown and is skipped (warned) rather than misgrouped.
    const norm = layers.get("blocks.0.norm").?;
    const norm_json = try std.json.Stringify.valueAlloc(allocator, norm, .{});
    defer allocator.free(norm_json);
    try testing.expectEqual(.unknown, parseComfyQuantScheme(norm_json));
}

test "resolveClusterBase: prefix-less metadata key resolves against prefixed tensor names" {
    // comfy-kitchen `_quantization_metadata` keys are logical layer names that may omit the
    // "model.diffusion_model." prefix the actual tensors carry. resolveClusterBase must
    // match fully-qualified keys directly via `name_map` and prefix-less keys by dot-separated
    // suffix, returning the full source-tensor base name.
    const T = types.Tensor;
    var dims = [_]usize{}; // unused by resolveClusterBase; a mutable []usize placeholder
    var tensors = [_]T{
        .{ .name = "model.diffusion_model.blocks.0.attn.gate.weight", .type = "I8", .dims = &dims, .size = 0, .offset = 0 },
        .{ .name = "model.diffusion_model.blocks.0.attn.gate.weight_scale", .type = "F32", .dims = &dims, .size = 0, .offset = 0 },
        .{ .name = "model.diffusion_model.blocks.0.norm.weight", .type = "F32", .dims = &dims, .size = 0, .offset = 0 },
    };
    const src = .{ .tensors = .{ .items = @as([]T, &tensors) } };
    const alloc = std.testing.allocator;

    var name_map = std.StringHashMap(usize).init(alloc);
    defer name_map.deinit();
    for (tensors, 0..) |t, i| try name_map.put(t.name, i);

    // Bare key → full prefixed base name (suffix scan).
    try testing.expectEqualStrings(
        "model.diffusion_model.blocks.0.attn.gate",
        (try resolveClusterBase(src, &name_map, "blocks.0.attn.gate", alloc)).?,
    );
    // Fully-qualified key resolves via the name_map fast path.
    try testing.expectEqualStrings(
        "model.diffusion_model.blocks.0.attn.gate",
        (try resolveClusterBase(src, &name_map, "model.diffusion_model.blocks.0.attn.gate", alloc)).?,
    );
    // No `<key>.weight` tensor → null (the fallback then skips with a warning).
    try testing.expectEqual(@as(?[]const u8, null), try resolveClusterBase(src, &name_map, "blocks.0.attn.missing", alloc));
    // Suffix that does not fall on a dot boundary must not match ("ate.weight" ⊄ ".gate.weight").
    try testing.expectEqual(@as(?[]const u8, null), try resolveClusterBase(src, &name_map, "ate", alloc));
}

test "resolveClusterBase: ambiguous prefix-less key matching multiple tensors resolves to null" {
    // A prefix-less key that suffix-matches more than one `.weight` tensor is ambiguous; rather
    // than silently binding to whichever comes first, resolveClusterBase returns null so the
    // caller skips the layer (warned). A fully-qualified key still resolves unambiguously.
    const T = types.Tensor;
    var dims = [_]usize{};
    var tensors = [_]T{
        .{ .name = "a.blocks.0.attn.gate.weight", .type = "I8", .dims = &dims, .size = 0, .offset = 0 },
        .{ .name = "b.blocks.0.attn.gate.weight", .type = "I8", .dims = &dims, .size = 0, .offset = 0 },
    };
    const src = .{ .tensors = .{ .items = @as([]T, &tensors) } };
    const alloc = std.testing.allocator;

    var name_map = std.StringHashMap(usize).init(alloc);
    defer name_map.deinit();
    for (tensors, 0..) |t, i| try name_map.put(t.name, i);

    // Prefix-less "blocks.0.attn.gate" suffix-matches both a.* and b.* → ambiguous → null.
    try testing.expectEqual(@as(?[]const u8, null), try resolveClusterBase(src, &name_map, "blocks.0.attn.gate", alloc));
    // The fully-qualified key is unambiguous and resolves via the fast path.
    try testing.expectEqualStrings(
        "a.blocks.0.attn.gate",
        (try resolveClusterBase(src, &name_map, "a.blocks.0.attn.gate", alloc)).?,
    );
}

test "groupFromQuantMetadata: routes every supported header scheme to its cluster list" {
    // Mirrors ComfyUI convert_old_quants: a marker-less `_quantization_metadata` header maps each
    // layer to {format, ...}, and each supported scheme must land in its own cluster list with a
    // null comfy_quant (no per-tensor marker in this layout). Grouping reads only tensor metadata
    // (names/dims/size), so a lightweight mock source suffices — no file IO.
    const T = types.Tensor;
    const alloc = std.testing.allocator;

    var d_full = [_]usize{ 4, 8 };  // fp8 / mxfp8 / int8 store [rows, cols]
    var d_half = [_]usize{ 4, 4 };  // nvfp4 stores [rows, cols/2] -> logical cols = 8
    var d_scale = [_]usize{ 4, 1 };

    var tensors = [_]T{
        .{ .name = "fp8.weight", .type = "F8_E4M3", .dims = &d_full, .size = 32, .offset = 0 },
        .{ .name = "fp8.weight_scale", .type = "F32", .dims = &d_scale, .size = 4, .offset = 0 },
        .{ .name = "nv.weight", .type = "U8", .dims = &d_half, .size = 16, .offset = 0 },
        .{ .name = "nv.weight_scale", .type = "F8_E4M3", .dims = &d_scale, .size = 4, .offset = 0 },
        .{ .name = "nv.weight_scale_2", .type = "F32", .dims = &d_scale, .size = 4, .offset = 0 },
        .{ .name = "mx.weight", .type = "F8_E4M3", .dims = &d_full, .size = 32, .offset = 0 },
        .{ .name = "mx.weight_scale", .type = "U8", .dims = &d_scale, .size = 4, .offset = 0 },
        .{ .name = "i8.weight", .type = "I8", .dims = &d_full, .size = 32, .offset = 0 },
        .{ .name = "i8.weight_scale", .type = "F32", .dims = &d_scale, .size = 16, .offset = 0 },
    };
    const src = .{ .tensors = .{ .items = @as([]T, &tensors) } };

    var name_map = std.StringHashMap(usize).init(alloc);
    defer name_map.deinit();
    for (tensors, 0..) |t, i| try name_map.put(t.name, i);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    var fp4_list: std.ArrayList(Fp4Cluster) = .empty;
    var float8_list: std.ArrayList(Float8Cluster) = .empty;
    var mxfp4_list: std.ArrayList(Mxfp4Cluster) = .empty;
    var mxfp8_list: std.ArrayList(Mxfp8Cluster) = .empty;
    var int8_list: std.ArrayList(Int8ConvrotCluster) = .empty;
    var int4_list: std.ArrayList(Int4Cluster) = .empty;

    var seen = std.StringHashMap(void).init(alloc);
    defer seen.deinit();

    // The "norm" bf16 layer resolves to .unknown and is skipped (warned), not misgrouped.
    const header =
        \\{"layers": {
        \\  "fp8": {"format": "float8_e4m3fn"},
        \\  "nv":  {"format": "nvfp4"},
        \\  "mx":  {"format": "mxfp8"},
        \\  "i8":  {"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 4},
        \\  "norm": {"format": "bf16"}
        \\}}
    ;

    try groupFromQuantMetadata(src, &name_map, header, &fp4_list, &float8_list, &mxfp4_list, &mxfp8_list, &int8_list, &int4_list, &seen, aa, alloc);

    try testing.expectEqual(@as(usize, 1), fp4_list.items.len);
    try testing.expectEqual(@as(usize, 1), float8_list.items.len);
    try testing.expectEqual(@as(usize, 1), mxfp8_list.items.len);
    try testing.expectEqual(@as(usize, 1), int8_list.items.len);
    try testing.expectEqual(@as(usize, 0), mxfp4_list.items.len); // no mxfp4 layer in the header

    // Header-sourced clusters carry a null comfy_quant, and dims decode correctly.
    try testing.expectEqual(@as(?types.Tensor, null), fp4_list.items[0].comfy_quant);
    try testing.expectEqual(@as(usize, 8), fp4_list.items[0].cols); // nvfp4: dims[1]*2
    try testing.expectEqual(@as(?types.Tensor, null), float8_list.items[0].comfy_quant);
    try testing.expectEqual(@as(usize, 8), float8_list.items[0].cols);
    try testing.expectEqual(@as(?types.Tensor, null), int8_list.items[0].comfy_quant);
    try testing.expect(int8_list.items[0].convrot);
    try testing.expectEqual(@as(usize, 4), int8_list.items[0].group_size);
}

test "groupFromQuantMetadata: skips bases already grouped via a per-tensor marker" {
    // If both a marker and a header describe the same layer, `seen` (populated by the marker pass)
    // suppresses the header entry so the cluster is not created twice.
    const T = types.Tensor;
    const alloc = std.testing.allocator;

    var d_full = [_]usize{ 4, 8 };
    var d_scale = [_]usize{ 4, 1 };
    var tensors = [_]T{
        .{ .name = "fp8.weight", .type = "F8_E4M3", .dims = &d_full, .size = 32, .offset = 0 },
        .{ .name = "fp8.weight_scale", .type = "F32", .dims = &d_scale, .size = 4, .offset = 0 },
    };
    const src = .{ .tensors = .{ .items = @as([]T, &tensors) } };

    var name_map = std.StringHashMap(usize).init(alloc);
    defer name_map.deinit();
    for (tensors, 0..) |t, i| try name_map.put(t.name, i);

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const aa = arena.allocator();

    var fp4_list: std.ArrayList(Fp4Cluster) = .empty;
    var float8_list: std.ArrayList(Float8Cluster) = .empty;
    var mxfp4_list: std.ArrayList(Mxfp4Cluster) = .empty;
    var mxfp8_list: std.ArrayList(Mxfp8Cluster) = .empty;
    var int8_list: std.ArrayList(Int8ConvrotCluster) = .empty;
    var int4_list: std.ArrayList(Int4Cluster) = .empty;

    var seen = std.StringHashMap(void).init(alloc);
    defer seen.deinit();
    try seen.put("fp8", {}); // pretend the marker pass already grouped this base

    const header = "{\"layers\": {\"fp8\": {\"format\": \"float8_e4m3fn\"}}}";
    try groupFromQuantMetadata(src, &name_map, header, &fp4_list, &float8_list, &mxfp4_list, &mxfp8_list, &int8_list, &int4_list, &seen, aa, alloc);

    try testing.expectEqual(@as(usize, 0), float8_list.items.len); // suppressed by `seen`
}

test "ConvRot INT8 cluster: quantize → dequantize round-trip (convrot on)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB0A710);
    const rand = prng.random();

    const rows: usize = 6;
    const cols: usize = 256;
    const gs: usize = 256;

    const w = try allocator.alloc(f32, rows * cols);
    defer allocator.free(w);
    for (w) |*v| v.* = (rand.float(f32) * 2.0 - 1.0) * 0.1;

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 2 });
    defer pool.deinit();

    const enc = try DataTransform.Quantizer.quantizeToConvrotInt8(allocator, w, rows, cols, gs, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    const got = try dequantizeInt8ConvrotRaw(enc.weight, enc.scale, rows, cols, true, gs, allocator, &pool);
    defer allocator.free(got);

    // The rotation is orthogonal, so it preserves the L2 norm of the quantization
    // error but redistributes it across each group; check the aggregate relative
    // L2 error rather than a per-element bound. INT8 (~7 effective bits) → well under 5%.
    var err_sq: f64 = 0;
    var sig_sq: f64 = 0;
    for (got, w) |g, orig| {
        const d = g - orig;
        err_sq += @as(f64, d) * @as(f64, d);
        sig_sq += @as(f64, orig) * @as(f64, orig);
    }
    const rel = @sqrt(err_sq / sig_sq);
    try testing.expect(rel < 0.05);
}

test "ConvRot INT8 dequant: fixture from real ComfyUI model matches comfy_kitchen" {
    // Validates dequantizeInt8ConvrotRaw against convrot_expected.f32 produced by
    // comfy_kitchen's TensorWiseINT8Layout.dequantize (gen_convrot_fixtures.py).
    // The fast Hadamard transform reorders the summation vs. comfy's dense matmul,
    // so results match to f32 rounding rather than bit-for-bit.
    const allocator = std.testing.allocator;

    const weight_bytes = (try loadFixture(allocator, "convrot_weight.i8")) orelse return error.SkipZigTest;
    defer allocator.free(weight_bytes);
    const scale_raw = (try loadFixture(allocator, "convrot_scale.f32")) orelse return error.SkipZigTest;
    defer allocator.free(scale_raw);
    const expected_raw = (try loadFixture(allocator, "convrot_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_raw);

    const rows: usize = 16;
    const cols: usize = 6144;
    const gs: usize = 256;
    try testing.expectEqual(rows * cols, weight_bytes.len);
    try testing.expectEqual(rows, scale_raw.len / 4);
    const scale: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(scale_raw)));
    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_raw)));
    try testing.expectEqual(rows * cols, expected.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 4 });
    defer pool.deinit();

    const got = try dequantizeInt8ConvrotRaw(weight_bytes, scale, rows, cols, true, gs, allocator, &pool);
    defer allocator.free(got);

    var max_abs_err: f32 = 0;
    for (got, expected) |g, e| max_abs_err = @max(max_abs_err, @abs(g - e));
    try testing.expect(max_abs_err < 1e-4);
}

test "ConvRot INT8 quantize: matches comfy_kitchen re-quantization within rounding" {
    // Validates quantizeToConvrotInt8 against convrot_requant.i8 — comfy_kitchen's own
    // re-quantization of the dequantized weights. Fast vs. dense rotation can flip a
    // rare rounding boundary, so we require the vast majority exact and all within ±1.
    const allocator = std.testing.allocator;

    const expected_raw = (try loadFixture(allocator, "convrot_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_raw);
    const requant = (try loadFixture(allocator, "convrot_requant.i8")) orelse return error.SkipZigTest;
    defer allocator.free(requant);

    const rows: usize = 16;
    const cols: usize = 6144;
    const gs: usize = 256;
    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_raw)));
    try testing.expectEqual(rows * cols, expected.len);
    try testing.expectEqual(rows * cols, requant.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 4 });
    defer pool.deinit();

    const enc = try DataTransform.Quantizer.quantizeToConvrotInt8(allocator, expected, rows, cols, gs, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    var mismatches: usize = 0;
    var over_one: usize = 0;
    for (enc.weight, requant) |ours_u8, theirs_i8| {
        const ours: i16 = @as(i8, @bitCast(ours_u8));
        const theirs: i16 = @as(i8, @bitCast(theirs_i8));
        const d = @abs(ours - theirs);
        if (d != 0) mismatches += 1;
        if (d > 1) over_one += 1;
    }
    try testing.expectEqual(@as(usize, 0), over_one);            // never off by more than one step
    try testing.expect(mismatches * 100 < enc.weight.len);        // < 1% differ by ±1
}

test "plain INT8 quantize: matches comfy_kitchen bit-for-bit (no rotation)" {
    // With convrot=false there is no fast-vs-dense rotation difference, so our per-row
    // quantizer must reproduce comfy_kitchen's quantize_int8_rowwise exactly.
    const allocator = std.testing.allocator;

    const expected_raw = (try loadFixture(allocator, "convrot_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_raw);
    const plain_q = (try loadFixture(allocator, "int8_plain_weight.i8")) orelse return error.SkipZigTest;
    defer allocator.free(plain_q);
    const plain_s_raw = (try loadFixture(allocator, "int8_plain_scale.f32")) orelse return error.SkipZigTest;
    defer allocator.free(plain_s_raw);

    const rows: usize = 16;
    const cols: usize = 6144;
    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_raw)));
    const comfy_scale: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(plain_s_raw)));
    try testing.expectEqual(rows * cols, expected.len);
    try testing.expectEqual(rows * cols, plain_q.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 4 });
    defer pool.deinit();

    const enc = try DataTransform.Quantizer.quantizeToInt8(allocator, expected, rows, cols, false, 0, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    for (enc.weight, plain_q) |ours, theirs| try testing.expectEqual(@as(i8, @bitCast(theirs)), @as(i8, @bitCast(ours)));
    for (enc.scale, comfy_scale) |ours, theirs| try testing.expectApproxEqAbs(theirs, ours, 1e-9);
}

test "plain INT8 cluster: quantize → dequantize round-trip (convrot off, any shape)" {
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1117);
    const rand = prng.random();

    // Non-power-of-4 columns: plain int8 must work where convrot cannot.
    const rows: usize = 10;
    const cols: usize = 200;

    const w = try allocator.alloc(f32, rows * cols);
    defer allocator.free(w);
    for (w) |*v| v.* = (rand.float(f32) * 2.0 - 1.0) * 0.1;

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 2 });
    defer pool.deinit();

    const enc = try DataTransform.Quantizer.quantizeToInt8(allocator, w, rows, cols, false, 0, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    const got = try dequantizeInt8ConvrotRaw(enc.weight, enc.scale, rows, cols, false, 0, allocator, &pool);
    defer allocator.free(got);

    // No rotation, so error is bounded per element by half the per-row quantization step.
    for (0..rows) |r| {
        var amax: f32 = 0;
        for (w[r * cols .. r * cols + cols]) |v| amax = @max(amax, @abs(v));
        const step = @max(amax / 127.0, 1e-30);
        for (0..cols) |c| {
            try testing.expect(@abs(got[r * cols + c] - w[r * cols + c]) <= step * 0.5 + 1e-6);
        }
    }
}

test "ConvRot int4 quantize: matches comfy_kitchen convrot_w4a4 within rounding" {
    // Reference is comfy_kitchen's quantize_convrot_w4a4_weight (stochastic_rounding=0, so
    // deterministic round-half-even, clamp [-7,7]). Our fast radix-4 Hadamard vs. comfy's dense
    // matmul can flip a rare rounding boundary, so we require the vast majority of nibbles exact
    // and all within ±1 int4 step. Nibbles are compared after unpacking so a single flipped
    // nibble is not masked by the shared byte. seed=0 selects the deterministic path.
    const allocator = std.testing.allocator;

    const input_raw = (try loadFixture(allocator, "convrot_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(input_raw);
    const ref_w = (try loadFixture(allocator, "int4_convrot_weight.u8")) orelse return error.SkipZigTest;
    defer allocator.free(ref_w);

    const rows: usize = 16;
    const cols: usize = 6144;
    const gs: usize = 256;
    const input: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(input_raw)));
    try testing.expectEqual(rows * (cols / 2), ref_w.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 4 });
    defer pool.deinit();

    const enc = try DataTransform.Quantizer.quantizeToInt4(allocator, input, rows, cols, true, gs, 0, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    var mismatches: usize = 0;
    var over_one: usize = 0;
    for (enc.weight, ref_w) |ours_b, theirs_b| {
        const nib_ours = [2]i16{ signExtendNibble(ours_b & 0x0F), signExtendNibble(ours_b >> 4) };
        const nib_theirs = [2]i16{ signExtendNibble(theirs_b & 0x0F), signExtendNibble(theirs_b >> 4) };
        for (nib_ours, nib_theirs) |o, t| {
            const d = @abs(o - t);
            if (d != 0) mismatches += 1;
            if (d > 1) over_one += 1;
        }
    }
    const n_nibbles = enc.weight.len * 2;
    try testing.expectEqual(@as(usize, 0), over_one);        // never off by more than one step
    try testing.expect(mismatches * 100 < n_nibbles);        // < 1% differ by ±1
}

test "ConvRot int4 dequant: matches comfy_kitchen convrot_w4a4 within rounding" {
    // Feed comfy_kitchen's own convrot_w4a4 weight+scale through our dequant; only the
    // fast-vs-dense un-rotation can differ, so results match within a tight tolerance.
    const allocator = std.testing.allocator;

    const ref_w = (try loadFixture(allocator, "int4_convrot_weight.u8")) orelse return error.SkipZigTest;
    defer allocator.free(ref_w);
    const ref_s_raw = (try loadFixture(allocator, "int4_convrot_scale.f32")) orelse return error.SkipZigTest;
    defer allocator.free(ref_s_raw);
    const expected_raw = (try loadFixture(allocator, "int4_convrot_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_raw);

    const rows: usize = 16;
    const cols: usize = 6144;
    const gs: usize = 256;
    const ref_scale: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(ref_s_raw)));
    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_raw)));

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 4 });
    defer pool.deinit();

    const got = try dequantizeInt4Raw(ref_w, ref_scale, rows, cols, true, gs, allocator, &pool);
    defer allocator.free(got);

    for (got, expected) |ours, theirs| try testing.expectApproxEqAbs(theirs, ours, 1e-4);
}

test "int4 stochastic rounding: reproducible, seed-dependent, same scale, unbiased" {
    // Verifies the SR contract: (1) a nonzero seed is fully reproducible regardless of thread
    // count, (2) different seeds produce different nibbles, (3) the per-row scale is independent
    // of the rounding mode, and (4) stochastic rounding stays within ±1 step of deterministic
    // rounding while remaining (statistically) unbiased about it.
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1114);
    const rand = prng.random();

    const rows: usize = 10;
    const cols: usize = 256; // valid convrot group size (power of 4)

    const w = try allocator.alloc(f32, rows * cols);
    defer allocator.free(w);
    for (w) |*v| v.* = (rand.float(f32) * 2.0 - 1.0) * 0.1;

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 2 });
    defer pool.deinit();
    var pool1: thread_pool_mod.ThreadPool = undefined;
    try pool1.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool1.deinit();

    const gs: usize = 256;
    const det = try DataTransform.Quantizer.quantizeToInt4(allocator, w, rows, cols, true, gs, 0, &pool);
    defer allocator.free(det.weight);
    defer allocator.free(det.scale);

    const sr_a = try DataTransform.Quantizer.quantizeToInt4(allocator, w, rows, cols, true, gs, 42, &pool);
    defer allocator.free(sr_a.weight);
    defer allocator.free(sr_a.scale);
    // Same seed but a different thread count → identical bytes (RNG keyed by element index).
    const sr_a1 = try DataTransform.Quantizer.quantizeToInt4(allocator, w, rows, cols, true, gs, 42, &pool1);
    defer allocator.free(sr_a1.weight);
    defer allocator.free(sr_a1.scale);
    const sr_b = try DataTransform.Quantizer.quantizeToInt4(allocator, w, rows, cols, true, gs, 7, &pool);
    defer allocator.free(sr_b.weight);
    defer allocator.free(sr_b.scale);

    // (1) reproducible across thread counts.
    for (sr_a.weight, sr_a1.weight) |x, y| try testing.expectEqual(x, y);
    // (3) scale is the same regardless of rounding mode/seed.
    for (det.scale, sr_a.scale) |x, y| try testing.expectEqual(x, y);
    for (det.scale, sr_b.scale) |x, y| try testing.expectEqual(x, y);

    // (2) different seeds differ somewhere, and (4) SR stays within ±1 step of deterministic.
    var seed_diffs: usize = 0;
    var signed_delta: i64 = 0;
    for (det.weight, sr_a.weight, sr_b.weight) |d, a, b| {
        if (a != b) seed_diffs += 1;
        const nd = [2]i16{ signExtendNibble(d & 0x0F), signExtendNibble(d >> 4) };
        const na = [2]i16{ signExtendNibble(a & 0x0F), signExtendNibble(a >> 4) };
        for (nd, na) |dv, av| {
            try testing.expect(@abs(av - dv) <= 1); // never more than one step from deterministic
            signed_delta += @as(i64, av - dv);
        }
    }
    try testing.expect(seed_diffs > 0);
    // (4) unbiased: mean signed deviation over ~2560 nibbles is small (not a systematic drift).
    const n_nibbles: i64 = @intCast(det.weight.len * 2);
    try testing.expect(@abs(signed_delta) * 10 < n_nibbles); // |mean| < 0.1 step
}

test "NVFP4 cuBLAS tiled block index: spot-checks" {
    // This matches the formula in dequantizeFp4Cluster.
    // Parameters: cols=256 → num_scale_cols=16, n_col_blocks=(16+3)/4=4
    const cols: usize = 256;
    const num_scale_cols: usize = cols / 16;
    const n_col_blocks: usize = (num_scale_cols + 3) / 4;
    try testing.expectEqual(@as(usize, 4), n_col_blocks);

    const blockIdx = struct {
        fn f(row: usize, col: usize, ncb: usize) usize {
            const scale_col = col / 16;
            const r0 = row / 128;
            const r1 = row % 128;
            const c0 = scale_col / 4;
            const c1 = scale_col % 4;
            return (r0 * ncb + c0) * 512 + (r1 % 32) * 16 + (r1 / 32) * 4 + c1;
        }
    }.f;

    // Row 0, first 4 scale columns are consecutive bytes 0-3
    try testing.expectEqual(@as(usize, 0), blockIdx(0,   0, n_col_blocks)); // scale_col=0, c0=0, c1=0
    try testing.expectEqual(@as(usize, 1), blockIdx(0,  16, n_col_blocks)); // scale_col=1, c0=0, c1=1
    try testing.expectEqual(@as(usize, 2), blockIdx(0,  32, n_col_blocks)); // scale_col=2, c0=0, c1=2
    try testing.expectEqual(@as(usize, 3), blockIdx(0,  48, n_col_blocks)); // scale_col=3, c0=0, c1=3
    // scale_col=4 starts the next 4-column group (c0=1)
    try testing.expectEqual(@as(usize, 512), blockIdx(0, 64, n_col_blocks)); // (0*4+1)*512+0
    // Row 1: (r1%32)*16 = 16
    try testing.expectEqual(@as(usize, 16), blockIdx(1,  0, n_col_blocks));
    // Row 128: r0=1, r1=0 → (1*4+0)*512 = 2048
    try testing.expectEqual(@as(usize, 2048), blockIdx(128, 0, n_col_blocks));
    // Row 32: r1=32, (r1%32)*16=0, (r1/32)*4=4 → index=4
    try testing.expectEqual(@as(usize, 4), blockIdx(32, 0, n_col_blocks));
}

test "MXFP4/MXFP8 linear block scale index" {
    // Linear layout: scale_idx = row * (cols/32) + col/32
    const cols: usize = 128;
    const num_scale_cols: usize = cols / 32; // = 4
    try testing.expectEqual(@as(usize, 4), num_scale_cols);

    const scaleIdx = struct {
        fn f(row: usize, col: usize, nsc: usize) usize {
            return row * nsc + col / 32;
        }
    }.f;

    try testing.expectEqual(@as(usize, 0),  scaleIdx(0, 0,   num_scale_cols));
    try testing.expectEqual(@as(usize, 1),  scaleIdx(0, 32,  num_scale_cols));
    try testing.expectEqual(@as(usize, 2),  scaleIdx(0, 64,  num_scale_cols));
    try testing.expectEqual(@as(usize, 3),  scaleIdx(0, 96,  num_scale_cols));
    try testing.expectEqual(@as(usize, 4),  scaleIdx(1, 0,   num_scale_cols));
    try testing.expectEqual(@as(usize, 7),  scaleIdx(1, 96,  num_scale_cols));
    try testing.expectEqual(@as(usize, 8),  scaleIdx(2, 0,   num_scale_cols));
}

test "MXFP4 cluster dequant: inline spot-check with E8M0 scale" {
    // Verify the MXFP4 element decode formula used in dequantizeMxfp4Cluster:
    //   nibble → lut_fp4_e2m1[nibble] * e8m0_to_f32(scale_byte)
    // Test nibble=2 (fp4=1.0), scale_byte=128 (2.0) → expected 2.0
    const scale = DataTransform.Quantizer.e8m0_to_f32(128);
    try testing.expectApproxEqAbs(@as(f32, 2.0), scale, 1e-6);

    const fp4_val = DataTransform.Quantizer.lut_fp4_e2m1[2]; // nibble 2 = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), fp4_val, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2.0), fp4_val * scale, 1e-6);

    // nibble=5 (fp4=3.0), scale_byte=130 (8.0) → expected 24.0
    const scale2 = DataTransform.Quantizer.e8m0_to_f32(130);
    try testing.expectApproxEqAbs(@as(f32, 8.0), scale2, 1e-6);
    const fp4_val2 = DataTransform.Quantizer.lut_fp4_e2m1[5]; // nibble 5 = 3.0
    try testing.expectApproxEqAbs(@as(f32, 24.0), fp4_val2 * scale2, 1e-6);
}

test "MXFP8 cluster dequant: inline spot-check with E8M0 scale and F8 LUT" {
    // Verify the MXFP8 element decode formula:
    //   lut_e4m3[byte] * e8m0_to_f32(scale_byte)
    // F8_E4M3 byte 0x38: biased_exp=7, mant=0 → 2^(7-7)*1.0 = 1.0
    const f8_val = DataTransform.Quantizer.lut_e4m3[0x38];
    try testing.expectApproxEqAbs(@as(f32, 1.0), f8_val, 1e-6);

    // scale_byte=128 → 2.0; product = 2.0
    const scale = DataTransform.Quantizer.e8m0_to_f32(128);
    try testing.expectApproxEqAbs(@as(f32, 2.0), f8_val * scale, 1e-6);
}

test "MXFP8 dequant: fixture from real ComfyUI model (128×128 slice)" {
    // Validates dequantizeMxfp8Raw against data extracted from
    // etCenterV1_v10MXFP8.safetensors by gen_mxfp8_fixtures.py.
    // Linear row-major scale layout, block size 32, no cuBLAS tiling.
    const allocator = std.testing.allocator;

    const weight_bytes = (try loadFixture(allocator, "mxfp8_weight.u8")) orelse return error.SkipZigTest;
    defer allocator.free(weight_bytes);
    const scale_bytes = (try loadFixture(allocator, "mxfp8_weight_scale.u8")) orelse return error.SkipZigTest;
    defer allocator.free(scale_bytes);
    const expected_bytes = (try loadFixture(allocator, "mxfp8_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const rows: usize = 128;
    const cols: usize = 128;
    try testing.expectEqual(rows * cols, weight_bytes.len);
    try testing.expectEqual(rows * (cols / 32), scale_bytes.len);

    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try testing.expectEqual(rows * cols, expected.len);

    const got = try dequantizeMxfp8Raw(weight_bytes, scale_bytes, rows, cols, allocator);
    defer allocator.free(got);

    var mismatches: usize = 0;
    for (got, expected, 0..) |g, e, i| {
        if (@abs(g - e) > 1e-6) {
            if (mismatches < 8) {
                std.debug.print("  MXFP8[{}]: got={d:.8} expected={d:.8}\n", .{ i, g, e });
            }
            mismatches += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), mismatches);
}

test "NVFP4 encode→decode round-trip: fixture data" {
    // Encodes nvfp4_expected.f32 (dequantized from a real NVFP4 model) with
    // quantizeToNvFp4Raw, then decodes with dequantizeFp4Raw, and checks that
    // every element round-trips within FP4 quantization error.
    //
    // Using already-quantized data as input means the encode should be very
    // faithful; the tolerance is set to 30% relative + small absolute floor,
    // which is well within the theoretical ~50% worst-case FP4 step error.
    const allocator = std.testing.allocator;

    const input_bytes = (try loadFixture(allocator, "nvfp4_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(input_bytes);

    const input: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(input_bytes)));
    const rows: usize = 128;
    const cols: usize = 256;
    try testing.expectEqual(rows * cols, input.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    const enc = try quantizeToNvFp4Raw(input, rows, cols, allocator, &pool);
    defer allocator.free(enc.weight);
    defer allocator.free(enc.scale);

    try testing.expectEqual(rows * (cols / 2), enc.weight.len);
    try testing.expectEqual(rows * (cols / 16), enc.scale.len);

    const got = try dequantizeFp4Raw(enc.weight, enc.scale, enc.global_scale, rows, cols, allocator, &pool);
    defer allocator.free(got);

    var max_input: f32 = 0.0;
    for (input) |v| max_input = @max(max_input, @abs(v));

    var mismatches: usize = 0;
    for (got, input, 0..) |g, e, i| {
        const tol = @max(@abs(e) * 0.30, max_input * 0.005);
        if (@abs(g - e) > tol) {
            if (mismatches < 8) {
                std.debug.print("  NVFP4-enc[{}]: got={d:.6} input={d:.6} tol={d:.6}\n", .{ i, g, e, tol });
            }
            mismatches += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), mismatches);
}

test "NVFP4 dequant: fixture from real ComfyUI model (128×256 slice)" {
    // Validates dequantizeFp4Raw against data extracted from
    // ernieImageQuants_turboNVFP4.safetensors by gen_nvfp4_fixtures.py.
    // Covers the full r1 range (0..127) and all four cuBLAS column-block
    // groups (n_col_blocks=4 for COLS=256), using scale indices 0..2047.
    const allocator = std.testing.allocator;

    const weight_bytes = (try loadFixture(allocator, "nvfp4_weight.u8")) orelse return error.SkipZigTest;
    defer allocator.free(weight_bytes);
    const scale_bytes = (try loadFixture(allocator, "nvfp4_weight_scale.u8")) orelse return error.SkipZigTest;
    defer allocator.free(scale_bytes);
    const gs_bytes = (try loadFixture(allocator, "nvfp4_weight_scale_2.f32")) orelse return error.SkipZigTest;
    defer allocator.free(gs_bytes);
    const expected_bytes = (try loadFixture(allocator, "nvfp4_expected.f32")) orelse return error.SkipZigTest;
    defer allocator.free(expected_bytes);

    const rows: usize = 128;
    const cols: usize = 256;
    try testing.expectEqual(rows * (cols / 2), weight_bytes.len);
    try testing.expectEqual(@as(usize, 2048), scale_bytes.len);

    const global_scale: f32 = @bitCast(std.mem.readInt(u32, gs_bytes[0..4], .little));
    const expected: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(expected_bytes)));
    try testing.expectEqual(rows * cols, expected.len);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = allocator, .n_jobs = 1 });
    defer pool.deinit();

    const got = try dequantizeFp4Raw(weight_bytes, scale_bytes, global_scale, rows, cols, allocator, &pool);
    defer allocator.free(got);

    var mismatches: usize = 0;
    for (got, expected, 0..) |g, e, i| {
        if (@abs(g - e) > 1e-6) {
            if (mismatches < 8) {
                std.debug.print("  NVFP4[{}]: got={d:.8} expected={d:.8}\n", .{ i, g, e });
            }
            mismatches += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), mismatches);
}

// Helper for the collapse tests: a physical tensor referenced by name only.
fn tc_named(name: []const u8) types.Tensor {
    return .{ .name = name, .type = "", .dims = &[_]usize{}, .size = 0, .offset = 0 };
}

test "collapseModelTensors preserve_quant: cluster weight keeps its logical quant type, sidecars dropped" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Two int8 layers (one ConvRot, one plain) plus an untouched passthrough tensor.
    // This mirrors a ComfyUI int8 model: each layer stores weight + weight_scale + comfy_quant.
    var dims_a = [_]usize{ 2048, 1024 };
    var dims_b = [_]usize{ 512, 768 };
    var dims_norm = [_]usize{2048};
    var model: std.ArrayList(types.Tensor) = .empty;
    try model.append(a, .{ .name = "blocks.0.attn.q_proj.weight", .type = "I8", .dims = &dims_a, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.attn.q_proj.weight_scale", .type = "F32", .dims = &dims_a, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.attn.q_proj.comfy_quant", .type = "U8", .dims = &dims_a, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.mlp.fc.weight", .type = "I8", .dims = &dims_b, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.mlp.fc.weight_scale", .type = "F32", .dims = &dims_b, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.mlp.fc.comfy_quant", .type = "U8", .dims = &dims_b, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.norm.weight", .type = "BF16", .dims = &dims_norm, .size = 0, .offset = 0 });

    var int8_clusters = [_]Int8ConvrotCluster{
        .{
            .base_name = "blocks.0.attn.q_proj",
            .weight = tc_named("blocks.0.attn.q_proj.weight"),
            .weight_scale = tc_named("blocks.0.attn.q_proj.weight_scale"),
            .comfy_quant = tc_named("blocks.0.attn.q_proj.comfy_quant"),
            .rows = 2048,
            .cols = 1024,
            .convrot = true,
            .group_size = 256,
        },
        .{
            .base_name = "blocks.0.mlp.fc",
            .weight = tc_named("blocks.0.mlp.fc.weight"),
            .weight_scale = tc_named("blocks.0.mlp.fc.weight_scale"),
            .comfy_quant = tc_named("blocks.0.mlp.fc.comfy_quant"),
            .rows = 512,
            .cols = 768,
            .convrot = false,
            .group_size = 256,
        },
    };
    const groups = GroupResult{
        .fp4_clusters = &.{},
        .float8_clusters = &.{},
        .mxfp4_clusters = &.{},
        .mxfp8_clusters = &.{},
        .int8_convrot_clusters = &int8_clusters,
        .int4_clusters = &.{},
    };

    try collapseModelTensors(&model, &groups, .preserve_quant, a);

    // sidecars gone; each cluster collapsed to one weight with its logical quant type.
    try testing.expectEqual(@as(usize, 3), model.items.len);

    const q = model.items[0];
    try testing.expectEqualStrings("blocks.0.attn.q_proj.weight", q.name);
    try testing.expectEqualStrings("INT8_CONVROT", q.type); // convrot=true
    try testing.expectEqual(@as(usize, 2048), q.dims[0]);
    try testing.expectEqual(@as(usize, 1024), q.dims[1]);

    const fc = model.items[1];
    try testing.expectEqualStrings("blocks.0.mlp.fc.weight", fc.name);
    try testing.expectEqualStrings("INT8", fc.type); // convrot=false → plain int8_tensorwise

    const norm = model.items[2];
    try testing.expectEqualStrings("blocks.0.norm.weight", norm.name);
    try testing.expectEqualStrings("BF16", norm.type); // passthrough untouched

    // No leftover .weight_scale / .comfy_quant entries.
    for (model.items) |t| {
        try testing.expect(!std.mem.endsWith(u8, t.name, ".weight_scale"));
        try testing.expect(!std.mem.endsWith(u8, t.name, ".comfy_quant"));
    }
}

test "collapseModelTensors dequant: same clusters collapse to BF16 (conversion path unchanged)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var dims_a = [_]usize{ 2048, 1024 };
    var model: std.ArrayList(types.Tensor) = .empty;
    try model.append(a, .{ .name = "blocks.0.attn.q_proj.weight", .type = "I8", .dims = &dims_a, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.attn.q_proj.weight_scale", .type = "F32", .dims = &dims_a, .size = 0, .offset = 0 });
    try model.append(a, .{ .name = "blocks.0.attn.q_proj.comfy_quant", .type = "U8", .dims = &dims_a, .size = 0, .offset = 0 });

    var int8_clusters = [_]Int8ConvrotCluster{.{
        .base_name = "blocks.0.attn.q_proj",
        .weight = tc_named("blocks.0.attn.q_proj.weight"),
        .weight_scale = tc_named("blocks.0.attn.q_proj.weight_scale"),
        .comfy_quant = tc_named("blocks.0.attn.q_proj.comfy_quant"),
        .rows = 2048,
        .cols = 1024,
        .convrot = true,
        .group_size = 256,
    }};
    const groups = GroupResult{
        .fp4_clusters = &.{},
        .float8_clusters = &.{},
        .mxfp4_clusters = &.{},
        .mxfp8_clusters = &.{},
        .int8_convrot_clusters = &int8_clusters,
        .int4_clusters = &.{},
    };

    try collapseModelTensors(&model, &groups, .dequant, a);

    try testing.expectEqual(@as(usize, 1), model.items.len);
    try testing.expectEqualStrings("BF16", model.items[0].type);
    try testing.expectEqual(@as(u64, 2048 * 1024 * 2), model.items[0].size);
}
