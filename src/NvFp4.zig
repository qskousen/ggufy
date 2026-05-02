const std = @import("std");
const types = @import("types.zig");
const st = @import("Safetensor.zig");
const DataTransform = @import("DataTransform.zig");

pub const ComfyQuantScheme = enum { nvfp4, float8_e4m3fn, unknown };

pub const NvFp4Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,
    weight_scale: types.Tensor,
    weight_scale_2: types.Tensor,
    comfy_quant: types.Tensor,
    rows: usize,
    cols: usize,
};

pub const Float8Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,
    weight_scale: types.Tensor,
    comfy_quant: types.Tensor,
    rows: usize,
    cols: usize,
};

pub const GroupResult = struct {
    nvfp4_clusters: []NvFp4Cluster,
    float8_clusters: []Float8Cluster,
};

/// Parse the JSON payload of a comfy_quant blob to identify the quantization scheme.
pub fn parseComfyQuantScheme(data: []const u8) ComfyQuantScheme {
    if (std.mem.indexOf(u8, data, "nvfp4") != null) return .nvfp4;
    if (std.mem.indexOf(u8, data, "float8_e4m3fn") != null) return .float8_e4m3fn;
    return .unknown;
}

/// Returns true if `stripped` matches `full_name` exactly or as a dot-separated suffix.
fn nameSuffixMatch(full_name: []const u8, stripped: []const u8) bool {
    if (std.mem.eql(u8, full_name, stripped)) return true;
    return full_name.len > stripped.len and
        full_name[full_name.len - stripped.len - 1] == '.' and
        std.mem.endsWith(u8, full_name, stripped);
}

/// Scan source tensors and group them into NVFP4 and FP8 clusters via comfy_quant markers.
/// Result slices are arena-allocated; `allocator` is used for temporary work only.
pub fn groupClusters(
    source: *st,
    arena_alloc: std.mem.Allocator,
    allocator: std.mem.Allocator,
) !GroupResult {
    var name_map = std.StringHashMap(usize).init(allocator);
    defer name_map.deinit();
    for (source.tensors.items, 0..) |t, i| try name_map.put(t.name, i);

    var nvfp4_list: std.ArrayList(NvFp4Cluster) = .{};
    var float8_list: std.ArrayList(Float8Cluster) = .{};

    const comfy_suffix = ".comfy_quant";
    var read_buf: [256]u8 = undefined;

    for (source.tensors.items) |t| {
        if (!std.mem.endsWith(u8, t.name, comfy_suffix)) continue;

        var reader = try source.getReaderForTensor(t.name, &read_buf);
        try reader.seekTo(t.offset + source.current_data_begin);
        const data = try (&reader.interface).readAlloc(allocator, t.size);
        defer allocator.free(data);

        const scheme = parseComfyQuantScheme(data);
        const base_name = t.name[0 .. t.name.len - comfy_suffix.len];

        switch (scheme) {
            .nvfp4 => {
                const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
                defer allocator.free(wname);
                const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
                defer allocator.free(wsname);
                const ws2name = try std.fmt.allocPrint(allocator, "{s}.weight_scale_2", .{base_name});
                defer allocator.free(ws2name);

                const wi = name_map.get(wname) orelse {
                    std.log.warn("NvFp4: missing .weight for cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("NvFp4: missing .weight_scale for cluster {s}", .{base_name});
                    continue;
                };
                const ws2i = name_map.get(ws2name) orelse {
                    std.log.warn("NvFp4: missing .weight_scale_2 for cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];
                const weight_scale_2 = source.tensors.items[ws2i];

                try nvfp4_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .weight_scale_2 = weight_scale_2,
                    .comfy_quant = t,
                    .rows = weight.dims[0],
                    .cols = weight.dims[1] * 2,
                });
                std.log.debug("NvFp4: grouped nvfp4 cluster {s} [{}, {}]", .{
                    base_name, weight.dims[0], weight.dims[1] * 2,
                });
            },
            .float8_e4m3fn => {
                const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
                defer allocator.free(wname);
                const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
                defer allocator.free(wsname);

                const wi = name_map.get(wname) orelse {
                    std.log.warn("NvFp4: missing .weight for fp8 cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("NvFp4: missing .weight_scale for fp8 cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];

                try float8_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .comfy_quant = t,
                    .rows = weight.dims[0],
                    .cols = weight.dims[1],
                });
                std.log.debug("NvFp4: grouped fp8 cluster {s} [{}, {}]", .{
                    base_name, weight.dims[0], weight.dims[1],
                });
            },
            .unknown => {},
        }
    }

    std.log.info("NvFp4: found {} nvfp4 and {} fp8 clusters", .{
        nvfp4_list.items.len, float8_list.items.len,
    });

    return GroupResult{
        .nvfp4_clusters = nvfp4_list.items,
        .float8_clusters = float8_list.items,
    };
}

/// Dequantize an NVFP4 cluster to a flat F32 slice of [rows * cols] elements.
/// Caller owns the returned slice.
pub fn dequantizeNvFp4Cluster(
    cluster: NvFp4Cluster,
    source: *st,
    allocator: std.mem.Allocator,
) ![]f32 {
    if (cluster.cols % 16 != 0) return error.InvalidClusterShape;

    var read_buf: [4096]u8 = undefined;

    var w_reader = try source.getReaderForTensor(cluster.weight.name, &read_buf);
    try w_reader.seekTo(cluster.weight.offset + source.current_data_begin);
    const weight_bytes = try (&w_reader.interface).readAlloc(allocator, cluster.weight.size);
    defer allocator.free(weight_bytes);

    var ws_reader = try source.getReaderForTensor(cluster.weight_scale.name, &read_buf);
    try ws_reader.seekTo(cluster.weight_scale.offset + source.current_data_begin);
    const scale_bytes = try (&ws_reader.interface).readAlloc(allocator, cluster.weight_scale.size);
    defer allocator.free(scale_bytes);

    var ws2_reader = try source.getReaderForTensor(cluster.weight_scale_2.name, &read_buf);
    try ws2_reader.seekTo(cluster.weight_scale_2.offset + source.current_data_begin);
    const gs_buf = try (&ws2_reader.interface).readAlloc(allocator, 4);
    defer allocator.free(gs_buf);
    const global_scale: f32 = @bitCast(std.mem.readInt(u32, gs_buf[0..4], .little));

    const rows = cluster.rows;
    const cols = cluster.cols;
    const out = try allocator.alloc(f32, rows * cols);
    errdefer allocator.free(out);

    for (0..rows) |row| {
        for (0..cols) |col| {
            const byte_idx = row * (cols / 2) + col / 2;
            const nibble: u4 = if (col % 2 == 0)
                @intCast(weight_bytes[byte_idx] & 0xF)
            else
                @intCast(weight_bytes[byte_idx] >> 4);

            const fp4_val = DataTransform.Quantizer.lut_fp4_e2m1[nibble];
            const scale_idx = row * (cols / 16) + col / 16;
            const block_scale = DataTransform.Quantizer.lut_e4m3[scale_bytes[scale_idx]];
            out[row * cols + col] = fp4_val * block_scale * global_scale;
        }
    }

    return out;
}

/// Dequantize an FP8 (ComfyUI) cluster to F32: f8_weight × scalar_scale.
/// Caller owns the returned slice.
pub fn dequantizeFloat8Cluster(
    cluster: Float8Cluster,
    source: *st,
    allocator: std.mem.Allocator,
) ![]f32 {
    var read_buf: [4096]u8 = undefined;

    var w_reader = try source.getReaderForTensor(cluster.weight.name, &read_buf);
    try w_reader.seekTo(cluster.weight.offset + source.current_data_begin);
    const weight_bytes = try (&w_reader.interface).readAlloc(allocator, cluster.weight.size);
    defer allocator.free(weight_bytes);

    var ws_reader = try source.getReaderForTensor(cluster.weight_scale.name, &read_buf);
    try ws_reader.seekTo(cluster.weight_scale.offset + source.current_data_begin);
    const scale_buf = try (&ws_reader.interface).readAlloc(allocator, cluster.weight_scale.size);
    defer allocator.free(scale_buf);
    const scalar_scale: f32 = @bitCast(std.mem.readInt(u32, scale_buf[0..4], .little));

    const out = try allocator.alloc(f32, weight_bytes.len);
    errdefer allocator.free(out);

    for (weight_bytes, 0..) |byte, i| {
        out[i] = DataTransform.Quantizer.lut_e4m3[byte] * scalar_scale;
    }

    return out;
}

/// Check whether `dest_tensor` belongs to any cluster in `groups`.
/// If so, dequantize it and return the F32 buffer. Returns null if not cluster-sourced.
/// Caller owns the returned slice.
pub fn tryDequantCluster(
    dest_tensor: types.Tensor,
    source: *st,
    groups: *const GroupResult,
    allocator: std.mem.Allocator,
    pool: *std.Thread.Pool,
) !?[]f32 {
    _ = pool;

    for (groups.nvfp4_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeNvFp4Cluster(cluster, source, allocator);
        }
    }
    for (groups.float8_clusters) |cluster| {
        if (nameSuffixMatch(cluster.weight.name, dest_tensor.name)) {
            return try dequantizeFloat8Cluster(cluster, source, allocator);
        }
    }
    return null;
}

/// Replace cluster physical tensors in `model_tensors` with single logical BF16 tensors.
/// Companion tensors (weight_scale, weight_scale_2, comfy_quant) are removed from the list.
pub fn collapseModelTensors(
    model_tensors: *std.ArrayList(types.Tensor),
    groups: *const GroupResult,
    arena_alloc: std.mem.Allocator,
) !void {
    if (groups.nvfp4_clusters.len == 0 and groups.float8_clusters.len == 0) return;

    var new_tensors: std.ArrayList(types.Tensor) = .{};

    for (model_tensors.items) |t| {
        var handled = false;

        for (groups.nvfp4_clusters) |cluster| {
            if (nameSuffixMatch(cluster.weight.name, t.name)) {
                var new_t = t;
                new_t.dims = try arena_alloc.dupe(usize, &[_]usize{ cluster.rows, cluster.cols });
                new_t.type = "BF16";
                new_t.size = cluster.rows * cluster.cols * 2;
                try new_tensors.append(arena_alloc, new_t);
                handled = true;
                break;
            }
            if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                nameSuffixMatch(cluster.weight_scale_2.name, t.name) or
                nameSuffixMatch(cluster.comfy_quant.name, t.name))
            {
                handled = true; // drop companion tensor
                break;
            }
        }

        if (!handled) {
            for (groups.float8_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.type = "BF16";
                    var n: usize = 1;
                    for (t.dims) |d| n *= d;
                    new_t.size = n * 2;
                    try new_tensors.append(arena_alloc, new_t);
                    handled = true;
                    break;
                }
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    nameSuffixMatch(cluster.comfy_quant.name, t.name))
                {
                    handled = true;
                    break;
                }
            }
        }

        if (!handled) try new_tensors.append(arena_alloc, t);
    }

    model_tensors.* = new_tensors;
}
