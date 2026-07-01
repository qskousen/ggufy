const std = @import("std");
const types = @import("types.zig");
const DataTransform = @import("DataTransform.zig");
const thread_pool_mod = @import("ThreadPool.zig");

pub const ComfyQuantScheme = enum { nvfp4, float8_e4m3fn, mxfp4, mxfp8_e4m3fn, int8_convrot, unknown };

pub const Fp4Cluster = struct {
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
    input_scale: ?types.Tensor,  // optional activation scale (scalar, dropped on conversion)
    comfy_quant: types.Tensor,
    rows: usize,
    cols: usize,
};

/// MXFP4 cluster (OCP MX spec): E2M1 packed nibbles + E8M0 per-block scale, block size 32.
/// Scale layout is linear row-major, distinct from NVFP4's cuBLAS-tiled layout.
pub const Mxfp4Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // U8 packed nibbles, [rows, cols/2]
    weight_scale: types.Tensor,  // U8 E8M0, [rows, cols/32]
    comfy_quant: types.Tensor,
    rows: usize,
    cols: usize,
};

/// MXFP8 cluster (OCP MX spec): E4M3 elements + E8M0 per-block scale, block size 32.
pub const Mxfp8Cluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // F8_E4M3, [rows, cols]
    weight_scale: types.Tensor,  // U8 E8M0, [rows, cols/32]
    comfy_quant: types.Tensor,
    rows: usize,
    cols: usize,
};

/// ConvRot INT8 cluster (ComfyUI "int8_tensorwise" with convrot + per_row):
/// I8 weight rotated by a group-wise Hadamard, F32 per-row scale.
pub const Int8ConvrotCluster = struct {
    base_name: []const u8,
    weight: types.Tensor,        // I8, [rows, cols]
    weight_scale: types.Tensor,  // F32, [rows, 1]
    comfy_quant: types.Tensor,
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
fn nameSuffixMatch(full_name: []const u8, stripped: []const u8) bool {
    if (std.mem.eql(u8, full_name, stripped)) return true;
    return full_name.len > stripped.len and
        full_name[full_name.len - stripped.len - 1] == '.' and
        std.mem.endsWith(u8, full_name, stripped);
}

/// Scan source tensors and group them into NVFP4 and FP8 clusters via comfy_quant markers.
/// Result slices are arena-allocated; `allocator` is used for temporary work only.
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

    const comfy_suffix = ".comfy_quant";

    for (source.tensors.items) |t| {
        if (!std.mem.endsWith(u8, t.name, comfy_suffix)) continue;

        const tensor_file = try source.openFileForTensor(t.name);
        const data = try allocator.alloc(u8, t.size);
        defer allocator.free(data);
        _ = try tensor_file.readPositionalAll(source.io, data, t.offset + source.current_data_begin);

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
                    std.log.warn("TensorClusters: missing .weight for cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("TensorClusters: missing .weight_scale for cluster {s}", .{base_name});
                    continue;
                };
                const ws2i = name_map.get(ws2name) orelse {
                    std.log.warn("TensorClusters: missing .weight_scale_2 for cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];
                const weight_scale_2 = source.tensors.items[ws2i];

                try fp4_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .weight_scale_2 = weight_scale_2,
                    .comfy_quant = t,
                    .rows = weight.dims[0],
                    .cols = weight.dims[1] * 2,
                });
                std.log.debug("TensorClusters: grouped nvfp4 cluster {s} [{}, {}]", .{
                    base_name, weight.dims[0], weight.dims[1] * 2,
                });
            },
            .float8_e4m3fn => {
                const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
                defer allocator.free(wname);
                const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
                defer allocator.free(wsname);
                const isname = try std.fmt.allocPrint(allocator, "{s}.input_scale", .{base_name});
                defer allocator.free(isname);

                const wi = name_map.get(wname) orelse {
                    std.log.warn("TensorClusters: missing .weight for fp8 cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("TensorClusters: missing .weight_scale for fp8 cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];
                const input_scale: ?types.Tensor = if (name_map.get(isname)) |isi|
                    source.tensors.items[isi]
                else
                    null;

                try float8_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .input_scale = input_scale,
                    .comfy_quant = t,
                    .rows = weight.dims[0],
                    .cols = weight.dims[1],
                });
                std.log.debug("TensorClusters: grouped fp8 cluster {s} [{}, {}]", .{
                    base_name, weight.dims[0], weight.dims[1],
                });
            },
            .mxfp4 => {
                const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
                defer allocator.free(wname);
                const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
                defer allocator.free(wsname);

                const wi = name_map.get(wname) orelse {
                    std.log.warn("TensorClusters: missing .weight for mxfp4 cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("TensorClusters: missing .weight_scale for mxfp4 cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];

                try mxfp4_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .comfy_quant = t,
                    .rows = weight.dims[0],
                    .cols = weight.dims[1] * 2,
                });
                std.log.debug("TensorClusters: grouped mxfp4 cluster {s} [{}, {}]", .{
                    base_name, weight.dims[0], weight.dims[1] * 2,
                });
            },
            .mxfp8_e4m3fn => {
                const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
                defer allocator.free(wname);
                const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
                defer allocator.free(wsname);

                const wi = name_map.get(wname) orelse {
                    std.log.warn("TensorClusters: missing .weight for mxfp8 cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("TensorClusters: missing .weight_scale for mxfp8 cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];

                try mxfp8_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .comfy_quant = t,
                    .rows = weight.dims[0],
                    .cols = weight.dims[1],
                });
                std.log.debug("TensorClusters: grouped mxfp8 cluster {s} [{}, {}]", .{
                    base_name, weight.dims[0], weight.dims[1],
                });
            },
            .int8_convrot => {
                const wname = try std.fmt.allocPrint(allocator, "{s}.weight", .{base_name});
                defer allocator.free(wname);
                const wsname = try std.fmt.allocPrint(allocator, "{s}.weight_scale", .{base_name});
                defer allocator.free(wsname);

                const wi = name_map.get(wname) orelse {
                    std.log.warn("TensorClusters: missing .weight for int8_convrot cluster {s}", .{base_name});
                    continue;
                };
                const wsi = name_map.get(wsname) orelse {
                    std.log.warn("TensorClusters: missing .weight_scale for int8_convrot cluster {s}", .{base_name});
                    continue;
                };

                const weight = source.tensors.items[wi];
                const weight_scale = source.tensors.items[wsi];

                if (weight.dims.len != 2) {
                    std.log.warn("TensorClusters: int8_convrot cluster {s} weight is not 2-D; skipping", .{base_name});
                    continue;
                }
                const rows = weight.dims[0];
                const cols = weight.dims[1];
                // Per-row scaling only: expect one F32 scale per output row.
                if (weight_scale.size != rows * 4) {
                    std.log.warn("TensorClusters: int8_convrot cluster {s} has non-per-row scale ({} bytes, expected {}); skipping", .{ base_name, weight_scale.size, rows * 4 });
                    continue;
                }

                const convrot = comfyQuantBool(data, "convrot", false);
                const group_size = comfyQuantInt(data, "convrot_groupsize", 256);
                if (convrot and (!DataTransform.Quantizer.isValidHadamardSize(group_size) or cols % group_size != 0)) {
                    std.log.warn("TensorClusters: int8_convrot cluster {s} has incompatible group_size {} for cols {}; skipping", .{ base_name, group_size, cols });
                    continue;
                }

                try int8_convrot_list.append(arena_alloc, .{
                    .base_name = base_name,
                    .weight = weight,
                    .weight_scale = weight_scale,
                    .comfy_quant = t,
                    .rows = rows,
                    .cols = cols,
                    .convrot = convrot,
                    .group_size = group_size,
                });
                std.log.debug("TensorClusters: grouped int8_convrot cluster {s} [{}, {}] convrot={} gs={}", .{
                    base_name, rows, cols, convrot, group_size,
                });
            },
            .unknown => {},
        }
    }

    std.log.info("TensorClusters: found {} nvfp4, {} fp8, {} mxfp4, {} mxfp8, {} int8_convrot clusters", .{
        fp4_list.items.len, float8_list.items.len, mxfp4_list.items.len, mxfp8_list.items.len, int8_convrot_list.items.len,
    });

    return GroupResult{
        .fp4_clusters = fp4_list.items,
        .float8_clusters = float8_list.items,
        .mxfp4_clusters = mxfp4_list.items,
        .mxfp8_clusters = mxfp8_list.items,
        .int8_convrot_clusters = int8_convrot_list.items,
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

    const w_file = try source.openFileForTensor(cluster.weight.name);
    const weight_bytes = try allocator.alloc(u8, cluster.weight.size);
    defer allocator.free(weight_bytes);
    _ = try w_file.readPositionalAll(source.io, weight_bytes, cluster.weight.offset + source.current_data_begin);

    const ws_file = try source.openFileForTensor(cluster.weight_scale.name);
    const scale_bytes = try allocator.alloc(u8, cluster.weight_scale.size);
    defer allocator.free(scale_bytes);
    _ = try ws_file.readPositionalAll(source.io, scale_bytes, cluster.weight_scale.offset + source.current_data_begin);

    const ws2_file = try source.openFileForTensor(cluster.weight_scale_2.name);
    var gs_buf: [4]u8 = undefined;
    _ = try ws2_file.readPositionalAll(source.io, &gs_buf, cluster.weight_scale_2.offset + source.current_data_begin);
    const global_scale: f32 = @bitCast(std.mem.readInt(u32, gs_buf[0..4], .little));

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
    const w_file = try source.openFileForTensor(cluster.weight.name);
    const weight_bytes = try allocator.alloc(u8, cluster.weight.size);
    defer allocator.free(weight_bytes);
    _ = try w_file.readPositionalAll(source.io, weight_bytes, cluster.weight.offset + source.current_data_begin);

    const ws_file = try source.openFileForTensor(cluster.weight_scale.name);
    const scale_buf = try allocator.alloc(u8, cluster.weight_scale.size);
    defer allocator.free(scale_buf);
    _ = try ws_file.readPositionalAll(source.io, scale_buf, cluster.weight_scale.offset + source.current_data_begin);
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

    const w_file = try source.openFileForTensor(cluster.weight.name);
    const weight_bytes = try allocator.alloc(u8, cluster.weight.size);
    defer allocator.free(weight_bytes);
    _ = try w_file.readPositionalAll(source.io, weight_bytes, cluster.weight.offset + source.current_data_begin);

    const ws_file = try source.openFileForTensor(cluster.weight_scale.name);
    const scale_bytes = try allocator.alloc(u8, cluster.weight_scale.size);
    defer allocator.free(scale_bytes);
    _ = try ws_file.readPositionalAll(source.io, scale_bytes, cluster.weight_scale.offset + source.current_data_begin);

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

    const w_file = try source.openFileForTensor(cluster.weight.name);
    const weight_bytes = try allocator.alloc(u8, cluster.weight.size);
    defer allocator.free(weight_bytes);
    _ = try w_file.readPositionalAll(source.io, weight_bytes, cluster.weight.offset + source.current_data_begin);

    const ws_file = try source.openFileForTensor(cluster.weight_scale.name);
    const scale_bytes = try allocator.alloc(u8, cluster.weight_scale.size);
    defer allocator.free(scale_bytes);
    _ = try ws_file.readPositionalAll(source.io, scale_bytes, cluster.weight_scale.offset + source.current_data_begin);

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
    const w_file = try source.openFileForTensor(cluster.weight.name);
    const weight_bytes = try allocator.alloc(u8, cluster.weight.size);
    defer allocator.free(weight_bytes);
    _ = try w_file.readPositionalAll(source.io, weight_bytes, cluster.weight.offset + source.current_data_begin);

    const ws_file = try source.openFileForTensor(cluster.weight_scale.name);
    const scale_raw = try allocator.alloc(u8, cluster.weight_scale.size);
    defer allocator.free(scale_raw);
    _ = try ws_file.readPositionalAll(source.io, scale_raw, cluster.weight_scale.offset + source.current_data_begin);
    const scale_f32: []const f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(scale_raw)));

    return dequantizeInt8ConvrotRaw(weight_bytes, scale_f32, cluster.rows, cluster.cols, cluster.convrot, cluster.group_size, allocator, pool);
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
    return null;
}

/// Replace cluster physical tensors in `model_tensors` with single logical BF16 tensors.
/// Companion tensors (weight_scale, weight_scale_2, comfy_quant) are removed from the list.
pub fn collapseModelTensors(
    model_tensors: *std.ArrayList(types.Tensor),
    groups: *const GroupResult,
    arena_alloc: std.mem.Allocator,
) !void {
    if (groups.fp4_clusters.len == 0 and groups.float8_clusters.len == 0 and
        groups.mxfp4_clusters.len == 0 and groups.mxfp8_clusters.len == 0 and
        groups.int8_convrot_clusters.len == 0) return;

    var new_tensors: std.ArrayList(types.Tensor) = .empty;

    for (model_tensors.items) |t| {
        var handled = false;

        for (groups.fp4_clusters) |cluster| {
            if (nameSuffixMatch(cluster.weight.name, t.name)) {
                var new_t = t;
                new_t.dims = try arena_alloc.dupe(usize, &[_]usize{ cluster.rows, cluster.cols });
                new_t.type = "BF16";
                new_t.size = cluster.rows * cluster.cols * 2;
                try new_tensors.append(arena_alloc,new_t);
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
                    try new_tensors.append(arena_alloc,new_t);
                    handled = true;
                    break;
                }
                const input_scale_match = if (cluster.input_scale) |is|
                    nameSuffixMatch(is.name, t.name)
                else
                    false;
                if (nameSuffixMatch(cluster.weight_scale.name, t.name) or
                    nameSuffixMatch(cluster.comfy_quant.name, t.name) or
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
                    new_t.type = "BF16";
                    new_t.size = cluster.rows * cluster.cols * 2;
                    try new_tensors.append(arena_alloc,new_t);
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

        if (!handled) {
            for (groups.mxfp8_clusters) |cluster| {
                if (nameSuffixMatch(cluster.weight.name, t.name)) {
                    var new_t = t;
                    new_t.type = "BF16";
                    var n: usize = 1;
                    for (t.dims) |d| n *= d;
                    new_t.size = n * 2;
                    try new_tensors.append(arena_alloc,new_t);
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

        if (!handled) {
            for (groups.int8_convrot_clusters) |cluster| {
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
                    nameSuffixMatch(cluster.comfy_quant.name, t.name))
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
