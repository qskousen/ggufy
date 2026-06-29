const std = @import("std");
const types = @import("types.zig");
const gguf = @import("Gguf.zig");
const DataTransform = @import("DataTransform.zig");
const cb = @import("callbacks.zig");
const ScaledQuant = @import("TensorClusters.zig");
const Convert = @import("Convert.zig");
const thread_pool_mod = @import("ThreadPool.zig");

path: []const u8,
io: std.Io,
allocator: std.mem.Allocator,
arena_alloc: std.mem.Allocator,

// We'll store parsed data here
json_data: ?std.json.Parsed(std.json.Value),
tensors: std.ArrayList(types.Tensor),
metadata: ?std.json.ObjectMap = null,

current_file_handle: ?std.Io.File = null,
current_open_path: []const u8 = "",
current_data_begin: u64 = 0,

const Safetensors = @This();

/// Opens a safetensors file or directory for reading or writing. `target` indicates the file will be opened for read/write.
/// Writing only supports a file target, not a directory.
pub fn init(path: []const u8, io: std.Io, allocator: std.mem.Allocator, arena_alloc: std.mem.Allocator, target: bool, overwrite: bool) !Safetensors {
    // 1. Detect if directory or file
    // 2. If directory (or index file), handle sharding logic
    // 3. Parse headers and populate tensors list

    var self = Safetensors{
        .path = path,
        .io = io,
        .allocator = allocator,
        .arena_alloc = arena_alloc,
        .json_data = null,
        .tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, 200),
        .metadata = null,
        .current_file_handle = null,
        .current_open_path = "",
        .current_data_begin = 0,
    };

    if (target) {
        // handle opening a target file
        var file: std.Io.File = undefined;

        if (overwrite) {
            file = try std.Io.Dir.cwd().createFile(io, path, .{ .read = true, .truncate = true });
        } else {
            // will return an error if file exists already
            // TODO: test if this is true
            file = try std.Io.Dir.cwd().createFile(io, path, .{ .read = true });
        }

        self.current_file_handle = file;
        self.current_open_path = path;
        return self;
    }

    // Determine effective path (if directory provided, look for index or single file)
    // For simplicity, let's assume 'path' is the entry point file (either .safetensors or .index.json)
    // or the directory containing them.

    var entry_path = path;
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (stat.kind == .directory) {
        // Look for index.json (model.safetensors.index.json first, then diffusion_pytorch_model)
        // These candidate paths are arena-allocated: the selected one becomes entry_path
        // (used throughout parsing) and the rest are reclaimed when the arena is freed.
        const paths_index = [_][]const u8{ path, "model.safetensors.index.json" };
        const index_path = try std.fs.path.join(arena_alloc, &paths_index);
        if (std.Io.Dir.cwd().access(io, index_path, .{})) {
            entry_path = index_path;
        } else |_| {
            // Look for model.safetensors
            const paths_single = [_][]const u8{ path, "model.safetensors" };
            const single_path = try std.fs.path.join(arena_alloc, &paths_single);
            if (std.Io.Dir.cwd().access(io, single_path, .{})) {
                entry_path = single_path;
            } else |_| {
                // Look for diffusion_pytorch_model.safetensors.index.json
                const dp_index_path = try std.fs.path.join(arena_alloc, &[_][]const u8{ path, "diffusion_pytorch_model.safetensors.index.json" });
                if (std.Io.Dir.cwd().access(io, dp_index_path, .{})) {
                    entry_path = dp_index_path;
                } else |_| {
                    // Look for diffusion_pytorch_model.safetensors
                    const dp_single_path = try std.fs.path.join(arena_alloc, &[_][]const u8{ path, "diffusion_pytorch_model.safetensors" });
                    if (std.Io.Dir.cwd().access(io, dp_single_path, .{})) {
                        entry_path = dp_single_path;
                    } else |_| {
                        return error.ModelNotFound;
                    }
                }
            }
        }
    }

    if (std.mem.endsWith(u8, entry_path, "index.json")) {
        try self.loadSharded(entry_path);
    } else {
        try self.loadSingle(entry_path);
    }

    return self;
}

pub fn deinit(self: *Safetensors) void {
    if (self.current_file_handle) |h| h.close(self.io);
    if (self.json_data) |*jd| jd.deinit();
    // tensors and metadata are in an arena allocator, so we don't need to free them specifically
}

/// Returns source metadata as an optional, matching the Gguf convention.
pub fn getSourceMetadata(self: Safetensors) ?std.json.ObjectMap {
    return self.metadata;
}

fn loadSingle(self: *Safetensors, path: []const u8) !void {
    const file = try std.Io.Dir.cwd().openFile(self.io, path, .{ .mode = .read_only });
    defer file.close(self.io);
    var read_buffer: [1024]u8 = undefined;
    var reader = file.reader(self.io, &read_buffer);

    const len = try reader.interface.readAlloc(self.allocator, 8);
    defer self.allocator.free(len);
    const header_len = std.mem.readInt(u64, len[0..8], .little);
    self.current_data_begin = header_len + 8;

    const header_bytes = try reader.interface.readAlloc(self.allocator, header_len);

    self.json_data = try std.json.parseFromSlice(std.json.Value, self.allocator, header_bytes, .{});
    self.allocator.free(header_bytes); // Safe because parseFromSlice copies strings by default

    const root = self.json_data.?.value.object;
    if (root.get("__metadata__")) |m| {
        self.metadata = m.object;
    }

    try self.extractTensorsFromObject(root, path);
}

fn loadSharded(self: *Safetensors, index_path: []const u8) !void {
    const dir = std.fs.path.dirname(index_path) orelse ".";

    // Read index file
    const file = try std.Io.Dir.cwd().openFile(self.io, index_path, .{});
    defer file.close(self.io);
    var index_read_buf: [4096]u8 = undefined;
    var index_reader = file.reader(self.io, &index_read_buf);
    const content = try index_reader.interface.allocRemaining(self.allocator, .unlimited);
    defer self.allocator.free(content);

    const index_json = try std.json.parseFromSlice(std.json.Value, self.allocator, content, .{});
    defer index_json.deinit();

    // We need to store *some* metadata, ideally from the first shard found or config?
    // Usually one of the shards has the __metadata__ field.
    // We will lazy-load shards.

    var filenames = try std.ArrayList([]const u8).initCapacity(self.allocator, 3);
    defer {
        for (filenames.items) |n| self.allocator.free(n);
        filenames.deinit(self.allocator);
    }

    if (index_json.value.object.get("weight_map")) |weight_map| {
        var it = weight_map.object.iterator();
        while (it.next()) |entry| {
            const fname = entry.value_ptr.string;
            var exists = false;
            for (filenames.items) |e| {
                if (std.mem.eql(u8, e, fname)) {
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                try filenames.append(self.allocator, try self.allocator.dupe(u8, fname));
            }
        }
    }

    var first = true;
    for (filenames.items) |fname| {
        const paths = [_][]const u8{ dir, fname };
        const full_path = try std.fs.path.join(self.allocator, &paths);
        defer self.allocator.free(full_path);

        const shard_file = try std.Io.Dir.cwd().openFile(self.io, full_path, .{});
        defer shard_file.close(self.io);
        var read_buffer: [1024]u8 = undefined;
        var reader = shard_file.reader(self.io, &read_buffer);

        const len = try reader.interface.readAlloc(self.allocator, 8);
        defer self.allocator.free(len);
        const header_len = std.mem.readInt(u64, len[0..8], .little);
        self.current_data_begin = header_len + 8;
        const header_bytes = try reader.interface.readAlloc(self.allocator, header_len);

        const shard_json = try std.json.parseFromSlice(std.json.Value, self.allocator, header_bytes, .{});
        self.allocator.free(header_bytes);

        if (first) {
            self.json_data = shard_json; // Keep ownership of the first one for metadata/structure
            if (shard_json.value.object.get("__metadata__")) |m| {
                self.metadata = m.object;
            }
            first = false;
            try self.extractTensorsFromObject(self.json_data.?.value.object, full_path);
        } else {
            defer shard_json.deinit();
            if (self.metadata == null) {
                if (shard_json.value.object.get("__metadata__")) |m| {
                    _ = m; // metadata in non-first shard; copy not implemented yet
                }
            }
            try self.extractTensorsFromObject(shard_json.value.object, full_path);
            // defer shard_json.deinit() runs here, after extractTensorsFromObject copied strings
        }
    }
}

fn extractTensorsFromObject(self: *Safetensors, root: std.json.ObjectMap, source_path: []const u8) !void {
    var it = root.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

        const name = try self.arena_alloc.dupe(u8, entry.key_ptr.*);

        const obj = entry.value_ptr.object;
        const dtype_str = obj.get("dtype").?.string;
        const dtype = try DType.fromString(dtype_str);

        const shape = obj.get("shape").?.array;
        var dims = try self.arena_alloc.alloc(usize, shape.items.len);
        errdefer self.arena_alloc.free(dims);
        for (shape.items, 0..) |item, i| {
            dims[i] = @intCast(item.integer);
        }

        const offsets = obj.get("data_offsets").?.array;
        const start = @as(u64, @intCast(offsets.items[0].integer));
        const end = @as(u64, @intCast(offsets.items[1].integer));

        try self.tensors.append(self.arena_alloc, .{
            .name = name,
            .type = @tagName(dtype),
            .dims = dims,
            .size = end - start,
            .offset = start,
            .source_path = try self.arena_alloc.dupe(u8, source_path),
        });
    }
}

/// Opens (or reuses) the file for the named tensor and returns the file handle.
/// After this call, `self.current_data_begin` is set for the opened file.
/// Callers should use `file.readPositionalAll(self.io, buf, self.current_data_begin + tensor.offset)`
/// to read tensor data at the correct position.
pub fn openFileForTensor(self: *Safetensors, name: []const u8) !std.Io.File {
    for (self.tensors.items) |t| {
        if (std.mem.eql(u8, t.name, name)) {
            const tensor_path = t.source_path orelse self.path;

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

pub fn printMetadata(self: Safetensors, writer: *std.Io.Writer) !void {
    if (self.metadata) |meta| {
        var it = meta.iterator();
        while (it.next()) |entry| {
            try writer.print("{s}: ", .{entry.key_ptr.*});
            switch (entry.value_ptr.*) {
                .string => |str| {
                    // Try to pretty print if it looks like JSON
                    if (str.len > 0 and (str[0] == '{' or str[0] == '[')) {
                        if (std.json.parseFromSlice(std.json.Value, self.allocator, str, .{})) |nested_json| {
                            defer nested_json.deinit();
                            var w: std.json.Stringify = .{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
                            try w.write(nested_json.value);
                        } else |_| {
                            try writer.print("{s}", .{str});
                        }
                    } else {
                        try writer.print("{s}", .{str});
                    }
                },
                else => {
                    var w: std.json.Stringify = .{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
                    try w.write(entry.value_ptr.*);
                },
            }
            try writer.writeAll("\n");
        }
    } else {
        try writer.print("No metadata found.\n", .{});
    }
}

// Define a node structure that can represent our hierarchical data
pub const TensorNode = struct {
    name: []const u8,
    children: std.StringHashMap(*TensorNode),
    parent: ?*TensorNode = null,
    // Tensor metadata if this is a leaf node
    dtype: ?DType = null,
    shape: ?[]usize = null,
    data_offsets: ?[2]usize = null,

    pub fn init(allocator: std.mem.Allocator, name: []const u8) !*TensorNode {
        const node = try allocator.create(TensorNode);
        node.* = .{
            .name = name,
            .children = std.StringHashMap(*TensorNode).init(allocator),
            .parent = null,
        };
        return node;
    }

    pub fn deinit(self: *TensorNode, allocator: std.mem.Allocator) void {
        var it = self.children.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit(allocator);
        }
        self.children.deinit();
        allocator.destroy(self);
    }

    // Helper function to get full path from root
    pub fn getFullPath(self: *TensorNode, allocator: std.mem.Allocator) ![]const u8 {
        var list: std.ArrayList([]const u8) = .empty;
        defer list.deinit(allocator);

        var current: ?*TensorNode = self;
        while (current) |node| : (current = node.parent) {
            try list.append(allocator, node.name);
        }

        // Now join the parts in reverse order
        var result: std.ArrayList(u8) = .empty;
        var i: usize = list.items.len;
        while (i > 0) {
            i -= 1;
            if (i < list.items.len - 1) {
                try result.appendSlice(allocator, ".");
            }
            try result.appendSlice(allocator, list.items[i]);
        }

        return result.toOwnedSlice(allocator);
    }

    pub fn getSize(self: *TensorNode) ?usize {
        // Only leaf nodes with dtype and shape can have a size
        if (self.children.count() > 0 or self.dtype == null or self.shape == null) {
            return null;
        }

        const shape = self.shape.?;

        // Calculate total number of elements
        var total_elements: usize = 1;
        for (shape) |dim| {
            total_elements *= dim;
        }

        return self.dtype.?.calcSizeInBytes(@intCast(total_elements));
    }
};

pub const DType = enum {
    F8_E4M3,
    F8_E5M2,
    SCALED_F8_E4M3,
    F4_E2M1,
    MXFP4,
    MXFP8_E4M3,
    NVFP4,
    BF16,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,

    pub fn fromString(str: []const u8) !DType {
        // Convert to uppercase for case-insensitive comparison
        var buf: [20]u8 = undefined;
        if (str.len > buf.len) return error.UnknownDType;
        const upper = std.ascii.upperString(&buf, str);

        inline for (std.meta.fields(DType)) |field| {
            if (std.mem.eql(u8, upper, field.name)) {
                return @field(DType, field.name);
            }
        }
        std.log.debug("Unknown dtype: {s}", .{str});
        return error.UnknownDType;
    }

    pub fn getSizeInBytes(self: DType) usize {
        return switch (self) {
            .BF16, .F16 => 2,
            .F32, .I32, .U32 => 4,
            .F64, .I64, .U64 => 8,
            .I8, .U8, .F8_E4M3, .F8_E5M2, .SCALED_F8_E4M3 => 1,
            .I16, .U16 => 2,
            .F4_E2M1, .MXFP4 => 1, // sub-byte: use calcSizeInBytes(n) for accurate byte counts
            .MXFP8_E4M3 => 1,
            .NVFP4 => 1,
        };
    }

    /// Returns byte size for n elements, correctly handling sub-byte types like F4_E2M1/MXFP4.
    pub fn calcSizeInBytes(self: DType, n: u64) u64 {
        return switch (self) {
            .F4_E2M1, .MXFP4, .NVFP4 => (n + 1) / 2,
            .MXFP8_E4M3 => n,
            else => self.getSizeInBytes() * n,
        };
    }
};

pub fn buildTensorTree(self: Safetensors) !*TensorNode {
    const root = try TensorNode.init(self.arena_alloc, "root");
    var json_data = try self.parseHeader();
    defer json_data.deinit();

    var it = json_data.value.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        // Skip metadata entry
        if (std.mem.eql(u8, key, "__metadata__")) continue;

        // Split the key on dots
        var parts = std.mem.splitAny(u8, key, ".");
        //var parts = std.mem.split(u8, key, ".");
        var current = root;

        // Build the tree structure
        while (parts.next()) |part| {
            const node = try current.children.getOrPut(part);
            if (!node.found_existing) {
                node.value_ptr.* = try TensorNode.init(self.arena_alloc, part);
                node.value_ptr.*.parent = current;
            }
            current = node.value_ptr.*;
        }

        // Add tensor metadata to the leaf node
        if (entry.value_ptr.*.object.get("dtype")) |dtype| {
            current.dtype = try DType.fromString(dtype.string);
        }
        if (entry.value_ptr.*.object.get("shape")) |shape| {
            var shape_list = try std.ArrayList(usize).initCapacity(self.arena_alloc, shape.array.items.len);
            for (shape.array.items) |item| {
                try shape_list.append(self.arena_alloc, @intCast(item.integer));
            }
            current.shape = try shape_list.toOwnedSlice(self.arena_alloc);
        }
        if (entry.value_ptr.*.object.get("data_offsets")) |offsets| {
            current.data_offsets = .{
                @intCast(offsets.array.items[0].integer),
                @intCast(offsets.array.items[1].integer),
            };
        }
    }

    return root;
}

pub fn printTensorTree(self: Safetensors, writer: *std.Io.Writer) !void {
    const root = try TensorNode.init(self.arena_alloc, "root");

    for (self.tensors.items) |t| {
        var parts = std.mem.splitAny(u8, t.name, ".");
        var current = root;
        while (parts.next()) |part| {
            const node = try current.children.getOrPut(part);
            if (!node.found_existing) {
                node.value_ptr.* = try TensorNode.init(self.arena_alloc, part);
                node.value_ptr.*.parent = current;
            }
            current = node.value_ptr.*;
        }

        // Populate leaf
        current.dtype = try DType.fromString(t.type);
        current.shape = t.dims; // Note: referencing slice in `t`, careful if t moves/reallocs?
        // ArrayList reallocs invalidate pointers, but slices point to heap?
        // t.dims is []usize allocated on heap. It's safe as long as we don't free t.
        current.data_offsets = .{ t.offset, t.offset + t.size };
    }

    try self.printNode(root, writer, 0);
}

fn printNode(self: Safetensors, node: *TensorNode, writer: *std.Io.Writer, depth: usize) !void {
    // Print indentation
    var j: usize = 0;
    while (j < depth * 2) : (j += 1) {
        try writer.writeByte(' ');
    }

    // Print node name
    try writer.print("{s}", .{node.name});

    // Print metadata if this is a leaf node (has no children)
    if (node.children.count() == 0) {
        if (node.dtype) |dt| {
            try writer.print(" (dtype: {}", .{dt});
            if (node.shape) |shape| {
                try writer.print(", shape: [", .{});
                for (shape, 0..) |dim, i| {
                    if (i > 0) try writer.print(", ", .{});
                    try writer.print("{}", .{dim});
                }
                try writer.print("]", .{});
            }
            if (node.data_offsets) |offsets| {
                try writer.print(", offsets: [{}, {}]", .{ offsets[0], offsets[1] });
            }
            if (node.getSize()) |size| {
                try writer.print(", size: {}", .{size});
            }
            try writer.print(")", .{});
        }
    }
    try writer.print("\n", .{});

    // Sort children keys for consistent output
    var keys = try std.ArrayList([]const u8).initCapacity(self.allocator, node.children.count());
    defer keys.deinit(self.allocator);

    var it = node.children.iterator();
    while (it.next()) |entry| {
        try keys.append(self.allocator, entry.key_ptr.*);
    }

    std.sort.block([]const u8, keys.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);

    // Print all children in sorted order
    for (keys.items) |key| {
        if (node.children.get(key)) |child| {
            try self.printNode(child, writer, depth + 1);
        }
    }
}

pub fn printHeader(self: Safetensors, writer: *std.Io.Writer) !void {
    var dtype_counts = std.AutoHashMap(DType, usize).init(self.allocator);
    defer dtype_counts.deinit();

    var bad_offset_count: u64 = 0;

    try writer.print("{{\n", .{});
    // do some sanity checks
    for (self.tensors.items, 0..) |t, i| {
        const dt = try DType.fromString(t.type);

        // Offset safety check
        var bad_offset = false;

        // Calculate expected size from dtype + shape
        var n_elements: u64 = 1;
        for (t.dims) |d| n_elements *= d;
        const expected_size: u64 = dt.calcSizeInBytes(n_elements);

        // The stored size (end - start) must match expected
        if (t.size != expected_size) {
            std.log.warn(
                "Tensor {s}: stored size {} does not match expected size {} (dtype={s}, elements={})",
                .{ t.name, t.size, expected_size, t.type, n_elements },
            );
            bad_offset = true;
            bad_offset_count += 1;
        }

        // For non-last tensors: next tensor's offset must immediately follow this one
        if (i < self.tensors.items.len - 1) {
            const next_offset = self.tensors.items[i + 1].offset;
            const allocated_size = std.math.sub(u64, next_offset, t.offset) catch {
                std.log.warn(
                    "Tensor {s}: overflow computing allocated size (offset={}, next_offset={})",
                    .{ t.name, t.offset, next_offset },
                );
                bad_offset = true;
                bad_offset_count += 1;
                continue;
            };
            if (allocated_size != expected_size) {
                std.log.warn(
                    "Tensor {s}: allocated region {} does not match expected size {}",
                    .{ t.name, allocated_size, expected_size },
                );
                bad_offset = true;
                bad_offset_count += 1;
            }
        } else {
            // Last tensor: verify it doesn't exceed the file's data section
            if (self.current_data_begin > 0) {
                const start_pos = std.math.add(u64, self.current_data_begin, t.offset) catch {
                    std.log.warn(
                        "Tensor {s}: overflow computing file start position (data_begin={}, offset={})",
                        .{ t.name, self.current_data_begin, t.offset },
                    );
                    bad_offset = true;
                    bad_offset_count += 1;
                    continue;
                };
                const end_pos = std.math.add(u64, start_pos, expected_size) catch {
                    std.log.warn(
                        "Tensor {s}: overflow computing end position (start={}, expected_size={})",
                        .{ t.name, start_pos, expected_size },
                    );
                    bad_offset = true;
                    bad_offset_count += 1;
                    continue;
                };
                _ = end_pos; // could compare against file size if available
            } else {
                std.log.warn("Tensor {s}: cannot validate last tensor offset, data_begin not set", .{t.name});
            }
        }

        try writer.print("  \"{s}\": {{\n", .{t.name});
        try writer.print("    \"dtype\": \"{s}\",\n", .{t.type});
        try writer.print("    \"shape\": [", .{});
        for (t.dims, 0..) |d, di| {
            if (di > 0) try writer.print(", ", .{});
            try writer.print("{}", .{d});
        }
        try writer.print("],\n", .{});
        try writer.print("    \"offset_from_data_start_and_file_start\": [{}, {}]", .{ t.offset, t.offset + self.current_data_begin });
        if (bad_offset) try writer.print(" <-- BAD OFFSET", .{});
        try writer.print("\n  }}", .{});
        if (i < self.tensors.items.len - 1) try writer.print(",", .{});
        try writer.print("\n", .{});

        const g = try dtype_counts.getOrPut(dt);
        if (!g.found_existing) g.value_ptr.* = 0;
        g.value_ptr.* += 1;
    }
    try writer.print("}}\n", .{});

    if (dtype_counts.count() > 0) {
        try writer.print("\n\nTensor Type Statistics:\n", .{});
        var stats_it = dtype_counts.iterator();
        while (stats_it.next()) |entry| {
            try writer.print("  {s}: {}\n", .{ @tagName(entry.key_ptr.*), entry.value_ptr.* });
        }
    }

    if (bad_offset_count > 0) {
        try writer.print("\nTensors with bad offsets/sizes: {}\n", .{bad_offset_count});
    }
}

pub fn getTensors(self: Safetensors) !std.ArrayList(types.Tensor) {
    const json_data = try self.parseHeader();
    defer json_data.deinit();

    var tensors = try std.ArrayList(types.Tensor).initCapacity(self.arena_alloc, 200);

    var it = json_data.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

        const name = try self.arena_alloc.dupe(u8, entry.key_ptr.*);

        const obj = entry.value_ptr.object;
        const dtype_str = obj.get("dtype").?.string;
        const dtype = try DType.fromString(dtype_str);

        const shape = obj.get("shape").?.array;
        var dims = try self.arena_alloc.alloc(usize, shape.items.len);
        for (shape.items, 0..) |item, i| {
            dims[i] = @intCast(item.integer);
        }

        const offsets = obj.get("data_offsets").?.array;
        const start = @as(u64, @intCast(offsets.items[0].integer));
        const end = @as(u64, @intCast(offsets.items[1].integer));

        try tensors.append(self.arena_alloc, .{
            .name = name,
            .type = @tagName(dtype),
            .dims = dims,
            .size = end - start,
            .offset = start,
        });
    }
    return tensors;
}

pub fn parseHeader(self: Safetensors) !std.json.Parsed(std.json.Value) {
    const file = try std.Io.Dir.cwd().openFile(self.io, self.path, .{ .mode = .read_only });
    defer file.close(self.io);
    var read_buf: [1024]u8 = undefined;
    var reader = file.reader(self.io, &read_buf);

    const len = try reader.interface.readAlloc(self.arena_alloc, 8);
    defer self.arena_alloc.free(len);
    const header_len = std.mem.readInt(u64, len[0..8], .little);

    const data = try reader.interface.readAlloc(self.arena_alloc, header_len);

    return std.json.parseFromSlice(std.json.Value, self.arena_alloc, data, .{});
}

pub fn writeTensorData(
    self: Safetensors,
    t: types.Tensor,
    source_dtype: types.DataType,
    source_data: []const u8,
    writer: *std.Io.Writer,
    pool: *thread_pool_mod.ThreadPool
) !void {
    const target_dtype = try types.DataType.fromString(t.type);

    // Calculate the source tensor size based on source type
    var n_elements: u64 = 1;
    for (t.dims) |d| n_elements *= d;

    // Convert if types differ, otherwise write directly
    if (source_dtype.equivalentType(@tagName(target_dtype))) {
        std.log.debug("Using direct data copy for {s} to {s}.", .{@tagName(source_dtype), @tagName(target_dtype)});
        try writer.writeAll(source_data);
    } else {
        // Use DataTransform to convert the data
        std.log.debug("Converting data from {s} to {s}.", .{@tagName(source_dtype), @tagName(target_dtype)});
        const converted_data = try DataTransform.Quantizer.convertTensorData(
            self.allocator,
            source_data,
            source_dtype,
            target_dtype,
            n_elements,
            pool,
        );
        defer self.allocator.free(converted_data);

        try writer.writeAll(converted_data);
    }
}

pub fn saveWithSTData(self: Safetensors, source: anytype, threads: usize, callbacks: cb.ConvertCallbacks, groups: *const ScaledQuant.GroupResult) !void {
    // Build the full header JSON object (tensor entries + __metadata__)
    var header_obj = std.json.ObjectMap.empty;

    // Add __metadata__ under its special key
    if (self.metadata) |meta| {
        try header_obj.put(self.arena_alloc,"__metadata__", .{ .object = meta });
    }

    // Add each tensor entry
    for (self.tensors.items) |t| {
        // SCALED_F8_E4M3: expand into a 3-tensor ComfyUI FP8 cluster.
        // Region layout: [F8 weight bytes][F32 scale 4 bytes][comfy_quant JSON]
        // The on-disk dtype for weight and sidecar tensors uses "F8_E4M3" (ComfyUI compatibility).
        if (std.mem.eql(u8, t.type, "SCALED_F8_E4M3")) {
            var n_elements: u64 = 1;
            for (t.dims) |d| n_elements *= d;
            const comfy_json = Convert.fp8_comfy_json;

            const scale_start = t.offset + n_elements;
            const comfy_start = scale_start + 4;

            const weight_suffix = ".weight";
            const base = if (std.mem.endsWith(u8, t.name, weight_suffix))
                t.name[0 .. t.name.len - weight_suffix.len]
            else
                t.name;

            // weight: dtype F8_E4M3, original shape
            {
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "F8_E4M3" });
                var shape = std.json.Array.init(self.arena_alloc);
                for (t.dims) |d| try shape.append(.{ .integer = @intCast(d) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(t.offset) });
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,t.name, .{ .object = obj });
            }
            // weight_scale: dtype F32, shape []
            {
                const sname = try std.fmt.allocPrint(self.arena_alloc, "{s}.weight_scale", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "F32" });
                try obj.put(self.arena_alloc,"shape", .{ .array = std.json.Array.init(self.arena_alloc) });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,sname, .{ .object = obj });
            }
            // comfy_quant: dtype U8, shape [27]
            {
                const cname = try std.fmt.allocPrint(self.arena_alloc, "{s}.comfy_quant", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(comfy_json.len) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start + comfy_json.len) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,cname, .{ .object = obj });
            }
            continue;
        }

        // MXFP4 safetensors: expand into a 3-tensor ComfyUI cluster.
        // The region [t.offset, t.offset+t.size) is split into:
        //   [weight nibbles][E8M0 scale bytes][comfy_quant JSON]
        if (std.mem.eql(u8, t.type, "MXFP4")) {
            const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
            var n_rows: u64 = 1;
            if (t.dims.len >= 2) {
                for (t.dims[0 .. t.dims.len - 1]) |d| n_rows *= d;
            }
            const weight_bytes: u64 = n_rows * n_cols / 2;
            const scale_bytes:  u64 = n_rows * ((n_cols + 31) / 32);  // U8 E8M0, 1 byte each
            const comfy_json = Convert.mxfp4_comfy_json;

            const scale_start = t.offset + weight_bytes;
            const comfy_start = scale_start + scale_bytes;

            const weight_suffix = ".weight";
            const base = if (std.mem.endsWith(u8, t.name, weight_suffix))
                t.name[0 .. t.name.len - weight_suffix.len]
            else
                t.name;

            // weight: dtype U32, shape [n_rows, n_cols/8]  (8 nibbles per U32)
            {
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U32" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(n_rows) });
                try shape.append(.{ .integer = @intCast(n_cols / 8) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(t.offset) });
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,t.name, .{ .object = obj });
            }
            // weight_scale: dtype U8 E8M0, shape [n_rows, n_cols/32]
            {
                const sname = try std.fmt.allocPrint(self.arena_alloc, "{s}.weight_scale", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(n_rows) });
                try shape.append(.{ .integer = @intCast((n_cols + 31) / 32) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,sname, .{ .object = obj });
            }
            // comfy_quant: dtype U8, shape [len(json)]
            {
                const cname = try std.fmt.allocPrint(self.arena_alloc, "{s}.comfy_quant", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(comfy_json.len) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start + comfy_json.len) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,cname, .{ .object = obj });
            }
            continue;
        }

        // MXFP8_E4M3 safetensors: expand into a 3-tensor ComfyUI cluster.
            // The region [t.offset, t.offset+t.size) is split into:
            //   [F8_E4M3 weight bytes][E8M0 scale bytes][comfy_quant JSON]
            if (std.mem.eql(u8, t.type, "MXFP8_E4M3")) {
            const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
            var n_rows: u64 = 1;
            if (t.dims.len >= 2) {
                for (t.dims[0 .. t.dims.len - 1]) |d| n_rows *= d;
            }
            const weight_bytes: u64 = n_rows * n_cols;  // 1 byte per F8_E4M3 element
                // cuBLAS blocked scale: padded to [n_row_blocks*128, n_col_blocks*4]
                const n_scale_cols_hdr: u64 = (n_cols + 31) / 32;
                const n_row_blocks_hdr: u64 = (n_rows + 127) / 128;
                const n_col_blocks_hdr: u64 = (n_scale_cols_hdr + 3) / 4;
                const scale_bytes:  u64 = n_row_blocks_hdr * 128 * n_col_blocks_hdr * 4;
                const comfy_json = Convert.mxfp8_comfy_json;

            const scale_start = t.offset + weight_bytes;
            const comfy_start = scale_start + scale_bytes;

            const weight_suffix = ".weight";
            const base = if (std.mem.endsWith(u8, t.name, weight_suffix))
                t.name[0 .. t.name.len - weight_suffix.len]
            else
                t.name;

            // weight: dtype F8_E4M3, original shape
                {
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "F8_E4M3" });
                var shape = std.json.Array.init(self.arena_alloc);
                for (t.dims) |d| try shape.append(.{ .integer = @intCast(d) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(t.offset) });
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,t.name, .{ .object = obj });
            }
            // weight_scale: dtype U8 E8M0, cuBLAS blocked shape [n_row_blocks*128, n_col_blocks*4]
                {
                const sname = try std.fmt.allocPrint(self.arena_alloc, "{s}.weight_scale", .{base});
                const n_scale_cols: u64 = (n_cols + 31) / 32;
                const n_row_blocks: u64 = (n_rows + 127) / 128;
                const n_col_blocks: u64 = (n_scale_cols + 3) / 4;
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(n_row_blocks * 128) });
                try shape.append(.{ .integer = @intCast(n_col_blocks * 4) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,sname, .{ .object = obj });
            }
            // comfy_quant: dtype U8, shape [len(json)]
                {
                const cname = try std.fmt.allocPrint(self.arena_alloc, "{s}.comfy_quant", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(comfy_json.len) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start + comfy_json.len) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,cname, .{ .object = obj });
            }
            continue;
        }

        // NVFP4 safetensors: expand into a 4-tensor ComfyUI cluster.
        // The region [t.offset, t.offset+t.size) is split into:
        //   [weight nibbles][F8_E4M3 scale cuBLAS-tiled][F32 global scale][comfy_quant JSON]
        if (std.mem.eql(u8, t.type, "NVFP4")) {
            const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
            var n_rows: u64 = 1;
            if (t.dims.len >= 2) {
                for (t.dims[0 .. t.dims.len - 1]) |d| n_rows *= d;
            }
            const weight_bytes: u64 = n_rows * (n_cols / 2);
            const scale_bytes:  u64 = n_rows * (n_cols / 16);  // F8_E4M3, cuBLAS-tiled
            const gs_bytes:     u64 = 4;  // F32 global scalar
            const comfy_json = Convert.nvfp4_comfy_json;

            const scale_start  = t.offset + weight_bytes;
            const gs_start     = scale_start + scale_bytes;
            const comfy_start  = gs_start + gs_bytes;

            const weight_suffix = ".weight";
            const base = if (std.mem.endsWith(u8, t.name, weight_suffix))
                t.name[0 .. t.name.len - weight_suffix.len]
            else
                t.name;

            // weight: dtype U8, shape [n_rows, n_cols/2]  (packed nibbles)
            {
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(n_rows) });
                try shape.append(.{ .integer = @intCast(n_cols / 2) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(t.offset) });
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,t.name, .{ .object = obj });
            }
            // weight_scale: dtype F8_E4M3, shape [n_rows, n_cols/16]  (cuBLAS-tiled)
            {
                const sname = try std.fmt.allocPrint(self.arena_alloc, "{s}.weight_scale", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "F8_E4M3" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(n_rows) });
                try shape.append(.{ .integer = @intCast(n_cols / 16) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(scale_start) });
                try offsets.append(.{ .integer = @intCast(gs_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,sname, .{ .object = obj });
            }
            // weight_scale_2: dtype F32, shape []  (global scalar)
            {
                const s2name = try std.fmt.allocPrint(self.arena_alloc, "{s}.weight_scale_2", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "F32" });
                try obj.put(self.arena_alloc,"shape", .{ .array = std.json.Array.init(self.arena_alloc) });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(gs_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,s2name, .{ .object = obj });
            }
            // comfy_quant: dtype U8, shape [len(json)]
            {
                const cname = try std.fmt.allocPrint(self.arena_alloc, "{s}.comfy_quant", .{base});
                var obj = std.json.ObjectMap.empty;
                try obj.put(self.arena_alloc,"dtype", .{ .string = "U8" });
                var shape = std.json.Array.init(self.arena_alloc);
                try shape.append(.{ .integer = @intCast(comfy_json.len) });
                try obj.put(self.arena_alloc,"shape", .{ .array = shape });
                var offsets = std.json.Array.init(self.arena_alloc);
                try offsets.append(.{ .integer = @intCast(comfy_start) });
                try offsets.append(.{ .integer = @intCast(comfy_start + comfy_json.len) });
                try obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets });
                try header_obj.put(self.arena_alloc,cname, .{ .object = obj });
            }
            continue;
        }

        var tensor_obj = std.json.ObjectMap.empty;

        try tensor_obj.put(self.arena_alloc,"dtype", .{ .string = t.type });

        var shape_arr = std.json.Array.init(self.arena_alloc);
        for (t.dims) |d| {
            try shape_arr.append(.{ .integer = @intCast(d) });
        }
        try tensor_obj.put(self.arena_alloc,"shape", .{ .array = shape_arr });

        var offsets_arr = std.json.Array.init(self.arena_alloc);
        try offsets_arr.append(.{ .integer = @intCast(t.offset) });
        try offsets_arr.append(.{ .integer = @intCast(t.offset + t.size) });
        try tensor_obj.put(self.arena_alloc,"data_offsets", .{ .array = offsets_arr });

        try header_obj.put(self.arena_alloc,t.name, .{ .object = tensor_obj });
    }

    // Serialize the full header to bytes
    var aw: std.Io.Writer.Allocating = .init(self.allocator);
    defer aw.deinit();

    const header_value = std.json.Value{ .object = header_obj };
    try std.json.Stringify.value(header_value, .{}, &aw.writer);

    const header_bytes = aw.written();
    const header_size: u64 = header_bytes.len;
    std.log.debug("Header size: {}", .{header_size});

    // writer for ourselves
    var write_buffer: [1024 * 1024]u8 = undefined;
    var writer = self.current_file_handle.?.writer(self.io, &write_buffer);

    // write header size
    try writer.interface.writeInt(u64, header_size, .little);
    // write header
    _ = try writer.interface.write(header_bytes);

    var pool: thread_pool_mod.ThreadPool = undefined;
    try pool.init(.{ .allocator = self.allocator, .n_jobs = threads });
    defer pool.deinit();

    // write the tensors

    // we need to write them in the order of our tensors, but there is no guarantee that the source tensors will be in the same order
    // the names might also be different, so we have to do some matching
    const total_tensors: u32 = @intCast(self.tensors.items.len);
    var count: u32 = 1;
    for (self.tensors.items) |t| {
        // Check for cancellation before writing each tensor
        if (callbacks.isCancelled()) return error.Cancelled;

        var elements: usize = 1;
        for (t.dims) |d| elements *= d;

        var matched = false;

        // SCALED_F8_E4M3 cluster output: quantize source data and write weight+scale+comfy_quant
        if (!matched and std.mem.eql(u8, t.type, "SCALED_F8_E4M3")) {
            var n_elements: u64 = 1;
            for (t.dims) |d| n_elements *= d;
            for (source.tensors.items) |source_tensor| {
                const is_match = std.mem.eql(u8, source_tensor.name, t.name) or
                    (source_tensor.name.len > t.name.len and
                     source_tensor.name[source_tensor.name.len - t.name.len - 1] == '.' and
                     std.mem.endsWith(u8, source_tensor.name, t.name));
                if (!is_match) continue;
                matched = true;

                const source_dtype = try types.DataType.fromString(source_tensor.type);
                const source_size: usize = switch (source_dtype.formatType()) {
                    .safetensors => blk: {
                        const stype = try Safetensors.DType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                    .gguf => blk: {
                        const stype = try gguf.GgmlType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                };
                const source_file = try source.openFileForTensor(source_tensor.name);
                const src_bytes = try self.allocator.alloc(u8, source_size);
                defer self.allocator.free(src_bytes);
                _ = try source_file.readPositionalAll(source.io, src_bytes, source_tensor.offset + source.current_data_begin);

                const f32_bytes = try DataTransform.Quantizer.convertTensorData(
                    self.allocator, src_bytes, source_dtype, .F32, n_elements, &pool,
                );
                defer self.allocator.free(f32_bytes);
                const f32_slice: []const f32 = @ptrCast(@alignCast(std.mem.bytesAsSlice(f32, f32_bytes)));

                const cluster = try DataTransform.Quantizer.quantizeToComfyFp8(
                    self.allocator, f32_slice, &pool,
                );
                defer self.allocator.free(cluster.weight);

                var scale_bytes_buf: [4]u8 = undefined;
                std.mem.writeInt(u32, &scale_bytes_buf, @bitCast(cluster.scale), .little);

                std.log.info("Writing FP8 cluster {s} from {s}, scale={d:.6}", .{
                    t.name, source_tensor.type, cluster.scale,
                });
                try (&writer.interface).writeAll(cluster.weight);
                try (&writer.interface).writeAll(&scale_bytes_buf);
                try (&writer.interface).writeAll(Convert.fp8_comfy_json);
                callbacks.reportProgress(count, total_tensors, t.name, source_tensor.type, "F8_E4M3", @intCast(n_elements));
                count += 1;
                break;
            }
        }

        // MXFP4 cluster output: quantize source data and write weight+scale+comfy_quant
        if (!matched and std.mem.eql(u8, t.type, "MXFP4")) {
            const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
            var n_rows: u64 = 1;
            if (t.dims.len >= 2) {
                for (t.dims[0 .. t.dims.len - 1]) |d| n_rows *= d;
            }
            for (source.tensors.items) |source_tensor| {
                const is_match = std.mem.eql(u8, source_tensor.name, t.name) or
                    (source_tensor.name.len > t.name.len and
                     source_tensor.name[source_tensor.name.len - t.name.len - 1] == '.' and
                     std.mem.endsWith(u8, source_tensor.name, t.name));
                if (!is_match) continue;
                matched = true;

                // Read source bytes
                const source_dtype = try types.DataType.fromString(source_tensor.type);
                const n_elements: u64 = n_rows * n_cols;
                const source_size: usize = switch (source_dtype.formatType()) {
                    .safetensors => blk: {
                        const stype = try Safetensors.DType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                    .gguf => blk: {
                        const stype = try gguf.GgmlType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                };
                const source_file = try source.openFileForTensor(source_tensor.name);
                const src_bytes = try self.allocator.alloc(u8, source_size);
                defer self.allocator.free(src_bytes);
                _ = try source_file.readPositionalAll(source.io, src_bytes, source_tensor.offset + source.current_data_begin);

                // Convert to F32
                const f32_bytes = try DataTransform.Quantizer.convertTensorData(
                    self.allocator, src_bytes, source_dtype, .F32, n_elements, &pool,
                );
                defer self.allocator.free(f32_bytes);
                const f32_slice: []const f32 = @ptrCast(@alignCast(std.mem.bytesAsSlice(f32, f32_bytes)));

                // Quantize to ComfyUI MXFP4
                const cluster = try DataTransform.Quantizer.quantizeToComfyMxfp4(
                    self.allocator, f32_slice, &pool,
                );
                defer self.allocator.free(cluster.weight);
                defer self.allocator.free(cluster.scale);

                std.log.info("Writing MXFP4 cluster {s} [{}, {}] from {s}", .{
                    t.name, n_rows, n_cols, source_tensor.type,
                });
                try (&writer.interface).writeAll(cluster.weight);
                try (&writer.interface).writeAll(cluster.scale);
                try (&writer.interface).writeAll(Convert.mxfp4_comfy_json);
                callbacks.reportProgress(count, total_tensors, t.name, source_tensor.type, "MXFP4", @intCast(n_elements));
                count += 1;
                break;
            }
        }

        // MXFP8 cluster output: quantize source data and write weight+scale+comfy_quant
        if (!matched and std.mem.eql(u8, t.type, "MXFP8_E4M3")) {
            const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
            var n_rows: u64 = 1;
            if (t.dims.len >= 2) {
                for (t.dims[0 .. t.dims.len - 1]) |d| n_rows *= d;
            }
            for (source.tensors.items) |source_tensor| {
                const is_match = std.mem.eql(u8, source_tensor.name, t.name) or
                    (source_tensor.name.len > t.name.len and
                        source_tensor.name[source_tensor.name.len - t.name.len - 1] == '.' and
                        std.mem.endsWith(u8, source_tensor.name, t.name));
                if (!is_match) continue;
                matched = true;

                // Read source bytes
                const source_dtype = try types.DataType.fromString(source_tensor.type);
                const n_elements: u64 = n_rows * n_cols;
                const source_size: usize = switch (source_dtype.formatType()) {
                    .safetensors => blk: {
                        const stype = try Safetensors.DType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                    .gguf => blk: {
                        const stype = try gguf.GgmlType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                };
                const source_file = try source.openFileForTensor(source_tensor.name);
                const src_bytes = try self.allocator.alloc(u8, source_size);
                defer self.allocator.free(src_bytes);
                _ = try source_file.readPositionalAll(source.io, src_bytes, source_tensor.offset + source.current_data_begin);

                // Convert to F32
                const f32_bytes = try DataTransform.Quantizer.convertTensorData(
                    self.allocator, src_bytes, source_dtype, .F32, n_elements, &pool,
                );
                defer self.allocator.free(f32_bytes);
                const f32_slice: []const f32 = @ptrCast(@alignCast(std.mem.bytesAsSlice(f32, f32_bytes)));

                // Quantize to ComfyUI MXFP8
                const cluster = try DataTransform.Quantizer.quantizeToComfyMxfp8(
                    self.allocator, f32_slice, &pool,
                );
                defer self.allocator.free(cluster.weight);
                defer self.allocator.free(cluster.scale);

                // Apply cuBLAS blocking to scales before writing
                const n_scale_cols: usize = @intCast((n_cols + 31) / 32);
                const blocked_scale = try DataTransform.Quantizer.toBlockedMxfp8(
                    self.allocator, cluster.scale, @intCast(n_rows), n_scale_cols,
                );
                defer self.allocator.free(blocked_scale);

                std.log.info("Writing MXFP8 cluster {s} [{}, {}] from {s}", .{
                    t.name, n_rows, n_cols, source_tensor.type,
                });
                try (&writer.interface).writeAll(cluster.weight);
                try (&writer.interface).writeAll(blocked_scale);
                try (&writer.interface).writeAll(Convert.mxfp8_comfy_json);
                callbacks.reportProgress(count, total_tensors, t.name, source_tensor.type, "MXFP8_E4M3", @intCast(n_elements));
                count += 1;
                break;
            }
        }

        // NVFP4 cluster output: quantize source data and write weight+scale+global_scale+comfy_quant
        if (!matched and std.mem.eql(u8, t.type, "NVFP4")) {
            const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
            var n_rows: u64 = 1;
            if (t.dims.len >= 2) {
                for (t.dims[0 .. t.dims.len - 1]) |d| n_rows *= d;
            }
            for (source.tensors.items) |source_tensor| {
                const is_match = std.mem.eql(u8, source_tensor.name, t.name) or
                    (source_tensor.name.len > t.name.len and
                     source_tensor.name[source_tensor.name.len - t.name.len - 1] == '.' and
                     std.mem.endsWith(u8, source_tensor.name, t.name));
                if (!is_match) continue;
                matched = true;

                // Read source bytes
                const source_dtype = try types.DataType.fromString(source_tensor.type);
                const n_elements: u64 = n_rows * n_cols;
                const source_size: usize = switch (source_dtype.formatType()) {
                    .safetensors => blk: {
                        const stype = try Safetensors.DType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                    .gguf => blk: {
                        const stype = try gguf.GgmlType.fromString(@tagName(source_dtype));
                        break :blk stype.calcSizeInBytes(n_elements);
                    },
                };
                const source_file = try source.openFileForTensor(source_tensor.name);
                const src_bytes = try self.allocator.alloc(u8, source_size);
                defer self.allocator.free(src_bytes);
                _ = try source_file.readPositionalAll(source.io, src_bytes, source_tensor.offset + source.current_data_begin);

                // Convert to F32
                const f32_bytes = try DataTransform.Quantizer.convertTensorData(
                    self.allocator, src_bytes, source_dtype, .F32, n_elements, &pool,
                );
                defer self.allocator.free(f32_bytes);
                const f32_slice: []const f32 = @ptrCast(@alignCast(std.mem.bytesAsSlice(f32, f32_bytes)));

                // Quantize to NVFP4 cluster
                const cluster = try ScaledQuant.quantizeToNvFp4Raw(
                    f32_slice, n_rows, n_cols, self.allocator, &pool,
                );
                defer self.allocator.free(cluster.weight);
                defer self.allocator.free(cluster.scale);

                var gs_buf: [4]u8 = undefined;
                std.mem.writeInt(u32, &gs_buf, @bitCast(cluster.global_scale), .little);

                std.log.info("Writing NVFP4 cluster {s} [{}, {}] from {s}", .{
                    t.name, n_rows, n_cols, source_tensor.type,
                });
                try (&writer.interface).writeAll(cluster.weight);
                try (&writer.interface).writeAll(cluster.scale);
                try (&writer.interface).writeAll(&gs_buf);
                try (&writer.interface).writeAll(Convert.nvfp4_comfy_json);
                callbacks.reportProgress(count, total_tensors, t.name, source_tensor.type, "NVFP4", @intCast(n_elements));
                count += 1;
                break;
            }
        }

        // Cluster dequantization path: source is an NVFP4/FP8/MX* cluster → dequant to F32
        if (!matched) {
            if (try ScaledQuant.tryDequantCluster(t, source, groups, self.allocator, &pool)) |f32_buf| {
                defer self.allocator.free(f32_buf);
                const target_dtype = try types.DataType.fromString(t.type);
                std.log.info("Writing tensor data for tensor {}/{} {s} - nvfp4/fp8 to {s}, {} elements", .{
                    count, total_tensors, t.name, t.type, elements,
                });
                const out = try DataTransform.Quantizer.convertTensorData(
                    self.allocator,
                    std.mem.sliceAsBytes(f32_buf),
                    .F32,
                    target_dtype,
                    f32_buf.len,
                    &pool,
                );
                defer self.allocator.free(out);
                try (&writer.interface).writeAll(out);
                callbacks.reportProgress(count, total_tensors, t.name, "nvfp4", t.type, @intCast(elements));
                count += 1;
                matched = true;
            }
        }

        if (!matched) {
            // Normal single-tensor path
            for (source.tensors.items) |source_tensor| {
                if (std.mem.eql(u8, source_tensor.name, t.name) or
                    (source_tensor.name.len > t.name.len and
                        source_tensor.name[source_tensor.name.len - t.name.len - 1] == '.' and
                        std.mem.endsWith(u8, source_tensor.name, t.name)))
                    {
                        matched = true;
                        std.log.info("Writing tensor data for tensor {}/{} {s} - {s} to {s}, {} elements", .{
                            count,
                            total_tensors,
                            t.name,
                            source_tensor.type,
                            t.type,
                            elements,
                        });
                        const source_dtype = try types.DataType.fromString(source_tensor.type);
                        var n_elements_st: u64 = 1;
                        for (t.dims) |d| n_elements_st *= d;
                        const source_size_st: usize = switch (source_dtype.formatType()) {
                            .safetensors => blk: {
                                const stype = try Safetensors.DType.fromString(@tagName(source_dtype));
                                break :blk stype.calcSizeInBytes(n_elements_st);
                            },
                            .gguf => blk: {
                                const stype = try gguf.GgmlType.fromString(@tagName(source_dtype));
                                break :blk stype.calcSizeInBytes(n_elements_st);
                            },
                        };
                        const source_file_st = try source.openFileForTensor(source_tensor.name);
                        const src_bytes_st = try self.allocator.alloc(u8, source_size_st);
                        defer self.allocator.free(src_bytes_st);
                        _ = try source_file_st.readPositionalAll(source.io, src_bytes_st, source_tensor.offset + source.current_data_begin);
                        try self.writeTensorData(t, source_dtype, src_bytes_st, &writer.interface, &pool);
                        callbacks.reportProgress(count, total_tensors, t.name, source_tensor.type, t.type, @intCast(elements));
                        count += 1;
                    }
            }
        }

        if (!matched) {
            std.log.warn("Could not find source tensor match for tensor {s}!", .{t.name});
            return error.NoMatchingSourceTensor;
        }
    }

    try writer.interface.flush();
}
