const std = @import("std");
const types = @import("types.zig");

path: []const u8,
allocator: std.mem.Allocator,
arena_alloc: std.mem.Allocator,

// We'll store parsed data here
json_data: std.json.Parsed(std.json.Value),
tensors: std.ArrayList(types.Tensor),
metadata: ?std.json.ObjectMap = null,

current_file_handle: ?std.fs.File = null,
current_open_path: []const u8 = "",
current_data_begin: u64 = 0,

pub const formatType = types.FileType.safetensors;

const Safetensors = @This();

pub fn init(path: []const u8, allocator: std.mem.Allocator, arena_alloc: std.mem.Allocator) !Safetensors {
    // 1. Detect if directory or file
    // 2. If directory (or index file), handle sharding logic
    // 3. Parse headers and populate tensors list

    var self = Safetensors{
        .path = path,
        .allocator = allocator,
        .arena_alloc = arena_alloc,
        .json_data = undefined,
        .tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, 200),
        .metadata = null,
        .current_file_handle = null,
        .current_open_path = "",
        .current_data_begin = 0,
    };

    // Determine effective path (if directory provided, look for index or single file)
    // For simplicity, let's assume 'path' is the entry point file (either .safetensors or .index.json)
    // or the directory containing them.

    var entry_path = path;
    const stat = try std.fs.cwd().statFile(path);
    if (stat.kind == .directory) {
        // Look for index.json
        const paths_index = [_][]const u8{ path, "model.safetensors.index.json" };
        const index_path = try std.fs.path.join(allocator, &paths_index);
        if (std.fs.cwd().access(index_path, .{})) {
            entry_path = index_path;
        } else |_| {
            // Look for model.safetensors
            const paths_single = [_][]const u8{ path, "model.safetensors" };
            const single_path = try std.fs.path.join(allocator, &paths_single);
            if (std.fs.cwd().access(single_path, .{})) {
                entry_path = single_path;
            } else |_| {
                return error.ModelNotFound;
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
    if (self.current_file_handle) |h| h.close();
    self.json_data.deinit();
    // tensors and metadata are in an arena allocator, so we don't need to free them specifically
}

fn loadSingle(self: *Safetensors, path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();
    var read_buffer: [1024]u8 = undefined;
    var reader = file.reader(&read_buffer);

    const len = try reader.interface.readAlloc(self.allocator, 8);
    defer self.allocator.free(len);
    const header_len = std.mem.readInt(u64, len[0..8], .little);

    const header_bytes = try reader.interface.readAlloc(self.allocator, header_len);

    self.json_data = try std.json.parseFromSlice(std.json.Value, self.allocator, header_bytes, .{});
    self.allocator.free(header_bytes); // Safe because parseFromSlice copies strings by default

    const root = self.json_data.value.object;
    if (root.get("__metadata__")) |m| {
        self.metadata = m.object;
    }

    try self.extractTensorsFromObject(root, path);
}

fn loadSharded(self: *Safetensors, index_path: []const u8) !void {
    const dir = std.fs.path.dirname(index_path) orelse ".";

    // Read index file
    const file = try std.fs.cwd().openFile(index_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(self.allocator, 10 * 1024 * 1024);
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

        const shard_file = try std.fs.cwd().openFile(full_path, .{});
        defer shard_file.close();
        var read_buffer: [1024]u8 = undefined;
        var reader = shard_file.reader(&read_buffer);

        const len = try reader.interface.readAlloc(self.allocator, 8);
        defer self.allocator.free(len);
        const header_len = std.mem.readInt(u64, len[0..8], .little);
        const header_bytes = try reader.interface.readAlloc(self.allocator, header_len);

        const shard_json = try std.json.parseFromSlice(std.json.Value, self.allocator, header_bytes, .{});
        self.allocator.free(header_bytes);

        if (first) {
            self.json_data = shard_json; // Keep ownership of the first one for metadata/structure
            if (shard_json.value.object.get("__metadata__")) |m| {
                self.metadata = m.object;
            }
            first = false;
        } else {
            // For subsequent shards, we just extract tensors and discard the JSON
            // Check for metadata if we haven't found it yet
            if (self.metadata == null) {
                if (shard_json.value.object.get("__metadata__")) |m| {
                    // We need to copy this metadata out because we are about to deinit shard_json
                    // Actually, simpler to just swap ownership of this json_data if we find metadata
                    // But that gets messy.
                    // Let's assume metadata is in the first shard or duplicated.
                    _ = m;
                }
            }
            defer shard_json.deinit();
        }

        // We use the JSON object to extract tensors
        // Note: For the 'first' shard, we are using self.json_data which is valid.
        // For others, we use shard_json.
        //const root = if (!first and self.json_data.value == shard_json.value) self.json_data.value.object else shard_json.value.object;
        try self.extractTensorsFromObject(self.json_data.value.object, full_path);
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

pub fn getReaderForTensor(self: *Safetensors, name: []const u8, buffer: []u8) !std.fs.File.Reader {
    for (self.tensors.items) |t| {
        if (std.mem.eql(u8, t.name, name)) {
            const tensor_path = t.source_path orelse self.path;

            if (!std.mem.eql(u8, self.current_open_path, tensor_path)) {
                if (self.current_file_handle) |h| h.close();

                const new_file = try std.fs.cwd().openFile(tensor_path, .{});
                self.current_file_handle = new_file;
                self.current_open_path = tensor_path;

                var len_bytes: [8]u8 = undefined;
                _ = try new_file.readAll(&len_bytes);
                const st_len = std.mem.readInt(u64, len_bytes[0..8], .little);
                self.current_data_begin = 8 + st_len;
            }

            if (self.current_file_handle) |h| {
                try h.seekTo(self.current_data_begin + t.offset);
                return h.reader(buffer);
            }
        }
    }
    return error.TensorNotFound;
}

pub fn printMetadata(self: Safetensors, writer: *std.io.Writer) !void {
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
        var list = std.ArrayList([]const u8).init(allocator);
        defer list.deinit();

        var current: ?*TensorNode = self;
        while (current) |node| : (current = node.parent) {
            try list.append(node.name);
        }

        // Now join the parts in reverse order
        var result = std.ArrayList(u8).init(allocator);
        var i: usize = list.items.len;
        while (i > 0) {
            i -= 1;
            if (i < list.items.len - 1) {
                try result.appendSlice(".");
            }
            try result.appendSlice(list.items[i]);
        }

        return result.toOwnedSlice();
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

        return total_elements * self.dtype.?.getSizeInBytes();
    }
};

pub const DType = enum {
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
        var buf: [8]u8 = undefined;
        const upper = std.ascii.upperString(&buf, str);

        inline for (std.meta.fields(DType)) |field| {
            if (std.mem.eql(u8, upper, field.name)) {
                return @field(DType, field.name);
            }
        }
        return error.UnknownDType;
    }

    pub fn getSizeInBytes(self: DType) u8 {
        return switch (self) {
            .BF16, .F16 => 2,
            .F32, .I32, .U32 => 4,
            .F64, .I64, .U64 => 8,
            .I8, .U8 => 1,
            .I16, .U16 => 2,
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

pub fn printTensorTree(self: Safetensors, writer: *std.io.Writer) !void {
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

fn printNode(self: Safetensors, node: *TensorNode, writer: *std.io.Writer, depth: usize) !void {
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

pub fn printHeader(self: Safetensors, writer: *std.io.Writer) !void {
    var dtype_counts = std.AutoHashMap(DType, usize).init(self.allocator);
    defer dtype_counts.deinit();

    try writer.print("{{\n", .{});
    for (self.tensors.items, 0..) |t, i| {
        try writer.print("  \"{s}\": {{\n", .{t.name});
        try writer.print("    \"dtype\": \"{s}\",\n", .{t.type});
        try writer.print("    \"shape\": [", .{});
        for (t.dims, 0..) |d, di| {
            if (di > 0) try writer.print(", ", .{});
            try writer.print("{}", .{d});
        }
        try writer.print("],\n", .{});
        try writer.print("    \"offset_from_data_start_and_file_start\": [{}, {}]\n", .{ t.offset, t.offset + self.current_data_begin });
        try writer.print("  }}", .{});
        if (i < self.tensors.items.len - 1) try writer.print(",", .{});
        try writer.print("\n", .{});

        const dt = try DType.fromString(t.type);
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
    const len = try self.reader.readAlloc(self.arena_alloc, 8);
    const header_len = std.mem.readInt(u64, len[0..8], .little);

    const data = try self.reader.readAlloc(self.arena_alloc, header_len);

    return std.json.parseFromSlice(std.json.Value, self.arena_alloc, data, .{});
}
