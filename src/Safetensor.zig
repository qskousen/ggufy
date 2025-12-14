const std = @import("std");
const types = @import("types.zig");

reader: *std.io.Reader,
allocator: std.mem.Allocator,

pub const formatType = types.FileType.safetensors;

const Safetensors = @This();

pub fn init(data_reader: *std.io.Reader, mem_allocator: std.mem.Allocator) Safetensors {
    return .{
        .reader = data_reader,
        .allocator = mem_allocator,
    };
}

pub fn printMetadata(self: Safetensors, writer: *std.io.Writer) !void {
    var data = try self.parseHeader();
    defer data.deinit();
    if (data.value.object.get("__metadata__")) |metadata| {
        switch (metadata) {
            .object => |obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    try writer.print("{s}: ", .{entry.key_ptr.*});

                    switch (entry.value_ptr.*) {
                        .string => |str| {
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
            },
            else => try writer.print("__metadata__ is not an object\n", .{}),
        }
    } else {
        return error.NoMetadataHeader;
    }
}

// split getting the header and printing the header
// same with metadata
// make it a nice data structure for consooming
// might not work out to just split them since for safetensors it just kinda prints it all out

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

// Add this to your Safetensors struct
pub fn buildTensorTree(self: Safetensors) !*TensorNode {
    const root = try TensorNode.init(self.allocator, "root");
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
                node.value_ptr.* = try TensorNode.init(self.allocator, part);
                node.value_ptr.*.parent = current;
            }
            current = node.value_ptr.*;
        }

        // Add tensor metadata to the leaf node
        if (entry.value_ptr.*.object.get("dtype")) |dtype| {
            current.dtype = try DType.fromString(dtype.string);
        }
        if (entry.value_ptr.*.object.get("shape")) |shape| {
            var shape_list = try std.ArrayList(usize).initCapacity(self.allocator, shape.array.items.len);
            for (shape.array.items) |item| {
                try shape_list.append(self.allocator, @intCast(item.integer));
            }
            current.shape = try shape_list.toOwnedSlice(self.allocator);
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
    var root = try self.buildTensorTree();
    defer root.deinit(self.allocator);
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
    var w: std.json.Stringify = .{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
    const data = try self.parseHeader();
    defer data.deinit();
    try w.write(data.value);

    var dtype_counts = std.AutoHashMap(DType, usize).init(self.allocator);
    defer dtype_counts.deinit();

    var it = data.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

        if (entry.value_ptr.* == .object) {
            if (entry.value_ptr.object.get("dtype")) |dtype_val| {
                if (dtype_val == .string) {
                    if (DType.fromString(dtype_val.string)) |dt| {
                        const g = try dtype_counts.getOrPut(dt);
                        if (!g.found_existing) g.value_ptr.* = 0;
                        g.value_ptr.* += 1;
                    } else |_| {}
                }
            }
        }
    }

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

    var tensors = try std.ArrayList(types.Tensor).initCapacity(self.allocator, 200);
    errdefer {
        for (tensors.items) |t| {
            self.allocator.free(t.name);
            self.allocator.free(t.dims);
        }
        tensors.deinit(self.allocator);
    }

    var it = json_data.value.object.iterator();
    while (it.next()) |entry| {
        if (std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) continue;

        const name = try self.allocator.dupe(u8, entry.key_ptr.*);
        errdefer self.allocator.free(name);

        const obj = entry.value_ptr.object;
        const dtype_str = obj.get("dtype").?.string;
        const dtype = try DType.fromString(dtype_str);

        const shape = obj.get("shape").?.array;
        var dims = try self.allocator.alloc(usize, shape.items.len);
        errdefer self.allocator.free(dims);
        for (shape.items, 0..) |item, i| {
            dims[i] = @intCast(item.integer);
        }

        const offsets = obj.get("data_offsets").?.array;
        const start = @as(u64, @intCast(offsets.items[0].integer));
        const end = @as(u64, @intCast(offsets.items[1].integer));

        try tensors.append(self.allocator, .{
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
    const len = try self.reader.readAlloc(self.allocator, 8);
    const header_len = std.mem.readInt(u64, len[0..8], .little);

    const data = try self.reader.readAlloc(self.allocator, header_len);

    return std.json.parseFromSlice(std.json.Value, self.allocator, data, .{});
}
