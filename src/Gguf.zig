const std = @import("std");
const nw = @import("NullWriter.zig");
const types = @import("types.zig");

reader: *std.io.Reader,
allocator: std.mem.Allocator,

tensors: std.ArrayList(types.Tensor),
metadata: std.json.ObjectMap,
version: u32 = 0,

data_offset: u64 = 0,
file_size: u64 = 0,

const Gguf = @This();

pub fn init(data_reader: *std.io.Reader, mem_allocator: std.mem.Allocator) !Gguf {
    var self = Gguf{
        .reader = data_reader,
        .allocator = mem_allocator,
        .tensors = try std.ArrayList(types.Tensor).initCapacity(mem_allocator, 200),
        .metadata = std.json.ObjectMap.init(mem_allocator),
    };
    errdefer {
        self.metadata.deinit();
        self.tensors.deinit(mem_allocator);
    }

    const version_buffer = try data_reader.readAlloc(mem_allocator, 4);
    defer mem_allocator.free(version_buffer);
    self.version = std.mem.readInt(u32, version_buffer[0..4], .little);

    const tensor_count_buf = try data_reader.readAlloc(mem_allocator, 8);
    defer mem_allocator.free(tensor_count_buf);
    const tensor_count = std.mem.readInt(u64, tensor_count_buf[0..8], .little);

    const metadata_count_buf = try data_reader.readAlloc(mem_allocator, 8);
    defer mem_allocator.free(metadata_count_buf);
    const metadata_count = std.mem.readInt(u64, metadata_count_buf[0..8], .little);

    var i: u64 = 0;
    while (i < metadata_count) : (i += 1) {
        var metadata = try GgufMetadata.read(data_reader, mem_allocator);
        defer metadata.deinit(mem_allocator);

        const val = try self.readGgufValueAsJson(metadata.value_type);
        try self.metadata.put(try mem_allocator.dupe(u8, metadata.title), val);
    }

    try self.tensors.ensureTotalCapacity(self.allocator, tensor_count);
    var ti: u64 = 0;
    while (ti < tensor_count) : (ti += 1) {
        const str_len_buf = try data_reader.readAlloc(mem_allocator, 8);
        defer mem_allocator.free(str_len_buf);
        const str_len = std.mem.readInt(i64, str_len_buf[0..8], .little);
        const name = try data_reader.readAlloc(mem_allocator, @intCast(str_len));
        errdefer mem_allocator.free(name);

        const dim_count_buf = try data_reader.readAlloc(mem_allocator, 4);
        defer mem_allocator.free(dim_count_buf);
        const dim_count = std.mem.readInt(u32, dim_count_buf[0..4], .little);

        var dims = try mem_allocator.alloc(usize, dim_count);
        errdefer mem_allocator.free(dims);

        var j: u64 = 0;
        while (j < dim_count) : (j += 1) {
            const buf = try data_reader.readAlloc(mem_allocator, 8);
            defer mem_allocator.free(buf);
            dims[j] = @intCast(std.mem.readInt(u64, buf[0..8], .little));
        }

        const type_buf = try data_reader.readAlloc(mem_allocator, 4);
        defer mem_allocator.free(type_buf);
        const tensor_type = try GgmlType.fromInt(std.mem.readInt(u32, type_buf[0..4], .little));

        const offset_buf = try data_reader.readAlloc(mem_allocator, 8);
        defer mem_allocator.free(offset_buf);
        const tensor_offset = std.mem.readInt(u64, offset_buf[0..8], .little);

        try self.tensors.append(self.allocator, .{
            .name = name,
            .type = @tagName(tensor_type),
            .dims = dims,
            .size = 0, // GGUF tensor info doesn't store size, needs calculation
            .offset = tensor_offset,
        });
    }

    return self;
}

pub fn deinit(self: *Gguf) void {
    self.metadata.deinit();
    for (self.tensors.items) |t| {
        self.allocator.free(t.name);
        self.allocator.free(t.dims);
    }
    self.tensors.deinit(self.allocator);
}

pub const GgufValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

pub const GgmlType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    q4_2 = 4, // Support has been removed from gguf files
    q4_3 = 5, // Support has been removed from gguf files
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    q4_0_4_4 = 31, // Support has been removed from gguf files
    q4_0_4_8 = 32, // Support has been removed from gguf files
    q4_0_8_8 = 33, // Support has been removed from gguf files
    tq1_0 = 34,
    tq2_0 = 35,
    iq4_nl_4_4 = 36, // Support has been removed from gguf files
    iq4_nl_4_8 = 37, // Support has been removed from gguf files
    iq4_nl_8_8 = 38, // Support has been removed from gguf files
    mxfp4 = 39,
    count = 40,

    pub fn fromInt(value: u32) !GgmlType {
        return std.meta.intToEnum(GgmlType, value) catch error.InvalidGgmlType;
    }

    pub fn isUnsupported(self: GgmlType) bool {
        return switch (self) {
            .q4_2, .q4_3, .q4_0_4_4, .q4_0_4_8, .q4_0_8_8, .iq4_nl_4_4, .iq4_nl_4_8, .iq4_nl_8_8 => true,
            else => false,
        };
    }

    pub fn fromSafetensorsType(str: []const u8) !GgmlType {
        if (std.ascii.eqlIgnoreCase(str, "F32")) return .f32;
        if (std.ascii.eqlIgnoreCase(str, "F16")) return .f16;
        if (std.ascii.eqlIgnoreCase(str, "BF16")) return .bf16;
        if (std.ascii.eqlIgnoreCase(str, "I32")) return .i32;
        if (std.ascii.eqlIgnoreCase(str, "I16")) return .i16;
        if (std.ascii.eqlIgnoreCase(str, "I8")) return .i8;
        if (std.ascii.eqlIgnoreCase(str, "F64")) return .f64;
        if (std.ascii.eqlIgnoreCase(str, "I64")) return .i64;
        return error.UnsupportedSafetensorType;
    }

    pub fn getBlockSize(self: GgmlType) u64 {
        return switch (self) {
            .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
            .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .iq1_m => 256,
            else => 1,
        };
    }

    pub fn getBytesPerBlock(self: GgmlType) u64 {
        return switch (self) {
            .f32, .i32 => 4,
            .f16, .bf16, .i16 => 2,
            .f64, .i64 => 8,
            .i8 => 1,

            // Legacy
            .q4_0 => 18,
            .q4_1 => 20,
            .q5_0 => 22,
            .q5_1 => 24,
            .q8_0 => 34,
            .q8_1 => 36,

            // K-Quants
            .q2_k => 84,
            .q3_k => 110,
            .q4_k => 144,
            .q5_k => 176,
            .q6_k => 210,
            .q8_k => 292,

            else => 0,
        };
    }

    pub fn calcSizeInBytes(self: GgmlType, n_elements: u64) u64 {
        const bs = self.getBlockSize();
        // For quantized types, n_elements must be a multiple of the block size.
        // Integer division handles this, assuming valid GGUF constraints.
        return (n_elements / bs) * self.getBytesPerBlock();
    }
};

const GgufMetadata = struct {
    title_len: u64,
    title: []const u8,
    value_type: GgufValueType,

    pub fn read(reader: *std.io.Reader, allocator: std.mem.Allocator) !GgufMetadata {
        var len_buffer = try reader.readAlloc(allocator, 8);
        const title_len = std.mem.readInt(u64, len_buffer[0..8], .little);

        // Add sanity check for title length
        if (title_len == 0 or title_len > 1024 * 1024) { // 1 MiB cap for title length
            std.debug.print("Invalid metadata title length: {}\n", .{title_len});
            return error.InvalidMetadataTitleLength;
        }

        const title = try reader.readAlloc(allocator, title_len);

        const type_buf = try reader.readAlloc(allocator, 4);
        const value_type = @as(GgufValueType, @enumFromInt(std.mem.readInt(u32, type_buf[0..4], .little)));

        return GgufMetadata{
            .title_len = title_len,
            .title = title,
            .value_type = value_type,
        };
    }

    pub fn deinit(self: *GgufMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.title);
    }
};

pub fn readGgufVersion(self: Gguf) !u32 {
    return self.version;
}

pub fn getTensors(self: Gguf) !std.ArrayList(types.Tensor) {
    const count_buf = try self.reader.readAlloc(self.allocator, 8);
    const tensor_count = std.mem.readInt(u64, count_buf[0..8], .little);

    // skip all the metadata
    var null_buffer: [1]u8 = undefined;
    var nullwrite = nw.NullWriter.init(&null_buffer);
    try self.readGgufMetadata(&nullwrite.interface);

    var tensors = try std.ArrayList(types.Tensor).initCapacity(self.allocator, 200);
    errdefer {
        for (tensors.items) |t| {
            self.allocator.free(t.name);
            self.allocator.free(t.dims);
        }
        tensors.deinit(self.allocator);
    }

    var i: u64 = 0;
    while (i < tensor_count) : (i += 1) {
        // tensor name string
        const str_len_buf = try self.reader.readAlloc(self.allocator, 8);
        const str_len = std.mem.readInt(i64, str_len_buf[0..8], .little);
        const str = try self.reader.readAlloc(self.allocator, @intCast(str_len));
        errdefer self.allocator.free(str);

        // number of dimensions
        const dim_count_buf = try self.reader.readAlloc(self.allocator, 4);
        const dim_count = std.mem.readInt(u32, dim_count_buf[0..4], .little);

        // Allocate a slice for the dimensions
        var dims = try self.allocator.alloc(usize, dim_count);
        errdefer self.allocator.free(dims);

        var j: u64 = 0;
        while (j < dim_count) : (j += 1) {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            dims[j] = @intCast(std.mem.readInt(u64, buf[0..8], .little));
        }

        // type of tensor
        const type_buf = try self.reader.readAlloc(self.allocator, 4);
        const tensor_type = try GgmlType.fromInt(std.mem.readInt(u32, type_buf[0..4], .little));

        // offset for tensor
        const offset_buf = try self.reader.readAlloc(self.allocator, 8);
        const tensor_offset = std.mem.readInt(u64, offset_buf[0..8], .little);

        try tensors.append(self.allocator, .{
            .name = str,
            .type = @tagName(tensor_type),
            .dims = dims,
            .size = 0,
            .offset = tensor_offset,
        });
    }

    return tensors;
}

pub fn writeHeader(writer: *std.io.Writer, tensor_count: u64, metadata_count: u64) !void {
    _ = try writer.write("GGUF");
    try writer.writeInt(u32, 3, .little); // Version 3
    try writer.writeInt(u64, tensor_count, .little);
    try writer.writeInt(u64, metadata_count, .little);
}

pub fn writeString(writer: *std.io.Writer, str: []const u8) !void {
    try writer.writeInt(u64, str.len, .little);
    _ = try writer.write(str);
}

pub fn writeMetadataKVString(writer: *std.io.Writer, key: []const u8, value: []const u8) !void {
    try writeString(writer, key);
    try writer.writeInt(u32, @intFromEnum(GgufValueType.string), .little);
    try writeString(writer, value);
}

pub fn writeMetadataKVU32(writer: *std.io.Writer, key: []const u8, value: u32) !void {
    try writeString(writer, key);
    try writer.writeInt(u32, @intFromEnum(GgufValueType.uint32), .little);
    try writer.writeInt(u32, value, .little);
}

pub fn writeMetadataKVJson(writer: *std.io.Writer, key: []const u8, value: std.json.Value) !void {
    try writeString(writer, key);
    switch (value) {
        .bool => |b| {
            try writer.writeInt(u32, @intFromEnum(GgufValueType.bool), .little);
            try writer.writeByte(if (b) 1 else 0);
        },
        .integer => |i| {
            // Heuristic: default to u32 for small positive integers if not specified
            // GGUF expects specific types. For now we assume u32 or i64 based on size.
            if (i >= 0 and i <= std.math.maxInt(u32)) {
                try writer.writeInt(u32, @intFromEnum(GgufValueType.uint32), .little);
                try writer.writeInt(u32, @intCast(i), .little);
            } else {
                try writer.writeInt(u32, @intFromEnum(GgufValueType.int64), .little);
                try writer.writeInt(i64, i, .little);
            }
        },
        .float => |f| {
            try writer.writeInt(u32, @intFromEnum(GgufValueType.float32), .little);
            try writer.writeInt(u32, @bitCast(@as(f32, @floatCast(f))), .little);
        },
        .string => |s| {
            try writer.writeInt(u32, @intFromEnum(GgufValueType.string), .little);
            try writeString(writer, s);
        },
        .array => |a| {
            try writer.writeInt(u32, @intFromEnum(GgufValueType.array), .little);
            // Assume all elements in array have same type as first element
            if (a.items.len == 0) return error.EmptyMetadataArray;
            const first = a.items[0];
            const array_type: GgufValueType = switch (first) {
                .bool => .bool,
                .integer => .uint32, // Heuristic
                .float => .float32,
                .string => .string,
                else => return error.UnsupportedMetadataArrayType,
            };
            try writer.writeInt(u32, @intFromEnum(array_type), .little);
            try writer.writeInt(u64, a.items.len, .little);
            for (a.items) |item| {
                switch (item) {
                    .bool => |b| try writer.writeByte(if (b) 1 else 0),
                    .integer => |i| try writer.writeInt(u32, @intCast(i), .little),
                    .float => |f| try writer.writeInt(u32, @bitCast(@as(f32, @floatCast(f))), .little),
                    .string => |s| try writeString(writer, s),
                    else => unreachable,
                }
            }
        },
        else => return error.UnsupportedMetadataType,
    }
}

pub fn writeTensorInfo(writer: *std.io.Writer, name: []const u8, dims: []const usize, type_: GgmlType, offset: u64) !void {
    try writeString(writer, name);
    try writer.writeInt(u32, @intCast(dims.len), .little);
    // GGUF expects dimensions in reverse order (e.g. [width, height, channels, batch])
    var i = dims.len;
    while (i > 0) {
        i -= 1;
        try writer.writeInt(u64, @intCast(dims[i]), .little);
    }
    try writer.writeInt(u32, @intFromEnum(type_), .little);
    try writer.writeInt(u64, offset, .little);
}

pub fn readGgufMetadata(self: Gguf, writer: *std.io.Writer) !void {
    try writer.print("Metadata count: {}\n", .{self.metadata.count()});

    var it = self.metadata.iterator();
    while (it.next()) |entry| {
        try writer.print("{s}: ", .{entry.key_ptr.*});
        try self.printJsonValue(writer, entry.value_ptr.*, 0);
        try writer.writeAll("\n");
    }
}

fn printJsonValue(self: Gguf, writer: *std.io.Writer, val: std.json.Value, depth: usize) !void {
    switch (val) {
        .bool => |b| try writer.print("{}", .{b}),
        .integer => |i| try writer.print("{}", .{i}),
        .float => |f| try writer.print("{d}", .{f}),
        .string => |s| try writer.print("\"{s}\"", .{s}),
        .array => |a| {
            if (a.items.len > 10) {
                try writer.print("[array of {} elements]: !! Array size greater than 10, not outputting !!", .{a.items.len});
            } else {
                try writer.writeAll("[");
                for (a.items, 0..) |item, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try self.printJsonValue(writer, item, depth + 1);
                }
                try writer.writeAll("]");
            }
        },
        .object => |o| {
            try writer.writeAll("{ ");
            var it = o.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try writer.writeAll(", ");
                first = false;
                try writer.print("\"{s}\": ", .{entry.key_ptr.*});
                try self.printJsonValue(writer, entry.value_ptr.*, depth + 1);
            }
            try writer.writeAll(" }");
        },
        .null => try writer.writeAll("null"),
        .number_string => |s| try writer.print("\"{s}\"", .{s}),
    }
}

pub fn readGgufTensorHeader(self: Gguf, stdout: *std.io.Writer) !void {
    try stdout.print("Tensor count: {}\n", .{self.tensors.items.len});

    var type_counts = std.AutoHashMap(GgmlType, usize).init(self.allocator);
    defer type_counts.deinit();

    var bad_size_count: u64 = 0;

    for (self.tensors.items, 0..) |tensor, i| {
        const tensor_type = std.meta.stringToEnum(GgmlType, tensor.type);

        var bad_size = false;

        if (tensor_type) |tt| {
            const g = try type_counts.getOrPut(tt);
            if (!g.found_existing) g.value_ptr.* = 0;
            g.value_ptr.* += 1;

            // calculate the total number of elements in this tensor
            var tensor_elements: u64 = 1;
            for (tensor.dims) |d| {
                tensor_elements *= d;
            }
            // calculate the total size in bytes for this tensor
            const total_bytes = tt.calcSizeInBytes(tensor_elements);

            const alignment = self.metadata.get("general.alignment") orelse std.json.Value{ .integer = 32 };
            const align_val: u64 = @intCast(alignment.integer);

            var expected_padded_size = total_bytes;
            if (expected_padded_size % align_val != 0) {
                expected_padded_size += align_val - (expected_padded_size % align_val);
            }

            // Validate against next tensor offset or end of file
            if (i < self.tensors.items.len - 1) {
                const next_offset = self.tensors.items[i + 1].offset;
                const allocated_size = next_offset - tensor.offset;

                if (allocated_size != expected_padded_size) {
                    bad_size = true;
                    bad_size_count += 1;
                }
            } else if (self.file_size > 0 and self.data_offset > 0) {
                // Total data section size available on disk from this tensor's start
                const disk_size_remaining = self.file_size - (self.data_offset + tensor.offset);

                // For the last tensor, the GGUF file should end exactly after the
                // tensor data (optionally including its specific alignment padding).
                // If disk_size_remaining is larger than expected_padded_size,
                // there is extra data/garbage at the end of the file.
                if (disk_size_remaining != total_bytes and disk_size_remaining != expected_padded_size) {
                    bad_size = true;
                    bad_size_count += 1;
                }
            } else {
                try stdout.print("Could not check size of last tensor: file size ({}) or data offset ({}) not set!\n", .{ self.file_size, self.data_offset });
            }

            if (tt.isUnsupported()) {
                try stdout.print("{s}: {} (Unsupported type!!!) [", .{ tensor.name, tt });
            } else if (bad_size) {
                try stdout.print("{s}: {} (BAD SIZE) [", .{ tensor.name, tt });
            } else {
                try stdout.print("{s}: {} [", .{ tensor.name, tt });
            }
        } else {
            try stdout.print("{s}: Unknown Type [", .{tensor.name});
        }

        for (tensor.dims, 0..) |dim, j| {
            try stdout.print("{}", .{dim});
            if (j < tensor.dims.len - 1) {
                try stdout.print(", ", .{});
            }
        }
        try stdout.print("] offset {}\n", .{tensor.offset});
    }

    if (type_counts.count() > 0) {
        try stdout.print("\n\nTensor Type Statistics:\n", .{});
        var stats_it = type_counts.iterator();
        while (stats_it.next()) |entry| {
            try stdout.print("  {s}: {}\n", .{ @tagName(entry.key_ptr.*), entry.value_ptr.* });
        }
    }
    if (bad_size_count > 0) {
        try stdout.print("\nTensors found with a bad size (tensor data + alignment doesn't match size on disk): {}", .{bad_size_count});
    }
}

pub fn writeTemplate(self: Gguf, writer: *std.io.Writer) !void {
    var root_obj = std.json.ObjectMap.init(self.allocator);
    defer root_obj.deinit();

    // Metadata - we need to deep copy because the value will be owned by root_obj
    // and stringify might deinit it depending on usage, or we just point to existing.
    // Actually, std.json.Value doesn't have an easy deep copy.
    // We can just construct it manually or use a wrapper.

    try root_obj.put("metadata", std.json.Value{ .object = self.metadata });

    // Tensors
    var tensors_obj = std.json.ObjectMap.init(self.allocator);
    errdefer tensors_obj.deinit();

    for (self.tensors.items) |t| {
        var t_obj = std.json.ObjectMap.init(self.allocator);
        errdefer t_obj.deinit();

        var shape_arr = std.json.Array.init(self.allocator);
        errdefer shape_arr.deinit();
        for (t.dims) |dim| {
            try shape_arr.append(std.json.Value{ .integer = @intCast(dim) });
        }
        try t_obj.put("shape", std.json.Value{ .array = shape_arr });
        try t_obj.put("type", std.json.Value{ .string = t.type });

        try tensors_obj.put(try self.allocator.dupe(u8, t.name), std.json.Value{ .object = t_obj });
    }
    try root_obj.put("tensors", std.json.Value{ .object = tensors_obj });

    // Stringify without taking ownership
    var stringifier = std.json.Stringify{
        .writer = writer,
        .options = .{
            .whitespace = .indent_2,
        },
    };

    try stringifier.write(std.json.Value{ .object = root_obj });

    // Remove metadata and tensors from root_obj so they don't get double-freed
    _ = root_obj.swapRemove("metadata");
    // tensors_obj keys were duped, but values were newly created so they are fine to deinit.
    // However we must be careful with tensors_obj deinit.
}

fn readGgufValueAsJson(self: Gguf, value_type: GgufValueType) !std.json.Value {
    return switch (value_type) {
        .bool => {
            const buf = try self.reader.readAlloc(self.allocator, 1);
            return std.json.Value{ .bool = buf[0] != 0 };
        },
        .uint8 => {
            const buf = try self.reader.readAlloc(self.allocator, 1);
            return std.json.Value{ .integer = buf[0] };
        },
        .int8 => {
            const buf = try self.reader.readAlloc(self.allocator, 1);
            return std.json.Value{ .integer = @as(i8, @bitCast(buf[0])) };
        },
        .uint16 => {
            const buf = try self.reader.readAlloc(self.allocator, 2);
            return std.json.Value{ .integer = std.mem.readInt(u16, buf[0..2], .little) };
        },
        .int16 => {
            const buf = try self.reader.readAlloc(self.allocator, 2);
            return std.json.Value{ .integer = std.mem.readInt(i16, buf[0..2], .little) };
        },
        .uint32 => {
            const buf = try self.reader.readAlloc(self.allocator, 4);
            return std.json.Value{ .integer = std.mem.readInt(u32, buf[0..4], .little) };
        },
        .int32 => {
            const buf = try self.reader.readAlloc(self.allocator, 4);
            return std.json.Value{ .integer = std.mem.readInt(i32, buf[0..4], .little) };
        },
        .float32 => {
            const buf = try self.reader.readAlloc(self.allocator, 4);
            return std.json.Value{ .float = @as(f32, @bitCast(std.mem.readInt(u32, buf[0..4], .little))) };
        },
        .uint64 => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            return std.json.Value{ .integer = @intCast(std.mem.readInt(u64, buf[0..8], .little)) };
        },
        .int64 => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            return std.json.Value{ .integer = std.mem.readInt(i64, buf[0..8], .little) };
        },
        .float64 => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            return std.json.Value{ .float = @as(f64, @bitCast(std.mem.readInt(u64, buf[0..8], .little))) };
        },
        .string => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            const str_len = std.mem.readInt(i64, buf[0..8], .little);
            const str = try self.reader.readAlloc(self.allocator, @intCast(str_len));
            return std.json.Value{ .string = str };
        },
        .array => {
            const type_buf = try self.reader.readAlloc(self.allocator, 4);
            const array_type = @as(GgufValueType, @enumFromInt(std.mem.readInt(u32, type_buf[0..4], .little)));
            var len_buf = try self.reader.readAlloc(self.allocator, 8);
            const arr_len = std.mem.readInt(i64, len_buf[0..8], .little);

            var arr = std.json.Array.init(self.allocator);
            errdefer arr.deinit();
            var j: i64 = 0;
            while (j < arr_len) : (j += 1) {
                try arr.append(try self.readGgufValueAsJson(array_type));
            }
            return std.json.Value{ .array = arr };
        },
    };
}

fn readGgufArrayValue(self: Gguf, writer: *std.io.Writer, value_type: GgufValueType, depth: usize) !void {
    switch (value_type) {
        .bool => {
            const buf = try self.reader.readAlloc(self.allocator, 1);
            try writer.print("{}", .{buf[0] != 0});
        },
        .uint8 => {
            const buf = try self.reader.readAlloc(self.allocator, 1);
            try writer.print("{}", .{buf[0]});
        },
        .int8 => {
            const buf = try self.reader.readAlloc(self.allocator, 1);
            try writer.print("{}", .{@as(i8, @bitCast(buf[0]))});
        },
        .uint16 => {
            const buf = try self.reader.readAlloc(self.allocator, 2);
            try writer.print("{}", .{std.mem.readInt(u16, buf[0..2], .little)});
        },
        .int16 => {
            const buf = try self.reader.readAlloc(self.allocator, 2);
            try writer.print("{}", .{std.mem.readInt(i16, buf[0..2], .little)});
        },
        .uint32 => {
            const buf = try self.reader.readAlloc(self.allocator, 4);
            try writer.print("{}", .{std.mem.readInt(u32, buf[0..4], .little)});
        },
        .int32 => {
            const buf = try self.reader.readAlloc(self.allocator, 4);
            try writer.print("{}", .{std.mem.readInt(i32, buf[0..4], .little)});
        },
        .float32 => {
            const buf = try self.reader.readAlloc(self.allocator, 4);
            try writer.print("{d}", .{@as(f32, @bitCast(std.mem.readInt(u32, buf[0..4], .little)))});
        },
        .uint64 => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            try writer.print("{}", .{std.mem.readInt(u64, buf[0..8], .little)});
        },
        .int64 => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            try writer.print("{}", .{std.mem.readInt(i64, buf[0..8], .little)});
        },
        .float64 => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            try writer.print("{d}", .{@as(f64, @bitCast(std.mem.readInt(u64, buf[0..8], .little)))});
        },
        .array => {
            // Read nested array type
            const type_buf = try self.reader.readAlloc(self.allocator, 4);
            const array_type = @as(GgufValueType, @enumFromInt(std.mem.readInt(u32, type_buf[0..4], .little)));

            // Read array length
            const len_buf = try self.reader.readAlloc(self.allocator, 8);
            const arr_len = std.mem.readInt(i64, len_buf[0..8], .little);

            try writer.print("[\n", .{});
            var i: i64 = 0;
            while (i < arr_len) : (i += 1) {
                // Indent based on depth
                try writeIndent(writer, depth + 1);
                try self.readGgufArrayValue(writer, array_type, depth + 1);
                if (i < arr_len - 1) {
                    try writer.print(",", .{});
                }
                try writer.print("\n", .{});
            }
            try writeIndent(writer, depth);
            try writer.print("]", .{});
        },
        .string => {
            const buf = try self.reader.readAlloc(self.allocator, 8);
            const str_len = std.mem.readInt(i64, buf[0..8], .little);
            const str = try self.reader.readAlloc(self.allocator, @intCast(str_len));
            try writer.print("\"{s}\"", .{str});
        },
    }
}

fn writeIndent(writer: *std.io.Writer, depth: usize) !void {
    var i: usize = 0;
    while (i < depth * 2) : (i += 1) {
        try writer.writeAll(" ");
    }
}
