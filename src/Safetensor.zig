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

pub fn printHeader(self: Safetensors, writer: *std.io.Writer) !void {
    var w: std.json.Stringify = .{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
    const data = try self.parseHeader();
    return w.write(data.value);
}

pub fn parseHeader(self: Safetensors) !std.json.Parsed(std.json.Value) {
    const len = try self.reader.readAlloc(self.allocator, 8);
    const header_len = std.mem.readInt(u64, len[0..8], .little);

    const data = try self.reader.readAlloc(self.allocator, header_len);

    return std.json.parseFromSlice(std.json.Value, self.allocator, data, .{});
}