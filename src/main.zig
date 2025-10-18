const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const nw = @import("NullWriter.zig");

const Command = enum {
    header,
    metadata,

    pub fn parse(str: []const u8) !Command {
        inline for (std.meta.fields(Command)) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @field(Command, field.name);
            }
        }
        return error.UnknownCommand;
    }
};

const GgufValueType = enum(u32) {
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

const GgmlType = enum(u32) {
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
            .q4_2, .q4_3, .q4_0_4_4, .q4_0_4_8, .q4_0_8_8,
            .iq4_nl_4_4, .iq4_nl_4_8, .iq4_nl_8_8 => true,
            else => false,
        };
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

fn readGgufVersion(reader: *std.io.Reader, allocator: std.mem.Allocator) !u32 {
    const version_buffer = try reader.readAlloc(allocator, 4);
    return std.mem.readInt(u32, version_buffer[0..4], .little);
}

fn readGgufMetadata(reader: *std.io.Reader, writer: *std.io.Writer, allocator: std.mem.Allocator) !void {
    var count_buffer = try reader.readAlloc(allocator, 8);
    const count = std.mem.readInt(u64, count_buffer[0..8], .little);

    try writer.print("Metadata count: {}\n", .{count});

    var i: u64 = 0;
    while (i < count) : (i += 1) {
        var metadata = try GgufMetadata.read(reader, allocator);
        defer metadata.deinit(allocator);

        try writer.print("{s}: ", .{metadata.title});

        switch (metadata.value_type) {
            .bool => {
                const buf = try reader.readAlloc(allocator, 1);
                try writer.print("{}\n", .{buf[0] != 0});
            },
            .uint8 => {
                const buf = try reader.readAlloc(allocator, 1);
                try writer.print("{}\n", .{buf[0]});
            },
            .int8 => {
                const buf = try reader.readAlloc(allocator, 1);
                try writer.print("{}\n", .{@as(i8, @bitCast(buf[0]))});
            },
            .uint16 => {
                const buf = try reader.readAlloc(allocator, 2);
                try writer.print("{}\n", .{std.mem.readInt(u16, buf[0..2], .little)});
            },
            .int16 => {
                const buf = try reader.readAlloc(allocator, 2);
                try writer.print("{}\n", .{std.mem.readInt(i16, buf[0..2], .little)});
            },
            .uint32 => {
                const buf = try reader.readAlloc(allocator, 4);
                try writer.print("{}\n", .{std.mem.readInt(u32, buf[0..4], .little)});
            },
            .int32 => {
                const buf = try reader.readAlloc(allocator, 4);
                try writer.print("{}\n", .{std.mem.readInt(i32, buf[0..4], .little)});
            },
            .float32 => {
                const buf = try reader.readAlloc(allocator, 4);
                try writer.print("{d}\n", .{@as(f32, @bitCast(std.mem.readInt(u32, buf[0..4], .little)))});
            },
            .uint64 => {
                const buf = try reader.readAlloc(allocator, 8);
                try writer.print("{}\n", .{std.mem.readInt(u64, buf[0..8], .little)});
            },
            .int64 => {
                const buf = try reader.readAlloc(allocator, 8);
                try writer.print("{}\n", .{std.mem.readInt(i64, buf[0..8], .little)});
            },
            .float64 => {
                const buf = try reader.readAlloc(allocator, 8);
                try writer.print("{d}\n", .{@as(f64, @bitCast(std.mem.readInt(u64, buf[0..8], .little)))});
            },
            .string => {
                const buf = try reader.readAlloc(allocator, 8);
                const str_len = std.mem.readInt(i64, buf[0..8], .little);
                const str = try reader.readAlloc(allocator, @intCast(str_len));
                try writer.print("\"{s}\"\n", .{str});
            },
            .array => {
                const type_buf = try reader.readAlloc(allocator, 4);
                const array_type = @as(GgufValueType, @enumFromInt(std.mem.readInt(u32, type_buf[0..4], .little)));
                var len_buf = try reader.readAlloc(allocator, 8);
                const arr_len = std.mem.readInt(i64, len_buf[0..8], .little);
                try writer.print("[array of {} elements of type {}]: ", .{arr_len, array_type});
                try writer.print("[", .{});

                var j: i64 = 0;
                while (j < arr_len) : (j += 1) {
                    if (j > 0) try writer.print(", ", .{});
                    try readGgufArrayValue(reader, writer, array_type, 0, allocator);
                }

                try writer.print("]\n", .{});
                try writer.print("\n", .{});
            },
        }
    }
}

fn readGgufArrayValue(reader: *std.io.Reader, writer: *std.io.Writer, value_type: GgufValueType, depth: usize, allocator: std.mem.Allocator) !void {
    switch (value_type) {
        .bool => {
            const buf = try reader.readAlloc(allocator, 1);
            try writer.print("{}", .{buf[0] != 0});
        },
        .uint8 => {
            const buf = try reader.readAlloc(allocator, 1);
            try writer.print("{}", .{buf[0]});
        },
        .int8 => {
            const buf = try reader.readAlloc(allocator, 1);
            try writer.print("{}", .{@as(i8, @bitCast(buf[0]))});
        },
        .uint16 => {
            const buf = try reader.readAlloc(allocator, 2);
            try writer.print("{}", .{std.mem.readInt(u16, buf[0..2], .little)});
        },
        .int16 => {
            const buf = try reader.readAlloc(allocator, 2);
            try writer.print("{}", .{std.mem.readInt(i16, buf[0..2], .little)});
        },
        .uint32 => {
            const buf = try reader.readAlloc(allocator, 4);
            try writer.print("{}", .{std.mem.readInt(u32, buf[0..4], .little)});
        },
        .int32 => {
            const buf = try reader.readAlloc(allocator, 4);
            try writer.print("{}", .{std.mem.readInt(i32, buf[0..4], .little)});
        },
        .float32 => {
            const buf = try reader.readAlloc(allocator, 4);
            try writer.print("{d}", .{@as(f32, @bitCast(std.mem.readInt(u32, buf[0..4], .little)))});
        },
        .uint64 => {
            const buf = try reader.readAlloc(allocator, 8);
            try writer.print("{}", .{std.mem.readInt(u64, buf[0..8], .little)});
        },
        .int64 => {
            const buf = try reader.readAlloc(allocator, 8);
            try writer.print("{}", .{std.mem.readInt(i64, buf[0..8], .little)});
        },
        .float64 => {
            const buf = try reader.readAlloc(allocator, 8);
            try writer.print("{d}", .{@as(f64, @bitCast(std.mem.readInt(u64, buf[0..8], .little)))});
        },
        .array => {
            // Read nested array type
            const type_buf = try reader.readAlloc(allocator, 4);
            const array_type = @as(GgufValueType, @enumFromInt(std.mem.readInt(u32, type_buf[0..4], .little)));

            // Read array length
            const len_buf = try reader.readAlloc(allocator, 8);
            const arr_len = std.mem.readInt(i64, len_buf[0..8], .little);

            try writer.print("[\n", .{});
            var i: i64 = 0;
            while (i < arr_len) : (i += 1) {
                // Indent based on depth
                try writeIndent(writer, depth + 1);
                try readGgufArrayValue(reader, writer, array_type, depth + 1, allocator);
                if (i < arr_len - 1) {
                    try writer.print(",", .{});
                }
                try writer.print("\n", .{});
            }
            try writeIndent(writer, depth);
            try writer.print("]", .{});
        },
        .string => {
            const buf = try reader.readAlloc(allocator, 8);
            const str_len = std.mem.readInt(i64, buf[0..8], .little);
            const str = try reader.readAlloc(allocator, @intCast(str_len));
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

fn readGgufTensorHeader(reader: *std.io.Reader, stdout: *std.io.Writer, allocator: std.mem.Allocator) !void {
    // get the count of tensors
    const count_buf = try reader.readAlloc(allocator, 8);
    const tensor_count = std.mem.readInt(u64, count_buf[0..8], .little);
    try stdout.print("Tensor count: {}\n", .{tensor_count});

    // skip all the metadata
    var null_buffer: [1]u8 = undefined;
    var nullwrite = nw.NullWriter.init(&null_buffer);
    try readGgufMetadata(reader, &nullwrite.interface, allocator);
    var i: u64 = 0;
    while (i < tensor_count) : (i += 1) {
        // tensor name string
        const str_len_buf = try reader.readAlloc(allocator, 8);
        const str_len = std.mem.readInt(i64, str_len_buf[0..8], .little);
        const str = try reader.readAlloc(allocator, @intCast(str_len));

        // number of dimensions (currently gguf only supports up to 4, but we will do whatever)
        const dim_count_buf = try reader.readAlloc(allocator, 4);
        const dim_count = std.mem.readInt(u32, dim_count_buf[0..4], .little);
        // Allocate a slice for the dimensions
        var dims = try allocator.alloc(u64, dim_count);
        defer allocator.free(dims);
        var j: u64 = 0;
        while (j < dim_count) : (j += 1) {
            const buf = try reader.readAlloc(allocator, 8);
            dims[j] = std.mem.readInt(u64, buf[0..8], .little);
        }

        // type of tensor
        const buf = try reader.readAlloc(allocator, 4);
        const tensor_type = try GgmlType.fromInt(std.mem.readInt(u32, buf[0..4], .little));

        // offset for tensor
        const offset_buf = try reader.readAlloc(allocator, 8);
        const tensor_offset = std.mem.readInt(u64, offset_buf[0..8], .little);

        // print it!
        if (tensor_type.isUnsupported()) {
            try stdout.print("{s}: {} (Unsupported type!!!) [", .{str, tensor_type});
        } else {
            try stdout.print("{s}: {} [", .{str, tensor_type});
        }
        j = 0;
        while (j < dim_count) : (j += 1) {
            try stdout.print("{}", .{dims[j]});
            if (j < dim_count - 1) {
                try stdout.print(", ", .{});
            }
        }
        try stdout.print("] offset {}\n", .{tensor_offset});
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args_it = try std.process.argsWithAllocator(allocator);
    defer args_it.deinit();

    var stderr_buffer: [256]u8 = undefined;
    var err_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &err_writer.interface;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    _ = args_it.next(); // program name
    const command_str = args_it.next() orelse {
        try stderr.print(
            \\Usage: ggufy <command> <file.safetensors>
            \\Commands:
            \\  header   - Show full safetensors header
            \\  metadata - Show only metadata section
            \\
        , .{});
        try stderr.flush();
        return error.InvalidArgs;
    };

    const command = Command.parse(command_str) catch |err| {
        try stderr.print("Unknown command: {s}\n", .{command_str});
        try stderr.flush();
        return err;
    };
    const path = args_it.next() orelse {
        try stderr.print("Usage: ggufy <file.safetensors>\n", .{});
        try stderr.flush();
        return error.InvalidArgs;
    };

    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();

    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(&read_buffer);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const file_type = try types.FileType.detect(&reader.interface, arena_alloc);
    try reader.seekTo(0);
    switch (file_type) {
        .safetensors => {
            var f = st.init(&reader.interface, arena_alloc);

            switch (command) {
                .header => {
                    try f.printHeader(stdout);
                },
                .metadata => {
                    try f.printMetadata(stdout);
                },
            }
        },
        .gguf => {
            try reader.seekTo(4); // Skip GGUF magic
            const version = try readGgufVersion(&reader.interface, arena_alloc);
            try stdout.print("GGUF format version {}\n", .{version});
            try stdout.flush();
            switch (command) {
                .header => {
                    try readGgufTensorHeader(&reader.interface, stdout, arena_alloc);
                },
                .metadata => {
                    // skip the tensors count section
                    try reader.seekBy(8);
                    try readGgufMetadata(&reader.interface, stdout, arena_alloc);
                },
            }
        },
    }
    try stdout.flush();
}
