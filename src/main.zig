const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const nw = @import("NullWriter.zig");
const gguf = @import("Gguf.zig");

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
            var f = gguf.init(&reader.interface, arena_alloc);
            const version = try f.readGgufVersion();
            try stdout.print("GGUF format version {}\n", .{version});
            try stdout.flush();
            switch (command) {
                .header => {
                    try f.readGgufTensorHeader(stdout);
                },
                .metadata => {
                    // skip the tensors count section
                    try reader.seekBy(8);
                    try f.readGgufMetadata(stdout);
                },
            }
        },
    }
    try stdout.flush();
}
