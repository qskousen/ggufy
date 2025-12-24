const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const nw = @import("NullWriter.zig");
const gguf = @import("Gguf.zig");

const Command = enum {
    header,
    tree,
    metadata,
    convert,

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
            \\Usage: ggufy <command> <file.safetensors> <convert_to_filetype> <convert_to_datatype>
            \\Commands:
            \\  header   - Show tensor names and shapes from header
            \\  tree     - Show the tensors as a tree
            \\  metadata - Show metadata
            \\  convert  - Convert a file from one format to another
            \\           - Filetype argument: safetensor, gguf
            \\           - Datatype argument: f16
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

    const file_type = types.FileType.detect_from_file(&reader.interface, arena_alloc)
        catch types.FileType.safetensors;
    try reader.seekTo(0);
    switch (file_type) {
        .safetensors => {
            var f = try st.init(path, arena_alloc);
            defer f.deinit();

            switch (command) {
                .header => {
                    try f.printHeader(stdout);
                },
                .tree => {
                    try f.printTensorTree(stdout);
                },
                .metadata => {
                    try f.printMetadata(stdout);
                },
                .convert => {
                    const target_filetype_str = args_it.next() orelse {
                        try stderr.print("Usage: ggufy convert <file.safetensors> <convert_to_filetype> <convert_to_datatype>\n", .{});
                        try stderr.print("Example usage: ggufy convert file.safetensors gguf f16\n", .{});
                        try stderr.flush();
                        return error.InvalidArgs;
                    };
                    const target_filetype = types.FileType.parse_from_string(target_filetype_str) catch |err| {
                        try stderr.print("Unknown target filetype: {s}\n", .{target_filetype_str});
                        try stderr.flush();
                        return err;
                    };
                    const target_datatype_str = args_it.next() orelse {
                        try stderr.print("Usage: ggufy convert <file.safetensors> <convert_to_filetype> <convert_to_datatype>\n", .{});
                        try stderr.print("Example usage: ggufy convert file.safetensors gguf f16\n", .{});
                        try stderr.flush();
                        return error.InvalidArgs;
                    };
                    const target_datatype = st.DType.fromString(target_datatype_str) catch |err| {
                        try stderr.print("Unknown target datatype: {s}\n", .{target_datatype_str});
                        try stderr.flush();
                        return err;
                    };

                    // TODO: if the target datatype is higher precision than the source, print an error warning the user
                    // of the dangers of upcasting not resulting in higher precision/less perplexity and exit, unless
                    // they pass a flag acknowledging that they understand the issues involved and want to proceed.

                    switch (target_filetype) {
                        .gguf => {
                            const out_filename = try std.fmt.allocPrint(arena_alloc, "{s}.gguf", .{std.fs.path.stem(path)});
                            const out_file = try std.fs.cwd().createFile(out_filename, .{ .truncate = true });
                            defer out_file.close();
                            var writer_buffer: [1024]u8 = undefined;
                            var out_writer = out_file.writer(&writer_buffer);
                            var writer = &out_writer.interface;

                            // Sort tensors
                            std.sort.block(types.Tensor, f.tensors.items, {}, struct {
                                fn lessThan(_: void, a: types.Tensor, b: types.Tensor) bool {
                                    return std.mem.lessThan(u8, a.name, b.name);
                                }
                            }.lessThan);

                            // Metadata count
                            var metadata_count: u64 = 1; // general.alignment
                            if (f.metadata) |meta| {
                                var it = meta.iterator();
                                while (it.next()) |entry| {
                                    if (entry.value_ptr.* == .string) metadata_count += 1;
                                }
                            }

                            try gguf.writeHeader(writer, @intCast(f.tensors.items.len), metadata_count);
                            try gguf.writeMetadataKVU32(writer, "general.alignment", 32);

                            if (f.metadata) |meta| {
                                var it = meta.iterator();
                                while (it.next()) |entry| {
                                    if (entry.value_ptr.* == .string) {
                                        try gguf.writeMetadataKVString(writer, entry.key_ptr.*, entry.value_ptr.string);
                                    }
                                }
                            }

                            var current_offset: u64 = 0;
                            for (f.tensors.items) |t| {
                                const ggml_type = try gguf.GgmlType.fromSafetensorsType(t.type);
                                try gguf.writeTensorInfo(writer, t.name, t.dims, ggml_type, current_offset);

                                const byte_size = t.size;
                                var next_offset = current_offset + byte_size;
                                const remainder = next_offset % 32;
                                if (remainder != 0) next_offset += (32 - remainder);
                                current_offset = next_offset;
                            }

                            try writer.flush();

                            // Padding for data start
                            const header_pos = try out_file.getPos();
                            const padding_len = (32 - (header_pos % 32)) % 32;
                            if (padding_len > 0) {
                                const zeros = [_]u8{0} ** 32;
                                try writer.writeAll(zeros[0..padding_len]);
                                try writer.flush();
                            }

                            // Data Copy
                            var copy_buf = try arena_alloc.alloc(u8, 1024 * 1024);
                            var current_open_path: []const u8 = "";
                            var current_file_handle: ?std.fs.File = null;
                            var current_data_begin: u64 = 0;
                            defer if (current_file_handle) |h| h.close();

                            for (f.tensors.items) |t| {
                                try stdout.print("Converting tensor {s}\n", .{t.name});
                                try stdout.flush();
                                const tensor_path = t.source_path orelse path;

                                if (!std.mem.eql(u8, current_open_path, tensor_path)) {
                                    if (current_file_handle) |h| h.close();

                                    try stdout.print("Opening file {s}\n", .{tensor_path});
                                    try stdout.flush();
                                    const new_file = try std.fs.cwd().openFile(tensor_path, .{});
                                    current_file_handle = new_file;
                                    current_open_path = tensor_path;

                                    var len_bytes: [8]u8 = undefined;
                                    _ = try new_file.readAll(&len_bytes);
                                    const st_len = std.mem.readInt(u64, len_bytes[0..8], .little);
                                    current_data_begin = 8 + st_len;
                                }

                                if (current_file_handle) |h| {
                                    try h.seekTo(current_data_begin + t.offset);
                                    var left = t.size;
                                    while (left > 0) {
                                        const n = try h.read(copy_buf);
                                        if (n == 0) return error.UnexpectedEof;
                                        const take = @min(left, n);
                                        try writer.writeAll(copy_buf[0..take]);
                                        left -= take;
                                    }
                                }

                                const padding = (32 - (t.size % 32)) % 32;
                                if (padding > 0) {
                                    const zeros = [_]u8{0} ** 32;
                                    try writer.writeAll(zeros[0..padding]);
                                }
                            }
                            try writer.flush();
                            try stdout.print("Converted to {s}\n", .{out_filename});
                        },
                        .safetensors => {
                            return error.Unimplimented;
                        },
                    }

                    _ = target_datatype;
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
                .tree => {
                    return error.Unimplemented;
                },
                .metadata => {
                    // skip the tensors count section
                    try reader.seekBy(8);
                    try f.readGgufMetadata(stdout);
                },
                .convert => {
                    return error.Unimplimented;
                },
            }
        },
    }
    try stdout.print("Total bytes used in arena allocator: {}\n", .{arena.queryCapacity()});
    try stdout.flush();
}
