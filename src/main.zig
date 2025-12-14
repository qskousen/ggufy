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

    const file_type = try types.FileType.detect_from_file(&reader.interface, arena_alloc);
    try reader.seekTo(0);
    switch (file_type) {
        .safetensors => {
            var f = st.init(&reader.interface, arena_alloc);

            switch (command) {
                .header => {
                    try f.printHeader(stdout);
                },
                .tree => {
                    try f.printTensorTree(stdout);
                },
                .metadata => {
                    f.printMetadata(stdout) catch |err| {
                        if (err == error.NoMetadataHeader) {
                            try stderr.print("This file does not have a metadata header.\n", .{});
                            try stderr.flush();
                        } else {
                            return err;
                        }
                    };
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
                            // We need to convert the metadata and tensor header to gguf format, then convert each tensor to gguf.
                            const out_filename = try std.fmt.allocPrint(arena_alloc, "{s}.gguf", .{std.fs.path.stem(path)});
                            const out_file = try std.fs.cwd().createFile(out_filename, .{ .truncate = true });
                            defer out_file.close();
                            var writer_buffer: [1024]u8 = undefined;
                            var out_writer = out_file.writer(&writer_buffer);
                            var writer = &out_writer.interface;

                            // Parse header again to get the JSON object
                            const header_json = try f.parseHeader();
                            defer header_json.deinit();
                            const root_obj = header_json.value.object;

                            // 1. Collect and Sort Tensors
                            try reader.seekTo(0);
                            const tensors = try f.getTensors();

                            var metadata_count: u64 = 0;
                            // We need to store the metadata object to iterate it later
                            var metadata_obj: ?std.json.ObjectMap = null;

                            if (root_obj.get("__metadata__")) |meta_val| {
                                metadata_obj = meta_val.object;
                            }

                            // Sort tensors by name
                            std.sort.block(types.Tensor, tensors.items, {}, struct {
                                fn lessThan(_: void, a: types.Tensor, b: types.Tensor) bool {
                                    return std.mem.lessThan(u8, a.name, b.name);
                                }
                            }.lessThan);

                            // Calculate exact metadata count
                            // Start with 1 for general.alignment
                            metadata_count = 1;
                            if (metadata_obj) |meta| {
                                var meta_it = meta.iterator();
                                while (meta_it.next()) |entry| {
                                    if (entry.value_ptr.* == .string) {
                                        metadata_count += 1;
                                    }
                                }
                            }

                            // Write GGUF Header
                            try gguf.writeHeader(writer, @intCast(tensors.items.len), metadata_count);

                            // Write Metadata
                            try gguf.writeMetadataKVU32(writer, "general.alignment", 32);

                            if (metadata_obj) |meta| {
                                var meta_it = meta.iterator();
                                while (meta_it.next()) |entry| {
                                    try stdout.print("metadata: {s} value: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.string });
                                    // Safetensors metadata values are usually strings
                                    if (entry.value_ptr.* == .string) {
                                        try gguf.writeMetadataKVString(writer, entry.key_ptr.*, entry.value_ptr.string);
                                    }
                                }
                            }

                            // Write Tensor Info and calculate offsets
                            var current_offset: u64 = 0;
                            for (tensors.items) |t| {
                                const ggml_type = try gguf.GgmlType.fromSafetensorsType(t.type);

                                // Validation 1: Check data start/offset vs dims/dtype
                                const st_dtype = try st.DType.fromString(t.type);
                                var num_elements: u64 = 1;
                                for (t.dims) |d| num_elements *= d;
                                const expected_size = num_elements * st_dtype.getSizeInBytes();

                                if (expected_size != t.size) {
                                    try stderr.print("\nWARNING: Data size mismatch for tensor '{s}'\n", .{t.name});
                                    try stderr.print("  Expected: {} bytes (dims: {any} * type: {s})\n", .{ expected_size, t.dims, t.type });
                                    try stderr.print("  Actual:   {} bytes\n", .{ t.size });
                                    try stderr.flush();
                                }

                                // Write info
                                try gguf.writeTensorInfo(writer, t.name, t.dims, ggml_type, current_offset);

                                // Use the ACTUAL size from Safetensors to advance offset
                                // This guarantees that the GGUF offset points to where we actually write the data
                                const byte_size = t.size;

                                // Calculate next offset with 32-byte alignment
                                var next_offset = current_offset + byte_size;
                                const remainder = next_offset % 32;
                                if (remainder != 0) {
                                    next_offset += (32 - remainder);
                                }
                                current_offset = next_offset;
                            }

                            // IMPORTANT: Flush headers before writing bulk data
                            try writer.flush();

                            // Write padding to align data section start to 32 bytes
                            const header_pos = try out_file.getPos();
                            const padding_len = (32 - (header_pos % 32)) % 32;
                            if (padding_len > 0) {
                                const zeros = [_]u8{0} ** 32;
                                try writer.writeAll(zeros[0..padding_len]);
                                try writer.flush(); // Flush padding
                            }

                            // --- Write Tensor Data ---

                            // 1. Determine start of data section in Safetensors
                            try reader.seekTo(0);
                            const len_bytes = try reader.interface.readAlloc(allocator, 8);
                            defer allocator.free(len_bytes);
                            const st_header_len = std.mem.readInt(u64, len_bytes[0..8], .little);
                            const st_data_begin = 8 + st_header_len;

                            // 2. Buffer for copying
                            var copy_buf = try allocator.alloc(u8, 1 * 1024 * 1024); // 1MB buffer
                            defer allocator.free(copy_buf);

                            // 3. Iterate SORTED tensors again to copy data
                            for (tensors.items) |t| {
                                const size = t.size;

                                // Seek to tensor data in input
                                try reader.seekTo(st_data_begin + t.offset);

                                // Copy data in chunks
                                var left = size;
                                while (left > 0) {
                                    const n = try reader.interface.readSliceShort(copy_buf);
                                    if (n == 0) return error.UnexpectedEof;

                                    const size_left = @min(left, n);
                                    try writer.writeAll(copy_buf[0..size_left]);
                                    left -= size_left;
                                }

                                // Write padding to align next tensor to 32 bytes
                                const padding = (32 - (size % 32)) % 32;
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
