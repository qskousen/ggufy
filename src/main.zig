const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const nw = @import("NullWriter.zig");
const gguf = @import("Gguf.zig");
const clap = @import("clap");

const Command = enum {
    header,
    tree,
    metadata,
    convert,
    template,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                 Display this help and exit.
        \\-d, --datatype <DATATYPE>  When converting, the target datatype (default fp16).
        \\-f, --filetype <FILETYPE>  When converting, the target filetype (default gguf).
        \\-t, --template <FILENAME>  When converting, specify a template to use.
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template
        \\<FILENAME>   The file to use for input
    );

    const parsers = comptime .{
        .DATATYPE = clap.parsers.enumeration(st.DType),
        .FILETYPE = clap.parsers.enumeration(types.FileType),
        .COMMAND = clap.parsers.enumeration(Command),
        .FILENAME = clap.parsers.string,
    };

    // Initialize our diagnostics, which can be used for reporting useful errors.
    // This is optional. You can also pass `.{}` to `clap.parse` if you don't
    // care about the extra information `Diagnostic` provides.
    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        // Report useful error and exit.
        try diag.reportToFile(.stderr(), err);
        return err;
    };
    defer res.deinit();

    if (res.args.help != 0)
        return clap.helpToFile(.stderr(), clap.Help, &params, .{});

    var stderr_buffer: [256]u8 = undefined;
    var err_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &err_writer.interface;
    _ = stderr;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    const command = res.positionals[0] orelse return error.MissingCommand;
    const path = res.positionals[1] orelse return error.MissingModelPath;
    const filetype = res.args.filetype orelse types.FileType.gguf;
    const datatype = res.args.datatype orelse st.DType.F16;
    const template_path = res.args.template;

    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
    defer file.close();

    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(&read_buffer);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const file_type = types.FileType.detect_from_file(&reader.interface, arena_alloc) catch types.FileType.safetensors;
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
                    var template_metadata: ?std.json.ObjectMap = null;
                    if (template_path) |tp| {
                        const t_file = try std.fs.cwd().openFile(tp, .{});
                        defer t_file.close();
                        const t_content = try t_file.readToEndAlloc(arena_alloc, 10 * 1024 * 1024);
                        const t_json = try std.json.parseFromSlice(std.json.Value, arena_alloc, t_content, .{});

                        if (t_json.value.object.get("metadata")) |m| {
                            template_metadata = m.object;
                        }

                        const t_tensors = t_json.value.object.get("tensors") orelse return error.InvalidTemplate;
                        var filtered_tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, f.tensors.items.len);

                        var it = t_tensors.object.iterator();
                        while (it.next()) |entry| {
                            const target_name = entry.key_ptr.*;
                            const target_info = entry.value_ptr.object;

                            // Find source tensor using fuzzy matching (ends_with)
                            var source_tensor: ?types.Tensor = null;
                            for (f.tensors.items) |st_tensor| {
                                if (std.mem.eql(u8, st_tensor.name, target_name) or
                                    (st_tensor.name.len > target_name.len and
                                        st_tensor.name[st_tensor.name.len - target_name.len - 1] == '.' and
                                        std.mem.endsWith(u8, st_tensor.name, target_name)))
                                {
                                    source_tensor = st_tensor;
                                    break;
                                }
                            }

                            if (source_tensor) |st_t| {
                                const target_shape_arr = target_info.get("shape").?.array;
                                var target_dims = try arena_alloc.alloc(usize, target_shape_arr.items.len);
                                var target_elements: u64 = 1;
                                for (target_shape_arr.items, 0..) |item, i| {
                                    // Templates generated from GGUF have reversed dimensions.
                                    // We flip them back here to match the logical row-major shape.
                                    target_dims[target_shape_arr.items.len - 1 - i] = @intCast(item.integer);
                                    target_elements *= @intCast(item.integer);
                                }

                                var source_elements: u64 = 1;
                                for (st_t.dims) |d| source_elements *= d;

                                if (source_elements != target_elements) {
                                    try stdout.print("Error: Tensor {s} shape mismatch. Source elements: {}, Target elements: {}\n", .{ target_name, source_elements, target_elements });
                                    return error.ShapeMismatch;
                                }

                                var new_t = st_t;
                                new_t.dims = target_dims;
                                new_t.name = target_name;
                                try filtered_tensors.append(arena_alloc, new_t);
                            } else {
                                try stdout.print("Warning: Template tensor {s} not found in source file.\n", .{target_name});
                            }
                        }
                        f.tensors = filtered_tensors;
                    }

                    // TODO: if the target datatype is higher precision than the source, print an error warning the user
                    // of the dangers of upcasting not resulting in higher precision/less perplexity and exit, unless
                    // they pass a flag acknowledging that they understand the issues involved and want to proceed.

                    switch (filetype) {
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
                            if (template_metadata) |meta| {
                                metadata_count = @intCast(meta.count());
                                metadata_count += 1;
                            } else if (f.metadata) |meta| {
                                var it = meta.iterator();
                                while (it.next()) |entry| {
                                    if (entry.value_ptr.* == .string) metadata_count += 1;
                                }
                            }

                            try gguf.writeHeader(writer, @intCast(f.tensors.items.len), metadata_count);

                            try gguf.writeMetadataKVU32(writer, "general.alignment", 32);
                            if (template_metadata) |meta| {
                                var it = meta.iterator();
                                while (it.next()) |entry| {
                                    try gguf.writeMetadataKVJson(writer, entry.key_ptr.*, entry.value_ptr.*);
                                }
                            } else {
                                if (f.metadata) |meta| {
                                    var it = meta.iterator();
                                    while (it.next()) |entry| {
                                        if (entry.value_ptr.* == .string) {
                                            try gguf.writeMetadataKVString(writer, entry.key_ptr.*, entry.value_ptr.string);
                                        }
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
                            const zeros = [_]u8{0} ** 32;
                            // why does this work???
                            try writer.writeAll(zeros[0..7]);
                            try writer.flush();
                            try stdout.print("Converted to {s}\n", .{out_filename});
                        },
                        .safetensors => {
                            return error.Unimplimented;
                        },
                    }

                    _ = datatype;
                },
                .template => {
                    return error.Unimplimented;
                },
            }
        },
        .gguf => {
            const file_size = try file.getEndPos();
            try reader.seekTo(4); // Skip GGUF magic
            var f = try gguf.init(&reader.interface, arena_alloc);
            defer f.deinit();

            f.file_size = file_size;
            // The data section start must be aligned.
            // GGUF header ends exactly where the last tensor info was read.
            const header_end = try file.getPos();
            try stdout.print("Header end raw: {}\n", .{header_end});
            const alignment = f.metadata.get("general.alignment") orelse std.json.Value{ .integer = 32 };
            const align_val: u64 = @intCast(alignment.integer);

            // The data section starts at the first multiple of align_val >= header_end
            f.data_offset = ((header_end + align_val - 1) / align_val) * align_val;

            try stdout.print("GGUF format version {}\n", .{f.version});
            try stdout.flush();
            switch (command) {
                .header => {
                    try f.readGgufTensorHeader(stdout);
                },
                .tree => {
                    return error.Unimplemented;
                },
                .metadata => {
                    try f.readGgufMetadata(stdout);
                },
                .convert => {
                    return error.Unimplimented;
                },
                .template => {
                    const out_file = try std.fs.cwd().createFile("template.json", .{ .truncate = true });
                    defer out_file.close();
                    var writer_buffer: [1024]u8 = undefined;
                    var out_writer = out_file.writer(&writer_buffer);
                    var writer = &out_writer.interface;

                    try f.writeTemplate(writer);
                    try writer.flush();
                    try stdout.print("Template exported to template.json\n", .{});
                },
            }
        },
    }
    try stdout.print("Total bytes used in arena allocator: {}\n", .{arena.queryCapacity()});
    try stdout.flush();
}
