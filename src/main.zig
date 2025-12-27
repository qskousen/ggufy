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

fn stripPrefix(name: []const u8, allocator: std.mem.Allocator) ![]const u8 {
    // Prefixes for mixed state dict
    const mixed_prefixes = [_][]const u8{
        "model.diffusion_model.",
        "model.",
    };

    // Prefixes for uniform state dict (would need to check if ALL tensors have this)
    // For now, we'll just handle mixed prefixes
    const uniform_prefixes = [_][]const u8{
        "net.",
    };

    // Check mixed prefixes (any tensor can have these)
    for (mixed_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            const stripped = name[prefix.len..];
            return try allocator.dupe(u8, stripped);
        }
    }

    // Check uniform prefixes
    for (uniform_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            const stripped = name[prefix.len..];
            return try allocator.dupe(u8, stripped);
        }
    }

    // No prefix found, return original name
    return try allocator.dupe(u8, name);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                 Display this help and exit.
        \\-d, --datatype <DATATYPE>  When converting, the target datatype (default fp16).
        \\-f, --filetype <FILETYPE>  When converting, the target filetype: gguf (default), safetensors.
        \\-t, --template <FILENAME>  When converting, specify a template to use.
        \\-o, --output-dir <DIR>     Output directory (default: same as source file).
        \\-n, --output-name <FILENAME>   Output filename without extension (default: source name + datatype).
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template
        \\<FILENAME>   The file to use for input
    );

    const parsers = comptime .{
        .DATATYPE = clap.parsers.enumeration(gguf.GgmlType),
        .FILETYPE = clap.parsers.enumeration(types.FileType),
        .COMMAND = clap.parsers.enumeration(Command),
        .FILENAME = clap.parsers.string,
        .DIR = clap.parsers.string,
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
    const datatype = res.args.datatype orelse gguf.GgmlType.f16;
    const template_path = res.args.template;
    const output_dir = res.args.@"output-dir";
    const output_name = res.args.@"output-name";

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
                    // First, filter out any tensors that don't start with "model."
                    var model_tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, f.tensors.items.len);
                    for (f.tensors.items) |t| {
                        if (std.mem.startsWith(u8, t.name, "model.")) {
                            try model_tensors.append(arena_alloc, t);
                        } else {
                            try stdout.print("Filtering out tensor: {s}\n", .{t.name});
                        }
                    }
                    // Strip prefixes from tensor names
                    for (model_tensors.items) |*t| {
                        const stripped_name = try stripPrefix(t.name, arena_alloc);
                        t.name = stripped_name;
                    }

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
                        var filtered_tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, model_tensors.items.len);

                        var it = t_tensors.object.iterator();
                        while (it.next()) |entry| {
                            const target_name = entry.key_ptr.*;
                            const target_info = entry.value_ptr.object;

                            // Find source tensor using fuzzy matching (ends_with)
                            var source_tensor: ?types.Tensor = null;
                            for (model_tensors.items) |st_tensor| {
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
                        model_tensors = filtered_tensors;
                    }

                    // Shape Fix Logic (similar to convert.py for SDXL/SD1)
                    // Initializes a map to hold extra metadata generated by the fix.
                    var extra_metadata = std.StringArrayHashMap(std.json.Value).init(arena_alloc);
                    const REARRANGE_THRESHOLD = 512;

                    // NOTE: This should ideally be enabled only for SDXL/SD1 architectures.
                    // You can make this a CLI flag or add architecture detection later.
                    const enable_shape_fix = true;

                    if (enable_shape_fix) {
                        for (model_tensors.items) |*t| {
                            var n_elements: u64 = 1;
                            for (t.dims) |d| n_elements *= @intCast(d);

                            const n_dims = t.dims.len;
                            const last_dim = if (n_dims > 0) t.dims[n_dims - 1] else 0;

                            // Criteria for conversion:
                            // 1. More than 1 dimension
                            // 2. Total params >= 512
                            // 3. Total params divisible by 256
                            // 4. Last dimension NOT divisible by 256
                            if (n_dims > 1 and n_elements >= REARRANGE_THRESHOLD and n_elements % 256 == 0 and @mod(last_dim, 256) != 0) {
                                // Store original shape in metadata as [d0, d1, ...]
                                var orig_shape_arr = std.json.Array.init(arena_alloc);
                                for (t.dims) |d| {
                                    try orig_shape_arr.append(.{ .integer = @intCast(d) });
                                }
                                const key = try std.fmt.allocPrint(arena_alloc, "comfy.gguf.orig_shape.{s}", .{t.name});
                                try extra_metadata.put(key, .{ .array = orig_shape_arr });

                                // Reshape tensor to (N/256, 256)
                                // Note: This changes the logical dimensions stored in GGUF.
                                // It does NOT require reordering the raw bytes on disk (row-major standard).
                                var new_dims = try arena_alloc.alloc(usize, 2);
                                new_dims[0] = n_elements / 256;
                                new_dims[1] = 256;
                                t.dims = new_dims;

                                try stdout.print("Applied shape fix to {s}: new shape {{ {}, {} }}\n", .{ t.name, new_dims[0], new_dims[1] });
                            }
                        }
                    }
                    try stdout.flush();

                    // Sort tensors
                    std.sort.block(types.Tensor, model_tensors.items, {}, struct {
                        fn lessThan(_: void, a: types.Tensor, b: types.Tensor) bool {
                            return std.mem.lessThan(u8, a.name, b.name);
                        }
                    }.lessThan);

                    // TODO: if the target datatype is higher precision than the source, print an error warning the user
                    // of the dangers of upcasting not resulting in higher precision/less perplexity and exit, unless
                    // they pass a flag acknowledging that they understand the issues involved and want to proceed.

                    switch (filetype) {
                        .gguf => {
                            // Determine output directory
                            const dir_path = if (output_dir) |od|
                                od
                            else
                                std.fs.path.dirname(path) orelse ".";

                            // Determine output filename
                            const base_name = if (output_name) |on|
                                on
                            else blk: {
                                const stem = std.fs.path.stem(path);
                                // Append datatype to filename
                                const dtype_str = @tagName(datatype);
                                break :blk try std.fmt.allocPrint(arena_alloc, "{s}-{s}", .{ stem, dtype_str });
                            };

                            const out_filename = try std.fs.path.join(arena_alloc, &[_][]const u8{ dir_path, try std.fmt.allocPrint(arena_alloc, "{s}.gguf", .{base_name}) });

                            var out_gguf = try gguf.init(out_filename, arena_alloc, true);
                            defer out_gguf.deinit();

                            out_gguf.tensors = model_tensors;

                            // standard metadata
                            // TODO: determine from the source type
                            try out_gguf.metadata.put("general.architecture", .{ .string = "sdxl" });
                            try out_gguf.metadata.put("general.quantization_version", .{ .integer = 2 });
                            // TODO: determine from the target dtype
                            try out_gguf.metadata.put("general.file_type", .{ .integer = 1 });

                            // for all following metadata we use getOrPut so we don't overwrite
                            // any metadata from the template
                            if (template_metadata) |meta| {
                                var it = meta.iterator();
                                while (it.next()) |entry| {
                                    if (!out_gguf.metadata.contains(entry.key_ptr.*)) {
                                        try out_gguf.metadata.put(entry.key_ptr.*, entry.value_ptr.*);
                                    }
                                }
                            } else {
                                // any metadata from the source file
                                if (f.metadata) |meta| {
                                    var it = meta.iterator();
                                    while (it.next()) |entry| {
                                        if (!out_gguf.metadata.contains(entry.key_ptr.*)) {
                                            try out_gguf.metadata.put(entry.key_ptr.*, entry.value_ptr.*);
                                        }
                                    }
                                }
                            }

                            // any extra metadata such as shape fix
                            var extra_it = extra_metadata.iterator();
                            while (extra_it.next()) |entry| {
                                if (!out_gguf.metadata.contains(entry.key_ptr.*)) {
                                    try out_gguf.metadata.put(entry.key_ptr.*, entry.value_ptr.*);
                                }
                            }

                            try out_gguf.saveWithSTData(&f, stdout);

                            try stdout.print("Converted to {s}\n", .{out_filename});
                        },
                        .safetensors => {
                            return error.Unimplimented;
                        },
                    }
                },
                .template => {
                    return error.Unimplimented;
                },
            }
        },
        .gguf => {
            var f = try gguf.init(path, arena_alloc, false);
            defer f.deinit();

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
