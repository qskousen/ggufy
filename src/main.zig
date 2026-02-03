const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const nw = @import("NullWriter.zig");
const gguf = @import("Gguf.zig");
const clap = @import("clap");
const imagearch = @import("ImageArch.zig");

const Command = enum {
    header,
    tree,
    metadata,
    convert,
    template,
};

const QUANTIZATION_THRESHOLD = 1024;

/// Quantization type hierarchy from lowest to highest precision
const QuantizationLevel = enum(u8) {
    q2_k = 0,
    q3_k = 1,
    q4_0 = 2,
    q4_1 = 3,
    q4_k = 4,
    q5_0 = 5,
    q5_1 = 6,
    q5_k = 7,
    q6_k = 8,
    q8_0 = 9,
    f16 = 10,
    bf16 = 11,
    f32 = 12,
    f64 = 13,

    pub fn fromString(s: []const u8) !QuantizationLevel {
        var lower: [12]u8 = [_]u8{0} ** 12;
        return std.meta.stringToEnum(QuantizationLevel, std.ascii.lowerString(&lower, s)) orelse error.UnknownQuantizationType;
    }
};

/// Calculate quantization level for a layer based on sensitivity and aggressiveness
/// sensitivity: 1-100, where 1 = least sensitive, 100 = most sensitive
/// aggressiveness: 1-100, where 1 = aggressive (quantize more), 100 = conservative (keep higher precision)
/// target_level: the base quantization level requested (e.g., q2_k)
/// source_type: the original data type (e.g., f16, f32)
fn calculateQuantizationLevel(
    sensitivity: f32,
    aggressiveness: f32,
    target_level: QuantizationLevel,
    source_type: []const u8,
) !QuantizationLevel {
    // Clamp inputs to valid ranges
    const sens = std.math.clamp(sensitivity, 1.0, 100.0);
    const hard = std.math.clamp(aggressiveness, 1.0, 100.0);

    // Get source and target level indices
    const source_level = try QuantizationLevel.fromString(source_type);
    const target_idx: f32 = @floatFromInt(@intFromEnum(target_level));
    const source_idx: f32 = @floatFromInt(@intFromEnum(source_level));

    // Normalize sensitivity to 0-1 range
    const norm_sens = (sens - 1.0) / 99.0;

    // Adjust the curve based on hardness
    // Lower hardness = more aggressive quantization (stays near target)
    // Higher hardness = more conservative (jumps to higher precision faster)
    const hardness_factor = hard / 100.0;

    // Use an exponential curve that's adjusted by hardness
    // The exponent controls how quickly we move from target to source precision
    const exponent = 0.5 + (hardness_factor * 3.0); // Range: 0.5 to 3.5
    const adjusted_sens = std.math.pow(f32, norm_sens, exponent);

    // Interpolate between target and source quantization levels
    const result_idx = target_idx + (adjusted_sens * (source_idx - target_idx));

    // Round to nearest quantization level
    const rounded_idx: u8 = @intFromFloat(@round(result_idx));

    // Ensure we don't exceed the source precision
    const final_idx = @min(rounded_idx, @intFromEnum(source_level));

    return @enumFromInt(final_idx);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                     Display this help and exit.
        \\-d, --datatype <DATATYPE>      When converting, the target datatype (default fp16).
        \\-f, --filetype <FILETYPE>      When converting, the target filetype: gguf (default), safetensors.
        \\-t, --template <FILENAME>      When converting, specify a template to use.
        \\-o, --output-dir <DIR>         Output directory (default: same as source file).
        \\-n, --output-name <FILENAME>   Output filename without extension (default: source name + datatype).
        \\-j, --threads <INT>            Threads to use when quantizing. Defaults to number of cores - 2.
        \\-a, --aggressiveness <INT>     How aggressive to quantize layers when using sensitivity. 1 is most aggressive, 100 is least.
        \\-x, --skip-sensitivity         Pass this to not use a built-in layer sensitivity file and just blindly quantize to target type.
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template
        \\<FILENAME>   The file to use for input
    );

    const parsers = comptime .{
        .DATATYPE = clap.parsers.enumeration(types.DataType),
        .FILETYPE = clap.parsers.enumeration(types.FileType),
        .COMMAND = clap.parsers.enumeration(Command),
        .FILENAME = clap.parsers.string,
        .DIR = clap.parsers.string,
        .INT = clap.parsers.int(usize, 10),
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

    var stderr_buffer: [256]u8 = undefined;
    var err_writer = std.fs.File.stderr().writer(&stderr_buffer);
    const stderr = &err_writer.interface;
    _ = stderr;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (res.args.help != 0) {
        try stdout.print("ggufy is a tool for LLM model files, particularly for converting between file types.\n\n", .{});
        try stdout.print("Usage: ggufy <COMMAND> <FILENAME> [options]\n\n", .{});
        try stdout.print("Possible commands:\n", .{});
        try stdout.print("  header         Shows header information for the specified file\n", .{});
        try stdout.print("  tree           Output tensor data in a tree format (SafeTensors only)\n", .{});
        try stdout.print("  metadata       Shows metadata information for the specified file\n", .{});
        try stdout.print("  convert        Convert the specified file into a different format or datatype\n", .{});
        try stdout.print("  template       Creates a json template from the specified file\n\n", .{});
        try stdout.print("Options:\n", .{});
        try stdout.flush();
        return clap.helpToFile(.stderr(), clap.Help, &params, .{});
    }

    const command = res.positionals[0] orelse {
        std.log.err("No command given. Use --help to get more information.", .{});
        return;
    };
    const path = res.positionals[1] orelse {
        std.log.err("No model file specified.", .{});
        return;
    };
    const filetype = res.args.filetype orelse types.FileType.gguf;
    const datatype: ?types.DataType = res.args.datatype;
    const template_path = res.args.template;
    const output_dir = res.args.@"output-dir";
    const output_name = res.args.@"output-name";
    const threads = res.args.threads orelse @max(1, try std.Thread.getCpuCount() - 2);
    const skip_sensitivity = res.args.@"skip-sensitivity" != 0;
    const quantization_aggressiveness: f32 = @floatFromInt(res.args.aggressiveness orelse 50);

    const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });

    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(&read_buffer);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const file_type = types.FileType.detect_from_file(&reader.interface, allocator) catch types.FileType.safetensors;
    file.close();
    switch (file_type) {
        .safetensors => {
            var f = try st.init(path, allocator, arena_alloc);
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
                    // tensors array for the converted file
                    var model_tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, f.tensors.items.len);

                    // detect architecture
                    const arch = try imagearch.detectArchFromTensorsOrError(f.tensors.items, allocator);
                    const threshhold = arch.threshhold orelse QUANTIZATION_THRESHOLD;
                    std.log.info("Detected architecture: {s}", .{arch.name});

                    // First, check if there are any tensors that start with "model."
                    // Full checkpoints will have "model." tensors for the unet
                    // along with other tensors for text encoding, vae, etc.
                    // but unet-only safetensors won't have those other tensors or the "model." prefix
                    var has_model_prefix = false;
                    for (f.tensors.items) |t| {
                        if (std.mem.startsWith(u8, t.name, "model.")) {
                            has_model_prefix = true;
                            break;
                        }
                    }

                    // If there are tensors with "model." prefix, filter out those without it
                    // Otherwise, include all tensors
                    for (f.tensors.items) |t| {
                        if (has_model_prefix) {
                            if (std.mem.startsWith(u8, t.name, "model.")) {
                                if (!arch.shouldIgnore(t.name)) {
                                    try model_tensors.append(arena_alloc, try t.dupe(arena_alloc));
                                }
                            } else {
                                std.log.info("Filtering out tensor: {s}", .{t.name});
                            }
                        } else {
                            if (!arch.shouldIgnore(t.name)) {
                                try model_tensors.append(arena_alloc, try t.dupe(arena_alloc));
                            }
                        }
                    }
                    // Strip prefixes from tensor names
                    for (model_tensors.items) |*t| {
                        const stripped_name = imagearch.stripPrefix(t.name);
                        t.name = try arena_alloc.dupe(u8, stripped_name);
                    }

                    var template_metadata: ?std.json.ObjectMap = null;
                    if (template_path) |tp| {
                        std.log.info("Using template {s}", .{tp});
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
                                const target_dims = try arena_alloc.alloc(usize, target_shape_arr.items.len);
                                var target_elements: u64 = 1;
                                for (target_shape_arr.items, 0..) |item, i| {
                                    // Templates generated from GGUF have reversed dimensions.
                                    // We flip them back here to match the logical row-major shape.
                                    //_ = i;
                                    target_dims[target_shape_arr.items.len - 1 - i] = @intCast(item.integer);
                                    target_elements *= @intCast(item.integer);
                                }

                                const target_type = target_info.get("type").?.string;

                                var source_elements: u64 = 1;
                                for (st_t.dims) |d| source_elements *= d;

                                if (source_elements != target_elements) {
                                    std.log.err("Tensor {s} shape mismatch. Source elements: {}, Target elements: {}", .{ target_name, source_elements, target_elements });
                                    return error.ShapeMismatch;
                                }

                                var new_t = st_t;
                                new_t.dims = target_dims;
                                new_t.name = target_name;

                                const ggml_type = try gguf.GgmlType.fromString(target_type);
                                const bs = ggml_type.getBlockSize();
                                if (bs > 1 and source_elements % bs != 0) {
                                    std.log.err("Tensor {s} cannot be quantized to type {s}. Element count {} is not a multiple of block size {}",
                                        .{ target_name, target_type, source_elements, bs });
                                    return error.InvalidSizeForQuantization;
                                }

                                // set the data type of tensor to the passed in datatype
                                new_t.type = @tagName(ggml_type);
                                new_t.size = ggml_type.calcSizeInBytes(target_elements);
                                try filtered_tensors.append(arena_alloc, new_t);
                                std.log.info("Matched target tensor {s} to source tensor {s}, setting to type {s}", .{target_name, source_tensor.?.name, new_t.type});
                            } else {
                                std.log.warn("Warning: Template tensor {s} not found in source file.", .{target_name});
                            }
                        }
                        model_tensors = filtered_tensors;
                    } else {
                        // do we have a sensitivities file?
                        var sensitivity = false;
                        var sensitivities: std.json.Parsed(std.json.Value) = undefined;

                        if (arch.sensitivities.len > 1 and ! skip_sensitivity)  {
                            std.log.debug("Using sensitivities file for {s}", .{arch.name});
                            sensitivities = try std.json.parseFromSlice(std.json.Value, arena_alloc, arch.sensitivities, .{});
                            sensitivity = true;
                        }

                        var offset: u64 = 0;
                        for (model_tensors.items) |*t| {
                            var num_elements: u64 = 1;
                            for (t.dims) |d| num_elements *= d;
                            if (num_elements < threshhold) {
                                // this one is too small to quantize
                                // pass it through unchanged
                            } else if (arch.isHighPrecision(t.name)) {
                                // this one is sensitive to quantization
                                // pass through unchanged
                            } else {
                                if (datatype) |dtype| {
                                    // convert from datatype to ggml type
                                    const ggml_type = try gguf.GgmlType.fromString(@tagName(dtype));
                                    const bs = ggml_type.getBlockSize();
                                    if (bs > 1 and num_elements % bs == 0) {
                                        if (sensitivity) {
                                            const sens_value = sensitivities.value.object.get(t.name);
                                            if (sens_value) |sv| {
                                                const sens: f32 = switch (sv) {
                                                    .float => |fl| @floatCast(fl),
                                                    .integer => |i| @floatFromInt(i),
                                                    else => return error.InvalidSensitivityValue,
                                                };

                                                // Calculate the appropriate quantization level
                                                const target_level = try QuantizationLevel.fromString(@tagName(dtype));
                                                const quant_level = try calculateQuantizationLevel(
                                                    sens,
                                                    quantization_aggressiveness,
                                                    target_level,
                                                    t.type,
                                                );

                                                const final_type_str = @tagName(quant_level);
                                                const final_ggml_type = try gguf.GgmlType.fromString(final_type_str);

                                                std.log.info("Layer {s}: sensitivity={d:.1}, hardness={d}, {s} -> {s}",
                                                    .{t.name, sens, quantization_aggressiveness, @tagName(dtype), final_type_str});

                                                t.type = final_type_str;
                                                t.size = final_ggml_type.calcSizeInBytes(num_elements);
                                            } else {
                                                std.log.warn("No sensitivity data for layer {s}, using target type", .{t.name});
                                                t.type = @tagName(ggml_type);
                                                t.size = ggml_type.calcSizeInBytes(num_elements);
                                            }
                                        } else {
                                            // set the data type of tensor to the passed in datatype
                                            std.log.debug("Will convert tensor {s} to type {s}", .{t.name, @tagName(ggml_type)});
                                            t.type = @tagName(ggml_type);
                                            t.size = ggml_type.calcSizeInBytes(num_elements);
                                        }
                                    } else {
                                        // we can't quantize this tensor with this blocksize?
                                        std.log.warn("Cannot convert tensor {s} to type {s} because {} is not a multiple of blocksize {}",
                                            .{t.name, @tagName(ggml_type), num_elements, bs});
                                    }
                                }
                            }
                            // comfyui gguf can't do f64 currently, need to downcast these to f32
                            // TODO: make this optional
                            if (std.mem.eql(u8, t.type, "f64") or std.mem.eql(u8, t.type, "F64")) {
                                std.log.info("Downcasting unsupported f64 to f32 for tensor {s}", .{t.name});
                                t.type = "f32";
                                const fat_type = try gguf.GgmlType.fromString(t.type);
                                t.size = fat_type.calcSizeInBytes(num_elements);
                            }
                            try stdout.print("Calculated size {} for type {s} with num elements {} with dims [", .{ t.size, t.type, num_elements });
                            for (t.dims) |d| try stdout.print("{}, ", .{d});
                            try stdout.print("]\n", .{});
                            try stdout.flush();
                            // TODO: make this work with configurable alignment!
                            const padding_len = (32 - (t.size % 32)) % 32;
                            t.offset = offset;
                            offset += t.size + padding_len;
                        }
                    }

                    // Initializes a map to hold extra metadata generated by the fix.
                    var extra_metadata = std.StringArrayHashMap(std.json.Value).init(arena_alloc);
                    const REARRANGE_THRESHOLD = 512;

                    if (arch.shape_fix) {
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
                                var new_dims = try arena_alloc.alloc(usize, 2);
                                new_dims[0] = n_elements / 256;
                                new_dims[1] = 256;
                                t.dims = new_dims;

                                std.log.info("Applied shape fix to {s}: new shape {{ {}, {} }}", .{ t.name, new_dims[0], new_dims[1] });
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
                                const dtype_str = @tagName(datatype orelse types.DataType.F16);
                                break :blk try std.fmt.allocPrint(arena_alloc, "{s}-{s}", .{ stem, dtype_str });
                            };

                            const out_filename = try std.fs.path.join(arena_alloc, &[_][]const u8{ dir_path, try std.fmt.allocPrint(arena_alloc, "{s}.gguf", .{base_name}) });

                            var out_gguf = try gguf.init(out_filename, allocator, arena_alloc, true);
                            defer out_gguf.deinit();

                            out_gguf.tensors = model_tensors;

                            // standard metadata
                            try out_gguf.metadata.put(try arena_alloc.dupe(u8, "general.architecture"), .{ .string = arch.name });
                            try out_gguf.metadata.put(try arena_alloc.dupe(u8, "general.quantization_version"), .{ .integer = 2 });
                            // TODO: determine from the target dtype
                            try out_gguf.metadata.put(try arena_alloc.dupe(u8, "general.file_type"), .{ .integer = 7 });

                            // for all following metadata we use getOrPut so we don't overwrite
                            // any metadata from the template
                            if (template_metadata) |meta| {
                                var it = meta.iterator();
                                while (it.next()) |entry| {
                                    if (!out_gguf.metadata.contains(entry.key_ptr.*)) {
                                        try out_gguf.metadata.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
                                    }
                                }
                            } else {
                                // any metadata from the source file
                                if (f.metadata) |meta| {
                                    var it = meta.iterator();
                                    while (it.next()) |entry| {
                                        if (!out_gguf.metadata.contains(entry.key_ptr.*)) {
                                            try out_gguf.metadata.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
                                        }
                                    }
                                }
                            }

                            // any extra metadata such as shape fix
                            var extra_it = extra_metadata.iterator();
                            while (extra_it.next()) |entry| {
                                if (!out_gguf.metadata.contains(entry.key_ptr.*)) {
                                    try out_gguf.metadata.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
                                }
                            }

                            try out_gguf.saveWithSTData(&f, stdout, threads);

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
            var f = try gguf.init(path, allocator, arena_alloc, false);
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
    //try stdout.print("Total bytes used in arena allocator: {}\n", .{arena.queryCapacity()});
    try stdout.flush();
}
