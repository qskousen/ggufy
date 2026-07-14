const std = @import("std");
const ggufy = @import("ggufy");
const st = ggufy.safetensor;
const types = ggufy.types;
const gguf = ggufy.gguf;
const clap = @import("clap");
const imagearch = ggufy.imageArch;
const conv = ggufy.convert;

const build_options = @import("build_options");

const Command = enum {
    header,
    tree,
    metadata,
    convert,
    template,
    names,
    sensitivities,
    version,
};

/// Format a byte count into a human-readable string (binary units) in `buf`.
fn formatBytes(bytes: u64, buf: []u8) []const u8 {
    const units = [_][]const u8{ "B", "KiB", "MiB", "GiB", "TiB" };
    var value: f64 = @floatFromInt(bytes);
    var unit: usize = 0;
    while (value >= 1024.0 and unit < units.len - 1) : (unit += 1) value /= 1024.0;
    return std.fmt.bufPrint(buf, "{d:.2} {s}", .{ value, units[unit] }) catch buf[0..0];
}

/// Predict and print the final output size for a convert, without writing anything.
fn reportPredictedSize(
    f: anytype,
    opts: conv.ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    stdout: *std.Io.Writer,
) !void {
    const size = conv.predictOutputSize(f, opts, allocator, arena_alloc) catch |err| {
        if (err == error.UnknownArchitecture) {
            std.log.err("Architecture not recognized. Pass --allow-unknown-arch (-u) to calculate size anyway. Results may be suboptimal.", .{});
            return;
        }
        return err;
    };
    var buf: [32]u8 = undefined;
    try stdout.print("Estimated output size: {s} ({d} bytes)\n", .{ formatBytes(size, &buf), size });
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const start_ts = std.Io.Clock.Timestamp.now(io, .awake);
    const allocator = init.gpa;

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                     Display this help and exit.
        \\-d, --datatype <DATATYPE>      When converting, the target datatype (default fp16).
        \\-f, --filetype <FILETYPE>      When converting, the target filetype: gguf (default), safetensors.
        \\-t, --template <FILENAME>      When converting, specify a template to use.
        \\-o, --output-dir <DIR>         Output directory (default: same as source file).
        \\-n, --output-name <FILENAME>   Output filename without extension (default: source name + datatype).
        \\-j, --threads <INT>            Threads to use when quantizing. Defaults to number of cores.
        \\-a, --aggressiveness <INT>     How aggressively to quantize layers when using sensitivity. 100 is most aggressive, 1 is least.
        \\-x, --skip-sensitivity         Pass this to not use a built-in layer sensitivity file and just blindly quantize to target type.
        \\-s, --sensitivities <FILENAME> Path to a sensitivities JSON file to use (overrides built-in sensitivities) Sensitivities are only used for GGUF model output.
        \\-q, --use-quant-types <QTYPES> Quantization families to use with sensitivity (e.g. "k", "0,k", "0,1,k"). Default: match datatype.
        \\-m, --model-only               When output is safetensors, convert only the main model (UNet/transformer). Ignored for GGUF output.
        \\-u, --allow-unknown-arch       Allow converting files with unrecognized architectures. Results may be suboptimal.
        \\-U, --allow-upscale            Allow converting from a lower-precision (quantized/FP8) source to a higher-precision target. The extra bits are fill-in; no quality is recovered.
        \\-A, --arch <NAME>              Set the architecture name written to the GGUF metadata (GGUF output only). Free-form; does not affect conversion behaviour.
        \\-R, --stochastic-rounding <SEED> Seed for INT4_CONVROT_SR stochastic rounding. Omit for the built-in default seed; pass 0 to disable (deterministic, for comparison). Ignored by other types.
        \\-c, --calculate-size           With convert: compute and print the exact final output size without writing any file.
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template, version
        \\<FILENAME>   The file to use for input (not required for the version command)
    );

    const parsers = comptime .{
        .DATATYPE = clap.parsers.enumeration(types.DataType),
        .FILETYPE = clap.parsers.enumeration(types.FileType),
        .COMMAND = clap.parsers.enumeration(Command),
        .FILENAME = clap.parsers.string,
        .DIR = clap.parsers.string,
        .INT = clap.parsers.int(usize, 10),
        .QTYPES = clap.parsers.string,
        .NAME = clap.parsers.string,
        .SEED = clap.parsers.int(u64, 10),
    };

    // Initialize our diagnostics, which can be used for reporting useful errors.
    // This is optional. You can also pass `.{}` to `clap.parse` if you don't
    // care about the extra information `Diagnostic` provides.
    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, parsers, init.minimal.args, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        // Report useful error and exit.
        try diag.reportToFile(io, std.Io.File.stderr(), err);
        return err;
    };
    defer res.deinit();

    var stderr_buffer: [256]u8 = undefined;
    var err_writer = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &err_writer.interface;
    _ = stderr;

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    if (res.args.help != 0) {
        try stdout.print("ggufy is a tool for LLM model files, particularly for converting between file types.\n\n", .{});
        try stdout.print("Usage: ggufy <COMMAND> <FILENAME> [options]\n\n", .{});
        try stdout.print("Possible commands:\n", .{});
        try stdout.print("  header         Shows header information for the specified file\n", .{});
        try stdout.print("  tree           Output tensor data in a tree format (SafeTensors only)\n", .{});
        try stdout.print("  metadata       Shows metadata information for the specified file\n", .{});
        try stdout.print("  convert        Convert the specified file into a different format or datatype\n", .{});
        try stdout.print("  template       Creates a json template from the specified file\n", .{});
        try stdout.print("  names          Dump tensor names as a JSON array (for test fixtures)\n", .{});
        try stdout.print("  sensitivities  Generate a sensitivities JSON template from the specified file\n", .{});
        try stdout.print("  version        Print version information\n\n", .{});
        try stdout.print("Options:\n", .{});
        try stdout.flush();
        return clap.helpToFile(io, std.Io.File.stderr(), clap.Help, &params, .{});
    }

    const command = res.positionals[0] orelse {
        std.log.err("No command given. Use --help to get more information.", .{});
        return;
    };

    if (command == .version) {
        try stdout.print("ggufy {s}\n", .{build_options.version});
        try stdout.flush();
        return;
    }

    const path = res.positionals[1] orelse {
        std.log.err("No model file specified.", .{});
        return;
    };
    const filetype = res.args.filetype orelse types.FileType.gguf;
    const datatype: ?types.DataType = res.args.datatype;
    const template_path = res.args.template;
    const output_dir = res.args.@"output-dir";
    const output_name = res.args.@"output-name";
    const threads = res.args.threads orelse @max(1, try std.Thread.getCpuCount());
    const skip_sensitivity = res.args.@"skip-sensitivity" != 0;
    const quantization_aggressiveness: f32 = @floatFromInt(res.args.aggressiveness orelse 50);
    const sensitivities_path = res.args.sensitivities;

    const model_only = res.args.@"model-only" != 0;
    const allow_unknown_arch = res.args.@"allow-unknown-arch" != 0;
    const allow_upscale = res.args.@"allow-upscale" != 0;
    const arch_override = res.args.arch;
    const calculate_size = res.args.@"calculate-size" != 0;

    const allowed_quant_families: ?conv.QuantizationFamilies = if (res.args.@"use-quant-types") |s|
        conv.QuantizationFamilies.parse(s) catch {
            std.log.err("Invalid --use-quant-types value '{s}'. Use a comma-separated list of: 0, 1, k", .{s});
            return;
        }
    else
        null;

    // Shared conversion options — used by both the real convert path and the
    // --calculate-size prediction, so the predicted size matches what convert writes.
    const convert_opts = conv.ConvertOptions{
        .io = io,
        .path = path,
        .filetype = filetype,
        .datatype = datatype,
        .template_path = template_path,
        .output_dir = output_dir,
        .output_name = output_name,
        .threads = threads,
        .skip_sensitivity = skip_sensitivity,
        .quantization_aggressiveness = quantization_aggressiveness,
        .sensitivities_path = sensitivities_path,
        .allowed_quant_families = allowed_quant_families,
        .model_only = model_only,
        .allow_unknown_arch = allow_unknown_arch,
        .allow_upscale = allow_upscale,
        .arch_override = arch_override,
        .stochastic_rounding = res.args.@"stochastic-rounding",
    };

    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });

    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(io, &read_buffer);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const file_type = types.FileType.detect_from_file(&reader.interface, allocator) catch types.FileType.safetensors;
    file.close(io);
    switch (file_type) {
        .safetensors => {
            var f = try st.init(path, io, allocator, arena_alloc, false, false);
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
                    if (calculate_size) {
                        try reportPredictedSize(&f, convert_opts, allocator, arena_alloc, stdout);
                    } else conv.convert(&f, convert_opts, allocator, arena_alloc) catch |err| {
                        if (err == error.UnknownArchitecture) {
                            std.log.err("Architecture not recognized. Pass --allow-unknown-arch (-u) to convert anyway. Results may be suboptimal.", .{});
                            return;
                        }
                        if (err == error.UpscalingNotAllowed) return;
                        return err;
                    };
                },
                .template => {
                    const out_path = if (output_name) |n|
                        try std.fmt.allocPrint(arena_alloc, "{s}.json", .{n})
                    else
                        "template.json";
                    const out_file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true });
                    defer out_file.close(io);
                    var writer_buffer: [8192]u8 = undefined;
                    var out_writer = out_file.writer(io, &writer_buffer);
                    var writer = &out_writer.interface;
                    const arch_ptr = try imagearch.detectArchFromTensors(f.tensors.items, allocator);
                    try conv.writeTemplateFromTensors(
                        f.tensors.items,
                        arch_ptr,
                        true, // reverse dims: safetensors → GGUF template convention
                        writer,
                        arena_alloc,
                    );
                    try writer.flush();
                    std.log.info("Template exported to {s}", .{out_path});
                },
                .sensitivities => {
                    const out_path = if (output_name) |n|
                        try std.fmt.allocPrint(arena_alloc, "{s}.json", .{n})
                    else
                        "sensitivities.json";
                    const out_file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true });
                    defer out_file.close(io);
                    var writer_buffer: [8192]u8 = undefined;
                    var out_writer = out_file.writer(io, &writer_buffer);
                    var writer = &out_writer.interface;
                    const arch_ptr = try imagearch.detectArchFromTensors(f.tensors.items, allocator);
                    const threshold: u64 = if (arch_ptr) |a| (a.threshhold orelse conv.QUANTIZATION_THRESHOLD) else conv.QUANTIZATION_THRESHOLD;
                    try conv.generateSensitivitiesFromTensors(
                        f.tensors.items,
                        arch_ptr,
                        threshold,
                        writer,
                        arena_alloc,
                    );
                    try writer.flush();
                    std.log.info("Sensitivities exported to {s}", .{out_path});
                },
                .names => {
                    const name_list = try allocator.alloc([]const u8, f.tensors.items.len);
                    defer allocator.free(name_list);
                    for (f.tensors.items, 0..) |t, i| name_list[i] = t.name;
                    const json = try std.json.Stringify.valueAlloc(allocator, name_list, .{ .whitespace = .indent_2 });
                    defer allocator.free(json);
                    try stdout.writeAll(json);
                    try stdout.writeByte('\n');
                },
                .version => unreachable,
            }
        },
        .gguf => {
            var f = try gguf.init(path, io, allocator, arena_alloc, false);
            defer f.deinit();

            std.log.info("GGUF format version {}", .{f.version});
            switch (command) {
                .header => {
                    try f.readGgufTensorHeader();
                },
                .tree => {
                    return error.Unimplemented;
                },
                .metadata => {
                    try f.readGgufMetadata(stdout);
                },
                .convert => {
                    if (calculate_size) {
                        try reportPredictedSize(&f, convert_opts, allocator, arena_alloc, stdout);
                    } else conv.convert(&f, convert_opts, allocator, arena_alloc) catch |err| {
                        if (err == error.UnknownArchitecture) {
                            std.log.err("Architecture not recognized. Pass --allow-unknown-arch (-u) to convert anyway. Results may be suboptimal.", .{});
                            return;
                        }
                        if (err == error.UpscalingNotAllowed) return;
                        return err;
                    };
                },
                .names => {
                    const name_list = try allocator.alloc([]const u8, f.tensors.items.len);
                    defer allocator.free(name_list);
                    for (f.tensors.items, 0..) |t, i| name_list[i] = t.name;
                    const json = try std.json.Stringify.valueAlloc(allocator, name_list, .{ .whitespace = .indent_2 });
                    defer allocator.free(json);
                    try stdout.writeAll(json);
                    try stdout.writeByte('\n');
                },
                .template => {
                    const out_path = if (output_name) |n|
                        try std.fmt.allocPrint(arena_alloc, "{s}.json", .{n})
                    else
                        "template.json";
                    const out_file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true });
                    defer out_file.close(io);
                    var writer_buffer: [8192]u8 = undefined;
                    var out_writer = out_file.writer(io, &writer_buffer);
                    var writer = &out_writer.interface;
                    try f.writeTemplate(writer);
                    try writer.flush();
                    std.log.info("Template exported to {s}", .{out_path});
                },
                .sensitivities => {
                    const out_path = if (output_name) |n|
                        try std.fmt.allocPrint(arena_alloc, "{s}.json", .{n})
                    else
                        "sensitivities.json";
                    const out_file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true });
                    defer out_file.close(io);
                    var writer_buffer: [8192]u8 = undefined;
                    var out_writer = out_file.writer(io, &writer_buffer);
                    var writer = &out_writer.interface;
                    const arch_ptr = try imagearch.detectArchFromTensors(f.tensors.items, allocator);
                    const threshold: u64 = if (arch_ptr) |a| (a.threshhold orelse conv.QUANTIZATION_THRESHOLD) else conv.QUANTIZATION_THRESHOLD;
                    try conv.generateSensitivitiesFromTensors(
                        f.tensors.items,
                        arch_ptr,
                        threshold,
                        writer,
                        arena_alloc,
                    );
                    try writer.flush();
                    std.log.info("Sensitivities exported to {s}", .{out_path});
                },
                .version => unreachable,
            }
        },
    }
    try stdout.flush();
    std.log.info("Total bytes used in arena allocator: {}", .{arena.queryCapacity()});
    const elapsed = start_ts.durationTo(std.Io.Clock.Timestamp.now(io, .awake));
    std.log.info("Completed in {d:.2} seconds.", .{@as(f64, @floatFromInt(elapsed.raw.nanoseconds)) / std.time.ns_per_s});
}
