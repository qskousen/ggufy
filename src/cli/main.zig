const std = @import("std");
const ggufy = @import("ggufy");
const st = ggufy.safetensor;
const types = ggufy.types;
const gguf = ggufy.gguf;
const clap = @import("clap");
const imagearch = ggufy.imageArch;
const conv = ggufy.convert;
const calib = ggufy.calibrate;
const sens = ggufy.sensitivity;

const build_options = @import("build_options");

const Command = enum {
    header,
    tree,
    metadata,
    convert,
    template,
    names,
    sensitivities,
    calibrate,
    sensitivity,
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

/// Dump the tensor list as a JSON test fixture. With `with_shapes`, each entry
/// becomes {"name":…,"shape":[…]} — required by architectures that are only
/// separable by dimension (see `Arch.shape_detect` in ImageArch.zig).
fn dumpNames(
    tensors: []const types.Tensor,
    with_shapes: bool,
    allocator: std.mem.Allocator,
    stdout: *std.Io.Writer,
) !void {
    const json = if (with_shapes) blk: {
        const Entry = struct { name: []const u8, shape: []const usize };
        const entries = try allocator.alloc(Entry, tensors.len);
        defer allocator.free(entries);
        for (tensors, 0..) |t, i| entries[i] = .{ .name = t.name, .shape = t.dims };
        break :blk try std.json.Stringify.valueAlloc(allocator, entries, .{ .whitespace = .indent_1 });
    } else blk: {
        const names = try allocator.alloc([]const u8, tensors.len);
        defer allocator.free(names);
        for (tensors, 0..) |t, i| names[i] = t.name;
        break :blk try std.json.Stringify.valueAlloc(allocator, names, .{ .whitespace = .indent_2 });
    };
    defer allocator.free(json);
    try stdout.writeAll(json);
    try stdout.writeByte('\n');
}

/// Human-readable duration, for the capture summary.
fn formatDuration(ns: u64, buf: []u8) []const u8 {
    const s = @as(f64, @floatFromInt(ns)) / std.time.ns_per_s;
    if (s < 90) return std.fmt.bufPrint(buf, "{d:.1}s", .{s}) catch buf[0..0];
    if (s < 5400) return std.fmt.bufPrint(buf, "{d:.1}m", .{s / 60}) catch buf[0..0];
    return std.fmt.bufPrint(buf, "{d:.2}h", .{s / 3600}) catch buf[0..0];
}

/// `ggufy calibrate <dit> -e <text-encoder> -v <vae> [...]` — run the prompt set
/// through real diffusion forwards and write the calibration cache.
fn runCalibrate(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    path: []const u8,
    args: anytype,
    stdout: *std.Io.Writer,
) !void {
    const text_encoder = args.@"text-encoder" orelse {
        std.log.err("calibrate needs the text encoder checkpoint: -e/--text-encoder <path>", .{});
        return error.MissingArgument;
    };
    const vae = args.vae orelse {
        std.log.err("calibrate needs the VAE checkpoint: -v/--vae <path>", .{});
        return error.MissingArgument;
    };

    const backend_name = args.backend orelse "cpu";
    const backend = calib.Backend.fromStr(backend_name) orelse {
        std.log.err("Unknown backend '{s}'. Use one of: cpu, vulkan, zig-cuda, cuda", .{backend_name});
        return error.InvalidArgument;
    };

    // The prompt set is part of the measurement, so a user-supplied one is held
    // to the same shape as the built-in and carries its own id into the cache.
    const prompt_json = if (args.prompts) |p|
        try std.Io.Dir.cwd().readFileAlloc(io, p, arena_alloc, .unlimited)
    else
        calib.default_prompts_json;
    var set = calib.parsePromptSet(allocator, prompt_json) catch |err| {
        std.log.err("Could not read the prompt set: {t}. Expected {{\"id\": …, \"prompts\": [ … ]}}", .{err});
        return err;
    };
    defer set.deinit();

    const stem = std.fs.path.stem(path);
    const out_name = if (args.@"output-name") |n|
        try std.fmt.allocPrint(arena_alloc, "{s}.safetensors", .{n})
    else
        try std.fmt.allocPrint(arena_alloc, "{s}.calib.safetensors", .{stem});
    const out_path = if (args.@"output-dir") |d|
        try std.fs.path.join(arena_alloc, &.{ d, out_name })
    else
        out_name;

    // TensorPencil's load/step notes go to stdout as they happen: a CPU capture
    // is long enough that silence would be indistinguishable from a hang.
    const opts = calib.Options{
        .dit_path = path,
        .text_encoder_path = text_encoder,
        .vae_path = vae,
        .output_path = out_path,
        .backend = backend,
        .resolution = args.resolution orelse 512,
        .steps = args.steps orelse 4,
        .seed = args.seed orelse 0,
        .sample_rows = args.rows orelse 64,
        .buckets = args.buckets orelse 3,
        .arch = args.arch orelse "",
        .producer = try std.fmt.allocPrint(arena_alloc, "ggufy {s}", .{build_options.version}),
        .log = stdout,
    };

    try stdout.print(
        "capture: {d} prompts x {d} steps at {d}^2 on {s}, {d} rows x {d} buckets\n",
        .{ set.prompts.len, opts.steps, opts.resolution, @tagName(backend), opts.sample_rows, opts.buckets },
    );
    try stdout.flush();

    const summary = try calib.run(allocator, io, opts, set);

    var tbuf: [32]u8 = undefined;
    var sbuf: [32]u8 = undefined;
    try stdout.print(
        "\ncaptured {d} layers, {d} tokens, in {s}\nwrote {s} ({s})\n",
        .{
            summary.layers,
            summary.tokens,
            formatDuration(summary.elapsed_ns, &tbuf),
            out_path,
            formatBytes(summary.cache_bytes, &sbuf),
        },
    );
    if (summary.untagged > 0) {
        // Not fatal — a caller may run its own untagged GEMMs — but a large count
        // means the loader is not tagging and the cache is thinner than it looks.
        try stdout.print("note: {d} GEMMs had no tensor tag and were not recorded\n", .{summary.untagged});
    }
}

/// `ggufy sensitivity <model> -C <cache> [...]` — level 1: measure per-layer
/// output error on the captured activations and write a measured sensitivities
/// JSON the converter can route on.
fn runSensitivity(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    path: []const u8,
    args: anytype,
    threads: usize,
    stdout: *std.Io.Writer,
) !void {
    const calib_path = args.calib orelse {
        std.log.err("sensitivity needs a calibration cache: -C/--calib <path> (produce one with `ggufy calibrate`)", .{});
        return error.MissingArgument;
    };

    const formats: []const sens.Format = if (args.formats) |spec|
        sens.parseFormats(arena_alloc, spec) catch |err| {
            std.log.err("Bad --formats value '{s}': {t}", .{ spec, err });
            return err;
        }
    else
        &sens.default_formats;

    const opts = sens.Options{
        .model_path = path,
        .calib_path = calib_path,
        .formats = formats,
        .bucket = args.bucket,
        .kernel_arm = args.@"no-kernel-arm" == 0,
        .imatrix_arm = args.@"no-imatrix-arm" == 0,
        .max_layers = args.@"max-layers",
        .threads = threads,
        .log = stdout,
    };

    var report = try sens.run(allocator, io, opts);
    defer report.deinit();

    try stdout.writeByte('\n');
    try sens.writeMarkdown(&report, stdout, args.top orelse 25);

    // The hand-authored file, when given, is the thing being checked — this is
    // the first real audit of scores nobody ever measured.
    if (args.sensitivities) |heur_path| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, heur_path, arena_alloc, .unlimited);
        const parsed = try std.json.parseFromSlice(std.json.Value, arena_alloc, text, .{});
        defer parsed.deinit();
        try sens.writeHeuristicDiff(&report, &parsed.value, stdout, args.top orelse 25);
    }
    try stdout.flush();

    const stem = if (args.@"output-name") |n|
        n
    else if (report.arch.len > 0)
        report.arch
    else
        std.fs.path.stem(path);

    const json_path = try joinOut(arena_alloc, args.@"output-dir", stem, ".json");
    try writeReportFile(io, json_path, &report, sens.writeSensitivitiesJson);
    const csv_path = try joinOut(arena_alloc, args.@"output-dir", stem, ".csv");
    try writeReportFile(io, csv_path, &report, sens.writeCsv);

    try stdout.print("\nwrote {s} ({d} layers) and {s}\n", .{ json_path, report.layers.len, csv_path });
}

fn joinOut(arena_alloc: std.mem.Allocator, dir: ?[]const u8, stem: []const u8, ext: []const u8) ![]const u8 {
    const name = try std.fmt.allocPrint(arena_alloc, "{s}{s}", .{ stem, ext });
    return if (dir) |d| std.fs.path.join(arena_alloc, &.{ d, name }) else name;
}

fn writeReportFile(
    io: std.Io,
    out_path: []const u8,
    report: *const sens.Report,
    emit: *const fn (*const sens.Report, *std.Io.Writer) anyerror!void,
) !void {
    const file = try std.Io.Dir.cwd().createFile(io, out_path, .{ .truncate = true });
    defer file.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = file.writer(io, &buf);
    try emit(report, &fw.interface);
    try fw.interface.flush();
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
        \\-S, --shapes                   With names: emit {"name":…,"shape":[…]} objects instead of bare names, for architectures detected by shape.
        \\-b, --backend <BACKEND>        With calibrate: compute backend (cpu, vulkan, zig-cuda, cuda). Default cpu.
        \\-e, --text-encoder <FILENAME>  With calibrate: path to the text encoder checkpoint (required).
        \\-v, --vae <FILENAME>           With calibrate: path to the VAE checkpoint (required).
        \\-r, --resolution <INT>         With calibrate: square capture resolution in pixels, multiple of 16. Default 512.
        \\-p, --prompts <FILENAME>       With calibrate: prompt-set JSON ({"id":…,"prompts":[…]}). Default: the built-in set.
        \\    --steps <INT>              With calibrate: denoising steps per prompt. Default 4.
        \\    --rows <INT>               With calibrate: token rows retained per layer per bucket. Default 64.
        \\    --buckets <INT>            With calibrate: schedule buckets to keep separate statistics for. Default 3.
        \\    --seed <SEED>              With calibrate: sampler seed. Default 0.
        \\-C, --calib <FILENAME>         The calibration cache written by calibrate. Required by sensitivity; optional for convert, where it makes k-quant scale searches activation-aware (GGUF output only).
        \\-F, --formats <NAME>           With sensitivity: comma-separated formats to measure (e.g. "q4_k,nvfp4,int4_convrot"). Default: all.
        \\    --bucket <INT>             With sensitivity: measure only this schedule bucket. Default: all buckets together.
        \\    --top <INT>                With sensitivity: how many layers to list in the report. Default 25.
        \\    --max-layers <INT>         With sensitivity: stop after this many layers (a quick look, not a trustworthy ranking).
        \\    --no-kernel-arm            With sensitivity: skip the native-kernel arm and measure format loss only.
        \\    --no-imatrix-arm           With sensitivity: skip the activation-weighted (imatrix) arm.
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template, calibrate, sensitivity, version
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
        .BACKEND = clap.parsers.string,
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
        try stdout.print("  names          Dump tensor names as a JSON array (for test fixtures; -S to include shapes)\n", .{});
        try stdout.print("  sensitivities  Generate a sensitivities JSON template from the specified file\n", .{});
        try stdout.print("  calibrate      Run diffusion forwards on the specified model and record a calibration cache\n", .{});
        try stdout.print("  sensitivity    Measure per-layer sensitivity from a calibration cache and write a measured sensitivities JSON\n", .{});
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
        .calibration_path = res.args.calib,
        .allowed_quant_families = allowed_quant_families,
        .model_only = model_only,
        .allow_unknown_arch = allow_unknown_arch,
        .allow_upscale = allow_upscale,
        .arch_override = arch_override,
        .stochastic_rounding = res.args.@"stochastic-rounding",
    };

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    // Capture drives inference rather than reading the file as a converter
    // input, so it is handled before file-type detection.
    if (command == .calibrate) {
        try runCalibrate(io, allocator, arena_alloc, path, res.args, stdout);
        try stdout.flush();
        return;
    }
    if (command == .sensitivity) {
        try runSensitivity(io, allocator, arena_alloc, path, res.args, threads, stdout);
        try stdout.flush();
        return;
    }

    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });

    var read_buffer: [8]u8 = undefined;
    var reader = file.reader(io, &read_buffer);

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
                    try conv.writeTemplateFromFile(
                        &f,
                        arch_ptr,
                        true, // reverse dims: safetensors → GGUF template convention
                        writer,
                        allocator,
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
                .names => try dumpNames(f.tensors.items, res.args.shapes != 0, allocator, stdout),
                // All three are dispatched before file-type detection.
                .calibrate, .sensitivity, .version => unreachable,
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
                .names => try dumpNames(f.tensors.items, res.args.shapes != 0, allocator, stdout),
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
                // All three are dispatched before file-type detection.
                .calibrate, .sensitivity, .version => unreachable,
            }
        },
    }
    try stdout.flush();
    std.log.info("Total bytes used in arena allocator: {}", .{arena.queryCapacity()});
    const elapsed = start_ts.durationTo(std.Io.Clock.Timestamp.now(io, .awake));
    std.log.info("Completed in {d:.2} seconds.", .{@as(f64, @floatFromInt(elapsed.raw.nanoseconds)) / std.time.ns_per_s});
}
