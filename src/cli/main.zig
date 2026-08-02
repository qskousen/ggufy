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
const vrd = ggufy.verdict;
const dvg = ggufy.divergence;
const het = ggufy.heterogeneity;

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
    verdict,
    divergence,
    heterogeneity,
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

/// True when `path` is a checkpoint that carries its own text encoder and VAE — an
/// LDM single-file SD checkpoint. Those commands normally require `-e`/`-v`, and for
/// such a file the flags are not only unnecessary but misleading: TensorPencil loads
/// the conditioner and decoder out of the same container.
///
/// Cheap: only the header is parsed. A GGUF (ggufy's own model-only output) never
/// qualifies, which is correct — a quantized UNet does need the original checkpoint
/// for CLIP and the VAE.
fn checkpointIsSelfContained(io: std.Io, allocator: std.mem.Allocator, arena_alloc: std.mem.Allocator, path: []const u8) bool {
    var f = ggufy.fileLoader.TensorFile.loadFile(io, allocator, arena_alloc, path) catch return false;
    defer f.deinit();
    var has_clip = false;
    var has_vae = false;
    for (f.tensors.items) |t| {
        if (std.mem.startsWith(u8, t.name, "cond_stage_model.")) has_clip = true;
        if (std.mem.startsWith(u8, t.name, "first_stage_model.")) has_vae = true;
        if (has_clip and has_vae) return true;
    }
    return false;
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
    // An LDM single-file SD checkpoint holds all three models, so -e/-v are optional
    // there and required everywhere else.
    const self_contained = checkpointIsSelfContained(io, allocator, arena_alloc, path);
    const text_encoder = args.@"text-encoder" orelse blk: {
        if (self_contained) break :blk path;
        std.log.err("calibrate needs the text encoder checkpoint: -e/--text-encoder <path>", .{});
        return error.MissingArgument;
    };
    const vae = args.vae orelse blk: {
        if (self_contained) break :blk path;
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

/// `ggufy verdict <reference> --candidates a,b [...]` — level 3/4: compare rendered
/// images across quantization arms and emit a table plus a contact sheet.
fn runVerdict(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    path: []const u8,
    args: anytype,
    stdout: *std.Io.Writer,
) !void {
    const list = args.candidates orelse {
        std.log.err("verdict needs at least one candidate: --candidates <dir>[,<dir>...]", .{});
        return error.MissingArgument;
    };

    var cands: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |tok| {
        const t = std.mem.trim(u8, tok, " \t");
        if (t.len > 0) try cands.append(arena_alloc, t);
    }

    var report = vrd.run(allocator, .{
        .io = io,
        .reference = path,
        .candidates = cands.items,
        .lpips_weights = args.lpips,
    }) catch |err| {
        std.log.err("verdict failed: {t}", .{err});
        return err;
    };
    defer report.deinit();

    try vrd.writeMarkdown(stdout, &report);

    if (args.html) |html_path| {
        const dir = std.fs.path.dirname(html_path) orelse ".";
        const f = try std.Io.Dir.cwd().createFile(io, html_path, .{ .truncate = true });
        defer f.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fw = f.writer(io, &buf);
        try vrd.writeHtml(&fw.interface, &report, dir);
        try fw.interface.flush();
        try stdout.print("\nWrote contact sheet to {s}\n", .{html_path});
    }
}

/// `ggufy divergence <reference> --candidates a,b [...]` — level 2: one-pass
/// velocity divergence on the reference's own trajectory (teacher-forced), the
/// only measurement here with no trajectory drift in it.
fn runDivergence(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    path: []const u8,
    args: anytype,
    stdout: *std.Io.Writer,
) !void {
    const per_tensor = args.@"per-tensor" != 0;
    if (!per_tensor and args.candidates == null) {
        std.log.err("divergence needs at least one candidate checkpoint: --candidates <file>[,<file>...]", .{});
        return error.MissingArgument;
    }
    if (per_tensor and args.candidates != null) {
        // Refused rather than ignored: a report listing candidates nobody measured
        // is worse than no report.
        std.log.err("--per-tensor measures tensors of the reference itself; it has no use for --candidates", .{});
        return error.InvalidArgument;
    }
    const list = args.candidates orelse "";
    // Same as calibrate: an LDM single-file SD checkpoint supplies its own conditioner
    // and decoder, so the flags are optional for it and required otherwise.
    const self_contained = checkpointIsSelfContained(io, allocator, arena_alloc, path);
    const text_encoder = args.@"text-encoder" orelse blk: {
        if (self_contained) break :blk path;
        std.log.err("divergence needs the text encoder checkpoint: -e/--text-encoder <path>", .{});
        return error.MissingArgument;
    };
    const vae = args.vae orelse blk: {
        if (self_contained) break :blk path;
        std.log.err("divergence needs the VAE checkpoint: -v/--vae <path>", .{});
        return error.MissingArgument;
    };

    const backend_name = args.backend orelse "cpu";
    const backend = dvg.Backend.fromStr(backend_name) orelse {
        std.log.err("Unknown backend '{s}'. Use one of: cpu, vulkan, zig-cuda, cuda", .{backend_name});
        return error.InvalidArgument;
    };
    // Levels 1-2 anchor on the CPU f32 path (ACTIVATION_AWARE.md hygiene rule 2):
    // a GPU reduction-order difference between the two arms would land in the same
    // number as the quantization difference we are trying to measure.
    if (backend != .cpu) {
        std.log.warn(
            "backend '{s}': level 2 compares two models' velocities, and GPU reduction order " ++
                "differs run to run — the drift becomes this measurement's noise floor. " ++
                "Use --backend cpu for the reference number.",
            .{@tagName(backend)},
        );
    }

    var cands: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |tok| {
        const t = std.mem.trim(u8, tok, " \t");
        if (t.len > 0) try cands.append(arena_alloc, t);
    }

    // The prompt set is part of the measurement, exactly as for `calibrate`, and
    // the same file works for both.
    const prompt_json = if (args.prompts) |p|
        try std.Io.Dir.cwd().readFileAlloc(io, p, arena_alloc, .unlimited)
    else
        calib.default_prompts_json;
    var set = calib.parsePromptSet(allocator, prompt_json) catch |err| {
        std.log.err("Could not read the prompt set: {t}. Expected {{\"id\": …, \"prompts\": [ … ]}}", .{err});
        return err;
    };
    defer set.deinit();

    const res = args.resolution orelse 512;
    const steps = args.steps orelse 8;
    const opts = dvg.Options{
        .io = io,
        .reference = path,
        .candidates = cands.items,
        .text_encoder_path = text_encoder,
        .vae_path = vae,
        .prompts = set.prompts,
        .width = res,
        .height = res,
        .steps = steps,
        .seed = args.seed orelse 0,
        .backend = backend,
        .log = stdout,
    };

    if (per_tensor) {
        try runDivergencePerTensor(io, allocator, arena_alloc, opts, args, stdout);
        return;
    }

    try stdout.print(
        "level 2: {d} prompts x {d} steps at {d}^2 on {s} = {d} points per arm, {d} arms\n" ++
            "reference {s}\n",
        .{ set.prompts.len, steps, res, @tagName(backend), set.prompts.len * steps, cands.items.len, path },
    );
    try stdout.flush();

    var report = dvg.run(allocator, opts) catch |err| {
        std.log.err("divergence failed: {t}", .{err});
        return err;
    };
    defer report.deinit();

    try stdout.writeByte('\n');
    try dvg.writeMarkdown(stdout, &report);
    try stdout.flush();

    const stem = args.@"output-name" orelse "divergence";
    const csv_path = try joinOut(arena_alloc, args.@"output-dir", stem, ".csv");
    {
        const file = try std.Io.Dir.cwd().createFile(io, csv_path, .{ .truncate = true });
        defer file.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &buf);
        try dvg.writeCsv(&fw.interface, &report);
        try fw.interface.flush();
    }
    try stdout.print("\nwrote {s}\n", .{csv_path});
}

/// `ggufy divergence <reference> --per-tensor -F <format> [...]` — level 2, one
/// tensor at a time: the arm that checks whether level 1's per-layer ranking is the
/// ranking the whole model agrees with.
fn runDivergencePerTensor(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    opts: dvg.Options,
    args: anytype,
    stdout: *std.Io.Writer,
) !void {
    const spec = args.formats orelse {
        std.log.err("--per-tensor needs exactly one format to measure: -F <format> (e.g. -F q4_k)", .{});
        return error.MissingArgument;
    };
    const fmts = sens.parseFormats(arena_alloc, spec) catch |err| {
        std.log.err("Could not parse --formats '{s}': {t}", .{ spec, err });
        return err;
    };
    if (fmts.len != 1) {
        // Every extra format multiplies the run by the number of tensors; make the
        // cost an explicit choice rather than a comma.
        std.log.err("--per-tensor measures one format per run ({d} given). Run it once per format.", .{fmts.len});
        return error.InvalidArgument;
    }

    var names: std.ArrayList([]const u8) = .empty;
    if (args.tensors) |t| {
        var it = std.mem.splitScalar(u8, t, ',');
        while (it.next()) |tok| {
            const s = std.mem.trim(u8, tok, " \t");
            if (s.len > 0) try names.append(arena_alloc, s);
        }
    }
    if (args.@"tensors-from") |path| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, arena_alloc, .unlimited);
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const s = std.mem.trim(u8, line, " \t\r");
            if (s.len == 0 or s[0] == '#') continue;
            try names.append(arena_alloc, s);
        }
    }

    // A set's arm is named by its *label*, not by a tensor, so a sensitivities file
    // written from a set run would be keyed on names no checkpoint has — and a miss
    // is only a warning in the converter, so it would look like it had worked.
    if (args.sets != null and args.@"emit-sensitivities" != null) {
        std.log.err("--emit-sensitivities needs per-tensor arms; it cannot be combined with --sets", .{});
        return error.InvalidArgument;
    }

    // --sets label:file[,label:file...]
    var sets: std.ArrayList(dvg.PerTensor.Set) = .empty;
    if (args.sets) |sets_spec| {
        var it = std.mem.splitScalar(u8, sets_spec, ',');
        while (it.next()) |tok| {
            const entry = std.mem.trim(u8, tok, " \t");
            if (entry.len == 0) continue;
            const colon = std.mem.indexOfScalar(u8, entry, ':') orelse {
                std.log.err("--sets wants 'label:file' entries (got '{s}')", .{entry});
                return error.InvalidArgument;
            };
            const label = entry[0..colon];
            const path = entry[colon + 1 ..];
            const text = std.Io.Dir.cwd().readFileAlloc(io, path, arena_alloc, .unlimited) catch |err| {
                std.log.err("--sets: could not read '{s}': {t}", .{ path, err });
                return err;
            };
            var members: std.ArrayList([]const u8) = .empty;
            var lines = std.mem.splitScalar(u8, text, '\n');
            while (lines.next()) |line| {
                const nm = std.mem.trim(u8, line, " \t\r");
                if (nm.len == 0 or nm[0] == '#') continue;
                try members.append(arena_alloc, nm);
            }
            if (members.items.len == 0) {
                std.log.err("--sets: '{s}' lists no tensors", .{path});
                return error.InvalidArgument;
            }
            try sets.append(arena_alloc, .{ .label = label, .tensors = members.items });
        }
    }

    const patch_dtype: dvg.PerTensor.PatchDtype = blk: {
        const name = args.@"patch-dtype" orelse break :blk .f32;
        if (std.mem.eql(u8, name, "f32")) break :blk .f32;
        if (std.mem.eql(u8, name, "bf16")) break :blk .bf16;
        std.log.err("--patch-dtype must be 'f32' or 'bf16' (got '{s}')", .{name});
        return error.InvalidArgument;
    };
    const pt = dvg.PerTensor{
        .format = fmts[0],
        .tensors = names.items,
        .max_tensors = args.@"max-tensors",
        .controls = args.@"no-controls" == 0,
        .threads = args.threads orelse 0,
        .patch_dtype = patch_dtype,
        .sets = sets.items,
    };

    // Optional: the level-1 CSV this ranking is compared against. Loaded BEFORE the
    // measurement — a typo in the path should not cost an hour of forwards.
    var l1: ?dvg.Level1 = null;
    defer if (l1) |*m| m.deinit();
    if (args.level1) |csv_path| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, csv_path, arena_alloc, .unlimited);
        const arm = args.@"level1-arm" orelse "format";
        l1 = dvg.Level1.parseCsv(allocator, text, sens.formatName(fmts[0]), arm) catch |err| {
            std.log.err("Could not read level-1 rows for format '{s}', arm '{s}' from {s}: {t}", .{ sens.formatName(fmts[0]), arm, csv_path, err });
            return err;
        };
    }

    const n_tensors: usize = if (names.items.len > 0) names.items.len else 0;
    try stdout.print(
        "level 2 per-tensor: format {s}, {d} prompts x {d} steps at {d}^2 on {s} = {d} points per arm\n" ++
            "reference {s}\n{s}\n",
        .{
            sens.formatName(fmts[0]),   opts.prompts.len,
            opts.steps,                 opts.width,
            @tagName(opts.backend),     opts.prompts.len * opts.steps,
            opts.reference,
            if (n_tensors > 0) "tensors: from --tensors" else "tensors: every matrix weight in the checkpoint (this is a long run)",
        },
    );
    try stdout.flush();

    // The CSV is opened BEFORE the run and streamed into, one flush per arm: a
    // 263-tensor sweep is hours, and the report is only assembled at the end, so
    // without this an interruption in hour three leaves nothing measured behind.
    const stem = args.@"output-name" orelse "divergence-per-tensor";
    const csv_out = try joinOut(arena_alloc, args.@"output-dir", stem, ".csv");
    const csv_file = try std.Io.Dir.cwd().createFile(io, csv_out, .{ .truncate = true });
    defer csv_file.close(io);
    var csv_buf: [1 << 16]u8 = undefined;
    var csv_w = csv_file.writer(io, &csv_buf);
    try csv_w.interface.writeAll(dvg.tensor_csv_header);
    try csv_w.interface.flush();

    var streamed = pt;
    streamed.stream_csv = &csv_w.interface;

    var report = dvg.runPerTensor(allocator, opts, streamed) catch |err| {
        std.log.err("per-tensor divergence failed: {t} (partial results are in {s})", .{ err, csv_out });
        return err;
    };
    defer report.deinit();

    try stdout.writeByte('\n');
    try dvg.writeTensorMarkdown(stdout, &report, if (l1) |*m| m else null);
    try stdout.flush();

    if (args.@"emit-sensitivities") |sens_path| {
        const file = try std.Io.Dir.cwd().createFile(io, sens_path, .{ .truncate = true });
        defer file.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &buf);
        // The converter looks up STRIPPED names (`filterAndStripTensors` removes the
        // container prefix), so a file keyed on the names the probe saw would match
        // nothing — and silently, since a miss is only a warning.
        dvg.writeSensitivitiesJson(arena_alloc, &fw.interface, &report, "model.diffusion_model.", report.arch) catch |err| {
            std.log.err("could not write sensitivities to {s}: {t}", .{ sens_path, err });
            return err;
        };
        try fw.interface.flush();
        try stdout.print("wrote {s} (scored by measured per-layer damage)\n", .{sens_path});
    }

    const md_out = try joinOut(arena_alloc, args.@"output-dir", stem, ".md");
    {
        const file = try std.Io.Dir.cwd().createFile(io, md_out, .{ .truncate = true });
        defer file.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &buf);
        try dvg.writeTensorMarkdown(&fw.interface, &report, if (l1) |*m| m else null);
        try fw.interface.flush();
    }
    try stdout.print("\nwrote {s}\nwrote {s}\n", .{ csv_out, md_out });
}

/// `ggufy sensitivity <model> -C <cache> [...]` — level 1: measure per-layer
/// output error on the captured activations and write a measured sensitivities
/// JSON the converter can route on.
/// `ggufy heterogeneity <checkpoint> [-F q4_k]` — the free half of the routing
/// question: how much per-tensor quantization error VARIES inside one checkpoint,
/// in weight space, with no inference and no calibration cache.
///
/// It exists because the routing verdict turned on krea2's routable damage spanning
/// only 2.3x around its median, and whether that is a fact about krea2 or about
/// diffusion models cannot be answered until TensorPencil can run a second
/// architecture. See `Heterogeneity.zig` for what this can and cannot conclude — it
/// promotes an architecture for the expensive measurement, it never rules one out.
fn runHeterogeneity(
    io: std.Io,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    path: []const u8,
    args: anytype,
    threads: usize,
    stdout: *std.Io.Writer,
) !void {
    const fmt: het.Format = blk: {
        const spec = args.formats orelse break :blk .q4_k;
        const fmts = sens.parseFormats(arena_alloc, spec) catch |err| {
            std.log.err("Bad --formats value '{s}': {t}", .{ spec, err });
            return err;
        };
        if (fmts.len != 1) {
            std.log.err("heterogeneity screens one format per run ({d} given)", .{fmts.len});
            return error.InvalidArgument;
        }
        break :blk fmts[0];
    };

    var report = het.run(allocator, io, .{
        .model_path = path,
        .format = fmt,
        .threads = threads,
    }) catch |err| {
        std.log.err("heterogeneity screen failed: {t}", .{err});
        return err;
    };
    defer report.deinit();

    try het.writeMarkdown(allocator, &report, stdout, args.top orelse 10);
    try stdout.flush();

    // The CSV is the per-tensor table, for cross-architecture analysis. There is
    // deliberately no JSON: a weight-space ranking must not be shippable as a
    // sensitivities file (see the module header).
    if (args.@"output-name") |stem| {
        const csv_path = try joinOut(arena_alloc, args.@"output-dir", stem, ".csv");
        const file = try std.Io.Dir.cwd().createFile(io, csv_path, .{ .truncate = true });
        defer file.close(io);
        var buf: [1 << 16]u8 = undefined;
        var fw = file.writer(io, &buf);
        try het.writeCsv(&report, &fw.interface);
        try fw.interface.flush();
        try stdout.print("\nwrote {s}\n", .{csv_path});
    }
}

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

    // §8B is off unless asked for: unlike the weighted arm it is a full extra
    // quantization plus GEMM per (layer, format, α), and nothing ships it yet.
    const eq_alphas: []const f32 = if (args.@"eq-alphas") |spec|
        if (std.mem.eql(u8, spec, "default"))
            &ggufy.equalize.default_alphas
        else
            parseAlphas(arena_alloc, spec) catch |err| {
                std.log.err("Bad --eq-alphas value '{s}': {t} (expected e.g. \"0.25,0.5\" or \"default\")", .{ spec, err });
                return err;
            }
    else
        &.{};

    // §8C, same reasoning as §8B and more so: it is the most expensive arm here and
    // nothing ships it, so it only runs when asked for.
    const gptq_damp: f32 = if (args.@"gptq-damp") |spec|
        std.fmt.parseFloat(f32, spec) catch |err| {
            std.log.err("Bad --gptq-damp value '{s}': {t}", .{ spec, err });
            return err;
        }
    else
        ggufy.gptq.default_damp;
    if (!(gptq_damp > 0)) {
        // A zero ridge is not a more aggressive setting, it is a singular one: the
        // Gram has rank at most the number of sampled rows.
        std.log.err("--gptq-damp must be greater than zero (the Gram is rank-deficient without it)", .{});
        return error.InvalidArgument;
    }

    const opts = sens.Options{
        .model_path = path,
        .calib_path = calib_path,
        .formats = formats,
        .bucket = args.bucket,
        .kernel_arm = args.@"no-kernel-arm" == 0,
        .weighted_arm = args.@"no-weighted-arm" == 0,
        .eq_alphas = eq_alphas,
        .gptq = args.gptq != 0,
        .gptq_damp = gptq_damp,
        .gptq_holdout = args.@"gptq-holdout" orelse 3,
        .gptq_eval_calib = args.@"calib-eval",
        .gptq_train_rows = args.@"gptq-train-rows",
        .convrot_group = args.@"convrot-group" orelse ggufy.sensitivity.default_convrot_group,
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

/// Parse `--eq-alphas` ("0.25,0.5,0.75"). Rejects anything outside [0, 1]: α is a
/// fraction of the importance to move into the weights, and a value outside that
/// range is a typo, not an experiment.
fn parseAlphas(arena_alloc: std.mem.Allocator, spec: []const u8) ![]const f32 {
    var out: std.ArrayList(f32) = .empty;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const tok = std.mem.trim(u8, raw, " \t");
        if (tok.len == 0) continue;
        const v = try std.fmt.parseFloat(f32, tok);
        if (!(v >= 0 and v <= 1)) return error.OutOfRange;
        try out.append(arena_alloc, v);
    }
    if (out.items.len == 0) return error.NoAlphas;
    return out.items;
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
        \\    --no-weighted-arm          With sensitivity: skip the activation-weighted arm (imatrix + clipping search).
        \\    --eq-alphas <NAME>         With sensitivity: also measure activation-equalization arms (§8B) at these alphas, e.g. "0.25,0.5" or "default". Off by default; costs one extra quantize+GEMM per layer, format and alpha.
        \\    --gptq                     With sensitivity: measure the GPTQ error-compensation arm (§8C) for the int8/int4 formats. With convert: apply it, choosing each weight's level by compensation rather than rounding (needs -C, safetensors output only, adds minutes).
        \\    --gptq-damp <NAME>         With --gptq: ridge as a fraction of the mean Gram diagonal. Default 0.01.
        \\    --gptq-holdout <INT>       With --gptq: hold 1 row in N back from the Hessian and measure on those. Default 3; 2 splits the sample evenly.
        \\    --calib-eval <FILENAME>    With --gptq: score against this second cache (same checkpoint, disjoint prompts) instead of a held-out row split. The number that decides whether GPTQ generalizes.
        \\    --gptq-train-rows <INT>    With --gptq: fit the Hessian on at most this many token rows (evenly subsampled). Sweep it to see how the win scales with calibration data.
        \\    --convrot-group <INT>      With sensitivity: ConvRot rotation group for the rotating arms. Default 64 (the harness's); pass 256 to measure what convert actually writes.
        \\    --candidates <NAME>        With verdict: comma-separated candidate image dirs (or files) to compare against the reference. With divergence: candidate DiT checkpoints.
        \\    --per-tensor               With divergence: quantize ONE tensor at a time (format from -F) and measure the whole model's velocity error — the arm that checks whether level 1 ranks layers the way the model does. Refuses --candidates.
        \\    --tensors <NAME>           With --per-tensor: comma-separated checkpoint tensor names to measure. Default: every matrix weight (263 on krea2, hours). A list stratified over level 1's ranking is what answers the correlation question cheaply.
        \\    --tensors-from <FILENAME>  With --per-tensor: read the tensor names from a file, one per line (blank lines and #-comments skipped). For lists too long or too scripted for a flag.
        \\    --max-tensors <INT>        With --per-tensor: keep at most this many of the selected tensors, sampled at an even stride.
        \\    --no-controls              With --per-tensor: skip the two control arms (overlay-inertness and the f32 dequant floor). They are what make the table interpretable.
        \\    --level1 <FILENAME>        With --per-tensor: a level-1 CSV (from sensitivity) to correlate the per-tensor ranking against. Uses -F's format name.
        \\    --level1-arm <NAME>        With --level1: which level-1 arm's rows to read. Default "format"; "weighted" for the imatrix arm.
        \\    --patch-dtype <NAME>       With --per-tensor: dtype the substituted tensor is written back as, "f32" (default, exact) or "bf16". bf16 is what makes this arm runnable on a GPU — the GPU DiTs need one weight class per model — at the cost of ~0.2% extra rounding.
        \\    --emit-sensitivities <FILENAME>  With --per-tensor: write a sensitivities JSON from the MEASURED per-layer damage, scored by src/LadderScore.zig (one bit per doubling of damage above the routable median — never a percentile rank), for convert to route on. This is the cheap form of the brute-force per-layer method the hand-built arch files came from.
        \\    --sets <NAME>              With --per-tensor: measure SETS of tensors instead of one at a time, as "label:file[,label:file...]" where each file lists tensor names. One arm per set, every member quantized together — the direct test of whether per-layer damages compose.
        \\    --html <FILENAME>          With verdict: also write an HTML contact sheet here.
        \\    --lpips <FILENAME>         With verdict: LPIPS weights (AlexNet, from TensorPencil's tools/gen_lpips_fixtures.py). Adds the LPIPS column - the metric that tracks human judgement.
        \\<COMMAND>    Specify a command: header, tree, metadata, convert, template, calibrate, sensitivity, verdict, divergence, heterogeneity, version
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
        try stdout.print("  verdict        Compare rendered images across quantization arms (PSNR, SSIM, LPIPS, detail, contact sheet)\n", .{});
        try stdout.print("  divergence     Level 2: one-pass velocity divergence between checkpoints, drift-free\n", .{});
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

    // §8C. Two ways to ask for compensation and not get it, both worth refusing
    // loudly rather than producing a file that looks like what was asked for:
    // without a cache there is nothing to compensate against, and GGUF output does
    // not carry the cluster formats §8C reaches at all.
    //
    // Scoped to `convert`: `--gptq` also selects the level-1 arm on `sensitivity`,
    // which has no output file and inherits the default gguf filetype, so an
    // unscoped guard rejects the measurement command that the flag mostly exists for.
    const want_gptq = res.args.gptq != 0;
    if (want_gptq and command == .convert and res.args.calib == null) {
        std.log.err("--gptq needs a calibration cache: -C/--calib <path>", .{});
        return;
    }
    if (want_gptq and command == .convert and filetype == .gguf) {
        std.log.err(
            "--gptq applies to the int4/int8 cluster formats, which are safetensors-only; " ++
                "GGUF output would silently ignore it. Use -f safetensors, or -C alone for GGUF " ++
                "(activation-aware k-quants, measured at 9-19%).",
            .{},
        );
        return;
    }

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
        .gptq = want_gptq and command == .convert,
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
    // Weight-space only: it needs the checkpoint and nothing else, so it is dispatched
    // beside the inference commands rather than through the converter's file handling.
    if (command == .heterogeneity) {
        try runHeterogeneity(io, allocator, arena_alloc, path, res.args, threads, stdout);
        try stdout.flush();
        return;
    }
    // Verdict compares rendered images, so its positional is an image path (or a
    // directory of them) rather than a model — handled before file-type sniffing.
    if (command == .verdict) {
        try runVerdict(io, allocator, arena_alloc, path, res.args, stdout);
        try stdout.flush();
        return;
    }

    if (command == .divergence) {
        try runDivergence(io, allocator, arena_alloc, path, res.args, stdout);
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
                .calibrate, .sensitivity, .verdict, .divergence, .heterogeneity, .version => unreachable,
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
                .calibrate, .sensitivity, .verdict, .divergence, .heterogeneity, .version => unreachable,
            }
        },
    }
    try stdout.flush();
    std.log.info("Total bytes used in arena allocator: {}", .{arena.queryCapacity()});
    const elapsed = start_ts.durationTo(std.Io.Clock.Timestamp.now(io, .awake));
    std.log.info("Completed in {d:.2} seconds.", .{@as(f64, @floatFromInt(elapsed.raw.nanoseconds)) / std.time.ns_per_s});
}
