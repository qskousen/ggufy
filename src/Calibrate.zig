//! `ggufy calibrate` — drive real diffusion forwards and record what every
//! linear layer actually sees.
//!
//! This is the capture half of activation-aware quantization
//! (ACTIVATION_AWARE_PLAN.md §6). It loads a checkpoint through TensorPencil's
//! pipeline, installs `ops.matmul.probe`, runs the prompt set, and writes a
//! calibration cache. Nothing here knows how to quantize; the cache is the
//! product, and `sensitivity` / `convert --calib` are its consumers.
//!
//! Two things about the shape of a capture are worth stating plainly, because
//! they look like bugs otherwise:
//!
//!   - **Statistics accumulate across prompts.** One `Collector` spans the whole
//!     run: `diag` sums, `amax` maxes, and the row reservoir samples uniformly
//!     over every token of every prompt. That is the aggregation §4 asks for, and
//!     it is why the cache is per-(model, prompt-set), not per-prompt.
//!   - **Buckets follow the denoising schedule, and only the DiT respects them.**
//!     A diffusion model's activation distribution moves along the schedule, so
//!     statistics are kept per schedule bucket. The text encoder runs before the
//!     first step and the VAE after the last, so their layers land in the first
//!     and last bucket respectively. Their per-bucket split is meaningless; their
//!     totals are not.
//!
//! Capture does **not** need determinism — statistics tolerate GPU
//! reduction-order drift — which is why this may run on any backend. The level
//! 2–4 verdict does need it, and that is a different code path.

const std = @import("std");
const tp = @import("TensorPencil");
const Activations = @import("Activations.zig");
const CalibrationCache = @import("CalibrationCache.zig");
const cb = @import("callbacks.zig");

pub const Backend = tp.pipeline.Backend;

/// The built-in prompt set. In-repo because it is part of the measurement: a
/// cache is only reproducible if the prompts are.
pub const default_prompts_json = @embedFile("prompts/calibration.json");

pub const PromptSet = struct {
    id: []const u8,
    prompts: []const []const u8,
    /// Backing storage when the set was parsed from JSON.
    parsed: ?std.json.Parsed(std.json.Value) = null,

    pub fn deinit(self: *PromptSet) void {
        if (self.parsed) |p| p.deinit();
        self.* = undefined;
    }
};

/// Parse a prompt set from `{"id": ..., "prompts": [...]}`. Used for both the
/// built-in set and `--prompts <file>`, so a user-supplied file is held to the
/// same shape (and carries its own id into the cache's provenance).
pub fn parsePromptSet(gpa: std.mem.Allocator, json: []const u8) !PromptSet {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    errdefer parsed.deinit();
    if (parsed.value != .object) return error.InvalidPromptSet;
    const root = parsed.value.object;

    const id_val = root.get("id") orelse return error.InvalidPromptSet;
    if (id_val != .string) return error.InvalidPromptSet;

    const arr_val = root.get("prompts") orelse return error.InvalidPromptSet;
    if (arr_val != .array) return error.InvalidPromptSet;
    const items = arr_val.array.items;
    if (items.len == 0) return error.EmptyPromptSet;

    const prompts = try parsed.arena.allocator().alloc([]const u8, items.len);
    for (items, prompts) |item, *p| {
        if (item != .string) return error.InvalidPromptSet;
        p.* = item.string;
    }
    return .{ .id = id_val.string, .prompts = prompts, .parsed = parsed };
}

pub const Options = struct {
    /// Checkpoint to capture from — the denoiser. This is the model whose tensor
    /// names key the cache.
    dit_path: []const u8,
    text_encoder_path: []const u8,
    vae_path: []const u8,
    /// Where the cache is written.
    output_path: []const u8,

    backend: Backend = .cpu,
    /// Square capture resolution, in pixels; must be a multiple of 16.
    resolution: usize = 512,
    steps: usize = 4,
    /// Sampler seed. Fixed by default so a capture is reproducible.
    seed: u64 = 0,
    /// Cap on device memory (bytes; 0 = the driver's live budget).
    vram_budget: u64 = 0,

    /// Token rows retained per layer per bucket.
    sample_rows: usize = 64,
    /// Schedule buckets to keep separate statistics for.
    buckets: usize = 3,
    /// Row-sampling seed (not the sampler seed).
    sample_seed: u64 = Activations.Options.default_seed,

    /// Architecture label recorded in the cache. Free-form; `sensitivity` uses it
    /// to name the JSON it emits.
    arch: []const u8 = "",
    /// Tool + version string recorded in the cache.
    producer: []const u8 = "ggufy",

    /// Human-readable log of load timings and per-step notes. Passed straight to
    /// TensorPencil, which is verbose here; null to silence it.
    log: ?*std.Io.Writer = null,
    callbacks: cb.CaptureCallbacks = .{},
};

pub const Summary = struct {
    layers: usize,
    /// Tokens accumulated across every layer and bucket.
    tokens: u64,
    /// GEMMs observed with no tag. A large count means the loader is not tagging
    /// and the capture is thinner than it looks.
    untagged: u64,
    prompts: usize,
    /// Bytes written.
    cache_bytes: u64,
    /// Wall time of the capture, nanoseconds.
    elapsed_ns: u64,
};

/// Advances the collector's schedule bucket and forwards progress. Installed as
/// TensorPencil's `on_step` hook, which fires *after* each step — so it sets the
/// bucket for the step about to run, and the first step's bucket is set before
/// `generate`.
const StepHook = struct {
    collector: *Activations.Collector,
    callbacks: cb.CaptureCallbacks,
    prompt: u32,
    prompts: u32,

    fn onStep(ctx: *anyopaque, done: usize, total: usize, preview: ?tp.pipeline.Preview) void {
        _ = preview;
        const self: *StepHook = @ptrCast(@alignCast(ctx));
        // `done` steps are finished, so the next forward sits at `done / total`
        // along the schedule. At the last step this saturates to the final
        // bucket, which is where the VAE decode's GEMMs then land.
        const progress = @as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total));
        self.collector.setBucket(self.collector.bucketForProgress(progress));
        self.callbacks.reportProgress(self.prompt, self.prompts, @intCast(done), @intCast(total));
    }
};

/// Hash a checkpoint's full contents. This is the cache's identity check: a
/// header-only hash would collide between two finetunes of the same
/// architecture, which is exactly the mix-up worth catching. Streamed, so it
/// costs a sequential read of the file (seconds warm on the 12.8 GB krea2 DiT)
/// and no memory.
pub fn hashModel(io: std.Io, path: []const u8, out: *[16]u8) !void {
    return hashModelIn(io, std.Io.Dir.cwd(), path, out);
}

pub fn hashModelIn(io: std.Io, dir: std.Io.Dir, path: []const u8, out: *[16]u8) !void {
    const file = try dir.openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    var buf: [1 << 20]u8 = undefined;
    var reader = file.reader(io, &buf);
    var hasher = std.hash.Wyhash.init(0);
    var len: u64 = 0;
    while (true) {
        const chunk = reader.interface.peekGreedy(1) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        hasher.update(chunk);
        len += chunk.len;
        reader.interface.toss(chunk.len);
    }
    // Fold the length in: a hash of contents alone is blind to truncation that
    // happens to land on a block boundary of zeros.
    hasher.update(std.mem.asBytes(&len));
    _ = std.fmt.bufPrint(out, "{x:0>16}", .{hasher.final()}) catch unreachable;
}

/// Run a capture and write the cache. Returns what was measured, so a caller can
/// report it without re-reading the file.
pub fn run(gpa: std.mem.Allocator, io: std.Io, opts: Options, set: PromptSet) !Summary {
    if (opts.resolution % 16 != 0) return error.SizeNotMultipleOf16;
    if (opts.steps < 1) return error.NoSteps;
    if (set.prompts.len == 0) return error.EmptyPromptSet;

    const start: i96 = std.Io.Clock.real.now(io).nanoseconds;

    var model_hash: [16]u8 = undefined;
    try hashModel(io, opts.dit_path, &model_hash);

    var collector = Activations.Collector.init(gpa, .{
        .sample_rows = opts.sample_rows,
        .buckets = opts.buckets,
        .seed = opts.sample_seed,
    });
    defer collector.deinit();

    // TensorPencil's cancel is an atomic it polls throughout a forward; ggufy's
    // callback convention is a predicate. Bridge them by mirroring the predicate
    // into the atomic between steps and prompts — which is as often as a capture
    // can act on it anyway.
    var cancel: std.atomic.Value(bool) = .init(false);

    var base_opts: tp.pipeline.Options = .{
        .prompt = set.prompts[0],
        .width = opts.resolution,
        .height = opts.resolution,
        .steps = opts.steps,
        .seed = opts.seed,
        .backend = opts.backend,
        .vram_budget = opts.vram_budget,
        .dit_path = opts.dit_path,
        .text_encoder_path = opts.text_encoder_path,
        .vae_path = opts.vae_path,
        .cancel = &cancel,
    };

    // The probe hangs off `ops.matmul`, and only the CPU denoiser goes through
    // it — `dit_gpu` / `dit_cuda` own their upload and GEMM. Until the
    // device-side probe lands (ACTIVATION_AWARE_PLAN.md §3.5 / item 9), a GPU
    // capture records whatever happens to fall back to CPU and silently misses
    // the rest. Say so: a thin cache that looks well-formed is precisely the
    // failure this whole module is built to avoid.
    if (opts.backend != .cpu) {
        std.log.warn(
            "backend '{s}': the activation probe currently only observes the CPU path, " ++
                "so this capture will miss the GEMMs that run on the device. " ++
                "Use --backend cpu for a complete cache until device-side probe stats land.",
            .{@tagName(opts.backend)},
        );
    }

    var session = try tp.pipeline.Session.init(io, gpa, base_opts, opts.log);
    defer session.deinit();

    // The probe is a process-wide single-threaded global, like TensorPencil's own
    // `gpu_dispatch` / `cancel.token`. Restore whatever was there so a GUI that
    // runs two captures in sequence — or a test — is not left with a dangling one.
    const prev_probe = tp.ops.matmul.probe;
    defer tp.ops.matmul.probe = prev_probe;
    tp.ops.matmul.probe = collector.probe();

    for (set.prompts, 0..) |prompt, i| {
        if (opts.callbacks.isCancelled()) {
            cancel.store(true, .release);
            return error.Canceled;
        }

        var hook: StepHook = .{
            .collector = &collector,
            .callbacks = opts.callbacks,
            .prompt = @intCast(i),
            .prompts = @intCast(set.prompts.len),
        };

        base_opts.prompt = prompt;
        base_opts.on_step = .{ .ctx = &hook, .step = StepHook.onStep };
        // The text encoder runs before the first step, so start every prompt at
        // the first bucket rather than wherever the previous prompt ended.
        collector.setBucket(0);

        var image = try session.generate(base_opts, opts.log);
        // The pixels are not the product here — the statistics are. A capture
        // that also wrote images would be a different command.
        image.deinit(gpa);
    }

    try collector.checkOk();

    var prov: CalibrationCache.Provenance = .{
        .model_path = opts.dit_path,
        .model_hash = &model_hash,
        .arch = opts.arch,
        .prompt_set = set.id,
        .backend = @tagName(opts.backend),
        .producer = opts.producer,
        .resolution = @intCast(opts.resolution),
        .steps = @intCast(opts.steps),
        .seed = opts.seed,
    };
    prov.sample_seed = opts.sample_seed;

    try CalibrationCache.writeFile(gpa, io, opts.output_path, &collector, prov);

    // Read the cache back and run the sanity gate against the checkpoint it was
    // captured from, before telling anyone the capture succeeded. A capture is
    // expensive and its output is consumed by tools that cannot tell a subtly
    // wrong cache from a good one, so the cheap end of that check belongs here
    // rather than in each consumer. It also proves the round-trip on real data
    // every single run, which no unit test can.
    {
        var cache = try CalibrationCache.Cache.open(gpa, io, opts.output_path);
        defer cache.deinit();
        var ck = try tp.safetensors.SafeTensors.open(gpa, io, opts.dit_path);
        defer ck.deinit();

        var diag: CalibrationCache.Diagnostic = .{};
        CalibrationCache.validate(&cache, .{
            .model_hash = &model_hash,
            .checkpoint = &ck,
        }, &diag) catch |err| {
            std.log.err("the cache just written to '{s}' does not validate: {s}", .{ opts.output_path, diag.msg });
            return err;
        };
    }

    var tokens: u64 = 0;
    var it = collector.iterator();
    while (it.next()) |e| {
        for (e.value_ptr.buckets) |b| tokens += b.count;
    }

    const stat = try std.Io.Dir.cwd().statFile(io, opts.output_path, .{});
    return .{
        .layers = collector.layerCount(),
        .tokens = tokens,
        .untagged = collector.untagged,
        .prompts = set.prompts.len,
        .cache_bytes = stat.size,
        .elapsed_ns = @intCast(@max(0, std.Io.Clock.real.now(io).nanoseconds - start)),
    };
}

// ---------------------------------------------------------------------------
// Tests
//
// `run` itself needs a 19 GB checkpoint and minutes of compute, so it is
// exercised by hand (and, later, by the level-1 harness's own gate) rather than
// here. What is testable in milliseconds is the parsing and the identity hash —
// both of which decide whether a cache is usable, and both of which fail
// silently if wrong.
// ---------------------------------------------------------------------------

const testing = std.testing;

test "the built-in prompt set parses and covers the damage-revealing cases" {
    var set = try parsePromptSet(testing.allocator, default_prompts_json);
    defer set.deinit();

    try testing.expectEqualStrings("ggufy-calib-v1", set.id);
    try testing.expectEqual(@as(usize, 16), set.prompts.len);
    for (set.prompts) |p| try testing.expect(p.len > 0);

    // Quantization damage shows up first in rendered text, fine repeating
    // texture and faces/hands (ACTIVATION_AWARE.md), so a set without them would
    // measure the easy cases only. Assert the intent, not the wording.
    var text = false;
    var texture = false;
    var people = false;
    for (set.prompts) |p| {
        if (std.mem.indexOfScalar(u8, p, '"') != null) text = true;
        if (std.mem.indexOf(u8, p, "repeating") != null or std.mem.indexOf(u8, p, "threads") != null) texture = true;
        if (std.mem.indexOf(u8, p, "portrait") != null or std.mem.indexOf(u8, p, "hands") != null) people = true;
    }
    try testing.expect(text);
    try testing.expect(texture);
    try testing.expect(people);
}

test "a malformed prompt set is refused rather than silently reduced" {
    const gpa = testing.allocator;
    try testing.expectError(error.InvalidPromptSet, parsePromptSet(gpa, "[]"));
    try testing.expectError(error.InvalidPromptSet, parsePromptSet(gpa, "{\"prompts\":[\"a\"]}"));
    try testing.expectError(error.InvalidPromptSet, parsePromptSet(gpa, "{\"id\":\"x\"}"));
    try testing.expectError(error.EmptyPromptSet, parsePromptSet(gpa, "{\"id\":\"x\",\"prompts\":[]}"));
    try testing.expectError(error.InvalidPromptSet, parsePromptSet(gpa, "{\"id\":\"x\",\"prompts\":[1]}"));

    var ok = try parsePromptSet(gpa, "{\"id\":\"mine\",\"prompts\":[\"a cat\",\"a dog\"]}");
    defer ok.deinit();
    try testing.expectEqualStrings("mine", ok.id);
    try testing.expectEqualSlices(u8, "a dog", ok.prompts[1]);
}

test "the model hash distinguishes files that differ anywhere, including in length" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = tmp.dir;

    // Larger than the reader's buffer so the streaming path is actually chunked.
    const n = 3 << 20;
    const a = try gpa.alloc(u8, n);
    defer gpa.free(a);
    for (a, 0..) |*v, i| v.* = @truncate(i);

    try dir.writeFile(io, .{ .sub_path = "a.bin", .data = a });
    // One flipped byte deep inside the file.
    a[n - 7] ^= 0xff;
    try dir.writeFile(io, .{ .sub_path = "b.bin", .data = a });
    a[n - 7] ^= 0xff;
    // Same prefix, truncated.
    try dir.writeFile(io, .{ .sub_path = "c.bin", .data = a[0 .. n - 1] });

    var ha: [16]u8 = undefined;
    var hb: [16]u8 = undefined;
    var hc: [16]u8 = undefined;
    var ha2: [16]u8 = undefined;
    for ([_][]const u8{ "a.bin", "b.bin", "c.bin", "a.bin" }, [_]*[16]u8{ &ha, &hb, &hc, &ha2 }) |name, out| {
        try hashModelIn(io, dir, name, out);
    }

    try testing.expectEqualSlices(u8, &ha, &ha2); // stable
    try testing.expect(!std.mem.eql(u8, &ha, &hb)); // one flipped byte
    try testing.expect(!std.mem.eql(u8, &ha, &hc)); // truncation
}
