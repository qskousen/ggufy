//! Verdict.zig — `ggufy verdict`: compare rendered images across quantization arms.
//!
//! Level 3's image half and level 4's contact sheet (ACTIVATION_AWARE_PLAN.md §7).
//! Levels 0–1 measure error in weights and in one layer's output; this is the only
//! place the question is asked the way a user asks it — *does the picture still
//! look right*. The programme exists because those can disagree: the
//! learned-rounding experiment cut projected weight error by 98% and rendered pure
//! noise.
//!
//! Input is one directory per arm, plus a reference directory, with **matching
//! filenames** across them — render each arm with the same prompts and seeds and
//! the names line up for free. A single file works too, for the one-image case.
//!
//! Output is a markdown table and a self-contained HTML contact sheet. The sheet
//! matters as much as the numbers: §7 level 4 is "the one thing that can't be
//! automated; make it cheap to do", and a person flipping between two renders
//! spots a mangled letterform or a collapsed texture that no scalar reports.
//!
//! ### On the metrics
//!
//! **PSNR here is confounded, and the report says so.** Two arms of a diffusion
//! model with the same seed do not merely differ in fidelity — quantization
//! perturbs the denoising trajectory, so composition drifts, and a pixel metric
//! mixes "drew a different picture" with "drew the same picture worse". Those call
//! for different responses, so `detail` (per-image gradient energy, from
//! `tp.image.detailEnergy`) is reported alongside: it says whether fine texture
//! survived *independently* of whether the image matched. A candidate that keeps
//! the reference's detail energy while scoring poorly on PSNR drifted; one that
//! loses detail energy degraded.
//!
//! **LPIPS is the metric that actually tracks human judgement**, and it is here
//! now (`tp.models.lpips`, validated against the `lpips` pip package to 1e-5).
//! It is opt-in only because it needs a weights file: pass `--lpips <path>` and
//! every pair gets an LPIPS column. Prefer it over PSNR for a verdict — PSNR's
//! standard error at n=12 was 0.26–0.51 dB against effects of 0.08–0.43 dB,
//! which is why the §8C conclusion needed a metric with more power.
//!
//! SSIM (`tp.image.ssim`, scikit-image's 7x7 uniform convention, fixture-pinned)
//! is reported unconditionally now that it can be pinned against an external
//! reference. It is a structural metric, not a perceptual one; it sits between
//! PSNR and LPIPS and is included because it is nearly free once the pixels are
//! decoded.

const std = @import("std");
const tp = @import("TensorPencil");

/// One image's numbers against the reference.
pub const ImageResult = struct {
    name: []const u8,
    /// Null when this arm has no file matching the reference's.
    psnr: ?f64,
    mse: ?f64,
    /// Mean structural similarity against the reference, 1 = identical.
    ssim: ?f64,
    /// LPIPS perceptual distance against the reference, 0 = identical. Null when
    /// no weights file was given.
    lpips: ?f64,
    /// Gradient energy of the candidate itself, comparable to the reference's.
    detail: ?f64,
    /// The file actually compared, so a report can point at it.
    path: []const u8,
    /// The `parameters` tEXt chunk, when the renderer wrote one.
    params: ?[]const u8,
    /// Set when the file exists but could not be compared (size mismatch, etc.).
    note: ?[]const u8 = null,
};

pub const ArmResult = struct {
    name: []const u8,
    dir: []const u8,
    images: []ImageResult,

    /// Mean PSNR over the images that produced one. Null when none did.
    pub fn meanPsnr(self: ArmResult) ?f64 {
        var sum: f64 = 0;
        var n: usize = 0;
        for (self.images) |im| {
            const p = im.psnr orelse continue;
            // An identical render gives infinite PSNR; averaging it would swallow
            // the whole column. Count it, report it separately.
            if (std.math.isInf(p)) continue;
            sum += p;
            n += 1;
        }
        return if (n == 0) null else sum / @as(f64, @floatFromInt(n));
    }

    pub fn identical(self: ArmResult) usize {
        var n: usize = 0;
        for (self.images) |im| {
            const p = im.psnr orelse continue;
            if (std.math.isInf(p)) n += 1;
        }
        return n;
    }

    pub fn meanDetail(self: ArmResult) ?f64 {
        return self.meanOf("detail");
    }

    pub fn meanSsim(self: ArmResult) ?f64 {
        return self.meanOf("ssim");
    }

    pub fn meanLpips(self: ArmResult) ?f64 {
        return self.meanOf("lpips");
    }

    /// Mean of one optional per-image field over the images that produced it.
    /// Unlike PSNR, none of these has an infinite value to exclude.
    fn meanOf(self: ArmResult, comptime field: []const u8) ?f64 {
        var sum: f64 = 0;
        var n: usize = 0;
        for (self.images) |im| {
            sum += @field(im, field) orelse continue;
            n += 1;
        }
        return if (n == 0) null else sum / @as(f64, @floatFromInt(n));
    }

    pub fn missing(self: ArmResult) usize {
        var n: usize = 0;
        for (self.images) |im| {
            if (im.psnr == null) n += 1;
        }
        return n;
    }
};

pub const Report = struct {
    arena: std.heap.ArenaAllocator,
    /// The reference path exactly as the caller gave it, for the report header.
    reference: []const u8,
    ref_dir: []const u8,
    /// Reference image names, in sorted order; every arm's `images` is parallel.
    names: []const []const u8,
    /// The reference file each name resolves to.
    ref_paths: []const []const u8,
    ref_detail: []const f64,
    ref_params: []const ?[]const u8,
    arms: []ArmResult,

    pub fn deinit(self: *Report) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Whether any arm produced an LPIPS figure — i.e. whether the run was given
    /// weights. Drives whether the reports carry the column at all.
    pub fn hasLpips(self: *const Report) bool {
        for (self.arms) |a| {
            for (a.images) |im| if (im.lpips != null) return true;
        }
        return false;
    }
};

pub const Options = struct {
    io: std.Io,
    /// Directory the paths below are resolved against. Defaults to the process
    /// cwd; tests point it at a temporary directory instead.
    dir: std.Io.Dir = std.Io.Dir.cwd(),
    /// Reference render(s) — a directory, or a single file.
    reference: []const u8,
    /// Candidate arms. Each is a directory (or file) whose entries are matched to
    /// the reference by filename.
    candidates: []const []const u8,
    /// LPIPS weights (`tools/gen_lpips_fixtures.py` in TensorPencil). When null,
    /// the LPIPS column is omitted rather than silently replaced by something
    /// else — a report that says "LPIPS" has to mean LPIPS.
    lpips_weights: ?[]const u8 = null,
    /// Which SSIM convention to report. scikit-image's default, so the number is
    /// comparable to what most tools print.
    ssim_window: tp.image.SsimWindow = .uniform7,
};

fn strLessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.order(u8, a, b) == .lt;
}

fn isPng(name: []const u8) bool {
    return std.ascii.endsWithIgnoreCase(name, ".png");
}

/// Read a PNG and return its pixels plus whatever `parameters` chunk it carries.
const Loaded = struct {
    dec: tp.image.DecodedPng,
    params: ?[]const u8,
};

fn loadPng(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !Loaded {
    const bytes = try dir.readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(bytes);

    const dec = try tp.image.decodePngRgb(gpa, bytes);
    errdefer gpa.free(dec.pixels);

    // Metadata is best-effort: a render from a tool that writes none is still a
    // perfectly good comparison subject.
    var params: ?[]const u8 = null;
    if (tp.image.readTextChunks(gpa, bytes)) |chunks| {
        defer gpa.free(chunks);
        if (tp.image.findTextChunk(chunks, "parameters")) |p| params = try arena.dupe(u8, p);
    } else |_| {}

    return .{ .dec = dec, .params = params };
}

/// The image files a path contributes, sorted. A file yields itself; a directory
/// yields its `.png` entries.
fn listPngs(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) ![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;

    var sub = dir.openDir(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.NotDir => {
            // A plain file: the caller named one image directly.
            try out.append(arena, try arena.dupe(u8, std.fs.path.basename(path)));
            return out.toOwnedSlice(arena);
        },
        else => return err,
    };
    defer sub.close(io);

    var it = sub.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!isPng(entry.name)) continue;
        try out.append(arena, try arena.dupe(u8, entry.name));
    }
    _ = gpa;
    const names = try out.toOwnedSlice(arena);
    std.mem.sort([]const u8, names, {}, strLessThan);
    return names;
}

/// The directory an input path denotes: itself if it is a directory, else its parent.
fn dirOf(arena: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) ![]const u8 {
    if (isDir(io, dir, path)) return arena.dupe(u8, path);
    return arena.dupe(u8, std.fs.path.dirname(path) orelse ".");
}

/// Is this path a directory? A single named file and a directory of files are
/// resolved differently, and getting that backwards silently compares the
/// reference against itself.
fn isDir(io: std.Io, dir: std.Io.Dir, path: []const u8) bool {
    if (dir.openDir(io, path, .{})) |d| {
        var dd = d;
        dd.close(io);
        return true;
    } else |_| return false;
}

/// A short arm label from its path — the directory name, or the file stem.
fn armLabel(arena: std.mem.Allocator, path: []const u8) ![]const u8 {
    const base = std.fs.path.basename(path);
    if (isPng(base)) return arena.dupe(u8, base[0 .. base.len - 4]);
    return arena.dupe(u8, base);
}

pub fn run(gpa: std.mem.Allocator, opts: Options) !Report {
    if (opts.candidates.len == 0) return error.NoCandidates;

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    errdefer arena_state.deinit();
    const arena = arena_state.allocator();

    const ref_dir = try dirOf(arena, opts.io, opts.dir, opts.reference);
    const names = try listPngs(gpa, arena, opts.io, opts.dir, opts.reference);
    if (names.len == 0) return error.NoReferenceImages;

    // Loaded once and reused across every image and arm: ~10 MB of weights, and
    // the tower is stateless.
    var lpips_net: ?tp.models.lpips.Lpips = if (opts.lpips_weights) |p|
        try tp.models.lpips.Lpips.loadPath(gpa, opts.io, p)
    else
        null;
    defer if (lpips_net) |*n| n.deinit();

    const ref_is_dir = isDir(opts.io, opts.dir, opts.reference);
    const ref_detail = try arena.alloc(f64, names.len);
    const ref_params = try arena.alloc(?[]const u8, names.len);
    const ref_paths = try arena.alloc([]const u8, names.len);

    // Reference pixels are held for the whole run: every arm is compared against
    // them, and re-decoding per arm would dominate the runtime.
    const ref_px = try gpa.alloc([]u8, names.len);
    defer {
        for (ref_px) |p| if (p.len > 0) gpa.free(p);
        gpa.free(ref_px);
    }
    const ref_dims = try gpa.alloc([2]usize, names.len);
    defer gpa.free(ref_dims);

    for (names, 0..) |n, i| {
        // A named file is used verbatim; only a directory is indexed by name.
        const path = if (ref_is_dir) try std.fs.path.join(arena, &.{ ref_dir, n }) else opts.reference;
        ref_paths[i] = path;
        const l = try loadPng(gpa, arena, opts.io, opts.dir, path);
        ref_px[i] = l.dec.pixels;
        ref_dims[i] = .{ l.dec.width, l.dec.height };
        ref_detail[i] = try tp.image.detailEnergy(l.dec.pixels, l.dec.width, l.dec.height);
        ref_params[i] = l.params;
    }

    const arms = try arena.alloc(ArmResult, opts.candidates.len);
    for (opts.candidates, 0..) |cand, ai| {
        const cdir = try dirOf(arena, opts.io, opts.dir, cand);
        const cand_is_dir = isDir(opts.io, opts.dir, cand);
        // A single named candidate file can only stand in for a single reference
        // image. Silently pairing it with the first of many would compare
        // unrelated renders and report a confident number for it.
        if (!cand_is_dir and names.len != 1) return error.CandidateFileNeedsSingleReference;
        const images = try arena.alloc(ImageResult, names.len);

        for (names, 0..) |n, i| {
            const path = if (cand_is_dir) try std.fs.path.join(arena, &.{ cdir, n }) else cand;
            images[i] = .{
                .name = n,
                .psnr = null,
                .mse = null,
                .ssim = null,
                .lpips = null,
                .detail = null,
                .params = null,
                .path = path,
            };

            const l = loadPng(gpa, arena, opts.io, opts.dir, path) catch |err| {
                // A missing or unreadable candidate is reported per image rather
                // than aborting the run — a partial sweep is still informative,
                // and silently dropping it would overstate coverage.
                images[i].note = if (err == error.FileNotFound)
                    "missing"
                else
                    try std.fmt.allocPrint(arena, "unreadable: {t}", .{err});
                continue;
            };
            defer gpa.free(l.dec.pixels);

            images[i].params = l.params;
            if (l.dec.width != ref_dims[i][0] or l.dec.height != ref_dims[i][1]) {
                images[i].note = try std.fmt.allocPrint(arena, "size {d}x{d} vs reference {d}x{d}", .{
                    l.dec.width, l.dec.height, ref_dims[i][0], ref_dims[i][1],
                });
                continue;
            }

            images[i].psnr = try tp.image.psnr(ref_px[i], l.dec.pixels);
            images[i].mse = try tp.image.mse(ref_px[i], l.dec.pixels);
            images[i].detail = try tp.image.detailEnergy(l.dec.pixels, l.dec.width, l.dec.height);
            // An image below the window / the AlexNet tower's floor is reported as
            // a missing figure, not as a failed run: the other metrics still hold.
            images[i].ssim = tp.image.ssim(ref_px[i], l.dec.pixels, l.dec.width, l.dec.height, opts.ssim_window) catch |err| switch (err) {
                error.ImageTooSmall => null,
                else => return err,
            };
            if (lpips_net) |*net| {
                images[i].lpips = net.distance(opts.io, gpa, ref_px[i], l.dec.pixels, l.dec.width, l.dec.height) catch |err| switch (err) {
                    error.ImageTooSmall => null,
                    else => return err,
                };
            }
        }

        arms[ai] = .{ .name = try armLabel(arena, cand), .dir = cdir, .images = images };
    }

    return .{
        .arena = arena_state,
        .reference = try arena.dupe(u8, opts.reference),
        .ref_dir = ref_dir,
        .names = names,
        .ref_paths = ref_paths,
        .ref_detail = ref_detail,
        .ref_params = ref_params,
        .arms = arms,
    };
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

fn printOpt(w: *std.Io.Writer, v: ?f64, comptime fmt: []const u8) !void {
    if (v) |x| {
        if (std.math.isInf(x)) {
            try w.writeAll("identical");
        } else {
            try w.print(fmt, .{x});
        }
    } else {
        try w.writeAll("—");
    }
}

pub fn writeMarkdown(w: *std.Io.Writer, report: *const Report) !void {
    try w.writeAll("# Image verdict\n\n");
    try w.print("- reference: `{s}` ({d} image{s})\n", .{
        report.reference, report.names.len, if (report.names.len == 1) "" else "s",
    });
    for (report.arms) |a| {
        try w.print("- arm `{s}`: `{s}`", .{ a.name, if (a.images.len == 1) a.images[0].path else a.dir });
        if (a.missing() > 0) try w.print(" — **{d} images missing or unusable**", .{a.missing()});
        try w.writeByte('\n');
    }

    const any_lpips = report.hasLpips();

    try w.writeAll("\n## Arms, averaged\n\n");
    try w.writeAll("| arm | mean PSNR | mean SSIM |");
    if (any_lpips) try w.writeAll(" mean LPIPS |");
    try w.writeAll(" mean detail | vs reference detail |\n|---|---:|---:|");
    if (any_lpips) try w.writeAll("---:|");
    try w.writeAll("---:|---:|\n");

    var ref_detail_sum: f64 = 0;
    for (report.ref_detail) |d| ref_detail_sum += d;
    const ref_detail_mean = ref_detail_sum / @as(f64, @floatFromInt(report.ref_detail.len));
    try w.print("| _reference_ | — | — |{s} {d:.3} | — |\n", .{ if (any_lpips) " — |" else "", ref_detail_mean });

    for (report.arms) |a| {
        try w.print("| {s} | ", .{a.name});
        try printOpt(w, a.meanPsnr(), "{d:.3} dB");
        try w.writeAll(" | ");
        try printOpt(w, a.meanSsim(), "{d:.4}");
        try w.writeAll(" | ");
        if (any_lpips) {
            try printOpt(w, a.meanLpips(), "{d:.4}");
            try w.writeAll(" | ");
        }
        try printOpt(w, a.meanDetail(), "{d:.3}");
        try w.writeAll(" | ");
        if (a.meanDetail()) |d| {
            try w.print("{d:.1}%", .{100.0 * d / ref_detail_mean});
        } else {
            try w.writeAll("—");
        }
        try w.writeAll(" |");
        if (a.identical() > 0) try w.print(" _{d} identical to reference_", .{a.identical()});
        try w.writeByte('\n');
    }

    try w.writeAll(
        "\n> **PSNR between two diffusion renders is confounded.** Quantization perturbs the denoising\n" ++
            "> trajectory, so an arm can draw a *different* picture rather than a worse one, and PSNR\n" ++
            "> cannot tell those apart. `detail` is per-image gradient energy: an arm holding the\n" ++
            "> reference's detail while scoring badly on PSNR drifted in composition; one losing detail\n" ++
            "> degraded. Neither is a substitute for looking at the contact sheet.\n",
    );
    if (any_lpips) {
        try w.writeAll(
            ">\n> **LPIPS (lower is better) is the metric to weight most heavily** — it is the one\n" ++
                "> trained against human judgement, and it has more power at small n than PSNR, whose\n" ++
                "> standard error over a 12-image set was measured at 0.26–0.51 dB.\n",
        );
    } else {
        try w.writeAll(
            ">\n> _No LPIPS column: pass `--lpips <weights.safetensors>` to add the one metric here that\n" ++
                "> tracks human judgement._\n",
        );
    }

    try w.writeAll("\n## Per image\n\n");
    try w.writeAll("| image |");
    for (report.arms) |a| {
        try w.print(" {s} PSNR | {s} SSIM |", .{ a.name, a.name });
        if (any_lpips) try w.print(" {s} LPIPS |", .{a.name});
        try w.print(" {s} detail |", .{a.name});
    }
    try w.writeAll("\n|---|");
    for (report.arms) |_| {
        try w.writeAll("---:|---:|---:|");
        if (any_lpips) try w.writeAll("---:|");
    }
    try w.writeByte('\n');

    for (report.names, 0..) |n, i| {
        try w.print("| `{s}` (ref detail {d:.2}) |", .{ n, report.ref_detail[i] });
        for (report.arms) |a| {
            const im = a.images[i];
            try w.writeAll(" ");
            if (im.note) |note| {
                try w.print("_{s}_", .{note});
            } else {
                try printOpt(w, im.psnr, "{d:.3}");
            }
            try w.writeAll(" | ");
            try printOpt(w, im.ssim, "{d:.4}");
            try w.writeAll(" | ");
            if (any_lpips) {
                try printOpt(w, im.lpips, "{d:.4}");
                try w.writeAll(" | ");
            }
            try printOpt(w, im.detail, "{d:.2}");
            try w.writeAll(" |");
        }
        try w.writeByte('\n');
    }

    // Generation parameters, so the report is self-describing rather than relying
    // on whoever ran it to remember the seed.
    var any_params = false;
    for (report.ref_params) |p| {
        if (p != null) any_params = true;
    }
    if (any_params) {
        try w.writeAll("\n## Reference generation parameters\n\n");
        for (report.names, report.ref_params) |n, p| {
            if (p) |text| try w.print("**`{s}`**\n\n```\n{s}\n```\n\n", .{ n, text });
        }
    }
}

/// A self-contained contact sheet: one row per image, arms side by side, images
/// referenced by relative path so the file stays small and the originals stay
/// authoritative.
pub fn writeHtml(w: *std.Io.Writer, report: *const Report, out_dir: []const u8) !void {
    try w.writeAll(
        \\<!doctype html><meta charset="utf-8"><title>ggufy image verdict</title>
        \\<style>
        \\body{font:14px/1.5 system-ui,sans-serif;margin:2rem;background:#111;color:#ddd}
        \\h1{font-size:1.3rem} h2{font-size:1rem;font-weight:600;margin:2rem 0 .5rem}
        \\.row{display:flex;gap:.5rem;align-items:flex-start;overflow-x:auto;padding-bottom:.5rem}
        \\figure{margin:0;flex:0 0 auto}
        \\img{max-height:70vh;width:auto;display:block;background:#222}
        \\figcaption{font-size:12px;color:#9ab;padding:.25rem 0}
        \\.miss{padding:2rem;color:#c66;border:1px dashed #444}
        \\code{color:#9c9}
        \\
    );
    try w.writeAll("</style>\n<h1>Image verdict</h1>\n");
    try w.print("<p>reference <code>{s}</code>, {d} images, {d} arms.</p>\n", .{
        report.reference, report.names.len, report.arms.len,
    });
    try w.writeAll(
        "<p>PSNR between diffusion renders mixes composition drift with quality loss — " ++
            "compare <code>detail</code> against the reference's to separate them, and trust your eyes over both.</p>\n",
    );
    if (report.hasLpips()) try w.writeAll(
        "<p><b>lpips</b> (lower is better) is the perceptual metric trained against human judgement; " ++
            "weight it above PSNR when they disagree.</p>\n",
    );

    for (report.names, 0..) |n, i| {
        try w.print("<h2>{s}</h2>\n<div class=\"row\">\n", .{n});

        const ref_rel = try relativeTo(report.arena.child_allocator, out_dir, report.ref_paths[i]);
        defer report.arena.child_allocator.free(ref_rel);
        try w.print(
            "<figure><img src=\"{s}\" loading=\"lazy\"><figcaption>reference · detail {d:.2}</figcaption></figure>\n",
            .{ ref_rel, report.ref_detail[i] },
        );

        for (report.arms) |a| {
            const im = a.images[i];
            if (im.note) |note| {
                try w.print("<figure><div class=\"miss\">{s}</div><figcaption>{s}</figcaption></figure>\n", .{ note, a.name });
                continue;
            }
            const rel = try relativeTo(report.arena.child_allocator, out_dir, im.path);
            defer report.arena.child_allocator.free(rel);
            try w.print("<figure><img src=\"{s}\" loading=\"lazy\"><figcaption>{s} · ", .{ rel, a.name });
            try printOpt(w, im.psnr, "{d:.2} dB");
            if (im.ssim) |s| try w.print(" · ssim {d:.4}", .{s});
            if (im.lpips) |l| try w.print(" · <b>lpips {d:.4}</b>", .{l});
            if (im.detail) |d| try w.print(" · detail {d:.2}", .{d});
            try w.writeAll("</figcaption></figure>\n");
        }
        try w.writeAll("</div>\n");
    }
}

/// `path` expressed relative to the directory the sheet is written into, so a
/// sheet written next to the images just references their filenames.
fn relativeTo(gpa: std.mem.Allocator, from: []const u8, path: []const u8) ![]const u8 {
    const dir = std.fs.path.dirname(path) orelse ".";
    if (std.mem.eql(u8, from, dir)) return gpa.dupe(u8, std.fs.path.basename(path));
    return gpa.dupe(u8, path);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Write a PNG of a solid colour with an optional `parameters` chunk.
fn writeTestPng(
    gpa: std.mem.Allocator,
    dir: std.Io.Dir,
    io: std.Io,
    name: []const u8,
    w: usize,
    h: usize,
    fill: u8,
    checker: bool,
    params: ?[]const u8,
) !void {
    const px = try gpa.alloc(u8, w * h * 3);
    defer gpa.free(px);
    for (0..h) |y| for (0..w) |x| {
        const v: u8 = if (checker and (x + y) % 2 == 1) 255 - fill else fill;
        const i = (y * w + x) * 3;
        px[i] = v;
        px[i + 1] = v;
        px[i + 2] = v;
    };

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    if (params) |p| {
        try tp.image.encodePngRgbText(gpa, &out, px, w, h, &.{.{ .keyword = "parameters", .text = p }});
    } else {
        try tp.image.encodePngRgb(gpa, &out, px, w, h);
    }

    const f = try dir.createFile(io, name, .{ .truncate = true });
    defer f.close(io);
    var buf: [1 << 16]u8 = undefined;
    var fw = f.writer(io, &buf);
    try fw.interface.writeAll(out.items);
    try fw.interface.flush();
}

test "a verdict run scores each arm and survives a missing candidate" {
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = tmp.dir;

    try root.createDirPath(io, "ref");
    try root.createDirPath(io, "good");
    try root.createDirPath(io, "bad");

    var ref_d = try root.openDir(io, "ref", .{});
    defer ref_d.close(io);
    var good_d = try root.openDir(io, "good", .{});
    defer good_d.close(io);
    var bad_d = try root.openDir(io, "bad", .{});
    defer bad_d.close(io);

    // Two reference images, one textured; `good` reproduces them exactly, `bad`
    // is flat (all detail gone) and is missing the second image entirely.
    try writeTestPng(gpa, ref_d, io, "a.png", 8, 8, 100, true, "Steps: 16, Seed: 80085");
    try writeTestPng(gpa, ref_d, io, "b.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, good_d, io, "a.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, good_d, io, "b.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, bad_d, io, "a.png", 8, 8, 100, false, null);

    var report = try run(gpa, .{ .io = io, .dir = root, .reference = "ref", .candidates = &.{ "good", "bad" } });
    defer report.deinit();

    try testing.expectEqual(@as(usize, 2), report.names.len);
    try testing.expectEqualStrings("a.png", report.names[0]);
    try testing.expectEqual(@as(usize, 2), report.arms.len);

    // Identical renders must read as identical, not as some large finite PSNR
    // that a mean would then quietly absorb.
    const good = report.arms[0];
    try testing.expectEqualStrings("good", good.name);
    try testing.expectEqual(@as(usize, 2), good.identical());
    try testing.expect(good.meanPsnr() == null);
    try testing.expectEqual(@as(usize, 0), good.missing());

    // The flat arm scores finitely, loses essentially all detail, and its second
    // image is reported missing rather than dropped.
    const bad = report.arms[1];
    try testing.expect(bad.meanPsnr().? < 20.0);
    try testing.expect(bad.images[0].detail.? < 1.0);
    // The reference checkerboard alternates 100 <-> 155, so its gradient RMS is
    // ~55 — the point is that it is orders above the flat arm, not its exact value.
    try testing.expect(report.ref_detail[0] > 50.0);
    try testing.expectEqual(@as(usize, 1), bad.missing());
    try testing.expectEqualStrings("missing", bad.images[1].note.?);

    // Generation parameters come back off the reference image.
    try testing.expectEqualStrings("Steps: 16, Seed: 80085", report.ref_params[0].?);
    try testing.expect(report.ref_params[1] == null);

    // Both report forms must render without erroring, and must mention the
    // confound rather than presenting PSNR bare.
    var md: std.Io.Writer.Allocating = .init(gpa);
    defer md.deinit();
    try writeMarkdown(&md.writer, &report);
    try testing.expect(std.mem.indexOf(u8, md.written(), "confounded") != null);
    try testing.expect(std.mem.indexOf(u8, md.written(), "missing") != null);
    try testing.expect(std.mem.indexOf(u8, md.written(), "identical") != null);

    var html: std.Io.Writer.Allocating = .init(gpa);
    defer html.deinit();
    try writeHtml(&html.writer, &report, "ref");
    try testing.expect(std.mem.indexOf(u8, html.written(), "<img") != null);
    try testing.expect(std.mem.indexOf(u8, html.written(), "a.png") != null);
}

test "a size mismatch is reported, not silently compared" {
    // Comparing a 512-square render against a 1120x1680 one would either crash or
    // produce a meaningless number; both are worse than saying so.
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = tmp.dir;
    try root.createDirPath(io, "r");
    try root.createDirPath(io, "c");

    var r = try root.openDir(io, "r", .{});
    defer r.close(io);
    var c = try root.openDir(io, "c", .{});
    defer c.close(io);
    try writeTestPng(gpa, r, io, "x.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, c, io, "x.png", 8, 4, 100, true, null);

    var report = try run(gpa, .{ .io = io, .dir = root, .reference = "r", .candidates = &.{"c"} });
    defer report.deinit();

    try testing.expect(report.arms[0].images[0].psnr == null);
    try testing.expect(std.mem.indexOf(u8, report.arms[0].images[0].note.?, "size 8x4") != null);
}

test "named files are compared to each other, not the reference to itself" {
    // The n=1 case is how this gets used first, and it is where the resolution
    // logic is easiest to get wrong: a directory is indexed by filename, but a
    // named file must be used verbatim. Joining the reference's *name* onto the
    // candidate's *directory* makes every arm re-load the reference and report a
    // confident "identical" — which is exactly what happened, and what the
    // original `psnr > 0` assertion failed to catch, since infinity passes it.
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = tmp.dir;
    try writeTestPng(gpa, root, io, "ref.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, root, io, "cand_a.png", 8, 8, 120, true, null);
    try writeTestPng(gpa, root, io, "cand_b.png", 8, 8, 140, true, null);

    var report = try run(gpa, .{
        .io = io,
        .dir = root,
        .reference = "ref.png",
        .candidates = &.{ "cand_a.png", "cand_b.png" },
    });
    defer report.deinit();

    try testing.expectEqual(@as(usize, 1), report.names.len);
    try testing.expectEqualStrings("cand_a", report.arms[0].name);
    try testing.expectEqualStrings("cand_b", report.arms[1].name);

    // Each arm must have loaded its OWN file: finite PSNR, and the two arms must
    // differ from each other since their fills do.
    const a = report.arms[0].images[0];
    const b = report.arms[1].images[0];
    try testing.expect(std.math.isFinite(a.psnr.?));
    try testing.expect(std.math.isFinite(b.psnr.?));
    try testing.expect(a.psnr.? != b.psnr.?);
    try testing.expectEqual(@as(usize, 0), report.arms[0].identical());
    try testing.expectEqualStrings("cand_a.png", a.path);
    try testing.expectEqualStrings("cand_b.png", b.path);

    // ...and cand_b is further from the reference than cand_a (fill 140 vs 120
    // against a reference of 100), so the ordering must come out right too.
    try testing.expect(b.psnr.? < a.psnr.?);
}

test "a named candidate file cannot stand in for a directory of references" {
    // Pairing one file against many references would compare unrelated renders
    // and report a confident number for each.
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = tmp.dir;
    try root.createDirPath(io, "ref");
    var r = try root.openDir(io, "ref", .{});
    defer r.close(io);
    try writeTestPng(gpa, r, io, "a.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, r, io, "b.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, root, io, "one.png", 8, 8, 120, true, null);

    try testing.expectError(error.CandidateFileNeedsSingleReference, run(gpa, .{
        .io = io,
        .dir = root,
        .reference = "ref",
        .candidates = &.{"one.png"},
    }));
}

test "no candidates and no reference images are refused" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try testing.expectError(error.NoCandidates, run(gpa, .{ .io = io, .dir = tmp.dir, .reference = ".", .candidates = &.{} }));
    try testing.expectError(error.NoReferenceImages, run(gpa, .{ .io = io, .dir = tmp.dir, .reference = ".", .candidates = &.{"."} }));
}

test "ssim is reported per image, and reads 1 for an identical render" {
    // SSIM comes from tp.image and is pinned against scikit-image there; what this
    // checks is that the verdict wires it per (arm, image) rather than, say,
    // comparing the reference to itself.
    const gpa = testing.allocator;
    const io = testing.io;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = tmp.dir;
    try writeTestPng(gpa, root, io, "ref.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, root, io, "same.png", 8, 8, 100, true, null);
    try writeTestPng(gpa, root, io, "flat.png", 8, 8, 100, false, null);

    var report = try run(gpa, .{
        .io = io,
        .dir = root,
        .reference = "ref.png",
        .candidates = &.{ "same.png", "flat.png" },
    });
    defer report.deinit();

    try testing.expectApproxEqAbs(1.0, report.arms[0].images[0].ssim.?, 1e-12);
    // A flat render against a checkerboard shares no structure: the luminance term
    // survives (both average ~100-127) but the contrast and covariance terms
    // collapse, which lands it near 0.07 rather than at 0.
    try testing.expect(report.arms[1].images[0].ssim.? < 0.1);
    try testing.expectApproxEqAbs(report.arms[0].images[0].ssim.?, report.arms[0].meanSsim().?, 1e-12);

    // No weights were given, so there is no LPIPS column — and the report says how
    // to get one rather than leaving a reader to wonder.
    try testing.expect(!report.hasLpips());
    var md: std.Io.Writer.Allocating = .init(gpa);
    defer md.deinit();
    try writeMarkdown(&md.writer, &report);
    try testing.expect(std.mem.indexOf(u8, md.written(), "mean SSIM") != null);
    try testing.expect(std.mem.indexOf(u8, md.written(), "LPIPS") != null);
    try testing.expect(std.mem.indexOf(u8, md.written(), "--lpips") != null);
}

/// The LPIPS weights are a user-supplied checkpoint, not a repo fixture (~10 MB),
/// so this test self-skips when they are absent. The path is where
/// TensorPencil's `tools/gen_lpips_fixtures.py` writes them, spelled relative to
/// this repo the same way `build.zig.zon` spells the TP dependency itself.
const lpips_weights_path = "../../../../dump/projects/zig/TensorPencil/models/lpips/lpips_alex.safetensors";

fn lpipsWeightsPresent(io: std.Io) bool {
    std.Io.Dir.cwd().access(io, lpips_weights_path, .{}) catch return false;
    return true;
}

test "the lpips column is populated when weights are supplied" {
    const gpa = testing.allocator;
    const io = testing.io;
    if (!lpipsWeightsPresent(io)) return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = tmp.dir;
    // 64-square: below ~32 the AlexNet tower has no spatial extent left, and the
    // 8-square images the other tests use would come back null.
    try writeTestPng(gpa, root, io, "ref.png", 64, 64, 100, true, null);
    try writeTestPng(gpa, root, io, "same.png", 64, 64, 100, true, null);
    try writeTestPng(gpa, root, io, "flat.png", 64, 64, 100, false, null);

    var report = try run(gpa, .{
        .io = io,
        .dir = root,
        .reference = "ref.png",
        .candidates = &.{ "same.png", "flat.png" },
        .lpips_weights = lpips_weights_path,
    });
    defer report.deinit();

    try testing.expect(report.hasLpips());
    // Identical pixels give identical features, so the distance is exactly zero —
    // not merely small.
    try testing.expectEqual(@as(f64, 0), report.arms[0].images[0].lpips.?);
    // Losing the entire checkerboard is a large perceptual distance, and it must
    // come out *larger* than the identical arm's, i.e. the sign is right.
    try testing.expect(report.arms[1].images[0].lpips.? > 0.1);

    var md: std.Io.Writer.Allocating = .init(gpa);
    defer md.deinit();
    try writeMarkdown(&md.writer, &report);
    try testing.expect(std.mem.indexOf(u8, md.written(), "mean LPIPS") != null);
    try testing.expect(std.mem.indexOf(u8, md.written(), "human judgement") != null);

    var html: std.Io.Writer.Allocating = .init(gpa);
    defer html.deinit();
    try writeHtml(&html.writer, &report, ".");
    try testing.expect(std.mem.indexOf(u8, html.written(), "lpips") != null);
}
