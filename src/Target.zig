//! Which program an output is meant for, and which programs can load the file a
//! conversion actually writes. The loader facts here are checked against each
//! reader's source; when one changes, change the table, not the callers.

const std = @import("std");
const types = @import("types.zig");
const HfLlm = @import("HfLlm.zig");

pub const Target = enum {
    /// ComfyUI for a diffusion model, llama.cpp for an LLM.
    auto,
    comfyui,
    llamacpp,
    transformers,
    /// No derivation: -f, -H and -A as given.
    custom,

    pub fn label(self: Target) []const u8 {
        return switch (self) {
            .auto => "automatic",
            .comfyui => "ComfyUI",
            .llamacpp => "llama.cpp",
            .transformers => "transformers",
            .custom => "custom",
        };
    }
};

pub const Kind = enum { diffusion, llm };

pub const Consumer = enum {
    comfyui,
    llamacpp,
    transformers,

    pub fn label(self: Consumer) []const u8 {
        return switch (self) {
            .comfyui => "ComfyUI",
            .llamacpp => "llama.cpp",
            .transformers => "transformers",
        };
    }
};

// Upstream city96/ComfyUI-GGUF loader.py. ComfyUI loads GGUF only through an
// extension, and other extensions or patched copies take more, so a missing
// architecture is warned about rather than refused.
const comfy_gguf_image_archs = [_][]const u8{ "flux", "sd1", "sdxl", "sd3", "aura", "hidream", "cosmos", "ltxv", "hyvid", "wan", "lumina2", "qwen_image" };
pub const comfy_gguf_text_archs = [_][]const u8{ "t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3" };

fn listed(list: []const []const u8, name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

pub fn comfyGgufLoadsImage(arch_name: []const u8) bool {
    return listed(&comfy_gguf_image_archs, arch_name);
}

pub fn comfyGgufLoadsText(arch_name: []const u8) bool {
    return listed(&comfy_gguf_text_archs, arch_name);
}

fn comfyGgufLoads(kind: Kind, arch_name: []const u8) bool {
    return switch (kind) {
        .diffusion => comfyGgufLoadsImage(arch_name),
        .llm => comfyGgufLoadsText(arch_name),
    };
}

/// The ggml types ComfyUI-GGUF's dequant.py decodes, plus the two it views directly.
fn comfyGgufDecodes(t: types.DataType) bool {
    return switch (t) {
        .f32, .f16, .bf16, .q8_0, .q5_1, .q5_0, .q4_1, .q4_0, .q6_k, .q5_k, .q4_k, .q3_k, .q2_k, .iq4_nl, .iq4_xs => true,
        else => false,
    };
}

/// The rest of what gguf-py's numpy dequantize decodes, which ComfyUI-GGUF falls back to on
/// the CPU for every forward pass.
fn ggufPyDecodes(t: types.DataType) bool {
    return switch (t) {
        .tq1_0, .tq2_0, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq1_s, .iq1_m => true,
        else => false,
    };
}

/// The first ComfyUI release that loads a safetensors tensor of this type, or
/// null when no release does. Plain floats and unscaled fp8 load everywhere.
pub fn comfyMinVersion(t: types.DataType) ?[]const u8 {
    return switch (t) {
        .F64, .F32, .F16, .BF16, .F8_E4M3, .F8_E5M2, .I8, .I16, .I32, .I64, .U8, .U16, .U32, .U64 => "",
        .SCALED_F8_E4M3 => "v0.3.68",
        .NVFP4 => "v0.8.0",
        .MXFP8_E4M3 => "v0.18.0",
        .INT8, .INT8_CONVROT => "v0.27.0",
        .INT4_CONVROT, .INT4_CONVROT_SR => "v0.28.0",
        .ASYM_W4A8_INT8 => "v0.31.0",
        .W6A8_INT8 => "v0.38.0",
        else => null,
    };
}

/// What every ComfyUI release gets wrong about a type it does load, or null.
pub fn comfyCaveat(t: types.DataType) ?[]const u8 {
    return if (t == .ASYM_W4A8_INT8_ZP)
        "every ComfyUI release drops weight_correction on load and decodes these layers wrong; comfy_kitchen reads them"
    else
        null;
}

/// Why ComfyUI's loader for `ft` cannot read a tensor of type `t`, or null when
/// it can. A passthrough tensor can sit in the other container's file.
fn comfyIssue(ft: types.FileType, t: types.DataType) ?[]const u8 {
    return switch (ft) {
        .safetensors => comfyCaveat(t) orelse if (comfyMinVersion(t) == null)
            "ComfyUI has no loader for this type"
        else
            null,
        .gguf => if (comfyGgufDecodes(t))
            null
        else if (ggufPyDecodes(t))
            "ComfyUI-GGUF decodes this type only through gguf-py's CPU fallback, which is very slow"
        else
            "ComfyUI-GGUF has no decoder for this type",
    };
}

/// Why `target` cannot load a tensor of type `t`, or null when it can. Only
/// the type is judged here; the architecture and naming are `assess`'s.
pub fn dtypeIssue(target: Target, t: types.DataType) ?[]const u8 {
    return switch (target) {
        .comfyui => comfyIssue(t.formatType(), t),
        .llamacpp => if (t.formatType() == .gguf) null else "llama.cpp reads GGUF types only",
        .transformers => if (t.formatType() == .safetensors and (t.isFloatType() or plainInt(t))) null else "transformers cannot load this tool's quantized layouts; use F16, BF16 or F32",
        .auto, .custom => null,
    };
}

/// A plain integer tensor, such as a position_ids buffer, which torch reads as is.
fn plainInt(t: types.DataType) bool {
    return switch (t) {
        .I8, .I16, .I32, .I64, .U8, .U16, .U32, .U64 => true,
        else => false,
    };
}

fn versionLess(a: []const u8, b: []const u8) bool {
    var ia = std.mem.tokenizeAny(u8, a, "v.");
    var ib = std.mem.tokenizeAny(u8, b, "v.");
    while (true) {
        const x = ia.next();
        const y = ib.next();
        if (x == null and y == null) return false;
        const xn = if (x) |s| std.fmt.parseInt(u32, s, 10) catch 0 else 0;
        const yn = if (y) |s| std.fmt.parseInt(u32, s, 10) catch 0 else 0;
        if (xn != yn) return xn < yn;
    }
}

/// Facts about a prepared output, gathered after type assignment.
pub const Output = struct {
    kind: Kind,
    filetype: types.FileType,
    /// general.architecture as written, for GGUF; the detected name otherwise.
    arch_name: []const u8,
    /// Tensor names are HF state-dict names rather than llama.cpp's or ComfyUI's.
    hf_names: bool = false,
    /// config.json and tokenizer files are written beside the weights.
    hf_sidecars: bool = false,
    /// The vision tower goes to a separate mmproj GGUF beside the text model.
    mmproj: bool = false,
    /// The model has a vision tower, wherever it ends up.
    has_vision: bool = false,
    llama_heads: ?struct { head_count: u32, head_count_kv: u32 } = null,
    dtypes: std.EnumSet(types.DataType) = .initEmpty(),
};

pub const Level = enum { yes, partial, no };

pub const Verdict = struct {
    consumer: Consumer,
    level: Level,
    /// Why not, or what it needs; empty for a plain yes.
    note: []const u8 = "",
    /// The note is worth a warning even though the output may load.
    warn: bool = false,
};

pub fn assess(out: Output, alloc: std.mem.Allocator) ![3]Verdict {
    return .{
        try assessComfy(out, alloc),
        assessLlamaCpp(out),
        try assessTransformers(out, alloc),
    };
}

fn assessComfy(out: Output, alloc: std.mem.Allocator) !Verdict {
    if (out.filetype == .gguf) return assessComfyGguf(out, alloc);
    var need: []const u8 = "";
    var it = out.dtypes.iterator();
    while (it.next()) |t| {
        if (comfyIssue(.safetensors, t)) |why| return .{ .consumer = .comfyui, .level = .no, .note = try std.fmt.allocPrint(alloc, "{s}: {s}", .{ @tagName(t), why }) };
        const min = comfyMinVersion(t).?;
        if (versionLess(need, min)) need = min;
    }
    if (out.kind == .llm) {
        if (!out.hf_names) return .{ .consumer = .comfyui, .level = .no, .note = "ComfyUI text encoders take HF tensor names (-H)" };
        if (out.hf_sidecars) return .{ .consumer = .comfyui, .level = .partial, .note = "as a text encoder; the HF directory's config and tokenizer files are not needed there" };
        return .{ .consumer = .comfyui, .level = .partial, .note = "as a text encoder, for a model that uses this LLM" };
    }
    var note: []const u8 = "";
    if (need.len > 0) note = try std.fmt.allocPrint(alloc, "{s} or newer", .{need});
    if (out.dtypes.contains(.MXFP8_E4M3)) note = try std.fmt.allocPrint(alloc, "{s}, with a comfy_kitchen that has MXFP8", .{note});
    return .{ .consumer = .comfyui, .level = .yes, .note = note };
}

const via_ext = "through a GGUF loader extension such as ComfyUI-GGUF";

// Judged against upstream ComfyUI-GGUF, but another extension or a later release may do
// better, so what it cannot do is a warning, never a refusal.
fn assessComfyGguf(out: Output, alloc: std.mem.Allocator) !Verdict {
    var warns: std.ArrayList([]const u8) = .empty;
    var it = out.dtypes.iterator();
    while (it.next()) |t| if (comfyIssue(.gguf, t)) |why| {
        try warns.append(alloc, try std.fmt.allocPrint(alloc, "{s}: {s}", .{ @tagName(t), why }));
    };
    if (!comfyGgufLoads(out.kind, out.arch_name))
        try warns.append(alloc, try std.fmt.allocPrint(alloc, "ComfyUI-GGUF does not load {s}, so it needs a GGUF loader extension that does", .{out.arch_name}));
    if (out.kind == .llm and std.mem.eql(u8, out.arch_name, "llama")) {
        // Its llama branch un-permutes Q/K with hardcoded head counts.
        if (out.llama_heads) |h| {
            if (h.head_count != 32 or h.head_count_kv != 8)
                try warns.append(alloc, try std.fmt.allocPrint(alloc, "ComfyUI-GGUF un-permutes llama Q/K assuming 32 heads and 8 KV heads; this model has {d} and {d}, so its attention would be scrambled", .{ h.head_count, h.head_count_kv }));
        } else try warns.append(alloc, "ComfyUI-GGUF un-permutes llama Q/K assuming 32 heads and 8 KV heads; this file does not say how many it has");
    }
    if (warns.items.len > 0) return .{ .consumer = .comfyui, .level = .partial, .warn = true, .note = try std.mem.join(alloc, "; ", warns.items) };

    if (out.kind == .llm) {
        if (out.has_vision) {
            if (std.mem.eql(u8, out.arch_name, "qwen2vl") and !out.mmproj)
                return .{ .consumer = .comfyui, .level = .partial, .note = "as a text encoder without vision: ComfyUI-GGUF reads Qwen2.5-VL's tower from an mmproj beside the file" };
            if (std.mem.eql(u8, out.arch_name, "qwen3vl") and !out.hf_names)
                return .{ .consumer = .comfyui, .level = .partial, .note = "as a text encoder without vision: Qwen3-VL's tower has to be in the same file, under HF names (--target comfyui)" };
        }
        return .{ .consumer = .comfyui, .level = .yes, .note = "as a text encoder, " ++ via_ext };
    }
    return .{ .consumer = .comfyui, .level = .yes, .note = via_ext };
}

fn assessLlamaCpp(out: Output) Verdict {
    const no: Verdict = .{ .consumer = .llamacpp, .level = .no };
    if (out.filetype != .gguf) return no;
    if (out.kind != .llm) return .{ .consumer = .llamacpp, .level = .no, .note = "llama.cpp runs LLMs only" };
    if (out.hf_names) return .{ .consumer = .llamacpp, .level = .no, .note = "llama.cpp needs its own blk.* tensor names" };
    if (std.mem.eql(u8, out.arch_name, "unknown")) return .{ .consumer = .llamacpp, .level = .no, .note = "the file names no architecture" };
    if (out.has_vision and !out.mmproj) return .{ .consumer = .llamacpp, .level = .partial, .note = "text only: no mmproj is written" };
    if (out.mmproj) return .{ .consumer = .llamacpp, .level = .yes, .note = "with the mmproj beside it for vision" };
    return .{ .consumer = .llamacpp, .level = .yes };
}

fn assessTransformers(out: Output, alloc: std.mem.Allocator) !Verdict {
    const no: Verdict = .{ .consumer = .transformers, .level = .no };
    if (out.filetype != .safetensors or out.kind != .llm) return no;
    if (!out.hf_names) return .{ .consumer = .transformers, .level = .no, .note = "transformers needs HF tensor names (-H)" };
    var it = out.dtypes.iterator();
    while (it.next()) |t| if (dtypeIssue(.transformers, t)) |why| {
        return .{ .consumer = .transformers, .level = .no, .note = try std.fmt.allocPrint(alloc, "{s}: {s}", .{ @tagName(t), why }) };
    };
    if (!out.hf_sidecars) return .{ .consumer = .transformers, .level = .partial, .note = "with the source checkpoint's config.json and tokenizer copied beside it" };
    return .{ .consumer = .transformers, .level = .yes };
}

pub fn verdictFor(verdicts: []const Verdict, c: Consumer) Verdict {
    for (verdicts) |v| if (v.consumer == c) return v;
    unreachable;
}

/// The verdict of the program a target means, or null for auto and custom.
pub fn targetVerdict(verdicts: []const Verdict, t: Target) ?Verdict {
    return switch (t) {
        .comfyui => verdictFor(verdicts, .comfyui),
        .llamacpp => verdictFor(verdicts, .llamacpp),
        .transformers => verdictFor(verdicts, .transformers),
        .auto, .custom => null,
    };
}

/// One line naming everything that loads the output, e.g.
/// "ComfyUI (v0.28.0 or newer)". "nothing this tool knows of" when empty.
pub fn worksWith(verdicts: []const Verdict, alloc: std.mem.Allocator) ![]const u8 {
    var parts: std.ArrayList([]const u8) = .empty;
    for (verdicts) |v| {
        if (v.level == .no) continue;
        try parts.append(alloc, if (v.note.len == 0)
            v.consumer.label()
        else if (v.warn)
            try std.fmt.allocPrint(alloc, "{s} (may not work, or work poorly: {s})", .{ v.consumer.label(), v.note })
        else
            try std.fmt.allocPrint(alloc, "{s} ({s})", .{ v.consumer.label(), v.note }));
    }
    if (parts.items.len == 0) return "nothing this tool knows of";
    return std.mem.join(alloc, "; ", parts.items);
}

pub const Plan = struct {
    target: Target,
    filetype: types.FileType,
    hf_names: bool,
};

pub const PlanError = error{ TargetCannotLoad, OutOfMemory };

/// Auto as the target it stands for: ComfyUI for a diffusion model, llama.cpp for an LLM.
pub fn resolve(target: Target, kind: Kind) Target {
    return if (target == .auto) switch (kind) {
        .diffusion => .comfyui,
        .llm => .llamacpp,
    } else target;
}

/// Whether `target` reads a `ft` file of this model at all; `plan` picks the
/// container from this.
pub fn readsContainer(target: Target, kind: Kind, ft: types.FileType) bool {
    return switch (resolve(target, kind)) {
        .comfyui => true,
        .llamacpp => kind == .llm and ft == .gguf,
        .transformers => kind == .llm and ft == .safetensors,
        .auto, .custom => true,
    };
}

/// Why `target` takes only one container for this model, or null when it reads both.
pub fn containerReason(target: Target, kind: Kind) ?[]const u8 {
    if (readsContainer(target, kind, .gguf) and readsContainer(target, kind, .safetensors)) return null;
    return switch (resolve(target, kind)) {
        .llamacpp => "llama.cpp reads GGUF only.",
        .transformers => "transformers reads safetensors.",
        else => unreachable,
    };
}

/// Whether `target` reads this LLM's tensors under HF names in a `ft` file. A
/// safetensors source already carries them, so only a GGUF one asks for -H.
/// Custom leaves it to the caller.
pub fn hfNames(target: Target, kind: Kind, arch_name: []const u8, ft: types.FileType, source_is_gguf: bool) bool {
    if (kind != .llm) return false;
    return switch (resolve(target, kind)) {
        // ComfyUI takes HF names from safetensors, and from GGUF only for
        // Qwen3-VL, whose tower it reads from the same file.
        .comfyui => if (ft == .safetensors) source_is_gguf else std.mem.eql(u8, arch_name, "qwen3vl"),
        .transformers => source_is_gguf,
        .llamacpp, .auto, .custom => false,
    };
}

/// Whether `target` takes this architecture's split MLP fused into one gate_up tensor.
/// ComfyUI maps LoRAs under either naming onto a fused MLP but only the split names onto a
/// split one, and only its fused MLP takes the int8 SwiGLU fast path.
pub fn fusesMlp(target: Target, arch_name: []const u8) bool {
    return target == .comfyui and std.mem.eql(u8, arch_name, "qwen_image21");
}

/// What a target asks of a conversion of `arch_name`. `datatype` null means the
/// default, and `requested` is a container the user asked for, which the target
/// plans around instead of choosing one. A safetensors LLM source already
/// carries HF names, so only a GGUF one asks for -H.
/// Fails when the target cannot load this output, with the reason in `why`.
pub fn plan(target: Target, kind: Kind, arch_name: []const u8, datatype: ?types.DataType, requested: ?types.FileType, source_is_gguf: bool, hf_config_writable: bool, alloc: std.mem.Allocator, why: *[]const u8) PlanError!Plan {
    const t = resolve(target, kind);
    std.debug.assert(t != .custom);
    const container: ?types.FileType = requested orelse if (datatype) |d| d.formatType() else null;

    switch (t) {
        .comfyui => {
            // With no type the output is unquantized, which ComfyUI loads as safetensors with no
            // extension. A GGUF source stays GGUF: safetensors from it would need a generated HF layout.
            const ft = container orelse if (source_is_gguf) types.FileType.gguf else .safetensors;
            const hf = hfNames(t, kind, arch_name, ft, source_is_gguf);
            // HF names out of a GGUF take the HF layout, config.json and all.
            if (hf and ft == .safetensors and !hf_config_writable) {
                why.* = try std.fmt.allocPrint(alloc, "ComfyUI takes a text encoder as safetensors under HF names, and this tool writes that layout for {s} only, not {s}", .{ HfLlm.hf_config_archs_text, arch_name });
                return error.TargetCannotLoad;
            }
            return .{ .target = t, .filetype = ft, .hf_names = hf };
        },
        .llamacpp => {
            if (kind != .llm) {
                why.* = try std.fmt.allocPrint(alloc, "llama.cpp runs LLMs; {s} is a diffusion model", .{arch_name});
                return error.TargetCannotLoad;
            }
            if (container == .safetensors) {
                why.* = if (datatype != null and datatype.?.formatType() == .safetensors)
                    try std.fmt.allocPrint(alloc, "llama.cpp reads GGUF; {s} is a safetensors type. Choose a GGUF type (f16, q8_0, q4_k, ...)", .{@tagName(datatype.?)})
                else
                    "llama.cpp reads GGUF, not safetensors";
                return error.TargetCannotLoad;
            }
            return .{ .target = t, .filetype = .gguf, .hf_names = hfNames(t, kind, arch_name, .gguf, source_is_gguf) };
        },
        .transformers => {
            if (kind != .llm) {
                why.* = try std.fmt.allocPrint(alloc, "transformers output is for LLMs; {s} is a diffusion model", .{arch_name});
                return error.TargetCannotLoad;
            }
            if (datatype) |d| if (dtypeIssue(.transformers, d)) |reason| {
                why.* = try std.fmt.allocPrint(alloc, "{s}: {s}", .{ @tagName(d), reason });
                return error.TargetCannotLoad;
            };
            if (container == .gguf) {
                why.* = "transformers reads safetensors, not GGUF";
                return error.TargetCannotLoad;
            }
            if (source_is_gguf and !hf_config_writable) {
                why.* = try std.fmt.allocPrint(alloc, "this tool writes config.json for {s} only; {s} is not one", .{ HfLlm.hf_config_archs_text, arch_name });
                return error.TargetCannotLoad;
            }
            return .{ .target = t, .filetype = .safetensors, .hf_names = hfNames(t, kind, arch_name, .safetensors, source_is_gguf) };
        },
        .auto, .custom => unreachable,
    }
}

/// The targets worth offering for a model of `kind`, custom last. Unknown
/// models get custom only.
pub fn targetsFor(kind: ?Kind) []const Target {
    const k = kind orelse return &.{.custom};
    return switch (k) {
        .diffusion => &.{ .comfyui, .custom },
        .llm => &.{ .comfyui, .llamacpp, .transformers, .custom },
    };
}

/// What to tell the user when `t` cannot load an output: the plan's refusal
/// `why` when there was one, else the verdict on the prepared output. Empty
/// when it loads.
pub fn problem(why: []const u8, verdicts: []const Verdict, t: Target, alloc: std.mem.Allocator) ![]const u8 {
    if (why.len > 0) return why;
    const v = targetVerdict(verdicts, t) orelse return "";
    if (v.level != .no) return "";
    return std.fmt.allocPrint(alloc, "{s} cannot load this output: {s}", .{ t.label(), v.note });
}

/// What to warn about when `t` may load an output only with help. Empty otherwise.
pub fn warning(verdicts: []const Verdict, t: Target, alloc: std.mem.Allocator) ![]const u8 {
    const v = targetVerdict(verdicts, t) orelse return "";
    if (!v.warn or v.level == .no) return "";
    return std.fmt.allocPrint(alloc, "{s} may not load this output, or may run it poorly: {s}", .{ t.label(), v.note });
}

const testing = std.testing;

fn set(ts: []const types.DataType) std.EnumSet(types.DataType) {
    var s: std.EnumSet(types.DataType) = .initEmpty();
    for (ts) |t| s.insert(t);
    return s;
}

test "comfyui target picks the container a ComfyUI loader reads" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    // With no type, safetensors for every family, and GGUF from a GGUF source.
    const sdxl = try plan(.comfyui, .diffusion, "sdxl", null, null, false, false, a, &why);
    try testing.expectEqual(types.FileType.safetensors, sdxl.filetype);
    const krea = try plan(.comfyui, .diffusion, "krea2", null, null, false, false, a, &why);
    try testing.expectEqual(types.FileType.safetensors, krea.filetype);
    try testing.expectEqual(types.FileType.gguf, (try plan(.comfyui, .diffusion, "sdxl", null, null, true, false, a, &why)).filetype);
    try testing.expectEqual(types.FileType.gguf, (try plan(.comfyui, .diffusion, "sdxl", .q4_k, null, false, false, a, &why)).filetype);
    try testing.expectEqual(types.FileType.safetensors, (try plan(.comfyui, .diffusion, "sdxl", .INT8_CONVROT, null, false, false, a, &why)).filetype);
    // Upstream ComfyUI-GGUF refuses krea2, but another GGUF extension may load it.
    try testing.expectEqual(types.FileType.gguf, (try plan(.comfyui, .diffusion, "krea2", .q4_k, null, false, false, a, &why)).filetype);
    try testing.expectEqualStrings("", why);
}

test "comfyui target names LLM tensors the way each ComfyUI loader expects" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const q3 = try plan(.comfyui, .llm, "qwen3", .q8_0, null, false, false, a, &why);
    try testing.expect(!q3.hf_names);
    const q3vl = try plan(.comfyui, .llm, "qwen3vl", .q8_0, null, false, false, a, &why);
    try testing.expect(q3vl.hf_names);
    const q2vl = try plan(.comfyui, .llm, "qwen2vl", .q8_0, null, false, false, a, &why);
    try testing.expect(!q2vl.hf_names);
    const st_ = try plan(.comfyui, .llm, "qwen3", .BF16, null, true, true, a, &why);
    try testing.expect(st_.hf_names);
    try testing.expectEqual(types.FileType.gguf, (try plan(.comfyui, .llm, "qwen2", .q4_k, null, false, false, a, &why)).filetype);
    try testing.expectEqual(types.FileType.safetensors, (try plan(.comfyui, .llm, "qwen2", null, null, false, false, a, &why)).filetype);
}

test "auto resolves by model kind and refuses nothing that loads" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    try testing.expectEqual(Target.comfyui, (try plan(.auto, .diffusion, "flux", null, null, false, false, a, &why)).target);
    const l = try plan(.auto, .llm, "qwen2", .q4_k, null, false, false, a, &why);
    try testing.expectEqual(Target.llamacpp, l.target);
    try testing.expectEqual(types.FileType.gguf, l.filetype);
}

test "llama.cpp and transformers targets refuse what they cannot read" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    try testing.expectError(error.TargetCannotLoad, plan(.llamacpp, .diffusion, "sdxl", null, null, false, false, a, &why));
    try testing.expectError(error.TargetCannotLoad, plan(.llamacpp, .llm, "qwen3", .BF16, null, false, false, a, &why));
    try testing.expectError(error.TargetCannotLoad, plan(.transformers, .llm, "qwen3", .q4_k, null, true, true, a, &why));
    try testing.expectError(error.TargetCannotLoad, plan(.transformers, .llm, "qwen3", .INT8_CONVROT, null, true, true, a, &why));
    try testing.expectError(error.TargetCannotLoad, plan(.transformers, .llm, "qwen3moe", .BF16, null, true, false, a, &why));
    const tf = try plan(.transformers, .llm, "qwen3", .BF16, null, true, true, a, &why);
    try testing.expect(tf.hf_names and tf.filetype == .safetensors);
    try testing.expect(!(try plan(.transformers, .llm, "qwen3vl", .BF16, null, false, false, a, &why)).hf_names);
}

test "a requested container decides the naming the target plans" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    // ComfyUI would pick GGUF for qwen3, but safetensors from a GGUF source takes HF names.
    try testing.expect(!(try plan(.comfyui, .llm, "qwen3", null, null, true, true, a, &why)).hf_names);
    const st_ = try plan(.comfyui, .llm, "qwen3", null, .safetensors, true, true, a, &why);
    try testing.expectEqual(types.FileType.safetensors, st_.filetype);
    try testing.expect(st_.hf_names);
    try testing.expectEqual(types.FileType.gguf, (try plan(.comfyui, .diffusion, "krea2", null, .gguf, false, false, a, &why)).filetype);
    try testing.expectError(error.TargetCannotLoad, plan(.llamacpp, .llm, "qwen3", null, .safetensors, false, false, a, &why));
    try testing.expectError(error.TargetCannotLoad, plan(.transformers, .llm, "qwen3", null, .gguf, true, true, a, &why));
}

test "targetsFor offers exactly the targets that plan a model of that kind" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    for ([_]Kind{ .diffusion, .llm }) |k| {
        for ([_]Target{ .comfyui, .llamacpp, .transformers }) |t| {
            const offered = std.mem.indexOfScalar(Target, targetsFor(k), t) != null;
            const plans = if (plan(t, k, "qwen3", null, null, false, true, a, &why)) |_| true else |_| false;
            try testing.expectEqual(plans, offered);
        }
        try testing.expectEqual(Target.custom, targetsFor(k)[targetsFor(k).len - 1]);
    }
    try testing.expectEqualSlices(Target, &.{.custom}, targetsFor(null));
}

test "dtypeIssue flags only the types a target cannot read" {
    try testing.expect(dtypeIssue(.comfyui, .ASYM_W4A8_INT8_ZP) != null);
    try testing.expect(dtypeIssue(.comfyui, .MXFP4) != null);
    try testing.expect(dtypeIssue(.comfyui, .mxfp4) != null);
    try testing.expect(dtypeIssue(.comfyui, .INT4_CONVROT) == null);
    try testing.expect(dtypeIssue(.comfyui, .q4_k) == null);
    try testing.expect(dtypeIssue(.transformers, .INT8_CONVROT) != null);
    try testing.expect(dtypeIssue(.transformers, .BF16) == null);
    try testing.expect(dtypeIssue(.llamacpp, .q2_k) == null);
    try testing.expect(dtypeIssue(.custom, .ASYM_W4A8_INT8_ZP) == null);
}

test "ComfyUI version floor follows the newest format in the file" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try assess(.{ .kind = .diffusion, .filetype = .safetensors, .arch_name = "sdxl", .dtypes = set(&.{ .BF16, .F32, .INT8_CONVROT, .INT4_CONVROT }) }, arena.allocator());
    const c = verdictFor(&v, .comfyui);
    try testing.expectEqual(Level.yes, c.level);
    try testing.expectEqualStrings("v0.28.0 or newer", c.note);
    try testing.expectEqual(Level.no, verdictFor(&v, .llamacpp).level);

    const zp = try assess(.{ .kind = .diffusion, .filetype = .safetensors, .arch_name = "sdxl", .dtypes = set(&.{ .BF16, .ASYM_W4A8_INT8_ZP }) }, arena.allocator());
    try testing.expectEqual(Level.no, verdictFor(&zp, .comfyui).level);
    const mx = try assess(.{ .kind = .diffusion, .filetype = .safetensors, .arch_name = "flux", .dtypes = set(&.{ .BF16, .MXFP4 }) }, arena.allocator());
    try testing.expectEqual(Level.no, verdictFor(&mx, .comfyui).level);
    // A GGUF type passed through into a safetensors file has no ComfyUI loader.
    const passthrough = try assess(.{ .kind = .diffusion, .filetype = .safetensors, .arch_name = "flux", .dtypes = set(&.{ .BF16, .q4_k }) }, arena.allocator());
    try testing.expectEqual(Level.no, verdictFor(&passthrough, .comfyui).level);
    const plain = try assess(.{ .kind = .diffusion, .filetype = .safetensors, .arch_name = "flux", .dtypes = set(&.{ .BF16, .F32 }) }, arena.allocator());
    try testing.expectEqualStrings("", verdictFor(&plain, .comfyui).note);
}

test "ComfyUI GGUF verdicts warn, never refuse, outside what ComfyUI-GGUF loads well" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ok = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "flux", .dtypes = set(&.{ .q4_k, .f32, .bf16 }) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&ok, .comfyui).level);
    try testing.expectEqualStrings("ComfyUI (through a GGUF loader extension such as ComfyUI-GGUF)", try worksWith(&ok, a));
    try testing.expectEqualStrings("", try warning(&ok, .comfyui, a));
    const unlisted = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "qwen_image21", .dtypes = set(&.{.q4_k}) }, a);
    try testing.expectEqual(Level.partial, verdictFor(&unlisted, .comfyui).level);
    try testing.expectEqualStrings("", try problem("", &unlisted, .comfyui, a));
    try testing.expectEqualStrings("ComfyUI may not load this output, or may run it poorly: ComfyUI-GGUF does not load qwen_image21, so it needs a GGUF loader extension that does", try warning(&unlisted, .comfyui, a));
    try testing.expectEqualStrings("ComfyUI (may not work, or work poorly: ComfyUI-GGUF does not load qwen_image21, so it needs a GGUF loader extension that does)", try worksWith(&unlisted, a));
    try testing.expectEqualStrings("", try warning(&unlisted, .custom, a));

    // No decoder, a slow CPU-only one, and both problems at once all warn.
    const mx = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "flux", .dtypes = set(&.{ .q4_k, .mxfp4 }) }, a);
    try testing.expectEqual(Level.partial, verdictFor(&mx, .comfyui).level);
    try testing.expect(std.mem.indexOf(u8, try warning(&mx, .comfyui, a), "no decoder") != null);
    const iq = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "flux", .dtypes = set(&.{ .q4_k, .iq2_xxs }) }, a);
    try testing.expectEqual(Level.partial, verdictFor(&iq, .comfyui).level);
    try testing.expect(std.mem.indexOf(u8, try warning(&iq, .comfyui, a), "CPU fallback") != null);
    const both = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "anima", .dtypes = set(&.{ .q4_k, .iq3_s }) }, a);
    const w = try warning(&both, .comfyui, a);
    try testing.expect(std.mem.indexOf(u8, w, "iq3_s") != null and std.mem.indexOf(u8, w, "anima") != null);
    try testing.expectEqualStrings("", try problem("", &both, .comfyui, a));
}

test "LLM GGUF verdicts: naming, vision tower placement, llama head counts" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const native = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "qwen3", .dtypes = set(&.{ .q4_k, .f32 }) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&native, .llamacpp).level);
    try testing.expectEqual(Level.yes, verdictFor(&native, .comfyui).level);

    const vl_split = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "qwen3vl", .mmproj = true, .has_vision = true, .dtypes = set(&.{.q8_0}) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&vl_split, .llamacpp).level);
    try testing.expectEqual(Level.partial, verdictFor(&vl_split, .comfyui).level);
    const vl_one = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "qwen3vl", .hf_names = true, .has_vision = true, .dtypes = set(&.{.q8_0}) }, a);
    try testing.expectEqual(Level.no, verdictFor(&vl_one, .llamacpp).level);
    try testing.expectEqual(Level.yes, verdictFor(&vl_one, .comfyui).level);
    const q2vl = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "qwen2vl", .mmproj = true, .has_vision = true, .dtypes = set(&.{.q8_0}) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&q2vl, .comfyui).level);

    const l8b = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "llama", .llama_heads = .{ .head_count = 32, .head_count_kv = 8 }, .dtypes = set(&.{.q4_k}) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&l8b, .comfyui).level);
    const l1b = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "llama", .llama_heads = .{ .head_count = 32, .head_count_kv = 4 }, .dtypes = set(&.{.q4_k}) }, a);
    try testing.expectEqual(Level.partial, verdictFor(&l1b, .comfyui).level);
    try testing.expect(std.mem.indexOf(u8, try warning(&l1b, .comfyui, a), "32 and 4") != null);
    try testing.expectEqual(Level.yes, verdictFor(&l1b, .llamacpp).level);

    const qwen2 = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "qwen2", .dtypes = set(&.{.q4_k}) }, a);
    try testing.expectEqual(Level.partial, verdictFor(&qwen2, .comfyui).level);
    try testing.expect((try warning(&qwen2, .comfyui, a)).len > 0);
}

test "transformers verdicts need HF names, sidecars and float weights" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const full = try assess(.{ .kind = .llm, .filetype = .safetensors, .arch_name = "qwen3", .hf_names = true, .hf_sidecars = true, .dtypes = set(&.{.BF16}) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&full, .transformers).level);
    try testing.expectEqual(Level.partial, verdictFor(&full, .comfyui).level);
    const bare = try assess(.{ .kind = .llm, .filetype = .safetensors, .arch_name = "qwen3", .hf_names = true, .dtypes = set(&.{.BF16}) }, a);
    try testing.expectEqual(Level.partial, verdictFor(&bare, .transformers).level);
    const quant = try assess(.{ .kind = .llm, .filetype = .safetensors, .arch_name = "qwen3", .hf_names = true, .hf_sidecars = true, .dtypes = set(&.{ .BF16, .INT8_CONVROT }) }, a);
    try testing.expectEqual(Level.no, verdictFor(&quant, .transformers).level);
    try testing.expectEqual(Level.partial, verdictFor(&quant, .comfyui).level);
}

test "targetVerdict is the verdict of the program the target means" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "sdxl", .dtypes = set(&.{.q4_k}) }, arena.allocator());
    try testing.expectEqual(Level.yes, targetVerdict(&v, .comfyui).?.level);
    try testing.expectEqual(Level.no, targetVerdict(&v, .llamacpp).?.level);
    try testing.expect(targetVerdict(&v, .custom) == null);

    const unlisted = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "krea2", .dtypes = set(&.{.q4_k}) }, arena.allocator());
    try testing.expect(targetVerdict(&unlisted, .comfyui).?.warn);
}

test "ComfyUI as safetensors from a GGUF refuses an LLM this tool has no HF layout for" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    try testing.expectError(error.TargetCannotLoad, plan(.comfyui, .llm, "gemma3", .BF16, null, true, false, a, &why));
    try testing.expect(std.mem.indexOf(u8, why, "gemma3") != null);
    // Its GGUF loads as it is, and a safetensors source already carries HF names.
    try testing.expectEqual(types.FileType.gguf, (try plan(.comfyui, .llm, "gemma3", null, null, true, false, a, &why)).filetype);
    try testing.expect(!(try plan(.comfyui, .llm, "gemma3", .BF16, null, false, false, a, &why)).hf_names);
}

test "the ComfyUI-GGUF text list is upstream's" {
    try testing.expect(comfyGgufLoadsText("gemma3"));
    // Patched copies list it; upstream refuses it.
    try testing.expect(!comfyGgufLoadsText("ministral3_3b"));
}

test "plain integer and f64 tensors passed through do not make an output unloadable" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const te = try assess(.{ .kind = .llm, .filetype = .safetensors, .arch_name = "qwen3", .hf_names = true, .hf_sidecars = true, .dtypes = set(&.{ .BF16, .I64 }) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&te, .transformers).level);
    const img = try assess(.{ .kind = .diffusion, .filetype = .safetensors, .arch_name = "flux", .dtypes = set(&.{ .BF16, .F64, .U32 }) }, a);
    try testing.expectEqual(Level.yes, verdictFor(&img, .comfyui).level);
    // ComfyUI-GGUF has no decoder for these, and gguf-py's fallback has none either.
    const g = try assess(.{ .kind = .diffusion, .filetype = .gguf, .arch_name = "flux", .dtypes = set(&.{ .q4_k, .i64 }) }, a);
    try testing.expect(verdictFor(&g, .comfyui).warn);
}

test "a llama GGUF with no head counts is not promised to ComfyUI-GGUF" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try assess(.{ .kind = .llm, .filetype = .gguf, .arch_name = "llama", .dtypes = set(&.{.q4_k}) }, arena.allocator());
    try testing.expectEqual(Level.partial, verdictFor(&v, .comfyui).level);
    try testing.expect(verdictFor(&v, .comfyui).warn);
}
