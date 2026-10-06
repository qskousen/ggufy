//! Convert.zig — SafeTensors → GGUF conversion logic.
//! Extracted from main.zig so the convert command has its own home.

const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const gguf = @import("Gguf.zig");
const imagearch = @import("ImageArch.zig");
const cb = @import("callbacks.zig");
const TensorClusters = @import("TensorClusters.zig");
const DataTransform = @import("DataTransform.zig");
const HfLlm = @import("HfLlm.zig");
const ImatrixMod = @import("Imatrix.zig");
const ggml = @import("ggml.h");
const build_options = @import("build_options");

// Cluster identity blobs + layout now live in TensorClusters (the shared source of truth).

// ============================================================================
// Public API
// ============================================================================

/// All options that drive a conversion, collected from CLI args.
pub const ConvertOptions = struct {
    io: std.Io,
    path: []const u8,
    filetype: types.FileType,
    datatype: ?types.DataType,
    template_path: ?[]const u8,
    output_dir: ?[]const u8,
    output_name: ?[]const u8,
    threads: usize,
    skip_sensitivity: bool,
    quantization_aggressiveness: f32,
    sensitivities_path: ?[]const u8 = null,
    /// Which quantization families are permitted when sensitivity-scaling.
    /// Non-quantized types (f16, bf16, f32, f64) and q8_0 are always allowed.
    /// Defaults to null, meaning "infer from datatype".
    allowed_quant_families: ?QuantizationFamilies = null,
    /// When true and output is safetensors, only include the main model tensors
    /// (same filtering as GGUF output) and strip name prefixes. Ignored for GGUF output.
    model_only: bool = false,
    /// When true, allow conversion even if the architecture is not recognized.
    /// Results may be suboptimal. Defaults to false.
    allow_unknown_arch: bool = false,
    /// Keep HF state-dict names in the output. Safetensors output: write the HF
    /// LLM layout, native LLM tensor names mapped back to HF state-dict names
    /// with HF shapes, and llama-family GGUF sources RoPE-unpermuted. GGUF
    /// output: leave an LLM's HF names alone instead of renaming to llama.cpp's
    /// blk.* and claiming its architecture.
    hf_names: bool = false,
    /// With `hf_names`, overwrite HF sidecars already in the output directory.
    /// Without it the conversion refuses rather than replacing them: their names
    /// are fixed, so with no `output_dir` they land on the source directory's own.
    force: bool = false,
    /// Path to a llama.cpp imatrix GGUF. When set, its per-column weights steer
    /// every ggml block quantizer's scale search. Tensors it has no entry for
    /// quantize unweighted.
    imatrix_path: ?[]const u8 = null,
    /// When set, write this string as the `general.architecture` metadata value
    /// in the GGUF output instead of the auto-detected name. Free-form; does
    /// not affect conversion behaviour. Ignored for safetensors output.
    arch_override: ?[]const u8 = null,
    /// When true, suppress the upscaling guard that fires when the target type
    /// has higher nominal precision than some source tensors (e.g. q4_k → f16).
    /// Upscaling does NOT recover lost precision; the extra bits are fill-in.
    allow_upscale: bool = false,
    /// Stochastic-rounding seed for the INT4_CONVROT_SR output type. `null` uses the
    /// baked-in default seed (TensorClusters.default_stochastic_seed); `0` disables
    /// stochastic rounding (making SR output identical to plain INT4_CONVROT — the
    /// deterministic path used for bit-for-bit comparison). Ignored by all other types.
    stochastic_rounding: ?u64 = null,
    /// Optional GUI progress/cancel hooks.  No-ops when null.
    callbacks: cb.ConvertCallbacks = .{},
};

/// Which sub-families of quantized types may be used during sensitivity-aware
/// quantization.  Non-quantized types (f16, bf16, f32, f64) and q8_0 are always
/// permitted regardless of this setting.
pub const QuantizationFamilies = struct {
    allow_0: bool = false, // qX_0
    allow_1: bool = false, // qX_1
    allow_k: bool = false, // qX_k

    /// Parse a comma-separated string of "0", "1", "k".
    /// E.g. "0,k" enables qX_0 and qX_k families.
    pub fn parse(s: []const u8) !QuantizationFamilies {
        var result = QuantizationFamilies{};
        var it = std.mem.splitScalar(u8, s, ',');
        while (it.next()) |tok| {
            const trimmed = std.mem.trim(u8, tok, " \t");
            if (std.mem.eql(u8, trimmed, "0")) {
                result.allow_0 = true;
            } else if (std.mem.eql(u8, trimmed, "1")) {
                result.allow_1 = true;
            } else if (std.mem.eql(u8, trimmed, "k")) {
                result.allow_k = true;
            } else {
                return error.InvalidQuantizationFamily;
            }
        }
        if (!result.allow_0 and !result.allow_1 and !result.allow_k) {
            return error.InvalidQuantizationFamily;
        }
        return result;
    }

    /// Derive the default families from a datatype — matches the family of
    /// the requested type so the output stays "within family" by default.
    pub fn fromDataType(dtype: types.DataType) QuantizationFamilies {
        const name = @tagName(dtype);
        return .{
            .allow_0 = std.mem.endsWith(u8, name, "_0"),
            .allow_1 = std.mem.endsWith(u8, name, "_1"),
            .allow_k = std.mem.endsWith(u8, name, "_k"),
        };
    }

    /// Returns true when the given QuantizationLevel is permitted.
    /// Non-quantized levels are always permitted.
    pub fn allows(self: QuantizationFamilies, level: QuantizationLevel) bool {
        return switch (level) {
            // Non-quantized (and q8_0) — always permitted.
            .q8_0, .f16, .bf16, .f32, .f64 => true,
            // Quantized — check family.
            .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => self.allow_k,
            .q4_0, .q5_0 => self.allow_0,
            .q4_1, .q5_1 => self.allow_1,
        };
    }
};

/// Rough precision rank — higher means more information per element.
/// Returns 255 for types that are not meaningful to compare (count, unknowns).
fn precisionRank(type_str: []const u8) u8 {
    const dt = types.DataType.fromString(type_str) catch return 255;
    return switch (dt) {
        .q1_0, .tq1_0, .iq1_s, .iq1_m => 1,
        .q2_k, .iq2_xxs, .iq2_xs, .iq2_s, .tq2_0 => 2,
        .q3_k, .iq3_xxs, .iq3_s => 3,
        .q4_0, .q4_1, .q4_k, .iq4_nl, .iq4_xs, .nvfp4, .mxfp4, .NVFP4, .MXFP4, .F4_E2M1 => 4,
        .q5_0, .q5_1, .q5_k => 5,
        .q6_k => 6,
        .q8_0, .q8_1, .q8_k, .F8_E4M3, .F8_E5M2, .SCALED_F8_E4M3, .MXFP8_E4M3 => 7,
        .f16, .F16, .bf16, .BF16, .i8, .I8, .I16, .i16, .U8, .U16 => 8,
        .f32, .F32, .i32, .I32, .U32 => 9,
        .f64, .F64, .i64, .I64, .U64 => 10,
        else => 255,
    };
}

/// Returns true when converting `source_tensors` to `target_dtype` would store
/// data at a higher nominal precision than the source — meaning the extra bits
/// are fill-in only, no information is recovered.
/// Skipped when target_dtype is null (template-based conversion).
pub fn detectUpscaling(source_tensors: []const types.Tensor, target_dtype: ?types.DataType) bool {
    const tgt = target_dtype orelse return false;
    const target_rank = precisionRank(@tagName(tgt));
    if (target_rank == 255) return false;
    for (source_tensors) |t| {
        const src_rank = precisionRank(t.type);
        if (src_rank != 255 and target_rank > src_rank) return true;
    }
    return false;
}

/// Whether `datatype` is actually storable in `filetype`.
///
/// The two type vocabularies overlap by equivalence rather than by name, so a mismatched
/// spelling is not by itself a problem: `-d f16 -f safetensors` is a legitimate request for
/// `F16` and `-d BF16 -f gguf` for `bf16`, and `DataType.forFormat` performs that mapping
/// during type assignment. What does not fit is a type with no counterpart at all — the
/// block-quantized GGUF types (q2_k…q6_k, q4_0, q8_0, …) in a SafeTensors file, or the
/// ComfyUI cluster types (SCALED_F8_E4M3, INT4_CONVROT, NVFP4, …) in a GGUF one.
///
/// Kept separate from `validateDatatypeForFiletype` (which logs) so the decision itself is
/// testable: a `std.log.err` from a passing test makes the Zig build runner report the whole
/// test binary as a failed command.
pub fn datatypeFitsFiletype(datatype: ?types.DataType, filetype: types.FileType) bool {
    // A template-driven conversion has no target datatype; per-tensor types come from the
    // template and are checked against the output format in `resolveTemplateTensorType`.
    const dt = datatype orelse return true;
    _ = dt.forFormat(filetype) catch return false;
    return true;
}

/// Refuse a target type the output container cannot hold, with a diagnostic naming the fix.
///
/// These combinations used to be accepted silently and produce a structurally invalid file:
/// `-d q4_k -f safetensors` wrote genuine q4_k blocks into a SafeTensors container under
/// `"dtype":"q4_k"`, and every reader rejects it (ComfyUI raises `KeyError`). Refusing here
/// costs only a conversion that was never going to load.
pub fn validateDatatypeForFiletype(datatype: ?types.DataType, filetype: types.FileType) !void {
    const dt = datatype orelse return;
    if (!datatypeFitsFiletype(dt, filetype)) {
        switch (filetype) {
            .safetensors => std.log.err(
                "{s} is a GGUF block-quantized type and cannot be stored in a SafeTensors file. " ++
                    "Use -f gguf to write a GGUF, or choose a SafeTensors type " ++
                    "(F16, BF16, F8_E4M3, SCALED_F8_E4M3, INT8, INT8_CONVROT, INT4_CONVROT, ASYM_W4A8_INT8, ASYM_W4A8_INT8_ZP, W6A8_INT8, MXFP4, MXFP8_E4M3, NVFP4).",
                .{@tagName(dt)},
            ),
            .gguf => std.log.err(
                "{s} has no GGUF representation and can only be written to a SafeTensors file. " ++
                    "Use -f safetensors, or choose a GGUF type (f16, bf16, f32, q8_0, q6_k, q5_k, q4_k, q3_k, q2_k, ...).",
                .{@tagName(dt)},
            ),
        }
        return error.DatatypeNotRepresentableInFiletype;
    }
}

/// The narrowest type a template assigns, by bits per weight. A template names
/// a type per tensor and so has no single target datatype, but the rest of the
/// tool already names mixed output after its floor: sensitivity routing only
/// ever moves a tensor *up* from `-d`, so `<stem>-q4_k` means "nothing here is
/// narrower than q4_k". This keeps that promise for templates.
///
/// Returns null when the template cannot be read, which leaves the caller on
/// its old fallback rather than failing a filename.
fn lowestTemplateType(opts: ConvertOptions, arena_alloc: std.mem.Allocator) ?types.DataType {
    const path = opts.template_path orelse return null;
    const file = std.Io.Dir.cwd().openFile(opts.io, path, .{ .mode = .read_only }) catch return null;
    defer file.close(opts.io);
    var buf: [8192]u8 = undefined;
    var reader = file.reader(opts.io, &buf);
    const content = reader.interface.allocRemaining(arena_alloc, .unlimited) catch return null;
    const parsed = std.json.parseFromSlice(std.json.Value, arena_alloc, content, .{}) catch return null;
    const tensors = switch (parsed.value.object.get("tensors") orelse return null) {
        .object => |o| o,
        else => return null,
    };

    var total_bits: f64 = 0;
    var total_elems: f64 = 0;
    var it = tensors.iterator();
    while (it.next()) |e| {
        const info = switch (e.value_ptr.*) {
            .object => |o| o,
            else => continue,
        };
        const ty = switch (info.get("type") orelse continue) {
            .string => |s| s,
            else => continue,
        };
        const dt = types.DataType.fromString(ty) catch continue;
        var elems: f64 = 1;
        switch (info.get("shape") orelse continue) {
            .array => |arr| for (arr.items) |d| switch (d) {
                .integer => |v| elems *= @floatFromInt(v),
                else => {},
            },
            else => continue,
        }
        total_bits += bitsPerWeight(dt) * elems;
        total_elems += elems;
    }
    if (total_elems == 0) return null;
    return nearestTypeByBpw(total_bits / total_elems);
}

/// Bits per weight, measured over a block so a block-quantized type is counted
/// including its scales rather than by its nibble width.
fn bitsPerWeight(dt: types.DataType) f64 {
    return @as(f64, @floatFromInt(dt.calcSizeInBytes(256) * 8)) / 256.0;
}

/// The standard type whose own bit rate is closest to `bpw`.
fn nearestTypeByBpw(bpw: f64) types.DataType {
    const classes = [_]types.DataType{
        .iq1_s, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s,
        .q2_k,  .q3_k,    .iq4_xs, .q4_k,  .iq4_nl,  .q4_0,
        .q5_k,  .q5_0,    .q6_k,   .q8_0,  .f16,     .bf16,
        .f32,
    };
    var best = classes[0];
    var best_d: f64 = std.math.inf(f64);
    for (classes) |c| {
        const d = @abs(bitsPerWeight(c) - bpw);
        if (d < best_d) {
            best_d = d;
            best = c;
        }
    }
    return best;
}

/// Drop a trailing `-<dtype>` from a file stem
pub fn stripDtypeSuffix(stem: []const u8) []const u8 {
    const dash = std.mem.lastIndexOfScalar(u8, stem, '-') orelse return stem;
    if (dash == 0 or dash + 1 >= stem.len) return stem;
    const suffix = stem[dash + 1 ..];
    return if (looksLikeDtype(suffix)) stem[0..dash] else stem;
}

/// Case-insensitive, because the name may not have come from here: llama.cpp
/// writes `-Q4_K_M` and `-IQ4_XS`, ggufy writes `-q4_k` and `-bf16`.
fn looksLikeDtype(suffix: []const u8) bool {
    var buf: [32]u8 = undefined;
    if (suffix.len > buf.len) return false;
    const lower = std.ascii.lowerString(buf[0..suffix.len], suffix);
    if (types.DataType.fromString(lower) != error.InvalidDataType) return true;

    const underscore = std.mem.lastIndexOfScalar(u8, lower, '_') orelse return false;
    if (underscore == 0) return false;
    const tail = lower[underscore + 1 ..];
    if (tail.len == 0 or tail.len > 2) return false;
    for (tail) |c| if (!std.ascii.isAlphabetic(c)) return false;
    return types.DataType.fromString(lower[0..underscore]) != error.InvalidDataType;
}

/// Compute the output file path that `convert` would write to, without
/// performing any actual conversion.  Useful for overwrite checks in the GUI.
pub fn computeOutputPath(opts: ConvertOptions, hf_layout: bool, arena_alloc: std.mem.Allocator) ![]const u8 {
    const dir_path = if (opts.output_dir) |od| od else std.fs.path.dirname(opts.path) orelse ".";
    const ext: []const u8 = switch (opts.filetype) {
        .gguf => "gguf",
        .safetensors => "safetensors",
    };
    const base_name = if (opts.output_name) |on| on else if (hf_layout) "model" else blk: {
        const stem = stripDtypeSuffix(std.fs.path.stem(opts.path));
        const resolved = opts.datatype orelse lowestTemplateType(opts, arena_alloc);
        const dtype_str = switch (opts.filetype) {
            .gguf => @tagName(resolved orelse types.DataType.f16),
            .safetensors => @tagName(resolved orelse types.DataType.F16),
        };
        break :blk try std.fmt.allocPrint(arena_alloc, "{s}-{s}", .{ stem, dtype_str });
    };
    return std.fs.path.join(
        arena_alloc,
        &[_][]const u8{ dir_path, try std.fmt.allocPrint(arena_alloc, "{s}.{s}", .{ base_name, ext }) },
    );
}

/// Entry point: convert a SafeTensors or GGUF file according to `opts`.
/// `f` is the already-opened source handle (either *st or *gguf.Gguf).
/// The result of running the shape/type/metadata-determining half of a conversion,
/// stopping just short of writing any output. Shared by `convert` (which then writes)
/// and `predictOutputSize` (which then measures) so both operate on identical tensor
/// types, sizes, offsets and grouping. All members are backed by `arena_alloc`.
pub const PreparedConversion = struct {
    arch: *const imagearch.Arch,
    model_tensors: std.ArrayList(types.Tensor),
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.json.ObjectMap,
    groups: TensorClusters.GroupResult,
    hf: ?HfLlm.Model = null,
    hf_renamed: bool = false,
    hf_permute: ?HfLlm.RopeHeads = null,
    hf_unpermute: ?HfLlm.RopeHeads = null,
    imatrix: ?ImatrixMod.Imatrix = null,
    vision_plan: ?HfLlm.VisionPlan = null,
    hf_vision: ?HfLlm.Vision = null,
};

/// Run everything from architecture detection through quantization-type assignment,
/// shape-fix and tensor sorting — i.e. the full conversion pipeline up to (but not
/// including) the output write. After this returns, every tensor's `type`, `size` and
/// `offset` are final, which is exactly what both writing and size prediction need.
pub fn prepareConversion(
    f: anytype,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !PreparedConversion {
    if (opts.arch_override) |name| {
        if (std.mem.trim(u8, name, " \t\r\n").len != name.len or name.len == 0) {
            std.log.warn("--arch needs an architecture name: \"{s}\" is empty or padded with whitespace", .{name});
            std.log.warn("it becomes general.architecture and the prefix of the <arch>.block_count keys llama.cpp looks up", .{});
            return error.InvalidArchOverride;
        }
    }

    // --- Detect architecture --------------------------------------------------
    const keep_hf_names = opts.filetype == .gguf and opts.hf_names;

    var hf: ?HfLlm.Model = null;
    var hf_failure: HfLlm.LoadFailure = .no_config;
    if (opts.filetype == .gguf and !keep_hf_names and @hasField(std.meta.Child(@TypeOf(f)), "current_open_path")) {
        if (hfDirCandidates(opts.io, opts.path)) |dir| {
            if (std.Io.Dir.cwd().openDir(opts.io, dir, .{})) |dh| {
                defer dh.close(opts.io);
                hf = try HfLlm.load(opts.io, dh, opts.path, arena_alloc, &hf_failure);
            } else |_| {}
        }
    }

    var expert_stacks: []const TensorClusters.ExpertStack = &.{};
    var vision_tensors: std.ArrayList(types.Tensor) = .empty;
    var vision_plan: ?HfLlm.VisionPlan = null;
    if (hf) |m| rename: {
        var forward: usize = 0;
        var reverse: usize = 0;
        for (f.tensors.items) |t| {
            if (try HfLlm.mapName(arena_alloc, m.arch, t.name, m.block_count) != null) forward += 1;
            if (try HfLlm.mapNameReverse(arena_alloc, m.arch, t.name, m.block_count) != null) reverse += 1;
        }
        if (forward == 0 and reverse == 0) {
            hf = null;
            break :rename;
        }
        var kept: std.ArrayList(types.Tensor) = .empty;
        var unmapped: std.ArrayList([]const u8) = .empty;
        var experts: std.ArrayList(types.Tensor) = .empty;
        var vision: usize = 0;
        for (f.tensors.items) |*t| {
            if (HfLlm.expertPart(t.name, m.block_count) != null) {
                try kept.append(arena_alloc, t.*);
                try experts.append(arena_alloc, t.*);
                continue;
            }
            const nn = if (forward > 0)
                try HfLlm.mapName(arena_alloc, m.arch, t.name, m.block_count)
            else if (try HfLlm.mapNameReverse(arena_alloc, m.arch, t.name, m.block_count) != null)
                t.name
            else
                null;
            if (nn) |native| {
                t.name = native;
                // HF stores the DeltaNet convolution as [channels, 1, kernel];
                // llama.cpp models it two-dimensional
                if (std.mem.endsWith(u8, native, ".ssm_conv1d.weight") and t.dims.len == 3 and t.dims[1] == 1) {
                    const squeezed = try arena_alloc.alloc(usize, 2);
                    squeezed[0] = t.dims[0];
                    squeezed[1] = t.dims[2];
                    t.dims = squeezed;
                }
                try kept.append(arena_alloc, t.*);
            } else if (HfLlm.isVisionTensor(t.name)) {
                vision += 1;
                try vision_tensors.append(arena_alloc, t.*);
            } else if (!HfLlm.isIgnoredTensor(t.name)) {
                try unmapped.append(arena_alloc, t.name);
            }
        }
        // A multimodal checkpoint's vision tower belongs in its own mmproj file
        if (vision > 0 and (m.vision == null or opts.filetype != .gguf)) {
            std.log.warn("dropping {} vision tower tensors; this writes the text model only", .{vision});
            if (m.vision == null and opts.filetype == .gguf) {
                std.log.warn("config.json has no vision_config this tool can turn into an mmproj file", .{});
            }
            vision_tensors.clearRetainingCapacity();
        }

        if (forward > 0) f.native_names_applied = true;

        if (unmapped.items.len > 0) {
            const joined = try std.mem.join(arena_alloc, ", ", unmapped.items);
            std.log.warn("{} tensors have no {s} name in llama.cpp's vocabulary: {s}", .{ unmapped.items.len, m.arch.hf_model_type, joined });
            if (!opts.allow_unknown_arch) {
                std.log.warn("converting would write a model missing them; pass --allow-unknown-arch to drop them anyway", .{});
                return error.UnmappedTensors;
            }
            std.log.warn("dropping them: --allow-unknown-arch was given", .{});
        }
        expert_stacks = try HfLlm.planExpertStacks(arena_alloc, experts.items, m.expert_count, m.block_count);
        vision_plan = try planMmproj(m, vision_tensors.items, opts, arena_alloc);
        f.tensors = kept;
    }

    if (hf) |*m| if (m.tokens.len == 0) {
        warnVocabless(m.*, opts.path);
    } else {
        if (!m.is_spm and m.tokenizer_pre.len == 0) {
            std.log.warn("the pre-tokenizer in {s}'s tokenizer.json in unknown", .{opts.path});
            if (!opts.allow_unknown_arch) {
                std.log.warn("converting would tag the GGUF with splitting the model does not use; pass --allow-unknown-arch to write llama.cpp's \"default\" tag anyway", .{});
                return error.UnknownPretokenizer;
            }
            std.log.warn("tagging it \"default\": --allow-unknown-arch was given, and llama.cpp will tokenize it as gpt2", .{});
            m.tokenizer_pre = "default";
        }
        // Unlike the tag above there is no fallback to offer
        if (!m.is_spm and m.merges.len == 0) {
            std.log.warn("{s} has a BPE vocabulary with no merge table: tokenizer.json lists no usable merges and there is no merges.txt beside it", .{opts.path});
            return error.MissingMerges;
        }
    };

    const detected = blk: {
        if (hf) |m| break :blk m.arch;
        // A vision-language text GGUF's names are its text tower's, so for -H
        // the file's own architecture picks the HF class to rebuild.
        if (opts.filetype == .safetensors and opts.hf_names) if (sourceArchName(f)) |sa| {
            if (HfLlm.visionArchByGgufName(sa)) |vl| break :blk vl;
        };
        const result = imagearch.detectArchFromTensorsOrError(f.tensors.items, allocator);
        if (result) |a| {
            break :blk a;
        } else |err| {
            if (err == error.UnknownArchitecture and opts.allow_unknown_arch) {
                std.log.warn("Unknown architecture; proceeding anyway. Results may be suboptimal.", .{});
                break :blk &imagearch.generic_arch;
            }
            return err;
        }
    };

    const bare_llm = blk: {
        if (hf != null or keep_hf_names or opts.filetype != .gguf) break :blk false;
        if (detected.hf_model_type.len == 0) break :blk false;
        for (f.tensors.items) |t| {
            if (try HfLlm.mapName(arena_alloc, detected, t.name, null) != null) break :blk true;
        }
        break :blk false;
    };
    if (bare_llm) {
        // Which sidecar to go fix differs per failure, so name it.
        switch (hf_failure) {
            .no_config => std.log.warn("{s} carries HF tensor names, but there is no config.json beside it", .{opts.path}),
            .unknown_model_type => std.log.warn("{s} carries HF tensor names, but the config.json beside it names a model type this tool does not map", .{opts.path}),
            .incomplete_config => std.log.warn("{s} carries HF tensor names, but the config.json beside it is missing fields the GGUF metadata needs (hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, vocab_size, rms_norm_eps, max_position_embeddings)", .{opts.path}),
            .no_tokenizer => {
                std.log.warn("{s} carries HF tensor names, but its tokenizer could not be read: tokenizer.json is missing (beside it and in a diffusers tokenizer/ sibling), malformed, or of a model type this tool does not map", .{opts.path});
                std.log.warn("a GGUF without vocabulary metadata does not load; fix or supply tokenizer.json", .{});
                return error.HfTokenizerUnreadable;
            },
        }
        std.log.warn("a GGUF with HF names and no architecture may fail to load", .{});
        std.log.warn("put the checkpoint's config.json beside the weights (and its tokenizer, for llama.cpp) and convert again", .{});
        std.log.warn("or, pass --hf-names to write the HF-named file anyway, for a reader of your own", .{});
        return error.HfConfigUnusable;
    }

    const arch = blk: {
        if (detected.hf_model_type.len == 0 or !keep_hf_names) break :blk detected;
        if (comfyLoadsTextArch(detected.name) and !std.mem.eql(u8, detected.name, "llama")) {
            std.log.info("--hf-names: keeping HF tensor names under architecture {s}, which ComfyUI-GGUF's text-encoder loader takes as they are", .{detected.name});
            break :blk detected;
        }
        std.log.info("--hf-names: {s} detected, but keeping HF tensor names and writing no architecture", .{detected.name});
        const anon = try arena_alloc.create(imagearch.Arch);
        anon.* = detected.*;
        anon.name = imagearch.generic_arch.name;
        anon.gguf_arch_id = false;
        anon.hf_model_type = "";
        break :blk @as(*const imagearch.Arch, anon);
    };
    // ComfyUI keys its Qwen3-VL detection on the prefixed names
    if (keep_hf_names) {
        if (HfLlm.visionLayout(detected.name)) |layout| if (std.mem.eql(u8, layout.prefix, "model.visual.")) {
            var prefixed: usize = 0;
            for (f.tensors.items) |*t| {
                if (std.mem.startsWith(u8, t.name, "language_model.") or std.mem.startsWith(u8, t.name, "visual.")) {
                    t.name = try std.fmt.allocPrint(arena_alloc, "model.{s}", .{t.name});
                    prefixed += 1;
                }
            }
            if (prefixed > 0) {
                std.log.info("--hf-names: prefixed {d} bare-model names with \"model.\"", .{prefixed});
            }
        };
    }
    const threshold = arch.threshhold orelse QUANTIZATION_THRESHOLD;
    std.log.info("Detected architecture: {s}", .{arch.name});

    // HF input (HF safetensors -> GGUF)
    var hf_permute: ?HfLlm.RopeHeads = null;
    if (hf) |m| {
        if (f.native_names_applied and std.mem.eql(u8, arch.name, "llama")) {
            hf_permute = .{ .head_count = m.head_count, .head_count_kv = m.head_count_kv };
        }
    }

    // HF-name output (GGUF -> HF safetensors)
    // Rewrite source tensors to HF names and HF dim order
    var hf_unpermute: ?HfLlm.RopeHeads = null;
    var hf_renamed = false;
    var hf_vision: ?HfLlm.Vision = null;
    if (opts.filetype == .safetensors and opts.hf_names) {
        if (arch.hf_model_type.len == 0) {
            std.log.err("Unknown model type for HF naming: detected {s}", .{arch.name});
            return error.HfNamesUnsupported;
        }

        var vision_part: std.ArrayList(types.Tensor) = .empty;
        if (HfLlm.visionLayout(arch.name)) |layout| {
            if (@hasField(std.meta.Child(@TypeOf(f)), "companion")) {
                if (f.companion == null) if (try findMmproj(opts.io, opts.path, arena_alloc)) |p| {
                    std.log.info("reading vision tower from {s}", .{p});
                    try f.attachCompanion(p);
                };
                if (f.companion) |c| {
                    var text: std.ArrayList(types.Tensor) = .empty;
                    for (f.tensors.items) |t| {
                        const theirs = if (t.source_path) |sp| std.mem.eql(u8, sp, c.path) else false;
                        try (if (theirs) &vision_part else &text).append(arena_alloc, t);
                    }
                    f.tensors = text;
                    const v = try HfLlm.visionFromClip(c.metadata, vision_part.items, arena_alloc) orelse {
                        std.log.warn("{s} is not a recognized vision projector (clip.projector_type {s})", .{ c.path, HfLlm.metaStr(c.metadata, "clip.projector_type") orelse "missing" });
                        return error.MmprojUnusable;
                    };
                    if (v.projector != layout.projector) {
                        std.log.warn("{s} is a {s} projector, but {s} is a {s} model", .{ c.path, v.projector.clipName(), opts.path, arch.name });
                        return error.MmprojUnusable;
                    }
                    hf_vision = v;
                }
            }
            if (hf_vision == null) {
                const want = try mmprojPath(opts.path, arena_alloc);
                if (!std.mem.eql(u8, arch.name, "qwen35")) {
                    std.log.warn("{s} is the text half of a {s} model; the vision half is required but not found", .{ opts.path, arch.hf_model_type });
                    std.log.warn("put its mmproj beside it as {s} (or as .gguf whose name holds \"mmproj\" and this file's name) and convert again", .{want});
                    return error.MmprojMissing;
                }
                std.log.info("no mmproj beside {s} (looked for {s}); the directory holds the text model only", .{ opts.path, want });
            }
        }

        var rewrote = false;
        var already = false;
        var hf_source = false;
        var unmapped: std.ArrayList([]const u8) = .empty;
        var ignored: usize = 0;

        const mtp_base = mtpBaseFromSource(f, sourceArchName(f) orelse arch.name, arena_alloc);
        for (f.tensors.items) |*t| {
            if (try HfLlm.mapNameReverse(arena_alloc, arch, t.name, mtp_base)) |nn| {
                t.name = nn;
                rewrote = true;
            } else if (try HfLlm.mapName(arena_alloc, arch, t.name, null) != null) {
                if (f.hf_names_applied) already = true else hf_source = true;
            } else if (HfLlm.isIgnoredNativeTensor(t.name)) {
                ignored += 1;
            } else {
                try unmapped.append(arena_alloc, t.name);
            }
        }

        if (rewrote) f.hf_names_applied = true;
        hf_renamed = rewrote or already;

        if (unmapped.items.len > 0 and (hf_renamed or !hf_source)) {
            const joined = try std.mem.join(arena_alloc, ", ", unmapped.items);
            std.log.warn("{} tensors have no {s} name in transformers' vocabulary: {s}", .{ unmapped.items.len, arch.hf_model_type, joined });
            if (!opts.allow_unknown_arch) {
                std.log.warn("converted model would have a directory missing them. Pass --allow-unknown-arch to convert anyway", .{});
                return error.UnmappedTensors;
            }
            std.log.warn("dropping: --allow-unknown-arch was given", .{});
        }
        if (hf_renamed and (unmapped.items.len > 0 or ignored > 0)) {
            var kept: std.ArrayList(types.Tensor) = .empty;
            for (f.tensors.items) |t| {
                if (try HfLlm.mapName(arena_alloc, arch, t.name, null) != null) try kept.append(arena_alloc, t);
            }
            f.tensors = kept;
        }
        if (hf_source and !hf_renamed) {
            std.log.info("--hf-names: the source already carries HF names; writing them through as they are", .{});
        }
        if (hf_renamed and std.mem.eql(u8, arch.name, "llama")) {
            const src_arch = sourceArchName(f) orelse {
                std.log.err("--hf-names needs the source's general.architecture to know whether its Q/K rows were permuted", .{});
                return error.HfNamesUnsupported;
            };
            if (!std.mem.eql(u8, src_arch, "llama")) {
                std.log.err("--hf-names maps llama-shaped GGUFs only where general.architecture is llama; this is {s}", .{src_arch});
                return error.HfNamesUnsupported;
            }
            hf_unpermute = ropeHeadsFromSource(f, src_arch, arena_alloc) catch |e| {
                std.log.err("--hf-names needs {s}.attention.head_count and head_count_kv in the source metadata to un-permute the Q/K rows", .{src_arch});
                return e;
            };
        }
        if (hf_vision) |v| {
            const plan = try HfLlm.planVisionReverse(arena_alloc, v, HfLlm.visionLayout(arch.name).?.prefix, vision_part.items);
            if (plan.unmapped.len > 0) {
                const joined = try std.mem.join(arena_alloc, ", ", plan.unmapped);
                std.log.warn("{} mmproj tensors have no {s} name in transformers' vocabulary: {s}", .{ plan.unmapped.len, arch.hf_model_type, joined });
                if (!opts.allow_unknown_arch) {
                    std.log.warn("converting would write a directory missing them; pass --allow-unknown-arch to convert anyway", .{});
                    return error.UnmappedTensors;
                }
                std.log.warn("dropping: --allow-unknown-arch was given", .{});
            }
            try f.tensors.appendSlice(arena_alloc, plan.sources);
            expert_stacks = plan.stacks;
        }
    }

    // --- Filter and normalise tensor list ------------------------------------
    var model_tensors = try filterAndStripTensors(f, arch, opts.filetype, opts.model_only, arena_alloc);

    // --- Undo any prior GGUF shape_fix ----------------------------------------
    // When the source is a GGUF written by this tool, restore each tensor's
    // logical shape from its "comfy.gguf.orig_shape.<name>" metadata. This makes
    // round-trips faithful (gguf -> safetensors) and lets shape_fix re-apply
    // cleanly for gguf -> gguf. No-op for sources without the metadata.
    try restoreOrigShapes(&model_tensors, f.getSourceMetadata(), arena_alloc);

    // --- Group and collapse NVFP4/FP8 clusters --------------------------------
    var groups = try TensorClusters.groupClusters(f, arena_alloc, allocator);
    groups.expert_stacks = expert_stacks;
    try TensorClusters.collapseModelTensors(&model_tensors, &groups, .dequant, arena_alloc);

    // --- Assign quantization types (template or auto) -------------------------
    var template_metadata: ?std.json.ObjectMap = null;
    if (opts.template_path) |tp| {
        template_metadata = try applyTemplate(opts.io, tp, &model_tensors, opts.filetype, arena_alloc);
    } else {
        try assignQuantTypes(&model_tensors, arch, threshold, opts, arena_alloc);
    }

    // --- Importance weights, and the check that the assignment is runnable ----
    // Before the shape fix and before any writer opens a file: a type whose
    // encoder needs weights it will not get must fail now, not partway through.
    var imatrix = try loadImatrix(opts, allocator);
    errdefer if (imatrix) |*im| im.deinit();
    try checkImatrixCoverage(model_tensors.items, if (imatrix) |*im| im else null, arena_alloc);

    // --- Shape fix ------------------------------------------------------------
    var extra_metadata: std.json.ObjectMap = .empty;
    if (arch.shape_fix and opts.filetype == .gguf) {
        try applyShapeFix(&model_tensors, &extra_metadata, arena_alloc);
    }

    // --- Sort tensors alphabetically -----------------------------------------
    if (opts.filetype == .gguf) {
        std.sort.block(types.Tensor, model_tensors.items, {}, struct {
            fn lessThan(_: void, a: types.Tensor, b: types.Tensor) bool {
                return std.mem.lessThan(u8, a.name, b.name);
            }
        }.lessThan);
    }

    // --- Assign final on-disk sizes/offsets (both template and auto paths) ----
    // Must run after any reordering so offsets describe the final tensor order.
    try assignOutputLayout(&model_tensors, opts, arena_alloc);

    return .{
        .arch = arch,
        .model_tensors = model_tensors,
        .template_metadata = template_metadata,
        .extra_metadata = extra_metadata,
        .groups = groups,
        .hf = hf,
        .hf_renamed = hf_renamed,
        .hf_permute = hf_permute,
        .hf_unpermute = hf_unpermute,
        .imatrix = imatrix,
        .vision_plan = vision_plan,
        .hf_vision = hf_vision,
    };
}

pub fn convert(
    f: anytype,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !void {
    // --- Upscaling guard ------------------------------------------------------
    if (!opts.allow_upscale and detectUpscaling(f.tensors.items, opts.datatype)) {
        std.log.err(
            "Source contains lossy-quantized tensors; converting to a higher-precision " ++
                "format will NOT recover lost information — the extra bits are fill-in only. " ++
                "Pass --allow-upscale (-U) to convert anyway.",
            .{},
        );
        return error.UpscalingNotAllowed;
    }

    if (opts.datatype == .ASYM_W4A8_INT8_ZP) {
        std.log.warn(
            "ASYM_W4A8_INT8_ZP: current ComfyUI drops weight_correction when loading, and decodes these layers wrong. comfy_kitchen and ggufy read them correctly.",
            .{},
        );
    }

    var prep = try prepareConversion(f, opts, allocator, arena_alloc);

    // --- Write output ---------------------------------------------------------
    switch (opts.filetype) {
        .gguf => try writeGguf(
            f,
            prep.model_tensors,
            prep.arch,
            prep.template_metadata,
            prep.extra_metadata,
            opts,
            allocator,
            arena_alloc,
            &prep.groups,
            prep.hf,
            prep.hf_permute,
            if (prep.imatrix) |*im| im else null,
        ),
        .safetensors => try writeSafetensors(
            f,
            prep.model_tensors,
            prep.template_metadata,
            prep.extra_metadata,
            opts,
            allocator,
            arena_alloc,
            &prep.groups,
            prep.hf_renamed,
            prep.hf_unpermute,
            prep.hf_vision,
        ),
    }

    if (prep.vision_plan) |plan| try writeMmproj(f, plan, prep.hf.?, opts, allocator, arena_alloc);
}

/// Compute the exact size in bytes of the file `convert` would produce with these
/// options, WITHOUT reading tensor data or writing anything. Reuses the same
/// pipeline (`prepareConversion`) and the same serializers the writers use, so the
/// prediction matches the real output byte-for-byte.
pub fn predictOutputSize(
    f: anytype,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !u64 {
    const prep = try prepareConversion(f, opts, allocator, arena_alloc);

    switch (opts.filetype) {
        .gguf => {
            var metadata: std.json.ObjectMap = .empty;
            try buildGgufMetadata(&metadata, f, prep.arch, prep.template_metadata, prep.extra_metadata, opts, arena_alloc, prep.hf);
            var size = try gguf.calculateFileSize(prep.model_tensors.items, &metadata, 32);
            // The mmproj is part of the output, so the estimate counts both files.
            if (prep.vision_plan) |plan| {
                var mm = try mmprojOutput(plan, prep.hf.?, opts, arena_alloc);
                const mm_size = try gguf.calculateFileSize(mm.tensors.items, &mm.metadata, 32);
                std.log.info("text GGUF {d} bytes, vision tower {s} {d} bytes", .{ size, std.fs.path.basename(mm.path), mm_size });
                size += mm_size;
            }
            return size;
        },
        .safetensors => {
            var metadata: ?std.json.ObjectMap = null;
            try buildSafetensorsMetadata(&metadata, f, prep.template_metadata, prep.extra_metadata, prep.hf_renamed, arena_alloc);
            return try st.calculateFileSize(prep.model_tensors.items, metadata, arena_alloc, allocator);
        },
    }
}

// ============================================================================
// Quantization level helpers
// ============================================================================

// Tensors with fewer than this many elements are never block-quantized; they
// are kept in a float type instead. This matches the ComfyUI-GGUF (city96)
// convention: small tensors (modulation tables, projectors, etc.) are loaded by
// ComfyUI as raw float parameters, so a block-quantized version fails to load.
// Together with the "never quantize 1D tensors" rule in assignTensorType, this
// generalizes what would otherwise be a per-architecture hi-precision list.
pub const QUANTIZATION_THRESHOLD: u64 = 256 * 256;

/// Quantization type hierarchy from lowest to highest precision.
pub const QuantizationLevel = enum(u8) {
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
        if (s.len > lower.len) return error.UnknownQuantizationType;
        return std.meta.stringToEnum(QuantizationLevel, std.ascii.lowerString(lower[0..s.len], s)) orelse error.UnknownQuantizationType;
    }
};

fn cheapestKQuantAtLeast(want: u8) ?types.DataType {
    const rungs = [_]types.DataType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_0, .f16, .f32 };
    for (rungs) |r| {
        if (r.nominalBits() >= want) return r;
    }
    return null;
}

/// Lift `target` to the cheapest type carrying at least `min` bits per weight,
/// staying inside `target`'s own format family. Returns `target` unchanged when it
/// already clears the floor, and null when the family has no rung that does - the
/// caller then leaves the tensor at its source precision.

pub fn liftToFloor(target: types.DataType, min: imagearch.Precision) ?types.DataType {
    const want = @intFromEnum(min);
    if (target.nominalBits() >= want) return target;

    return switch (target) {
        // Rotated-int cluster line: the 8-bit rung keeps the Hadamard rotation and
        // the per-row scale, so only the element width changes.
        .INT4_CONVROT, .INT4_CONVROT_SR, .W6A8_INT8 => if (want <= 8) .INT8_CONVROT else null,
        // W6A8 is W4A8's own layout and kernel at 6 bits, so it is the nearer rung.
        .ASYM_W4A8_INT8, .ASYM_W4A8_INT8_ZP => if (want <= 6) .W6A8_INT8 else if (want <= 8) .INT8_CONVROT else null,
        // Block-scaled FP line. Both lift to MXFP8 rather than to SCALED_F8_E4M3,
        // which is also 8-bit and slightly smaller: scaled-fp8 carries a single F32
        // for the whole tensor, so it would buy four bits of element precision while
        // throwing away the per-block scaling. On a tensor that is floored precisely
        // because its activations have outliers, that is the wrong half to give up -
        // one tensor-wide scale is set by the outliers and everything under them
        // loses resolution. MXFP8 keeps a scale per 32 elements.
        .NVFP4, .MXFP4 => if (want <= 8) .MXFP8_E4M3 else null,
        // GGUF: walk the existing level ordering upward, keeping to the families the
        // requested type already implies (q8_0 and the float types are always in).
        else => blk: {
            if (target.formatType() != .gguf) break :blk null;
            const families = QuantizationFamilies.fromDataType(target);
            const start = QuantizationLevel.fromString(@tagName(target)) catch {
                // IQ types are not rungs here (the ladder also drives
                // sensitivity routing) and top out at iq4_xs, so a floor above
                // that comes off the k-quant line.
                break :blk cheapestKQuantAtLeast(want);
            };
            var i: u8 = @intFromEnum(start) + 1;
            while (i <= @intFromEnum(QuantizationLevel.f64)) : (i += 1) {
                const cand: QuantizationLevel = @enumFromInt(i);
                if (!families.allows(cand)) continue;
                const dt = types.DataType.fromString(@tagName(cand)) catch continue;
                if (dt.nominalBits() >= want) break :blk dt;
            }
            break :blk null;
        },
    };
}

/// Calculate an appropriate quantization level for one tensor given its
/// sensitivity score and the user's aggressiveness setting.
///
/// Builds a filtered list of only the permitted levels (from target up to
/// source precision), then interpolates directly within that list so the
/// full sensitivity range maps evenly across all allowed levels.
///
/// - `sensitivity`:    1–100, 1 = least sensitive, 100 = most sensitive.
/// - `aggressiveness`: 1–100, higher = more aggressive (stays near target).
/// - `target_level`:   the base type requested by the user (e.g. q2_k).
/// - `source_type`:    the tensor's current type string (e.g. "f16").
/// - `families`:       which quantized sub-families are allowed.
pub fn calculateQuantizationLevel(
    sensitivity: f32,
    aggressiveness: f32,
    target_level: QuantizationLevel,
    source_type: []const u8,
    families: QuantizationFamilies,
) !QuantizationLevel {
    const sens = std.math.clamp(sensitivity, 1.0, 100.0);
    const hard = std.math.clamp(aggressiveness, 1.0, 100.0);

    // Build the filtered level list: all allowed levels from target up to source.
    // Maximum possible entries is the number of enum variants (14).
    var allowed_buf: [14]QuantizationLevel = undefined;
    var allowed_len: usize = 0;

    const source_level = try QuantizationLevel.fromString(source_type);
    const source_idx: u8 = @intFromEnum(source_level);

    // When source precision is strictly above f16 (i.e. bf16/f32/f64), selecting
    // f16 as a "lossless" fallback wastes no space but silently loses precision
    // compared to bf16.  Exclude it from the candidate set in that case so the
    // sensitivity scaling skips straight to bf16.
    const skip_f16 = source_idx > @intFromEnum(QuantizationLevel.f16);

    var i: u8 = @intFromEnum(target_level);
    while (i <= source_idx) : (i += 1) {
        if (skip_f16 and i == @intFromEnum(QuantizationLevel.f16)) continue;
        const candidate: QuantizationLevel = @enumFromInt(i);
        if (families.allows(candidate)) {
            allowed_buf[allowed_len] = candidate;
            allowed_len += 1;
        }
    }

    // If nothing matched (shouldn't happen given q8_0/f-types are always allowed),
    // fall back to source precision.
    if (allowed_len == 0) return source_level;

    // Interpolate within the filtered list.
    // norm_sens 0.0 → allowed_buf[0] (target/lowest), 1.0 → allowed_buf[last] (highest).
    const norm_sens = (sens - 1.0) / 99.0;
    const hardness_factor = hard / 100.0;
    const exponent = 0.5 + (hardness_factor * 3.0);
    const adjusted_sens = std.math.pow(f32, norm_sens, exponent);

    const max_idx: f32 = @floatFromInt(allowed_len - 1);
    const raw = adjusted_sens * max_idx;
    const picked: usize = @intFromFloat(@round(raw));
    return allowed_buf[@min(picked, allowed_len - 1)];
}

// ============================================================================
// Step 1 — Filter tensors and strip name prefixes
// ============================================================================

/// The directory whose config.json describes these weights: the parent of a
/// model.safetensors / index.json path, otherwise the path itself when it is a
/// directory. load() still has to find a readable config.json for HF mode.
/// The architectures ComfyUI-GGUF's text-encoder loader accepts (TXT_ARCH_LIST
/// in its loader.py). It refuses any other general.architecture outright.
const comfy_gguf_text_archs = [_][]const u8{ "t5", "t5encoder", "llama", "qwen2vl", "qwen3", "qwen3vl", "gemma3", "ministral3_3b" };

fn comfyLoadsTextArch(name: []const u8) bool {
    for (comfy_gguf_text_archs) |n| if (std.mem.eql(u8, n, name)) return true;
    return false;
}

fn warnVocabless(m: HfLlm.Model, path: []const u8) void {
    const a = m.arch.name;
    std.log.warn("{s} has a config.json but no tokenizer, beside it or in a diffusers tokenizer/ sibling", .{path});
    std.log.warn("writing {s} names and metadata without tokenizer.ggml.* keys: llama.cpp will refuse the file for want of a vocabulary", .{a});
    if (!comfyLoadsTextArch(a)) {
        std.log.warn("ComfyUI-GGUF will refuse it too: its text-encoder loader does not list {s}", .{a});
    } else if (std.mem.eql(u8, a, "llama") and (m.head_count != 32 or m.head_count_kv != 8)) {
        // Its llama branch un-permutes Q/K with hardcoded head counts.
        std.log.warn("ComfyUI-GGUF un-permutes llama Q/K rows assuming 32 heads and 8 KV heads; this model has {d} and {d}, so it will load with scrambled attention", .{ m.head_count, m.head_count_kv });
    } else {
        std.log.warn("ComfyUI-GGUF loads it as a text encoder", .{});
    }
}

fn hfDirCandidates(io: std.Io, path: []const u8) ?[]const u8 {
    var dir = path;
    while (dir.len > 0 and dir[dir.len - 1] == '/') dir = dir[0 .. dir.len - 1];
    const base = std.fs.path.basename(dir);
    if (std.mem.endsWith(u8, base, ".safetensors") or std.mem.endsWith(u8, base, ".json")) {
        dir = std.fs.path.dirname(dir) orelse ".";
    } else {
        const stat = std.Io.Dir.cwd().statFile(io, dir, .{}) catch return null;
        if (stat.kind != .directory) return null;
    }
    return dir;
}

/// Returns a new list containing only the tensors that should be converted,
/// with name prefixes stripped for GGUF output and model-only safetensors output.
///
/// For full safetensors output (the default), all tensors are included with
/// their original names so VAE, text encoders, and other components in a
/// bundled checkpoint are preserved in the output file.
fn filterAndStripTensors(
    f: anytype,
    arch: *const imagearch.Arch,
    output_filetype: types.FileType,
    model_only: bool,
    arena_alloc: std.mem.Allocator,
) !std.ArrayList(types.Tensor) {
    var model_tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, f.tensors.items.len);

    if (output_filetype == .safetensors and !model_only) {
        // Preserve all tensors and their original names so bundled checkpoints
        // (VAE, text encoders, etc.) round-trip intact.
        for (f.tensors.items) |t| {
            if (!arch.shouldIgnore(t.name)) {
                try model_tensors.append(arena_alloc, try t.dupe(arena_alloc));
            }
        }
        return model_tensors;
    }

    // GGUF output or model-only safetensors: filter to main model tensors and strip name prefixes.

    // Arches that already ship just the model keep every tensor under its raw
    // name; the prefix heuristic below would drop lm_head.weight and rename layers.
    if (!arch.component_filter) {
        for (f.tensors.items) |t| {
            if (!arch.shouldIgnore(t.name)) {
                try model_tensors.append(arena_alloc, try t.dupe(arena_alloc));
            }
        }
        return model_tensors;
    }

    // Full checkpoints mix "model." (unet) tensors with VAE/text encoder tensors.
    // UNet-only files have no such prefix — we need to know which case we're in.
    var has_model_prefix = false;
    for (f.tensors.items) |t| {
        if (std.mem.startsWith(u8, t.name, "model.")) {
            has_model_prefix = true;
            break;
        }
    }

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

    // Strip "model.diffusion_model." etc. from names.
    for (model_tensors.items) |*t| {
        t.name = try arena_alloc.dupe(u8, imagearch.stripPrefix(t.name));
    }

    return model_tensors;
}

// ============================================================================
// Step 2a — Apply a JSON template
// ============================================================================

/// Filters `model_tensors` down to only the tensors listed in the template,
/// applying the shapes and types specified there. Returns any template-level
/// metadata found under the "metadata" key.
fn applyTemplate(
    io: std.Io,
    template_path: []const u8,
    model_tensors: *std.ArrayList(types.Tensor),
    output_filetype: types.FileType,
    arena_alloc: std.mem.Allocator,
) !?std.json.ObjectMap {
    std.log.info("Using template {s}", .{template_path});

    const t_file = try std.Io.Dir.cwd().openFile(io, template_path, .{ .mode = .read_only });
    defer t_file.close(io);
    var t_reader_buf: [8192]u8 = undefined;
    var t_reader = t_file.reader(io, &t_reader_buf);
    const t_content = try t_reader.interface.allocRemaining(arena_alloc, .unlimited);
    const t_json = try std.json.parseFromSlice(std.json.Value, arena_alloc, t_content, .{});

    var template_metadata: ?std.json.ObjectMap = null;
    if (t_json.value.object.get("metadata")) |m| {
        template_metadata = m.object;
    }

    const t_tensors = t_json.value.object.get("tensors") orelse return error.InvalidTemplate;
    var filtered = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, model_tensors.items.len);

    var it = t_tensors.object.iterator();
    while (it.next()) |entry| {
        const target_name = entry.key_ptr.*;
        const target_info = entry.value_ptr.object;

        const source_tensor = findSourceTensor(model_tensors.items, target_name);

        if (source_tensor) |src| {
            const new_t = try applyTemplateEntry(src, target_name, target_info, output_filetype, arena_alloc);
            try filtered.append(arena_alloc, new_t);
            std.log.info("Matched target tensor {s} to source tensor {s}, setting to type {s}", .{ target_name, src.name, new_t.type });
        } else {
            std.log.warn("Warning: Template tensor {s} not found in source file.", .{target_name});
        }
    }

    model_tensors.* = filtered;
    return template_metadata;
}

/// Fuzzy-match a target name against the source tensor list.
/// Accepts exact matches or suffix matches separated by '.'.
fn findSourceTensor(tensors: []const types.Tensor, target_name: []const u8) ?types.Tensor {
    for (tensors) |t| {
        if (std.mem.eql(u8, t.name, target_name)) return t;
        if (t.name.len > target_name.len and
            t.name[t.name.len - target_name.len - 1] == '.' and
            std.mem.endsWith(u8, t.name, target_name)) return t;
    }
    return null;
}

/// Build a new Tensor for a single template entry, validating shapes and types.
fn applyTemplateEntry(
    src: types.Tensor,
    target_name: []const u8,
    target_info: std.json.ObjectMap,
    output_filetype: types.FileType,
    arena_alloc: std.mem.Allocator,
) !types.Tensor {
    const target_shape_arr = target_info.get("shape").?.array;
    const target_dims = try arena_alloc.alloc(usize, target_shape_arr.items.len);
    var target_elements: u64 = 1;
    for (target_shape_arr.items, 0..) |item, i| {
        // Templates from GGUF have reversed dimensions — flip them back.
        target_dims[target_shape_arr.items.len - 1 - i] = @intCast(item.integer);
        target_elements *= @intCast(item.integer);
    }

    const target_type = target_info.get("type").?.string;

    var source_elements: u64 = 1;
    for (src.dims) |d| source_elements *= d;

    if (source_elements != target_elements) {
        std.log.err("Tensor {s} shape mismatch. Source elements: {}, Target elements: {}", .{ target_name, source_elements, target_elements });
        return error.ShapeMismatch;
    }

    const raw_type = try types.DataType.fromString(target_type);
    // Normalize to the canonical type for the output format (e.g. bf16 → BF16 for safetensors).
    const data_type = try raw_type.forFormat(output_filetype);

    // Block-size validation only applies to quantized GGUF types.
    if (data_type.formatType() == .gguf) {
        const ggml_type = try gguf.GgmlType.fromString(@tagName(data_type));
        const bs = ggml_type.getBlockSize();
        if (bs > 1 and source_elements % bs != 0) {
            std.log.err("Tensor {s} cannot be quantized to type {s}. Element count {} is not a multiple of block size {}", .{ target_name, @tagName(data_type), source_elements, bs });
            return error.InvalidSizeForQuantization;
        }
    }

    var new_t = src;
    new_t.name = target_name;
    new_t.dims = target_dims;
    new_t.type = @tagName(data_type);
    new_t.size = data_type.calcSizeInBytes(target_elements);
    return new_t;
}

// ============================================================================
// Step 2b — Auto-assign quantization types
// ============================================================================

/// Iterates over all tensors and assigns GGUF types based on the target
/// datatype, sensitivities file (if any), and architecture-specific rules.
/// Also computes offsets and prints per-tensor progress.
fn assignQuantTypes(
    model_tensors: *std.ArrayList(types.Tensor),
    arch: *const imagearch.Arch,
    threshold: u64,
    opts: ConvertOptions,
    arena_alloc: std.mem.Allocator,
) !void {
    // Load sensitivities if available and not skipped.
    var use_sensitivity = false;
    var sensitivities: std.json.Parsed(std.json.Value) = undefined;

    if (!opts.skip_sensitivity and opts.filetype == .gguf) {
        if (opts.sensitivities_path) |sp| {
            // User-supplied file overrides the built-in one.
            std.log.info("Using user-supplied sensitivities file: {s}", .{sp});
            const sens_file = try std.Io.Dir.cwd().openFile(opts.io, sp, .{ .mode = .read_only });
            defer sens_file.close(opts.io);
            var sens_reader_buf: [8192]u8 = undefined;
            var sens_reader = sens_file.reader(opts.io, &sens_reader_buf);
            const sens_content = try sens_reader.interface.allocRemaining(arena_alloc, .unlimited);
            sensitivities = try std.json.parseFromSlice(std.json.Value, arena_alloc, sens_content, .{});
            use_sensitivity = true;
        } else if (arch.sensitivities.len > 1) {
            // Fall back to built-in sensitivities for this architecture.
            std.log.debug("Using built-in sensitivities file for {s}", .{arch.name});
            sensitivities = try std.json.parseFromSlice(std.json.Value, arena_alloc, arch.sensitivities, .{});
            use_sensitivity = true;
        }
    }

    for (model_tensors.items) |*t| {
        var num_elements: u64 = 1;
        for (t.dims) |d| num_elements *= d;

        try assignTensorType(
            t,
            num_elements,
            arch,
            threshold,
            opts,
            use_sensitivity,
            if (use_sensitivity) &sensitivities else null,
            arena_alloc,
        );

        // f64 is unsupported in ComfyUI GGUF — downcast to f32.
        // TODO: make this optional via a flag.
        if (opts.filetype == .gguf and (std.mem.eql(u8, t.type, "f64") or std.mem.eql(u8, t.type, "F64"))) {
            std.log.info("Downcasting unsupported f64 to f32 for tensor {s}", .{t.name});
            t.type = "f32";
            const fat_type = try gguf.GgmlType.fromString(t.type);
            t.size = fat_type.calcSizeInBytes(num_elements);
        }

        var dims_buf = try std.ArrayList(u8).initCapacity(arena_alloc, 5);
        for (t.dims, 0..) |d, i| {
            if (i > 0) try dims_buf.appendSlice(arena_alloc, ", ");
            try dims_buf.print(arena_alloc, "{}", .{d});
        }
        std.log.debug("{s}: Calculated size {} for type {s} with num elements {} with dims [{s}]", .{ t.name, t.size, t.type, num_elements, dims_buf.items });
    }
}

/// Assign every tensor's final on-disk `offset` (and, for ComfyUI cluster dest types on
/// safetensors, its full cluster `size`) in list order. Runs for BOTH the template and
/// auto-assign paths — and after any reordering — so output offsets are always a fresh,
/// contiguous layout instead of meaningless inherited source offsets.
///
/// Cluster dest types expand on disk into `[weight][scale...][comfy_quant]`; their recorded
/// size must be the full cluster size (`clusterWriteSize`), not the packed-weight-only size
/// that `DataType.calcSizeInBytes` returns. The auto path already sets this in
/// `assignTensorType`, but the template path (`applyTemplateEntry`) does not, so we make it
/// authoritative here for both.
fn assignOutputLayout(
    model_tensors: *std.ArrayList(types.Tensor),
    opts: ConvertOptions,
    arena_alloc: std.mem.Allocator,
) !void {
    var offset: u64 = 0;
    for (model_tensors.items) |*t| {
        if (opts.filetype == .safetensors) {
            if (types.DataType.fromString(t.type)) |dt| {
                if (TensorClusters.isClusterType(dt)) {
                    if (try TensorClusters.clusterWriteSize(arena_alloc, dt, t.dims)) |cluster_size| {
                        t.size = cluster_size;
                    }
                }
            } else |_| {}
        }

        // padding only applies to gguf
        if (opts.filetype == .gguf) {
            // TODO: make alignment configurable.
            const padding_len = (32 - (t.size % 32)) % 32;
            t.offset = offset;
            offset += t.size + padding_len;
        } else {
            t.offset = offset;
            offset += t.size;
        }
    }
}

/// Returns the effective QuantizationFamilies: user-supplied if present,
/// otherwise inferred from the target datatype, otherwise all-enabled.
fn resolvedFamilies(opts: ConvertOptions) QuantizationFamilies {
    if (opts.allowed_quant_families) |f| return f;
    if (opts.datatype) |dt| {
        const derived = QuantizationFamilies.fromDataType(dt);
        // If the datatype has no family suffix (e.g. f16), allow all quant families.
        if (derived.allow_0 or derived.allow_1 or derived.allow_k) return derived;
    }
    return .{ .allow_0 = true, .allow_1 = true, .allow_k = true };
}

/// Whether `name` is a token-embedding lookup table (an nn.Embedding weight).
///
/// These are 2D float tensors, indistinguishable by shape from a Linear weight,
/// but they must not be quantized (see the call site in assignTensorType). We
/// match a curated set of terminal module names that are unambiguously
/// nn.Embedding. Positional/patch/time "embedder" projections (pos_embedder,
/// x_embedder, patch_embed, add_embedding, context_embedder, img_emb, ...) are
/// Linear/Conv and are intentionally NOT matched — they quantize fine, and the
/// suffixes below don't collide with them. The misclassification cost is
/// asymmetric: wrongly protecting a Linear only costs a little file size, while
/// wrongly quantizing an Embedding breaks the model, so we err toward matching.
fn isEmbeddingWeight(name: []const u8) bool {
    const suffixes = [_][]const u8{
        ".embed.weight", // Anima llm_adapter, misc adapters
        ".embed_tokens.weight", // HF-style token embeddings
        ".token_embedding.weight",
        ".token_embed.weight",
        ".word_embeddings.weight",
        ".tok_embeddings.weight",
        ".wte.weight", // GPT-2 style
    };
    for (suffixes) |s| {
        if (std.mem.endsWith(u8, name, s)) return true;
    }
    return false;
}

/// Decide the GGUF type for a single tensor and update its `type` and `size`
/// fields in-place. Does not touch `offset` — that's done by the caller.
fn assignTensorType(
    t: *types.Tensor,
    num_elements: u64,
    arch: *const imagearch.Arch,
    threshold: u64,
    opts: ConvertOptions,
    use_sensitivity: bool,
    sensitivities: ?*const std.json.Parsed(std.json.Value),
    arena_alloc: std.mem.Allocator,
) !void {
    // Architecture-specific overrides first.
    if (arch.shouldUpcast(t.name) and opts.filetype == .gguf) {
        std.log.info("Forcing layer {s} to f32 for compatability", .{t.name});
        const ggml_type = gguf.GgmlType.f32;
        t.type = @tagName(ggml_type);
        t.size = ggml_type.calcSizeInBytes(num_elements);
        return;
    }

    // ComfyUI-GGUF shapes a tensor itself only when it is f16 or f32, and wraps
    // no layer that takes a rank-5 weight (the Qwen-VL patch Conv3d), so that
    // one has to arrive as plain floats. llama.cpp holds nothing past rank 4.
    if (opts.filetype == .gguf and t.dims.len >= 5) {
        const ggml_type: gguf.GgmlType = if (opts.datatype == .f32) .f32 else .f16;
        t.type = @tagName(ggml_type);
        t.size = ggml_type.calcSizeInBytes(num_elements);
        return;
    }

    // ComfyUI compatibility: never block-quantize 1D tensors (norms, biases,
    // modulation/scale vectors). ComfyUI loads these as raw float parameters
    // rather than wrapping them for on-the-fly dequant, so an int-quantized
    // version fails to load with a shape/dtype mismatch (the packed byte length
    // is read as the element count). This is a structural rule that generalizes
    // across architectures, replacing most per-model hi-precision lists.
    if (opts.filetype == .gguf and t.dims.len <= 1) return nearestCompatibleType(t, opts, num_elements, arch, .f32);

    // ComfyUI compatibility: never quantize token-embedding lookup tables. An
    // nn.Embedding weight is a 2D float table indexed by token IDs, not a Linear
    // matmul weight, and ComfyUI's quant loaders only wrap Linear layers. A block-
    // or int-quantized Embedding therefore fails to load (e.g. Anima's
    // llm_adapter.embed.weight → "Only Tensors of floating point and complex dtype
    // can require gradients"). A safetensors/GGUF file carries no module-type
    // metadata, so we can't introspect Linear-vs-Embedding; we classify by name,
    // the same way llama.cpp/GPTQ/AWQ treat token_embd. Like the 1D rule above,
    // this is a structural rule that generalizes across architectures, and it
    // applies to every output format (leaving the tensor in its source precision).
    // Safetensors only: ComfyUI's comfy_quant loader wraps Linear and not
    // Embedding, so a quantized table throws there; GGUF consumers read one fine.
    if (opts.filetype == .safetensors and isEmbeddingWeight(t.name)) {
        return nearestCompatibleType(t, opts, num_elements, arch, .family);
    }

    // Too small to quantize - use nearest compatible type
    if (num_elements < threshold) return nearestCompatibleType(t, opts, num_elements, arch, .family);

    // High-precision tensors (e.g. norms, gates) - use nearest compatible type.
    // The llama.cpp families protect only the MoE router here, which ggml
    // multiplies against an f32 residual like a norm vector. Image arches
    // protect whole 2-D weights, so an f32 floor there would double the bytes
    // of a transformer block for nothing.
    if (arch.isHighPrecision(t.name)) {
        return if (arch.row_aligned_blocks)
            nearestCompatibleType(t, opts, num_elements, arch, .f32)
        else
            nearestCompatibleType(t, opts, num_elements, arch, .family);
    }

    // Whole-name protected tables (token_embd, output, lm_head, MoE router)
    if (arch.isHighPrecisionNamed(t.name)) return nearestCompatibleType(t, opts, num_elements, arch, .family);

    // Same, for layers that only need protecting in a narrow-row form (see NarrowRule).
    if (arch.isNarrowHighPrecision(t.name, t.dims)) return nearestCompatibleType(t, opts, num_elements, arch, .family);

    // Apply the target datatype, lifted to any precision floor this architecture
    // declares for the tensor. The floor sits ahead of every format branch, so it
    // holds for GGUF and SafeTensors alike and neither `-a` nor `-x` can undo it.
    var ttype = opts.datatype orelse {
        // No target keeps the source type, which for a block-quantized GGUF
        // tensor has no SafeTensors spelling.
        if (opts.filetype == .safetensors) nearestCompatibleType(t, opts, num_elements, arch, .family);
        return;
    };
    if (arch.precisionFloor(t.name)) |min| {
        const lifted = liftToFloor(ttype, min) orelse
            return nearestCompatibleType(t, opts, num_elements, arch, .family);
        if (lifted != ttype) {
            std.log.info("Precision floor: {s} {s} -> {s}", .{ t.name, @tagName(ttype), @tagName(lifted) });
            ttype = lifted;
        }
    }
    if (opts.filetype == .gguf) {
        const ggml_type = gguf.GgmlType.fromString(@tagName(ttype)) catch unreachable;
        const bs = ggml_type.getBlockSize();

        if (bs > 1 and num_elements % bs != 0) {
            std.log.warn("Cannot convert tensor {s} to type {s} because {} is not a multiple of blocksize {}", .{ t.name, @tagName(ggml_type), num_elements, bs });
            return;
        }

        // Flat blocking (above) scrambles a row-aligned consumer unless the rows
        // themselves divide the block size. Drop to a narrower block before
        // giving up on quantizing at all: n_embd 896 is a q4_k row misalignment
        // on a stock Qwen2.5-0.5B, and sparing every 2-D weight there would
        // write an f16 file against a q4_k request.
        if (bs > 1 and arch.row_aligned_blocks and t.dims[t.dims.len - 1] % bs != 0) {
            const narrower = rowAlignedFallback(ttype, num_elements, t.dims[t.dims.len - 1]) orelse
                return nearestCompatibleType(t, opts, num_elements, arch, .family);
            std.log.info("Row blocking: {s} {s} -> {s}", .{ t.name, @tagName(ttype), @tagName(narrower) });
            ttype = narrower;
        }
    }

    // ComfyUI cluster safetensors outputs. Eligibility (which tensors get clustered vs. fall
    // back to their source type) is format-specific and lives here; the physical byte layout
    // is owned by TensorClusters.clusterWriteLayout, shared with the header/data writers.
    if (opts.filetype == .safetensors and TensorClusters.isClusterType(ttype)) {
        if (clusterEligible(t, ttype, num_elements, arch)) {
            t.type = @tagName(ttype);
            t.size = (try TensorClusters.clusterWriteSize(arena_alloc, ttype, t.dims)).?;
            return;
        }
        return nearestCompatibleType(t, opts, num_elements, arch, .family);
    }

    if (use_sensitivity) {
        const source_type = t.type;
        const source_size = t.size;
        try applySensitivityQuantization(t, num_elements, ttype, opts.quantization_aggressiveness, resolvedFamilies(opts), sensitivities.?);
        // The ladder runs across quant families, so it can hand back a larger
        // block size than the target was checked against above (q4_0's 32 up to
        // q4_k's 256): re-run both blocking checks on the type actually picked.
        if (opts.filetype == .gguf) {
            const bs = (gguf.GgmlType.fromString(t.type) catch unreachable).getBlockSize();
            if (bs > 1 and (num_elements % bs != 0 or
                (arch.row_aligned_blocks and t.dims[t.dims.len - 1] % bs != 0)))
            {
                const picked = types.DataType.fromString(t.type) catch unreachable;
                // Put the tensor back as it arrived: the fallback reads t.type as
                // the source precision, and leaves both fields alone for a source
                // it has no upcast for, so a stale size would be written straight
                // into the output layout.
                t.type = source_type;
                t.size = source_size;
                if (num_elements % bs == 0) {
                    if (rowAlignedFallback(picked, num_elements, t.dims[t.dims.len - 1])) |narrower| {
                        std.log.info("Row blocking: {s} {s} -> {s}", .{ t.name, @tagName(picked), @tagName(narrower) });
                        t.type = @tagName(narrower);
                        t.size = narrower.calcSizeInBytes(num_elements);
                        return;
                    }
                }
                return nearestCompatibleType(t, opts, num_elements, arch, .family);
            }
        }
    } else {
        std.log.debug("Will convert tensor {s} from type {s} to {s}", .{ t.name, t.type, @tagName(ttype) });
        t.type = @tagName(ttype);
        t.size = ttype.calcSizeInBytes(num_elements);
    }
}

/// Decide whether tensor `t` is eligible for cluster quantization to `ttype`, vs. falling
/// back to its source type. Only weight matrices are clustered, and each format has its own
/// shape constraints (block/tiling divisibility, per-arch NVFP4 passthrough).
fn clusterEligible(t: *const types.Tensor, ttype: types.DataType, num_elements: u64, arch: *const imagearch.Arch) bool {
    if (!std.mem.endsWith(u8, t.name, ".weight")) return false;
    const n_cols: u64 = if (t.dims.len >= 1) t.dims[t.dims.len - 1] else 0;
    return switch (ttype) {
        .SCALED_F8_E4M3 => true,
        // Require at least one full 32-element block; tiny last dims fall back.
        .MXFP4, .MXFP8_E4M3 => t.dims.len >= 1 and n_cols >= 32,
        // cuBLAS tiling: cols % 64 == 0 and rows % 128 == 0, minus per-arch passthrough.
        .NVFP4 => t.dims.len >= 1 and n_cols >= 64 and n_cols % 64 == 0 and
            (num_elements / n_cols) % 128 == 0 and !arch.isNvfp4Passthrough(t.name),
        .INT8 => t.dims.len == 2 and n_cols >= 1,
        .INT8_CONVROT => t.dims.len == 2 and n_cols % TensorClusters.int8_convrot_group_size == 0,
        // convrot_w4a4 rotates in column-groups of convrot_groupsize, so the input dim must
        // be divisible by it (which also guarantees the even column count nibble-packing needs).
        .INT4_CONVROT, .INT4_CONVROT_SR => t.dims.len == 2 and n_cols % TensorClusters.int4_convrot_group_size == 0,
        // Two separate constraints on the input dim: the rotation group and the scale group.
        // The rotation group is a multiple of the scale group today, so the first check covers
        // both, but the scale group is a per-layer field and need not stay 16.
        .ASYM_W4A8_INT8, .ASYM_W4A8_INT8_ZP, .W6A8_INT8 => t.dims.len == 2 and
            n_cols % TensorClusters.asym_w4a8_convrot_group_size == 0 and
            n_cols % TensorClusters.asym_w4a8_group_size == 0,
        else => false,
    };
}

/// The type to quantize to when `ttype`'s 256-element blocks do not divide a
/// row, following llama.cpp's convert_incompatible_tensor ladder down to a
/// 32-element block type. Null when nothing narrower fits, which sends the
/// tensor to nearestCompatibleType instead.
fn rowAlignedFallback(ttype: types.DataType, num_elements: u64, row: u64) ?types.DataType {
    const narrower: types.DataType = switch (ttype) {
        .q4_k => .q5_0,
        .q5_k => .q5_1,
        .q6_k => .q8_0,
        // llama.cpp sends these to iq4_nl, which ggufy has no encoder for.
        .q2_k, .q3_k => .q4_0,
        else => return null,
    };
    const bs = (gguf.GgmlType.fromString(@tagName(narrower)) catch unreachable).getBlockSize();
    if (num_elements % bs != 0 or row % bs != 0) return null;
    return narrower;
}

/// nearestCompatibleType converts a tensor type to the nearest type compatible with the output (or leaves it the same, if it's already compatible)
/// Why the tensor was spared the target type. .f32 keeps the old f32 floor
/// (1-D parameters, MoE routers - both llama.cpp and ComfyUI expect f32);
/// .family lets llama.cpp-family arches follow the target's float width for
/// 2-D tables while everything else still upcasts bf16 to f32.
pub const SpareReason = enum { family, f32 };

fn nearestCompatibleType(
    t: *types.Tensor,
    opts: ConvertOptions,
    num_elements: u64,
    arch: *const imagearch.Arch,
    comptime reason: SpareReason,
) void {
    const sourceType = types.DataType.fromString(t.type) catch unreachable;
    if (opts.filetype == .gguf) {
        // The f32 floor takes every narrower float, not just bf16: ggml has no
        // f32-by-f16 binary op, so an f16 norm vector aborts llama.cpp at the
        // first ggml_mul against the f32 residual.
        if (reason == .f32) {
            switch (sourceType) {
                .F8_E4M3, .F8_E5M2, .BF16, .F16, .bf16, .f16 => {
                    const ggml_type = gguf.GgmlType.f32;
                    t.type = @tagName(ggml_type);
                    t.size = ggml_type.calcSizeInBytes(num_elements);
                },
                else => {},
            }
            return;
        }
        // FP8 has no GGUF equivalent — upcast to F16.
        if (sourceType == .F8_E4M3 or sourceType == .F8_E5M2) {
            const ggml_type = gguf.GgmlType.f16;
            t.type = @tagName(ggml_type);
            t.size = ggml_type.calcSizeInBytes(num_elements);
            return;
        }
        if (sourceType == .BF16) {
            // ComfyUI cannot carry bf16 in its shape math, so float tensors
            // upcast to F32 there. llama.cpp-family arches instead follow the
            // target's float width for 2-D tables (llama.cpp's own converter
            // casts bf16 checkpoints to f16).
            const target: gguf.GgmlType = if (arch.row_aligned_blocks and t.dims.len > 1) blk: {
                const want = opts.datatype orelse break :blk .f16;
                const g = gguf.GgmlType.fromString(@tagName(want)) catch break :blk .f16;
                break :blk switch (g) {
                    .f32, .f16, .bf16 => g,
                    else => .f16,
                };
            } else .f32;
            t.type = @tagName(target);
            t.size = target.calcSizeInBytes(num_elements);
            return;
        }
        return;
    }

    // Sparing a tensor keeps its source type, and a block-quantized GGUF type has
    // no SafeTensors spelling at all: the writer rejects the whole file over it.
    // A q4_K-quantized token_embd in any stock llama.cpp file reaches here, so
    // -H needs a float floor of its own. f16 when the target is not a float,
    // which is what llama.cpp's converter writes for these tables.
    _ = sourceType.forFormat(.safetensors) catch {
        const floor: types.DataType = blk: {
            const want = opts.datatype orelse break :blk .F16;
            if (!want.isFloatType()) break :blk .F16;
            break :blk want.forFormat(.safetensors) catch .F16;
        };
        t.type = @tagName(floor);
        t.size = floor.calcSizeInBytes(num_elements);
    };
}

/// Applies sensitivity-adjusted quantization to a single tensor.
fn applySensitivityQuantization(
    t: *types.Tensor,
    num_elements: u64,
    dtype: types.DataType,
    aggressiveness: f32,
    families: QuantizationFamilies,
    sensitivities: *const std.json.Parsed(std.json.Value),
) !void {
    const sens_value = sensitivities.value.object.get(t.name);
    if (sens_value) |sv| {
        const sens: f32 = switch (sv) {
            .float => |fl| @floatCast(fl),
            .integer => |i| @floatFromInt(i),
            else => return error.InvalidSensitivityValue,
        };

        const target_level = try QuantizationLevel.fromString(@tagName(dtype));
        const quant_level = try calculateQuantizationLevel(sens, aggressiveness, target_level, t.type, families);

        const final_type_str = @tagName(quant_level);
        const final_ggml_type = try gguf.GgmlType.fromString(final_type_str);

        std.log.info("Layer {s}: sensitivity={d:.1}, hardness={d}, {s} -> {s}", .{ t.name, sens, aggressiveness, @tagName(dtype), final_type_str });

        t.type = final_type_str;
        t.size = final_ggml_type.calcSizeInBytes(num_elements);
    } else {
        std.log.warn("No sensitivity data for layer {s}, using target type", .{t.name});
        t.type = @tagName(dtype);
        t.size = dtype.calcSizeInBytes(num_elements);
    }
}

// ============================================================================
// Step 3 — Shape fix
// ============================================================================

const REARRANGE_THRESHOLD = 512;

/// Inverse of applyShapeFix: when the source carries "comfy.gguf.orig_shape.<name>"
/// metadata (i.e. a GGUF previously written with shape_fix), restore each affected
/// tensor's dims to the recorded original shape. The rearrange is a pure
/// reinterpretation of contiguous data, so only the dims metadata changes.
/// No-op for sources without this metadata (e.g. plain safetensors).
fn restoreOrigShapes(
    model_tensors: *std.ArrayList(types.Tensor),
    src_meta: ?std.json.ObjectMap,
    arena_alloc: std.mem.Allocator,
) !void {
    const meta = src_meta orelse return;
    for (model_tensors.items) |*t| {
        const key = try std.fmt.allocPrint(arena_alloc, "comfy.gguf.orig_shape.{s}", .{t.name});
        const value = meta.get(key) orelse continue;
        const arr = switch (value) {
            .array => |a| a,
            else => continue,
        };
        const dims = try arena_alloc.alloc(usize, arr.items.len);
        var ok = true;
        for (arr.items, 0..) |item, idx| {
            switch (item) {
                .integer => |n| dims[idx] = @intCast(n),
                else => {
                    ok = false;
                    break;
                },
            }
        }
        if (!ok) continue;
        t.dims = dims;
        std.log.info("Restored original shape for {s}", .{t.name});
    }
}

/// Reshapes qualifying tensors to (N/256, 256) for ComfyUI compatibility and
/// records their original shapes under "comfy.gguf.orig_shape.<name>" in
/// `extra_metadata`.
fn applyShapeFix(
    model_tensors: *std.ArrayList(types.Tensor),
    extra_metadata: *std.json.ObjectMap,
    arena_alloc: std.mem.Allocator,
) !void {
    for (model_tensors.items) |*t| {
        var n_elements: u64 = 1;
        for (t.dims) |d| n_elements *= @intCast(d);

        const n_dims = t.dims.len;
        const last_dim = if (n_dims > 0) t.dims[n_dims - 1] else 0;

        // Criteria:
        //   1. More than one dimension
        //   2. Total elements >= 512
        //   3. Total elements divisible by 256
        //   4. Last dimension NOT divisible by 256
        if (n_dims <= 1) continue;
        if (n_elements < REARRANGE_THRESHOLD) continue;
        if (n_elements % 256 != 0) continue;
        if (@mod(last_dim, 256) == 0) continue;

        // Record original shape.
        var orig_shape_arr = std.json.Array.init(arena_alloc);
        for (t.dims) |d| try orig_shape_arr.append(.{ .integer = @intCast(d) });
        const key = try std.fmt.allocPrint(arena_alloc, "comfy.gguf.orig_shape.{s}", .{t.name});
        try extra_metadata.put(arena_alloc, key, .{ .array = orig_shape_arr });

        // Reshape to (N/256, 256).
        var new_dims = try arena_alloc.alloc(usize, 2);
        new_dims[0] = n_elements / 256;
        new_dims[1] = 256;
        t.dims = new_dims;

        std.log.info("Applied shape fix to {s}: new shape {{ {}, {} }}", .{ t.name, new_dims[0], new_dims[1] });
    }
}

/// Merges top-level keys from `base_json` into the `config` KV stored in
/// `metadata`, skipping any key that the source config already defines.
/// This fills in architectural constants (vae, audio_vae, vocoder) that
/// fine-tuned safetensors files typically omit.
fn mergeBaseConfig(
    metadata: *std.json.ObjectMap,
    base_json: []const u8,
    arena_alloc: std.mem.Allocator,
) !void {
    const base = try std.json.parseFromSlice(std.json.Value, arena_alloc, base_json, .{});

    const existing_str: []const u8 = if (metadata.get("config")) |v| switch (v) {
        .string => |s| s,
        else => "{}",
    } else "{}";
    const existing = try std.json.parseFromSlice(std.json.Value, arena_alloc, existing_str, .{});

    // Build merged object: start from base, let source override.
    var merged: std.json.ObjectMap = .empty;
    var base_it = base.value.object.iterator();
    while (base_it.next()) |e| try merged.put(arena_alloc, e.key_ptr.*, e.value_ptr.*);
    var src_it = existing.value.object.iterator();
    while (src_it.next()) |e| try merged.put(arena_alloc, e.key_ptr.*, e.value_ptr.*);

    const merged_str = try std.json.Stringify.valueAlloc(arena_alloc, std.json.Value{ .object = merged }, .{});
    try metadata.put(arena_alloc, try arena_alloc.dupe(u8, "config"), .{ .string = merged_str });
}

/// Maps the target quantization type to a GGUF general.file_type integer.
/// Values follow the llama.cpp LLAMA_FTYPE_* convention.
fn ggufFileType(datatype: ?types.DataType) i64 {
    const dt = datatype orelse return 1; // default: MOSTLY_F16
    return switch (dt) {
        .f32 => 0,
        .f16 => 1,
        .q4_0 => 2,
        .q4_1 => 3,
        .q5_0 => 8,
        .q5_1 => 9,
        .q8_0 => 7,
        .q2_k => 10,
        .q3_k => 12,
        .q4_k => 15,
        .q5_k => 17,
        .q6_k => 18,
        .bf16 => 32,
        else => 1,
    };
}

/// File-level metadata key (comfy-kitchen layout) mapping each layer to its source
/// quantization identity. Conversion dequantizes/re-encodes those layers, so this header
/// no longer describes the output — copying it verbatim makes ComfyUI try to load the
/// converted file as INT8 ConvRot and fail. It is stripped from every output.
const source_quant_metadata_key = "_quantization_metadata";

const ggufy_repo_url = "https://github.com/qskousen/ggufy";

/// Stamp ggufy's identity onto the converter-provenance metadata keys of every
/// output. Values a source carried from a prior tool (e.g. comfy-kitchen's INT8
/// ConvRot converter) misdescribe the file's origin after conversion and are
/// overwritten.
fn stampConverterProvenance(metadata: *std.json.ObjectMap, arena_alloc: std.mem.Allocator) !void {
    const stamps = [_]struct { key: []const u8, value: []const u8 }{
        .{ .key = "converted_by", .value = "ggufy " ++ build_options.version },
        .{ .key = "converter_url", .value = ggufy_repo_url },
        .{ .key = "converter_note", .value = "Converted with ggufy" },
    };
    for (stamps) |s| try metadata.put(arena_alloc, s.key, .{ .string = s.value });
}

/// The architecture the source file declares for itself, if any. Name detection
/// cannot recover it: every llama-shaped file detects as "llama" whatever its
/// metadata calls it, and the `<arch>.*` keys are prefixed with this, not that.
fn sourceMetaStr(f: anytype, key: []const u8) ?[]const u8 {
    const md = f.getSourceMetadata() orelse return null;
    const v = md.get(key) orelse return null;
    return switch (v) {
        .string => |s| if (s.len > 0) s else null,
        else => null,
    };
}

fn sourceArchName(f: anytype) ?[]const u8 {
    return sourceMetaStr(f, "general.architecture");
}

/// Copy the keys `metadata` does not already carry. Whoever wrote a key first
/// keeps it, so callers order their sources by priority.
fn fillMetadataGaps(metadata: *std.json.ObjectMap, from: std.json.ObjectMap, arena_alloc: std.mem.Allocator) !void {
    var it = from.iterator();
    while (it.next()) |entry| {
        if (!metadata.contains(entry.key_ptr.*))
            try metadata.put(arena_alloc, try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
    }
}

/// Assemble the final GGUF metadata map (added/overwritten standard keys, merged
/// template/source keys, extra shape-fix records, dropped source quant header,
/// stamped provenance, merged base config) into `metadata`. Shared by the real
/// writer (`writeGguf`) and the size predictor so both serialize identical metadata.
fn buildGgufMetadata(
    metadata: *std.json.ObjectMap,
    f: anytype,
    arch: *const imagearch.Arch,
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.json.ObjectMap,
    opts: ConvertOptions,
    arena_alloc: std.mem.Allocator,
    hf: ?HfLlm.Model,
) !void {
    // Standard metadata. The generic arch is a detection failure, not a result, so it
    // defers to a source/template general.architecture via the copies below instead of
    // clobbering it; the fallback after the copies restores "unknown" when none exists.
    // A name that is not a GGUF architecture id defers the same way, and so does a
    // family-wide name over a source that already says which member it is: renaming a
    // gemma3 file to "llama" leaves every gemma3.* key behind under the old prefix and
    // llama.cpp dies on a missing llama.block_count. A config.json (`hf`) names the
    // model outright and is not a detection guess, so it still writes.
    const arch_name = opts.arch_override orelse arch.name;
    const source_arch = sourceArchName(f);
    const defers_to_source = opts.arch_override == null and hf == null and
        arch.keys_family_wide and source_arch != null;
    const writes_arch = arch != &imagearch.generic_arch and (arch.gguf_arch_id or hf != null) and !defers_to_source;
    if (writes_arch or opts.arch_override != null) {
        try metadata.put(arena_alloc, try arena_alloc.dupe(u8, "general.architecture"), .{ .string = arch_name });
    }
    try metadata.put(arena_alloc, try arena_alloc.dupe(u8, "general.quantization_version"), .{ .integer = 2 });
    try metadata.put(arena_alloc, try arena_alloc.dupe(u8, "general.file_type"), .{ .integer = ggufFileType(opts.datatype) });

    // Every copy from here down only fills gaps, so the order is the priority:
    // template over HF config over source file. The HF keys go through the same
    // fill so a -t value can correct a config.json the checkpoint got wrong (a
    // short max_position_embeddings, a stale chat template). The architecture is
    // the one thing a template cannot reach over an hf, above: the name and the
    // prefix its dimension keys carry have to move together, which is -A.
    if (template_metadata) |meta| try fillMetadataGaps(metadata, meta, arena_alloc);

    if (hf) |m| {
        var hf_meta: std.json.ObjectMap = .empty;
        try HfLlm.addMetadata(m, &hf_meta, arch_name, arena_alloc);
        try fillMetadataGaps(metadata, hf_meta, arena_alloc);
    }

    // Source-file metadata, which -t replaces rather than joins.
    if (template_metadata == null) {
        if (f.getSourceMetadata()) |meta| try fillMetadataGaps(metadata, meta, arena_alloc);
    }

    // Extra metadata (e.g. shape-fix records).
    try fillMetadataGaps(metadata, extra_metadata, arena_alloc);

    if (!metadata.contains("general.architecture")) {
        // The copies above were the deferral, and -t skips the source one, so
        // take the source's name off the file directly rather than stamping the
        // detected one: that would claim an architecture whose <arch>.* keys sit
        // under another prefix. A name that is not a GGUF architecture id must
        // not arrive here either - no loader knows it.
        const fallback = if (opts.arch_override != null)
            arch_name
        else
            source_arch orelse if (arch.gguf_arch_id) arch_name else imagearch.generic_arch.name;
        try metadata.put(arena_alloc, try arena_alloc.dupe(u8, "general.architecture"), .{ .string = fallback });
    }

    // Drop the source quantization header — it describes the pre-conversion layout —
    // and correct any converter-provenance keys carried over from the source tool.
    _ = metadata.swapRemove(source_quant_metadata_key);
    try stampConverterProvenance(metadata, arena_alloc);

    // Merge arch base config (vae/audio_vae/vocoder etc.) into the `config` KV.
    // Fine-tuned source files often omit these sections; they are architectural
    // constants for the base model that ComfyUI needs to initialise decoders.
    if (arch.base_config_json.len > 0) {
        try mergeBaseConfig(metadata, arch.base_config_json, arena_alloc);
    }
}

// Resolved before the output file exists, so a source missing the head counts
// fails without leaving a truncated safetensors behind. `arch_name` prefixes the
// keys and must come from the source's own general.architecture, not detection.
fn ropeHeadsFromSource(f: anytype, arch_name: []const u8, arena_alloc: std.mem.Allocator) !HfLlm.RopeHeads {
    const md = f.getSourceMetadata() orelse return error.HfNamesUnsupported;
    const hc = md.get(try std.fmt.allocPrint(arena_alloc, "{s}.attention.head_count", .{arch_name})) orelse
        return error.HfNamesUnsupported;
    const hkv = md.get(try std.fmt.allocPrint(arena_alloc, "{s}.attention.head_count_kv", .{arch_name})) orelse
        return error.HfNamesUnsupported;
    const hc_i = switch (hc) {
        .integer => |v| v,
        else => return error.HfNamesUnsupported,
    };
    const hkv_i = switch (hkv) {
        .integer => |v| v,
        else => return error.HfNamesUnsupported,
    };
    if (hc_i <= 0 or hkv_i <= 0) return error.HfNamesUnsupported;
    return .{ .head_count = @intCast(hc_i), .head_count_kv = @intCast(hkv_i) };
}

/// The block index the MTP head occupies in a qwen35 GGUF, read off the file's
/// own keys: block_count counts it, nextn_predict_layers says how many of the
/// tail are it. null when the file names neither, which is a file with no MTP
/// head rather than an error.
fn mtpBaseFromSource(f: anytype, arch_name: []const u8, arena_alloc: std.mem.Allocator) ?u32 {
    const md = f.getSourceMetadata() orelse return null;
    const bc = md.get(std.fmt.allocPrint(arena_alloc, "{s}.block_count", .{arch_name}) catch return null) orelse return null;
    const nx = md.get(std.fmt.allocPrint(arena_alloc, "{s}.nextn_predict_layers", .{arch_name}) catch return null) orelse return null;
    const bc_i = switch (bc) {
        .integer => |v| v,
        else => return null,
    };
    const nx_i = switch (nx) {
        .integer => |v| v,
        else => return null,
    };
    if (bc_i <= nx_i or nx_i < 0) return null;
    return @intCast(bc_i - nx_i);
}

/// The hybrid's head geometry, read off a GGUF's own ssm.* keys, for the
/// GGUF -> HF direction where there is no config.json to read it from.
fn qwen35CtxFromSource(f: anytype, arch_name: []const u8, arena_alloc: std.mem.Allocator) ?VReorderCtx {
    const md = f.getSourceMetadata() orelse return null;
    const num = struct {
        fn get(m: anytype, a: std.mem.Allocator, arch: []const u8, key: []const u8) ?usize {
            const full = std.fmt.allocPrint(a, "{s}.ssm.{s}", .{ arch, key }) catch return null;
            const v = m.get(full) orelse return null;
            return switch (v) {
                .integer => |i| if (i > 0) @intCast(i) else null,
                else => null,
            };
        }
    };
    const k_heads = num.get(md, arena_alloc, arch_name, "group_count") orelse return null;
    const v_heads = num.get(md, arena_alloc, arch_name, "time_step_rank") orelse return null;
    const state = num.get(md, arena_alloc, arch_name, "state_size") orelse return null;
    const inner = num.get(md, arena_alloc, arch_name, "inner_size") orelse return null;
    if (v_heads % k_heads != 0 or inner % v_heads != 0) return null;
    return .{
        .to_hf = true,
        .k_heads = k_heads,
        .v_per_k = v_heads / k_heads,
        .qk_rows = 2 * state * k_heads,
        .value_head_dim = inner / v_heads,
    };
}

const RopePatchCtx = struct { head_count: usize, head_count_kv: usize, to_hf: bool };

fn ropePatchApply(
    ctx: *anyopaque,
    alloc: std.mem.Allocator,
    name: []const u8,
    dims: []const usize,
    source_type: []const u8,
    bytes: []u8,
) !void {
    const heads: *const RopePatchCtx = @ptrCast(@alignCast(ctx));
    const groups = HfLlm.ropePermuteGroups(.{
        .head_count = heads.head_count,
        .head_count_kv = heads.head_count_kv,
    }, name) orelse return;

    // The writer dequantizes a block-quantized source to F32 before calling
    // this, so anything but a float here means the rows cannot be swapped and
    // the tensor would reach the output in the wrong RoPE order.
    const elem_size: usize = switch (try types.DataType.fromString(source_type)) {
        .BF16, .F16, .bf16, .f16 => 2,
        .F32, .f32 => 4,
        .F64, .f64 => 8,
        else => return error.RopePatchUnsupportedType,
    };
    if (dims.len == 0) return error.RopePatchNoDims;
    // Tensor structs keep dims in logical [out, in] order for either format.
    const rows = dims[0];
    var cols: usize = 1;
    for (dims[1..]) |d| cols *= d;
    try HfLlm.ropePermuteInPlace(alloc, bytes, elem_size, rows, cols, groups, heads.to_hf);
}

const VReorderCtx = struct {
    to_hf: bool,
    k_heads: usize,
    v_per_k: usize,
    /// Rows ahead of the V block in attn_qkv: the q and k projections.
    qk_rows: usize,
    value_head_dim: usize,
};

/// What a qwen35 tensor needs doing to it, recognised under either spelling:
/// the GGUF names on the way in, the HF ones on the way back out, because the
/// rename happens before the patch runs in both directions.
const Qwen35Role = enum {
    /// q and k rows sit ahead of the V block; only its tail moves.
    qkv_tail,
    /// Whole tensor is V rows, one head_dim-sized block per head.
    v_rows,
    /// Whole tensor is V rows, one row per head.
    v_scalars,
    /// Same, and the value stored is the log of the decay.
    a_log,
    /// The V space is the input dim, so columns move rather than rows.
    out_proj,
    /// Stored as w+1, with the linear-attention norm the one exception.
    norm_offset,
    none,
};

fn qwen35Role(name: []const u8) Qwen35Role {
    // The vision tower rides along under -H, and none of these rewrites are
    // its: its qkv and merger norm would otherwise match by suffix.
    if (HfLlm.isVisionTensor(name) or std.mem.startsWith(u8, name, "v.") or std.mem.startsWith(u8, name, "mm.")) return .none;
    const ends = std.mem.endsWith;
    if (ends(u8, name, ".attn_qkv.weight") or ends(u8, name, ".linear_attn.in_proj_qkv.weight")) return .qkv_tail;
    if (ends(u8, name, ".ssm_conv1d.weight") or ends(u8, name, ".linear_attn.conv1d.weight")) return .qkv_tail;
    if (ends(u8, name, ".attn_gate.weight") or ends(u8, name, ".linear_attn.in_proj_z.weight")) return .v_rows;
    if (ends(u8, name, ".ssm_out.weight") or ends(u8, name, ".linear_attn.out_proj.weight")) return .out_proj;
    if (ends(u8, name, ".ssm_a") or ends(u8, name, ".linear_attn.A_log")) return .a_log;
    if (ends(u8, name, ".ssm_alpha.weight") or ends(u8, name, ".linear_attn.in_proj_a.weight") or
        ends(u8, name, ".ssm_beta.weight") or ends(u8, name, ".linear_attn.in_proj_b.weight") or
        ends(u8, name, ".ssm_dt.bias") or ends(u8, name, ".linear_attn.dt_bias")) return .v_scalars;
    // The linear-attention block's own norm is the exception llama.cpp carves
    // out; every other norm carries the offset.
    if (ends(u8, name, ".ssm_norm.weight") or ends(u8, name, ".linear_attn.norm.weight")) return .none;
    // The MTP head's two input norms are the trap: as GGUF names they end in
    // "norm.weight" (nextn.enorm, nextn.hnorm), as HF names they do not
    // (pre_fc_norm_embedding, pre_fc_norm_hidden), so a suffix test alone
    // applies the offset one way and silently not the other.
    if (ends(u8, name, "pre_fc_norm_embedding.weight") or ends(u8, name, "pre_fc_norm_hidden.weight")) return .norm_offset;
    if (ends(u8, name, "norm.weight")) return .norm_offset;
    return .none;
}

/// Which rows a role moves, given the tensor's actual row count.
fn vReorderSpan(ctx: VReorderCtx, name: []const u8, rows: usize) ?struct { start: usize, len: usize, head_dim: usize } {
    return switch (qwen35Role(name)) {
        .qkv_tail => if (rows <= ctx.qk_rows) null else .{
            .start = ctx.qk_rows,
            .len = rows - ctx.qk_rows,
            .head_dim = ctx.value_head_dim,
        },
        .v_rows => .{ .start = 0, .len = rows, .head_dim = ctx.value_head_dim },
        .v_scalars, .a_log => .{ .start = 0, .len = rows, .head_dim = 1 },
        .out_proj, .norm_offset, .none => null,
    };
}


fn vReorderApply(
    ctx_ptr: *anyopaque,
    alloc: std.mem.Allocator,
    name: []const u8,
    dims: []const usize,
    source_type: []const u8,
    bytes: []u8,
) !void {
    const ctx: *const VReorderCtx = @ptrCast(@alignCast(ctx_ptr));
    if (dims.len == 0) return error.RopePatchNoDims;
    // force_f32 means every tensor reaching here is F32, which the arithmetic
    // below requires and the row moves do not mind.
    switch (try types.DataType.fromString(source_type)) {
        .F32, .f32 => {},
        else => return error.RopePatchUnsupportedType,
    }
    const rows = dims[0];
    var cols: usize = 1;
    for (dims[1..]) |d| cols *= d;
    const vals = std.mem.bytesAsSlice(f32, bytes);
    if (vals.len != rows * cols) return error.VReorderSizeMismatch;

    switch (qwen35Role(name)) {
        // HF stores the log of the decay, the GGUF the decay. log(-x) is not a
        // bit-exact inverse of -exp(x), so a round trip drifts here.
        .a_log => for (vals) |*v| {
            v.* = if (ctx.to_hf) @log(-v.*) else -@exp(v.*);
        },
        // Likewise w+1 then w-1: for a weight much smaller than 1 the offset
        // absorbs it, and subtracting does not bring it back.
        .norm_offset => {
            for (vals) |*v| v.* += if (ctx.to_hf) -1 else 1;
            return;
        },
        // out_proj reads the V space, so its columns move rather than its rows.
        .out_proj => {
            try vReorderColumns(alloc, vals, rows, cols, ctx.k_heads, ctx.v_per_k, ctx.value_head_dim, ctx.to_hf);
            return;
        },
        else => {},
    }

    const span = vReorderSpan(ctx.*, name, rows) orelse return;
    try HfLlm.vReorderInPlace(
        alloc,
        std.mem.sliceAsBytes(vals[span.start * cols ..][0 .. span.len * cols]),
        4,
        span.len,
        cols,
        ctx.k_heads,
        ctx.v_per_k,
        span.head_dim,
        ctx.to_hf,
    );
}

/// The same grouped-to-tiled move applied across a row instead of down a
/// column, for the projection whose input dim is the V space.
fn vReorderColumns(
    alloc: std.mem.Allocator,
    vals: []align(1) f32,
    rows: usize,
    cols: usize,
    k_heads: usize,
    v_per_k: usize,
    head_dim: usize,
    to_hf: bool,
) !void {
    if (v_per_k <= 1) return;
    if (cols != k_heads * v_per_k * head_dim) return error.VReorderRowsNotDivisible;
    const tmp = try alloc.alloc(f32, cols);
    defer alloc.free(tmp);
    for (0..rows) |r| {
        const row = vals[r * cols ..][0..cols];
        for (0..cols) |dst| {
            const d = dst % head_dim;
            const src = if (to_hf) blk: {
                const v = (dst / head_dim) % v_per_k;
                const k = dst / (head_dim * v_per_k);
                break :blk v * (k_heads * head_dim) + k * head_dim + d;
            } else blk: {
                const k = (dst / head_dim) % k_heads;
                const v = dst / (head_dim * k_heads);
                break :blk k * (v_per_k * head_dim) + v * head_dim + d;
            };
            tmp[dst] = row[src];
        }
        @memcpy(row, tmp);
    }
}

fn vReorderMatches(ctx_ptr: *anyopaque, name: []const u8) bool {
    _ = ctx_ptr;
    return qwen35Role(name) != .none;
}

fn ropePatchMatches(ctx: *anyopaque, name: []const u8) bool {
    const heads: *const RopePatchCtx = @ptrCast(@alignCast(ctx));
    return HfLlm.ropePermuteGroups(.{
        .head_count = heads.head_count,
        .head_count_kv = heads.head_count_kv,
    }, name) != null;
}

/// Refuse, before a byte is written, any tensor assigned a type whose encoder
/// aborts without importance weights. ggml calls GGML_ASSERT for these, which
/// kills the process partway through the file - by which point the header is
/// already on disk and the user has waited for however much of the model got
/// converted first.
fn checkImatrixCoverage(
    tensors: []const types.Tensor,
    imatrix: ?*const ImatrixMod.Imatrix,
    arena_alloc: std.mem.Allocator,
) !void {
    var missing: std.ArrayList([]const u8) = .empty;
    // More than one type can be at fault in a single assignment, so name the
    // set rather than whichever happened to come first.
    var kinds: std.ArrayList([]const u8) = .empty;
    for (tensors) |t| {
        const dt = types.DataType.fromString(t.type) catch continue;
        const gt = gguf.GgmlType.fromString(@tagName(dt)) catch continue;
        if (!ggml.ggml_quantize_requires_imatrix(@intCast(@intFromEnum(gt)))) continue;
        const covered = if (imatrix) |im| im.forTensor(t.name) != null else false;
        if (!covered) {
            try missing.append(arena_alloc, t.name);
            var seen = false;
            for (kinds.items) |k| {
                if (std.mem.eql(u8, k, t.type)) seen = true;
            }
            if (!seen) try kinds.append(arena_alloc, t.type);
        }
    }
    if (missing.items.len == 0) return;

    const shown = @min(missing.items.len, 6);
    std.log.err(
        "{d} tensors are assigned {s}, whose encoders cannot run without importance weights",
        .{ missing.items.len, try std.mem.join(arena_alloc, "/", kinds.items) },
    );
    std.log.err("  {s}{s}", .{
        try std.mem.join(arena_alloc, ", ", missing.items[0..shown]),
        if (missing.items.len > shown) ", ..." else "",
    });
    if (imatrix == null) {
        std.log.err("  pass -I <imatrix.gguf> to supply them", .{});
    } else {
        std.log.err("  the imatrix given covers no entry for these; collect one that does, or assign them a type that does not require weights", .{});
    }
    return error.TypeRequiresImatrix;
}

fn imatrixGet(ctx: *const anyopaque, t: types.Tensor) ?[]const f32 {
    const im: *const ImatrixMod.Imatrix = @ptrCast(@alignCast(ctx));
    return im.forTensor(t.name);
}

/// Open the imatrix named by `--imatrix`, or null when none was given. A path
/// that will not load is an error rather than a warning: an unweighted
/// conversion looks identical from the outside, so a typo must not pass as one.
fn loadImatrix(opts: ConvertOptions, gpa: std.mem.Allocator) !?ImatrixMod.Imatrix {
    const path = opts.imatrix_path orelse return null;
    var im = ImatrixMod.load(path, opts.io, gpa) catch |e| {
        std.log.err("could not read imatrix {s}: {s}", .{ path, @errorName(e) });
        return e;
    };
    std.log.info("imatrix {s}: {d} tensors, {d} chunks", .{ path, im.count(), im.chunk_count });
    return im;
}

fn writeGguf(
    f: anytype,
    model_tensors: std.ArrayList(types.Tensor),
    arch: *const imagearch.Arch,
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.json.ObjectMap,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    groups: *const TensorClusters.GroupResult,
    hf: ?HfLlm.Model,
    hf_permute: ?HfLlm.RopeHeads,
    imatrix: ?*const ImatrixMod.Imatrix,
) !void {
    // --- Resolve output path -------------------------------------------------
    const dir_path = if (opts.output_dir) |od| od else std.fs.path.dirname(opts.path) orelse ".";

    const dir_result = try std.Io.Dir.cwd().createDirPathStatus(opts.io, dir_path, .default_dir);
    if (dir_result == .created) std.log.info("Created directory {s}", .{dir_path});

    const out_filename = try computeOutputPath(opts, false, arena_alloc);

    // -n can name the source back (gguf to gguf, -n matching the input stem).
    // The writer truncates before any tensor is read, so a miss here destroys
    // the source.
    if (pathsCollide(opts.io, out_filename, opts.path)) {
        std.log.err("output {s} is the source file; pass -n or -o", .{out_filename});
        return error.OutputWouldOverwriteSource;
    }

    // --- Initialise GGUF writer ----------------------------------------------
    var out_gguf = try gguf.init(out_filename, opts.io, allocator, arena_alloc, true);
    defer out_gguf.deinit();
    out_gguf.tensors = model_tensors;

    try buildGgufMetadata(&out_gguf.metadata, f, arch, template_metadata, extra_metadata, opts, arena_alloc, hf);

    if (imatrix) |im| out_gguf.imatrix = .{ .ctx = im, .get = &imatrixGet };

    // HF llama checkpoints store Q/K rows in interleaved RoPE order; gguf
    // expects half-split, so patch the source bytes before (de/re)quantizing.
    // `prepareConversion` decides which sources those are.
    var rope_patch: ?gguf.SourcePatch = null;
    if (hf_permute) |heads| {
        const ctx = try arena_alloc.create(RopePatchCtx);
        ctx.* = .{ .head_count = heads.head_count, .head_count_kv = heads.head_count_kv, .to_hf = false };
        rope_patch = .{ .ctx = ctx, .apply = &ropePatchApply, .matches = &ropePatchMatches };
    }
    // The Gated DeltaNet hybrid needs the other row rewrite instead: its V heads
    // arrive grouped under their K head and ggml broadcasts tiled. The two never
    // apply to one file - the rope permute is the llama family's.
    // Installed whatever vPerK is: the decay and norm-offset rewrites ride on
    // this patch too, and llama.cpp wants them even where the reorder is a no-op.
    if (hf) |m| if (m.qwen35) |q| {
        const ctx = try arena_alloc.create(VReorderCtx);
        ctx.* = .{
            .to_hf = false,
            .k_heads = q.group_count,
            .v_per_k = q.vPerK(),
            .qk_rows = 2 * @as(usize, q.state_size) * @as(usize, q.group_count),
            .value_head_dim = q.value_head_dim,
        };
        rope_patch = .{ .ctx = ctx, .apply = &vReorderApply, .matches = &vReorderMatches, .force_f32 = true };
    };

    out_gguf.saveWithSTData(f, opts.threads, opts.callbacks, groups, rope_patch) catch |err| {
        if (err == error.Cancelled) {
            std.Io.Dir.deleteFileAbsolute(opts.io, out_filename) catch {};
        }
        return err;
    };
    std.log.info("Converted to {s}", .{out_filename});
}

/// Where the mmproj for the text GGUF at `text_path` goes: beside it, under
/// llama.cpp's converter prefix. ComfyUI-GGUF finds it by the text file's stem.
pub fn mmprojPath(text_path: []const u8, alloc: std.mem.Allocator) ![]const u8 {
    const dir = std.fs.path.dirname(text_path) orelse ".";
    const name = try std.fmt.allocPrint(alloc, "mmproj-{s}", .{std.fs.path.basename(text_path)});
    return std.fs.path.join(alloc, &.{ dir, name });
}

/// The mmproj -H reads beside the text GGUF at `text_path`: mmprojPath's name
/// first, else the one .gguf there whose name holds "mmproj" and the text
/// file's stem less its quant suffix, which is how ComfyUI-GGUF pairs them.
/// null when there is none, or more than one candidate.
fn findMmproj(io: std.Io, text_path: []const u8, alloc: std.mem.Allocator) !?[]const u8 {
    const exact = try mmprojPath(text_path, alloc);
    if (std.Io.Dir.cwd().access(io, exact, .{})) |_| return exact else |_| {}
    const dir_path = std.fs.path.dirname(text_path) orelse ".";
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    const lower_stem = try std.ascii.allocLowerString(alloc, std.fs.path.stem(text_path));
    const quantless = stripQuantSuffixComfy(lower_stem);
    const stem = if (quantless.len < lower_stem.len) quantless else stripDtypeSuffix(lower_stem);
    const self_name = std.fs.path.basename(text_path);
    var found: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .file and e.kind != .sym_link) continue;
        if (std.mem.eql(u8, e.name, self_name)) continue;
        const lower = try std.ascii.allocLowerString(alloc, e.name);
        if (!std.mem.endsWith(u8, lower, ".gguf")) continue;
        if (std.mem.indexOf(u8, lower, "mmproj") == null or std.mem.indexOf(u8, lower, stem) == null) continue;
        try found.append(alloc, try alloc.dupe(u8, e.name));
    }
    if (found.items.len == 1) return try std.fs.path.join(alloc, &.{ dir_path, found.items[0] });
    if (found.items.len > 1) {
        std.log.warn("more than one mmproj beside {s} could be its vision tower: {s}", .{ text_path, try std.mem.join(alloc, ", ", found.items) });
        std.log.warn("rename the right one to {s}", .{exact});
    }
    return null;
}

/// ComfyUI-GGUF's strip_quant_suffix, `[-_]?(?:ud-)?i?q[0-9]_[a-z0-9_\-]{1,8}$`
/// on a lowercased stem, so -H pairs a text GGUF with the mmproj it would.
fn stripQuantSuffixComfy(s: []const u8) []const u8 {
    var start: usize = 0;
    while (start < s.len) : (start += 1) {
        var i = start;
        if (s[i] == '-' or s[i] == '_') i += 1;
        if (std.mem.startsWith(u8, s[i..], "ud-")) i += 3;
        if (i < s.len and s[i] == 'i') i += 1;
        if (i + 3 > s.len or s[i] != 'q' or !std.ascii.isDigit(s[i + 1]) or s[i + 2] != '_') continue;
        const tail = s[i + 3 ..];
        if (tail.len < 1 or tail.len > 8) continue;
        for (tail) |c| {
            if (!std.ascii.isLower(c) and !std.ascii.isDigit(c) and c != '_' and c != '-') break;
        } else return s[0..start];
    }
    return s;
}

test "stripQuantSuffixComfy strips what ComfyUI-GGUF's pattern strips" {
    try testing.expectEqualStrings("dumpling-qwen2.5-vl-7b.", stripQuantSuffixComfy("dumpling-qwen2.5-vl-7b.q4_k_s"));
    try testing.expectEqualStrings("qwen2.5-vl-7b-instruct", stripQuantSuffixComfy("qwen2.5-vl-7b-instruct-q4_k_m"));
    try testing.expectEqualStrings("model", stripQuantSuffixComfy("model-ud-iq2_xxs"));
    try testing.expectEqualStrings("qwen3vl", stripQuantSuffixComfy("qwen3vl-q8_0"));
    try testing.expectEqualStrings("qwen3vl-f16", stripQuantSuffixComfy("qwen3vl-f16"));
    // Nine characters after the type's underscore is past the pattern's reach.
    try testing.expectEqualStrings("x-q4_abcdefghi", stripQuantSuffixComfy("x-q4_abcdefghi"));
}

/// Map the vision tower before anything is written, so a tensor with no clip
/// name stops the conversion the way an unmapped text tensor does. null when
/// there is no tower to write, or one the mmproj writer cannot read.
fn planMmproj(m: HfLlm.Model, tensors: []const types.Tensor, opts: ConvertOptions, arena_alloc: std.mem.Allocator) !?HfLlm.VisionPlan {
    if (tensors.len == 0) return null;
    const v = m.vision orelse return null;
    for (tensors) |t| {
        const dt = types.DataType.fromString(t.type) catch null;
        if (dt == null or !dt.?.isFloatType()) {
            std.log.warn("{s} is {s}; the mmproj writer reads float vision towers only, so this writes the text model only", .{ t.name, t.type });
            return null;
        }
    }
    const plan = try HfLlm.planVision(arena_alloc, v, tensors);
    if (plan.unmapped.len > 0) {
        const joined = try std.mem.join(arena_alloc, ", ", plan.unmapped);
        std.log.warn("{} vision tensors have no name in llama.cpp's clip vocabulary: {s}", .{ plan.unmapped.len, joined });
        if (!opts.allow_unknown_arch) {
            std.log.warn("converting would write an mmproj missing them; pass --allow-unknown-arch to drop them anyway", .{});
            return error.UnmappedTensors;
        }
        std.log.warn("dropping them: --allow-unknown-arch was given", .{});
    }
    return plan;
}

/// The vision tower as llama.cpp's separate mmproj GGUF, in the file type its
/// converter would choose for this target (see HfLlm.mmprojFileType).
const MmprojOutput = struct {
    tensors: std.ArrayList(types.Tensor),
    metadata: std.json.ObjectMap,
    path: []const u8,
};

/// The mmproj as it will be written: typed and laid-out tensors, metadata and
/// path. Shared by the writer and the size predictor so -c counts the same file.
fn mmprojOutput(plan: HfLlm.VisionPlan, m: HfLlm.Model, opts: ConvertOptions, arena_alloc: std.mem.Allocator) !MmprojOutput {
    const ftype = HfLlm.mmprojFileType(opts.datatype);
    var tensors: std.ArrayList(types.Tensor) = .empty;
    try tensors.appendSlice(arena_alloc, plan.outputs);
    for (tensors.items) |*t| {
        t.type = @tagName(HfLlm.mmprojTensorType(t.name, t.dims, ftype));
        t.size = try gguf.calculateTensorSize(t.*);
    }
    std.sort.block(types.Tensor, tensors.items, {}, struct {
        fn lessThan(_: void, a: types.Tensor, b: types.Tensor) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);
    var gguf_opts = opts;
    gguf_opts.filetype = .gguf;
    try assignOutputLayout(&tensors, gguf_opts, arena_alloc);

    var metadata: std.json.ObjectMap = .empty;
    try HfLlm.addVisionMetadata(m, &metadata, arena_alloc);
    try metadata.put(arena_alloc, "general.file_type", .{ .integer = ggufFileType(ftype) });
    try metadata.put(arena_alloc, "general.quantization_version", .{ .integer = 2 });
    try stampConverterProvenance(&metadata, arena_alloc);

    return .{
        .tensors = tensors,
        .metadata = metadata,
        .path = try mmprojPath(try computeOutputPath(opts, false, arena_alloc), arena_alloc),
    };
}

fn writeMmproj(
    f: anytype,
    plan: HfLlm.VisionPlan,
    m: HfLlm.Model,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !void {
    const mm = try mmprojOutput(plan, m, opts, arena_alloc);
    const out_path = mm.path;
    if (pathsCollide(opts.io, out_path, opts.path)) {
        std.log.err("mmproj output {s} is the source file; pass -n or -o", .{out_path});
        return error.OutputWouldOverwriteSource;
    }
    var out = try gguf.init(out_path, opts.io, allocator, arena_alloc, true);
    defer out.deinit();
    out.tensors = mm.tensors;
    out.metadata = mm.metadata;

    const groups = TensorClusters.GroupResult{
        .fp4_clusters = &.{},
        .float8_clusters = &.{},
        .mxfp4_clusters = &.{},
        .mxfp8_clusters = &.{},
        .int8_convrot_clusters = &.{},
        .int4_clusters = &.{},
        .asym_w4a8_clusters = &.{},
        .expert_stacks = plan.stacks,
    };
    // The writer finds each tensor's bytes by name in the source handle, so it
    // reads the tower's own list for this pass.
    const text_tensors = f.tensors;
    defer f.tensors = text_tensors;
    f.tensors = .empty;
    try f.tensors.appendSlice(arena_alloc, plan.sources);

    out.saveWithSTData(f, opts.threads, opts.callbacks, &groups, null) catch |err| {
        if (err == error.Cancelled) std.Io.Dir.deleteFileAbsolute(opts.io, out_path) catch {};
        return err;
    };
    std.log.info("Wrote the vision tower ({s}) to {s}", .{ @tagName(HfLlm.mmprojFileType(opts.datatype)), out_path });
}

/// Assemble the final SafeTensors `__metadata__` map (cloned source metadata, merged
/// template/source keys, extra records, dropped source quant header, stamped
/// provenance) into `metadata`. Shared by the real writer and the size predictor.
fn buildSafetensorsMetadata(
    metadata: *?std.json.ObjectMap,
    f: anytype,
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.json.ObjectMap,
    skip_source_meta: bool,
    arena_alloc: std.mem.Allocator,
) !void {
    // Copy any metadata from the source, if there is any. GGUF LLM metadata is
    // typed; the safetensors header stringifies it, so an HF-layout output
    // must not carry it forward.
    metadata.* = if (skip_source_meta) null else if (f.getSourceMetadata()) |meta| try meta.clone(arena_alloc) else null;
    if (metadata.* == null) metadata.* = std.json.ObjectMap.empty;

    // Template metadata takes priority over source-file metadata.
    if (template_metadata) |meta| {
        var it = meta.iterator();
        while (it.next()) |entry| {
            if (!metadata.*.?.contains(entry.key_ptr.*))
                try metadata.*.?.put(arena_alloc, try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }
    } else if (!skip_source_meta) {
        if (f.getSourceMetadata()) |meta| {
            var it = meta.iterator();
            while (it.next()) |entry| {
                if (!metadata.*.?.contains(entry.key_ptr.*))
                    try metadata.*.?.put(arena_alloc, try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
            }
        }
    }

    // Extra metadata
    var extra_it = extra_metadata.iterator();
    while (extra_it.next()) |entry| {
        if (!metadata.*.?.contains(entry.key_ptr.*))
            try metadata.*.?.put(arena_alloc, try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
    }

    // Drop the source quantization header. It describes the pre-conversion layout, so
    // leaving it would make ComfyUI load the dequantized output as a different format and fail.
    // Provenance is stamped on every output, so the map exists even without source metadata.
    if (metadata.* == null) metadata.* = std.json.ObjectMap.empty;
    if (metadata.*) |*m| {
        _ = m.swapRemove(source_quant_metadata_key);
        // transformers and huggingface_hub refuse a header that has a
        // __metadata__ map whose "format" they don't recognise, and the
        // provenance stamp means every output now has the map. A source that
        // names its own format keeps it.
        if (!m.contains("format")) try m.put(arena_alloc, "format", .{ .string = "pt" });
        try stampConverterProvenance(m, arena_alloc);
    }
}

/// Whether two paths name the same file. Both sides are built by joining a
/// directory onto a name, so "model.safetensors" and "./model.safetensors"
/// arrive as different strings for the same file; the output does not exist
/// yet, so the directories are canonicalized and the basenames compared.
pub fn pathsCollide(io: std.Io, a: []const u8, b: []const u8) bool {
    if (!std.mem.eql(u8, std.fs.path.basename(a), std.fs.path.basename(b))) return false;
    const dir_a = std.fs.path.dirname(a) orelse ".";
    const dir_b = std.fs.path.dirname(b) orelse ".";
    var buf_a: [std.fs.max_path_bytes]u8 = undefined;
    var buf_b: [std.fs.max_path_bytes]u8 = undefined;
    const len_a = std.Io.Dir.cwd().realPathFile(io, dir_a, &buf_a) catch
        return std.mem.eql(u8, dir_a, dir_b);
    const len_b = std.Io.Dir.cwd().realPathFile(io, dir_b, &buf_b) catch
        return std.mem.eql(u8, dir_a, dir_b);
    return std.mem.eql(u8, buf_a[0..len_a], buf_b[0..len_b]);
}

fn writeSafetensors(
    f: anytype,
    model_tensors: std.ArrayList(types.Tensor),
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.json.ObjectMap,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
    groups: *const TensorClusters.GroupResult,
    hf_renamed: bool,
    hf_unpermute: ?HfLlm.RopeHeads,
    hf_vision: ?HfLlm.Vision,
) !void {
    // --- Resolve output path -------------------------------------------------
    const dir_path = if (opts.output_dir) |od| od else std.fs.path.dirname(opts.path) orelse ".";

    const dir_result2 = try std.Io.Dir.cwd().createDirPathStatus(opts.io, dir_path, .default_dir);
    if (dir_result2 == .created) std.log.info("Created directory {s}", .{dir_path});

    // transformers loads model.safetensors (or an index naming it) and nothing
    // else, so the HF layout has one usable weights name: a <stem>-<dtype> file
    // beside the sidecars would be a directory that only looks loadable.
    const out_filename = try computeOutputPath(opts, hf_renamed, arena_alloc);

    // A fixed output name can land on the source itself (a safetensors already
    // named model.safetensors, converted in place). The writer truncates before
    // any tensor is read, so a miss here destroys the source.
    if (pathsCollide(opts.io, out_filename, opts.path)) {
        std.log.err("output {s} is the source file; pass -n or -o", .{out_filename});
        return error.OutputWouldOverwriteSource;
    }

    // --- HF sidecar directory -------------------------------------------------
    // Sidecar names are fixed, so with no -o they land on the source directory's
    // own config.json etc. Settle that here, before the output file exists.
    var sidecar_dir: ?std.Io.Dir = null;
    defer if (sidecar_dir) |d| d.close(opts.io);
    if (hf_renamed) {
        const md = f.getSourceMetadata() orelse std.json.ObjectMap.empty;
        // The sidecars are written after the weights, which are the gigabytes,
        // so what the vocabulary needs is settled here instead: otherwise a
        // tokenizer that cannot be rebuilt only surfaces over a finished
        // model.safetensors and half a layout, and the retry wants --force.
        HfLlm.sidecarPrecheck(md) catch |e| {
            switch (e) {
                error.UnknownPretokenizer => {},
                error.MissingArchitecture => {
                    std.log.warn("{s} has no general.architecture; there is nothing to name config.json's model type after", .{opts.path});
                    return e;
                },
                error.HfConfigUnsupported => {
                    const src_arch = sourceMetaStr(f, "general.architecture") orelse "?";
                    std.log.warn("{s} is a {s} GGUF, and this tool writes config.json for llama, qwen2, qwen3 and qwen35 only", .{ opts.path, src_arch });
                    return e;
                },
                error.MissingVocabulary => {
                    std.log.warn("{s} has no tokenizer.ggml.* metadata; there is no vocabulary to write beside the weights", .{opts.path});
                    return e;
                },
                error.MissingDimensions => {
                    const src_arch = sourceMetaStr(f, "general.architecture") orelse "?";
                    std.log.warn("{s} has no {s}.block_count/embedding_length/attention.head_count; config.json would describe a model of no layers and no width", .{ opts.path, src_arch });
                    return e;
                },
                error.MissingMerges => {
                    std.log.warn("{s} has a BPE vocabulary but no tokenizer.ggml.merges; tokenizer.json would merge nothing", .{opts.path});
                    std.log.warn("every prompt would tokenize differently from the model these weights came from", .{});
                    return e;
                },
            }
            const pre = sourceMetaStr(f, "tokenizer.ggml.pre") orelse "default";
            std.log.warn("no tokenizer.json reproduces the splitting llama.cpp's \"{s}\" pre-tokenizer applies", .{pre});
            if (!opts.allow_unknown_arch) {
                std.log.warn("the HF layout would carry a vocabulary that segments text differently from {s}", .{opts.path});
                std.log.warn("pass --allow-unknown-arch to write the layout without a tokenizer.json", .{});
                return e;
            }
            std.log.warn("leaving tokenizer.json out: --allow-unknown-arch was given, so AutoTokenizer will not load {s}", .{dir_path});
        };
        const d = try std.Io.Dir.cwd().openDir(opts.io, dir_path, .{});
        sidecar_dir = d;
        // The weights file clashes like a sidecar: model.safetensors is a
        // fixed name, so with no -o it lands on whatever model already sits
        // in the directory, and the writer truncates before reading a tensor.
        var clashes: std.ArrayList([]const u8) = .empty;
        if (!opts.force) {
            try clashes.appendSlice(arena_alloc, try HfLlm.existingSidecars(opts.io, d, md, std.fs.path.basename(out_filename), arena_alloc));
            if (hf_vision != null) for ([_][]const u8{ "preprocessor_config.json", "chat_template.jinja" }) |n| {
                if (d.access(opts.io, n, .{})) |_| try clashes.append(arena_alloc, n) else |_| {}
            };
        }
        if (clashes.items.len > 0) {
            std.log.err("HF layout files already in {s}: {s}", .{ dir_path, try std.mem.join(arena_alloc, ", ", clashes.items) });
            std.log.err("pass --force to overwrite them, or -o to write into another directory", .{});
            return error.SidecarExists;
        }
    }

    // --- Initialise Safetensors writer ----------------------------------------------
    var out_st = try st.init(out_filename, opts.io, allocator, arena_alloc, true, true);
    defer out_st.deinit();
    out_st.tensors = model_tensors;

    try buildSafetensorsMetadata(&out_st.metadata, f, template_metadata, extra_metadata, hf_renamed, arena_alloc);

    const sr_seed: u64 = opts.stochastic_rounding orelse TensorClusters.default_stochastic_seed;
    var rope_patch: ?gguf.SourcePatch = null;
    if (hf_unpermute) |heads| {
        const ctx = try arena_alloc.create(RopePatchCtx);
        ctx.* = .{ .head_count = heads.head_count, .head_count_kv = heads.head_count_kv, .to_hf = true };
        rope_patch = .{ .ctx = ctx, .apply = &ropePatchApply, .matches = &ropePatchMatches };
    }
    // The hybrid undoes its own rewrites instead: tiled V heads back to grouped,
    // the decay back to its log, the norm offset back off. The rope permute is
    // the llama family's and the two never apply to one file.
    if (hf_renamed) if (sourceArchName(f)) |src_arch| {
        if (std.mem.eql(u8, src_arch, "qwen35")) {
            if (qwen35CtxFromSource(f, src_arch, arena_alloc)) |c| {
                const ctx = try arena_alloc.create(VReorderCtx);
                ctx.* = c;
                rope_patch = .{ .ctx = ctx, .apply = &vReorderApply, .matches = &vReorderMatches, .force_f32 = true };
            } else {
                std.log.err("--hf-names needs {s}.ssm.* in the source metadata to undo the V-head tiling", .{src_arch});
                return error.HfNamesUnsupported;
            }
        }
    };
    out_st.saveWithSTData(f, opts.threads, opts.callbacks, groups, sr_seed, rope_patch) catch |err| {
        if (err == error.Cancelled) {
            std.Io.Dir.deleteFileAbsolute(opts.io, out_filename) catch {};
        }
        return err;
    };
    std.log.info("Converted to {s}", .{out_filename});

    // HF layout needs its sidecars too; without them the safetensors are not
    // loadable by transformers. All values come from the source GGUF metadata.
    if (sidecar_dir) |out_dir| {
        const md = f.getSourceMetadata() orelse std.json.ObjectMap.empty;
        var names = try std.ArrayList([]const u8).initCapacity(arena_alloc, model_tensors.items.len);
        var total_size: u64 = 0;
        for (model_tensors.items) |t| {
            names.appendAssumeCapacity(t.name);
            total_size += t.size;
        }
        const torch_dtype = try HfLlm.dominantTorchDtype(model_tensors.items, arena_alloc);
        // writeSidecars writes config.json first and the tokenizer last, so a
        // failure part-way leaves a directory from_pretrained cannot open. Say
        // so and fail: reporting success over half a layout sends the user off
        // to debug transformers instead of the missing vocabulary.
        HfLlm.writeSidecars(opts.io, out_dir, md, names.items, torch_dtype, true, opts.allow_unknown_arch, hf_vision, arena_alloc) catch |e| {
            std.log.err("HF sidecars failed: {s}", .{@errorName(e)});
            switch (e) {
                error.MissingVocabulary => std.log.err("{s} has no tokenizer.ggml.* metadata; there is no vocabulary to write", .{opts.path}),
                error.MissingArchitecture => std.log.err("{s} has no general.architecture to name config.json's model type after", .{opts.path}),
                error.MissingMerges => std.log.err("{s} has a BPE vocabulary but no tokenizer.ggml.merges to rebuild tokenizer.json from", .{opts.path}),
                else => {},
            }
            std.log.err("{s} holds the weights but is not loadable by from_pretrained", .{out_filename});
            return e;
        };
        // -n moved the weights off the one name transformers opens by itself.
        const weights_name = std.fs.path.basename(out_filename);
        if (!std.mem.eql(u8, weights_name, HfLlm.default_weights_name)) {
            // from_pretrained opens model.safetensors or an index naming the
            // file; without the index this directory is as unloadable as one
            // missing a sidecar, so it fails the same way rather than warning.
            HfLlm.writeWeightsIndex(opts.io, out_dir, weights_name, names.items, total_size, arena_alloc) catch |e| {
                std.log.err("{s} failed: {s}", .{ HfLlm.weights_index_name, @errorName(e) });
                std.log.err("{s} holds the weights but is not loadable by from_pretrained without it", .{out_filename});
                return e;
            };
        } else {
            // An index outranks model.safetensors when transformers loads, so one
            // left over from an earlier -n run in this directory sends it to the
            // old weights. Without --force existingSidecars already refused; with
            // it, the overwrite has to cover the index too.
            out_dir.deleteFile(opts.io, HfLlm.weights_index_name) catch {};
        }
    }
}

/// Filter and name-strip a tensor list using architecture rules.
/// Works on any already-loaded tensor slice — no SafeTensors object required.
fn filterTensorsForExport(
    tensors: []const types.Tensor,
    arch_opt: ?*const imagearch.Arch,
    arena_alloc: std.mem.Allocator,
) !std.ArrayList(types.Tensor) {
    var result = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, tensors.len);

    var has_model_prefix = false;
    for (tensors) |t| {
        if (std.mem.startsWith(u8, t.name, "model.")) {
            has_model_prefix = true;
            break;
        }
    }

    for (tensors) |t| {
        if (arch_opt) |a| {
            if (has_model_prefix and !std.mem.startsWith(u8, t.name, "model.")) continue;
            if (a.shouldIgnore(t.name)) continue;
        }
        var duped = try t.dupe(arena_alloc);
        duped.name = try arena_alloc.dupe(u8, imagearch.stripPrefix(duped.name));
        try result.append(arena_alloc, duped);
    }
    return result;
}

/// Write a JSON template from a tensor list.
/// Pass `reverse_dims = true` for SafeTensors source files so that
/// applyTemplate (which always un-reverses dimensions) round-trips correctly.
/// GGUF tensor structs already store dims in GGUF (reversed) order — pass false.
/// Generate a JSON template from an open source file (`Safetensors`/`Gguf`), collapsing
/// ComfyUI cluster layouts (`.weight` + `.weight_scale` + `.comfy_quant`) into a single
/// logical entry carrying the cluster's quant type (INT8_CONVROT, NVFP4, ...). This is
/// what makes a template round-trip: without it, a template exported from a cluster model
/// lists the raw sub-tensors as plain `I8`/`F32`/`U8`, which `convert` cannot re-ingest.
///
/// `f` must be an opened source file object (exposing `tensors`, `openFileForTensor`,
/// `getSourceMetadata`) so cluster detection can read the `.comfy_quant` markers.
pub fn writeTemplateFromFile(
    f: anytype,
    arch_opt: ?*const imagearch.Arch,
    reverse_dims: bool,
    writer: *std.Io.Writer,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !void {
    var tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, f.tensors.items.len);
    for (f.tensors.items) |t| try tensors.append(arena_alloc, t);

    var groups = try TensorClusters.groupClusters(f, arena_alloc, allocator);
    try TensorClusters.collapseModelTensors(&tensors, &groups, .preserve_quant, arena_alloc);

    try writeTemplateFromTensors(tensors.items, arch_opt, reverse_dims, writer, arena_alloc);
}

pub fn writeTemplateFromTensors(
    tensors: []const types.Tensor,
    arch_opt: ?*const imagearch.Arch,
    reverse_dims: bool,
    writer: *std.Io.Writer,
    arena_alloc: std.mem.Allocator,
) !void {
    const filtered = try filterTensorsForExport(tensors, arch_opt, arena_alloc);

    var tensors_obj: std.json.ObjectMap = .empty;
    defer tensors_obj.deinit(arena_alloc);

    for (filtered.items) |t| {
        var t_obj: std.json.ObjectMap = .empty;
        errdefer t_obj.deinit(arena_alloc);

        var shape_arr = std.json.Array.init(arena_alloc);
        errdefer shape_arr.deinit();
        if (reverse_dims) {
            var i: usize = t.dims.len;
            while (i > 0) {
                i -= 1;
                try shape_arr.append(.{ .integer = @intCast(t.dims[i]) });
            }
        } else {
            for (t.dims) |d| try shape_arr.append(.{ .integer = @intCast(d) });
        }
        try t_obj.put(arena_alloc, "shape", .{ .array = shape_arr });
        try t_obj.put(arena_alloc, "type", .{ .string = t.type });
        try tensors_obj.put(arena_alloc, try arena_alloc.dupe(u8, t.name), .{ .object = t_obj });
    }

    var root_obj: std.json.ObjectMap = .empty;
    defer root_obj.deinit(arena_alloc);
    try root_obj.put(arena_alloc, "tensors", .{ .object = tensors_obj });

    var stringifier = std.json.Stringify{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
    try stringifier.write(std.json.Value{ .object = root_obj });
    _ = root_obj.swapRemove("tensors");
}

fn safetensorTypePrecision(type_str: []const u8) u8 {
    var buf: [12]u8 = [_]u8{0} ** 12;
    const l = std.ascii.lowerString(&buf, type_str[0..@min(type_str.len, buf.len)]);
    if (std.mem.eql(u8, l, "f8_e4m3") or std.mem.eql(u8, l, "f8_e5m2")) return 0;
    if (std.mem.eql(u8, l, "f16") or std.mem.eql(u8, l, "fp16")) return 1;
    if (std.mem.eql(u8, l, "bf16")) return 2;
    if (std.mem.eql(u8, l, "f32") or std.mem.eql(u8, l, "fp32")) return 3;
    if (std.mem.eql(u8, l, "f64") or std.mem.eql(u8, l, "fp64")) return 4;
    return 1;
}

fn safetensorDisplayType(type_str: []const u8) []const u8 {
    var buf: [12]u8 = [_]u8{0} ** 12;
    const l = std.ascii.lowerString(&buf, type_str[0..@min(type_str.len, buf.len)]);
    if (std.mem.eql(u8, l, "f8_e4m3") or std.mem.eql(u8, l, "f8_e5m2")) return "FP8";
    return type_str;
}

/// Derive the filename type-suffix for a template-based conversion.
/// For GGUF: the lowest (most quantized) QuantizationLevel present.
/// For SafeTensors: the lowest-precision type; appends "-MIXED" when multiple
/// distinct types are present. Both FP8 variants display as "FP8".
pub fn templateTypeSuffix(
    io: std.Io,
    template_path: []const u8,
    filetype: types.FileType,
    arena_alloc: std.mem.Allocator,
) ![]const u8 {
    const t_file = try std.Io.Dir.cwd().openFile(io, template_path, .{ .mode = .read_only });
    defer t_file.close(io);
    var t_reader_buf: [8192]u8 = undefined;
    var t_reader = t_file.reader(io, &t_reader_buf);
    const content = try t_reader.interface.allocRemaining(arena_alloc, .unlimited);
    const parsed = try std.json.parseFromSlice(std.json.Value, arena_alloc, content, .{});

    const tensors_val = parsed.value.object.get("tensors") orelse return error.InvalidTemplate;
    var it = tensors_val.object.iterator();

    switch (filetype) {
        .gguf => {
            var min_level: ?u8 = null;
            var min_str: []const u8 = "f16";
            while (it.next()) |entry| {
                const type_val = entry.value_ptr.object.get("type") orelse continue;
                const level = QuantizationLevel.fromString(type_val.string) catch continue;
                const lv: u8 = @intFromEnum(level);
                if (min_level == null or lv < min_level.?) {
                    min_level = lv;
                    min_str = type_val.string;
                }
            }
            return min_str;
        },
        .safetensors => {
            var min_precision: ?u8 = null;
            var min_display: []const u8 = "F16";
            var seen: [16][]const u8 = undefined;
            var seen_count: usize = 0;
            while (it.next()) |entry| {
                const type_val = entry.value_ptr.object.get("type") orelse continue;
                const display = safetensorDisplayType(type_val.string);
                var already = false;
                for (seen[0..seen_count]) |s| {
                    if (std.mem.eql(u8, s, display)) {
                        already = true;
                        break;
                    }
                }
                if (!already and seen_count < seen.len) {
                    seen[seen_count] = display;
                    seen_count += 1;
                }
                const prec = safetensorTypePrecision(type_val.string);
                if (min_precision == null or prec < min_precision.?) {
                    min_precision = prec;
                    min_display = display;
                }
            }
            if (seen_count > 1) return try std.fmt.allocPrint(arena_alloc, "{s}-MIXED", .{min_display});
            return min_display;
        },
    }
}

/// Generate a sensitivities JSON file from a tensor list.
/// Only includes tensors that would actually be quantized:
/// not architecture-ignored, not too small, not high-precision, not upcast-forced.
/// All values are initialised to 50.0 (neutral sensitivity).
pub fn generateSensitivitiesFromTensors(
    tensors: []const types.Tensor,
    arch_opt: ?*const imagearch.Arch,
    threshold: u64,
    writer: *std.Io.Writer,
    arena_alloc: std.mem.Allocator,
) !void {
    const filtered = try filterTensorsForExport(tensors, arch_opt, arena_alloc);

    var sens_obj: std.json.ObjectMap = .empty;
    defer sens_obj.deinit(arena_alloc);

    for (filtered.items) |t| {
        var n_elements: u64 = 1;
        for (t.dims) |d| n_elements *= d;
        if (n_elements < threshold) continue;
        if (arch_opt) |a| {
            if (a.isHighPrecision(t.name)) continue;
            if (a.shouldUpcast(t.name)) continue;
        }
        try sens_obj.put(arena_alloc, try arena_alloc.dupe(u8, t.name), .{ .float = 50.0 });
    }

    var stringifier = std.json.Stringify{ .writer = writer, .options = .{ .whitespace = .indent_2 } };
    try stringifier.write(std.json.Value{ .object = sens_obj });
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

test "restoreOrigShapes: restores dims from orig_shape metadata, leaves others" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Source metadata as produced by applyShapeFix on a prior GGUF write.
    var meta: std.json.ObjectMap = .empty;
    var shape_arr = std.json.Array.init(a);
    try shape_arr.append(.{ .integer = 6144 });
    try shape_arr.append(.{ .integer = 64 });
    try meta.put(a, "comfy.gguf.orig_shape.first.weight", .{ .array = shape_arr });

    var dims_fixed = [_]usize{ 1536, 256 }; // rearranged form stored in the GGUF
    var dims_plain = [_]usize{ 6144, 6144 }; // no orig_shape entry
    var model_tensors: std.ArrayList(types.Tensor) = .empty;
    try model_tensors.append(a, .{ .name = "first.weight", .type = "bf16", .dims = &dims_fixed, .size = 0, .offset = 0 });
    try model_tensors.append(a, .{ .name = "blocks.0.attn.wq.weight", .type = "bf16", .dims = &dims_plain, .size = 0, .offset = 0 });

    try restoreOrigShapes(&model_tensors, meta, a);

    try testing.expectEqualSlices(usize, &.{ 6144, 64 }, model_tensors.items[0].dims);
    try testing.expectEqualSlices(usize, &.{ 6144, 6144 }, model_tensors.items[1].dims);
}

test "restoreOrigShapes: no-op when source has no metadata" {
    const a = testing.allocator;
    var dims = [_]usize{ 1536, 256 };
    var model_tensors: std.ArrayList(types.Tensor) = .empty;
    defer model_tensors.deinit(a);
    try model_tensors.append(a, .{ .name = "first.weight", .type = "bf16", .dims = &dims, .size = 0, .offset = 0 });

    try restoreOrigShapes(&model_tensors, null, a);

    try testing.expectEqualSlices(usize, &.{ 1536, 256 }, model_tensors.items[0].dims);
}

/// Minimal ConvertOptions for exercising assignTensorType in tests.
fn testOpts(datatype: types.DataType) ConvertOptions {
    return .{
        .io = undefined, // unused by assignTensorType
        .path = "",
        .filetype = .gguf,
        .datatype = datatype,
        .template_path = null,
        .output_dir = null,
        .output_name = null,
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
}

test "assignOutputLayout: contiguous offsets and full cluster size for a template-typed cluster" {
    const a = testing.allocator;

    // Mimics the template path: a cluster dest type whose size was NOT pre-expanded
    // (applyTemplateEntry leaves it at the packed-weight size / zero), plus plain tensors
    // whose sizes are already correct. assignOutputLayout must set the full cluster size
    // and lay every tensor out contiguously from offset 0.
    var dims_bf = [_]usize{ 4, 4 };
    var dims_cluster = [_]usize{ 2048, 1024 };
    var dims_bias = [_]usize{10};
    var model: std.ArrayList(types.Tensor) = .empty;
    defer model.deinit(a);
    try model.append(a, .{ .name = "a.weight", .type = "BF16", .dims = &dims_bf, .size = 32, .offset = 999 });
    try model.append(a, .{ .name = "b.attn.q.weight", .type = "INT8_CONVROT", .dims = &dims_cluster, .size = 0, .offset = 999 });
    try model.append(a, .{ .name = "c.bias", .type = "F32", .dims = &dims_bias, .size = 40, .offset = 999 });

    var opts = testOpts(.F16);
    opts.filetype = .safetensors;

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try assignOutputLayout(&model, opts, arena.allocator());

    // Full int8_convrot cluster size = I8 weight + F32 per-row scale + comfy_quant JSON.
    const expected_cluster_size: u64 = 2048 * 1024 + 2048 * 4 + TensorClusters.int8_convrot_comfy_json.len;
    try testing.expectEqual(expected_cluster_size, model.items[1].size);

    // Offsets are a fresh, contiguous layout starting at 0 (source offsets discarded).
    try testing.expectEqual(@as(u64, 0), model.items[0].offset);
    try testing.expectEqual(@as(u64, 32), model.items[1].offset);
    try testing.expectEqual(@as(u64, 32 + expected_cluster_size), model.items[2].offset);
}

test "assignTensorType: 1D tensors are never block-quantized" {
    const opts = testOpts(.q8_0);
    // 1D and larger than the small-tensor threshold: only the 1D rule can keep it float.
    var dims = [_]usize{200000};
    var t = types.Tensor{ .name = "blocks.0.mod.lin", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, 200000, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    // BF16 -> f32 via nearestCompatibleType; crucially NOT q8_0.
    try testing.expectEqualStrings("f32", t.type);
}

test "assignTensorType: a rank-5 conv is f16 in a GGUF whatever the target" {
    var dims = [_]usize{ 64, 3, 2, 16, 16 };
    for ([_]types.DataType{ .bf16, .q8_0, .q4_k, .f16 }) |dt| {
        var t = types.Tensor{ .name = "model.visual.patch_embed.proj.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
        try assignTensorType(&t, 64 * 3 * 2 * 16 * 16, &imagearch.qwen3vl, QUANTIZATION_THRESHOLD, testOpts(dt), false, null, testing.allocator);
        try testing.expectEqualStrings("f16", t.type);
    }
    var t = types.Tensor{ .name = "model.visual.patch_embed.proj.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, 64 * 3 * 2 * 16 * 16, &imagearch.qwen3vl, QUANTIZATION_THRESHOLD, testOpts(.f32), false, null, testing.allocator);
    try testing.expectEqualStrings("f32", t.type);
}

test "assignTensorType: small 2D tensors are kept float" {
    const opts = testOpts(.q8_0);
    // last.modulation.lin [2, 6144] = 12288 elements, below the threshold.
    var dims = [_]usize{ 2, 6144 };
    var t = types.Tensor{ .name = "last.modulation.lin", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, 12288, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    try testing.expectEqualStrings("f32", t.type);
}

test "assignTensorType: large 2D weights are quantized" {
    const opts = testOpts(.q8_0);
    var dims = [_]usize{ 6144, 6144 }; // ~37.7M elements
    const n: u64 = 6144 * 6144;
    var t = types.Tensor{ .name = "blocks.0.attn.wq.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, n, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    try testing.expectEqualStrings("q8_0", t.type);
}

test "assignTensorType: minimax_h3 pruned adaln_proj is kept float, full-width is quantized" {
    const opts = testOpts(.q4_k);
    // 96768 * 8 = 774144 elements: 2D, well above the threshold, and divisible by
    // the q4_k block size, so only the narrow rule can keep it float.
    var curve = [_]usize{ 96768, 8 };
    var t = types.Tensor{ .name = "blocks.0.adaln_proj.linear.weight", .type = "F16", .dims = &curve, .size = 0, .offset = 0 };
    try assignTensorType(&t, 96768 * 8, &imagearch.minimax_h3, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    try testing.expectEqualStrings("F16", t.type);

    // The unpruned form is 40% of the model's parameters and must still quantize.
    var full = [_]usize{ 96768, 2688 };
    var t2 = types.Tensor{ .name = "blocks.0.adaln_proj.linear.weight", .type = "BF16", .dims = &full, .size = 0, .offset = 0 };
    try assignTensorType(&t2, 96768 * 2688, &imagearch.minimax_h3, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    try testing.expectEqualStrings("q4_k", t2.type);
}

test "assignTensorType: minimax_h3 backbone linears quantize, conditioning path does not" {
    const opts = testOpts(.q4_k);
    var dims = [_]usize{ 28672, 5376 };
    const n: u64 = 28672 * 5376;
    var t = types.Tensor{ .name = "blocks.0.mlp.fc1.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, n, &imagearch.minimax_h3, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    try testing.expectEqualStrings("q4_k", t.type);

    // Same shape, inside the token refiner: protected.
    var t2 = types.Tensor{ .name = "token_refiner.blocks.0.mlp.fc1.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t2, n, &imagearch.minimax_h3, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    try testing.expectEqualStrings("f32", t2.type);
}

test "isEmbeddingWeight: matches embedding tables, not embedder projections" {
    // nn.Embedding lookup tables — must be protected.
    try testing.expect(isEmbeddingWeight("model.diffusion_model.llm_adapter.embed.weight"));
    try testing.expect(isEmbeddingWeight("text_model.embed_tokens.weight"));
    try testing.expect(isEmbeddingWeight("transformer.wte.weight"));
    try testing.expect(isEmbeddingWeight("foo.token_embedding.weight"));
    // Linear/Conv "embedder" projections and unrelated weights — must NOT match.
    try testing.expect(!isEmbeddingWeight("blocks.0.pos_embedder.weight"));
    try testing.expect(!isEmbeddingWeight("x_embedder.proj.weight"));
    try testing.expect(!isEmbeddingWeight("patch_embed.proj.weight"));
    try testing.expect(!isEmbeddingWeight("add_embedding.linear_1.weight"));
    try testing.expect(!isEmbeddingWeight("blocks.0.attn.wq.weight"));
}

test "assignTensorType: embedding tables are never quantized (gguf)" {
    const opts = testOpts(.q8_0);
    // Anima's T5 embedding table: 2D, well above threshold — only the embedding
    // rule can keep it float (vs. the 1D and small-tensor rules).
    var dims = [_]usize{ 32128, 1024 };
    const n: u64 = 32128 * 1024;
    var t = types.Tensor{ .name = "model.diffusion_model.llm_adapter.embed.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, n, &imagearch.anima, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    // BF16 -> f32 via nearestCompatibleType; crucially NOT q8_0.
    try testing.expectEqualStrings("f32", t.type);
}

test "assignTensorType: embedding tables are never clustered (safetensors INT8_CONVROT)" {
    var opts = testOpts(.INT8_CONVROT);
    opts.filetype = .safetensors;
    // 1024 % int8_convrot_group_size == 0, so absent the embedding rule this
    // would pass clusterEligible and be INT8-quantized (the reported bug).
    var dims = [_]usize{ 32128, 1024 };
    const n: u64 = 32128 * 1024;
    var t = types.Tensor{ .name = "model.diffusion_model.llm_adapter.embed.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    try assignTensorType(&t, n, &imagearch.anima, QUANTIZATION_THRESHOLD, opts, false, null, std.testing.allocator);
    // Left in source precision, NOT INT8_CONVROT.
    try testing.expectEqualStrings("BF16", t.type);
}

/// Write a minimal two-tensor SafeTensors file for the size-prediction test. Data is a
/// constant non-zero byte pattern (finite, non-zero) so cluster/quant scale computation
/// never hits a divide-by-zero on all-zero input.
fn writeTestSafetensors(io: std.Io, path: []const u8, a: std.mem.Allocator) !void {
    const header =
        \\{"unet.foo.weight":{"dtype":"BF16","shape":[512,512],"data_offsets":[0,524288]},"unet.bar.weight":{"dtype":"F32","shape":[256],"data_offsets":[524288,525312]}}
    ;
    const data = try a.alloc(u8, 525312);
    @memset(data, 0x3C); // finite, non-zero for both BF16 and F32 interpretations

    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var fw = file.writer(io, &buf);
    const w = &fw.interface;
    try w.writeInt(u64, header.len, .little);
    try w.writeAll(header);
    try w.writeAll(data);
    try w.flush();
}

test "predictOutputSize matches the actual written file size" {
    // The synthetic model has no recognizable architecture, so conversion logs an expected
    // "unknown architecture" warning. Silence it: a passing test must not write to stderr.
    testing.log_level = .err;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run: the test suite runs multiple binaries in parallel, all of
    // which include this test, so a shared path would race (one deletes it mid-use).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_predict_size_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const src_path = try std.fmt.allocPrint(a, "{s}/src.safetensors", .{dir});
    try writeTestSafetensors(io, src_path, a);

    const Case = struct { ft: types.FileType, dt: types.DataType, ext: []const u8 };
    const cases = [_]Case{
        .{ .ft = .gguf, .dt = .f16, .ext = "gguf" },
        .{ .ft = .gguf, .dt = .q8_0, .ext = "gguf" },
        .{ .ft = .gguf, .dt = .q4_k, .ext = "gguf" },
        .{ .ft = .safetensors, .dt = .F16, .ext = "safetensors" },
        .{ .ft = .safetensors, .dt = .SCALED_F8_E4M3, .ext = "safetensors" },
        .{ .ft = .safetensors, .dt = .MXFP4, .ext = "safetensors" },
        .{ .ft = .safetensors, .dt = .NVFP4, .ext = "safetensors" },
    };

    for (cases) |c| {
        var f = try st.init(src_path, io, gpa, a, false, false);
        defer f.deinit();

        const opts = ConvertOptions{
            .io = io,
            .path = src_path,
            .filetype = c.ft,
            .datatype = c.dt,
            .template_path = null,
            .output_dir = dir,
            .output_name = "out",
            .threads = 1,
            .skip_sensitivity = true,
            .quantization_aggressiveness = 0,
            .allow_unknown_arch = true,
        };

        const predicted = try predictOutputSize(&f, opts, gpa, a);
        try convert(&f, opts, gpa, a);

        const out_path = try std.fmt.allocPrint(a, "{s}/out.{s}", .{ dir, c.ext });
        const actual = (try std.Io.Dir.cwd().statFile(io, out_path, .{})).size;
        testing.expectEqual(predicted, actual) catch |err| {
            std.log.err("size mismatch for {s} {s}: predicted={} actual={}", .{ @tagName(c.ft), @tagName(c.dt), predicted, actual });
            return err;
        };
    }
}
test "datatypeFitsFiletype accepts cross-format equivalents and rejects the rest" {
    // Equivalent spellings are a legitimate request, not an error: the assignment path maps
    // them with DataType.forFormat.
    try testing.expect(datatypeFitsFiletype(.f16, .safetensors));
    try testing.expect(datatypeFitsFiletype(.F16, .gguf));
    try testing.expect(datatypeFitsFiletype(.bf16, .safetensors));
    try testing.expect(datatypeFitsFiletype(.BF16, .gguf));
    try testing.expect(datatypeFitsFiletype(.F32, .gguf));
    try testing.expect(datatypeFitsFiletype(.i8, .safetensors));

    // Same-format types always fit.
    try testing.expect(datatypeFitsFiletype(.q4_k, .gguf));
    try testing.expect(datatypeFitsFiletype(.q8_0, .gguf));
    try testing.expect(datatypeFitsFiletype(.SCALED_F8_E4M3, .safetensors));
    try testing.expect(datatypeFitsFiletype(.INT4_CONVROT, .safetensors));

    // A template-driven conversion has no target type.
    try testing.expect(datatypeFitsFiletype(null, .gguf));
    try testing.expect(datatypeFitsFiletype(null, .safetensors));

    // The regression: block-quantized GGUF types in a SafeTensors container. This used to be
    // accepted and wrote real q4_k blocks under `"dtype":"q4_k"`, which ComfyUI rejects.
    for ([_]types.DataType{ .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .iq4_nl, .mxfp4 }) |dt| {
        try testing.expect(!datatypeFitsFiletype(dt, .safetensors));
    }

    // And the mirror image: ComfyUI cluster types have no GGUF form.
    for ([_]types.DataType{ .SCALED_F8_E4M3, .INT8, .INT8_CONVROT, .INT4_CONVROT, .NVFP4, .MXFP4, .MXFP8_E4M3, .F8_E4M3 }) |dt| {
        try testing.expect(!datatypeFitsFiletype(dt, .gguf));
    }
}

test "liftToFloor stays inside the requested format's family" {
    const F = imagearch.Precision;
    // Rotated-int cluster line keeps its rotation and per-row scale at 8 bits.
    try testing.expectEqual(types.DataType.INT8_CONVROT, liftToFloor(.INT4_CONVROT, F.bits8).?);
    try testing.expectEqual(types.DataType.INT8_CONVROT, liftToFloor(.INT4_CONVROT_SR, F.bits8).?);
    try testing.expectEqual(types.DataType.INT8_CONVROT, liftToFloor(.ASYM_W4A8_INT8, F.bits8).?);
    try testing.expectEqual(types.DataType.W6A8_INT8, liftToFloor(.ASYM_W4A8_INT8, F.bits6).?);
    try testing.expectEqual(types.DataType.W6A8_INT8, liftToFloor(.ASYM_W4A8_INT8_ZP, F.bits6).?);
    try testing.expectEqual(types.DataType.INT8_CONVROT, liftToFloor(.ASYM_W4A8_INT8_ZP, F.bits8).?);
    try testing.expectEqual(types.DataType.W6A8_INT8, liftToFloor(.ASYM_W4A8_INT8, F.bits5).?);
    try testing.expectEqual(types.DataType.W6A8_INT8, liftToFloor(.W6A8_INT8, F.bits6).?);
    try testing.expectEqual(types.DataType.INT8_CONVROT, liftToFloor(.W6A8_INT8, F.bits8).?);
    // Float lines step to their own siblings, not to an int format.
    try testing.expectEqual(types.DataType.MXFP8_E4M3, liftToFloor(.NVFP4, F.bits8).?);
    try testing.expectEqual(types.DataType.MXFP8_E4M3, liftToFloor(.MXFP4, F.bits8).?);
    // GGUF walks the existing level ordering. q8_0 is the first rung at 8 bits.
    try testing.expectEqual(types.DataType.q8_0, liftToFloor(.q4_k, F.bits8).?);
    try testing.expectEqual(types.DataType.q6_k, liftToFloor(.q4_k, F.bits6).?);
    try testing.expectEqual(types.DataType.q5_k, liftToFloor(.q4_k, F.bits5).?);
    // A target already at or above the floor is returned untouched.
    try testing.expectEqual(types.DataType.INT8_CONVROT, liftToFloor(.INT8_CONVROT, F.bits8).?);
    try testing.expectEqual(types.DataType.q8_0, liftToFloor(.q8_0, F.bits8).?);
    // No cluster line carries 16 bits, so the caller falls back to the source dtype.
    try testing.expect(liftToFloor(.INT4_CONVROT, F.bits16) == null);
    try testing.expect(liftToFloor(.NVFP4, F.bits16) == null);
}

test "assignTensorType: sensenova_u15 floors down_proj on every output path" {
    testing.log_level = .err; // the floor logs each lift at info level

    const dims = [_]usize{ 4096, 12288 };
    const n: u64 = 4096 * 12288;
    const Case = struct { target: types.DataType, filetype: types.FileType, want: []const u8 };
    const cases = [_]Case{
        .{ .target = .q4_k, .filetype = .gguf, .want = "q8_0" },
        .{ .target = .q2_k, .filetype = .gguf, .want = "q8_0" },
        .{ .target = .INT4_CONVROT, .filetype = .safetensors, .want = "INT8_CONVROT" },
        .{ .target = .NVFP4, .filetype = .safetensors, .want = "MXFP8_E4M3" },
        .{ .target = .ASYM_W4A8_INT8, .filetype = .safetensors, .want = "INT8_CONVROT" },
        .{ .target = .W6A8_INT8, .filetype = .safetensors, .want = "INT8_CONVROT" },
        // Already at the floor: must pass through untouched, not get lifted again.
        .{ .target = .INT8_CONVROT, .filetype = .safetensors, .want = "INT8_CONVROT" },
    };

    // The cluster size path allocates sub-tensor specs expecting an arena teardown.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    for (cases) |c| {
        var opts = testOpts(c.target);
        opts.filetype = c.filetype;

        var t = types.Tensor{
            .name = "language_model.model.layers.0.mlp.down_proj.weight",
            .type = "BF16",
            .dims = @constCast(dims[0..]),
            .size = 0,
            .offset = 0,
        };
        try assignTensorType(&t, n, &imagearch.sensenova_u15, QUANTIZATION_THRESHOLD, opts, false, null, a);
        try testing.expectEqualStrings(c.want, t.type);

        // The MoT twin is measured fine at the target and must not be lifted: the
        // substring "mlp.down_proj" does not occur in "mlp_mot_gen.down_proj".
        var twin = types.Tensor{
            .name = "language_model.model.layers.0.mlp_mot_gen.down_proj.weight",
            .type = "BF16",
            .dims = @constCast(dims[0..]),
            .size = 0,
            .offset = 0,
        };
        try assignTensorType(&twin, n, &imagearch.sensenova_u15, QUANTIZATION_THRESHOLD, opts, false, null, a);
        try testing.expectEqualStrings(@tagName(c.target), twin.type);
    }
}

test "precision floors do not touch architectures that declare none" {
    testing.log_level = .err;
    const dims = [_]usize{ 4096, 12288 };
    const n: u64 = 4096 * 12288;
    var t = types.Tensor{
        .name = "language_model.model.layers.0.mlp.down_proj.weight",
        .type = "BF16",
        .dims = @constCast(dims[0..]),
        .size = 0,
        .offset = 0,
    };
    try assignTensorType(&t, n, &imagearch.krea2, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, std.testing.allocator);
    try testing.expectEqualStrings("q4_k", t.type);
}

const FakeMetaFile = struct {
    meta: ?std.json.ObjectMap,
    pub fn getSourceMetadata(self: FakeMetaFile) ?std.json.ObjectMap {
        return self.meta;
    }
};

fn expectGgufArch(meta: *std.json.ObjectMap, want: []const u8) !void {
    const v = meta.get("general.architecture") orelse return error.MissingArch;
    try testing.expectEqualStrings(want, v.string);
}

test "buildGgufMetadata: generic arch keeps the source general.architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "general.architecture", .{ .string = "llama" });

    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, &imagearch.generic_arch, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&meta, "llama");
}

test "buildGgufMetadata: generic arch with no source value falls back to unknown" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = null }, &imagearch.generic_arch, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&meta, "unknown");
}

test "buildGgufMetadata: arch override wins over the source value" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "general.architecture", .{ .string = "llama" });

    var opts = testOpts(.q4_k);
    opts.arch_override = "myarch";
    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, &imagearch.generic_arch, null, .empty, opts, a, null);
    try expectGgufArch(&meta, "myarch");
}

test "buildGgufMetadata: a detected arch still overwrites the source value" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "general.architecture", .{ .string = "llama" });

    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, &imagearch.lumina2, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&meta, "lumina2");
}

test "buildGgufMetadata: a name detection only guessed at never overwrites the source's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // qwen3next ships qwen35's tensor names, so detection cannot separate them
    // and the requantize path is the exposure: the <arch>.* keys copied below
    // keep their source prefix, and renaming the architecture out from under
    // them loses the file.
    var src: std.json.ObjectMap = .empty;
    try src.put(a, "general.architecture", .{ .string = "qwen3next" });

    for ([_]*const imagearch.Arch{ &imagearch.qwen35, &imagearch.qwen35moe }) |arch| {
        var meta: std.json.ObjectMap = .empty;
        try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, arch, null, .empty, testOpts(.q4_k), a, null);
        try expectGgufArch(&meta, "qwen3next");
    }

    // A family-wide name defers for the same reason even though it is one of
    // llama.cpp's own: qwen3's key set is equally gemma3's, and only the source
    // knows which of them it is.
    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, &imagearch.qwen3, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&meta, "qwen3next");

    // With nothing to defer to, detection's name is all there is.
    var bare: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&bare, FakeMetaFile{ .meta = null }, &imagearch.qwen3, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&bare, "qwen3");

    // -A overrules both.
    var forced_opts = testOpts(.q4_k);
    forced_opts.arch_override = "gemma3";
    var forced: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&forced, FakeMetaFile{ .meta = src }, &imagearch.qwen3, null, .empty, forced_opts, a, null);
    try expectGgufArch(&forced, "gemma3");

    // With no source to defer to, qwen35 names itself: llama.cpp implements
    // LLM_ARCH_QWEN35 under exactly that string.
    var q35: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&q35, FakeMetaFile{ .meta = null }, &imagearch.qwen35, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&q35, "qwen35");

    // Without a config.json saying so, qwen35moe declines to name itself.
    var none: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&none, FakeMetaFile{ .meta = null }, &imagearch.qwen35moe, null, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&none, "unknown");

    // -A still names either by hand.
    for ([_]*const imagearch.Arch{ &imagearch.qwen35, &imagearch.qwen35moe }) |arch| {
        var named: std.json.ObjectMap = .empty;
        try buildGgufMetadata(&named, FakeMetaFile{ .meta = null }, arch, null, .empty, forced_opts, a, null);
        try expectGgufArch(&named, "gemma3");
    }
}

test "buildGgufMetadata: -t keeps the architecture deferred to the source" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "general.architecture", .{ .string = "qwen3next" });

    // A template replaces the source-metadata copy the deferral rides on, so
    // the deferred name has to come off the source by another route.
    var tmpl: std.json.ObjectMap = .empty;
    try tmpl.put(a, "general.name", .{ .string = "Mini" });

    const archs = [_]*const imagearch.Arch{ &imagearch.qwen35, &imagearch.qwen35moe, &imagearch.qwen3 };
    for (archs) |arch| {
        var meta: std.json.ObjectMap = .empty;
        try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, arch, tmpl, .empty, testOpts(.q4_k), a, null);
        try expectGgufArch(&meta, "qwen3next");
    }

    // A template that names an architecture outright still outranks both.
    var named_tmpl: std.json.ObjectMap = .empty;
    try named_tmpl.put(a, "general.architecture", .{ .string = "gemma3" });
    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, &imagearch.qwen3, named_tmpl, .empty, testOpts(.q4_k), a, null);
    try expectGgufArch(&meta, "gemma3");
}

/// Enough of an HF checkpoint for the metadata pass: the shape fields and an
/// spm tokenizer, which needs no pre-tokenizer tag.
fn fakeHfModel(arch: *const imagearch.Arch) HfLlm.Model {
    return .{
        .arch = arch,
        .dir = "/models/Mini-1B-Chat",
        .block_count = 2,
        .embedding_length = 4,
        .feed_forward_length = 8,
        .head_count = 2,
        .head_count_kv = 1,
        .context_length = 4096,
        .rms_eps = 1e-5,
        .rope_theta = 10000,
        .rope_scaling = null,
        .vocab_size = 2,
        .head_dim = null,
        .is_spm = true,
        .scores = null,
        .tokens = &.{ "a", "b" },
        .token_types = &.{ 1, 1 },
        .merges = &.{},
        .tokenizer_pre = "",
        .chat_template = "hf",
        .add_bos_token = null,
        .add_eos_token = null,
        .add_sep_token = null,
        .bos_id = null,
        .eos_id = null,
        .pad_id = null,
        .unk_id = null,
        .sampling_temp = null,
        .sampling_top_k = null,
        .sampling_top_p = null,
        .license = null,
        .license_link = null,
        .tags = &.{},
        .languages = &.{},
        .datasets = &.{},
        .base_models = &.{},
    };
}

test "buildGgufMetadata: -t outranks the HF config, which outranks the source" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // A config.json the checkpoint got wrong: -t is the only way to correct it,
    // and it is worth nothing if the derived value wins.
    var tmpl: std.json.ObjectMap = .empty;
    try tmpl.put(a, "llama.context_length", .{ .integer = 131072 });
    try tmpl.put(a, "tokenizer.chat_template", .{ .string = "mine" });

    var meta: std.json.ObjectMap = .empty;
    const hf = fakeHfModel(&imagearch.llama);
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = null }, &imagearch.llama, tmpl, .empty, testOpts(.q4_k), a, hf);
    try testing.expectEqual(@as(i64, 131072), meta.get("llama.context_length").?.integer);
    try testing.expectEqualStrings("mine", meta.get("tokenizer.chat_template").?.string);
    // Everything the template says nothing about still comes off the config.
    try testing.expectEqual(@as(i64, 2), meta.get("llama.block_count").?.integer);

    // The source file is below both: a checkpoint's own metadata does not get
    // to contradict the config.json sitting beside it.
    var src: std.json.ObjectMap = .empty;
    try src.put(a, "llama.block_count", .{ .integer = 99 });
    var over_src: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&over_src, FakeMetaFile{ .meta = src }, &imagearch.llama, null, .empty, testOpts(.q4_k), a, hf);
    try testing.expectEqual(@as(i64, 2), over_src.get("llama.block_count").?.integer);
}

test "buildGgufMetadata: -A renames the HF key prefix along with the architecture" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Keys under the detected name beside a general.architecture saying
    // something else is a file llama.cpp aborts on, missing <arch>.block_count.
    var opts = testOpts(.q4_k);
    opts.arch_override = "mymodel";
    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = null }, &imagearch.llama, null, .empty, opts, a, fakeHfModel(&imagearch.llama));
    try expectGgufArch(&meta, "mymodel");
    try testing.expectEqual(@as(i64, 2), meta.get("mymodel.block_count").?.integer);
    try testing.expect(!meta.contains("llama.block_count"));
    // The family tests do not follow the label: this is still a llama, so it
    // still carries the keys llama.cpp's converter writes for one.
    try testing.expectEqual(@as(i64, 2), meta.get("mymodel.vocab_size").?.integer);
}

fn fakeTensorList(a: std.mem.Allocator, names: []const []const u8) !std.ArrayList(types.Tensor) {
    var list: std.ArrayList(types.Tensor) = .empty;
    for (names) |n| try list.append(a, .{ .name = n, .type = "BF16", .dims = &.{}, .size = 0, .offset = 0 });
    return list;
}

test "filterAndStripTensors: component_filter=false keeps HF names and lm_head" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const llmish = imagearch.Arch{ .name = "llmish", .keys_detect = &.{}, .threshhold = null, .component_filter = false };
    const tensors = try fakeTensorList(a, &.{ "model.layers.0.self_attn.q_proj.weight", "model.norm.weight", "lm_head.weight" });
    const out = try filterAndStripTensors(.{ .tensors = tensors }, &llmish, .gguf, false, a);

    try testing.expectEqual(@as(usize, 3), out.items.len);
    try testing.expectEqualStrings("model.layers.0.self_attn.q_proj.weight", out.items[0].name);
    try testing.expectEqualStrings("model.norm.weight", out.items[1].name);
    try testing.expectEqualStrings("lm_head.weight", out.items[2].name);
}

test "filterAndStripTensors: component_filter arches keep dropping non-model tensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const tensors = try fakeTensorList(a, &.{ "model.layers.0.self_attn.q_proj.weight", "model.norm.weight", "lm_head.weight" });
    const out = try filterAndStripTensors(.{ .tensors = tensors }, &imagearch.generic_arch, .gguf, false, a);

    try testing.expectEqual(@as(usize, 2), out.items.len);
    try testing.expectEqualStrings("layers.0.self_attn.q_proj.weight", out.items[0].name);
    try testing.expectEqualStrings("norm.weight", out.items[1].name);
}

test "assignTensorType: row_aligned_blocks narrows tensors whose rows break the block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const llmish = imagearch.Arch{ .name = "llmish", .keys_detect = &.{}, .threshhold = null, .row_aligned_blocks = true };
    const n: u64 = 4096 * 896; // 896 cols: real case, Qwen2.5-0.5B n_embd

    var dims_bad = [_]usize{ 4096, 896 };
    var misaligned = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_bad[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&misaligned, n, &llmish, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("q5_0", misaligned.type); // llama.cpp's q4_k fallback
    try testing.expectEqual(gguf.GgmlType.q5_0.calcSizeInBytes(n), misaligned.size);

    var misaligned6 = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_bad[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&misaligned6, n, &llmish, QUANTIZATION_THRESHOLD, testOpts(.q6_k), false, null, a);
    try testing.expectEqualStrings("q8_0", misaligned6.type);

    // Rows that divide neither block size have nothing to narrow to.
    var dims_odd = [_]usize{ 4096, 40 };
    const n_odd: u64 = 4096 * 40;
    var unfixable = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_odd[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&unfixable, n_odd, &llmish, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("f16", unfixable.type);

    var misaligned8 = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_bad[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&misaligned8, n, &llmish, QUANTIZATION_THRESHOLD, testOpts(.q8_0), false, null, a);
    try testing.expectEqualStrings("q8_0", misaligned8.type); // 896 % 32 == 0

    var dims_ok = [_]usize{ 4096, 1024 };
    var aligned = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_ok[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&aligned, 4096 * 1024, &llmish, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("q4_k", aligned.type);

    // Without the field, flat blocking still stands: same shape, same target.
    var legacy = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_bad[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&legacy, n, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("q4_k", legacy.type);
}

test "assignTensorType: sensitivity promotion into a wider block gets narrowed back" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const llmish = imagearch.Arch{ .name = "llmish", .keys_detect = &.{}, .threshhold = null, .row_aligned_blocks = true };
    // 41 lands one rung above the q4_0 target, i.e. on q4_k and its 256-element
    // blocks, which 896-wide rows do not divide.
    const sens = try std.json.parseFromSlice(std.json.Value, a,
        \\{"model.layers.0.mlp.up_proj.weight": 41}
    , .{});
    defer sens.deinit();

    var opts = testOpts(.q4_0);
    opts.skip_sensitivity = false;
    opts.quantization_aggressiveness = 50;
    opts.allowed_quant_families = try QuantizationFamilies.parse("0,k");

    var dims_bad = [_]usize{ 4096, 896 };
    var misaligned = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_bad[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&misaligned, 4096 * 896, &llmish, QUANTIZATION_THRESHOLD, opts, true, &sens, a);
    try testing.expectEqualStrings("q5_0", misaligned.type);
    try testing.expectEqual(gguf.GgmlType.q5_0.calcSizeInBytes(4096 * 896), misaligned.size);

    // Rows that do divide it still take the promotion.
    var dims_ok = [_]usize{ 4096, 1024 };
    var aligned = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = "BF16", .dims = @constCast(dims_ok[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&aligned, 4096 * 1024, &llmish, QUANTIZATION_THRESHOLD, opts, true, &sens, a);
    try testing.expectEqualStrings("q4_k", aligned.type);

    // Sources the fallback has no upcast for keep their own type, so nothing
    // resets the byte count the rejected promotion wrote: every later tensor
    // offset comes off t.size, so a stale one corrupts the whole file. Rows of
    // 40 divide neither block size, so there is no narrowing to rescue them.
    var dims_odd = [_]usize{ 4096, 40 };
    const n: u64 = 4096 * 40;
    const Case = struct { src: []const u8, bytes: u64 };
    for ([_]Case{ .{ .src = "F16", .bytes = n * 2 }, .{ .src = "F32", .bytes = n * 4 } }) |c| {
        var t = types.Tensor{ .name = "model.layers.0.mlp.up_proj.weight", .type = c.src, .dims = @constCast(dims_odd[0..]), .size = c.bytes, .offset = 0 };
        try assignTensorType(&t, n, &llmish, QUANTIZATION_THRESHOLD, opts, true, &sens, a);
        try testing.expectEqualStrings(if (std.mem.eql(u8, c.src, "F16")) "F16" else "F32", t.type);
        try testing.expectEqual(c.bytes, t.size);
    }
}

test "assignTensorType: llama floors the head and token table, protects the router" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const Case = struct { name: []const u8, cols: usize, rows: usize, src: []const u8, target: types.DataType = .q4_k, want: []const u8 };
    const cases = [_]Case{
        .{ .name = "blk.0.ffn_gate_inp.weight", .cols = 2048, .rows = 256, .src = "BF16", .want = "f32" },
        // f16 is no better than bf16 to ggml here: no f32-by-f16 binary op.
        .{ .name = "blk.0.ffn_gate_inp.weight", .cols = 2048, .rows = 256, .src = "F16", .want = "f32" },
        .{ .name = "blk.0.ffn_gate_inp_shexp.weight", .cols = 2048, .rows = 256, .src = "BF16", .want = "f32" },
        // The LM head quantizes, one rung above the body, under either name.
        .{ .name = "output.weight", .cols = 2048, .rows = 151936, .src = "BF16", .want = "q6_k" },
        .{ .name = "lm_head.weight", .cols = 2048, .rows = 151936, .src = "BF16", .want = "q6_k" },
        // A request already above the floor is left where it is.
        .{ .name = "output.weight", .cols = 2048, .rows = 151936, .src = "BF16", .target = .q8_0, .want = "q8_0" },
        // The token table takes the same floor as the head: six bits is enough
        // for either, and bf16 costs a sixth of the file for no measurable gain.
        .{ .name = "token_embd.weight", .cols = 2048, .rows = 151936, .src = "BF16", .want = "q6_k" },
        .{ .name = "model.embed_tokens.weight", .cols = 2048, .rows = 151936, .src = "BF16", .want = "q6_k" },
        // A request already above the floor stays where it is.
        .{ .name = "token_embd.weight", .cols = 2048, .rows = 151936, .src = "BF16", .target = .q8_0, .want = "q8_0" },
        // Neither the named list nor the head's floor may catch these:
        .{ .name = "blk.0.attn_output.weight", .cols = 5120, .rows = 4096, .src = "BF16", .want = "q4_k" },
        .{ .name = "blk.0.attn_q.weight", .cols = 5120, .rows = 8192, .src = "BF16", .want = "q4_k" },
        // HF MoE router protected via substring; gate_proj must stay quantizable.
        .{ .name = "model.layers.0.mlp.gate.weight", .cols = 2048, .rows = 256, .src = "BF16", .want = "f32" },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .cols = 5120, .rows = 12288, .src = "BF16", .want = "q4_k" },
    };

    for (cases) |c| {
        var dims = [_]usize{ c.rows, c.cols };
        var t = types.Tensor{ .name = c.name, .type = c.src, .dims = @constCast(dims[0..]), .size = 0, .offset = 0 };
        try assignTensorType(&t, c.rows * c.cols, &imagearch.qwen35moe, QUANTIZATION_THRESHOLD, testOpts(c.target), false, null, a);
        try testing.expectEqualStrings(c.want, t.type);
    }
}

test "assignTensorType: a spared block-quantized tensor is floored to a float for safetensors" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // -H on a stock llama.cpp file: the tables it protects are exactly the ones
    // that arrive quantized, and "q4_k" is not a dtype any safetensors reader
    // takes - the writer refuses the whole file over it.
    const Case = struct { name: []const u8, src: []const u8, target: types.DataType, want: []const u8, bytes: u64 };
    const rows: usize = 151936;
    const cols: usize = 2048;
    const n: u64 = rows * cols;
    const cases = [_]Case{
        .{ .name = "model.embed_tokens.weight", .src = "q4_k", .target = .F16, .want = "F16", .bytes = n * 2 },
        // Under either name for the table: the GGUF-native one arrives on the
        // way back out of a llama.cpp file.
        .{ .name = "token_embd.weight", .src = "q6_k", .target = .F16, .want = "F16", .bytes = n * 2 },
        .{ .name = "model.embed_tokens.weight", .src = "q4_k", .target = .F32, .want = "F32", .bytes = n * 4 },
        // A target that is not a float says nothing about the floor: f16, like
        // llama.cpp's own converter writes for these tables.
        .{ .name = "model.embed_tokens.weight", .src = "q4_k", .target = .INT8_CONVROT, .want = "F16", .bytes = n * 2 },
        // A source type safetensors can spell is left exactly as it was.
        .{ .name = "model.embed_tokens.weight", .src = "BF16", .target = .F16, .want = "BF16", .bytes = 0 },
    };

    for (cases) |c| {
        var dims = [_]usize{ rows, cols };
        var opts = testOpts(c.target);
        opts.filetype = .safetensors;
        var t = types.Tensor{ .name = c.name, .type = c.src, .dims = @constCast(dims[0..]), .size = 0, .offset = 0 };
        try assignTensorType(&t, n, &imagearch.llama, QUANTIZATION_THRESHOLD, opts, false, null, a);
        try testing.expectEqualStrings(c.want, t.type);
        try testing.expectEqual(c.bytes, t.size);
    }
}

test "assignTensorType: an image arch's high-precision 2-D weights keep their float width" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // anima spares "blocks.0." - a whole transformer block. The f32 floor is
    // for 1-D parameters and routers; applying it here would double the block.
    var dims = [_]usize{ 4096, 4096 };
    const n: u64 = 4096 * 4096;
    var half = types.Tensor{ .name = "blocks.0.self_attn.q_proj.weight", .type = "F16", .dims = @constCast(dims[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&half, n, &imagearch.anima, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("F16", half.type); // untouched; the writer lowercases


    // bf16 still goes to f32: ComfyUI cannot carry bf16 in its shape math.
    var brain = types.Tensor{ .name = "blocks.0.self_attn.q_proj.weight", .type = "BF16", .dims = @constCast(dims[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&brain, n, &imagearch.anima, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("f32", brain.type);

    // The same arch's 1-D parameters keep the floor.
    var norm_dims = [_]usize{4096};
    var norm = types.Tensor{ .name = "blocks.0.norm.weight", .type = "F16", .dims = @constCast(norm_dims[0..]), .size = 0, .offset = 0 };
    try assignTensorType(&norm, 4096, &imagearch.anima, QUANTIZATION_THRESHOLD, testOpts(.q4_k), false, null, a);
    try testing.expectEqualStrings("f32", norm.type);
}

test "pathsCollide sees through a ./ prefix" {
    const io = std.testing.io;
    // dirname("m.safetensors") is null -> ".", which path.join spells "./m...".
    try testing.expect(pathsCollide(io, "./m.safetensors", "m.safetensors"));
    try testing.expect(pathsCollide(io, "/tmp/../tmp/m.safetensors", "/tmp/m.safetensors"));
    try testing.expect(pathsCollide(io, "./m.gguf", "m.gguf")); // -n can name a gguf source back
    try testing.expect(!pathsCollide(io, "./m-q4_k.safetensors", "m.safetensors"));
    try testing.expect(!pathsCollide(io, "/tmp/m.safetensors", "/usr/m.safetensors"));
}

test "provenance is stamped on plain gguf output" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = null }, &imagearch.lumina2, null, .empty, testOpts(.q4_k), a, null);
    try testing.expectEqualStrings("Converted with ggufy", meta.get("converter_note").?.string);
    try testing.expectEqualStrings(ggufy_repo_url, meta.get("converter_url").?.string);
    try testing.expect(std.mem.startsWith(u8, meta.get("converted_by").?.string, "ggufy "));
}

test "source provenance from another tool is overwritten" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "converted_by", .{ .string = "comfy-kitchen" });

    var meta: std.json.ObjectMap = .empty;
    try buildGgufMetadata(&meta, FakeMetaFile{ .meta = src }, &imagearch.lumina2, null, .empty, testOpts(.q4_k), a, null);
    try testing.expect(std.mem.startsWith(u8, meta.get("converted_by").?.string, "ggufy "));
}

test "safetensors metadata gains provenance without source metadata" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var meta: ?std.json.ObjectMap = null;
    try buildSafetensorsMetadata(&meta, FakeMetaFile{ .meta = null }, null, .empty, false, a);
    try testing.expect(meta != null);
    try testing.expect(std.mem.startsWith(u8, meta.?.get("converted_by").?.string, "ggufy "));
}

test "safetensors metadata names a format transformers accepts" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // transformers and huggingface_hub reject a __metadata__ map whose "format"
    // they don't know, and the provenance stamp gives every output that map.
    var meta: ?std.json.ObjectMap = null;
    try buildSafetensorsMetadata(&meta, FakeMetaFile{ .meta = null }, null, .empty, true, a);
    try testing.expectEqualStrings("pt", meta.?.get("format").?.string);

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "format", .{ .string = "flax" });
    var kept: ?std.json.ObjectMap = null;
    try buildSafetensorsMetadata(&kept, FakeMetaFile{ .meta = src }, null, .empty, false, a);
    try testing.expectEqualStrings("flax", kept.?.get("format").?.string);
}

test "hf-layout safetensors drops source metadata" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    var src: std.json.ObjectMap = .empty;
    try src.put(a, "general.languages", .{ .string = "en" });

    var meta: ?std.json.ObjectMap = null;
    try buildSafetensorsMetadata(&meta, FakeMetaFile{ .meta = src }, null, .empty, true, a);
    try testing.expect(meta != null);
    try testing.expect(meta.?.get("general.languages") == null);
    try testing.expect(std.mem.startsWith(u8, meta.?.get("converted_by").?.string, "ggufy "));
}

// A BPE tokenizer HfLlm.load accepts: it refuses one whose pre-tokenizer matches
// no llama.cpp tag and one with no merge table, so a bare vocab is not enough.
// Llama-3's splitting, to go with the "llama" model_type these fixtures declare.
const test_bpe_tokenizer =
    \\{"pre_tokenizer":{"type":"Sequence","pretokenizers":[
    \\ {"type":"Split","behavior":"Isolated","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    \\ {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
    \\ "model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":["a b","ab c"]}}
;

// End-to-end RoPE plumbing check: HF checkpoint -> q4_k GGUF -> HF safetensors.
// Row constants (r+1)*(idx+1) are f16-exact and distinct per row (steps of at
// least 1 against a ulp of at most 4), so any missed or spurious row swap shows
// up byte-wise. Widths are multiples of 256 so every 2D weight q4_k-blocks;
// k_proj is non-square to expose a dim flip.
test "HF llama Q/K RoPE survives dequantizing a block-quantized GGUF" {
    testing.log_level = .err; // conversions log every tensor at info level

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var pool: @import("ThreadPool.zig").ThreadPool = undefined;
    try pool.init(.{ .allocator = a, .n_jobs = 1 });
    defer pool.deinit();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_rope_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const Spec = struct { name: []const u8, shape: []const usize };
    const specs = [_]Spec{
        .{ .name = "model.embed_tokens.weight", .shape = &.{ 100, 512 } },
        .{ .name = "model.norm.weight", .shape = &.{512} },
        .{ .name = "lm_head.weight", .shape = &.{ 100, 512 } },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &.{512} },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 512, 512 } },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 256, 512 } },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 256, 512 } },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &.{ 512, 512 } },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &.{ 512, 512 } },
        .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &.{ 512, 512 } },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 512, 512 } },
        .{ .name = "model.layers.0.post_attention_layernorm.weight", .shape = &.{512} },
    };

    const writeFile = struct {
        fn f(path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{ .truncate = true });
            defer file.close(std.testing.io);
            var buf: [8192]u8 = undefined;
            var fw = file.writer(std.testing.io, &buf);
            try fw.interface.writeAll(bytes);
            try fw.interface.flush();
        }
    }.f;

    try writeFile(
        try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"llama","num_hidden_layers":1,"hidden_size":512,"intermediate_size":512,"num_attention_heads":4,"num_key_value_heads":2,"rms_norm_eps":1e-5,"vocab_size":100,"max_position_embeddings":128}
        ++ "\n",
    );
    // load() refuses a model with no tokenizer.
    try writeFile(
        try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}),
        test_bpe_tokenizer,
    );

    var total: usize = 0;
    for (specs) |s| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        total += n;
    }
    const data = try a.alloc(u8, total * 2);
    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{");
    var off: u64 = 0;
    var q_src: []u8 = undefined;
    var k_src: []u8 = undefined;
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        const rows: usize = s.shape[0];
        const cols: usize = n / rows;
        const f16s: []f16 = @ptrCast(@alignCast(data[off..][0 .. n * 2])); // offsets stay 2-byte aligned
        for (0..rows) |r| {
            for (0..cols) |c| {
                f16s[r * cols + c] = @floatFromInt((r + 1) * (i + 1));
            }
        }
        if (i == 4) q_src = data[off..][0 .. n * 2];
        if (i == 5) k_src = data[off..][0 .. n * 2];
        if (i > 0) try header.appendSlice(a, ",");
        try header.print(a, "\"{s}\":{{\"dtype\":\"F16\",\"shape\":[", .{s.name});
        for (s.shape, 0..) |d, j| {
            if (j > 0) try header.appendSlice(a, ",");
            try header.print(a, "{}", .{d});
        }
        try header.print(a, "],\"data_offsets\":[{[0]},{[1]}]}}", .{ off, off + n * 2 });
        off += n * 2;
    }
    try header.appendSlice(a, "}");

    var st_file: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try st_file.appendSlice(a, &len_buf);
    try st_file.appendSlice(a, header.items);
    try st_file.appendSlice(a, data);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeFile(src_path, st_file.items);

    const findT = struct {
        fn f(ts: []const types.Tensor, name: []const u8) !types.Tensor {
            for (ts) |t| if (std.mem.eql(u8, t.name, name)) return t;
            return error.MissingTensor;
        }
    }.f;

    // Step A: HF dir -> q4_k GGUF. The permute must land before quantization,
    // so the on-disk blocks must equal q4_k(permute(W)) computed here directly.
    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    try convert(&f1, .{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .q4_k,
        .template_path = null,
        .output_dir = dir,
        .output_name = "q4k",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    }, gpa, a);

    const q4k_path = try std.fmt.allocPrint(a, "{s}/q4k.gguf", .{dir});
    var fg = try gguf.init(q4k_path, io, gpa, a, false);
    defer fg.deinit();

    const readG = struct {
        fn f(g: anytype, t: types.Tensor, alloc: std.mem.Allocator) ![]u8 {
            const buf = try alloc.alloc(u8, @intCast(t.size));
            const file = try g.openFileForTensor(t.name);
            _ = try file.readPositionalAll(g.io, buf, t.offset + g.current_data_begin);
            return buf;
        }
    }.f;

    // 1-D parameters land in f32 whatever float they came from: ggml has no
    // f32-by-f16 binary op and aborts on an f16 norm vector.
    try testing.expectEqualStrings("f32", (try findT(fg.tensors.items, "blk.0.attn_norm.weight")).type);
    try testing.expectEqualStrings("f32", (try findT(fg.tensors.items, "output_norm.weight")).type);

    const gq = try findT(fg.tensors.items, "blk.0.attn_q.weight");
    const gk = try findT(fg.tensors.items, "blk.0.attn_k.weight");
    const gv = try findT(fg.tensors.items, "blk.0.attn_v.weight");
    try testing.expectEqualStrings("q4_k", gq.type);
    const gq_bytes = try readG(&fg, gq, a);
    const gk_bytes = try readG(&fg, gk, a);
    const gv_bytes = try readG(&fg, gv, a);

    {
        const wq = try a.dupe(u8, q_src);
        try HfLlm.ropePermuteInPlace(a, wq, 2, 512, 512, 4, false);
        const ref = try DataTransform.Quantizer.convertTensorData(a, wq, .F16, .q4_k, 262144, &pool);
        try testing.expectEqualSlices(u8, ref, gq_bytes);

        const wk = try a.dupe(u8, k_src);
        try HfLlm.ropePermuteInPlace(a, wk, 2, 256, 512, 2, false);
        const refk = try DataTransform.Quantizer.convertTensorData(a, wk, .F16, .q4_k, 131072, &pool);
        try testing.expectEqualSlices(u8, refk, gk_bytes);
    }

    // Step B: q4_k GGUF -> HF safetensors. The un-permute must survive the
    // block-quantized source, so the output must equal dequantize -> un-permute
    // -> cast of the on-disk blocks. Before the fix, the Q/K rows stayed in
    // half-split order and this comparison went red.
    try convert(&fg, .{
        .io = io,
        .path = q4k_path,
        .filetype = .safetensors,
        .datatype = .F16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "deq",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
        .allow_upscale = true,
        .force = true, // dir still holds the config.json/tokenizer.json this test wrote
    }, gpa, a);

    var fo = try st.init(try std.fmt.allocPrint(a, "{s}/deq.safetensors", .{dir}), io, gpa, a, false, false);
    defer fo.deinit();

    const readS = struct {
        fn f(src: anytype, t: types.Tensor, alloc: std.mem.Allocator) ![]u8 {
            const buf = try alloc.alloc(u8, @intCast(t.size));
            const file = try src.openFileForTensor(t.name);
            _ = try file.readPositionalAll(src.io, buf, t.offset + src.current_data_begin);
            return buf;
        }
    }.f;

    const unpermute = struct {
        fn f(
            q4_bytes: []const u8,
            nelem: usize,
            rows: usize,
            cols: usize,
            groups: usize,
            alloc: std.mem.Allocator,
            tp: *@import("ThreadPool.zig").ThreadPool,
        ) ![]u8 {
            const f32b = try DataTransform.Quantizer.convertTensorData(alloc, q4_bytes, .q4_k, .F32, nelem, tp);
            try HfLlm.ropePermuteInPlace(alloc, f32b, 4, rows, cols, groups, true);
            return DataTransform.Quantizer.convertTensorData(alloc, f32b, .F32, .F16, nelem, tp);
        }
    }.f;

    const oq = try findT(fo.tensors.items, "model.layers.0.self_attn.q_proj.weight");
    const ok = try findT(fo.tensors.items, "model.layers.0.self_attn.k_proj.weight");
    const ov = try findT(fo.tensors.items, "model.layers.0.self_attn.v_proj.weight");
    try testing.expectEqualSlices(usize, &.{ 512, 512 }, oq.dims);
    try testing.expectEqualSlices(usize, &.{ 256, 512 }, ok.dims);

    try testing.expectEqualSlices(u8, try unpermute(gq_bytes, 262144, 512, 512, 4, a, &pool), try readS(&fo, oq, a));
    try testing.expectEqualSlices(u8, try unpermute(gk_bytes, 131072, 256, 512, 2, a, &pool), try readS(&fo, ok, a));

    // v_proj shares the shapes but no RoPE: it must pass through unpermuted.
    const f32v = try DataTransform.Quantizer.convertTensorData(a, gv_bytes, .q4_k, .F32, 131072, &pool);
    const refv = try DataTransform.Quantizer.convertTensorData(a, f32v, .F32, .F16, 131072, &pool);
    try testing.expectEqualSlices(u8, refv, try readS(&fo, ov, a));

    // -n put the weights somewhere transformers does not look by itself, so
    // the sidecars are only loadable alongside an index naming the real file.
    {
        const idx_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, HfLlm.weights_index_name });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, idx_path, a, .limited(1 << 20));
        const idx = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
        defer idx.deinit();
        const wm = idx.value.object.get("weight_map").?.object;
        try testing.expectEqual(fo.tensors.items.len, wm.count());
        try testing.expectEqualStrings("deq.safetensors", wm.get("model.layers.0.self_attn.q_proj.weight").?.string);
        var idx_total: u64 = 0;
        for (fo.tensors.items) |t| idx_total += t.size;
        try testing.expectEqual(idx_total, @as(u64, @intCast(idx.value.object.get("metadata").?.object.get("total_size").?.integer)));
    }

    // Step C: the same directory again, this time under the default weights
    // name. The index -n left behind outranks model.safetensors when
    // transformers loads, so --force has to take it with the rest.
    {
        var fg3 = try gguf.init(q4k_path, io, gpa, a, false);
        defer fg3.deinit();
        try convert(&fg3, .{
            .io = io,
            .path = q4k_path,
            .filetype = .safetensors,
            .datatype = .F16,
            .template_path = null,
            .output_dir = dir,
            .output_name = null,
            .threads = 1,
            .skip_sensitivity = true,
            .quantization_aggressiveness = 0,
            .hf_names = true,
            .allow_upscale = true,
            .force = true,
        }, gpa, a);

        const idx_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, HfLlm.weights_index_name });
        try testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, idx_path, .{}));
        const model_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, HfLlm.default_weights_name });
        try std.Io.Dir.cwd().access(io, model_path, .{});
    }

    // Step B renamed fg's tensors in place, so a second pass over the same
    // handle has nothing left to map: it must still report an HF layout and
    // still un-permute, or the next write would emit permuted Q/K rows.
    const again = try prepareConversion(&fg, .{
        .io = io,
        .path = q4k_path,
        .filetype = .safetensors,
        .datatype = .F16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "deq2",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
        .allow_upscale = true,
    }, gpa, a);
    try testing.expect(again.hf_renamed);
    try testing.expectEqual(@as(usize, 4), again.hf_unpermute.?.head_count);
    try testing.expectEqual(@as(usize, 2), again.hf_unpermute.?.head_count_kv);

    // -H against a source that is already HF has nothing to do: no rename, no
    // un-permute, no sidecars to clash with, and the source metadata kept.
    // Reading it as a renamed GGUF instead used to fail the conversion.
    var f_hf = try st.init(src_path, io, gpa, a, false, false);
    defer f_hf.deinit();
    const noop = try prepareConversion(&f_hf, .{
        .io = io,
        .path = src_path,
        .filetype = .safetensors,
        .datatype = .F16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "passthrough",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
    }, gpa, a);
    try testing.expect(!noop.hf_renamed);
    try testing.expectEqual(@as(?HfLlm.RopeHeads, null), noop.hf_unpermute);
    _ = try findT(noop.model_tensors.items, "model.layers.0.self_attn.q_proj.weight");
}

test "a tensor with no llama.cpp name stops the conversion" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_unmapped_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;

    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"llama","num_hidden_layers":1,"hidden_size":8,"intermediate_size":8,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-5,"vocab_size":4,
        \\ "max_position_embeddings":128}
    );
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    // One weight llama.cpp models, one it has never heard of, and one rotary
    // cache its own converter throws away.
    const header =
        \\{"model.layers.0.input_layernorm.weight":{"dtype":"F16","shape":[8],"data_offsets":[0,16]},
        \\"model.layers.0.self_attn.rotary_emb.inv_freq":{"dtype":"F16","shape":[8],"data_offsets":[16,32]},
        \\"model.layers.0.mlp.experts.0.w1.weight":{"dtype":"F16","shape":[8],"data_offsets":[32,48]}}
    ;
    var file_bytes: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.len), .little);
    try file_bytes.appendSlice(a, &len_buf);
    try file_bytes.appendSlice(a, header);
    try file_bytes.appendNTimes(a, 0, 48);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeFile(io, src_path, file_bytes.items);

    const opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "out",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };

    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    try testing.expectError(error.UnmappedTensors, prepareConversion(&f1, opts, gpa, a));

    // The escape hatch drops it, and the rotary cache goes quietly either way.
    var f2 = try st.init(src_path, io, gpa, a, false, false);
    defer f2.deinit();
    var permissive = opts;
    permissive.allow_unknown_arch = true;
    const prep = try prepareConversion(&f2, permissive, gpa, a);
    try testing.expectEqual(@as(usize, 1), prep.model_tensors.items.len);
    try testing.expectEqualStrings("blk.0.attn_norm.weight", prep.model_tensors.items[0].name);
}

test "a mergeless BPE checkpoint and a blank -A both stop before the write" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_merges_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;

    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"llama","num_hidden_layers":1,"hidden_size":8,"intermediate_size":8,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-5,"vocab_size":4,
        \\ "max_position_embeddings":128}
    );
    // Llama-3's splitting, so the pre-tokenizer tag is found and the merge table
    // is the only thing missing. No merges.txt beside it either.
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}),
        \\{"pre_tokenizer":{"type":"Sequence","pretokenizers":[
        \\ {"type":"Split","behavior":"Isolated","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
        \\ {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},
        \\ "model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}
    );

    const header =
        \\{"model.layers.0.input_layernorm.weight":{"dtype":"F16","shape":[8],"data_offsets":[0,16]}}
    ;
    var file_bytes: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.len), .little);
    try file_bytes.appendSlice(a, &len_buf);
    try file_bytes.appendSlice(a, header);
    try file_bytes.appendNTimes(a, 0, 16);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeFile(io, src_path, file_bytes.items);

    const opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "out",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };

    // llama.cpp reads a gpt2 vocabulary with no merges as "cannot find tokenizer
    // merges in model file", and -u is no way out of it: the file would not load
    // whatever else the flag allowed through.
    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    try testing.expectError(error.MissingMerges, convert(&f1, opts, gpa, a));
    var f2 = try st.init(src_path, io, gpa, a, false, false);
    defer f2.deinit();
    var permissive = opts;
    permissive.allow_unknown_arch = true;
    try testing.expectError(error.MissingMerges, convert(&f2, permissive, gpa, a));
    try testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}/out.gguf", .{dir}), .{}));

    // -A names general.architecture and prefixes the <arch>.* dimension keys, so
    // a blank one writes a file with neither under a name any loader knows.
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);
    for ([_][]const u8{ "", " " }) |blank| {
        var f3 = try st.init(src_path, io, gpa, a, false, false);
        defer f3.deinit();
        var named = opts;
        named.arch_override = blank;
        try testing.expectError(error.InvalidArchOverride, convert(&f3, named, gpa, a));
    }
    try testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}/out.gguf", .{dir}), .{}));

    // The same fixture with a merge table and a real name converts, so neither
    // refusal above is standing in for some other complaint.
    var f4 = try st.init(src_path, io, gpa, a, false, false);
    defer f4.deinit();
    var ok = opts;
    ok.arch_override = "llama";
    try convert(&f4, ok, gpa, a);
    try std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}/out.gguf", .{dir}), .{});
}

test "a GGUF tensor with no HF name stops -H" {
    testing.log_level = .err; // conversions log every tensor at info level

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_hf_unmapped_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Native names, so this converts as a GGUF-shaped source with no HF config
    // in sight: a dense qwen3 the reverse map covers, plus the two kinds it does
    // not - an expert stack (an MoE file detects as its dense arch, the expert
    // keys being exactly what nothing matches) and the rope cache transformers
    // recomputes from config.json.
    const Spec = struct { name: []const u8, shape: []const usize };
    const specs = [_]Spec{
        .{ .name = "token_embd.weight", .shape = &.{ 32, 64 } },
        .{ .name = "output_norm.weight", .shape = &.{64} },
        .{ .name = "blk.0.attn_norm.weight", .shape = &.{64} },
        .{ .name = "blk.0.attn_q.weight", .shape = &.{ 64, 64 } },
        .{ .name = "blk.0.attn_q_norm.weight", .shape = &.{16} },
        .{ .name = "blk.0.attn_k_norm.weight", .shape = &.{16} },
        .{ .name = "blk.0.ffn_down.weight", .shape = &.{ 64, 64 } },
        .{ .name = "blk.0.ffn_gate_exps.weight", .shape = &.{ 2, 64, 64 } },
        .{ .name = "rope_freqs.weight", .shape = &.{8} },
    };

    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{");
    var off: u64 = 0;
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        if (i > 0) try header.appendSlice(a, ",");
        try header.print(a, "\"{s}\":{{\"dtype\":\"F16\",\"shape\":[", .{s.name});
        for (s.shape, 0..) |d, j| {
            if (j > 0) try header.appendSlice(a, ",");
            try header.print(a, "{}", .{d});
        }
        try header.print(a, "],\"data_offsets\":[{[0]},{[1]}]}}", .{ off, off + n * 2 });
        off += n * 2;
    }
    try header.appendSlice(a, "}");

    var st_file: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try st_file.appendSlice(a, &len_buf);
    try st_file.appendSlice(a, header.items);
    try st_file.appendNTimes(a, 0, off);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    {
        const file = try std.Io.Dir.cwd().createFile(io, src_path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, st_file.items);
    }

    var f_st = try st.init(src_path, io, gpa, a, false, false);
    defer f_st.deinit();
    try convert(&f_st, .{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "native",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    }, gpa, a);

    const gguf_path = try std.fmt.allocPrint(a, "{s}/native.gguf", .{dir});
    const hf_opts = ConvertOptions{
        .io = io,
        .path = gguf_path,
        .filetype = .safetensors,
        .datatype = .F16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "hf",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
        .allow_upscale = true,
    };

    var fg1 = try gguf.init(gguf_path, io, gpa, a, false);
    defer fg1.deinit();
    try testing.expectError(error.UnmappedTensors, prepareConversion(&fg1, hf_opts, gpa, a));

    // The failed pass renamed in place, so the second one over the same handle
    // has nothing left to rename - and must still refuse. Predicting the size
    // and then converting reuses one handle exactly like this, and letting it
    // through would write the expert stack under its blk.* name beside a
    // config.json that has no such key.
    try testing.expectError(error.UnmappedTensors, prepareConversion(&fg1, hf_opts, gpa, a));

    // With the escape hatch the experts go, the rope cache goes quietly, and
    // nothing keeps a native name transformers would never open.
    var fg2 = try gguf.init(gguf_path, io, gpa, a, false);
    defer fg2.deinit();
    var permissive2 = hf_opts;
    permissive2.allow_unknown_arch = true;
    const prep2 = try prepareConversion(&fg2, permissive2, gpa, a);
    try testing.expectEqual(@as(usize, specs.len - 2), prep2.model_tensors.items.len);
    for (prep2.model_tensors.items) |t| {
        try testing.expect(!std.mem.startsWith(u8, t.name, "blk."));
        try testing.expect(std.mem.startsWith(u8, t.name, "model.") or std.mem.eql(u8, t.name, "lm_head.weight"));
    }
}

// A pre-tokenizer no llama.cpp tag names stops the conversion in both
// directions until -u, and -H stops while the output directory is still empty:
// its sidecars go down after the weights, so getting that wrong costs a full
// dequantize-and-write before the failure and leaves a directory holding
// weights plus the sidecars that precede the tokenizer.
test "a pre-tokenizer no tag names stops the conversion until -u" {
    testing.log_level = .err; // conversions log every tensor at info level

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_hf_pretok_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{ .truncate = true });
            defer file.close(std.testing.io);
            var buf: [8192]u8 = undefined;
            var fw = file.writer(std.testing.io, &buf);
            try fw.interface.writeAll(bytes);
            try fw.interface.flush();
        }
    }.f;

    try writeFile(try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"llama","num_hidden_layers":1,"hidden_size":64,"intermediate_size":64,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-5,"vocab_size":4,
        \\ "max_position_embeddings":128}
    );
    try writeFile(try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    const Spec = struct { name: []const u8, shape: []const usize };
    const specs = [_]Spec{
        .{ .name = "model.embed_tokens.weight", .shape = &.{ 4, 64 } },
        .{ .name = "model.norm.weight", .shape = &.{64} },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &.{64} },
        .{ .name = "model.layers.0.post_attention_layernorm.weight", .shape = &.{64} },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 32, 64 } },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 32, 64 } },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "lm_head.weight", .shape = &.{ 4, 64 } },
    };

    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{");
    var off: u64 = 0;
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        if (i > 0) try header.appendSlice(a, ",");
        try header.print(a, "\"{s}\":{{\"dtype\":\"F16\",\"shape\":[", .{s.name});
        for (s.shape, 0..) |d, j| {
            if (j > 0) try header.appendSlice(a, ",");
            try header.print(a, "{}", .{d});
        }
        try header.print(a, "],\"data_offsets\":[{[0]},{[1]}]}}", .{ off, off + n * 2 });
        off += n * 2;
    }
    try header.appendSlice(a, "}");

    var st_file: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try st_file.appendSlice(a, &len_buf);
    try st_file.appendSlice(a, header.items);
    try st_file.appendNTimes(a, 0, off);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeFile(src_path, st_file.items);

    var f_st = try st.init(src_path, io, gpa, a, false, false);
    defer f_st.deinit();
    try convert(&f_st, .{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "native",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    }, gpa, a);

    // The tag is read off tokenizer.json rather than guessed from model_type,
    // so it is one -H can turn back into a tokenizer.json.
    const gguf_path = try std.fmt.allocPrint(a, "{s}/native.gguf", .{dir});
    var fg = try gguf.init(gguf_path, io, gpa, a, false);
    defer fg.deinit();
    try testing.expectEqualStrings("llama-bpe", fg.metadata.get("tokenizer.ggml.pre").?.string);

    // What llama.cpp's own files carry, most of the time: a tag whose splitting
    // is several regexes, or none this tool maps. Faking a tokenizer.json for
    // one of those ships a vocabulary that segments text differently.
    const out_dir = try std.fmt.allocPrint(a, "{s}/out", .{dir});
    const hf_opts = ConvertOptions{
        .io = io,
        .path = gguf_path,
        .filetype = .safetensors,
        .datatype = .F16,
        .template_path = null,
        .output_dir = out_dir,
        .output_name = null,
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
        .allow_upscale = true,
    };
    try fg.metadata.put(a, "tokenizer.ggml.pre", .{ .string = "tekken" });
    try testing.expectError(error.UnknownPretokenizer, convert(&fg, hf_opts, gpa, a));

    var od = try std.Io.Dir.cwd().openDir(io, out_dir, .{});
    defer od.close(io);
    for ([_][]const u8{ "model.safetensors", "config.json", "tokenizer_config.json", "README.md" }) |name| {
        try testing.expectError(error.FileNotFound, od.access(io, name, .{}));
    }

    // -u is the same bargain it strikes for a tensor with no llama.cpp name:
    // write what maps, drop what does not. The weights and config land, the
    // vocabulary does not, and nothing carries another family's splitting.
    var permissive = hf_opts;
    permissive.allow_unknown_arch = true;
    var fg2 = try gguf.init(gguf_path, io, gpa, a, false);
    defer fg2.deinit();
    try fg2.metadata.put(a, "tokenizer.ggml.pre", .{ .string = "tekken" });
    try convert(&fg2, permissive, gpa, a);
    for ([_][]const u8{ "model.safetensors", "config.json", "tokenizer_config.json" }) |name| {
        try od.access(io, name, .{});
    }
    try testing.expectError(error.FileNotFound, od.access(io, "tokenizer.json", .{}));

    // The other direction, same rule: splitting no tag names stops the GGUF
    // rather than guessing a family, and -u takes llama.cpp's own fallback tag.
    try writeFile(try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}),
        \\{ "pre_tokenizer": { "type": "Split", "behavior": "Isolated", "pattern": { "Regex": "\\w+" } },
        \\  "model": { "type": "BPE", "vocab": { "a": 0, "b": 1, "c": 2, "d": 3 }, "merges": ["a b"] } }
    );
    var gguf_opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "untagged",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
    var f_bad = try st.init(src_path, io, gpa, a, false, false);
    defer f_bad.deinit();
    try testing.expectError(error.UnknownPretokenizer, convert(&f_bad, gguf_opts, gpa, a));

    gguf_opts.allow_unknown_arch = true;
    var f_ok = try st.init(src_path, io, gpa, a, false, false);
    defer f_ok.deinit();
    try convert(&f_ok, gguf_opts, gpa, a);
    var fu = try gguf.init(try std.fmt.allocPrint(a, "{s}/untagged.gguf", .{dir}), io, gpa, a, false);
    defer fu.deinit();
    try testing.expectEqualStrings("default", fu.metadata.get("tokenizer.ggml.pre").?.string);
}

const NameShape = struct { name: []const u8, shape: []const usize };

// An F16 safetensors of zeros with these tensors, for tests that only need names.
fn writeZeroSafetensors(io: std.Io, a: std.mem.Allocator, path: []const u8, specs: []const NameShape) !void {
    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{");
    var off: u64 = 0;
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        if (i > 0) try header.appendSlice(a, ",");
        try header.print(a, "\"{s}\":{{\"dtype\":\"F16\",\"shape\":[", .{s.name});
        for (s.shape, 0..) |d, j| {
            if (j > 0) try header.appendSlice(a, ",");
            try header.print(a, "{}", .{d});
        }
        try header.print(a, "],\"data_offsets\":[{[0]},{[1]}]}}", .{ off, off + n * 2 });
        off += n * 2;
    }
    try header.appendSlice(a, "}");
    var st_file: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try st_file.appendSlice(a, &len_buf);
    try st_file.appendSlice(a, header.items);
    try st_file.appendNTimes(a, 0, off);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, st_file.items);
}

test "-H keeps an HF-named qwen3 file's architecture, which ComfyUI-GGUF lists" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_hfnames_arch_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    // qwen3's HF detection set: llama's plus the per-head q/k norms.
    const specs = [_]NameShape{
        .{ .name = "model.embed_tokens.weight", .shape = &.{ 32, 64 } },
        .{ .name = "model.norm.weight", .shape = &.{64} },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &.{64} },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "model.layers.0.self_attn.q_norm.weight", .shape = &.{16} },
        .{ .name = "model.layers.0.self_attn.k_norm.weight", .shape = &.{16} },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 64, 64 } },
    };
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeZeroSafetensors(io, a, src_path, &specs);

    var opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "out",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
    // No config.json beside it, so the plain path refuses.
    var f0 = try st.init(src_path, io, gpa, a, false, false);
    defer f0.deinit();
    try testing.expectError(error.HfConfigUnusable, prepareConversion(&f0, opts, gpa, a));

    // -H writes it with HF names, still claiming qwen3: without the claim
    // ComfyUI-GGUF refuses the file, with it the loader passes HF names through.
    opts.hf_names = true;
    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    try convert(&f1, opts, gpa, a);
    var out = try gguf.init(try std.fmt.allocPrint(a, "{s}/out.gguf", .{dir}), io, gpa, a, false);
    defer out.deinit();
    try testing.expectEqualStrings("qwen3", out.metadata.get("general.architecture").?.string);
    try testing.expectEqual(specs.len, out.tensors.items.len);
    for (out.tensors.items) |t| {
        try testing.expect(std.mem.startsWith(u8, t.name, "model."));
    }
}

test "-H prefixes a bare Qwen3VLModel checkpoint into the layout ComfyUI detects" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_hfnames_prefix_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    // What a diffusers pipeline's Qwen3VLModel text_encoder saves: qwen3vl's
    // detection set without the outer "model.", plus the head kept as is.
    const specs = [_]NameShape{
        .{ .name = "language_model.embed_tokens.weight", .shape = &.{ 32, 64 } },
        .{ .name = "language_model.norm.weight", .shape = &.{64} },
        .{ .name = "language_model.layers.0.self_attn.q_proj.weight", .shape = &.{ 64, 64 } },
        .{ .name = "language_model.layers.0.self_attn.q_norm.weight", .shape = &.{16} },
        .{ .name = "visual.patch_embed.proj.weight", .shape = &.{ 16, 48 } },
        .{ .name = "visual.deepstack_merger_list.0.norm.weight", .shape = &.{16} },
        .{ .name = "lm_head.weight", .shape = &.{ 32, 64 } },
    };
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeZeroSafetensors(io, a, src_path, &specs);

    var f = try st.init(src_path, io, gpa, a, false, false);
    defer f.deinit();
    try convert(&f, .{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "out",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
    }, gpa, a);
    var out = try gguf.init(try std.fmt.allocPrint(a, "{s}/out.gguf", .{dir}), io, gpa, a, false);
    defer out.deinit();
    try testing.expectEqualStrings("qwen3vl", out.metadata.get("general.architecture").?.string);
    try testing.expectEqual(specs.len, out.tensors.items.len);
    for (specs) |want| {
        const expected = if (std.mem.startsWith(u8, want.name, "lm_head."))
            want.name
        else
            try std.fmt.allocPrint(a, "model.{s}", .{want.name});
        const found = for (out.tensors.items) |t| {
            if (std.mem.eql(u8, t.name, expected)) break true;
        } else false;
        if (!found) std.debug.print("{s} is missing from the output\n", .{expected});
        try testing.expect(found);
    }
}

test "a llama-shaped checkpoint with no usable config.json is refused unless -H keeps its names" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_noconfig_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;

    const config_path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir});
    const body =
        \\"num_hidden_layers":1,"hidden_size":8,"intermediate_size":8,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-5,"vocab_size":4,
        \\ "max_position_embeddings":128}
    ;
    // A llama-shaped model type with no rename table here.
    try writeFile(io, config_path, try std.fmt.allocPrint(a, "{{\"model_type\":\"olmo\",{s}", .{body}));
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    // Exactly llama's HF detection set - the shape Mistral, Gemma and SmolLM
    // all share.
    const header =
        \\{"model.embed_tokens.weight":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]},
        \\"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[8,8],"data_offsets":[64,192]},
        \\"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[8,8],"data_offsets":[192,320]},
        \\"model.layers.0.mlp.gate_proj.weight":{"dtype":"F16","shape":[8,8],"data_offsets":[320,448]},
        \\"model.layers.0.mlp.down_proj.weight":{"dtype":"F16","shape":[8,8],"data_offsets":[448,576]}}
    ;
    var file_bytes: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.len), .little);
    try file_bytes.appendSlice(a, &len_buf);
    try file_bytes.appendSlice(a, header);
    try file_bytes.appendNTimes(a, 0, 576);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeFile(io, src_path, file_bytes.items);

    const opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "out",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };

    // Name detection says "llama", but olmo has no rename table here, and HF
    // names under no architecture load nowhere: refuse.
    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    try testing.expectError(error.HfConfigUnusable, prepareConversion(&f1, opts, gpa, a));

    // No config.json at all is the same case.
    try std.Io.Dir.cwd().deleteFile(io, config_path);
    var f1n = try st.init(src_path, io, gpa, a, false, false);
    defer f1n.deinit();
    try testing.expectError(error.HfConfigUnusable, prepareConversion(&f1n, opts, gpa, a));

    // -H asks for the HF-named file outright, and gets it.
    var f2h = try st.init(src_path, io, gpa, a, false, false);
    defer f2h.deinit();
    var keep = opts;
    keep.hf_names = true;
    const prep_keep = try prepareConversion(&f2h, keep, gpa, a);
    try testing.expect(!prep_keep.arch.gguf_arch_id);
    try testing.expectEqual(@as(usize, 0), prep_keep.arch.hf_model_type.len);
    try testing.expectEqual(@as(usize, 5), prep_keep.model_tensors.items.len);
    for (prep_keep.model_tensors.items) |t| {
        try testing.expect(std.mem.startsWith(u8, t.name, "model."));
    }

    // A config.json naming a type we map takes the same tensors through.
    try writeFile(io, config_path, try std.fmt.allocPrint(a, "{{\"model_type\":\"llama\",{s}", .{body}));
    var f3 = try st.init(src_path, io, gpa, a, false, false);
    defer f3.deinit();
    const prep = try prepareConversion(&f3, opts, gpa, a);
    var found = false;
    for (prep.model_tensors.items) |t| {
        if (std.mem.eql(u8, t.name, "blk.0.attn_q.weight")) found = true;
    }
    try testing.expect(found);

    // Mistral is llama to llama.cpp: same names, same Q/K permute, same arch.
    try writeFile(io, config_path, try std.fmt.allocPrint(a, "{{\"model_type\":\"mistral\",{s}", .{body}));
    var fm = try st.init(src_path, io, gpa, a, false, false);
    defer fm.deinit();
    const prep_m = try prepareConversion(&fm, opts, gpa, a);
    try testing.expectEqualStrings("llama", prep_m.arch.name);
    try testing.expect(prep_m.hf_permute != null);
    var found_m = false;
    for (prep_m.model_tensors.items) |t| {
        if (std.mem.eql(u8, t.name, "blk.0.attn_q.weight")) found_m = true;
    }
    try testing.expect(found_m);

    // ...and -H overrides that too, all the way to the written file: a
    // recognized config.json beside the weights does not force the rename on a
    // caller who asked for HF names.
    var f3h = try st.init(src_path, io, gpa, a, false, false);
    defer f3h.deinit();
    var keep_named = keep;
    keep_named.output_name = "keep";
    try convert(&f3h, keep_named, gpa, a);
    var out = try gguf.init(try std.fmt.allocPrint(a, "{s}/keep.gguf", .{dir}), io, gpa, a, false);
    defer out.deinit();
    try testing.expectEqual(@as(usize, 5), out.tensors.items.len);
    for (out.tensors.items) |t| {
        try testing.expect(std.mem.startsWith(u8, t.name, "model."));
    }
    if (out.metadata.get("general.architecture")) |v| {
        try testing.expect(!std.mem.eql(u8, v.string, "llama"));
    }

    // An unreadable tokenizer beside a config.json we do understand is a
    // different thing to fix, and says so rather than blaming the config.
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), "{\"model\":{\"type\":\"WordPiece\"}}");
    var f4 = try st.init(src_path, io, gpa, a, false, false);
    defer f4.deinit();
    try testing.expectError(error.HfTokenizerUnreadable, prepareConversion(&f4, opts, gpa, a));
}

test "a config.json with no tokenizer anywhere writes a native text encoder" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_vocabless_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;

    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":8,"intermediate_size":8,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":1e-6,
        \\ "vocab_size":4,"max_position_embeddings":128,"rope_theta":1000000}
    );
    const header =
        \\{"model.embed_tokens.weight":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]},
        \\"model.layers.0.self_attn.q_proj.weight":{"dtype":"F16","shape":[8,8],"data_offsets":[64,192]},
        \\"model.layers.0.self_attn.k_proj.weight":{"dtype":"F16","shape":[4,8],"data_offsets":[192,256]},
        \\"model.layers.0.self_attn.q_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[256,264]},
        \\"model.layers.0.self_attn.k_norm.weight":{"dtype":"F16","shape":[4],"data_offsets":[264,272]},
        \\"model.layers.0.mlp.gate_proj.weight":{"dtype":"F16","shape":[8,8],"data_offsets":[272,400]},
        \\"model.norm.weight":{"dtype":"F16","shape":[8],"data_offsets":[400,416]}}
    ;
    var file_bytes: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.len), .little);
    try file_bytes.appendSlice(a, &len_buf);
    try file_bytes.appendSlice(a, header);
    try file_bytes.appendNTimes(a, 0, 416);
    const src_path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{dir});
    try writeFile(io, src_path, file_bytes.items);

    const opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "te",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
    var f = try st.init(src_path, io, gpa, a, false, false);
    defer f.deinit();
    try convert(&f, opts, gpa, a);

    var out = try gguf.init(try std.fmt.allocPrint(a, "{s}/te.gguf", .{dir}), io, gpa, a, false);
    defer out.deinit();
    try testing.expectEqual(@as(usize, 7), out.tensors.items.len);
    var found_q = false;
    for (out.tensors.items) |t| {
        try testing.expect(!std.mem.startsWith(u8, t.name, "model."));
        if (std.mem.eql(u8, t.name, "blk.0.attn_q.weight")) found_q = true;
    }
    try testing.expect(found_q);
    // ComfyUI-GGUF keys its text loader off exactly this string.
    try testing.expectEqualStrings("qwen3", out.metadata.get("general.architecture").?.string);
    try testing.expectEqual(@as(i64, 1), out.metadata.get("qwen3.block_count").?.integer);
    try testing.expectEqual(@as(i64, 4), out.metadata.get("qwen3.attention.key_length").?.integer);
    var it = out.metadata.iterator();
    while (it.next()) |e| try testing.expect(!std.mem.startsWith(u8, e.key_ptr.*, "tokenizer."));
}

/// F32 safetensors whose element i of spec k holds (k * 1000 + i) * scale, so a byte
/// compare after conversion shows which source element landed where.
const StSpec = struct { name: []const u8, dims: []const usize };

fn writeF32Safetensors(io: std.Io, a: std.mem.Allocator, path: []const u8, specs: []const StSpec, first_index: usize, scale: f32) !void {
    var header: std.ArrayList(u8) = .empty;
    var data: std.ArrayList(u8) = .empty;
    try header.append(a, '{');
    for (specs, 0..) |sp, k| {
        var n: usize = 1;
        for (sp.dims) |d| n *= d;
        const start = data.items.len;
        for (0..n) |i| {
            const v: f32 = @as(f32, @floatFromInt((first_index + k) * 1000 + i)) * scale;
            try data.appendSlice(a, std.mem.asBytes(&v));
        }
        if (k > 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{sp.name});
        for (sp.dims, 0..) |d, di| try header.print(a, "{s}{d}", .{ if (di > 0) "," else "", d });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ start, data.items.len });
    }
    try header.append(a, '}');
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try file.writeStreamingAll(io, &len_buf);
    try file.writeStreamingAll(io, header.items);
    try file.writeStreamingAll(io, data.items);
}

test "HF MoE experts stack into one tensor per projection, across shards and from the fused layout" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_moe_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"qwen3_moe","num_hidden_layers":1,"hidden_size":4,"intermediate_size":6,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"rms_norm_eps":1e-6,
        \\ "vocab_size":4,"max_position_embeddings":64,"num_experts":2,"num_experts_per_tok":1,
        \\ "moe_intermediate_size":3}
    );
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    const L = "model.layers.0.";
    const dense = [_]StSpec{
        .{ .name = "model.embed_tokens.weight", .dims = &.{ 4, 4 } },
        .{ .name = L ++ "self_attn.q_proj.weight", .dims = &.{ 4, 4 } },
        .{ .name = L ++ "self_attn.k_proj.weight", .dims = &.{ 2, 4 } },
        .{ .name = L ++ "self_attn.v_proj.weight", .dims = &.{ 2, 4 } },
        .{ .name = L ++ "self_attn.o_proj.weight", .dims = &.{ 4, 4 } },
        .{ .name = L ++ "self_attn.q_norm.weight", .dims = &.{2} },
        .{ .name = L ++ "self_attn.k_norm.weight", .dims = &.{2} },
        .{ .name = L ++ "input_layernorm.weight", .dims = &.{4} },
        .{ .name = L ++ "post_attention_layernorm.weight", .dims = &.{4} },
        .{ .name = L ++ "mlp.gate.weight", .dims = &.{ 2, 4 } },
        .{ .name = "model.norm.weight", .dims = &.{4} },
    };
    // Expert 0 in the first shard, expert 1 in the second, and the second
    // listed first within its file.
    const e0 = [_]StSpec{
        .{ .name = L ++ "mlp.experts.0.gate_proj.weight", .dims = &.{ 3, 4 } },
        .{ .name = L ++ "mlp.experts.0.up_proj.weight", .dims = &.{ 3, 4 } },
        .{ .name = L ++ "mlp.experts.0.down_proj.weight", .dims = &.{ 4, 3 } },
    };
    const e1 = [_]StSpec{
        .{ .name = L ++ "mlp.experts.1.down_proj.weight", .dims = &.{ 4, 3 } },
        .{ .name = L ++ "mlp.experts.1.gate_proj.weight", .dims = &.{ 3, 4 } },
        .{ .name = L ++ "mlp.experts.1.up_proj.weight", .dims = &.{ 3, 4 } },
    };
    // Spec indices: dense 0..10, e0 11..13 (shard 1 continues the count), e1 20..22.
    try writeF32Safetensors(io, a, try std.fmt.allocPrint(a, "{s}/model-00001-of-00002.safetensors", .{dir}), &(dense ++ e0), 0, 1);
    try writeF32Safetensors(io, a, try std.fmt.allocPrint(a, "{s}/model-00002-of-00002.safetensors", .{dir}), &e1, 20, 1);
    var idx: std.ArrayList(u8) = .empty;
    try idx.appendSlice(a, "{\"weight_map\":{");
    for (dense ++ e0, 0..) |sp, i| try idx.print(a, "{s}\"{s}\":\"model-00001-of-00002.safetensors\"", .{ if (i > 0) "," else "", sp.name });
    for (e1) |sp| try idx.print(a, ",\"{s}\":\"model-00002-of-00002.safetensors\"", .{sp.name});
    try idx.appendSlice(a, "}}");
    const idx_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{dir});
    try writeFile(io, idx_path, idx.items);

    const readG = struct {
        fn f(g: anytype, name: []const u8, alloc: std.mem.Allocator) !struct { t: types.Tensor, v: []const f32 } {
            for (g.tensors.items) |t| if (std.mem.eql(u8, t.name, name)) {
                const buf = try alloc.alloc(u8, @intCast(t.size));
                const file = try g.openFileForTensor(t.name);
                _ = try file.readPositionalAll(g.io, buf, t.offset + g.current_data_begin);
                return .{ .t = t, .v = @alignCast(std.mem.bytesAsSlice(f32, buf)) };
            };
            return error.TestExpectedTensor;
        }
    }.f;

    var opts = ConvertOptions{
        .io = io,
        .path = idx_path,
        .filetype = .gguf,
        .datatype = .f32,
        .template_path = null,
        .output_dir = dir,
        .output_name = "moe",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
    {
        var f = try st.init(idx_path, io, gpa, a, false, false);
        defer f.deinit();
        try convert(&f, opts, gpa, a);
        var g = try gguf.init(try std.fmt.allocPrint(a, "{s}/moe.gguf", .{dir}), io, gpa, a, false);
        defer g.deinit();

        // 11 dense tensors, and 6 expert weights become 3.
        try testing.expectEqual(@as(usize, 14), g.tensors.items.len);
        try testing.expectEqualStrings("qwen3moe", g.metadata.get("general.architecture").?.string);
        try testing.expectEqual(@as(i64, 2), g.metadata.get("qwen3moe.expert_count").?.integer);
        try testing.expectEqual(@as(i64, 1), g.metadata.get("qwen3moe.expert_used_count").?.integer);
        try testing.expectEqual(@as(i64, 3), g.metadata.get("qwen3moe.expert_feed_forward_length").?.integer);

        const gate = try readG(&g, "blk.0.ffn_gate_exps.weight", a);
        try testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, gate.t.dims);
        for (gate.v, 0..) |v, i| {
            const want: f32 = if (i < 12) @floatFromInt(11 * 1000 + i) else @floatFromInt(21 * 1000 + i - 12);
            try testing.expectEqual(want, v);
        }
        const down = try readG(&g, "blk.0.ffn_down_exps.weight", a);
        try testing.expectEqualSlices(usize, &.{ 2, 4, 3 }, down.t.dims);
        try testing.expectEqual(@as(f32, 13 * 1000), down.v[0]);
        try testing.expectEqual(@as(f32, 20 * 1000 + 11), down.v[23]);
        const router = try readG(&g, "blk.0.ffn_gate_inp.weight", a);
        try testing.expectEqualSlices(usize, &.{ 2, 4 }, router.t.dims);
        try testing.expectEqual(@as(f32, 9 * 1000 + 7), router.v[7]);
    }

    // Newer transformers write one fused [E, 2*ff, embd] gate_up and one
    // [E, embd, ff] down per layer; llama.cpp wants the gate and up halves apart.
    const fused = [_]StSpec{
        .{ .name = L ++ "mlp.experts.gate_up_proj", .dims = &.{ 2, 6, 4 } },
        .{ .name = L ++ "mlp.experts.down_proj", .dims = &.{ 2, 4, 3 } },
    };
    const fdir = try std.fmt.allocPrint(a, "{s}/fused", .{dir});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, fdir, .default_dir);
    try std.Io.Dir.cwd().copyFile(try std.fmt.allocPrint(a, "{s}/config.json", .{dir}), std.Io.Dir.cwd(), try std.fmt.allocPrint(a, "{s}/config.json", .{fdir}), io, .{});
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{fdir}), test_bpe_tokenizer);
    const fpath = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{fdir});
    try writeF32Safetensors(io, a, fpath, &(dense ++ fused), 0, 1);
    opts.path = fpath;
    opts.output_dir = fdir;
    {
        var f = try st.init(fpath, io, gpa, a, false, false);
        defer f.deinit();
        try convert(&f, opts, gpa, a);
        var g = try gguf.init(try std.fmt.allocPrint(a, "{s}/moe.gguf", .{fdir}), io, gpa, a, false);
        defer g.deinit();
        try testing.expectEqual(@as(usize, 14), g.tensors.items.len);
        const gate = try readG(&g, "blk.0.ffn_gate_exps.weight", a);
        const up = try readG(&g, "blk.0.ffn_up_exps.weight", a);
        try testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, gate.t.dims);
        try testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, up.t.dims);
        // Expert x's gate is rows 0..3 of its [6, 4] slab, up rows 3..6.
        for (0..2) |x| for (0..12) |i| {
            try testing.expectEqual(@as(f32, @floatFromInt(11 * 1000 + x * 24 + i)), gate.v[x * 12 + i]);
            try testing.expectEqual(@as(f32, @floatFromInt(11 * 1000 + x * 24 + 12 + i)), up.v[x * 12 + i]);
        };
        const down = try readG(&g, "blk.0.ffn_down_exps.weight", a);
        try testing.expectEqualSlices(usize, &.{ 2, 4, 3 }, down.t.dims);
        try testing.expectEqual(@as(f32, 12 * 1000 + 23), down.v[23]);
    }
}

test "an HF qwen3_5_moe checkpoint converts: hybrid rewrites with equal head counts, fused experts, shared expert" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_q35moe_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;
    // Two K heads and two V heads, so the V reorder has nothing to move and
    // only the decay and norm rewrites are left to show. No intermediate_size:
    // MoE configs may leave it out.
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text",
        \\ "num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,
        \\ "head_dim":2,"rms_norm_eps":1e-6,"vocab_size":4,"max_position_embeddings":64,
        \\ "linear_num_value_heads":2,"linear_num_key_heads":2,"linear_value_head_dim":2,
        \\ "linear_key_head_dim":2,"linear_conv_kernel_dim":4,"layer_types":["linear_attention"],
        \\ "num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":3,
        \\ "shared_expert_intermediate_size":5,
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.5,"mrope_section":[1,0,0]}}}
    );
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    const L = "model.language_model.layers.0.";
    const specs = [_]StSpec{
        .{ .name = "model.language_model.embed_tokens.weight", .dims = &.{ 4, 4 } },
        .{ .name = L ++ "linear_attn.in_proj_qkv.weight", .dims = &.{ 12, 4 } },
        .{ .name = L ++ "linear_attn.in_proj_z.weight", .dims = &.{ 4, 4 } },
        .{ .name = L ++ "linear_attn.in_proj_a.weight", .dims = &.{ 2, 4 } },
        .{ .name = L ++ "linear_attn.in_proj_b.weight", .dims = &.{ 2, 4 } },
        .{ .name = L ++ "linear_attn.conv1d.weight", .dims = &.{ 12, 1, 4 } },
        .{ .name = L ++ "linear_attn.norm.weight", .dims = &.{2} },
        .{ .name = L ++ "linear_attn.out_proj.weight", .dims = &.{ 4, 4 } },
        .{ .name = L ++ "linear_attn.A_log", .dims = &.{2} }, // spec 8
        .{ .name = L ++ "linear_attn.dt_bias", .dims = &.{2} },
        .{ .name = L ++ "input_layernorm.weight", .dims = &.{4} }, // spec 10
        .{ .name = L ++ "post_attention_layernorm.weight", .dims = &.{4} },
        .{ .name = L ++ "mlp.gate.weight", .dims = &.{ 2, 4 } },
        .{ .name = L ++ "mlp.experts.gate_up_proj", .dims = &.{ 2, 6, 4 } },
        .{ .name = L ++ "mlp.experts.down_proj", .dims = &.{ 2, 4, 3 } },
        .{ .name = L ++ "mlp.shared_expert.gate_proj.weight", .dims = &.{ 5, 4 } },
        .{ .name = L ++ "mlp.shared_expert.up_proj.weight", .dims = &.{ 5, 4 } },
        .{ .name = L ++ "mlp.shared_expert.down_proj.weight", .dims = &.{ 4, 5 } },
        .{ .name = L ++ "mlp.shared_expert_gate.weight", .dims = &.{ 1, 4 } },
        .{ .name = "model.language_model.norm.weight", .dims = &.{4} },
    };
    // Small values so -exp(A_log) stays finite.
    const scale: f32 = 1.0 / 16384.0;
    const src_path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{dir});
    try writeF32Safetensors(io, a, src_path, &specs, 0, scale);

    var src_f = try st.init(src_path, io, gpa, a, false, false);
    defer src_f.deinit();
    try convert(&src_f, .{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f32,
        .template_path = null,
        .output_dir = dir,
        .output_name = "q35moe",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    }, gpa, a);

    var g = try gguf.init(try std.fmt.allocPrint(a, "{s}/q35moe.gguf", .{dir}), io, gpa, a, false);
    defer g.deinit();
    const md = g.metadata;
    try testing.expectEqualStrings("qwen35moe", md.get("general.architecture").?.string);
    try testing.expectEqual(@as(i64, 2), md.get("qwen35moe.expert_count").?.integer);
    try testing.expectEqual(@as(i64, 3), md.get("qwen35moe.expert_feed_forward_length").?.integer);
    try testing.expectEqual(@as(i64, 5), md.get("qwen35moe.expert_shared_feed_forward_length").?.integer);
    try testing.expect(!md.contains("qwen35moe.feed_forward_length"));
    try testing.expect(md.contains("qwen35moe.ssm.group_count"));
    // 18 non-expert tensors, and the fused pair becomes three.
    try testing.expectEqual(@as(usize, 21), g.tensors.items.len);

    const readG = struct {
        fn f(gg: anytype, name: []const u8, alloc: std.mem.Allocator) !struct { t: types.Tensor, v: []const f32 } {
            for (gg.tensors.items) |t| if (std.mem.eql(u8, t.name, name)) {
                const buf = try alloc.alloc(u8, @intCast(t.size));
                const file = try gg.openFileForTensor(t.name);
                _ = try file.readPositionalAll(gg.io, buf, t.offset + gg.current_data_begin);
                return .{ .t = t, .v = @alignCast(std.mem.bytesAsSlice(f32, buf)) };
            };
            return error.TestExpectedTensor;
        }
    }.f;
    const src = struct {
        fn v(k: usize, i: usize, s: f32) f32 {
            return @as(f32, @floatFromInt(k * 1000 + i)) * s;
        }
    };
    const ssm_a = try readG(&g, "blk.0.ssm_a", a);
    for (ssm_a.v, 0..) |x, i| try testing.expectEqual(-@exp(src.v(8, i, scale)), x);
    const attn_norm = try readG(&g, "blk.0.attn_norm.weight", a);
    for (attn_norm.v, 0..) |x, i| try testing.expectEqual(src.v(10, i, scale) + 1, x);
    const ssm_norm = try readG(&g, "blk.0.ssm_norm.weight", a);
    for (ssm_norm.v, 0..) |x, i| try testing.expectEqual(src.v(6, i, scale), x);
    const conv = try readG(&g, "blk.0.ssm_conv1d.weight", a);
    try testing.expectEqualSlices(usize, &.{ 12, 4 }, conv.t.dims);
    const up = try readG(&g, "blk.0.ffn_up_exps.weight", a);
    try testing.expectEqualSlices(usize, &.{ 2, 3, 4 }, up.t.dims);
    try testing.expectEqual(src.v(13, 24 + 12, scale), up.v[12]);
    _ = try readG(&g, "blk.0.ffn_gate_inp_shexp.weight", a);
    _ = try readG(&g, "blk.0.ffn_down_shexp.weight", a);
}

test "a GGUF carrying HF names of its own is written through, not un-permuted" {
    testing.log_level = .err; // conversions log every tensor at info level

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_hfsource_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const Spec = struct { name: []const u8, shape: []const usize };
    const specs = [_]Spec{
        .{ .name = "model.embed_tokens.weight", .shape = &.{ 4, 8 } },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 8, 8 } },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 8, 8 } },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &.{ 8, 8 } },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 8, 8 } },
    };

    var total: usize = 0;
    for (specs) |s| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        total += n;
    }
    const data = try a.alloc(u8, total * 2);
    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{");
    var off: u64 = 0;
    var q_src: []u8 = undefined;
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        const cols: usize = n / s.shape[0];
        const f16s: []f16 = @ptrCast(@alignCast(data[off..][0 .. n * 2]));
        // Distinct per row, so a row permute of Q would show up below.
        for (0..s.shape[0]) |r| {
            for (0..cols) |c| f16s[r * cols + c] = @floatFromInt((r + 1) * (i + 1));
        }
        if (i == 1) q_src = data[off..][0 .. n * 2];
        if (i > 0) try header.appendSlice(a, ",");
        try header.print(a, "\"{s}\":{{\"dtype\":\"F16\",\"shape\":[", .{s.name});
        for (s.shape, 0..) |d, j| {
            if (j > 0) try header.appendSlice(a, ",");
            try header.print(a, "{}", .{d});
        }
        try header.print(a, "],\"data_offsets\":[{[0]},{[1]}]}}", .{ off, off + n * 2 });
        off += n * 2;
    }
    try header.appendSlice(a, "}");

    var st_file: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try st_file.appendSlice(a, &len_buf);
    try st_file.appendSlice(a, header.items);
    try st_file.appendSlice(a, data);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    {
        const file = try std.Io.Dir.cwd().createFile(io, src_path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, st_file.items);
    }

    // -H keeps the HF names on the way into the GGUF, which is the only way to
    // get the thing under test: a GGUF whose tensors are not named blk.*.
    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    try convert(&f1, .{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "hfnamed",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
    }, gpa, a);

    // Those HF names are the source's own, not a rename this tool performed, so
    // -H has nothing to do: no rename, and above all no RoPE un-permute of rows
    // that were never permuted.
    const gguf_path = try std.fmt.allocPrint(a, "{s}/hfnamed.gguf", .{dir});
    var fg = try gguf.init(gguf_path, io, gpa, a, false);
    defer fg.deinit();
    try convert(&fg, .{
        .io = io,
        .path = gguf_path,
        .filetype = .safetensors,
        .datatype = .F16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "back",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .hf_names = true,
    }, gpa, a);

    var fo = try st.init(try std.fmt.allocPrint(a, "{s}/back.safetensors", .{dir}), io, gpa, a, false, false);
    defer fo.deinit();
    var q: ?types.Tensor = null;
    for (fo.tensors.items) |t| {
        if (std.mem.eql(u8, t.name, "model.layers.0.self_attn.q_proj.weight")) q = t;
    }
    const qt = q orelse return error.MissingTensor;
    const buf = try a.alloc(u8, @intCast(qt.size));
    const file = try fo.openFileForTensor(qt.name);
    _ = try file.readPositionalAll(fo.io, buf, qt.offset + fo.current_data_begin);
    try testing.expectEqualSlices(u8, q_src, buf);
}

test "a safetensors carrying native names of its own is not permuted again" {
    testing.log_level = .err; // conversions log every tensor at info level

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_nativesource_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;

    // 16 wide over 2 heads is a head_dim of 8, so the permute swaps halves of 4.
    // A half of 2 would make it its own inverse and hide a second application.
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"llama","num_hidden_layers":1,"hidden_size":16,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-5,"vocab_size":100,"max_position_embeddings":128}
        ++ "\n");
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    const Spec = struct { name: []const u8, shape: []const usize };
    const specs = [_]Spec{
        .{ .name = "model.embed_tokens.weight", .shape = &.{ 100, 16 } },
        .{ .name = "model.norm.weight", .shape = &.{16} },
        .{ .name = "lm_head.weight", .shape = &.{ 100, 16 } },
        .{ .name = "model.layers.0.input_layernorm.weight", .shape = &.{16} },
        .{ .name = "model.layers.0.self_attn.q_proj.weight", .shape = &.{ 16, 16 } },
        .{ .name = "model.layers.0.self_attn.k_proj.weight", .shape = &.{ 8, 16 } },
        .{ .name = "model.layers.0.self_attn.v_proj.weight", .shape = &.{ 8, 16 } },
        .{ .name = "model.layers.0.self_attn.o_proj.weight", .shape = &.{ 16, 16 } },
        .{ .name = "model.layers.0.mlp.gate_proj.weight", .shape = &.{ 16, 16 } },
        .{ .name = "model.layers.0.mlp.up_proj.weight", .shape = &.{ 16, 16 } },
        .{ .name = "model.layers.0.mlp.down_proj.weight", .shape = &.{ 16, 16 } },
        .{ .name = "model.layers.0.post_attention_layernorm.weight", .shape = &.{16} },
    };

    var total: usize = 0;
    for (specs) |s| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        total += n;
    }
    const data = try a.alloc(u8, total * 2);
    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{");
    var off: u64 = 0;
    var q_src: []u8 = undefined;
    for (specs, 0..) |s, i| {
        var n: usize = 1;
        for (s.shape) |d| n *= d;
        const rows: usize = s.shape[0];
        const cols: usize = n / rows;
        const f16s: []f16 = @ptrCast(@alignCast(data[off..][0 .. n * 2])); // offsets stay 2-byte aligned
        // Distinct per row, so any row permute shows up in the bytes below.
        for (0..rows) |r| {
            for (0..cols) |c| f16s[r * cols + c] = @floatFromInt((r + 1) * (i + 1));
        }
        if (i == 4) q_src = data[off..][0 .. n * 2];
        if (i > 0) try header.appendSlice(a, ",");
        try header.print(a, "\"{s}\":{{\"dtype\":\"F16\",\"shape\":[", .{s.name});
        for (s.shape, 0..) |d, j| {
            if (j > 0) try header.appendSlice(a, ",");
            try header.print(a, "{}", .{d});
        }
        try header.print(a, "],\"data_offsets\":[{[0]},{[1]}]}}", .{ off, off + n * 2 });
        off += n * 2;
    }
    try header.appendSlice(a, "}");

    var st_file: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.items.len), .little);
    try st_file.appendSlice(a, &len_buf);
    try st_file.appendSlice(a, header.items);
    try st_file.appendSlice(a, data);
    const src_path = try std.fmt.allocPrint(a, "{s}/m.safetensors", .{dir});
    try writeFile(io, src_path, st_file.items);

    const readT = struct {
        fn f(src: anytype, name: []const u8, alloc: std.mem.Allocator) ![]u8 {
            for (src.tensors.items) |t| {
                if (!std.mem.eql(u8, t.name, name)) continue;
                const buf = try alloc.alloc(u8, @intCast(t.size));
                const file = try src.openFileForTensor(t.name);
                _ = try file.readPositionalAll(src.io, buf, t.offset + src.current_data_begin);
                return buf;
            }
            return error.MissingTensor;
        }
    }.f;

    const convertTo = struct {
        fn f(path: []const u8, ft: types.FileType, dt: types.DataType, name: []const u8, d: []const u8, io2: std.Io, g: std.mem.Allocator, al: std.mem.Allocator) !void {
            var f1 = try st.init(path, io2, g, al, false, false);
            defer f1.deinit();
            try convert(&f1, .{
                .io = io2,
                .path = path,
                .filetype = ft,
                .datatype = dt,
                .template_path = null,
                .output_dir = d,
                .output_name = name,
                .threads = 1,
                .skip_sensitivity = true,
                .quantization_aggressiveness = 0,
            }, g, al);
        }
    }.f;

    // The HF checkpoint -> GGUF: the Q rows are permuted on the way in.
    try convertTo(src_path, .gguf, .f16, "first", dir, io, gpa, a);
    const first_path = try std.fmt.allocPrint(a, "{s}/first.gguf", .{dir});
    var f_first = try gguf.init(first_path, io, gpa, a, false);
    defer f_first.deinit();
    const first_q = try readT(&f_first, "blk.0.attn_q.weight", a);
    try testing.expect(!std.mem.eql(u8, q_src, first_q));

    // That GGUF back out without -H: native blk.* names, landing beside the
    // config.json that makes the next conversion find an HF model again.
    {
        var fg = try gguf.init(first_path, io, gpa, a, false);
        defer fg.deinit();
        try convert(&fg, .{
            .io = io,
            .path = first_path,
            .filetype = .safetensors,
            .datatype = .F16,
            .template_path = null,
            .output_dir = dir,
            .output_name = "native",
            .threads = 1,
            .skip_sensitivity = true,
            .quantization_aggressiveness = 0,
        }, gpa, a);
    }

    // Its rows are already half-split, so converting it back must permute
    // nothing: same bytes as the first GGUF, not the permute applied twice.
    const native_path = try std.fmt.allocPrint(a, "{s}/native.safetensors", .{dir});
    try convertTo(native_path, .gguf, .f16, "second", dir, io, gpa, a);
    var f_second = try gguf.init(try std.fmt.allocPrint(a, "{s}/second.gguf", .{dir}), io, gpa, a, false);
    defer f_second.deinit();
    try testing.expectEqualSlices(u8, first_q, try readT(&f_second, "blk.0.attn_q.weight", a));
}

test "a file that is not the model its neighbouring config.json describes converts as itself" {
    testing.log_level = .err;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Unique dir per run (parallel binaries would race on a shared path).
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_sibling_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const writeFile = struct {
        fn f(io2: std.Io, path: []const u8, bytes: []const u8) !void {
            const file = try std.Io.Dir.cwd().createFile(io2, path, .{ .truncate = true });
            defer file.close(io2);
            try file.writeStreamingAll(io2, bytes);
        }
    }.f;

    try writeFile(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}),
        \\{"model_type":"llama","num_hidden_layers":1,"hidden_size":8,"intermediate_size":8,
        \\ "num_attention_heads":2,"num_key_value_heads":1,"rms_norm_eps":1e-5,"vocab_size":4,
        \\ "max_position_embeddings":128}
    );
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}), test_bpe_tokenizer);

    // A LoRA adapter in the model's own repo: nothing here maps in either
    // direction, so the rename must leave the file alone rather than drop
    // every tensor and write an empty GGUF.
    const header =
        \\{"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":{"dtype":"F16","shape":[4,8],"data_offsets":[0,64]},
        \\"base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":{"dtype":"F16","shape":[8,4],"data_offsets":[64,128]}}
    ;
    var file_bytes: std.ArrayList(u8) = .empty;
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, @intCast(header.len), .little);
    try file_bytes.appendSlice(a, &len_buf);
    try file_bytes.appendSlice(a, header);
    try file_bytes.appendNTimes(a, 0, 128);
    const src_path = try std.fmt.allocPrint(a, "{s}/adapter_model.safetensors", .{dir});
    try writeFile(io, src_path, file_bytes.items);

    const opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "out",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .allow_unknown_arch = true,
    };

    var f1 = try st.init(src_path, io, gpa, a, false, false);
    defer f1.deinit();
    const prep = try prepareConversion(&f1, opts, gpa, a);
    try testing.expectEqual(@as(?HfLlm.Model, null), prep.hf);
    try testing.expectEqual(@as(usize, 2), prep.model_tensors.items.len);

    // The same adapter beside a tokenizer that would stop the model it belongs
    // to: an empty merge table is the refusal with no -u to go on from, and it
    // is about that model's vocabulary, which this file does not carry.
    try writeFile(io, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{dir}),
        \\{"pre_tokenizer":{"type":"Split","behavior":"Isolated","pattern":{"Regex":"\\w+"}},
        \\ "model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3},"merges":[]}}
    );
    var f2 = try st.init(src_path, io, gpa, a, false, false);
    defer f2.deinit();
    const prep2 = try prepareConversion(&f2, opts, gpa, a);
    try testing.expectEqual(@as(?HfLlm.Model, null), prep2.hf);
    try testing.expectEqual(@as(usize, 2), prep2.model_tensors.items.len);
}

test "computeOutputPath names a template's output after its average bit rate" {
    var threaded = std.Io.Threaded.init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_tplname_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const tpl_path = try std.fmt.allocPrint(a, "{s}/t.json", .{dir});
    {
        const file = try std.Io.Dir.cwd().createFile(io, tpl_path, .{ .truncate = true });
        defer file.close(io);
        // f32 norms beside quantized weights: the narrowest is the fewest bits
        // per weight, not the lowest enum value, so f32 must not win by being
        // listed first or by any ordering accident.
        try file.writeStreamingAll(io,
            \\{"tensors":{
            \\ "output_norm.weight":{"shape":[8],"type":"f32"},
            \\ "blk.0.ffn_down.weight":{"shape":[8,8],"type":"q6_k"},
            \\ "blk.0.attn_q.weight":{"shape":[8,8],"type":"iq3_s"},
            \\ "blk.0.ffn_up.weight":{"shape":[8,8],"type":"q4_k"}
            \\}}
        );
    }

    var opts = testOpts(.q4_k);
    opts.io = io;
    opts.datatype = null;
    opts.template_path = tpl_path;
    opts.path = "model.gguf";
    opts.output_dir = dir;

    // Weighted by element count: the 8-element f32 norm barely counts against
    // three 64-element weights, so the label lands on the size class the file
    // actually occupies rather than on its narrowest or widest member.
    const got = try computeOutputPath(opts, false, a);
    try testing.expect(std.mem.endsWith(u8, got, "model-q5_k.gguf"));

    // q5_0 and q5_k are both 5.5 bits per weight, so the tie must resolve to
    // the k-quant rather than to whichever happens to be listed first.
    try testing.expectEqual(types.DataType.q5_k, nearestTypeByBpw(5.5));
    try testing.expectEqual(types.DataType.q4_k, nearestTypeByBpw(4.5));
    try testing.expectEqual(types.DataType.iq4_xs, nearestTypeByBpw(4.25));
    try testing.expectEqual(types.DataType.f32, nearestTypeByBpw(32));

    // An explicit -d still wins; the template only fills the gap.
    opts.datatype = .q5_k;
    const forced = try computeOutputPath(opts, false, a);
    try testing.expect(std.mem.endsWith(u8, forced, "model-q5_k.gguf"));

    // An unreadable template leaves the old fallback rather than failing a name.
    opts.datatype = null;
    opts.template_path = "/nonexistent/none.json";
    const fallback = try computeOutputPath(opts, false, a);
    try testing.expect(std.mem.endsWith(u8, fallback, "model-f16.gguf"));
}

test "stripDtypeSuffix removes a trailing dtype and nothing else" {
    const s = stripDtypeSuffix;
    // ggufy's own output names, in both namespaces.
    try testing.expectEqualStrings("model", s("model-bf16"));
    try testing.expectEqualStrings("model", s("model-BF16"));

    // Other tools' casing and llama.cpp's mix labels.
    try testing.expectEqualStrings("model", s("model-Q4_K"));
    try testing.expectEqualStrings("model", s("model-Q4_K_M"));
    try testing.expectEqualStrings("model", s("model-Q5_K_S"));
    try testing.expectEqualStrings("model", s("model-Q3_K_L"));
    try testing.expectEqualStrings("model", s("model-IQ4_XS"));
    try testing.expectEqualStrings("model", s("model-IQ2_XXS"));
    try testing.expectEqualStrings("model", s("model-F16"));
    try testing.expectEqualStrings("Meta-Llama-3-8B", s("Meta-Llama-3-8B-Q6_K"));

    // A two-letter tail that is not a mix label must not make a non-type match.
    try testing.expectEqualStrings("model-foo_xs", s("model-foo_xs"));
    try testing.expectEqualStrings("model-q4_zz", s("model-q4_zz"));
    try testing.expectEqualStrings("swift-qwen38", s("swift-qwen38-bf16"));
    try testing.expectEqualStrings("model", s("model-q4_k"));
    try testing.expectEqualStrings("model", s("model-iq3_s"));
    try testing.expectEqualStrings("model", s("model-f32"));

    // Only one layer comes off per conversion, which is all that can accumulate.
    try testing.expectEqualStrings("model-bf16", s("model-bf16-q4_k"));

    // A name that merely ends in a dash-separated word keeps it.
    try testing.expectEqualStrings("Qwen3-4B", s("Qwen3-4B"));
    try testing.expectEqualStrings("llama-2", s("llama-2"));
    try testing.expectEqualStrings("krea2_turbo", s("krea2_turbo"));
    try testing.expectEqualStrings("Illustrious-XL-v2.0", s("Illustrious-XL-v2.0"));

    // Degenerate inputs are left alone rather than emptied.
    try testing.expectEqualStrings("", s(""));
    try testing.expectEqualStrings("-bf16", s("-bf16"));
    try testing.expectEqualStrings("model-", s("model-"));
}

test "liftToFloor lifts an IQ target off the k-quant line, not to the source dtype" {
    // iq3_s is 3.44 bits against a 6-bit floor, and the IQ family stops at
    // iq4_xs (4.25), so the lift has to leave the family.
    try testing.expectEqual(types.DataType.q6_k, liftToFloor(.iq3_s, .bits6).?);
    try testing.expectEqual(types.DataType.q6_k, liftToFloor(.iq2_xxs, .bits6).?);
    try testing.expectEqual(types.DataType.q8_0, liftToFloor(.iq4_xs, .bits8).?);
    // An IQ type already clearing the floor is left alone.
    try testing.expectEqual(types.DataType.iq4_xs, liftToFloor(.iq4_xs, .bits4).?);

    // The k-quant line itself is unchanged.
    try testing.expectEqual(types.DataType.q6_k, liftToFloor(.q4_k, .bits6).?);
    try testing.expectEqual(types.DataType.q4_k, liftToFloor(.q4_k, .bits4).?);

    // An over-long name is not a rung, and must not assert.
    try testing.expectError(error.UnknownQuantizationType, QuantizationLevel.fromString("INT4_CONVROT_SR"));
}

test "the ComfyUI embedding rule applies to safetensors output only" {
    // Applying it to GGUF only inflated files: ComfyUI-GGUF and llama.cpp both
    // read a quantized table, which is why they quantize token_embd routinely.
    var dims = [_]usize{ 4096, 1024 };
    const n = 4096 * 1024;
    for ([_][]const u8{
        "model.diffusion_model.llm_adapter.embed.weight",
        "text_model.embed_tokens.weight",
        "transformer.wte.weight",
    }) |name| {
        var t_gguf: types.Tensor = .{ .name = name, .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
        var o_gguf = testOpts(.q4_k);
        o_gguf.filetype = .gguf;
        try assignTensorType(&t_gguf, n, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, o_gguf, false, null, testing.allocator);
        try testing.expectEqualStrings("q4_k", t_gguf.type);

        var t_st: types.Tensor = .{ .name = name, .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
        var o_st = testOpts(.q4_k);
        o_st.filetype = .safetensors;
        o_st.datatype = .INT8;
        try assignTensorType(&t_st, n, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, o_st, false, null, testing.allocator);
        try testing.expect(!std.mem.eql(u8, "INT8", t_st.type));
    }

    // A patch/positional embedder is Conv or Linear and quantizes on both paths.
    var patch: types.Tensor = .{ .name = "patch_embed.proj.weight", .type = "BF16", .dims = &dims, .size = 0, .offset = 0 };
    var o = testOpts(.q4_k);
    o.filetype = .gguf;
    try assignTensorType(&patch, n, &imagearch.generic_arch, QUANTIZATION_THRESHOLD, o, false, null, testing.allocator);
    try testing.expectEqualStrings("q4_k", patch.type);
}

// A micro Qwen checkpoint with a vision tower, as transformers saved it, in a
// fresh directory with the test tokenizer beside it. See gen_vision_fixtures.py.
fn writeVisionFixture(io: std.Io, a: std.mem.Allocator, comptime name: []const u8, tag: []const u8) ![]const u8 {
    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(tag.ptr));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_vision_{s}_{s}_{x}", .{ name, tag, token });
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    const files = .{
        .{ "model.safetensors", @embedFile("test_fixtures/vision/" ++ name ++ "/model.safetensors") },
        .{ "config.json", @embedFile("test_fixtures/vision/" ++ name ++ "/config.json") },
        .{ "preprocessor_config.json", @embedFile("test_fixtures/vision/" ++ name ++ "/preprocessor_config.json") },
        .{ "tokenizer.json", test_bpe_tokenizer },
    };
    inline for (files) |entry| {
        const file = try std.Io.Dir.cwd().createFile(io, try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, entry[0] }), .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, entry[1]);
    }
    return dir;
}

fn jsonValuesEqual(x: std.json.Value, y: std.json.Value) bool {
    return switch (x) {
        .bool => |b| y == .bool and y.bool == b,
        .integer => |n| y == .integer and y.integer == n,
        .float => |f| y == .float and y.float == f,
        .string => |s| y == .string and std.mem.eql(u8, s, y.string),
        .array => |arr| blk: {
            if (y != .array or y.array.items.len != arr.items.len) break :blk false;
            for (arr.items, y.array.items) |p, q| if (!jsonValuesEqual(p, q)) break :blk false;
            break :blk true;
        },
        else => false,
    };
}

fn readTensorBytes(g: anytype, t: types.Tensor, alloc: std.mem.Allocator) ![]u8 {
    const buf = try alloc.alloc(u8, @intCast(t.size));
    const file = try g.openFileForTensor(t.name);
    _ = try file.readPositionalAll(g.io, buf, t.offset + g.current_data_begin);
    return buf;
}

fn expectMmprojMatchesLlamaCpp(comptime name: []const u8) !void {
    testing.log_level = .err;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const dir = try writeVisionFixture(io, a, name, "mmproj");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const src_path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{dir});
    // A k-quant text target still gets an f16 mmproj, the type the fixture has.
    const opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .q4_k,
        .template_path = null,
        .output_dir = dir,
        .output_name = "vl",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
    const predicted = blk: {
        var fp = try st.init(src_path, io, gpa, a, false, false);
        defer fp.deinit();
        break :blk try predictOutputSize(&fp, opts, gpa, a);
    };
    {
        var f = try st.init(src_path, io, gpa, a, false, false);
        defer f.deinit();
        try convert(&f, opts, gpa, a);
    }
    // The estimate covers both files the conversion writes.
    const text_size = (try std.Io.Dir.cwd().statFile(io, try std.fmt.allocPrint(a, "{s}/vl.gguf", .{dir}), .{})).size;
    const mm_size = (try std.Io.Dir.cwd().statFile(io, try std.fmt.allocPrint(a, "{s}/mmproj-vl.gguf", .{dir}), .{})).size;
    try testing.expectEqual(text_size + mm_size, predicted);

    const ref_path = try std.fmt.allocPrint(a, "{s}/ref.gguf", .{dir});
    {
        const file = try std.Io.Dir.cwd().createFile(io, ref_path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, @embedFile("test_fixtures/vision/" ++ name ++ "/mmproj-f16.gguf"));
    }
    var ref = try gguf.init(ref_path, io, gpa, a, false);
    defer ref.deinit();
    var got = try gguf.init(try std.fmt.allocPrint(a, "{s}/mmproj-vl.gguf", .{dir}), io, gpa, a, false);
    defer got.deinit();

    // The name parts come from the directory, and provenance is this tool's.
    const skipped = [_][]const u8{ "general.name", "general.basename", "general.size_label", "general.finetune", "converted_by", "converter_url", "converter_note" };
    var it = ref.metadata.iterator();
    var compared: usize = 0;
    while (it.next()) |e| {
        const skip = for (skipped) |k| {
            if (std.mem.eql(u8, k, e.key_ptr.*)) break true;
        } else false;
        if (skip) continue;
        const mine = got.metadata.get(e.key_ptr.*) orelse {
            std.debug.print("missing key {s}\n", .{e.key_ptr.*});
            return error.TestExpectedEqual;
        };
        if (!jsonValuesEqual(e.value_ptr.*, mine)) {
            std.debug.print("key {s} differs\n", .{e.key_ptr.*});
            return error.TestExpectedEqual;
        }
        compared += 1;
    }
    try testing.expect(compared >= 15);

    try testing.expectEqual(ref.tensors.items.len, got.tensors.items.len);
    for (ref.tensors.items) |rt| {
        const gt = for (got.tensors.items) |t| {
            if (std.mem.eql(u8, t.name, rt.name)) break t;
        } else {
            std.debug.print("missing tensor {s}\n", .{rt.name});
            return error.TestExpectedEqual;
        };
        try testing.expectEqualStrings(rt.type, gt.type);
        try testing.expectEqualSlices(usize, rt.dims, gt.dims);
        const rb = try readTensorBytes(&ref, rt, a);
        const gb = try readTensorBytes(&got, gt, a);
        if (!std.mem.eql(u8, rb, gb)) {
            std.debug.print("tensor {s} bytes differ\n", .{rt.name});
            return error.TestExpectedEqual;
        }
    }
}

test "a Qwen3-VL checkpoint's vision tower becomes the mmproj llama.cpp's converter writes" {
    try expectMmprojMatchesLlamaCpp("qwen3vl");
}

test "a Qwen2.5-VL checkpoint's vision tower becomes the mmproj llama.cpp's converter writes" {
    try expectMmprojMatchesLlamaCpp("qwen25vl");
}

fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

// The rebuilt tensor `got` as f32, whatever float type the writer chose for it.
fn tensorAsF32(g: anytype, t: types.Tensor, alloc: std.mem.Allocator) ![]f32 {
    const bytes = try readTensorBytes(g, t, alloc);
    var n: usize = 1;
    for (t.dims) |d| n *= d;
    const out = try alloc.alloc(f32, n);
    if (std.mem.eql(u8, t.type, "F32")) {
        @memcpy(out, @as([]align(1) const f32, std.mem.bytesAsSlice(f32, bytes)));
    } else if (std.mem.eql(u8, t.type, "BF16")) {
        for (out, std.mem.bytesAsSlice(u16, bytes)) |*o, b| o.* = bf16ToF32(b);
    } else if (std.mem.eql(u8, t.type, "F16")) {
        for (out, std.mem.bytesAsSlice(f16, bytes)) |*o, h| o.* = h;
    } else return error.TestUnexpectedType;
    return out;
}

fn expectVisionRoundTrip(comptime name: []const u8) !void {
    testing.log_level = .err;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const dir = try writeVisionFixture(io, a, name, "roundtrip");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const src_path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{dir});
    const base = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f32,
        .template_path = null,
        .output_dir = dir,
        .output_name = "vl",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
        .allow_upscale = true,
    };
    {
        var f = try st.init(src_path, io, gpa, a, false, false);
        defer f.deinit();
        try convert(&f, base, gpa, a);
    }

    var src = try st.init(src_path, io, gpa, a, false, false);
    defer src.deinit();
    const text_path = try std.fmt.allocPrint(a, "{s}/vl.gguf", .{dir});

    // Lossless through f32, then again with llama.cpp's own f16 mmproj in
    // place of this tool's, whose tower comes back f16-rounded.
    for ([_]bool{ false, true }, [_][]const u8{ "hf", "hf_llamacpp" }) |llamacpp, out_name| {
        if (llamacpp) {
            const file = try std.Io.Dir.cwd().createFile(io, try std.fmt.allocPrint(a, "{s}/mmproj-vl.gguf", .{dir}), .{ .truncate = true });
            defer file.close(io);
            try file.writeStreamingAll(io, @embedFile("test_fixtures/vision/" ++ name ++ "/mmproj-f16.gguf"));
        }
        const out_dir = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, out_name });
        {
            var g = try gguf.init(text_path, io, gpa, a, false);
            defer g.deinit();
            var opts = base;
            opts.path = text_path;
            opts.filetype = .safetensors;
            opts.datatype = .F32;
            opts.hf_names = true;
            opts.output_dir = out_dir;
            opts.output_name = null;
            try convert(&g, opts, gpa, a);
        }
        const hf_path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{out_dir});
        var got = try st.init(hf_path, io, gpa, a, false, false);
        defer got.deinit();
        try testing.expectEqual(src.tensors.items.len, got.tensors.items.len);
        for (src.tensors.items) |stt| {
            const gt = for (got.tensors.items) |t| {
                if (std.mem.eql(u8, t.name, stt.name)) break t;
            } else {
                std.debug.print("{s} is missing from the rebuilt checkpoint\n", .{stt.name});
                return error.TestExpectedEqual;
            };
            try testing.expectEqualSlices(usize, stt.dims, gt.dims);
            const want_bits = try readTensorBytes(&src, stt, a);
            const have = try tensorAsF32(&got, gt, a);
            // Its converter keeps 1-D tensors and the position table f32.
            const rounded = llamacpp and HfLlm.isVisionTensor(stt.name) and stt.dims.len >= 2 and
                !std.mem.endsWith(u8, stt.name, "pos_embed.weight");
            for (std.mem.bytesAsSlice(u16, want_bits), have, 0..) |b, h, i| {
                const x = bf16ToF32(b);
                const want: f32 = if (rounded) @as(f16, @floatCast(x)) else x;
                if (want != h) {
                    std.debug.print("{s}[{d}]: {d} came back as {d}\n", .{ stt.name, i, want, h });
                    return error.TestExpectedEqual;
                }
            }
        }

        // The vision_config transformers builds the tower from.
        const cfg_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{out_dir}), a, .limited(1 << 20));
        const cfg = try std.json.parseFromSliceLeaky(std.json.Value, a, cfg_bytes, .{});
        const ref = try std.json.parseFromSliceLeaky(std.json.Value, a, @embedFile("test_fixtures/vision/" ++ name ++ "/config.json"), .{});
        const vc = cfg.object.get("vision_config").?.object;
        const rvc = ref.object.get("vision_config").?.object;
        for ([_][]const u8{ "depth", "hidden_size", "intermediate_size", "num_heads", "out_hidden_size", "patch_size", "spatial_merge_size", "temporal_patch_size", "num_position_embeddings", "deepstack_visual_indexes", "fullatt_block_indexes", "hidden_act" }) |k| {
            const want = rvc.get(k) orelse continue;
            const have = vc.get(k) orelse {
                std.debug.print("vision_config.{s} is missing\n", .{k});
                return error.TestExpectedEqual;
            };
            if (!jsonValuesEqual(want, have)) {
                std.debug.print("vision_config.{s} differs\n", .{k});
                return error.TestExpectedEqual;
            }
        }
        try testing.expectEqualStrings(ref.object.get("architectures").?.array.items[0].string, cfg.object.get("architectures").?.array.items[0].string);
    }
}

test "a Qwen3-VL text GGUF and its mmproj rebuild the HF checkpoint" {
    try expectVisionRoundTrip("qwen3vl");
}

test "a Qwen2.5-VL text GGUF and its mmproj rebuild the HF checkpoint" {
    try expectVisionRoundTrip("qwen25vl");
}

test "a Qwen3-VL text GGUF with no mmproj beside it is refused for -H" {
    testing.log_level = .err;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const dir = try writeVisionFixture(io, a, "qwen3vl", "lone");
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const src_path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{dir});
    var opts = ConvertOptions{
        .io = io,
        .path = src_path,
        .filetype = .gguf,
        .datatype = .f16,
        .template_path = null,
        .output_dir = dir,
        .output_name = "vl",
        .threads = 1,
        .skip_sensitivity = true,
        .quantization_aggressiveness = 0,
    };
    {
        var f = try st.init(src_path, io, gpa, a, false, false);
        defer f.deinit();
        try convert(&f, opts, gpa, a);
    }
    try std.Io.Dir.cwd().deleteFile(io, try std.fmt.allocPrint(a, "{s}/mmproj-vl.gguf", .{dir}));
    const text_path = try std.fmt.allocPrint(a, "{s}/vl.gguf", .{dir});
    var g = try gguf.init(text_path, io, gpa, a, false);
    defer g.deinit();
    opts.path = text_path;
    opts.filetype = .safetensors;
    opts.datatype = .F16;
    opts.hf_names = true;
    opts.output_dir = try std.fmt.allocPrint(a, "{s}/hf", .{dir});
    try testing.expectError(error.MmprojMissing, convert(&g, opts, gpa, a));
}
