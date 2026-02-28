
//! Convert.zig — SafeTensors → GGUF conversion logic.
//! Extracted from main.zig so the convert command has its own home.

const std = @import("std");
const st = @import("Safetensor.zig");
const types = @import("types.zig");
const gguf = @import("Gguf.zig");
const imagearch = @import("ImageArch.zig");

// ============================================================================
// Public API
// ============================================================================

/// All options that drive a conversion, collected from CLI args.
pub const ConvertOptions = struct {
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

/// Entry point: convert a SafeTensors file according to `opts`.
/// `f` is the already-opened SafeTensors handle.
pub fn convert(
    f: *st,
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !void {
    // --- Detect architecture --------------------------------------------------
    const arch = try imagearch.detectArchFromTensorsOrError(f.tensors.items, allocator);
    const threshold = arch.threshhold orelse QUANTIZATION_THRESHOLD;
    std.log.info("Detected architecture: {s}", .{arch.name});

    // --- Filter and normalise tensor list ------------------------------------
    var model_tensors = try filterAndStripTensors(f, arch, arena_alloc);

    // --- Assign quantization types (template or auto) -------------------------
    var template_metadata: ?std.json.ObjectMap = null;
    if (opts.template_path) |tp| {
        template_metadata = try applyTemplate(tp, &model_tensors, arena_alloc);
    } else {
        try assignQuantTypes(&model_tensors, arch, threshold, opts, arena_alloc);
    }

    // --- Shape fix ------------------------------------------------------------
    var extra_metadata = std.StringArrayHashMap(std.json.Value).init(arena_alloc);
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

    // --- Write output ---------------------------------------------------------
    switch (opts.filetype) {
        .gguf => try writeGguf(
            f,
            model_tensors,
            arch,
            template_metadata,
            extra_metadata,
            opts,
            allocator,
            arena_alloc,
        ),
        .safetensors => try writeSafetensors(
            f,
            model_tensors,
            arch,
            template_metadata,
            extra_metadata,
            opts,
            allocator,
            arena_alloc,
        ),
    }
}

// ============================================================================
// Quantization level helpers
// ============================================================================

const QUANTIZATION_THRESHOLD = 1024;

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
        return std.meta.stringToEnum(QuantizationLevel, std.ascii.lowerString(&lower, s)) orelse error.UnknownQuantizationType;
    }
};

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

/// Returns a new list containing only the tensors that should be converted,
/// with name prefixes stripped.
fn filterAndStripTensors(
    f: *st,
    arch: *const imagearch.Arch,
    arena_alloc: std.mem.Allocator,
) !std.ArrayList(types.Tensor) {
    var model_tensors = try std.ArrayList(types.Tensor).initCapacity(arena_alloc, f.tensors.items.len);

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
    template_path: []const u8,
    model_tensors: *std.ArrayList(types.Tensor),
    arena_alloc: std.mem.Allocator,
) !?std.json.ObjectMap {
    std.log.info("Using template {s}", .{template_path});

    const t_file = try std.fs.cwd().openFile(template_path, .{});
    defer t_file.close();
    const t_content = try t_file.readToEndAlloc(arena_alloc, 10 * 1024 * 1024);
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
            const new_t = try applyTemplateEntry(src, target_name, target_info, arena_alloc);
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

    const ggml_type = try gguf.GgmlType.fromString(target_type);
    const bs = ggml_type.getBlockSize();
    if (bs > 1 and source_elements % bs != 0) {
        std.log.err("Tensor {s} cannot be quantized to type {s}. Element count {} is not a multiple of block size {}", .{ target_name, target_type, source_elements, bs });
        return error.InvalidSizeForQuantization;
    }

    var new_t = src;
    new_t.name = target_name;
    new_t.dims = target_dims;
    new_t.type = @tagName(ggml_type);
    new_t.size = ggml_type.calcSizeInBytes(target_elements);
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

    if (!opts.skip_sensitivity) {
        if (opts.sensitivities_path) |sp| {
            // User-supplied file overrides the built-in one.
            std.log.info("Using user-supplied sensitivities file: {s}", .{sp});
            const sens_file = try std.fs.cwd().openFile(sp, .{});
            defer sens_file.close();
            const sens_content = try sens_file.readToEndAlloc(arena_alloc, 32 * 1024 * 1024);
            sensitivities = try std.json.parseFromSlice(std.json.Value, arena_alloc, sens_content, .{});
            use_sensitivity = true;
        } else if (arch.sensitivities.len > 1) {
            // Fall back to built-in sensitivities for this architecture.
            std.log.debug("Using built-in sensitivities file for {s}", .{arch.name});
            sensitivities = try std.json.parseFromSlice(std.json.Value, arena_alloc, arch.sensitivities, .{});
            use_sensitivity = true;
        }
    }

    var offset: u64 = 0;
    for (model_tensors.items) |*t| {
        var num_elements: u64 = 1;
        for (t.dims) |d| num_elements *= d;

        try assignTensorType(t, num_elements, arch, threshold, opts, use_sensitivity, if (use_sensitivity) &sensitivities else null);

        // f64 is unsupported in ComfyUI GGUF — downcast to f32.
        // TODO: make this optional via a flag.
        if (std.mem.eql(u8, t.type, "f64") or std.mem.eql(u8, t.type, "F64")) {
            std.log.info("Downcasting unsupported f64 to f32 for tensor {s}", .{t.name});
            t.type = "f32";
            const fat_type = try gguf.GgmlType.fromString(t.type);
            t.size = fat_type.calcSizeInBytes(num_elements);
        }

        var dims_buf = try std.ArrayList(u8).initCapacity(arena_alloc, 5);
        for (t.dims, 0..) |d, i| {
            if (i > 0) try dims_buf.appendSlice(arena_alloc, ", ");
            try std.fmt.format(dims_buf.writer(arena_alloc), "{}", .{d});
        }
        std.log.debug("Calculated size {} for type {s} with num elements {} with dims [{s}]", .{ t.size, t.type, num_elements, dims_buf.items });

        // TODO: make alignment configurable.
        const padding_len = (32 - (t.size % 32)) % 32;
        t.offset = offset;
        offset += t.size + padding_len;
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
) !void {
    // Architecture-specific overrides first.
    if (arch.shouldUpcast(t.name) and opts.filetype == .gguf) {
        std.log.info("Forcing layer {s} to f32 for compatability", .{t.name});
        const ggml_type = gguf.GgmlType.f32;
        t.type = @tagName(ggml_type);
        t.size = ggml_type.calcSizeInBytes(num_elements);
        return;
    }

    // Too small to quantize — leave it alone.
    if (num_elements < threshold) return;

    // High-precision tensors (e.g. norms, gates) — leave alone.
    if (arch.isHighPrecision(t.name)) return;

    // Apply the target datatype.
    const ttype = opts.datatype orelse return;
    if (opts.filetype == .gguf) {
        const ggml_type = gguf.GgmlType.fromString(@tagName(ttype)) catch unreachable;
        const bs = ggml_type.getBlockSize();

        if (bs > 1 and num_elements % bs != 0) {
            std.log.warn("Cannot convert tensor {s} to type {s} because {} is not a multiple of blocksize {}", .{ t.name, @tagName(ggml_type), num_elements, bs });
            return;
        }
    }

    if (use_sensitivity) {
        try applySensitivityQuantization(t, num_elements, ttype,opts.quantization_aggressiveness, resolvedFamilies(opts), sensitivities.?);
    } else {
        std.log.debug("Will convert tensor {s} to type {s}", .{ t.name, @tagName(ttype) });
        t.type = @tagName(ttype);
        t.size = ttype.calcSizeInBytes(num_elements);
    }
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

/// Reshapes qualifying tensors to (N/256, 256) for ComfyUI compatibility and
/// records their original shapes under "comfy.gguf.orig_shape.<name>" in
/// `extra_metadata`.
fn applyShapeFix(
    model_tensors: *std.ArrayList(types.Tensor),
    extra_metadata: *std.StringArrayHashMap(std.json.Value),
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
        try extra_metadata.put(key, .{ .array = orig_shape_arr });

        // Reshape to (N/256, 256).
        var new_dims = try arena_alloc.alloc(usize, 2);
        new_dims[0] = n_elements / 256;
        new_dims[1] = 256;
        t.dims = new_dims;

        std.log.info("Applied shape fix to {s}: new shape {{ {}, {} }}", .{ t.name, new_dims[0], new_dims[1] });
    }
}

fn writeGguf(
    f: *st,
    model_tensors: std.ArrayList(types.Tensor),
    arch: *const imagearch.Arch,
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.StringArrayHashMap(std.json.Value),
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !void {
    // --- Resolve output path -------------------------------------------------
    const dir_path = if (opts.output_dir) |od| od else std.fs.path.dirname(opts.path) orelse ".";

    var cwd = std.fs.cwd();
    const dir_result = try cwd.makePathStatus(dir_path);
    if (dir_result == .created) std.log.info("Created directory {s}", .{dir_path});

    const base_name = if (opts.output_name) |on| on else blk: {
        const stem = std.fs.path.stem(opts.path);
        const dtype_str = @tagName(opts.datatype orelse types.DataType.f16);
        break :blk try std.fmt.allocPrint(arena_alloc, "{s}-{s}", .{ stem, dtype_str });
    };

    const out_filename = try std.fs.path.join(
        arena_alloc,
        &[_][]const u8{ dir_path, try std.fmt.allocPrint(arena_alloc, "{s}.gguf", .{base_name}) },
    );

    // --- Initialise GGUF writer ----------------------------------------------
    var out_gguf = try gguf.init(out_filename, allocator, arena_alloc, true);
    defer out_gguf.deinit();
    out_gguf.tensors = model_tensors;

    // Standard metadata.
    try out_gguf.metadata.put(try arena_alloc.dupe(u8, "general.architecture"), .{ .string = arch.name });
    try out_gguf.metadata.put(try arena_alloc.dupe(u8, "general.quantization_version"), .{ .integer = 2 });
    // TODO: determine from the target dtype.
    try out_gguf.metadata.put(try arena_alloc.dupe(u8, "general.file_type"), .{ .integer = 7 });

    // Template metadata takes priority over source-file metadata.
    if (template_metadata) |meta| {
        var it = meta.iterator();
        while (it.next()) |entry| {
            if (!out_gguf.metadata.contains(entry.key_ptr.*))
                try out_gguf.metadata.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }
    } else if (f.metadata) |meta| {
        var it = meta.iterator();
        while (it.next()) |entry| {
            if (!out_gguf.metadata.contains(entry.key_ptr.*))
                try out_gguf.metadata.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }
    }

    // Extra metadata (e.g. shape-fix records).
    var extra_it = extra_metadata.iterator();
    while (extra_it.next()) |entry| {
        if (!out_gguf.metadata.contains(entry.key_ptr.*))
            try out_gguf.metadata.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
    }

    try out_gguf.saveWithSTData(f, opts.threads);
    std.log.info("Converted to {s}", .{out_filename});
}

fn writeSafetensors(
    f: *st,
    model_tensors: std.ArrayList(types.Tensor),
    arch: *const imagearch.Arch,
    template_metadata: ?std.json.ObjectMap,
    extra_metadata: std.StringArrayHashMap(std.json.Value),
    opts: ConvertOptions,
    allocator: std.mem.Allocator,
    arena_alloc: std.mem.Allocator,
) !void {
    // --- Resolve output path -------------------------------------------------
    const dir_path = if (opts.output_dir) |od| od else std.fs.path.dirname(opts.path) orelse ".";

    var cwd = std.fs.cwd();
    const dir_result = try cwd.makePathStatus(dir_path);
    if (dir_result == .created) std.log.info("Created directory {s}", .{dir_path});

    const base_name = if (opts.output_name) |on| on else blk: {
        const stem = std.fs.path.stem(opts.path);
        const dtype_str = @tagName(opts.datatype orelse types.DataType.F16);
        break :blk try std.fmt.allocPrint(arena_alloc, "{s}-{s}", .{ stem, dtype_str });
    };

    const out_filename = try std.fs.path.join(
        arena_alloc,
        &[_][]const u8{ dir_path, try std.fmt.allocPrint(arena_alloc, "{s}.safetensors", .{base_name}) },
    );

    // --- Initialise Safetensors writer ----------------------------------------------
    var out_st = try st.init(out_filename, allocator, arena_alloc, true, true);
    defer out_st.deinit();
    out_st.tensors = model_tensors;

    // Copy any metadata from the source, if there is any
    out_st.metadata = try f.metadata.?.clone();

    // Template metadata takes priority over source-file metadata.
    if (template_metadata) |meta| {
        var it = meta.iterator();
        while (it.next()) |entry| {
            if (!out_st.metadata.?.contains(entry.key_ptr.*))
                try out_st.metadata.?.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }
    } else if (f.metadata) |meta| {
        var it = meta.iterator();
        while (it.next()) |entry| {
            if (!out_st.metadata.?.contains(entry.key_ptr.*))
                try out_st.metadata.?.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
        }
    }

    // Extra metadata
    var extra_it = extra_metadata.iterator();
    while (extra_it.next()) |entry| {
        if (!out_st.metadata.?.contains(entry.key_ptr.*))
            try out_st.metadata.?.put(try arena_alloc.dupe(u8, entry.key_ptr.*), entry.value_ptr.*);
    }

    try out_st.saveWithSTData(f, opts.threads);
    std.log.info("Converted to {s}", .{out_filename});


    _ = arch;
}