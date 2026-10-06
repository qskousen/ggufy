const std = @import("std");
const Safetensors = @import("Safetensor.zig");
const gguf = @import("Gguf.zig");

pub const FileType = enum {
    safetensors,
    gguf,

    pub fn detect_from_file(reader: *std.Io.Reader, allocator: std.mem.Allocator) !FileType {
        _ = allocator;
        var file_header: [8]u8 = undefined;
        reader.readSliceAll(&file_header) catch return error.UnknownFormat;

        // Check for GGUF magic "GGUF" followed by version (2 bytes) and tensor count
        if (std.mem.eql(u8, file_header[0..4], "GGUF")) {
            return .gguf;
        }

        // For safetensors, the first 8 bytes are a u64 length
        // Try to interpret as safetensors header length
        const possible_length = std.mem.readInt(u64, file_header[0..8], .little);
        if (possible_length > 0 and possible_length < 128 * 1024 * 1024) { // 128 MiB cap
            return .safetensors;
        }

        return error.UnknownFormat;
    }

    pub fn parse_from_string(str: []const u8) !FileType {
        inline for (std.meta.fields(FileType)) |field| {
            if (std.mem.eql(u8, str, field.name)) {
                return @field(FileType, field.name);
            }
        }
        return error.UnknownFormat;
    }
};

/// `runs` spans of `elem_count` elements of `tensor`, the first `elem_offset`
/// elements in and each `stride` past the last. With `out_offset` null the
/// spans land back to back after the previous segment's; otherwise span i
/// lands at out_offset + i * out_stride, which is how two segments interleave.
pub const StackSegment = struct {
    tensor: Tensor,
    elem_offset: usize,
    elem_count: usize,
    runs: usize = 1,
    stride: usize = 0,
    out_offset: ?usize = null,
    out_stride: usize = 0,
};

/// An output tensor no single source tensor holds: HF's per-expert MoE weights
/// stacked into llama.cpp's one [n_expert, rows, cols] tensor, one half of a
/// fused gate_up_proj, or a vision tensor split or joined along an axis. The
/// segments fill `dims` exactly once between them. `dtype` is what the stacked
/// tensor is typed as before type assignment; each segment is read as its own
/// tensor's type.
pub const ExpertStack = struct {
    name: []const u8,
    dims: []const usize,
    dtype: []const u8,
    segments: []const StackSegment,

    /// First segment's source name: collapseModelTensors puts the stacked
    /// tensor where this one was.
    pub fn anchor(self: ExpertStack) []const u8 {
        return self.segments[0].tensor.name;
    }

    pub fn uses(self: ExpertStack, source_name: []const u8) bool {
        for (self.segments) |seg| if (std.mem.eql(u8, seg.tensor.name, source_name)) return true;
        return false;
    }
};

pub const Tensor = struct {
    name: []const u8,
    type: []const u8,
    dims: []usize,
    size: u64,
    offset: u64,
    source_path: ?[]const u8 = null,

    pub fn dupe(self: Tensor, allocator: std.mem.Allocator) !Tensor {
        return Tensor{
            .name = try allocator.dupe(u8, self.name),
            .type = try allocator.dupe(u8, self.type),
            .dims = try allocator.dupe(usize, self.dims),
            .size = self.size,
            .offset = self.offset,
            .source_path = if (self.source_path) |sp| allocator.dupe(u8, sp) catch null else null,
        };
    }
};

/// Optional in-place rewrite of freshly-read source bytes, keyed by source
/// tensor name/dims. Used by the HF llama paths to apply the Q/K RoPE row
/// permutation (forward and reverse are separate positional inverses) around
/// (de/re)quantization. Writers hand `apply` plain float values: quantized
/// sources are dequantized to F32 first, since block-packed bytes have no
/// addressable rows. `matches`, when set, skips `apply` for tensors the
/// patch ignores, so writers do not dequantize tensors for nothing.
pub const SourcePatch = struct {
    pub const MatchesFn = *const fn (ctx: *anyopaque, name: []const u8) bool;
    ctx: *anyopaque,
    apply: *const fn (ctx: *anyopaque, alloc: std.mem.Allocator, name: []const u8, dims: []const usize, source_type: []const u8, bytes: []u8) anyerror!void,
    matches: ?MatchesFn = null,
    /// Hand `apply` F32 even when the source is already a float. A patch that
    /// only moves rows around does not care, but one that does arithmetic does:
    /// adding 1 to a bf16 norm in bf16 rounds to a different number than adding
    /// it after the widening the output performs anyway.
    force_f32: bool = false,
};

/// Per-column importance weights, looked up by output tensor. Type-erased so
/// the file writers do not depend on where the weights came from.
pub const ImatrixLookup = struct {
    ctx: *const anyopaque,
    get: *const fn (ctx: *const anyopaque, t: Tensor) ?[]const f32,

    pub fn forTensor(self: ImatrixLookup, t: Tensor) ?[]const f32 {
        return self.get(self.ctx, t);
    }
};

/// Represents data types from all known formats
pub const DataType = enum {
    // Safetensor types
    F8_E4M3,
    F8_E5M2,
    SCALED_F8_E4M3, // ComfyUI scaled FP8 cluster: F8_E4M3 weight + F32 global scale + comfy_quant
    F4_E2M1,
    MXFP4,
    MXFP8_E4M3,
    NVFP4,
    INT8, // ComfyUI int8_tensorwise cluster: I8 weight + F32 per-row scale + comfy_quant (no rotation)
    INT8_CONVROT, // ComfyUI ConvRot cluster: I8 weight (Hadamard-rotated) + F32 per-row scale + comfy_quant
    INT4_CONVROT, // ComfyUI convrot_w4a4 cluster: I8 nibble-packed signed 4-bit weight (Hadamard-rotated) + F32 per-row scale + comfy_quant
    INT4_CONVROT_SR, // Same on-disk convrot_w4a4 format as INT4_CONVROT, but quantized with stochastic rounding ("SR")
    ASYM_W4A8_INT8, // ComfyUI asym_w4a8_int8 cluster: rotated 4-bit codebook indices + fp8 per-group scale + F32 per-row scale + F32 codebook + comfy_quant
    W6A8_INT8, // ComfyUI w6a8_int8 cluster: same layout as ASYM_W4A8_INT8 with uniform 6-bit codes and no codebook
    BF16,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    // ggml types
    f32,
    f16,
    q4_0,
    q4_1,
    q4_2, // Support has been removed from gguf files
    q4_3, // Support has been removed from gguf files
    q5_0,
    q5_1,
    q8_0,
    q8_1,
    q2_k,
    q3_k,
    q4_k,
    q5_k,
    q6_k,
    q8_k,
    iq2_xxs,
    iq2_xs,
    iq3_xxs,
    iq1_s,
    iq4_nl,
    iq3_s,
    iq2_s,
    iq4_xs,
    i8,
    i16,
    i32,
    i64,
    f64,
    iq1_m,
    bf16,
    q4_0_4_4, // Support has been removed from gguf files
    q4_0_4_8, // Support has been removed from gguf files
    q4_0_8_8, // Support has been removed from gguf files
    tq1_0,
    tq2_0,
    iq4_nl_4_4, // Support has been removed from gguf files
    iq4_nl_4_8, // Support has been removed from gguf files
    iq4_nl_8_8, // Support has been removed from gguf files
    mxfp4,
    nvfp4,
    q1_0,
    count,

    pub fn fromString(value: []const u8) !DataType {
        return std.meta.stringToEnum(DataType, value) orelse error.InvalidDataType;
    }

    /// Comptime table of equivalent (safetensors, gguf) type pairs.
    /// Types with no cross-format equivalent (quantized gguf, FP8, unsigned ints) are omitted.
    const equivalence_table = [_][2]DataType{
        .{ .F16, .f16 },
        .{ .F32, .f32 },
        .{ .F64, .f64 },
        .{ .BF16, .bf16 },
        .{ .I8, .i8 },
        .{ .I16, .i16 },
        .{ .I32, .i32 },
        .{ .I64, .i64 },
    };

    /// Convert this DataType to the equivalent type for the given file format.
    /// Returns error.NoEquivalentType if no cross-format equivalent exists
    /// (e.g. quantized gguf types have no safetensors counterpart).
    pub fn forFormat(self: DataType, filetype: FileType) !DataType {
        if (self.formatType() == filetype) return self;
        for (equivalence_table) |pair| {
            if (self == pair[0]) return pair[1];
            if (self == pair[1]) return pair[0];
        }
        return error.NoEquivalentType;
    }

    /// Returns true if `self` and `target` (parsed from string) represent the same
    /// underlying data type across formats. Same-format types must be identical.
    /// Types with no cross-format equivalent (FP8, unsigned ints, quantized gguf types)
    /// return false rather than an error.
    pub fn equivalentType(self: DataType, target: []const u8) bool {
        const t = DataType.fromString(target) catch return false;

        if (self.formatType() == t.formatType()) return self == t;

        // Determine which is the safetensors type and which is the gguf type.
        const st_type = if (self.formatType() == .safetensors) self else t;
        const gg_type = if (self.formatType() == .gguf) self else t;

        for (equivalence_table) |pair| {
            if (pair[0] == st_type and pair[1] == gg_type) return true;
        }
        return false;
    }

    /// True for types whose payload is plain float values in either format and
    /// can be row-permuted directly; quantized and FP8 payloads must go through F32 first.
    pub fn isFloatType(self: DataType) bool {
        return switch (self) {
            .BF16, .F16, .F32, .F64, .bf16, .f16, .f32, .f64 => true,
            else => false,
        };
    }

    pub fn formatType(self: DataType) FileType {
        return switch (self) {
            .F8_E4M3, .F8_E5M2, .SCALED_F8_E4M3, .F4_E2M1, .MXFP4, .MXFP8_E4M3, .NVFP4, .INT8, .INT8_CONVROT, .INT4_CONVROT, .INT4_CONVROT_SR, .ASYM_W4A8_INT8, .W6A8_INT8, .BF16, .F16, .F32, .F64, .I8, .I16, .I32, .I64, .U8, .U16, .U32, .U64 => FileType.safetensors,
            .f32, .f16, .q4_0, .q4_1, .q4_2, .q4_3, .q5_0, .q5_1, .q8_0, .q8_1, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .i8, .i16, .i32, .i64, .f64, .iq1_m, .bf16, .q4_0_4_4, .q4_0_4_8, .q4_0_8_8, .tq1_0, .tq2_0, .iq4_nl_4_4, .iq4_nl_4_8, .iq4_nl_8_8, .mxfp4, .nvfp4, .q1_0, .count => FileType.gguf,
        };
    }

    /// Nominal bits per weight for the format, ignoring per-tensor scale overhead.
    pub fn nominalBits(self: DataType) u8 {
        return switch (self) {
            .q1_0, .iq1_s, .iq1_m => 1,
            .q2_k, .iq2_xxs, .iq2_xs, .iq2_s, .tq1_0, .tq2_0 => 2,
            .q3_k, .iq3_xxs, .iq3_s => 3,
            .F4_E2M1, .MXFP4, .NVFP4, .INT4_CONVROT, .INT4_CONVROT_SR, .ASYM_W4A8_INT8 => 4,
            .q4_0, .q4_1, .q4_2, .q4_3, .q4_k, .mxfp4, .nvfp4 => 4,
            .iq4_nl, .iq4_xs, .iq4_nl_4_4, .iq4_nl_4_8, .iq4_nl_8_8 => 4,
            .q4_0_4_4, .q4_0_4_8, .q4_0_8_8 => 4,
            .q5_0, .q5_1, .q5_k => 5,
            .q6_k, .W6A8_INT8 => 6,
            .F8_E4M3, .F8_E5M2, .SCALED_F8_E4M3, .MXFP8_E4M3, .INT8, .INT8_CONVROT => 8,
            .I8, .U8, .i8, .q8_0, .q8_1, .q8_k => 8,
            .BF16, .F16, .I16, .U16, .bf16, .f16, .i16 => 16,
            .F32, .I32, .U32, .f32, .i32 => 32,
            .F64, .I64, .U64, .f64, .i64 => 64,
            .count => 0,
        };
    }

    pub fn calcSizeInBytes(self: DataType, n_elements: u64) u64 {
        // SCALED_F8_E4M3 and INT8_CONVROT are cluster types; actual total size is set by
        // assignQuantType. Report just the weight bytes (1 per element) as a conservative
        // lower bound.
        if (self == .SCALED_F8_E4M3 or self == .INT8_CONVROT or self == .INT8) return n_elements;
        // These pack two 4-bit values per byte; report the packed weight bytes. Their scale
        // tensors are added on by clusterWriteSize, like every other cluster type.
        if (self == .INT4_CONVROT or self == .INT4_CONVROT_SR or self == .ASYM_W4A8_INT8) return (n_elements + 1) / 2;
        if (self == .W6A8_INT8) return n_elements / 4 * 3;
        return switch (self.formatType()) {
            .safetensors => {
                const t = Safetensors.DType.fromString(@tagName(self)) catch unreachable;
                return t.calcSizeInBytes(n_elements);
            },
            .gguf => {
                const t = gguf.GgmlType.fromString(@tagName(self)) catch unreachable;
                return t.calcSizeInBytes(n_elements);
            },
        };
    }
};
