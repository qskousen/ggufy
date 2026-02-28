const std = @import("std");
const Safetensors = @import("Safetensor.zig");
const gguf = @import("Gguf.zig");

pub const FileType = enum {
    safetensors,
    gguf,

    pub fn detect_from_file(reader: *std.io.Reader, allocator: std.mem.Allocator) !FileType {
        const file_header = try reader.readAlloc(allocator, 8);
        defer allocator.free(file_header);

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
            .source_path = allocator.dupe(u8, self.source_path.?) catch null,
        };
    }
};

/// Represents data types from all known formats
pub const DataType = enum {
    // Safetensor types
    FP8_E4M3,
    FP8_E5M2,
    BF16,
    FP16,
    FP32,
    FP64,
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
    count,

    pub fn fromString(value: []const u8) !DataType {
        var lower: [12]u8 = [_]u8{0} ** 12;
        return std.meta.stringToEnum(DataType, std.ascii.lowerString(&lower, value)) orelse error.InvalidDataType;
    }

    /// Comptime table of equivalent (safetensors, gguf) type pairs.
    /// Types with no cross-format equivalent (quantized gguf, FP8, unsigned ints) are omitted.
    const equivalence_table = [_][2]DataType{
        .{ .FP16, .f16  },
        .{ .FP32, .f32  },
        .{ .FP64, .f64  },
        .{ .BF16, .bf16 },
        .{ .I8,   .i8   },
        .{ .I16,  .i16  },
        .{ .I32,  .i32  },
        .{ .I64,  .i64  },
    };

    /// Returns true if `self` and `target` (parsed from string) represent the same
    /// underlying data type across formats. Same-format types must be identical.
    /// Types with no cross-format equivalent (FP8, unsigned ints, quantized gguf types)
    /// return false rather than an error.
    pub fn equivalentType(self: DataType, target: []const u8) bool {
        const t = DataType.fromString(target) catch return false;

        if (self.formatType() == t.formatType()) return self == t;

        // Determine which is the safetensors type and which is the gguf type.
        const st_type = if (self.formatType() == .safetensors) self else t;
        const gg_type = if (self.formatType() == .gguf)         self else t;

        for (equivalence_table) |pair| {
            if (pair[0] == st_type and pair[1] == gg_type) return true;
        }
        return false;
    }

    pub fn formatType(self: DataType) FileType {
        return switch (self) {
            .FP8_E4M3, .FP8_E5M2, .BF16, .FP16, .FP32, .FP64, .I8, .I16, .I32, .I64, .U8, .U16, .U32, .U64 => FileType.safetensors,
            .f32, .f16, .q4_0, .q4_1, .q4_2, .q4_3, .q5_0, .q5_1, .q8_0, .q8_1, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k, .iq2_xxs, .iq2_xs, .iq3_xxs, .iq1_s, .iq4_nl, .iq3_s, .iq2_s, .iq4_xs, .i8, .i16, .i32, .i64, .f64, .iq1_m, .bf16, .q4_0_4_4, .q4_0_4_8, .q4_0_8_8, .tq1_0, .tq2_0, .iq4_nl_4_4, .iq4_nl_4_8, .iq4_nl_8_8, .mxfp4, .count => FileType.gguf,
        };
    }

    pub fn calcSizeInBytes(self: DataType, n_elements: u64) u64 {
        return switch (self.formatType()) {
            .safetensors => {
                const t = Safetensors.DType.fromString(@tagName(self)) catch unreachable;
                return t.getSizeInBytes() * n_elements;
            },
            .gguf => {
                const t = gguf.GgmlType.fromString(@tagName(self)) catch unreachable;
                return t.calcSizeInBytes(n_elements);
            },
        };
    }
};
