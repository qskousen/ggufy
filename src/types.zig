const std = @import("std");

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
        return Tensor {
            .name = try allocator.dupe(u8, self.name),
            .type = try allocator.dupe(u8, self.type),
            .dims = try allocator.dupe(usize, self.dims),
            .size = self.size,
            .offset = self.offset,
            .source_path = allocator.dupe(u8, self.source_path.?) catch null,
        };
    }
};