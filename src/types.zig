const std = @import("std");

pub const FileType = enum {
    safetensors,
    gguf,

    pub fn detect(reader: *std.io.Reader, allocator: std.mem.Allocator) !FileType {
        const file_header = try reader.readAlloc(allocator, 8);

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
};