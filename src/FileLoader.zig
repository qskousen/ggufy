const std = @import("std");
const types = @import("types.zig");
const st = @import("Safetensor.zig");
const gguf = @import("Gguf.zig");

pub const TensorFile = struct {
    type: types.FileType = .safetensors,
    tensors: std.ArrayList(types.Tensor) = undefined,
    metadata: ?std.json.ObjectMap = null,

    pub fn loadFile(allocator: std.mem.Allocator, arena_alloc: std.mem.Allocator, path: []const u8) !TensorFile {
        var ret: TensorFile = .{};
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });

        var read_buffer: [8]u8 = undefined;
        var reader = file.reader(&read_buffer);
        ret.type = types.FileType.detect_from_file(&reader.interface, allocator) catch types.FileType.safetensors;
        file.close();

        // load data
        switch (ret.type) {
            .safetensors => {
                var f = try st.init(path, allocator, arena_alloc, false, false);
                ret.metadata = f.metadata;
                ret.tensors = f.tensors;
                f.deinit();
            },
            .gguf => {
                var f = try gguf.init(path, allocator, arena_alloc, false);
                ret.metadata = f.metadata;
                ret.tensors = f.tensors;
                f.deinit();
            },
        }

        return ret;
    }
};