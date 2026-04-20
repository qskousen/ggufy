const std = @import("std");
const types = @import("types.zig");
const st = @import("Safetensor.zig");
const gguf = @import("Gguf.zig");
const arch = @import("ImageArch.zig");

pub const TensorFile = struct {
    type: types.FileType = .safetensors,
    arch: ?arch.Arch = null,
    tensors: std.ArrayList(types.Tensor) = undefined,
    metadata: ?std.json.ObjectMap = null,
    /// Owns the parsed JSON arena that `metadata` slices point into.
    /// Must be freed before this TensorFile is discarded.
    _st_json_data: ?std.json.Parsed(std.json.Value) = null,
    sizeInBytes: u64 = 0,
    type_counts: std.HashMap(types.DataType,usize,std.hash_map.AutoContext(types.DataType), 80) = undefined,
    types_line: []u8 = "",

    pub fn deinit(self: *TensorFile) void {
        if (self._st_json_data) |*jd| jd.deinit();
        self._st_json_data = null;
    }

    pub fn loadFile(allocator: std.mem.Allocator, arena_alloc: std.mem.Allocator, path: []const u8) !TensorFile {
        var ret: TensorFile = .{};
        const file = try std.fs.cwd().openFile(path, .{ .mode = .read_only });
        ret.sizeInBytes = try file.getEndPos();

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
                // Transfer json_data ownership so metadata slices remain valid.
                // deinit() on TensorFile will free it.
                ret._st_json_data = f.json_data;
                f.json_data = null;
                f.deinit();
            },
            .gguf => {
                var f = try gguf.init(path, allocator, arena_alloc, false);
                ret.metadata = f.metadata;
                ret.tensors = f.tensors;
                f.deinit();
            },
        }

        ret.type_counts = std.AutoHashMap(types.DataType, usize).init(arena_alloc);

        for (ret.tensors.items) |tensor| {
            const tensor_type = std.meta.stringToEnum(types.DataType, tensor.type);
            if (tensor_type) |tt| {
                const g = try ret.type_counts.getOrPut(tt);
                if (!g.found_existing) g.value_ptr.* = 0;
                g.value_ptr.* += 1;
            }
        }

        if (ret.type_counts.count() > 0) {
            var stats_it = ret.type_counts.iterator();
            while (stats_it.next()) |entry| {
                ret.types_line = try std.fmt.allocPrint(arena_alloc, "{s} {s}: {}", .{ ret.types_line, @tagName(entry.key_ptr.*), entry.value_ptr.* });
            }
        }

        ret.arch = if (try arch.detectArchFromTensors(ret.tensors.items, allocator)) |a| a.* else null;

        return ret;
    }
};