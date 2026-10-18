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

    pub fn loadFile(io: std.Io, allocator: std.mem.Allocator, arena_alloc: std.mem.Allocator, path: []const u8) !TensorFile {
        var ret: TensorFile = .{};
        const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
        ret.sizeInBytes = (try file.stat(io)).size;

        var read_buffer: [8]u8 = undefined;
        var reader = file.reader(io, &read_buffer);
        ret.type = types.FileType.detect_from_file(&reader.interface, allocator) catch types.FileType.safetensors;
        file.close(io);

        // load data
        switch (ret.type) {
            .safetensors => {
                var f = try st.init(path, io, allocator, arena_alloc, false, false);
                ret.metadata = f.metadata;
                ret.tensors = f.tensors;
                // Transfer json_data ownership so metadata slices remain valid.
                // deinit() on TensorFile will free it.
                ret._st_json_data = f.json_data;
                f.json_data = null;
                f.deinit();
            },
            .gguf => {
                var f = try gguf.init(path, io, allocator, arena_alloc, false);
                ret.metadata = f.metadata;
                ret.tensors = f.tensors;
                f.deinit();
            },
        }

        // A sharded checkpoint's path is its folder or index, so sum the shards instead.
        ret.sizeInBytes = (shardBytes(io, ret.tensors.items, arena_alloc) catch null) orelse ret.sizeInBytes;

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
                ret.types_line = try arena_alloc.print("{s} {s}: {}", .{ ret.types_line, @tagName(entry.key_ptr.*), entry.value_ptr.* });
            }
        }

        ret.arch = if (try arch.detectArchFromTensors(ret.tensors.items, allocator)) |a| a.* else null;

        return ret;
    }
};

/// The total size of the files `tensors` were read from, or null when none records its file.
fn shardBytes(io: std.Io, tensors: []const types.Tensor, alloc: std.mem.Allocator) !?u64 {
    var seen: std.ArrayList([]const u8) = .empty;
    var total: u64 = 0;
    outer: for (tensors) |t| {
        const p = t.source_path orelse continue;
        for (seen.items) |s| if (std.mem.eql(u8, s, p)) continue :outer;
        try seen.append(alloc, p);
        total += (try std.Io.Dir.cwd().statFile(io, p, .{})).size;
    }
    return if (seen.items.len == 0) null else total;
}

test "a sharded checkpoint's size is its shards', not its folder's" {
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const ts_ns: i96 = std.Io.Timestamp.now(io, .awake).nanoseconds;
    const token: u64 = @as(u64, @bitCast(@as(i64, @truncate(ts_ns)))) ^ @as(u64, @intFromPtr(&arena));
    const dir = try std.fmt.allocPrint(a, "/tmp/ggufy_shards_test_{x}", .{token});
    _ = try std.Io.Dir.cwd().createDirPathStatus(io, dir, .default_dir);
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};

    const shards = [_]struct { name: []const u8, tensor: []const u8, floats: usize }{
        .{ .name = "model-00001-of-00002.safetensors", .tensor = "a", .floats = 2 },
        .{ .name = "model-00002-of-00002.safetensors", .tensor = "b", .floats = 300 },
    };
    var want: u64 = 0;
    for (shards) |sh| {
        const header = try std.fmt.allocPrint(a, "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[{d}],\"data_offsets\":[0,{d}]}}}}", .{ sh.tensor, sh.floats, sh.floats * 4 });
        const file = try std.Io.Dir.cwd().createFile(io, try std.fs.path.join(a, &.{ dir, sh.name }), .{ .truncate = true });
        defer file.close(io);
        var buf: [256]u8 = undefined;
        var fw = file.writer(io, &buf);
        try fw.interface.writeInt(u64, header.len, .little);
        try fw.interface.writeAll(header);
        try fw.interface.splatByteAll(0, sh.floats * 4);
        try fw.interface.flush();
        want += 8 + header.len + sh.floats * 4;
    }
    {
        const file = try std.Io.Dir.cwd().createFile(io, try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" }), .{ .truncate = true });
        defer file.close(io);
        var buf: [256]u8 = undefined;
        var fw = file.writer(io, &buf);
        try fw.interface.print("{{\"weight_map\":{{\"a\":\"{s}\",\"b\":\"{s}\"}}}}", .{ shards[0].name, shards[1].name });
        try fw.interface.flush();
    }

    var f = try TensorFile.loadFile(io, a, a, dir);
    defer f.deinit();
    try std.testing.expectEqual(want, f.sizeInBytes);
}
