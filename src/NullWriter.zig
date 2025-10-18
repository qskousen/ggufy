const std = @import("std");

pub const NullWriter = struct {
    buffer: []u8,
    end: usize = 0,
    pos: u64 = 0,
    interface: std.Io.Writer,

    pub fn init(buffer: []u8) NullWriter {
        return .{
            .buffer = buffer,
            .interface = initInterface(buffer),
        };
    }

    fn initInterface(buffer: []u8) std.Io.Writer {
        return .{
            .vtable = &.{
                .drain = drain,
            },
            .buffer = buffer,
        };
    }

    fn drain(io_w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const w: *NullWriter = @alignCast(@fieldParentPtr("interface", io_w));
        _ = w;

        if (data.len == 0) return 0;
        const last = data[data.len - 1];
        if (last.len == 0 or splat == 0) return 0;

        const buffered = io_w.buffered();
        // Return the entire length since we're discarding everything
        return if (buffered.len > 0) buffered.len else last.len;
    }
};

pub fn nullWriter(buffer: []u8) NullWriter {
    return NullWriter.init(buffer);
}