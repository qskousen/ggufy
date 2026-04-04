const std = @import("std");
const ggufy = @import("ggufy");

pub const LoadState = enum(u8) { idle, loading, done, err };

pub const State = struct {
    load_state: std.atomic.Value(LoadState) = .init(.idle),
    dropping: bool = false,
    file_selected_buf: [std.fs.max_path_bytes]u8 = undefined,
    file_selected: ?[]u8 = null,
    file_selected_ready: std.atomic.Value(bool) = .init(false),
    file_dialog_open: bool = false,
    target_folder: ?[]u8 = null,
    target_filename: ?[]u8 = null,
    target_dtype: ?ggufy.types.DataType = null,
    load_error: ?anyerror = null,
    loaded_file: ?ggufy.fileLoader.TensorFile = null,
    wakeup_event_type: u32 = 0,
};