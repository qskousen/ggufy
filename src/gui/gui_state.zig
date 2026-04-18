const std = @import("std");
const ggufy = @import("ggufy");

pub const LoadState = enum(u8) { idle, loading, done, err };
pub const ConvertState = enum(u8) { idle, converting, done, err };

pub const State = struct {
    // File load
    load_state: std.atomic.Value(LoadState) = .init(.idle),
    dropping: bool = false,
    file_selected_buf: [std.fs.max_path_bytes]u8 = undefined,
    file_selected: ?[]u8 = null,
    file_selected_ready: std.atomic.Value(bool) = .init(false),
    file_dialog_open: bool = false,
    load_error: ?anyerror = null,
    loaded_file: ?ggufy.fileLoader.TensorFile = null,
    wakeup_event_type: u32 = 0,

    // Conversion options
    /// Set to true after first populating folder/filename buffers from the
    /// loaded file path so we don't clobber edits the user has already made.
    convert_options_initialized: bool = false,
    target_filetype: ggufy.types.FileType = .gguf,
    target_dtype: ?ggufy.types.DataType = null,
    /// Output folder — null-terminated; dvui textEntry writes here directly.
    target_folder_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    /// Output base filename without extension — null-terminated.
    target_filename_buf: [256]u8 = std.mem.zeroes([256]u8),
    /// Base file stem stored at init; used to regenerate filename when dtype changes.
    filename_base_stem_buf: [256]u8 = std.mem.zeroes([256]u8),
    filename_base_stem_len: usize = 0,
    /// Last dtype applied when auto-generating the filename. Null = never auto-applied.
    prev_target_dtype: ?ggufy.types.DataType = null,
    /// Length of template_path last time the filename was auto-generated (change detector).
    prev_template_path_len: usize = 0,
    /// CPU thread count for quantization. Populated at startup with getCpuCount().
    target_threads: usize = 4,
    cpu_count: usize = 4,
    /// 1-100: how aggressively to quantize sensitivity-scaled layers.
    target_aggressiveness: u8 = 50,
    skip_sensitivity: bool = false,
    sensitivity_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    sensitivity_path: ?[]u8 = null,
    template_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    template_path: ?[]u8 = null,
    /// Separate open-flag for each dialog so they don't interfere.
    folder_dialog_open: bool = false,
    sensitivity_dialog_open: bool = false,
    template_dialog_open: bool = false,
    export_template_dialog_open: bool = false,
    gen_sensitivities_dialog_open: bool = false,

    // Export template / generate sensitivities
    export_template_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    export_template_path: ?[]u8 = null,
    gen_sensitivities_path_buf: [std.fs.max_path_bytes]u8 = std.mem.zeroes([std.fs.max_path_bytes]u8),
    gen_sensitivities_path: ?[]u8 = null,
    export_template_requested: bool = false,
    gen_sensitivities_requested: bool = false,
    // Status message shown after a tool operation (template export / sensitivities gen)
    tool_status_buf: [256]u8 = std.mem.zeroes([256]u8),
    tool_status_len: usize = 0,
    tool_status_is_error: bool = false,

    // Conversion progress
    convert_state: std.atomic.Value(ConvertState) = .init(.idle),
    /// Index of the last completed tensor.  Written with .release so all
    /// preceding plain writes (tensor name/type/elements) are visible to the
    /// main thread after it loads this with .acquire.
    convert_progress: std.atomic.Value(u32) = .init(0),
    convert_total: u32 = 0,
    /// Set true in the GUI to request cancellation; cleared by the convert thread.
    cancel_requested: std.atomic.Value(bool) = .init(false),
    /// Set true in the main loop to spawn the convert thread on the next iteration.
    convert_requested: bool = false,
    convert_error: ?anyerror = null,
    convert_elapsed_ns: u64 = 0,
    convert_output_path_buf: [std.fs.max_path_bytes]u8 = undefined,
    convert_output_path: ?[]u8 = null,

    // Current tensor info — written by the convert thread BEFORE the
    // convert_progress .release store.  The main thread reads these fields
    // after a .acquire load of convert_progress, so no extra sync needed.
    convert_tensor_name_buf: [256]u8 = undefined,
    convert_tensor_name_len: usize = 0,
    convert_tensor_src_type_buf: [32]u8 = undefined,
    convert_tensor_src_type_len: usize = 0,
    convert_tensor_dst_type_buf: [32]u8 = undefined,
    convert_tensor_dst_type_len: usize = 0,
    convert_tensor_elements: u64 = 0,

    // Overwrite confirmation
    overwrite_pending_path_buf: [std.fs.max_path_bytes]u8 = undefined,
    overwrite_pending_path: ?[]u8 = null,

    // Misc UI state
    show_about: bool = false,
    same_file_error: bool = false,

    // Helpers
    pub fn targetFolder(self: *const State) []const u8 {
        return std.mem.sliceTo(&self.target_folder_buf, 0);
    }

    pub fn targetFilename(self: *const State) []const u8 {
        return std.mem.sliceTo(&self.target_filename_buf, 0);
    }

    pub fn currentTensorName(self: *const State) []const u8 {
        return self.convert_tensor_name_buf[0..self.convert_tensor_name_len];
    }

    pub fn currentTensorSrcType(self: *const State) []const u8 {
        return self.convert_tensor_src_type_buf[0..self.convert_tensor_src_type_len];
    }

    pub fn currentTensorDstType(self: *const State) []const u8 {
        return self.convert_tensor_dst_type_buf[0..self.convert_tensor_dst_type_len];
    }

    pub fn toolStatus(self: *const State) []const u8 {
        return self.tool_status_buf[0..self.tool_status_len];
    }
};
