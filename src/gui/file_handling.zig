const std = @import("std");
const guiState = @import("gui_state.zig");
const ggufy = @import("ggufy");
const SDLBackend = @import("backend");

fn pushWakeupEvent(state: *guiState.State) void {
    var ev: SDLBackend.c.SDL_Event = std.mem.zeroes(SDLBackend.c.SDL_Event);
    ev.type = state.wakeup_event_type;
    _ = SDLBackend.c.SDL_PushEvent(&ev);
}

/// loadFile loads the file in the state. also handles load_state switching. on error, sets load_state to err and
/// puts the error in state.load_error
pub fn loadFile(alloc: std.mem.Allocator, arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    state.load_state.store(.loading, .release);
    const path = state.file_selected.?;
    state.loaded_file = ggufy.fileLoader.TensorFile.loadFile(alloc, arena_alloc, path) catch |err| {
        state.load_error = err;
        state.load_state.store(.err, .release);
        pushWakeupEvent(state);
        return;
    };
    state.load_state.store(.done, .release);
    pushWakeupEvent(state);
}

// note that this function is called from a different thread
// last param is the filter selected; we don't really care so we ignore it
pub fn fileDialogCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));

    // filelist is null if an error occurred
    const files = filelist orelse {
        std.log.err("Dialog error: {s}", .{SDLBackend.c.SDL_GetError()});
        return;
    };

    // filelist[0] is null if user cancelled
    if (files[0] == null) {
        std.log.info("File open dialog cancelled", .{});
        return;
    }

    const path = std.mem.span(files[0]);
    std.log.info("Selected: {s}", .{path});
    const can_copy = path.len <= state.file_selected_buf.len;
    if (can_copy) {
        // dump the path into the state buffer so we own it
        @memcpy(state.file_selected_buf[0..path.len], path);
        state.file_selected = state.file_selected_buf[0..path.len];
        state.file_selected_ready.store(true, .release);
    } else {
        state.load_error = error.FilePathTooLong;
        state.load_state.store(.err, .release);
    }
}