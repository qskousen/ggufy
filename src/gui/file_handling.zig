const std = @import("std");
const guiState = @import("gui_state.zig");
const ggufy = @import("ggufy");
const conv = ggufy.convert;
const SDLBackend = @import("backend");

// Wakeup helper

fn pushWakeupEvent(state: *guiState.State) void {
    var ev: SDLBackend.c.SDL_Event = std.mem.zeroes(SDLBackend.c.SDL_Event);
    ev.type = state.wakeup_event_type;
    _ = SDLBackend.c.SDL_PushEvent(&ev);
}

// File loading

/// Load the file indicated by state.file_selected into state.loaded_file.
/// Handles load_state transitions.  Runs on a detached thread.
pub fn loadFile(alloc: std.mem.Allocator, arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    state.load_state.store(.loading, .release);
    const path = state.file_selected.?;
    state.loaded_file = ggufy.fileLoader.TensorFile.loadFile(state.io, alloc, arena_alloc, path) catch |err| {
        state.load_error = err;
        state.load_state.store(.err, .release);
        pushWakeupEvent(state);
        return;
    };
    state.load_state.store(.done, .release);
    pushWakeupEvent(state);
}

/// SDL open-file dialog callback — stores the chosen path and signals the main loop.
pub fn fileDialogCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.file_dialog_open = false;

    const files = filelist orelse {
        std.log.err("Dialog error: {s}", .{SDLBackend.c.SDL_GetError()});
        return;
    };
    if (files[0] == null) {
        std.log.info("File open dialog cancelled", .{});
        return;
    }

    const path = std.mem.span(files[0]);
    std.log.info("Selected: {s}", .{path});
    const can_copy = path.len <= state.file_selected_buf.len;
    if (can_copy) {
        @memcpy(state.file_selected_buf[0..path.len], path);
        state.file_selected = state.file_selected_buf[0..path.len];
        state.file_selected_ready.store(true, .release);
    } else {
        state.load_error = error.FilePathTooLong;
        state.load_state.store(.err, .release);
        pushWakeupEvent(state);
    }
}

// Output-folder dialog callback

pub fn folderDialogCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.folder_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return; // cancelled

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.target_folder_buf.len - 1);
    @memcpy(state.target_folder_buf[0..len], path[0..len]);
    state.target_folder_buf[len] = 0;
    pushWakeupEvent(state);
}

// Sensitivity file dialog callback

pub fn sensitivityFileCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.sensitivity_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.sensitivity_path_buf.len - 1);
    @memcpy(state.sensitivity_path_buf[0..len], path[0..len]);
    state.sensitivity_path_buf[len] = 0;
    state.sensitivity_path = state.sensitivity_path_buf[0..len];
    // Custom sensitivity and built-in sensitivity are mutually exclusive.
    state.template_path = null;
    state.skip_sensitivity = false;
    pushWakeupEvent(state);
}

// Template file dialog callback

pub fn templateFileCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.template_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.template_path_buf.len - 1);
    @memcpy(state.template_path_buf[0..len], path[0..len]);
    state.template_path_buf[len] = 0;
    state.template_path = state.template_path_buf[0..len];
    // Selecting a template clears any sensitivity file
    state.sensitivity_path = null;
    pushWakeupEvent(state);
}

// Conversion progress/cancel callbacks

fn progressCallback(
    ctx: ?*anyopaque,
    done: u32,
    total: u32,
    name: []const u8,
    src_type: []const u8,
    dst_type: []const u8,
    n_elements: u64,
) void {
    const state: *guiState.State = @ptrCast(@alignCast(ctx));

    // Write tensor info before the .release store so the main thread
    // observes consistent values after its .acquire load of convert_progress.
    const name_len = @min(name.len, state.convert_tensor_name_buf.len);
    @memcpy(state.convert_tensor_name_buf[0..name_len], name[0..name_len]);
    state.convert_tensor_name_len = name_len;

    const src_len = @min(src_type.len, state.convert_tensor_src_type_buf.len);
    @memcpy(state.convert_tensor_src_type_buf[0..src_len], src_type[0..src_len]);
    state.convert_tensor_src_type_len = src_len;

    const dst_len = @min(dst_type.len, state.convert_tensor_dst_type_buf.len);
    @memcpy(state.convert_tensor_dst_type_buf[0..dst_len], dst_type[0..dst_len]);
    state.convert_tensor_dst_type_len = dst_len;

    state.convert_tensor_elements = n_elements;
    state.convert_total = total;

    // Release store: signals all writes above are visible to main thread.
    state.convert_progress.store(done, .release);
    pushWakeupEvent(state);
}

fn cancelCallback(ctx: ?*anyopaque) bool {
    const state: *guiState.State = @ptrCast(@alignCast(ctx));
    return state.cancel_requested.load(.acquire);
}

// Export template / generate sensitivities

fn setToolStatus(state: *guiState.State, is_error: bool, comptime fmt: []const u8, args: anytype) void {
    const msg = std.fmt.bufPrint(&state.tool_status_buf, fmt, args) catch "(message too long)";
    state.tool_status_len = msg.len;
    state.tool_status_is_error = is_error;
}

pub fn exportTemplateCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.export_template_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.export_template_path_buf.len - 1);
    @memcpy(state.export_template_path_buf[0..len], path[0..len]);
    state.export_template_path_buf[len] = 0;
    state.export_template_path = state.export_template_path_buf[0..len];
    state.export_template_requested = true;
    pushWakeupEvent(state);
}

pub fn genSensitivitiesCallback(userdata: ?*anyopaque, filelist: [*c]const [*c]const u8, _: c_int) callconv(.c) void {
    const state: *guiState.State = @ptrCast(@alignCast(userdata));
    state.gen_sensitivities_dialog_open = false;

    const files = filelist orelse return;
    if (files[0] == null) return;

    const path = std.mem.span(files[0]);
    const len = @min(path.len, state.gen_sensitivities_path_buf.len - 1);
    @memcpy(state.gen_sensitivities_path_buf[0..len], path[0..len]);
    state.gen_sensitivities_path_buf[len] = 0;
    state.gen_sensitivities_path = state.gen_sensitivities_path_buf[0..len];
    state.gen_sensitivities_requested = true;
    pushWakeupEvent(state);
}

pub fn doExportTemplate(arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    const path = state.export_template_path.?;
    const loaded_file = &state.loaded_file.?;
    const arch_opt: ?*const ggufy.imageArch.Arch = if (loaded_file.arch != null) &(loaded_file.arch.?) else null;
    const reverse_dims = loaded_file.type == .safetensors;

    const out_file = std.Io.Dir.cwd().createFile(state.io, path, .{ .truncate = true }) catch |err| {
        setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
        return;
    };
    defer out_file.close(state.io);

    var write_buf: [8192]u8 = undefined;
    var file_writer = out_file.writer(state.io, &write_buf);
    const writer = &file_writer.interface;

    // Re-open the source file so cluster detection can read `.comfy_quant` markers and
    // collapse ComfyUI cluster layouts into a single logical quant-typed entry — the
    // TensorFile loaded in `state` holds no open handle. The arena backs both temporary
    // allocations and the file object for this one-shot export.
    const src_path = state.file_selected.?;
    switch (loaded_file.type) {
        .safetensors => {
            var f = ggufy.safetensor.init(src_path, state.io, arena_alloc, arena_alloc, false, false) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
            defer f.deinit();
            conv.writeTemplateFromFile(&f, arch_opt, reverse_dims, writer, arena_alloc, arena_alloc) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
        },
        .gguf => {
            var f = ggufy.gguf.init(src_path, state.io, arena_alloc, arena_alloc, false) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
            defer f.deinit();
            conv.writeTemplateFromFile(&f, arch_opt, reverse_dims, writer, arena_alloc, arena_alloc) catch |err| {
                setToolStatus(state, true, "Export failed: {s}", .{@errorName(err)});
                return;
            };
        },
    }
    writer.flush() catch {};
    setToolStatus(state, false, "Template exported to {s}", .{std.fs.path.basename(path)});
}

pub fn doGenSensitivities(arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    const path = state.gen_sensitivities_path.?;
    const loaded_file = &state.loaded_file.?;
    const arch_opt: ?*const ggufy.imageArch.Arch = if (loaded_file.arch != null) &(loaded_file.arch.?) else null;
    const threshold: u64 = if (arch_opt) |a| (a.threshhold orelse conv.QUANTIZATION_THRESHOLD) else conv.QUANTIZATION_THRESHOLD;

    const out_file = std.Io.Dir.cwd().createFile(state.io, path, .{ .truncate = true }) catch |err| {
        setToolStatus(state, true, "Failed: {s}", .{@errorName(err)});
        return;
    };
    defer out_file.close(state.io);

    var write_buf: [8192]u8 = undefined;
    var file_writer = out_file.writer(state.io, &write_buf);
    const writer = &file_writer.interface;

    conv.generateSensitivitiesFromTensors(
        loaded_file.tensors.items,
        arch_opt,
        threshold,
        writer,
        arena_alloc,
    ) catch |err| {
        setToolStatus(state, true, "Failed: {s}", .{@errorName(err)});
        return;
    };
    writer.flush() catch {};
    setToolStatus(state, false, "Sensitivities written to {s}", .{std.fs.path.basename(path)});
}

// Conversion

/// Convert the loaded file according to the options in state.
/// Handles convert_state transitions.  Runs on a detached thread.
/// Build a ConvertOptions from the current GUI state (without progress callbacks).
/// Shared by the real conversion and the output-size prediction so both agree on
/// exactly which options are in effect.
pub fn buildConvertOptions(state: *guiState.State) conv.ConvertOptions {
    const folder = state.targetFolder();
    const filename = state.targetFilename();
    return conv.ConvertOptions{
        .io = state.io,
        .path = state.file_selected.?,
        .filetype = state.target_filetype,
        .datatype = state.target_dtype,
        .output_dir = if (folder.len > 0) folder else null,
        .output_name = if (filename.len > 0) filename else null,
        .threads = state.target_threads,
        .skip_sensitivity = state.skip_sensitivity,
        .sensitivities_path = state.sensitivity_path,
        .template_path = state.template_path,
        .quantization_aggressiveness = @as(f32, @floatFromInt(state.target_aggressiveness)),
        .model_only = state.model_only,
        .allow_unknown_arch = state.allow_unknown_arch,
        .allow_upscale = state.allow_upscale,
        .arch_override = state.archOverride(),
    };
}

/// Predict the exact final output size for the current options without converting.
/// Opens the source file (header/metadata only — no tensor data) and runs the same
/// pipeline the real conversion uses. Returns null if the size can't be determined
/// (e.g. no data type selected, or an unrecognized architecture without override).
pub fn predictSize(alloc: std.mem.Allocator, state: *guiState.State) ?u64 {
    if (state.file_selected == null or state.target_dtype == null) return null;
    const path = state.file_selected.?;

    var pred_arena = std.heap.ArenaAllocator.init(alloc);
    defer pred_arena.deinit();
    const pa = pred_arena.allocator();

    const opts = buildConvertOptions(state);

    const file = std.Io.Dir.cwd().openFile(state.io, path, .{ .mode = .read_only }) catch return null;
    var read_buf: [8]u8 = undefined;
    var file_reader = file.reader(state.io, &read_buf);
    const file_type = ggufy.types.FileType.detect_from_file(&file_reader.interface, alloc) catch ggufy.types.FileType.safetensors;
    file.close(state.io);

    switch (file_type) {
        .safetensors => {
            var f = ggufy.safetensor.init(path, state.io, alloc, pa, false, false) catch return null;
            defer f.deinit();
            return conv.predictOutputSize(&f, opts, alloc, pa) catch null;
        },
        .gguf => {
            var f = ggufy.gguf.init(path, state.io, alloc, pa, false) catch return null;
            defer f.deinit();
            return conv.predictOutputSize(&f, opts, alloc, pa) catch null;
        },
    }
}

pub fn convertFile(alloc: std.mem.Allocator, arena_alloc: std.mem.Allocator, state: *guiState.State) void {
    state.convert_state.store(.converting, .release);
    pushWakeupEvent(state);

    const path = state.file_selected.?;

    var opts = buildConvertOptions(state);
    opts.callbacks = .{
        .progress_fn = progressCallback,
        .progress_ctx = state,
        .cancel_fn = cancelCallback,
        .cancel_ctx = state,
    };

    // Compute output path for display after completion.
    const output_path_str = conv.computeOutputPath(opts, arena_alloc) catch |err| {
        state.convert_error = err;
        state.convert_state.store(.err, .release);
        pushWakeupEvent(state);
        return;
    };

    // Detect source file type.
    const file_type = blk: {
        const file = std.Io.Dir.cwd().openFile(state.io, path, .{ .mode = .read_only }) catch |err| {
            state.convert_error = err;
            state.convert_state.store(.err, .release);
            pushWakeupEvent(state);
            return;
        };
        var read_buf: [8]u8 = undefined;
        var file_reader = file.reader(state.io, &read_buf);
        const ft = ggufy.types.FileType.detect_from_file(&file_reader.interface, alloc) catch ggufy.types.FileType.safetensors;
        file.close(state.io);
        break :blk ft;
    };

    const convert_start_ts = std.Io.Clock.Timestamp.now(state.io, .awake);

    switch (file_type) {
        .safetensors => {
            var f = ggufy.safetensor.init(path, state.io, alloc, arena_alloc, false, false) catch |err| {
                state.convert_error = err;
                state.convert_state.store(.err, .release);
                pushWakeupEvent(state);
                return;
            };
            defer f.deinit();

            conv.convert(&f, opts, alloc, arena_alloc) catch |err| {
                if (err == error.Cancelled) {
                    state.cancel_requested.store(false, .release);
                    state.convert_progress.store(0, .release);
                    state.convert_state.store(.idle, .release);
                } else {
                    state.convert_error = err;
                    state.convert_state.store(.err, .release);
                }
                pushWakeupEvent(state);
                return;
            };
        },
        .gguf => {
            var f = ggufy.gguf.init(path, state.io, alloc, arena_alloc, false) catch |err| {
                state.convert_error = err;
                state.convert_state.store(.err, .release);
                pushWakeupEvent(state);
                return;
            };
            defer f.deinit();

            conv.convert(&f, opts, alloc, arena_alloc) catch |err| {
                if (err == error.Cancelled) {
                    state.cancel_requested.store(false, .release);
                    state.convert_progress.store(0, .release);
                    state.convert_state.store(.idle, .release);
                } else {
                    state.convert_error = err;
                    state.convert_state.store(.err, .release);
                }
                pushWakeupEvent(state);
                return;
            };
        },
    }

    const elapsed = convert_start_ts.durationTo(std.Io.Clock.Timestamp.now(state.io, .awake));
    state.convert_elapsed_ns = @intCast(@max(@as(i96, 0), elapsed.raw.nanoseconds));

    // Store output path for display on the done screen.
    const path_len = @min(output_path_str.len, state.convert_output_path_buf.len);
    @memcpy(state.convert_output_path_buf[0..path_len], output_path_str[0..path_len]);
    state.convert_output_path = state.convert_output_path_buf[0..path_len];

    state.convert_state.store(.done, .release);
    pushWakeupEvent(state);
}
