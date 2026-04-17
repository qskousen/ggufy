const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const SDLBackend = @import("backend");
const ggufy = @import("ggufy");
const guiState = @import("gui_state.zig");
const fileHandling = @import("file_handling.zig");
const conv = ggufy.convert;

comptime {
    std.debug.assert(@hasDecl(SDLBackend, "SDLBackend"));
}

const window_icon_png = @embedFile("gg.png");

var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = gpa_instance.allocator();

var arena = std.heap.ArenaAllocator.init(gpa);
const arena_alloc = arena.allocator();

var state: guiState.State = .{};

var g_backend: ?SDLBackend = null;
var g_win: ?*dvui.Window = null;

// GGUF output types that we can convert
const gguf_target_types = [_]ggufy.types.DataType{
    .f32,  .f16,  .bf16,
    .q2_k, .q3_k,
    .q4_0, .q4_1, .q4_k,
    .q5_0, .q5_1, .q5_k,
    .q6_k,
    .q8_0,
};

const gguf_type_names = blk: {
    var names: [gguf_target_types.len][]const u8 = undefined;
    for (gguf_target_types, 0..) |t, i| names[i] = @tagName(t);
    break :blk names;
};

// Safetensors output types that we can convert
const st_target_types = [_]ggufy.types.DataType{
    .F32, .F16, .BF16, .F8_E4M3, .F8_E5M2,
};

const st_type_names = blk: {
    var names: [st_target_types.len][]const u8 = undefined;
    for (st_target_types, 0..) |t, i| names[i] = @tagName(t);
    break :blk names;
};

pub fn main() !void {
    if (@import("builtin").os.tag == .windows) {
        dvui.Backend.Common.windowsAttachConsole() catch {};
    }
    SDLBackend.enableSDLLogging();
    std.log.info("SDL version: {f}", .{SDLBackend.getSDLVersion()});

    defer if (gpa_instance.deinit() != .ok) @panic("Memory leak on exit!");
    defer arena.deinit();

    // Populate CPU count and thread default before first frame.
    state.cpu_count = std.Thread.getCpuCount() catch 4;
    state.target_threads = state.cpu_count;

    var backend = try SDLBackend.initWindow(.{
        .allocator = gpa,
        .size = .{ .w = 800.0, .h = 600.0 },
        .min_size = .{ .w = 250.0, .h = 350.0 },
        .vsync = true,
        .title = "ggufy",
        .icon = window_icon_png,
    });
    g_backend = backend;
    defer backend.deinit();

    _ = SDLBackend.c.SDL_EnableScreenSaver();
    state.wakeup_event_type = SDLBackend.c.SDL_RegisterEvents(1);

    var win = try dvui.Window.init(@src(), gpa, backend.backend(), .{
        .theme = switch (backend.preferredColorScheme() orelse .dark) {
            .light => dvui.Theme.builtin.adwaita_light,
            .dark => dvui.Theme.builtin.adwaita_dark,
        },
    });
    defer win.deinit();
    g_win = &win;

    // Register a synchronous event watch for drag-and-drop.  An event watch
    // fires the moment SDL pumps an OS event into its queue - before any
    // SDL_PollEvent caller (including addAllEvents) can consume it.  This
    // guarantees we never miss DROP_FILE regardless of frame timing.
    _ = SDLBackend.c.SDL_AddEventWatch(dropEventWatch, &state);

    var interrupted = false;

    main_loop: while (true) {
        const nstime = win.beginWait(interrupted);

        // File load trigger
        if (state.file_selected_ready.load(.acquire)) {
            std.log.debug("Loading file: {s}", .{state.file_selected.?});
            state.file_selected_ready.store(false, .release);
            _ = arena.reset(.free_all);
            state.loaded_file = null;
            state.convert_options_initialized = false;
            state.convert_state.store(.idle, .release);
            state.convert_progress.store(0, .release);
            state.convert_output_path = null;
            const thread = std.Thread.spawn(.{ .allocator = gpa }, fileHandling.loadFile, .{ gpa, arena_alloc, &state }) catch |err| {
                state.load_error = err;
                state.load_state.store(.err, .release);
                continue :main_loop;
            };
            thread.detach();
        }

        // Conversion trigger
        if (state.convert_requested) {
            state.convert_requested = false;
            state.convert_progress.store(0, .release);
            state.convert_tensor_name_len = 0;
            state.convert_tensor_src_type_len = 0;
            state.convert_tensor_dst_type_len = 0;
            state.convert_tensor_elements = 0;
            state.convert_error = null;
            const thread = std.Thread.spawn(.{ .allocator = gpa }, fileHandling.convertFile, .{ gpa, arena_alloc, &state }) catch |err| {
                state.convert_error = err;
                state.convert_state.store(.err, .release);
                continue :main_loop;
            };
            thread.detach();
        }

        try win.begin(nstime);

        // Let dvui's backend consume all pending SDL events (mouse, keyboard, etc.).
        // Drop events are handled by dropEventWatch above and are ignored here.
        try backend.addAllEvents(&win);

        _ = SDLBackend.c.SDL_SetRenderDrawColor(backend.renderer, 0, 0, 0, 0);
        _ = SDLBackend.c.SDL_RenderClear(backend.renderer);

        const keep_running = gui_frame();
        if (!keep_running) break :main_loop;

        const end_micros = try win.end(.{});
        try backend.setCursor(win.cursorRequested());
        try backend.textInputRect(win.textInputRequested());
        try backend.renderPresent();
        const wait_event_micros = win.waitTime(end_micros);
        interrupted = try backend.waitEventTimeout(wait_event_micros);
    }
}

// Top-level frame

fn gui_frame() bool {
    // Menu bar
    {
        var hbox = dvui.box(@src(), .{ .dir = .horizontal }, .{ .style = .window, .background = true, .expand = .horizontal, .name = "main" });
        defer hbox.deinit();

        var m = dvui.menu(@src(), .horizontal, .{});
        defer m.deinit();

        if (dvui.menuItemLabel(@src(), "File", .{ .submenu = true }, .{})) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r }, .{});
            defer fw.deinit();

            if (dvui.menuItemLabel(@src(), "Open File...", .{}, .{ .expand = .horizontal }) != null) {
                if (!state.file_dialog_open) {
                    state.file_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFileDialog(
                        fileHandling.fileDialogCallback,
                        &state,
                        g_backend.?.window,
                        &file_filters,
                        file_filters.len,
                        null,
                        false,
                    );
                }
            }

            if (dvui.menuItemLabel(@src(), "Exit", .{}, .{ .expand = .horizontal }) != null) {
                return false;
            }
        }

        if (dvui.menuItemLabel(@src(), "Help", .{ .submenu = true }, .{})) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r }, .{});
            defer fw.deinit();
            if (dvui.menuItemLabel(@src(), "About ggufy", .{}, .{ .expand = .horizontal }) != null) {
                state.show_about = true;
            }
            if (dvui.menuItemLabel(@src(), "GitHub Page", .{}, .{ .expand = .horizontal }) != null) {
                _ = SDLBackend.c.SDL_OpenURL("https://github.com/qskousen/ggufy");
            }
        }
    }

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();

    // Drop-zone highlight
    const dropping = state.dropping;
    const border_color: dvui.Color = .{ .r = if (dropping) 120 else 0, .g = if (dropping) 120 else 0, .b = if (dropping) 230 else 0, .a = 240 };
    const border: ?dvui.Rect = if (dropping) dvui.Rect.all(1) else null;
    const background_color: dvui.Color = .{ .r = if (dropping) 120 else 0, .g = if (dropping) 120 else 0, .b = if (dropping) 230 else 0, .a = 80 };

    var box = dvui.box(@src(), .{}, .{ .expand = .both, .color_border = border_color, .border = border, .color_fill = background_color, .background = true });
    defer box.deinit();

    switch (state.load_state.load(.acquire)) {
        .idle => showIntro(),
        .loading => showLoading(),
        .done => switch (state.convert_state.load(.acquire)) {
            .idle => showInputFile(),
            .converting => showConverting(),
            .done => showConvertDone(),
            .err => showConvertError(),
        },
        .err => showLoadError(),
    }

    // Overwrite dialog floats on top of everything
    if (state.overwrite_pending_path != null) showOverwriteDialog();

    // About modal
    if (state.show_about) showAboutModal();

    // Check for quit events
    for (dvui.events()) |*e| {
        if (e.evt == .window and e.evt.window.action == .close) return false;
        if (e.evt == .app and e.evt.app.action == .quit) return false;
    }

    return true;
}

// Helper: frame arena

fn frameArena() std.mem.Allocator {
    return dvui.currentWindow().arena();
}

// Formatting helpers

fn formatWithCommas(value: u64, buf: []u8) []u8 {
    var tmp: [20]u8 = undefined;
    var tmp_len: usize = 0;
    if (value == 0) { buf[0] = '0'; return buf[0..1]; }
    var v = value;
    while (v > 0) {
        tmp[tmp_len] = '0' + @as(u8, @intCast(v % 10));
        tmp_len += 1;
        v /= 10;
    }
    var out_idx: usize = 0;
    for (0..tmp_len) |i| {
        const digit_pos = tmp_len - 1 - i;
        if (i > 0 and (tmp_len - i) % 3 == 0) { buf[out_idx] = ','; out_idx += 1; }
        buf[out_idx] = tmp[digit_pos];
        out_idx += 1;
    }
    return buf[0..out_idx];
}

fn formatBytes(value: u64, buf: []u8) []u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB", "PB" };
    var idx: usize = 0;
    var scaled = @as(f64, @floatFromInt(value));
    while (scaled >= 1024.0 and idx < units.len - 1) { scaled /= 1024.0; idx += 1; }
    if (idx == 0) return std.fmt.bufPrint(buf, "{d}{s}", .{ value, units[idx] }) catch buf[0..0];
    if (scaled >= 100.0) return std.fmt.bufPrint(buf, "{d:.0}{s}", .{ scaled, units[idx] }) catch buf[0..0];
    if (scaled >= 10.0)  return std.fmt.bufPrint(buf, "{d:.1}{s}", .{ scaled, units[idx] }) catch buf[0..0];
    return std.fmt.bufPrint(buf, "{d:.2}{s}", .{ scaled, units[idx] }) catch buf[0..0];
}

// Screen: intro

fn showIntro() void {
    var box_inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer box_inner.deinit();

    dvui.label(@src(), "Convert a safetensors or gguf file", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });

    if (dvui.button(@src(), "Select a File", .{}, .{ .gravity_x = 0.5 })) {
        if (!state.file_dialog_open) {
            state.file_dialog_open = true;
            SDLBackend.c.SDL_ShowOpenFileDialog(
                fileHandling.fileDialogCallback,
                &state,
                g_backend.?.window,
                &file_filters,
                file_filters.len,
                null,
                false,
            );
        }
    }
    dvui.label(@src(), "Or drag and drop a file", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });
}

// Screen: loading

fn showLoading() void {
    var box_inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer box_inner.deinit();
    dvui.label(@src(), "Loading...", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });
}

// Screen: load error

fn showLoadError() void {
    var box_inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer box_inner.deinit();
    dvui.label(@src(), "Error loading file: {}", .{state.load_error.?}, .{ .gravity_x = 0.5, .font = .theme(.title) });
    if (dvui.button(@src(), "Back", .{}, .{ .gravity_x = 0.5 })) {
        state.load_state.store(.idle, .release);
        state.load_error = null;
    }
}

// Screen: input file + conversion options

fn showInputFile() void {
    const file = &state.loaded_file.?;
    const fa = frameArena();

    // Auto-populate folder/filename once on first display.
    const first_init = !state.convert_options_initialized;
    if (first_init) {
        state.convert_options_initialized = true;
        // Folder: directory of the source file
        const dir = std.fs.path.dirname(state.file_selected.?) orelse ".";
        const dir_len = @min(dir.len, state.target_folder_buf.len - 1);
        @memcpy(state.target_folder_buf[0..dir_len], dir[0..dir_len]);
        state.target_folder_buf[dir_len] = 0;
        // Store base stem for later auto-regeneration
        const stem = std.fs.path.stem(state.file_selected.?);
        const stem_len = @min(stem.len, state.filename_base_stem_buf.len - 1);
        @memcpy(state.filename_base_stem_buf[0..stem_len], stem[0..stem_len]);
        state.filename_base_stem_len = stem_len;
    }

    // File info banner
    {
        var info_box = dvui.box(@src(), .{}, .{ .expand = .horizontal, .border = dvui.Rect.all(1), .margin = .all(4), .padding = .all(6) });
        defer info_box.deinit();

        const arch_name = if (file.arch) |a| a.name else "Unknown";
        var size_buf: [32]u8 = undefined;
        var bytes_buf: [16]u8 = undefined;
        const size_str = formatWithCommas(file.sizeInBytes, &size_buf);
        const size_bytes_str = formatBytes(file.sizeInBytes, &bytes_buf);

        dvui.label(@src(), "{s}", .{state.file_selected.?}, .{ .font = .theme(.body) });
        dvui.label(
            @src(),
            "Format: {s}   Architecture: {s}   Size: {s} ({s})   Types:{s}",
            .{ @tagName(file.type), arch_name, size_bytes_str, size_str, file.types_line },
            .{},
        );
    }

    const dim_color = dvui.themeGet().color(.control, .text).opacity(0.45);

    // Conversion options
    {
        var opts_box = dvui.box(@src(), .{}, .{ .expand = .horizontal, .margin = .{ .x = 4, .y = 8, .w = 4, .h = 4 } });
        defer opts_box.deinit();

        const active_types: []const ggufy.types.DataType = if (state.target_filetype == .gguf) &gguf_target_types else &st_target_types;
        const active_names: []const []const u8 = if (state.target_filetype == .gguf) &gguf_type_names else &st_type_names;

        // Target format row
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Target format", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
            dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                "Choose the output file format.", .{}, .{});

            const prev_filetype = state.target_filetype;
            _ = dvui.dropdownEnum(@src(), ggufy.types.FileType, .{ .choice = &state.target_filetype }, .{}, .{ .gravity_y = 0.5 });
            if (state.target_filetype != prev_filetype) {
                // Reset dtype when format changes as old selection may be invalid.
                state.target_dtype = null;
            }
        }

        // Data type row
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Data type", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
            dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                "Quantization or precision type for the output tensors.", .{}, .{});

            // Find current index in the active list (may be null if none selected or format changed)
            var dtype_idx: ?usize = if (state.target_dtype) |dt| blk: {
                for (active_types, 0..) |t, i| { if (t == dt) break :blk i; }
                break :blk null;
            } else null;

            if (dvui.dropdown(@src(), active_names, .{ .choice_nullable = &dtype_idx }, .{ .placeholder = "Select type..." }, .{ .expand = .horizontal, .gravity_y = 0.5 })) {
                state.target_dtype = if (dtype_idx) |i| active_types[i] else null;
            }
        }

        // Auto-regenerate filename when dtype changes (or on first display).
        // Uses the stored base stem + the selected (or most-common) dtype name.
        if (first_init or state.target_dtype != state.prev_target_dtype) {
            state.prev_target_dtype = state.target_dtype;
            const type_name: []const u8 = if (state.target_dtype) |dt|
                @tagName(dt)
            else blk: {
                // No dtype selected - fall back to the most common type in the source file.
                var best: ?ggufy.types.DataType = null;
                var best_count: usize = 0;
                var it = file.type_counts.iterator();
                while (it.next()) |entry| {
                    if (entry.value_ptr.* > best_count) {
                        best_count = entry.value_ptr.*;
                        best = entry.key_ptr.*;
                    }
                }
                break :blk if (best) |b| @tagName(b) else "";
            };
            const stem = state.filename_base_stem_buf[0..state.filename_base_stem_len];
            if (type_name.len > 0) {
                const written = std.fmt.bufPrint(
                    state.target_filename_buf[0..state.target_filename_buf.len - 1],
                    "{s}-{s}",
                    .{ stem, type_name },
                ) catch state.target_filename_buf[0..stem.len];
                state.target_filename_buf[written.len] = 0;
            } else {
                @memcpy(state.target_filename_buf[0..stem.len], stem);
                state.target_filename_buf[stem.len] = 0;
            }
        }

        // Output folder row
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Output folder", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
            dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                "Directory where the output file will be written. Defaults to the source file's directory.", .{}, .{});

            var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = &state.target_folder_buf } }, .{ .expand = .horizontal, .gravity_y = 0.5 });
            te.deinit();

            if (dvui.button(@src(), "Browse...", .{}, .{ .gravity_y = 0.5 })) {
                if (!state.folder_dialog_open) {
                    state.folder_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFolderDialog(
                        fileHandling.folderDialogCallback,
                        &state,
                        g_backend.?.window,
                        null,
                        false,
                    );
                }
            }
        }

        // Output filename row
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Output filename", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
            dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                "Base filename without extension. The correct extension (.gguf / .safetensors) is appended automatically.", .{}, .{});

            var te = dvui.textEntry(@src(), .{ .text = .{ .buffer = &state.target_filename_buf } }, .{ .expand = .horizontal, .gravity_y = 0.5 });
            te.deinit();
        }

        // Advanced section
        dvui.label(@src(), "Advanced", .{}, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 4 } });

        // Skip sensitivity - only shown when the architecture has built-in data
        // AND no custom sensitivity file is loaded (they are mutually exclusive).
        const has_sensitivities = if (file.arch) |a| a.sensitivities.len > 1 else false;
        if (has_sensitivities and state.sensitivity_path == null) {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            var cwd: dvui.WidgetData = undefined;
            _ = dvui.checkbox(@src(), &state.skip_sensitivity, "Skip built-in sensitivity", .{ .gravity_y = 0.5, .data_out = &cwd });
            dvui.tooltip(@src(), .{ .active_rect = cwd.borderRectScale().r },
                "By default, per-layer sensitivity scores preserve precision on important layers. Check this to quantize all eligible layers uniformly.", .{}, .{});
        }

        // Sensitivity file row
        {
            const blocked_by_template = state.template_path != null;
            const row_color: ?dvui.Color = if (blocked_by_template) dim_color else null;

            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            dvui.label(@src(), "Sensitivity file", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .color_text = row_color });

            const sens_display = if (state.sensitivity_path) |p| p else "none";
            dvui.labelNoFmt(@src(), sens_display, .{}, .{ .expand = .horizontal, .gravity_y = 0.5, .color_text = row_color });

            var bwd: dvui.WidgetData = undefined;
            if (dvui.button(@src(), "Browse...", .{}, .{ .gravity_y = 0.5, .color_text = row_color, .data_out = &bwd })) {
                if (!blocked_by_template and !state.sensitivity_dialog_open) {
                    state.sensitivity_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFileDialog(
                        fileHandling.sensitivityFileCallback,
                        &state,
                        g_backend.?.window,
                        &json_filters,
                        json_filters.len,
                        null,
                        false,
                    );
                }
            }
            if (blocked_by_template) {
                dvui.tooltip(@src(), .{ .active_rect = bwd.borderRectScale().r },
                    "Cannot use a sensitivity file while a template is selected.", .{}, .{});
            }
        }

        // Template file row
        {
            const blocked_by_sens = state.sensitivity_path != null;
            const row_color: ?dvui.Color = if (blocked_by_sens) dim_color else null;

            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();
            dvui.label(@src(), "Template file", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .color_text = row_color });

            const tmpl_display = if (state.template_path) |p| p else "none";
            dvui.labelNoFmt(@src(), tmpl_display, .{}, .{ .expand = .horizontal, .gravity_y = 0.5, .color_text = row_color });

            var bwd: dvui.WidgetData = undefined;
            if (dvui.button(@src(), "Browse...", .{}, .{ .gravity_y = 0.5, .color_text = row_color, .data_out = &bwd })) {
                if (!blocked_by_sens and !state.template_dialog_open) {
                    state.template_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFileDialog(
                        fileHandling.templateFileCallback,
                        &state,
                        g_backend.?.window,
                        &json_filters,
                        json_filters.len,
                        null,
                        false,
                    );
                }
            }
            if (blocked_by_sens) {
                dvui.tooltip(@src(), .{ .active_rect = bwd.borderRectScale().r },
                    "Cannot use a template while a sensitivity file is selected.", .{}, .{});
            }
        }

        // Aggressiveness slider
        {
            // Disabled when: sensitivity is skipped, OR no built-in sensitivities AND no custom file
            const sens_active = !state.skip_sensitivity and
                (has_sensitivities or state.sensitivity_path != null);
            const agg_color: ?dvui.Color = if (!sens_active) dim_color else null;

            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();

            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Aggressiveness", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .color_text = agg_color, .data_out = &lwd });
            if (!sens_active) {
                dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                    "Enable sensitivity scoring to use aggressiveness control.", .{}, .{});
            } else {
                dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                    "How aggressively to quantize sensitive layers. Higher = smaller file, lower = better quality.", .{}, .{});
            }

            var agg_label_buf: [8]u8 = undefined;
            const agg_label = std.fmt.bufPrint(&agg_label_buf, "{d}", .{state.target_aggressiveness}) catch "?";
            dvui.labelNoFmt(@src(), agg_label, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 28 }, .color_text = agg_color });

            var agg_frac: f32 = (@as(f32, @floatFromInt(state.target_aggressiveness)) - 1.0) / 99.0;
            if (sens_active) {
                if (dvui.slider(@src(), .{ .fraction = &agg_frac }, .{ .expand = .horizontal, .gravity_y = 0.5 })) {
                    const raw: u8 = @intFromFloat(@round(agg_frac * 99.0));
                    state.target_aggressiveness = std.math.clamp(raw + 1, 1, 100);
                }
            } else {
                // Render a visual-only progress bar in place of the slider
                dvui.progress(@src(), .{ .percent = agg_frac }, .{ .expand = .horizontal, .gravity_y = 0.5, .color_fill = dim_color });
            }
        }

        // Threads slider
        {
            const cpu_f: f32 = @floatFromInt(state.cpu_count);
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .all(2) });
            defer row.deinit();

            var lwd: dvui.WidgetData = undefined;
            dvui.label(@src(), "Threads", .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 120 }, .data_out = &lwd });
            dvui.tooltip(@src(), .{ .active_rect = lwd.borderRectScale().r },
                "Number of CPU threads to use during quantization.", .{}, .{});

            var thr_label_buf: [8]u8 = undefined;
            const thr_label = std.fmt.bufPrint(&thr_label_buf, "{d}", .{state.target_threads}) catch "?";
            dvui.labelNoFmt(@src(), thr_label, .{}, .{ .gravity_y = 0.5, .min_size_content = .{ .w = 28 } });

            var thr_frac: f32 = (@as(f32, @floatFromInt(state.target_threads)) - 1.0) / (cpu_f - 1.0);
            if (dvui.slider(@src(), .{ .fraction = &thr_frac }, .{ .expand = .horizontal, .gravity_y = 0.5 })) {
                const raw = @as(usize, @intFromFloat(@round(thr_frac * (cpu_f - 1.0))));
                state.target_threads = std.math.clamp(raw + 1, 1, state.cpu_count);
            }
        }

        // Action buttons
        if (state.same_file_error) {
            dvui.label(@src(), "Output path cannot be the same as the input file.", .{}, .{
                .color_text = dvui.Color{ .r = 220, .g = 60, .b = 60, .a = 255 },
                .margin = .{ .x = 0, .y = 4, .w = 0, .h = 2 },
            });
        }
        {
            var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 10, .w = 0, .h = 4 } });
            defer row.deinit();

            if (dvui.button(@src(), "Convert", .{}, .{ .gravity_y = 0.5 })) {
                launchConversion(fa);
            }

            // Spacer pushes Unload to the right
            {
                var spacer = dvui.box(@src(), .{}, .{ .expand = .horizontal });
                defer spacer.deinit();
            }

            if (dvui.button(@src(), "Unload File", .{}, .{ .gravity_y = 0.5 })) {
                unloadFile();
            }
        }
    }

    // Model internals tree
    showModelInternals();
}

/// Compute the output path and either trigger the conversion or show the
/// overwrite-confirmation dialog.
fn launchConversion(fa: std.mem.Allocator) void {
    const folder = state.targetFolder();
    const filename = state.targetFilename();

    const opts = conv.ConvertOptions{
        .path = state.file_selected.?,
        .filetype = state.target_filetype,
        .datatype = state.target_dtype,
        .template_path = null,
        .output_dir = if (folder.len > 0) folder else null,
        .output_name = if (filename.len > 0) filename else null,
        .threads = 1,
        .skip_sensitivity = false,
        .quantization_aggressiveness = 50,
    };

    const out_path = conv.computeOutputPath(opts, fa) catch {
        // Couldn't compute path - just fire and let the thread report the error.
        state.same_file_error = false;
        state.convert_requested = true;
        return;
    };

    // Reject if output would overwrite the source file.
    if (std.mem.eql(u8, out_path, state.file_selected.?)) {
        state.same_file_error = true;
        return;
    }
    state.same_file_error = false;

    const file_exists = blk: {
        std.fs.cwd().access(out_path, .{}) catch break :blk false;
        break :blk true;
    };

    if (file_exists) {
        const len = @min(out_path.len, state.overwrite_pending_path_buf.len);
        @memcpy(state.overwrite_pending_path_buf[0..len], out_path[0..len]);
        state.overwrite_pending_path = state.overwrite_pending_path_buf[0..len];
    } else {
        state.convert_requested = true;
    }
}

fn unloadFile() void {
    _ = arena.reset(.free_all);
    state.loaded_file = null;
    state.convert_options_initialized = false;
    state.convert_state.store(.idle, .release);
    state.convert_progress.store(0, .release);
    state.convert_output_path = null;
    state.convert_error = null;
    state.same_file_error = false;
    state.load_state.store(.idle, .release);
}

// Screen: converting

fn showConverting() void {
    var outer = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5, .expand = .both });
    defer outer.deinit();

    var inner = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5, .min_size_content = .{ .w = 400 } });
    defer inner.deinit();

    // Use .acquire so we read tensor info written before this store.
    const done = state.convert_progress.load(.acquire);
    const total = state.convert_total;

    dvui.label(@src(), "Converting...", .{}, .{ .gravity_x = 0.5, .font = .theme(.title), .margin = .{ .x = 0, .y = 0, .w = 0, .h = 8 } });

    // Progress bar
    const frac: f32 = if (total > 0)
        @as(f32, @floatFromInt(done)) / @as(f32, @floatFromInt(total))
    else
        0.0;
    dvui.progress(@src(), .{ .percent = frac }, .{ .expand = .horizontal, .margin = .{ .x = 0, .y = 4, .w = 0, .h = 4 } });

    // Tensor count
    dvui.label(@src(), "{d} / {d} tensors", .{ done, total }, .{ .gravity_x = 0.5 });

    // Current tensor info
    if (done > 0) {
        const tensor_name = state.currentTensorName();
        const src_type = state.currentTensorSrcType();
        const dst_type = state.currentTensorDstType();
        const n_elem = state.convert_tensor_elements;

        if (tensor_name.len > 0) {
            var elem_buf: [32]u8 = undefined;
            const elem_str = formatWithCommas(n_elem, &elem_buf);

            dvui.labelNoFmt(@src(), tensor_name, .{}, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 6, .w = 0, .h = 2 } });
            dvui.label(@src(), "{s} -> {s}  |  {s} elements", .{ src_type, dst_type, elem_str }, .{ .gravity_x = 0.5 });
        }
    }

    // Cancel button
    if (dvui.button(@src(), "Cancel", .{}, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 12, .w = 0, .h = 0 } })) {
        state.cancel_requested.store(true, .release);
    }
}

// Screen: conversion done

fn showConvertDone() void {
    var outer = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer outer.deinit();

    dvui.label(@src(), "Conversion complete!", .{}, .{ .gravity_x = 0.5, .font = .theme(.title) });

    if (state.convert_output_path) |p| {
        dvui.label(@src(), "Output: {s}", .{p}, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 4, .w = 0, .h = 12 } });
    }

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 0 } });
        defer row.deinit();

        if (dvui.button(@src(), "Convert Again", .{}, .{ .margin = .all(4) })) {
            state.convert_state.store(.idle, .release);
            state.convert_progress.store(0, .release);
            state.convert_output_path = null;
        }

        if (dvui.button(@src(), "Open New File", .{}, .{ .margin = .all(4) })) {
            unloadFile();
        }
    }
}

// Screen: conversion error

fn showConvertError() void {
    var outer = dvui.box(@src(), .{}, .{ .gravity_x = 0.5, .gravity_y = 0.5 });
    defer outer.deinit();

    dvui.label(@src(), "Conversion failed: {}", .{state.convert_error.?}, .{ .gravity_x = 0.5, .font = .theme(.title) });

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 0.5, .margin = .{ .x = 0, .y = 8, .w = 0, .h = 0 } });
        defer row.deinit();

        if (dvui.button(@src(), "Try Again", .{}, .{ .margin = .all(4) })) {
            state.convert_state.store(.idle, .release);
            state.convert_error = null;
        }

        if (dvui.button(@src(), "Open New File", .{}, .{ .margin = .all(4) })) {
            unloadFile();
        }
    }
}

// Overwrite confirmation dialog

fn showOverwriteDialog() void {
    const path = state.overwrite_pending_path.?;

    var float = dvui.floatingWindow(@src(), .{ .modal = true }, .{ .min_size_content = .{ .w = 360, .h = 120 } });
    defer float.deinit();

    var content = dvui.box(@src(), .{}, .{ .expand = .both, .padding = .all(16) });
    defer content.deinit();

    dvui.label(@src(), "File already exists:", .{}, .{ .font = .theme(.title) });
    dvui.labelNoFmt(@src(), path, .{}, .{ .margin = .{ .x = 0, .y = 4, .w = 0, .h = 12 } });

    {
        var row = dvui.box(@src(), .{ .dir = .horizontal }, .{ .gravity_x = 1.0 });
        defer row.deinit();

        if (dvui.button(@src(), "Cancel", .{}, .{ .margin = .all(4) })) {
            state.overwrite_pending_path = null;
        }

        if (dvui.button(@src(), "Overwrite", .{}, .{ .margin = .all(4) })) {
            state.overwrite_pending_path = null;
            state.convert_requested = true;
        }
    }
}

// About modal

fn showAboutModal() void {
    var float = dvui.floatingWindow(@src(), .{ .modal = true }, .{ .min_size_content = .{ .w = 360, .h = 160 } });
    defer float.deinit();

    var content = dvui.box(@src(), .{}, .{ .expand = .both, .padding = .all(20) });
    defer content.deinit();

    dvui.label(@src(), "ggufy", .{}, .{ .font = .theme(.title), .margin = .{ .x = 0, .y = 0, .w = 0, .h = 6 } });
    dvui.label(@src(), "Convert ML model files between safetensors and GGUF formats.", .{}, .{ .margin = .{ .x = 0, .y = 0, .w = 0, .h = 4 } });
    dvui.labelNoFmt(@src(), "https://github.com/qskousen/ggufy", .{}, .{ .margin = .{ .x = 0, .y = 0, .w = 0, .h = 16 } });

    if (dvui.button(@src(), "Close", .{}, .{ .gravity_x = 0.5 })) {
        state.show_about = false;
    }
}

// Model internals tree

fn showModelInternals() void {
    const file = &state.loaded_file.?;
    const fa = frameArena();

    var tree = dvui.TreeWidget.tree(@src(), .{ .enable_reordering = false }, .{ .expand = .horizontal });
    defer tree.deinit();

    // Metadata branch
    if (file.metadata) |meta| {
        const header = std.fmt.allocPrint(fa, "Metadata ({d})", .{meta.count()}) catch "Metadata";
        const meta_branch = tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal });
        defer meta_branch.deinit();
        dvui.labelNoFmt(@src(), header, .{}, .{ .expand = .horizontal });
        if (meta_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
            var it = meta.iterator();
            var i: usize = 0;
            while (it.next()) |entry| : (i += 1) {
                const type_name = @tagName(entry.value_ptr.*);
                const entry_label = std.fmt.allocPrint(fa, "{s}  [{s}]", .{ entry.key_ptr.*, type_name }) catch entry.key_ptr.*;
                const entry_branch = tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal, .id_extra = i });
                defer entry_branch.deinit();
                dvui.labelNoFmt(@src(), entry_label, .{}, .{ .expand = .horizontal });
                if (entry_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal, .id_extra = i })) {
                    showJsonValue(entry.value_ptr.*);
                }
            }
        }
    }

    // Tensors branch
    {
        const header = std.fmt.allocPrint(fa, "Tensors ({d})", .{file.tensors.items.len}) catch "Tensors";
        const tensors_branch = tree.branch(@src(), .{ .expanded = false }, .{ .expand = .horizontal });
        defer tensors_branch.deinit();
        dvui.labelNoFmt(@src(), header, .{}, .{ .expand = .horizontal });
        if (tensors_branch.expander(@src(), .{ .indent = 10 }, .{ .expand = .horizontal })) {
            for (file.tensors.items, 0..) |tensor, i| {
                showTensorBranch(tensor, i);
            }
        }
    }
}

fn showJsonValue(value: std.json.Value) void {
    const json_str = std.json.Stringify.valueAlloc(frameArena(), value, .{ .whitespace = .indent_2 }) catch {
        dvui.labelNoFmt(@src(), "(error serializing value)", .{}, .{});
        return;
    };
    var tl = dvui.textLayout(@src(), .{}, .{ .expand = .horizontal });
    tl.addText(json_str, .{});
    tl.deinit();
}

fn showTensorBranch(tensor: ggufy.types.Tensor, idx: usize) void {
    const fa = frameArena();

    var dims_buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&dims_buf);
    const w = fbs.writer();
    w.writeByte('[') catch {};
    for (tensor.dims, 0..) |d, j| {
        if (j > 0) w.writeAll(" x ") catch {};
        w.print("{d}", .{d}) catch {};
    }
    w.writeByte(']') catch {};

    var n: u64 = 1;
    for (tensor.dims) |d| n *= d;

    var size_buf: [32]u8 = undefined;
    const size_str = formatBytes(tensor.size, &size_buf);

    const line = std.fmt.allocPrint(fa, "{s}  [{s}]  {s}  {d} elem{s}  offset:{d}", .{
        tensor.name, tensor.type, fbs.getWritten(), n, size_str, tensor.offset,
    }) catch tensor.name;

    dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .id_extra = idx });
}

// Drop events
// SDL_AddEventWatch fires synchronously when SDL pumps an OS event - before
// any SDL_PollEvent caller (dvui's addAllEvents included) can consume it.
// This guarantees we never lose drop events to timing races.

fn dropEventWatch(userdata: ?*anyopaque, event: [*c]SDLBackend.c.SDL_Event) callconv(.c) bool {
    const s: *guiState.State = @ptrCast(@alignCast(userdata));
    const ev = event.*;
    switch (ev.type) {
        SDLBackend.c.SDL_EVENT_DROP_BEGIN,
        SDLBackend.c.SDL_EVENT_DROP_POSITION => {
            std.log.debug("Drop begin/position", .{});
            s.dropping = true;
        },
        SDLBackend.c.SDL_EVENT_DROP_FILE => {
            // File has landed - clear hover whether we accept or not.
            s.dropping = false;
            const load_busy = s.load_state.load(.acquire) == .loading;
            const conv_busy = s.convert_state.load(.acquire) == .converting;
            if (load_busy or conv_busy) {
                std.log.info("Drop ignored: load/conversion in progress", .{});
            } else {
                const path = std.mem.span(ev.drop.data);
                std.log.info("Selected: {s}", .{path});
                const can_copy = path.len <= s.file_selected_buf.len;
                if (can_copy) {
                    @memcpy(s.file_selected_buf[0..path.len], path);
                    s.file_selected = s.file_selected_buf[0..path.len];
                    s.file_selected_ready.store(true, .release);
                } else {
                    s.load_error = error.FilePathTooLong;
                    s.load_state.store(.err, .release);
                }
            }
            dvui.refresh(g_win, @src(), null);
        },
        SDLBackend.c.SDL_EVENT_DROP_COMPLETE => {
            std.log.debug("Drop finished", .{});
            s.dropping = false;
            dvui.refresh(g_win, @src(), null);
        },
        else => {},
    }
    return true; // never filter events out
}

// File dialog filters

const file_filters = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "Compatible files", .pattern = "gguf;safetensors" },
    .{ .name = "GGUF files",       .pattern = "gguf" },
    .{ .name = "Safetensors files",.pattern = "safetensors" },
    .{ .name = "All files",        .pattern = "*" },
};

const json_filters = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "JSON files", .pattern = "json" },
    .{ .name = "All files",  .pattern = "*" },
};
