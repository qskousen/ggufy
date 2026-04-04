const std = @import("std");
const builtin = @import("builtin");
const dvui = @import("dvui");
const SDLBackend = @import("backend");
const ggufy = @import("ggufy");
const guiState = @import("gui_state.zig");
const fileHandling = @import("file_handling.zig");

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

pub fn main() !void {
    if (@import("builtin").os.tag == .windows) { // optional
        // on windows, graphical apps have no console so output goes to nowhere - attach it manually. related: https://github.com/ziglang/zig/issues/4196
        dvui.Backend.Common.windowsAttachConsole() catch {};
    }
    SDLBackend.enableSDLLogging();
    std.log.info("SDL version: {f}", .{SDLBackend.getSDLVersion()});


    defer if (gpa_instance.deinit() != .ok) @panic("Memory leak on exit!");
    defer arena.deinit();

    // init SDL backend (creates and owns OS window)
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

    var interrupted = false;

    main_loop: while (true) {
        // beginWait coordinates with waitTime below to run frames only when needed
        const nstime = win.beginWait(interrupted);

        if (state.file_selected_ready.load(.acquire)) {
            std.log.debug("Loading file: {s}", .{state.file_selected.?});
            state.file_selected_ready.store(false, .release);
            _ = arena.reset(.free_all);
            state.loaded_file = null;
            const thread = std.Thread.spawn(.{.allocator = gpa}, fileHandling.loadFile, .{gpa, arena_alloc, &state}) catch |err| {
                state.load_error = err;
                state.load_state.store(.err, .release);
                continue :main_loop;
            };
            thread.detach();
        }

        // marks the beginning of a frame for dvui, can call dvui functions after this
        try win.begin(nstime);

        // catch sdl events we care about
        var sdlEvent: SDLBackend.c.SDL_Event = undefined;
        while (SDLBackend.c.SDL_PollEvent(&sdlEvent)) {
            switch (sdlEvent.type) {
                SDLBackend.c.SDL_EVENT_DROP_BEGIN,
                SDLBackend.c.SDL_EVENT_DROP_POSITION,
                SDLBackend.c.SDL_EVENT_DROP_FILE,
                SDLBackend.c.SDL_EVENT_DROP_COMPLETE,
                SDLBackend.c.SDL_EVENT_DROP_TEXT => {
                    handleDropEvent(sdlEvent);
                    dvui.refresh(g_win, @src(), null); // wake dvui up
                },
                else => {
                    if (sdlEvent.type == state.wakeup_event_type) {
                        // wakeup event from load thread — just consume it, state check above handles refresh
                    } else {
                        // put it back so addAllEvents sees it
                        _ = SDLBackend.c.SDL_PushEvent(&sdlEvent);
                    }
                },
            }
        }

        // let events fall through to dvui
        try backend.addAllEvents(&win);

        // if dvui widgets might not cover the whole window, then need to clear
        // the previous frame's render
        _ = SDLBackend.c.SDL_SetRenderDrawColor(backend.renderer, 0, 0, 0, 0);
        _ = SDLBackend.c.SDL_RenderClear(backend.renderer);

        const keep_running = gui_frame();
        if (!keep_running) break :main_loop;

        // marks end of dvui frame, don't call dvui functions after this
        // - sends all dvui stuff to backend for rendering, must be called before renderPresent()
        const end_micros = try win.end(.{});

        // cursor management
        try backend.setCursor(win.cursorRequested());
        try backend.textInputRect(win.textInputRequested());

        // render frame to OS
        try backend.renderPresent();

        // waitTime and beginWait combine to achieve variable framerates
        const wait_event_micros = win.waitTime(end_micros);
        interrupted = try backend.waitEventTimeout(wait_event_micros);
    }
}

// both dvui and SDL drawing
// return false if user wants to exit the app
fn gui_frame() bool {
    {
        var hbox = dvui.box(@src(), .{ .dir = .horizontal }, .{ .style = .window, .background = true, .expand = .horizontal, .name = "main" });
        defer hbox.deinit();

        var m = dvui.menu(@src(), .horizontal, .{});
        defer m.deinit();

        if (dvui.menuItemLabel(@src(), "File", .{ .submenu = true }, .{})) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r }, .{});
            defer fw.deinit();

            if (dvui.menuItemLabel(@src(), "Open File...", .{}, .{ .expand = .horizontal }) != null) {
                if (! state.file_dialog_open) {
                    state.file_dialog_open = true;
                    SDLBackend.c.SDL_ShowOpenFileDialog(
                        fileHandling.fileDialogCallback,
                        &state,
                        g_backend.?.window,
                        &filters,
                        filters.len,
                        null,
                        false,
                    );
                }
            }

            if (dvui.menuItemLabel(@src(), "Exit", .{}, .{ .expand = .horizontal }) != null) {
                return false;
            }
        }

        if (dvui.menuItemLabel(@src(), "About", .{ .submenu = true }, .{})) |r| {
            var fw = dvui.floatingMenu(@src(), .{ .from = r }, .{});
            defer fw.deinit();
            _ = dvui.menuItemLabel(@src(), "Dummy", .{}, .{ .expand = .horizontal });
            _ = dvui.menuItemLabel(@src(), "Dummy Long", .{}, .{ .expand = .horizontal });
            _ = dvui.menuItemLabel(@src(), "Dummy Super Long", .{}, .{ .expand = .horizontal });
        }
    }

    var scroll = dvui.scrollArea(@src(), .{}, .{ .expand = .both });
    defer scroll.deinit();

    // dropping state handling
    const dropping = state.dropping;

    const border_color: dvui.Color = .{
        .r = if (dropping) 120 else 0,
        .g = if (dropping) 120 else 0,
        .b = if (dropping) 230 else 0,
        .a = 240,
    };

    const border: ?dvui.Rect = if (dropping) dvui.Rect.all(1) else null;

    const background_color: dvui.Color = .{
        .r = if (dropping) 120 else 0,
        .g = if (dropping) 120 else 0,
        .b = if (dropping) 230 else 0,
        .a = 80,
    };

    var box = dvui.box(@src(), .{}, .{.expand = .both, .color_border = border_color, .border = border, .color_fill = background_color, .background = true});
    defer box.deinit();

    switch (state.load_state.load(.acquire)) {
        .idle => showIntro(),
        .loading => showLoading(),
        .done => showInputFile(),
        .err => showError(),
    }

    // check for quitting
    for (dvui.events()) |*e| {
        // assume we only have a single window
        if (e.evt == .window and e.evt.window.action == .close) return false;
        if (e.evt == .app and e.evt.app.action == .quit) return false;
    }

    return true;
}

fn showLoading() void {
    var box_inner = dvui.box(@src(), .{}, .{.gravity_x = 0.5, .gravity_y = 0.5});
    defer box_inner.deinit();
    var tl = dvui.textLayout(@src(), .{}, .{.gravity_x = 0.5, .font = .theme(.title) });
    tl.addText("Loading...", .{});
    tl.deinit();
}

fn showIntro() void {
    var box_inner = dvui.box(@src(), .{}, .{.gravity_x = 0.5, .gravity_y = 0.5});
    defer box_inner.deinit();

    var tl = dvui.textLayout(@src(), .{}, .{.gravity_x = 0.5, .font = .theme(.title) });
    tl.addText("Convert a safetensors or gguf file", .{});
    tl.deinit();


    if (dvui.button(@src(), "Select a File", .{}, .{.gravity_x = 0.5})) {
        if (! state.file_dialog_open) {
            state.file_dialog_open = true;
            SDLBackend.c.SDL_ShowOpenFileDialog(
                fileHandling.fileDialogCallback,
                &state,
                g_backend.?.window,
                &filters,
                filters.len,
                null,
                false,
            );
        }
    }
    var tl2 = dvui.textLayout(@src(), .{}, .{.gravity_x = 0.5, .font = .theme(.title) });
    tl2.addText("Or drag and drop a file", .{});
    tl2.deinit();
}

fn frameArena() std.mem.Allocator {
    return dvui.currentWindow().arena();
}

fn showInputFile() void {
    const file = &state.loaded_file.?;
    const fa = frameArena();

    dvui.label(
        @src(),
        "Opened file: {s}",
        .{state.file_selected.?},
        .{.gravity_x = 0.5, .border = .all(1)}
    );

    // Metadata accordion
    if (file.metadata) |meta| {
        const header = std.fmt.allocPrint(fa, "Metadata ({d})", .{meta.count()}) catch "Metadata";
        if (dvui.expander(@src(), header, .{}, .{ .expand = .horizontal })) {
            showMetadataEntries(meta);
        }
    }

    // Tensors accordion
    {
        const header = std.fmt.allocPrint(fa, "Tensors ({d})", .{file.tensors.items.len}) catch "Tensors";
        if (dvui.expander(@src(), header, .{}, .{ .expand = .horizontal })) {
            showTensorsExpanded(file.tensors.items);
        }
    }
}

fn showTensorsExpanded(tensors: []const ggufy.types.Tensor) void {
    // Container wrapping the whole list so we can measure total height.
    var list_box = dvui.box(@src(), .{}, .{ .expand = .horizontal });
    defer list_box.deinit();
    const list_id = list_box.data().id;

    // Derive per-item height from previous frame's total height.
    // Self-consistent: spacers maintain total = N * item_h, so this is stable.
    var item_h: f32 = dvui.dataGet(null, list_id, "_ih", f32) orelse 0;
    {
        const prev_total = list_box.data().rect.h;
        if (prev_total > 0 and tensors.len > 0) {
            item_h = prev_total / @as(f32, @floatFromInt(tensors.len));
            dvui.dataSet(null, list_id, "_ih", item_h);
        }
    }

    if (item_h <= 0) {
        // First frame only: render everything to establish measurements.
        for (tensors, 0..) |tensor, i| {
            showTensorEntry(tensor, i);
        }
        return;
    }

    // Convert item height to physical pixels for clip comparison.
    const rs = list_box.data().borderRectScale();
    const item_h_phys = item_h * rs.s;
    const list_top_phys = rs.r.y;
    const clip = dvui.clipGet();

    const first_visible: usize = blk: {
        const above = clip.y - list_top_phys;
        if (above <= 0) break :blk 0;
        break :blk @min(@as(usize, @intFromFloat(above / item_h_phys)), tensors.len);
    };
    const last_visible: usize = blk: {
        const to_bottom = clip.y + clip.h - list_top_phys;
        if (to_bottom <= 0) break :blk 0;
        const idx = @as(usize, @intFromFloat(@ceil(to_bottom / item_h_phys))) + 2;
        break :blk @min(idx, tensors.len);
    };

    // Top spacer for items scrolled above the viewport.
    if (first_visible > 0) {
        var sp = dvui.box(@src(), .{}, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = @as(f32, @floatFromInt(first_visible)) * item_h },
        });
        sp.deinit();
    }

    for (first_visible..last_visible) |i| {
        showTensorEntry(tensors[i], i);
    }

    // Bottom spacer for items below the viewport.
    if (last_visible < tensors.len) {
        var sp = dvui.box(@src(), .{}, .{
            .expand = .horizontal,
            .min_size_content = .{ .h = @as(f32, @floatFromInt(tensors.len - last_visible)) * item_h },
        });
        sp.deinit();
    }
}

fn showMetadataEntries(meta: std.json.ObjectMap) void {
    const fa = frameArena();
    var it = meta.iterator();
    var i: usize = 0;
    while (it.next()) |entry| : (i += 1) {
        const type_name = @tagName(entry.value_ptr.*);
        const entry_label = std.fmt.allocPrint(fa, "{s}  [{s}]", .{ entry.key_ptr.*, type_name }) catch entry.key_ptr.*;
        if (dvui.expander(@src(), entry_label, .{}, .{ .expand = .horizontal, .id_extra = i })) {
            var indent = dvui.box(@src(), .{}, .{ .expand = .horizontal, .padding = .{ .x = 20 }, .id_extra = i });
            defer indent.deinit();
            showJsonValue(entry.value_ptr.*);
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

fn showTensorEntry(tensor: ggufy.types.Tensor, i: usize) void {
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

    const size_str = if (tensor.size == 0) ""
        else if (tensor.size >= 1024 * 1024 * 1024)
            std.fmt.allocPrint(fa, "  {d:.2} GB", .{@as(f64, @floatFromInt(tensor.size)) / (1024.0 * 1024.0 * 1024.0)}) catch ""
        else if (tensor.size >= 1024 * 1024)
            std.fmt.allocPrint(fa, "  {d:.2} MB", .{@as(f64, @floatFromInt(tensor.size)) / (1024.0 * 1024.0)}) catch ""
        else if (tensor.size >= 1024)
            std.fmt.allocPrint(fa, "  {d:.2} KB", .{@as(f64, @floatFromInt(tensor.size)) / 1024.0}) catch ""
        else
            std.fmt.allocPrint(fa, "  {d} B", .{tensor.size}) catch "";

    const line = std.fmt.allocPrint(fa, "{s}  [{s}]  {s}  {d} elem{s}  offset:{d}", .{
        tensor.name, tensor.type, fbs.getWritten(), n, size_str, tensor.offset,
    }) catch tensor.name;

    dvui.labelNoFmt(@src(), line, .{}, .{ .expand = .horizontal, .id_extra = i });
}

fn showError() void {
    var box_inner = dvui.box(@src(), .{}, .{.gravity_x = 0.5, .gravity_y = 0.5});
    defer box_inner.deinit();

    var tl = dvui.textLayout(@src(), .{}, .{.gravity_x = 0.5, .font = .theme(.title) });
    const err = std.fmt.allocPrint(gpa, "Error: {}", .{state.load_error.?}) catch "Out of memory";
    tl.addText(err, .{});
    tl.deinit();
}

fn handleDropEvent(sdlEvent: SDLBackend.c.SDL_Event) void {
    switch (sdlEvent.type) {
        SDLBackend.c.SDL_EVENT_DROP_BEGIN => {
            std.log.debug("Drop begin", .{});
            state.dropping = true;
        },
        SDLBackend.c.SDL_EVENT_DROP_POSITION => {
            // hovering — event.drop.x and event.drop.y give cursor position
            const x = sdlEvent.drop.x;
            const y = sdlEvent.drop.y;
            std.log.debug("Drop position: x {} y {}", .{x, y});
            state.dropping = true;
        },
        SDLBackend.c.SDL_EVENT_DROP_FILE => {
            // a file was dropped — event.drop.data is the path
            const path = std.mem.span(sdlEvent.drop.data);
            std.log.info("Selected: {s}", .{path});
            const can_copy = path.len <= state.file_selected_buf.len;
            if (can_copy) {
                // dump the path into the state buffer so we own it
                @memcpy(state.file_selected_buf[0..path.len], path);
                state.file_selected = state.file_selected_buf[0..path.len];
                state.file_selected_ready.store(true, .release);
                std.log.debug("File dropped: {s}", .{state.file_selected.?});
            } else {
                std.log.debug("Dropped file path too long: {s}", .{path});
                state.load_error = error.FilePathTooLong;
                state.load_state.store(.err, .release);
            }
        },
        SDLBackend.c.SDL_EVENT_DROP_COMPLETE => {
            // drop finished — clear your drop zone highlight
            std.log.debug("Drop finished", .{});
            state.dropping = false;
        },
        SDLBackend.c.SDL_EVENT_DROP_TEXT => {
            // text was dropped instead of a file
            const text = std.mem.span(sdlEvent.drop.data);
            std.log.debug("Text dropped: {s}", .{text});
        },
        else => {},
    }
}

const filters = [_]SDLBackend.c.SDL_DialogFileFilter{
    .{ .name = "Compatible files",  .pattern = "gguf;safetensors" },
    .{ .name = "GGUF files",  .pattern = "gguf" },
    .{ .name = "Safetensors files",  .pattern = "safetensors" },
    .{ .name = "All files",   .pattern = "*" },
};