const std = @import("std");
const imagearch = @import("ImageArch.zig");

/// Load fixture JSON, run detectArch, assert the expected architecture name.
fn expectArch(fixture_json: []const u8, expected_name: []const u8) !void {
    const allocator = std.testing.allocator;
    const parsed = try std.json.parseFromSlice([][]const u8, allocator, fixture_json, .{});
    defer parsed.deinit();
    const detected = imagearch.detectArch(parsed.value);
    if (detected == null) {
        std.debug.print("detectArch returned null (expected '{s}')\n", .{expected_name});
    }
    try std.testing.expect(detected != null);
    try std.testing.expectEqualStrings(expected_name, detected.?.name);
}

// ── Flux ──────────────────────────────────────────────────────────────────────

test "flux dev" {
    try expectArch(@embedFile("test_fixtures/flux.d.json"), "flux");
}

test "flux kontext" {
    try expectArch(@embedFile("test_fixtures/flux.kontext.json"), "flux");
}

test "flux2 dev" {
    try expectArch(@embedFile("test_fixtures/flux2.d.json"), "flux");
}

test "flux2 klein 9b" {
    try expectArch(@embedFile("test_fixtures/flux2.klein.9b.json"), "flux");
}

// ── SD1 / SDXL ────────────────────────────────────────────────────────────────

test "sd1.5" {
    try expectArch(@embedFile("test_fixtures/sd1.5.json"), "sd1");
}

test "sdxl" {
    try expectArch(@embedFile("test_fixtures/sdxl.json"), "sdxl");
}

test "illustrious (sdxl finetune, non-diffusers format)" {
    try expectArch(@embedFile("test_fixtures/illustrious.json"), "sdxl");
}

// ── Other ─────────────────────────────────────────────────────────────────────

test "anima (cosmos)" {
    try expectArch(@embedFile("test_fixtures/anima.json"), "cosmos");
}

test "lumina2 (zit, with model.diffusion_model prefix)" {
    try expectArch(@embedFile("test_fixtures/zit.json"), "lumina2");
}

test "lumina2 (zib, no prefix)" {
    try expectArch(@embedFile("test_fixtures/zib.json"), "lumina2");
}

test "qwen" {
    try expectArch(@embedFile("test_fixtures/qwen.json"), "qwen");
}

test "ernie" {
    try expectArch(@embedFile("test_fixtures/ernie.json"), "ernie");
}
