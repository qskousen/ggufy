const std = @import("std");
const imagearch = @import("ImageArch.zig");
const types = @import("types.zig");

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

/// Load a shape-carrying fixture (`[{"name":…,"shape":[…]}, …]`), run the
/// tensor-based detectArch, assert the expected architecture name. Needed for
/// architectures that can only be told apart by dimension (see `shape_detect`).
fn expectArchWithShapes(fixture_json: []const u8, expected_name: []const u8) !void {
    const allocator = std.testing.allocator;
    const Entry = struct { name: []const u8, shape: []const usize };
    const parsed = try std.json.parseFromSlice([]Entry, allocator, fixture_json, .{});
    defer parsed.deinit();

    const tensors = try allocator.alloc(types.Tensor, parsed.value.len);
    defer allocator.free(tensors);
    for (parsed.value, 0..) |e, i| {
        tensors[i] = .{
            .name = e.name,
            .type = "BF16",
            .dims = @constCast(e.shape),
            .size = 0,
            .offset = 0,
        };
    }

    const detected = try imagearch.detectArchFromTensors(tensors, allocator);
    if (detected == null) {
        std.debug.print("detectArchFromTensors returned null (expected '{s}')\n", .{expected_name});
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

// Anima = Cosmos-Predict2 + llm_adapter. It shares Cosmos's backbone but is
// detected distinctly (and must be matched before base cosmos, whose key set is
// a subset). See the `anima` arch note in ImageArch.zig.
test "anima (cosmos + llm_adapter)" {
    try expectArch(@embedFile("test_fixtures/anima.json"), "anima");
}

test "lumina2 (zit, with model.diffusion_model prefix)" {
    try expectArch(@embedFile("test_fixtures/zit.json"), "lumina2");
}

test "lumina2 (zib, no prefix)" {
    try expectArch(@embedFile("test_fixtures/zib.json"), "lumina2");
}

test "qwen" {
    try expectArch(@embedFile("test_fixtures/qwen.json"), "qwen_image");
}

test "ernie" {
    try expectArch(@embedFile("test_fixtures/ernie.json"), "ernie");
}

test "krea2 (native single-file)" {
    try expectArch(@embedFile("test_fixtures/krea2.json"), "krea2");
}

// Dumped from a pruned int8_convrot bake, so the fixture still carries the
// `.weight_scale` / `.comfy_quant` sidecars a cluster checkpoint ships with.
// Detection runs before cluster collapse and must not care.
test "minimax h3 (pruned int8_convrot)" {
    try expectArch(@embedFile("test_fixtures/minimax_h3.json"), "minimax_h3");
}

// Mage-Flow's tensor names are byte-for-byte the same set as Qwen-Image's, so
// this fixture (dumped from mageFlow_mageFlow4B.safetensors) only resolves
// correctly because detection also checks txt_norm/proj_out dimensions.
test "mageflow 4B (shape-disambiguated from qwen-image)" {
    try expectArchWithShapes(@embedFile("test_fixtures/mageflow.json"), "mage_flow");
}

// The same fixture without shapes must fall through to qwen: name-only
// detection has no way to confirm Mage-Flow's dimension constraints.
test "mageflow names alone are indistinguishable from qwen-image" {
    const allocator = std.testing.allocator;
    const Entry = struct { name: []const u8, shape: []const usize };
    const parsed = try std.json.parseFromSlice([]Entry, allocator, @embedFile("test_fixtures/mageflow.json"), .{});
    defer parsed.deinit();

    const names = try allocator.alloc([]const u8, parsed.value.len);
    defer allocator.free(names);
    for (parsed.value, 0..) |e, i| names[i] = e.name;

    try std.testing.expectEqualStrings("qwen_image", imagearch.detectArch(names).?.name);
}

// Dumped from the bf16 single-file release, which ships the MLP fused.
test "qwen image 2.1" {
    try expectArchWithShapes(@embedFile("test_fixtures/qwen21.json"), "qwen_image21");
}

// The unfused MLP layout ComfyUI also accepts: gate_up split into proj + gate_layer.
test "qwen image 2.1 with a split MLP" {
    const allocator = std.testing.allocator;
    const Entry = struct { name: []const u8, shape: []const usize };
    const parsed = try std.json.parseFromSlice([]Entry, allocator, @embedFile("test_fixtures/qwen21.json"), .{});
    defer parsed.deinit();

    var names: std.ArrayList([]const u8) = .empty;
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit(allocator);
    }
    for (parsed.value) |e| {
        if (std.mem.indexOf(u8, e.name, "img_mlp.gate_up.") != null) {
            for ([_][]const u8{ "proj", "gate_layer" }) |half| {
                const n = try std.mem.replaceOwned(u8, allocator, e.name, "gate_up", half);
                errdefer allocator.free(n);
                try names.append(allocator, n);
            }
        } else {
            try names.append(allocator, try allocator.dupe(u8, e.name));
        }
    }
    try std.testing.expectEqualStrings("qwen_image21", imagearch.detectArch(names.items).?.name);
}

// Dumped from a bf16 single-file checkpoint.
test "sensenova u1.5 8B MoT" {
    try expectArch(@embedFile("test_fixtures/sensenova_u15.json"), "sensenova_u15");
}

// Dumped from llama.cpp GGUF files on disk (native tensor names).
test "llama 24B GGUF-native" {
    try expectArch(@embedFile("test_fixtures/llama.json"), "llama");
}

test "qwen3 32B GGUF-native" {
    try expectArch(@embedFile("test_fixtures/qwen3.json"), "qwen3");
}

test "qwen35 9B hybrid GGUF-native" {
    try expectArch(@embedFile("test_fixtures/qwen35.json"), "qwen35");
}

test "qwen35 35B-A3B MoE GGUF-native" {
    try expectArch(@embedFile("test_fixtures/qwen35moe.json"), "qwen35moe");
}

// A two-block, eight-expert Qwen3MoeForCausalLM from transformers, and the GGUF
// this tool writes from it: the real models differ only in counts.
test "qwen3moe HF and GGUF-native" {
    try expectArch(@embedFile("test_fixtures/qwen3moe.hf.json"), "qwen3moe");
    try expectArch(@embedFile("test_fixtures/qwen3moe.json"), "qwen3moe");
}

// A Qwen3VLModel text encoder: the towers at the top level, not under model.
test "qwen3vl 4B HF" {
    try expectArch(@embedFile("test_fixtures/qwen3vl.hf.json"), "qwen3vl");
}
