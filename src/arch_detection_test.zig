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

// ── HunyuanVideo ──────────────────────────────────────────────────────────────

test "hunyuan video under Tencent's names" {
    try expectArch(@embedFile("test_fixtures/hyvid.json"), "hyvid");
}

// Its double_blocks are named like flux's; the token refiner tells them apart.
test "hunyuan video as Comfy-Org ships it, and without the model.model. prefix" {
    try expectArch(@embedFile("test_fixtures/hyvid.comfy.json"), "hyvid");

    const allocator = std.testing.allocator;
    const parsed = try std.json.parseFromSlice([][]const u8, allocator, @embedFile("test_fixtures/hyvid.comfy.json"), .{});
    defer parsed.deinit();
    for (parsed.value) |*n| n.* = n.*["model.model.".len..];
    try std.testing.expectEqualStrings("hyvid", imagearch.detectArch(parsed.value).?.name);
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

// Dumped from the step-distilled bf16 single file, which the BFL and ComfyUI
// releases share key for key.
test "flux3 image klein" {
    try expectArchWithShapes(@embedFile("test_fixtures/flux3_klein.json"), "flux3");
}

// ComfyUI's int8 bake fuses q/k/v into qkv and adds comfy_quant and attention configs.
test "flux3 image klein with fused qkv" {
    try expectArch(@embedFile("test_fixtures/flux3_klein_int8_convrot.json"), "flux3");
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

// ── Attention modules ─────────────────────────────────────────────────────────

/// The attention config names ggufy derives from a checkpoint's tensor names are
/// exactly the modules ComfyUI instantiates a ComfyAttention for in that
/// checkpoint (gen_attention_fixtures.py), and the skipped ones are a subset.
fn expectAttentionModules(fixture_json: []const u8, arch: *const imagearch.Arch) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Fixture = struct { source: []const u8, prefix: []const u8, tensors: []const []const u8, modules: []const []const u8 };
    const fx = try std.json.parseFromSliceLeaky(Fixture, a, fixture_json, .{});

    const got = try arch.attentionConfigNames(fx.tensors, true, a);
    var want: std.ArrayList([]const u8) = .empty;
    for (fx.modules) |m| try want.append(a, try std.mem.concat(a, u8, &.{ fx.prefix, m, ".comfy_attention.config" }));
    const lt = struct {
        fn f(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.f;
    const got_sorted = try a.dupe([]const u8, got);
    std.sort.block([]const u8, got_sorted, {}, lt);
    std.sort.block([]const u8, want.items, {}, lt);
    if (got_sorted.len != want.items.len) {
        std.debug.print("{s}: {d} modules found, ComfyUI has {d}\n", .{ fx.source, got_sorted.len, want.items.len });
        return error.TestExpectedEqual;
    }
    for (got_sorted, want.items) |g, w| try std.testing.expectEqualStrings(w, g);

    const kept = try arch.attentionConfigNames(fx.tensors, false, a);
    try std.testing.expect(kept.len > 0);
    try std.testing.expect(kept.len == got.len or arch.attention_skip.len > 0);
    for (arch.attention_skip) |skip| {
        var listed = false;
        for (arch.attention_modules) |m| listed = listed or std.mem.eql(u8, m, skip);
        try std.testing.expect(listed);
    }
}

test "attention modules match ComfyUI's for every family" {
    const cases = .{
        .{ "flux3", &imagearch.flux3 },          .{ "flux", &imagearch.flux },
        .{ "sd1", &imagearch.sd1 },              .{ "sdxl", &imagearch.sdxl },
        .{ "anima", &imagearch.anima },          .{ "cosmos", &imagearch.cosmos },
        .{ "ernie", &imagearch.ernie },          .{ "krea2", &imagearch.krea2 },
        .{ "mage_flow", &imagearch.mageflow },   .{ "qwen_image21", &imagearch.qwen21 },
        .{ "qwen_image", &imagearch.qwen },      .{ "sensenova_u15", &imagearch.sensenova_u15 },
        .{ "minimax_h3", &imagearch.minimax_h3 }, .{ "ltx2", &imagearch.ltx2 },
        .{ "wan", &imagearch.wan },              .{ "lumina2", &imagearch.lumina2 },
        .{ "sd3", &imagearch.sd3 },              .{ "aura", &imagearch.aura },
        .{ "hidream", &imagearch.hidream },      .{ "hyvid", &imagearch.hyvid },
        .{ "ltxv", &imagearch.ltxv },
    };
    inline for (cases) |c| {
        expectAttentionModules(@embedFile("test_fixtures/attention/" ++ c[0] ++ ".json"), c[1]) catch |e| {
            std.debug.print("family {s}\n", .{c[0]});
            return e;
        };
    }
}

test "flux3 int8 attention leaves out the blocks ComfyUI's bake leaves out" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names = try std.json.parseFromSliceLeaky([]const []const u8, a, @embedFile("test_fixtures/flux3_klein_int8_convrot.json"), .{});
    var baked: std.ArrayList([]const u8) = .empty;
    for (names) |n| if (std.mem.endsWith(u8, n, ".comfy_attention.config")) try baked.append(a, n);
    const kept = try imagearch.flux3.attentionConfigNames(names, false, a);
    try std.testing.expectEqual(baked.items.len, kept.len);
    for (kept) |k| {
        var hit = false;
        for (baked.items) |b| hit = hit or std.mem.eql(u8, b, k);
        try std.testing.expect(hit);
    }
}
