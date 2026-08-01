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
    try expectArch(@embedFile("test_fixtures/qwen.json"), "qwen");
}

test "ernie" {
    try expectArch(@embedFile("test_fixtures/ernie.json"), "ernie");
}

test "krea2 (native single-file)" {
    try expectArch(@embedFile("test_fixtures/krea2.json"), "krea2");
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

    try std.testing.expectEqualStrings("qwen", imagearch.detectArch(names).?.name);
}

// ---------------------------------------------------------------------------
// §8B foldability
// ---------------------------------------------------------------------------

test "krea2 foldability separates the text tower from the modulated blocks" {
    const k = imagearch.krea2;

    // The text tower is plain pre-norm, so its q/k/v/gate and mlp gate/up have a
    // static producer to divide.
    try std.testing.expectEqual(imagearch.Foldability.exact, k.foldability("txtfusion.refiner_blocks.1.attn.wq.weight"));
    try std.testing.expectEqual(imagearch.Foldability.exact, k.foldability("txtfusion.layerwise_blocks.0.mlp.up.weight"));
    try std.testing.expectEqual(imagearch.Foldability.exact, k.foldability("txtmlp.1.weight"));

    // The 28 main blocks look the same but are AdaLN-modulated, and the shift
    // comes from a projection shared by every block — nothing per-layer to fold
    // `1/s` into. Getting this wrong is the whole reason the relation exists: it
    // would ship a model computing a different function.
    try std.testing.expectEqual(imagearch.Foldability.runtime_shift, k.foldability("blocks.0.attn.wq.weight"));
    try std.testing.expectEqual(imagearch.Foldability.runtime_shift, k.foldability("blocks.27.mlp.gate.weight"));
    try std.testing.expectEqual(imagearch.Foldability.runtime_shift, k.foldability("last.linear.weight"));

    // Consumers of a computed activation have no producer at all.
    try std.testing.expectEqual(imagearch.Foldability.none, k.foldability("blocks.0.attn.wo.weight"));
    try std.testing.expectEqual(imagearch.Foldability.none, k.foldability("blocks.0.mlp.down.weight"));
    try std.testing.expectEqual(imagearch.Foldability.none, k.foldability("txtfusion.refiner_blocks.0.mlp.down.weight"));
    try std.testing.expectEqual(imagearch.Foldability.none, k.foldability("first.weight"));
    try std.testing.expectEqual(imagearch.Foldability.none, k.foldability("tproj.1.weight"));

    // Container-prefixed packaging must resolve the same way, since half the
    // krea2 checkpoints in circulation carry the prefix.
    try std.testing.expectEqual(
        imagearch.Foldability.runtime_shift,
        k.foldability("model.diffusion_model.blocks.3.attn.wv.weight"),
    );

    // An architecture with no relation yet claims nothing.
    try std.testing.expectEqual(imagearch.Foldability.none, imagearch.flux.foldability("double_blocks.0.img_attn.qkv.weight"));
}

test "on krea2 every exactly-foldable layer is already routed to high precision" {
    // The consequence of the two relations meeting, and the reason §8B ships
    // nothing on krea2 yet: the only layers whose fold is free (the txtfusion
    // tower and txtmlp.1) are the same layers `keys_hiprec` keeps out of the
    // aggressive formats, so equalizing them buys nothing. Every layer with real
    // quantization error to reclaim is `runtime_shift`.
    //
    // Pinned rather than written down, so that the day routing changes — a
    // measured sensitivity demoting txtfusion, say — this fails and says that
    // §8B's exact subset has become worth shipping.
    const allocator = std.testing.allocator;
    const parsed = try std.json.parseFromSlice([][]const u8, allocator, @embedFile("test_fixtures/krea2.json"), .{});
    defer parsed.deinit();

    var exact: usize = 0;
    var shift: usize = 0;
    for (parsed.value) |name| {
        switch (imagearch.krea2.foldability(name)) {
            .exact => {
                exact += 1;
                if (!imagearch.krea2.isHighPrecision(name)) {
                    std.debug.print("'{s}' is exactly foldable and NOT high-precision: §8B now has a shippable layer\n", .{name});
                    return error.TestUnexpectedResult;
                }
            },
            .runtime_shift => shift += 1,
            .none => {},
        }
    }
    // Both classes must be non-empty, or the relation is matching nothing and the
    // check above passes vacuously.
    try std.testing.expect(exact > 0);
    try std.testing.expect(shift > 0);
}

test "byName finds the architectures a calibration cache can name" {
    try std.testing.expectEqualStrings("krea2", imagearch.byName("krea2").?.name);
    try std.testing.expect(imagearch.byName("not-an-arch") == null);
    // Every arch must be reachable by its own name, or a cache captured from it
    // silently loses its per-arch rules.
    for (imagearch.arch_list) |a| try std.testing.expect(imagearch.byName(a.name) != null);
}
