const std = @import("std");
const types = @import("types.zig");

/// Represents a model architecture with its detection keys and configuration
pub const Arch = struct {
    /// String describing architecture name
    name: []const u8,
    /// Whether to reshape tensors for this architecture
    shape_fix: bool = false,
    /// List of key sets to match in state dict (any set matching = detected)
    /// Each inner slice is a set of keys that must ALL be present
    keys_detect: []const []const []const u8,
    /// Keys that mark model as invalid for conversion (e.g., wrong format)
    keys_banned: []const []const u8 = &.{},
    /// Keys that need to be kept in fp32/high precision
    keys_hiprec: []const []const u8 = &.{},
    /// Key substrings to ignore when found
    keys_ignore: []const []const u8 = &.{},
    /// Quantization threshhold specific to a model, or fall back to default
    threshhold: ?u64,
    /// Sensitivities filename; json dictionary of layer names and their relative sensitivity to quantization, 1-100
    sensitivities: []const u8 = "",
    /// Keys that should be upcast from bf16 to fp32 (start with a dot for end match)
    upcast_from_bf16: []const []const u8 = &.{},
    /// Keys that must pass through as-is in NVFP4 output (ComfyUI reads their shape[1] for arch detection)
    keys_nvfp4_passthrough: []const []const u8 = &.{},
    /// JSON object with base architecture configs (e.g. vae/audio_vae/vocoder) that may be absent
    /// from fine-tuned source files. Top-level keys are merged into the output `config` KV,
    /// with the source file's keys taking priority over these defaults.
    base_config_json: []const u8 = "",

    /// Check if this architecture matches the given tensor names
    pub fn matches(self: Arch, tensor_names: []const []const u8) bool {
        for (self.keys_detect) |key_set| {
            if (allKeysPresent(key_set, tensor_names)) {
                // Check if any banned keys are present; if so, skip this key set
                var banned = false;
                for (self.keys_banned) |banned_key| {
                    if (containsKey(tensor_names, banned_key)) {
                        std.log.debug("Skipping key set for architecture {s}: found banned key {s}", .{ self.name, banned_key });
                        banned = true;
                        break;
                    }
                }
                if (banned) continue;
                return true;
            }
        }
        return false;
    }

    /// Check if a key should be kept in high precision
    pub fn isHighPrecision(self: Arch, key: []const u8) bool {
        for (self.keys_hiprec) |hiprec| {
            if (std.mem.indexOf(u8, key, hiprec) != null) {
                return true;
            }
        }
        return false;
    }

    /// Check if a key should be ignored
    pub fn shouldIgnore(self: Arch, key: []const u8) bool {
        for (self.keys_ignore) |ignore| {
            if (std.mem.indexOf(u8, key, ignore) != null) {
                return true;
            }
        }
        return false;
    }

    /// Check if any of the given tensor names are banned for this architecture
    /// Returns the first banned key found, or null if none are banned
    pub fn findBannedKey(self: Arch, tensor_names: []const []const u8) ?[]const u8 {
        for (self.keys_banned) |banned| {
            if (containsKey(tensor_names, banned)) {
                return banned;
            }
        }
        return null;
    }

    /// Check if any of the given tensor names are banned (returns bool)
    pub fn hasBannedKeys(self: Arch, tensor_names: []const []const u8) bool {
        return self.findBannedKey(tensor_names) != null;
    }

    /// Check if a key must pass through unquantized in NVFP4 output for ComfyUI compat
    pub fn isNvfp4Passthrough(self: Arch, key: []const u8) bool {
        for (self.keys_nvfp4_passthrough) |pattern| {
            if (std.mem.indexOf(u8, key, pattern) != null) return true;
        }
        return false;
    }

    /// Check if the key should be upcast from bf16
    pub fn shouldUpcast(self: Arch, tensor_name: []const u8) bool {
        for (self.upcast_from_bf16) |pattern| {
            if (pattern.len > 0 and pattern[0] == '.') {
                // Dot-prefixed: match if tensor_name ends with this pattern
                if (std.mem.endsWith(u8, tensor_name, pattern)) return true;
            } else {
                // No dot: match if tensor_name equals this pattern
                if (std.mem.eql(u8, tensor_name, pattern)) return true;
            }
        }
        return false;
    }
};

fn allKeysPresent(key_set: []const []const u8, tensor_names: []const []const u8) bool {
    for (key_set) |key| {
        if (!containsKey(tensor_names, key)) {
            return false;
        }
    }
    return true;
}

fn containsKey(tensor_names: []const []const u8, key: []const u8) bool {
    for (tensor_names) |name| {
        const stripped = stripPrefix(name);
        if (std.mem.eql(u8, stripped, key)) {
            return true;
        }
    }
    return false;
}

/// Check if tensors contain any banned keys for a specific architecture
/// Returns the first banned key found, or null if none are banned
pub fn findBannedKeyInTensors(arch: *const Arch, tensors: []const types.Tensor) ?[]const u8 {
    var names: [4096][]const u8 = undefined;
    const count = @min(tensors.len, 4096);
    for (tensors[0..count], 0..) |t, i| {
        names[i] = t.name;
    }
    return arch.findBannedKey(names[0..count]);
}

/// Check if tensors contain any banned keys for a specific architecture (returns bool)
pub fn hasBannedKeysInTensors(arch: *const Arch, tensors: []const types.Tensor) bool {
    return findBannedKeyInTensors(arch, tensors) != null;
}

// ============================================================================
// Architecture Definitions
// ============================================================================

pub const flux = Arch{
    .name = "flux",
    .shape_fix = true,
    .keys_detect = &.{
        &.{"transformer_blocks.0.attn.norm_added_k.weight"},
        &.{"double_blocks.0.img_attn.proj.weight"},
    },
    .keys_banned = &.{"transformer_blocks.0.attn.norm_added_k.weight"},
    .threshhold = null,
    .upcast_from_bf16 = &.{
        ".norm.query_norm.scale",
        ".norm.key_norm.scale",
        ".norm.query_norm.weight",
        ".norm.key_norm.weight",
    },
    // ComfyUI infers in_channels from img_in.weight.shape[1], context_in_dim from
    // txt_in.weight.shape[1], and vec_in_dim from vector_in.in_layer.weight.shape[1].
    // NVFP4 nibble-packing halves the column count, so ComfyUI detects half the true
    // dimension and then clips the dequantized weight, causing shape mismatches at runtime.
    // Keep these as BF16 so ComfyUI reads the correct dimensions.
    .keys_nvfp4_passthrough = &.{
        "img_in.weight",
        "txt_in.weight",
        "vector_in.in_layer.weight",
    },
};

pub const sd3 = Arch{
    .name = "sd3",
    .keys_detect = &.{
        &.{"transformer_blocks.0.attn.add_q_proj.weight"},
        &.{"joint_blocks.0.x_block.attn.qkv.weight"},
    },
    .keys_banned = &.{"transformer_blocks.0.attn.add_q_proj.weight"},
    .threshhold = null,
    // ComfyUI infers adm_in_channels from y_embedder.mlp.0.weight.shape[1] and
    // context_dim from context_embedder.weight.shape[1]; NVFP4 packing halves both.
    .keys_nvfp4_passthrough = &.{
        "y_embedder.mlp.0.weight",
        "context_embedder.weight",
    },
};

pub const aura = Arch{
    .name = "aura",
    .keys_detect = &.{
        &.{"double_layers.3.modX.1.weight"},
        &.{"joint_transformer_blocks.3.ff_context.out_projection.weight"},
    },
    .keys_banned = &.{"joint_transformer_blocks.3.ff_context.out_projection.weight"},
    .threshhold = null,
};

pub const hidream = Arch{
    .name = "hidream",
    .keys_detect = &.{
        &.{
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight",
        },
    },
    .keys_hiprec = &.{
        ".ff_i.gate.weight",
        "img_emb.emb_pos",
    },
    .threshhold = null,
};

pub const cosmos = Arch{
    .name = "cosmos",
    .keys_detect = &.{
        &.{
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        },
    },
    .keys_hiprec = &.{"pos_embedder"},
    .keys_ignore = &.{ "_extra_state", "accum_" },
    .threshhold = null,
};

pub const hyvid = Arch{
    .name = "hyvid",
    .keys_detect = &.{
        &.{
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        },
    },
    .threshhold = null,
};

pub const wan = Arch{
    .name = "wan",
    .keys_detect = &.{
        &.{
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        },
    },
    .keys_hiprec = &.{".modulation"},
    .threshhold = null,
};

pub const ltxv = Arch{
    .name = "ltxv",
    .keys_detect = &.{
        &.{
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        },
    },
    .keys_hiprec = &.{"scale_shift_table"},
    .threshhold = null,
};

pub const ltx2 = Arch{
    // ComfyUI identifies both v1 and 2.x by the "ltxv" architecture string.
    .name = "ltxv",
    .base_config_json = @embedFile("configs/ltx23_base_config.json"),
    .keys_detect = &.{
        &.{
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.47.scale_shift_table",
            "patchify_proj.weight",
        },
    },
    // Tensors that must stay in source precision:
    //   - scale_shift_table: conditioning signals (multiple variants in 2.x)
    //   - _norm.weight: RMSNorm scale vectors
    //   - .bias: bias vectors must not be block-quantized
    //   - adaln_single: AdaLN conditioning projections, sensitive and small outer-dim shapes
    //   - patchify_proj.weight / proj_out.weight: patch embed/unembed, outer-dim = 128
    //   - learnable_registers: embedding tokens [128, X] — Python shape[-1]=128, not divisible by Q4_K block size 256
    .keys_hiprec = &.{
        "scale_shift_table",
        "_norm.weight",
        ".bias",
        "adaln_single",
        "patchify_proj.weight",
        "proj_out.weight",
        "learnable_registers",
    },
    .threshhold = null,
};

pub const sdxl = Arch{
    .name = "sdxl",
    .shape_fix = true,
    .keys_detect = &.{
        &.{ "down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight" },
        // Non-diffusers format
        &.{
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight",
            "output_blocks.5.2.conv.weight",
        },
        &.{"label_emb.0.0.weight"},
    },
    .threshhold = null,
    .sensitivities = @embedFile("sensitivities/sdxl.json"),
    // ComfyUI infers adm_in_channels from label_emb.0.0.weight.shape[1]; NVFP4 packing halves it.
    .keys_nvfp4_passthrough = &.{
        "label_emb.0.0.weight",
    },
};

pub const sd1 = Arch{
    .name = "sd1",
    .shape_fix = true,
    .keys_detect = &.{
        &.{"down_blocks.0.downsamplers.0.conv.weight"},
        // Non-diffusers format
        &.{
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight",
            "output_blocks.5.2.conv.weight",
            "output_blocks.8.2.conv.weight",
        },
    },
    .threshhold = null,
    .sensitivities = @embedFile("sensitivities/sd1.5.json"),
    // ComfyUI infers adm_in_channels from label_emb.0.0.weight.shape[1] on class-conditional
    // SD1 variants; NVFP4 packing halves it.
    .keys_nvfp4_passthrough = &.{
        "label_emb.0.0.weight",
    },
};

pub const lumina2 = Arch{
    .name = "lumina2",
    .keys_detect = &.{
        &.{ "cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight" },
    },
    .shape_fix = true,
    .keys_ignore = &.{
        "norm_final.weight",
    },
    .threshhold = 8192,
    .upcast_from_bf16 = &.{
        "cap_pad_token",
        "x_pad_token",
    },
    // ComfyUI infers cap_feat_dim from cap_embedder.1.weight.shape[1]. NVFP4 nibble-packing
    // halves that dimension, causing a shape mismatch when loading. Keep as BF16 so ComfyUI
    // reads the correct dimension.
    .keys_nvfp4_passthrough = &.{
        "cap_embedder.1.weight",
    },
};

pub const qwen = Arch{
    .name = "qwen",
    .keys_detect = &.{
        &.{
            "time_text_embed.timestep_embedder.linear_2.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
        },
    },
    .shape_fix = true,
    .threshhold = null,
    .upcast_from_bf16 = &.{
        "txt_norm.weight",
        ".norm_k.weight",
        ".norm_q.weight",
        ".norm_added_k.weight",
        ".norm_added_q.weight",
    },
    // ComfyUI infers in_channels from img_in.weight.shape[1]; NVFP4 packing halves it.
    .keys_nvfp4_passthrough = &.{
        "img_in.weight",
    },
};

pub const ernie = Arch{
    .name = "ernie",
    .keys_detect = &.{
        &.{
            "adaLN_modulation.1.weight",
            "x_embedder.proj.weight",
            "text_proj.weight",
            "layers.0.mlp.linear_fc2.weight",
        },
    },
    .shape_fix = true,
    .threshhold = null,
    .upcast_from_bf16 = &.{
        ".adaLN_sa_ln.weight",
        ".adaLN_mlp_ln.weight",
    },
};

pub const krea2 = Arch{
    .name = "krea2",
    // Detected on the native (ComfyUI single-file) naming used by Krea2 checkpoints.
    // qknorm/txtfusion are unique to Krea2, so two keys are enough to disambiguate.
    .keys_detect = &.{
        &.{
            "blocks.0.attn.qknorm.qnorm.scale",
            "txtfusion.projector.weight",
        },
    },
    .shape_fix = true,
    .threshhold = null,
    .keys_hiprec = &.{
        "txtfusion", // entire text-fusion / conditioning tower
        "tmlp", // timestep MLP (+ txtmlp text MLP)
        "tproj", // timestep projection
        "first.", // input projection (also shape-sensitive: ComfyUI reads in_channels here)
        "last.", // output projection
        ".projector",
    },
    // RMSNorm (q/k) and LayerNorm scales are precision-sensitive; keep them fp32.
    .upcast_from_bf16 = &.{
        ".qknorm.qnorm.scale",
        ".qknorm.knorm.scale",
        ".prenorm.scale",
        ".postnorm.scale",
    },
    // ComfyUI infers in_channels from first.weight.shape[1] (=64); NVFP4 nibble-packing
    // halves that dimension, so keep it as BF16 to preserve the shape.
    .keys_nvfp4_passthrough = &.{
        "first.weight",
    },
};

/// List of all known architectures, in detection priority order
pub const arch_list = [_]*const Arch{
    &flux,
    &sd3,
    &aura,
    &hidream,
    &cosmos,
    &ltx2,
    &ltxv,
    &hyvid,
    &wan,
    &sdxl,
    &sd1,
    &lumina2,
    &qwen,
    &ernie,
    &krea2,
};

/// Detect architecture from a list of tensor names
/// Returns the matching Arch or null if unknown
pub fn detectArch(tensor_names: []const []const u8) ?*const Arch {
    for (arch_list) |arch| {
        if (arch.matches(tensor_names)) {
            return arch;
        }
    }
    return null;
}

/// Detect architecture from a tensor list using an allocator for large models
pub fn detectArchFromTensors(tensors: []const types.Tensor, allocator: std.mem.Allocator) !?*const Arch {
    const names = try allocator.alloc([]const u8, tensors.len);
    defer allocator.free(names);

    for (tensors, 0..) |t, i| {
        names[i] = t.name;
    }
    return detectArch(names);
}

/// Detect architecture and return error if not found or invalid
pub fn detectArchOrError(tensor_names: []const []const u8) ArchError!*const Arch {
    for (arch_list) |arch| {
        if (arch.matches(tensor_names)) {
            return arch;
        }
    }
    return ArchError.UnknownArchitecture;
}

/// Detect architecture from tensors and return error if not found or invalid
pub fn detectArchFromTensorsOrError(tensors: []const types.Tensor, allocator: std.mem.Allocator) ArchError!*const Arch {
    const names = allocator.alloc([]const u8, tensors.len) catch return ArchError.OutOfMemory;
    defer allocator.free(names);

    for (tensors, 0..) |t, i| {
        names[i] = t.name;
    }

    for (arch_list) |arch| {
        if (arch.matches(names)) {
            return arch;
        }
    }
    return ArchError.UnknownArchitecture;
}

/// Error type for architecture validation
pub const ArchError = error{
    UnknownArchitecture,
    InvalidModelFormat,
    OutOfMemory,
};

/// Fallback used when allow_unknown_arch is set and no architecture matches.
/// Has no detection keys, no ignored keys, no shape fix, and no sensitivities.
pub const generic_arch: Arch = .{
    .name = "unknown",
    .keys_detect = &.{},
    .threshhold = null,
};

/// Strip prefixes from a tensor name (e.g. "model.diffusion_model.", etc.)
pub fn stripPrefix(name: []const u8) []const u8 {
    // Prefixes for mixed state dict
    const mixed_prefixes = [_][]const u8{
        "model.diffusion_model.",
        "model.",
    };

    // Prefixes for uniform state dict (would need to check if ALL tensors have this)
    // For now, we'll just handle mixed prefixes
    const uniform_prefixes = [_][]const u8{
        "net.",
    };

    // Check mixed prefixes (any tensor can have these)
    for (mixed_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            return name[prefix.len..];
        }
    }

    // Check uniform prefixes
    for (uniform_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            return name[prefix.len..];
        }
    }

    // No prefix found, return original name
    return name;
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Tests
// ============================================================================

test "detect flux architecture" {
    const names = [_][]const u8{"double_blocks.0.img_attn.proj.weight"};
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("flux", arch.?.name);
}

test "detect sdxl architecture" {
    const names = [_][]const u8{
        "down_blocks.0.downsamplers.0.conv.weight",
        "add_embedding.linear_1.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("sdxl", arch.?.name);
    try std.testing.expect(arch.?.shape_fix);
}

test "detect qwen architecture" {
    // this will match flux as well, but has a banned key, so it should skip flux and match qwen
    const names = [_][]const u8{
        "time_text_embed.timestep_embedder.linear_2.weight",
        "transformer_blocks.0.attn.norm_added_q.weight",
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.attn.norm_added_k.weight",
        "transformer_blocks.0.attn.norm_added_k.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("qwen", arch.?.name);
    try std.testing.expect(arch.?.shape_fix);
}

test "detect architecture from tensors with allocator" {
    const allocator = std.testing.allocator;
    const tensors = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    const arch = try detectArchFromTensors(&tensors, allocator);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("flux", arch.?.name);
}

test "detect architecture with prefix using allocator" {
    const allocator = std.testing.allocator;
    const tensors = [_]types.Tensor{
        .{ .name = "model.diffusion_model.double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    const arch = try detectArchFromTensors(&tensors, allocator);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("flux", arch.?.name);
}

test "high precision key detection" {
    try std.testing.expect(hidream.isHighPrecision("some.ff_i.gate.weight"));
    try std.testing.expect(!hidream.isHighPrecision("other.key"));
}

test "ignore key detection" {
    try std.testing.expect(cosmos.shouldIgnore("layer._extra_state.data"));
    try std.testing.expect(cosmos.shouldIgnore("accum_grad"));
    try std.testing.expect(!cosmos.shouldIgnore("normal.weight"));
}

test "banned key detection with allocator" {
    const tensors_with_banned = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.attn.norm_added_k.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    try std.testing.expect(hasBannedKeysInTensors(&flux, &tensors_with_banned));

    const tensors_without_banned = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
        .{ .name = "some.other.tensor", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    try std.testing.expect(! hasBannedKeysInTensors(&flux, &tensors_without_banned));
}

test "qwen upcast from bf16 - exact match" {
    try std.testing.expect(qwen.shouldUpcast("txt_norm.weight"));
    try std.testing.expect(!qwen.shouldUpcast("txt_norm.bias"));
    try std.testing.expect(!qwen.shouldUpcast("some.txt_norm.weight")); // not exact
}

test "qwen upcast from bf16 - suffix match" {
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.0.attn.norm_k.weight"));
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.5.attn.norm_q.weight"));
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.0.attn.norm_added_k.weight"));
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.0.attn.norm_added_q.weight"));
}

test "qwen upcast from bf16 - no false positives" {
    try std.testing.expect(!qwen.shouldUpcast("transformer_blocks.0.attn.norm_k.bias"));
    try std.testing.expect(!qwen.shouldUpcast("some.other.weight"));
    try std.testing.expect(!qwen.shouldUpcast("norm_k.weight.extra")); // suffix only, not contains
}

test "detect ltxv v1 architecture" {
    const names = [_][]const u8{
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "transformer_blocks.27.scale_shift_table",
        "caption_projection.linear_2.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("ltxv", arch.?.name);
}

test "detect ltx2 architecture" {
    const names = [_][]const u8{
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.weight",
        "model.diffusion_model.transformer_blocks.47.scale_shift_table",
        "model.diffusion_model.patchify_proj.weight",
        "model.diffusion_model.audio_adaln_single.linear.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    // Both ltxv and ltx2 write "ltxv" as general.architecture for ComfyUI compatibility.
    try std.testing.expectEqualStrings("ltxv", arch.?.name);
    // But it must resolve to the ltx2 constant, not ltxv, to get the correct hiprec list.
    try std.testing.expectEqual(&ltx2, arch.?);
}

test "detect krea2 architecture" {
    const names = [_][]const u8{
        "blocks.0.attn.qknorm.qnorm.scale",
        "txtfusion.projector.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("krea2", arch.?.name);
    try std.testing.expect(arch.?.shape_fix);
}

test "detect krea2 architecture with prefix" {
    const names = [_][]const u8{
        "model.diffusion_model.blocks.0.attn.qknorm.qnorm.scale",
        "model.diffusion_model.txtfusion.projector.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("krea2", arch.?.name);
}

test "krea2 upcast from bf16 - norm scales" {
    try std.testing.expect(krea2.shouldUpcast("blocks.0.attn.qknorm.qnorm.scale"));
    try std.testing.expect(krea2.shouldUpcast("blocks.27.attn.qknorm.knorm.scale"));
    try std.testing.expect(krea2.shouldUpcast("blocks.0.prenorm.scale"));
    try std.testing.expect(krea2.shouldUpcast("blocks.0.postnorm.scale"));
    try std.testing.expect(!krea2.shouldUpcast("blocks.0.attn.wq.weight"));
}

test "krea2 nvfp4 passthrough - first.weight" {
    try std.testing.expect(krea2.isNvfp4Passthrough("first.weight"));
    try std.testing.expect(krea2.isNvfp4Passthrough("model.diffusion_model.first.weight"));
    try std.testing.expect(!krea2.isNvfp4Passthrough("blocks.0.attn.wq.weight"));
}

test "krea2 high-precision policy matches ComfyUI reference (backbone-only quant)" {
    // Protected (kept high precision) — everything outside the main image DiT backbone.
    const protected = [_][]const u8{
        "txtfusion.layerwise_blocks.0.attn.wq.weight",
        "txtfusion.refiner_blocks.1.mlp.down.weight",
        "txtfusion.projector.weight",
        "tmlp.0.weight",
        "txtmlp.1.weight",
        "tproj.0.weight",
        "first.weight",
        "last.linear.weight",
        "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wo.weight",
    };
    for (protected) |k| try std.testing.expect(krea2.isHighPrecision(k));

    // Quantized — the image DiT backbone linears must NOT be protected.
    const backbone = [_][]const u8{
        "blocks.0.attn.wq.weight",
        "blocks.27.attn.wo.weight",
        "blocks.13.mlp.up.weight",
        "blocks.5.mlp.down.weight",
        "blocks.0.attn.gate.weight",
        "model.diffusion_model.blocks.9.mlp.gate.weight",
    };
    for (backbone) |k| try std.testing.expect(!krea2.isHighPrecision(k));
}