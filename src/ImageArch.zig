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

    /// Check if this architecture matches the given tensor names
    pub fn matches(self: Arch, tensor_names: []const []const u8) ArchMatchResult {
        for (self.keys_detect) |key_set| {
            if (allKeysPresent(key_set, tensor_names)) {
                // Check if any banned keys are present
                for (self.keys_banned) |banned| {
                    if (containsKey(tensor_names, banned)) {
                        return .{ .matched = false, .invalid = true };
                    }
                }
                return .{ .matched = true, .invalid = false };
            }
        }
        return .{ .matched = false, .invalid = false };
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
};

pub const ArchMatchResult = struct {
    matched: bool,
    invalid: bool,
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
    .keys_detect = &.{
        &.{"transformer_blocks.0.attn.norm_added_k.weight"},
        &.{"double_blocks.0.img_attn.proj.weight"},
    },
    .keys_banned = &.{"transformer_blocks.0.attn.norm_added_k.weight"},
    .threshhold = null,
};

pub const sd3 = Arch{
    .name = "sd3",
    .keys_detect = &.{
        &.{"transformer_blocks.0.attn.add_q_proj.weight"},
        &.{"joint_blocks.0.x_block.attn.qkv.weight"},
    },
    .keys_banned = &.{"transformer_blocks.0.attn.add_q_proj.weight"},
    .threshhold = null,
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
};

pub const lumina2 = Arch{
    .name = "lumina2",
    .keys_detect = &.{
        &.{ "cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight" },
    },
    .keys_ignore = &.{
        "norm_final.weight",
    },
    .threshhold = 8192,
};

/// List of all known architectures, in detection priority order
pub const arch_list = [_]*const Arch{
    &flux,
    &sd3,
    &aura,
    &hidream,
    &cosmos,
    &ltxv,
    &hyvid,
    &wan,
    &sdxl,
    &sd1,
    &lumina2,
};

/// Detect architecture from a list of tensor names
/// Returns the matching Arch or null if unknown
pub fn detectArch(tensor_names: []const []const u8) ?*const Arch {
    for (arch_list) |arch| {
        const result = arch.matches(tensor_names);
        if (result.matched) {
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
        const result = arch.matches(tensor_names);
        if (result.invalid) {
            return ArchError.InvalidModelFormat;
        }
        if (result.matched) {
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
        const result = arch.matches(names);
        if (result.invalid) {
            return ArchError.InvalidModelFormat;
        }
        if (result.matched) {
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
    const allocator = std.testing.allocator;
    const tensors_with_banned = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.attn.norm_added_k.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    try std.testing.expect(try hasBannedKeysInTensors(&flux, &tensors_with_banned, allocator));

    const tensors_without_banned = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
        .{ .name = "some.other.tensor", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    try std.testing.expect(!try hasBannedKeysInTensors(&flux, &tensors_without_banned, allocator));
}