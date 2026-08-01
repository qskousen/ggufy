const std = @import("std");
const types = @import("types.zig");

/// A single dimension constraint, used to tell apart architectures whose tensor
/// *names* are identical (Mage-Flow vs Qwen-Image). Mirrors what ComfyUI's
/// model_detection.py does in the same situation: read one dimension of one
/// named tensor and compare it against a known constant.
pub const ShapeRule = struct {
    /// Tensor name with any state-dict prefix already stripped (see stripPrefix).
    key: []const u8,
    /// Index into `Tensor.dims`, which is always outermost-first (PyTorch order)
    /// regardless of whether the source was SafeTensors or GGUF.
    dim: usize,
    /// Required extent of that dimension.
    extent: usize,
};

/// Can §8B's per-channel fold `x / s` be produced without a runtime change?
///
/// Equalization scales a weight's input columns by `s` and requires whatever
/// feeds it to divide by the same `s`. Whether that is free, cheap-but-invasive,
/// or impossible is a property of the architecture's graph, which is why the
/// relation lives here rather than in `Equalize.zig`.
pub const Foldability = enum {
    /// A static per-channel producer (an RMSNorm scale vector) feeds this weight,
    /// with nothing additive between it and the GEMM. Divide the producer's vector
    /// by `s` and the transform is exact at zero runtime cost — the case §8B was
    /// designed around.
    exact,
    /// A static norm scale exists, but the activations are additively modulated
    /// after it (AdaLN `(1 + a) ⊙ n + b`) by a runtime vector that no per-layer
    /// static tensor carries alone. `1/s` folds into the norm and commutes with
    /// the multiplicative half, but `W·diag(s)` also multiplies the shift, so the
    /// transform is **not** exact. Needs the runtime to divide the modulated
    /// activations, i.e. a per-layer vector in the file and one multiply at the
    /// modulate site.
    runtime_shift,
    /// The input is a computed activation with no per-channel producer at all
    /// (attention output → `wo`, the SwiGLU product → `down`, a raw embedding).
    /// Equalization here means a format change plus a kernel change on every
    /// backend, which §8B explicitly declines.
    none,
};

/// One rule in an architecture's foldability relation. Both bounds must match the
/// **stripped** tensor name; an empty `prefix` matches anything.
pub const FoldRule = struct {
    prefix: []const u8 = "",
    suffix: []const u8,
    fold: Foldability,
};

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
    /// Dimension constraints that must ALL hold in addition to `keys_detect`.
    /// Only usable when tensor shapes are available, so an architecture that
    /// declares these can only be detected via the tensor-based entry points
    /// (`detectArchFromTensors*`), never from a bare list of names.
    shape_detect: []const ShapeRule = &.{},
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
    /// §8B: which weights have a foldable per-channel producer. First matching rule
    /// wins; anything unmatched is `.none`. Empty means "not analyzed for this
    /// architecture", which reads as `.none` everywhere — conservative, and the
    /// right default, since claiming a fold that does not exist would ship a model
    /// computing a different function.
    eq_folds: []const FoldRule = &.{},

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

    /// Check this architecture's `shape_detect` constraints against a tensor list.
    /// A missing tensor, a missing dimension, or a mismatched extent all fail.
    pub fn shapesMatch(self: Arch, tensors: []const types.Tensor) bool {
        for (self.shape_detect) |rule| {
            const t = findTensor(tensors, rule.key) orelse return false;
            if (rule.dim >= t.dims.len) return false;
            if (t.dims[rule.dim] != rule.extent) return false;
        }
        return true;
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

    /// Whether §8B equalization of this weight's input channels can be folded
    /// into a static producer. Takes the tensor name in either packaging — it
    /// strips the container prefix itself, as the rest of this file's lookups do.
    pub fn foldability(self: Arch, tensor_name: []const u8) Foldability {
        const key = stripPrefix(tensor_name);
        for (self.eq_folds) |rule| {
            if (!std.mem.startsWith(u8, key, rule.prefix)) continue;
            if (!std.mem.endsWith(u8, key, rule.suffix)) continue;
            // A rule must not match the tensor it is describing the *producer* of,
            // which for a same-name prefix+suffix pair would mean overlapping
            // bounds on too short a name.
            if (key.len < rule.prefix.len + rule.suffix.len) continue;
            return rule.fold;
        }
        return .none;
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

fn findTensor(tensors: []const types.Tensor, key: []const u8) ?*const types.Tensor {
    for (tensors) |*t| {
        if (std.mem.eql(u8, stripPrefix(t.name), key)) return t;
    }
    return null;
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

// Anima is Cosmos-Predict2 (MiniTrainDIT) with an extra bolted-on T5 text
// adapter (`llm_adapter`). It shares Cosmos's entire backbone, so its detect
// keys are Cosmos's two plus the llm_adapter discriminator. This mirrors
// ComfyUI's own model_detection.py, which starts at "cosmos_predict2" and
// reclassifies to "anima" iff `llm_adapter.blocks.0.cross_attn.q_proj.weight`
// is present. "anima" is a valid `general.architecture` value for the
// ComfyUI-GGUF loader (it's in PIG_ARCH_LIST), so we can name it distinctly.
//
// Must be listed BEFORE `cosmos` in arch_list: base Cosmos's key set is a
// subset of Anima's, so cosmos would otherwise match first.
//
// The ENTIRE `llm_adapter` is kept high-precision (not just its embedding),
// matching the reference converter silveroxides/convert_to_quant (its
// ANIMA_LAYER_KEYNAMES lists "llm_adapter" as highprec). Two reasons:
//   1. ComfyUI: the adapter's `embed.weight` is an nn.Embedding table that
//      can't be block/int-quantized (also caught generically by
//      isEmbeddingWeight() in Convert.zig).
//   2. Forge Neo: its loader's `process_anima` MOVES the whole llm_adapter out
//      of the transformer and into the *text-encoder* component. If any adapter
//      tensor is quantized (carries `.comfy_quant`), Forge loads the text
//      encoder via its MixedPrecision path, which builds non-quantized layers
//      (the `embed`) at fp32 while the quantized projections dequantize to
//      bf16. The adapter then computes rotary embeddings from the fp32 embed
//      output and applies them to q/k, but v (no rope) stays bf16 — so
//      scaled_dot_product_attention gets mismatched dtypes and throws. Keeping
//      the adapter fully bf16 means the text encoder has no `.comfy_quant`, so
//      it loads in plain bf16 and everything matches. (bf16 Linears load fine
//      in ComfyUI too, so this does not regress the working ComfyUI path.)
pub const anima = Arch{
    .name = "anima",
    .keys_detect = &.{
        &.{
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
            "llm_adapter.blocks.0.cross_attn.q_proj.weight",
        },
    },
    // High-precision set mirrors silveroxides/convert_to_quant's ANIMA_LAYER_KEYNAMES
    // (the reference converter): the llm_adapter (see above), plus the first block,
    // block 1's adaln modulation, the final layer, and the timestep/patch embedders.
    // These are the small, sensitivity-critical layers that reference tool keeps in
    // full precision for quality. Patterns are bare substrings (isHighPrecision matches
    // the full tensor name), so "blocks.0." also covers the — already-hiprec — llm_adapter
    // block 0, which is harmless. pos_embedder is retained from the Cosmos base.
    .keys_hiprec = &.{
        "pos_embedder",
        "llm_adapter",
        "blocks.0.",
        "blocks.1.adaln_modulation",
        "final_layer",
        "t_embedder",
        "x_embedder",
    },
    .keys_ignore = &.{ "_extra_state", "accum_" },
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

// Mage-Flow (microsoft/Mage) is a 12-layer native-resolution MMDiT that reuses
// Qwen-Image's double-stream block verbatim. Its state dict has *exactly* the
// same set of tensor names as Qwen-Image — only the dimensions differ — so name
// matching alone cannot tell the two apart. ComfyUI's model_detection.py
// disambiguates purely by shape (txt_norm/proj_out are 2560/128 here vs
// 3584/64 for Qwen-Image), and so do we, via `shape_detect`.
//
// Must be listed BEFORE `qwen` in arch_list: Qwen-Image's key set matches
// Mage-Flow's file exactly, so qwen would otherwise win.
//
// `mage_flow` is the `image_model` string ComfyUI itself assigns, so we use it
// verbatim as `general.architecture`.
pub const mageflow = Arch{
    .name = "mage_flow",
    .keys_detect = &.{
        &.{
            "time_text_embed.timestep_embedder.linear_2.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
            "txt_norm.weight",
            "proj_out.weight",
        },
    },
    // The exact pair ComfyUI reads to separate Mage-Flow from Qwen-Image.
    .shape_detect = &.{
        .{ .key = "txt_norm.weight", .dim = 0, .extent = 2560 },
        .{ .key = "proj_out.weight", .dim = 0, .extent = 128 },
    },
    .shape_fix = true,
    .threshhold = null,
    // Only 12 blocks and 4.1B params, so the conditioning/IO path is under 1%
    // of the weights (~38M params) while carrying most of the quantization
    // risk. It lands as F32 in GGUF output, which costs ~144 MiB — about 6% of
    // a Q4_K build. Same trade-off the krea2/ltx2 entries above make.
    //   - txt_norm.weight: also load-bearing for detection. It is 1-D, which
    //     already protects it in GGUF output, but the SafeTensors cluster
    //     formats (MXFP4/MXFP8) would otherwise nibble-pack it and halve the
    //     2560 that ComfyUI matches on.
    //   - img_in / txt_in / proj_out: patch embed/unembed and text input proj.
    //   - norm_out.linear: final AdaLN modulation.
    //   - time_text_embed: timestep embedding MLP.
    .keys_hiprec = &.{
        "txt_norm.weight",
        "img_in.",
        "txt_in.",
        "proj_out.",
        "norm_out.linear",
        "time_text_embed",
    },
    // RMSNorm scales, as for Qwen-Image (shared block implementation).
    .upcast_from_bf16 = &.{
        "txt_norm.weight",
        ".norm_k.weight",
        ".norm_q.weight",
        ".norm_added_k.weight",
        ".norm_added_q.weight",
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
    // MEASURED, not hand-authored: percentile-ranked Q4_K output error on real
    // captured activations (`ggufy calibrate` + `ggufy sensitivity`, 512², 4
    // steps, 4 prompts, r=32×3 buckets), against the BF16 `animosity_krea2Ver10`
    // checkpoint so the baseline is genuine full precision. Keyed on the
    // canonical unprefixed names; `Convert.zig` strips container prefixes before
    // the fallback lookup, so this covers both packagings.
    //
    // The ranking reproduced at Spearman 0.9938 against a capture from a
    // *different* krea2 finetune (the fp8 checkpoint), which is what justifies
    // shipping one file for the architecture rather than per checkpoint.
    // See ACTIVATION_AWARE_PLAN.md §7.
    .sensitivities = @embedFile("sensitivities/krea2.json"),
    .keys_hiprec = &.{
        // Measurement agrees with this list: txtfusion's attention projections
        // are the most damaged layers in the model by a clear margin.
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
    // §8B. Read off TensorPencil's `dit.zig` (`txtBlockForward` and
    // `blockForward`), which mirrors comfy/ldm/krea2/model.py.
    //
    // The text tower is plain pre-norm — `x += attn(rmsNorm(x) ⊙ prenorm)` — so
    // its q/k/v/gate share one static producer and its mlp gate/up share another.
    // Folding there is exact, and free: `.prenorm.scale`/`.postnorm.scale` are
    // already kept at fp32 by `upcast_from_bf16` above, so dividing them by `s`
    // costs one f32 rounding per channel.
    //
    // The 28 main blocks look identical but are not: AdaLN-single inserts
    // `(1 + a) ⊙ n + b` between the norm and the projections, and `b`'s `tvec`
    // half comes from `tproj.1`, which every block shares. There is no per-block
    // static tensor holding block *i*'s shift, so nothing to divide — see
    // `Foldability.runtime_shift`. Same story for `last.linear`, whose shift is
    // `t + last.modulation.lin`.
    .eq_folds = &.{
        .{ .prefix = "txtfusion.", .suffix = ".attn.wq.weight", .fold = .exact },
        .{ .prefix = "txtfusion.", .suffix = ".attn.wk.weight", .fold = .exact },
        .{ .prefix = "txtfusion.", .suffix = ".attn.wv.weight", .fold = .exact },
        .{ .prefix = "txtfusion.", .suffix = ".attn.gate.weight", .fold = .exact },
        .{ .prefix = "txtfusion.", .suffix = ".mlp.gate.weight", .fold = .exact },
        .{ .prefix = "txtfusion.", .suffix = ".mlp.up.weight", .fold = .exact },
        // txtmlp.0.scale → txtmlp.1, a single consumer with no modulation.
        .{ .suffix = "txtmlp.1.weight", .fold = .exact },
        .{ .prefix = "blocks.", .suffix = ".attn.wq.weight", .fold = .runtime_shift },
        .{ .prefix = "blocks.", .suffix = ".attn.wk.weight", .fold = .runtime_shift },
        .{ .prefix = "blocks.", .suffix = ".attn.wv.weight", .fold = .runtime_shift },
        .{ .prefix = "blocks.", .suffix = ".attn.gate.weight", .fold = .runtime_shift },
        .{ .prefix = "blocks.", .suffix = ".mlp.gate.weight", .fold = .runtime_shift },
        .{ .prefix = "blocks.", .suffix = ".mlp.up.weight", .fold = .runtime_shift },
        .{ .suffix = "last.linear.weight", .fold = .runtime_shift },
        // Everything else is `.none` by default, which is correct and worth
        // naming: `attn.wo` reads the attention output, `mlp.down` the SwiGLU
        // product, `first` the raw patches, and `tmlp`/`tproj`/`.projector` the
        // timestep and per-layer conditioning paths. None has a per-channel
        // producer to fold into.
    },
};

/// List of all known architectures, in detection priority order
pub const arch_list = [_]*const Arch{
    &flux,
    &sd3,
    &aura,
    &hidream,
    &anima,
    &cosmos,
    &ltx2,
    &ltxv,
    &hyvid,
    &wan,
    &sdxl,
    &sd1,
    &lumina2,
    &mageflow,
    &qwen,
    &ernie,
    &krea2,
};

/// Look an architecture up by its `name`, for callers that were handed one as a
/// string rather than detecting it — a calibration cache records the arch it was
/// captured from, and the level-1 harness needs the per-arch rules behind it.
pub fn byName(name: []const u8) ?*const Arch {
    for (arch_list) |arch| if (std.mem.eql(u8, arch.name, name)) return arch;
    return null;
}

/// Core matcher: names must match, and any `shape_detect` rules must hold.
/// `tensors` is null when only names are known; an architecture that needs
/// shapes then cannot match, since we have no way to confirm its constraints.
fn archMatches(arch: *const Arch, names: []const []const u8, tensors: ?[]const types.Tensor) bool {
    if (!arch.matches(names)) return false;
    if (arch.shape_detect.len == 0) return true;
    const ts = tensors orelse return false;
    return arch.shapesMatch(ts);
}

fn detectImpl(names: []const []const u8, tensors: ?[]const types.Tensor) ?*const Arch {
    for (arch_list) |arch| {
        if (archMatches(arch, names, tensors)) return arch;
    }
    return null;
}

/// Detect architecture from a list of tensor names.
/// Returns the matching Arch or null if unknown.
///
/// Names alone cannot distinguish architectures that share a tensor-name set
/// (e.g. Mage-Flow vs Qwen-Image); those declare `shape_detect` and are only
/// reachable through `detectArchFromTensors`/`detectArchFromTensorsOrError`.
pub fn detectArch(tensor_names: []const []const u8) ?*const Arch {
    return detectImpl(tensor_names, null);
}

/// Detect architecture from a tensor list using an allocator for large models
pub fn detectArchFromTensors(tensors: []const types.Tensor, allocator: std.mem.Allocator) !?*const Arch {
    const names = try allocator.alloc([]const u8, tensors.len);
    defer allocator.free(names);

    for (tensors, 0..) |t, i| {
        names[i] = t.name;
    }
    return detectImpl(names, tensors);
}

/// Detect architecture and return error if not found or invalid
pub fn detectArchOrError(tensor_names: []const []const u8) ArchError!*const Arch {
    return detectImpl(tensor_names, null) orelse ArchError.UnknownArchitecture;
}

/// Detect architecture from tensors and return error if not found or invalid
pub fn detectArchFromTensorsOrError(tensors: []const types.Tensor, allocator: std.mem.Allocator) ArchError!*const Arch {
    const names = allocator.alloc([]const u8, tensors.len) catch return ArchError.OutOfMemory;
    defer allocator.free(names);

    for (tensors, 0..) |t, i| {
        names[i] = t.name;
    }

    return detectImpl(names, tensors) orelse ArchError.UnknownArchitecture;
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

test "anima vs cosmos detection priority" {
    // A base-Cosmos state dict (no llm_adapter) must resolve to cosmos.
    const cosmos_only = [_][]const u8{
        "net.blocks.0.mlp.layer1.weight",
        "net.blocks.0.adaln_modulation_cross_attn.1.weight",
    };
    try std.testing.expectEqualStrings("cosmos", detectArch(&cosmos_only).?.name);

    // Adding the llm_adapter discriminator must flip detection to anima, even
    // though the cosmos key set is still fully present.
    const anima_sd = [_][]const u8{
        "model.diffusion_model.blocks.0.mlp.layer1.weight",
        "model.diffusion_model.blocks.0.adaln_modulation_cross_attn.1.weight",
        "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
    };
    try std.testing.expectEqualStrings("anima", detectArch(&anima_sd).?.name);
}

test "anima keeps the whole llm_adapter high-precision" {
    // The entire adapter must be unquantized (Forge Neo reroutes it into the
    // text-encoder MixedPrecision path; a quantized adapter breaks its attention).
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.llm_adapter.embed.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.llm_adapter.out_proj.weight"));
    // Backbone weights stay quantizable.
    try std.testing.expect(!anima.isHighPrecision("model.diffusion_model.blocks.5.mlp.layer1.weight"));
    // Reference (silveroxides) hiprec layers: first block, final layer, embedders.
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.blocks.0.mlp.layer1.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.final_layer.linear.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.t_embedder.1.linear_1.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.x_embedder.proj.1.weight"));
    // Only block 1's adaln modulation is protected, not the rest of block 1.
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.blocks.1.adaln_modulation_mlp.1.weight"));
    try std.testing.expect(!anima.isHighPrecision("model.diffusion_model.blocks.1.mlp.layer1.weight"));
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

// Mage-Flow and Qwen-Image share an identical tensor-name set, so the only
// thing separating them is txt_norm.weight[0] (2560 vs 3584) and
// proj_out.weight[0] (128 vs 64) — the same pair ComfyUI keys off.
// The dim arrays live in per-instantiation static storage: `Tensor.dims` is a
// slice, so function-local arrays would dangle once the helper returns.
fn mmditTensors(comptime txt_norm_dim: usize, comptime proj_out_rows: usize) [5]types.Tensor {
    const dims = struct {
        var img_mlp = [_]usize{ 12288, 3072 };
        var linear2 = [_]usize{ 3072, 3072 };
        var norm_added_q = [_]usize{128};
        var txt_norm = [_]usize{txt_norm_dim};
        var proj_out = [_]usize{ proj_out_rows, 3072 };
    };
    return .{
        .{ .name = "time_text_embed.timestep_embedder.linear_2.weight", .type = "BF16", .dims = &dims.linear2, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.attn.norm_added_q.weight", .type = "BF16", .dims = &dims.norm_added_q, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.img_mlp.net.0.proj.weight", .type = "BF16", .dims = &dims.img_mlp, .size = 0, .offset = 0 },
        .{ .name = "txt_norm.weight", .type = "BF16", .dims = &dims.txt_norm, .size = 0, .offset = 0 },
        .{ .name = "proj_out.weight", .type = "BF16", .dims = &dims.proj_out, .size = 0, .offset = 0 },
    };
}

test "mage_flow vs qwen-image disambiguation is by shape only" {
    const allocator = std.testing.allocator;

    var mage = mmditTensors(2560, 128);
    try std.testing.expectEqualStrings("mage_flow", (try detectArchFromTensors(&mage, allocator)).?.name);

    // Qwen-Image: same names, different dims — must not be claimed by mage_flow.
    var qwen_image = mmditTensors(3584, 64);
    try std.testing.expectEqualStrings("qwen", (try detectArchFromTensors(&qwen_image, allocator)).?.name);

    // One matching dimension is not enough; both rules must hold.
    var half_match = mmditTensors(2560, 64);
    try std.testing.expectEqualStrings("qwen", (try detectArchFromTensors(&half_match, allocator)).?.name);
}

test "shape rules reject tensors with missing or too-few dims" {
    var no_dims = [_]usize{};
    var proj_out_dims = [_]usize{ 128, 3072 };
    const tensors = [_]types.Tensor{
        .{ .name = "txt_norm.weight", .type = "BF16", .dims = &no_dims, .size = 0, .offset = 0 },
        .{ .name = "proj_out.weight", .type = "BF16", .dims = &proj_out_dims, .size = 0, .offset = 0 },
    };
    try std.testing.expect(!mageflow.shapesMatch(&tensors));

    // Absent tensor also fails, rather than silently passing.
    try std.testing.expect(!mageflow.shapesMatch(tensors[1..]));
}

test "mage_flow keeps the conditioning and IO path high-precision" {
    const protected = [_][]const u8{
        "txt_norm.weight",
        "img_in.weight",
        "txt_in.weight",
        "proj_out.weight",
        "norm_out.linear.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "model.diffusion_model.proj_out.bias",
    };
    for (protected) |k| try std.testing.expect(mageflow.isHighPrecision(k));

    // The 12 double-stream blocks are the quantization target.
    const backbone = [_][]const u8{
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.11.img_mlp.net.0.proj.weight",
        "transformer_blocks.5.txt_mod.1.weight",
        "transformer_blocks.7.attn.add_v_proj.weight",
    };
    for (backbone) |k| try std.testing.expect(!mageflow.isHighPrecision(k));
}

test "mage_flow upcasts rmsnorm scales" {
    try std.testing.expect(mageflow.shouldUpcast("txt_norm.weight"));
    try std.testing.expect(mageflow.shouldUpcast("transformer_blocks.0.attn.norm_q.weight"));
    try std.testing.expect(mageflow.shouldUpcast("transformer_blocks.3.attn.norm_added_k.weight"));
    try std.testing.expect(!mageflow.shouldUpcast("transformer_blocks.0.attn.to_q.weight"));
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