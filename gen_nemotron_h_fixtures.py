#!/usr/bin/env python3
"""Generate the Nemotron-H MoE fixture: a micro checkpoint in the layout NVIDIA
ships (backbone.* names, f32, per-expert weights, a unitron image-token
manifest), and the GGUF llama.cpp's own converter writes for it.

Run with the ComfyUI venv that provides torch + safetensors + transformers:
    /home/qt/genai/comfyui/nvenv/bin/python gen_nemotron_h_fixtures.py

Outputs (into src/test_fixtures/nemotron_h/):
    model.safetensors                  f32 checkpoint
    config.json                        nemotron_h config, pattern "MEM*E"
    tokenizer.json                     tiny byte-level BPE with the real model's split regex
    tokenizer_config.json
    unitron_extension_manifest.json    framing and vision token ranges past the base vocab
    ref-f32.gguf                       convert_hf_to_gguf.py --outtype f32

The tokenizer is too small for llama.cpp's pre-tokenizer hash to know it, so
the converter runs with that lookup patched to return "pixtral", the tag it
gives the real Nemotron tokenizer.

$LLAMA_CPP points at the llama.cpp checkout (default ~/genai/llama.cpp).
"""
import json
import os
import runpy
import sys

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "src", "test_fixtures", "nemotron_h")
LLAMA = os.environ.get("LLAMA_CPP", os.path.expanduser("~/genai/llama.cpp"))

PATTERN = "MEM*E"
H = 16
MAMBA_HEADS, MAMBA_HEAD_DIM, N_GROUPS, STATE, CONV = 8, 8, 8, 8, 4
HEADS, KV_HEADS, HEAD_DIM = 4, 2, 8
EXPERTS, TOPK, MOE_FF, SHARED_FF = 4, 2, 8, 12
BASE_VOCAB, N_VISION, VOCAB = 32, 20, 64

CONFIG = {
    "architectures": ["NemotronHForCausalLM"],
    "model_type": "nemotron_h",
    "attention_bias": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "chunk_size": 128,
    "conv_kernel": CONV,
    "expand": 2,
    "head_dim": HEAD_DIM,
    "hidden_size": H,
    "hybrid_override_pattern": PATTERN,
    "intermediate_size": MOE_FF,
    "layer_norm_epsilon": 1e-05,
    "mamba_head_dim": MAMBA_HEAD_DIM,
    "mamba_hidden_act": "silu",
    "mamba_num_heads": MAMBA_HEADS,
    "mamba_proj_bias": False,
    "max_position_embeddings": 4096,
    "mlp_bias": False,
    "mlp_hidden_act": "relu2",
    "moe_intermediate_size": MOE_FF,
    "moe_shared_expert_intermediate_size": SHARED_FF,
    "n_group": 3,
    "n_groups": N_GROUPS,
    "n_routed_experts": EXPERTS,
    "n_shared_experts": 5,
    "norm_eps": 1e-05,
    "norm_topk_prob": True,
    "num_attention_heads": HEADS,
    "num_experts_per_tok": TOPK,
    "num_hidden_layers": len(PATTERN),
    "num_key_value_heads": KV_HEADS,
    "partial_rotary_factor": 1.0,
    "rope_theta": 10000,
    "routed_scaling_factor": 2.5,
    "ssm_state_size": STATE,
    "tie_word_embeddings": False,
    "topk_group": 1,
    "use_bias": False,
    "use_conv_bias": True,
    "vocab_size": VOCAB,
}

SPLIT = ("[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
         "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
         "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+")
SPECIAL = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"]


def tokenizer():
    pieces = SPECIAL + list("abcdehlnorstwĠ") + ["Ġt", "he", "Ġthe", "ll", "Ġw", "or", "ld", "Ġa", "an", "nd", "Ġs", "st", "er"]
    assert len(pieces) == BASE_VOCAB, len(pieces)
    vocab = {p: i for i, p in enumerate(pieces)}
    merges = [["Ġ", "t"], ["h", "e"], ["Ġt", "he"], ["l", "l"], ["Ġ", "w"], ["o", "r"], ["l", "d"],
              ["Ġ", "a"], ["a", "n"], ["n", "d"], ["Ġ", "s"], ["s", "t"], ["e", "r"]]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for i, s in enumerate(SPECIAL)],
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": merges},
    }


def manifest():
    start = BASE_VOCAB + 3
    return {
        "base_vocab_size": BASE_VOCAB,
        "tokenizer_mode": "virtual",
        "n_vision": N_VISION,
        "framing_tokens": ["<|img_start|>", "<|img_end|>", "<|vis_row|>"],
        "model_vocab_size": start + N_VISION,
        "embedding_rows": VOCAB,
        "unitron_token_ranges": {
            "framing": {"<|img_start|>": BASE_VOCAB, "<|img_end|>": BASE_VOCAB + 1, "<|vis_row|>": BASE_VOCAB + 2},
            "vision": {"token_format": "<|vis_{i}|>", "start_id": start, "end_id": start + N_VISION, "count": N_VISION},
        },
    }


def weights():
    g = torch.Generator().manual_seed(0)

    def r(*shape, scale=0.1, base=0.0):
        return (torch.randn(shape, generator=g) * scale + base).float()

    d_inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    conv_dim = d_inner + 2 * N_GROUPS * STATE
    in_proj = 2 * d_inner + 2 * N_GROUPS * STATE + MAMBA_HEADS
    t = {"backbone.embeddings.weight": r(VOCAB, H), "backbone.norm_f.weight": r(H, base=1.0),
         "lm_head.weight": r(VOCAB, H)}
    for i, kind in enumerate(PATTERN):
        p = f"backbone.layers.{i}."
        t[p + "norm.weight"] = r(H, base=1.0)
        m = p + "mixer."
        if kind == "M":
            t[m + "in_proj.weight"] = r(in_proj, H)
            t[m + "conv1d.weight"] = r(conv_dim, 1, CONV)
            t[m + "conv1d.bias"] = r(conv_dim)
            t[m + "A_log"] = r(MAMBA_HEADS, scale=0.5)
            t[m + "D"] = r(MAMBA_HEADS, base=1.0)
            t[m + "dt_bias"] = r(MAMBA_HEADS)
            t[m + "norm.weight"] = r(d_inner, base=1.0)
            t[m + "out_proj.weight"] = r(H, d_inner)
        elif kind == "*":
            t[m + "q_proj.weight"] = r(HEADS * HEAD_DIM, H)
            t[m + "k_proj.weight"] = r(KV_HEADS * HEAD_DIM, H)
            t[m + "v_proj.weight"] = r(KV_HEADS * HEAD_DIM, H)
            t[m + "o_proj.weight"] = r(H, HEADS * HEAD_DIM)
        else:
            t[m + "gate.weight"] = r(EXPERTS, H)
            t[m + "gate.e_score_correction_bias"] = r(EXPERTS)
            for x in range(EXPERTS):
                t[m + f"experts.{x}.up_proj.weight"] = r(MOE_FF, H)
                t[m + f"experts.{x}.down_proj.weight"] = r(H, MOE_FF)
            t[m + "shared_experts.up_proj.weight"] = r(SHARED_FF, H)
            t[m + "shared_experts.down_proj.weight"] = r(H, SHARED_FF)
    return t


def main():
    os.makedirs(OUT, exist_ok=True)
    save_file(weights(), os.path.join(OUT, "model.safetensors"))
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=1)
    with open(os.path.join(OUT, "tokenizer.json"), "w") as f:
        json.dump(tokenizer(), f, ensure_ascii=False)
    with open(os.path.join(OUT, "tokenizer_config.json"), "w") as f:
        json.dump({"bos_token": "<s>", "eos_token": "<|im_end|>", "unk_token": "<unk>", "pad_token": "<unk>",
                   "tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 4096}, f, indent=1)
    with open(os.path.join(OUT, "unitron_extension_manifest.json"), "w") as f:
        json.dump(manifest(), f, indent=1)

    sys.path.insert(0, LLAMA)
    sys.path.insert(1, os.path.join(LLAMA, "gguf-py"))
    import conversion.base
    conversion.base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "pixtral"
    sys.argv = ["convert_hf_to_gguf.py", OUT, "--outtype", "f32", "--outfile", os.path.join(OUT, "ref-f32.gguf")]
    runpy.run_path(os.path.join(LLAMA, "convert_hf_to_gguf.py"), run_name="__main__")


if __name__ == "__main__":
    main()
