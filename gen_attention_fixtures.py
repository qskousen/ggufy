#!/usr/bin/env python3
"""
Generate the attention-module fixtures: for one checkpoint per diffusion family,
the module paths at which ComfyUI accepts a `<module>.comfy_attention.config`
tensor (see ComfyUI's QUANTIZATION.md, "Diffusion attention preferences").

Each checkpoint's header is turned into meta tensors, run through ComfyUI's own
model detection, and instantiated on the meta device; every ComfyAttention in
the resulting module tree is a valid path. Nothing is read beyond the headers
(GGUF goes through ComfyUI-GGUF's loader, which memory-maps).

Outputs src/test_fixtures/attention/<family>.json:
  {"source": file, "prefix": diffusion-model key prefix, "tensors": [names],
   "modules": [paths relative to the diffusion model]}

Run from the project root with ComfyUI's venv:
  /home/qt/genai/comfyui/nvenv/bin/python gen_attention_fixtures.py COMFYUI_DIR [family ...]
COMFYUI_DIR must be a checkout new enough to know every family below.
"""

import importlib
import json
import os
import re
import struct
import sys
import urllib.request

D = "/mnt/watch/models/diffusion"
T = "/home/qt/genai/ggufy/test-models"
SOURCES = {
    "flux3": "/mnt/watch/flux3-klein/comfy/diffusion_models/flux3_image_klein_sd_bf16.safetensors",
    "flux": f"{D}/diffusion_models/flux.d/artsyVibe_v2FP16.safetensors",
    "sd1": f"{D}/checkpoints/dreamshaper_8.safetensors",
    "sdxl": f"{T}/illustrious_v110.safetensors",
    "anima": f"{D}/diffusion_models/anima/anima_baseV10.safetensors",
    "ernie": f"{D}/diffusion_models/ernie/ernie-image.safetensors",
    "krea2": f"{T}/krea2_turbo_bf16.safetensors",
    "mage_flow": f"{D}/diffusion_models/mageflow/mageFlow_mageFlow4B.safetensors",
    "qwen_image21": f"{D}/diffusion_models/qwen2.1/qwen_image_2.1_bf16.safetensors",
    "qwen_image": f"{D}/diffusion_models/qwen/hyphoria_qwen_v1_Q6_K.gguf",
    "sensenova_u15": f"{D}/diffusion_models/sensenova/sensenovaU158BMot_sft.safetensors",
    "moss_image": "hf:Comfy-Org/MOSS-Image:diffusion_models/moss_image_sft_bf16.safetensors",
    "minimax_h3": f"{D}/diffusion_models/h3/10erosMaxInt8Ref2va_v10Beta.safetensors",
    "ltx2": f"{D}/diffusion_models/ltx-2/ltx-2.3-22b-dev-Q4_K_M.gguf",
    "wan": f"{D}/diffusion_models/wan2.2/wan22EnhancedNSFWCameraPrompt_nsfwV2Q8High.gguf",
    "lumina2": f"{T}/z_image_bf16.safetensors",
    "sd3": "hf:Comfy-Org/stable-diffusion-3.5-fp8:sd3.5_large_fp8_scaled.safetensors",
    "aura": "hf:fal/AuraFlow-v0.3:aura_flow_0.3.safetensors",
    "hidream": "hf:Comfy-Org/HiDream-I1_ComfyUI:split_files/diffusion_models/hidream_i1_dev_bf16.safetensors",
    "cosmos": "hf:Comfy-Org/Cosmos_Predict2_repackaged:cosmos_predict2_2B_t2i.safetensors",
    "hyvid": "hf:Comfy-Org/HunyuanVideo_repackaged:split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors",
    "ltxv": "hf:Lightricks/LTX-Video:ltx-video-2b-v0.9.5.safetensors",
}
OUT_DIR = "src/test_fixtures/attention"

comfy_dir = sys.argv[1]
families = sys.argv[2:] or list(SOURCES)
out_dir = os.path.abspath(OUT_DIR)
os.chdir(comfy_dir)
sys.path.insert(0, comfy_dir)
import comfy.options
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]
import comfy.cli_args
import torch
import comfy.model_detection
import comfy.utils
from comfy.ldm.modules.attention import ComfyAttention

DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
}


def read_header(src):
    if src.startswith("hf:"):
        _, repo, path = src.split(":", 2)
        tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
        url = f"https://huggingface.co/{repo}/resolve/main/{path}"

        def get(a, b):
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok, "Range": f"bytes={a}-{b}"})
            return urllib.request.urlopen(req).read()

        n = struct.unpack("<Q", get(0, 7))[0]
        return json.loads(get(8, 8 + n - 1))
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def load(src):
    """State dict of meta tensors (or memory-mapped GGUF tensors), and metadata."""
    if src.endswith(".gguf"):
        sys.path.insert(0, "/home/qt/genai")
        gg = importlib.import_module("ComfyUI-GGUF.loader")
        sd, extra = gg.gguf_sd_loader(src)
        return sd, extra.get("metadata", {}) or {}
    h = read_header(src)
    meta = h.pop("__metadata__", {}) or {}
    sd = {k: torch.empty(v["shape"], dtype=DTYPES.get(v["dtype"], torch.float32), device="meta") for k, v in h.items()}
    return sd, meta


for fam in families:
    src = SOURCES[fam]
    sd, meta = load(src)
    names = sorted(sd)
    # As comfy.sd.load_diffusion_model_state_dict does: strip a checkpoint's
    # diffusion-model prefix when it has one, then detect.
    prefix = comfy.model_detection.unet_prefix_from_state_dict(sd)
    unet_sd = comfy.utils.state_dict_prefix_replace(sd, {prefix: ""}, filter_keys=True)
    if not unet_sd:
        prefix, unet_sd = "", sd
    cfg = comfy.model_detection.model_config_from_unet(unet_sd, "", metadata=meta)
    if cfg is None:
        print(f"{fam}: ComfyUI does not detect {src}", file=sys.stderr)
        sys.exit(1)
    with torch.device("meta"):
        model = cfg.get_model(unet_sd, "", device=torch.device("meta"))
    modules = sorted(n for n, m in model.diffusion_model.named_modules() if isinstance(m, ComfyAttention))
    # ComfyAttention is the child module; the config sits under it.
    modules = [m[: -len(".comfy_attention")] for m in modules]
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{fam}.json", "w") as f:
        json.dump({"source": os.path.basename(src.split(":")[-1]), "prefix": prefix, "tensors": names, "modules": modules}, f, indent=0)
    patterns = sorted(set(re.sub(r"(?<=\.)\d+(?=\.|$)|^\d+(?=\.)", "N", m) for m in modules))
    print(f"{fam}: {type(cfg).__name__}, {len(modules)} modules, prefix {prefix!r}: {patterns}", flush=True)
