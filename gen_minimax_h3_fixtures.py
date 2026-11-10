"""Fixtures for MiniMax H3's diffusers-to-ComfyUI repack, read by HTTP range from the
official diffusers release and Comfy-Org's repack of the same weights.

  minimax_h3_pdd_heads.f32     diffusers pdd_proj_out.{0..5}.bias, 6 x 96, head order
  minimax_h3_pdd_bank.f32      Comfy-Org final_layer.video_out.bias, 576
  minimax_h3_inv_freq.f32      Comfy-Org rope.inv_freq, 16
  minimax_h3_fl2va.json        Comfy-Org's full H3 FL2VA tensor names
  minimax_h3_diffusers.json    MiniMax-H3 transformer/ (diffusers) tensor names
  minimax_h3_image_diffusers.json  MiniMax-H3-Image transformer_diffusers/ tensor names
  minimax_h3_image.json        MiniMax-H3-Image transformer/ (MiniMax's own layout) tensor names

Usage: python gen_minimax_h3_fixtures.py
"""

import json
import struct
import urllib.request

import numpy as np

HF = "https://huggingface.co"
IMG = f"{HF}/MiniMaxAI/MiniMax-H3-Image/resolve/main"
AV = f"{HF}/MiniMaxAI/MiniMax-H3/resolve/main"
COMFY_IMG = f"{HF}/Comfy-Org/MiniMax-H3-Image/resolve/main/diffusion_models/minimax_h3_image_pruned_bf16.safetensors"
COMFY_AV = f"{HF}/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
OUT = "src/test_fixtures"

_headers = {}


def fetch(url, a=None, b=None):
    headers = {} if a is None else {"Range": f"bytes={a}-{b}"}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers)).read()


def header(url):
    if url not in _headers:
        n = struct.unpack("<Q", fetch(url, 0, 7))[0]
        h = json.loads(fetch(url, 8, 8 + n - 1))
        h.pop("__metadata__", None)
        _headers[url] = (n, h)
    return _headers[url]


def tensor_f32(url, name):
    n, h = header(url)
    t = h[name]
    assert t["dtype"] == "F32", (name, t["dtype"])
    a, b = t["data_offsets"]
    return np.frombuffer(fetch(url, 8 + n + a, 8 + n + b - 1), "<f4")


def index_names(url):
    return sorted(json.loads(fetch(url))["weight_map"])


def main():
    wm = json.loads(fetch(f"{IMG}/transformer_diffusers/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    heads = [tensor_f32(f"{IMG}/transformer_diffusers/{wm[k]}", k) for k in (f"pdd_proj_out.{i}.bias" for i in range(6))]
    np.concatenate(heads).astype("<f4").tofile(f"{OUT}/minimax_h3_pdd_heads.f32")
    tensor_f32(COMFY_IMG, "final_layer.video_out.bias").astype("<f4").tofile(f"{OUT}/minimax_h3_pdd_bank.f32")
    tensor_f32(COMFY_IMG, "rope.inv_freq").astype("<f4").tofile(f"{OUT}/minimax_h3_inv_freq.f32")

    lists = {
        "minimax_h3_fl2va.json": sorted(header(COMFY_AV)[1]),
        "minimax_h3_diffusers.json": index_names(f"{AV}/transformer/diffusion_pytorch_model.safetensors.index.json"),
        "minimax_h3_image_diffusers.json": sorted(wm),
        "minimax_h3_image.json": index_names(f"{IMG}/transformer/model.safetensors.index.json"),
    }
    for name, names in lists.items():
        with open(f"{OUT}/{name}", "w") as f:
            json.dump(names, f, indent=0)
        print(name, len(names))


if __name__ == "__main__":
    main()
