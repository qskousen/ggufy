#!/usr/bin/env python3
"""Generate ConvRot INT8 test fixtures from a real ComfyUI model, validated against
the authoritative comfy_kitchen dequantize/quantize implementation.

Run with the ComfyUI venv that provides comfy_kitchen + torch, e.g.:
    /home/qt/genai/comfyui/nvenv/bin/python gen_convrot_fixtures.py

Outputs (into src/test_fixtures/):
    convrot_weight.i8       int8 weight    [ROWS × COLS]   (raw bytes)
    convrot_scale.f32       per-row scale  [ROWS]
    convrot_expected.f32    comfy dequant  [ROWS × COLS]   (row-major)
    convrot_requant.i8      comfy re-quantized int8 of the dequant (for write-path check)
    convrot_meta.json       {rows, cols, group_size, convrot}

The default source is the krea2 Int8 model; override with $CONVROT_SRC.
We take the first ROWS rows and ALL columns of one quantized linear so the
Hadamard groups (size 256 along the input dim) stay intact.
"""
import json
import os
import struct
import sys

import numpy as np
import torch

from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

SRC = os.environ.get(
    "CONVROT_SRC",
    "/home/qt/genai/comfyui/models/diffusion_models/krea2/krea2CenterSemiraw_v10Int8.safetensors",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "test_fixtures")
ROWS = 16  # keep the fixture small; must be < full row count


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
    return hdr, base


def read_tensor(path, hdr, base, name):
    v = hdr[name]
    a, b = v["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        raw = f.read(b - a)
    dt = {"I8": np.int8, "F32": np.float32, "U8": np.uint8, "BF16": None}[v["dtype"]]
    if v["dtype"] == "BF16":
        arr = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).view(v["shape"])
        return arr
    arr = np.frombuffer(raw, dtype=dt).reshape(v["shape"])
    return torch.from_numpy(arr.copy())


def main():
    hdr, base = read_header(SRC)
    hdr.pop("__metadata__", None)

    # Find the first convrot cluster: a .comfy_quant whose JSON has convrot == true.
    base_name = None
    cfg = None
    for k in hdr:
        if not k.endswith(".comfy_quant"):
            continue
        blob = bytes(read_tensor(SRC, hdr, base, k).numpy()).decode("utf-8").rstrip("\x00")
        j = json.loads(blob)
        if j.get("convrot"):
            base_name = k[: -len(".comfy_quant")]
            cfg = j
            break
    if base_name is None:
        print("No convrot cluster found in", SRC, file=sys.stderr)
        sys.exit(1)

    gs = int(cfg.get("convrot_groupsize", 256))
    print(f"cluster={base_name}  cfg={cfg}")

    q_full = read_tensor(SRC, hdr, base, base_name + ".weight")          # int8 [out,in]
    s_full = read_tensor(SRC, hdr, base, base_name + ".weight_scale")    # f32  [out,1]
    out_dim, in_dim = q_full.shape
    rows = min(ROWS, out_dim)
    q = q_full[:rows].contiguous()
    s = s_full[:rows].contiguous().float()
    print(f"shape=({out_dim},{in_dim})  slice=({rows},{in_dim})  gs={gs}")
    assert in_dim % gs == 0, "cols must be divisible by group size"

    # Authoritative dequant via comfy_kitchen.
    params = TensorWiseINT8Layout.Params(
        scale=s,
        orig_dtype=torch.float32,
        orig_shape=(rows, in_dim),
        is_weight=True,
        convrot=True,
        convrot_groupsize=gs,
    )
    expected = TensorWiseINT8Layout.dequantize(q, params).float().contiguous()

    # Write-path reference: re-quantize the dequantized weights with the same options.
    requant_q, _ = TensorWiseINT8Layout.quantize(
        expected, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=gs
    )

    # Plain per-row INT8 (no rotation) reference — our quantizer should match this bit-for-bit
    # (identical formula: amax/127, round-half-even, clamp [-128,127]; no fast-transform rounding).
    plain_q, plain_params = TensorWiseINT8Layout.quantize(
        expected, is_weight=True, per_channel=True, convrot=False
    )
    plain_q.numpy().astype(np.int8).tofile(os.path.join(OUT_DIR, "int8_plain_weight.i8"))
    plain_params.scale.reshape(-1).numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "int8_plain_scale.f32"))

    os.makedirs(OUT_DIR, exist_ok=True)
    q.numpy().astype(np.int8).tofile(os.path.join(OUT_DIR, "convrot_weight.i8"))
    s.reshape(-1).numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "convrot_scale.f32"))
    expected.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "convrot_expected.f32"))
    requant_q.numpy().astype(np.int8).tofile(os.path.join(OUT_DIR, "convrot_requant.i8"))
    with open(os.path.join(OUT_DIR, "convrot_meta.json"), "w") as f:
        json.dump({"rows": rows, "cols": in_dim, "group_size": gs, "convrot": True}, f)

    print("wrote fixtures to", OUT_DIR)
    print(f"  expected range [{expected.min():.5f}, {expected.max():.5f}]  std {expected.std():.5f}")


if __name__ == "__main__":
    main()
