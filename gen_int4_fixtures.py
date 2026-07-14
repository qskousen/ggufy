#!/usr/bin/env python3
"""Generate int4 ConvRot ("convrot_w4a4") test fixtures from the authoritative
comfy_kitchen implementation, so the Zig quantizer/dequantizer is validated against
exactly what ComfyUI produces and loads.

Run with the ComfyUI venv that provides comfy_kitchen + torch, e.g.:
    /home/qt/genai/comfyui/nvenv/bin/python gen_int4_fixtures.py

Reference: comfy_kitchen.backends.eager.convrot_w4a4
    quantize_convrot_w4a4_weight(w, convrot_groupsize=256, quant_group_size=64,
                                 stochastic_rounding=0)   # 0 => deterministic round-half-even
      -> (qweight int8 [rows, cols//2] nibble-packed, scales f32 [rows])
    dequantize_convrot_w4a4_weight(qweight, scales, convrot_groupsize=256)
      -> f32 [rows, cols]   (un-rotated back to the original basis)

stochastic_rounding=0 makes the reference deterministic (round-half-to-even, clamp [-7,7]),
which is the path ggufy matches bit-for-bit modulo the fast-vs-dense Hadamard boundary.

Input is the real krea2 weight slice already committed as convrot_expected.f32 (16 x 6144 f32).

Outputs (into src/test_fixtures/):
    int4_convrot_weight.u8     packed int4 weight, convrot   [ROWS x COLS/2]  (raw int8 bytes)
    int4_convrot_scale.f32     per-row scale, convrot        [ROWS]
    int4_convrot_expected.f32  convrot dequant (un-rotated)  [ROWS x COLS]
    int4_meta.json             {rows, cols, group_size}
"""
import json
import os

import numpy as np
import torch

from comfy_kitchen.backends.eager.convrot_w4a4 import (
    quantize_convrot_w4a4_weight,
    dequantize_convrot_w4a4_weight,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "test_fixtures")
GROUP_SIZE = 256
QUANT_GROUP_SIZE = 64


def main():
    src = os.path.join(OUT_DIR, "convrot_expected.f32")
    meta = json.load(open(os.path.join(OUT_DIR, "convrot_meta.json")))
    rows, cols = int(meta["rows"]), int(meta["cols"])
    inp = np.fromfile(src, dtype=np.float32).reshape(rows, cols)
    assert cols % GROUP_SIZE == 0 and cols % QUANT_GROUP_SIZE == 0
    print(f"input {src} shape=({rows},{cols})  convrot_gs={GROUP_SIZE}  quant_gs={QUANT_GROUP_SIZE}")

    weight = torch.from_numpy(inp.copy())  # float32 [rows, cols]

    # Deterministic reference (stochastic_rounding=0).
    qweight, scales = quantize_convrot_w4a4_weight(
        weight,
        convrot_groupsize=GROUP_SIZE,
        quant_group_size=QUANT_GROUP_SIZE,
        stochastic_rounding=0,
    )
    assert qweight.dtype == torch.int8 and tuple(qweight.shape) == (rows, cols // 2)
    scales = scales.reshape(rows).to(torch.float32)

    expected = dequantize_convrot_w4a4_weight(
        qweight, scales, convrot_groupsize=GROUP_SIZE, output_dtype=torch.float32
    ).contiguous()

    # Weight bytes are written verbatim (int8 two's-complement == the u8 nibble bytes ggufy compares).
    qweight.numpy().astype(np.int8).tofile(os.path.join(OUT_DIR, "int4_convrot_weight.u8"))
    scales.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "int4_convrot_scale.f32"))
    expected.numpy().astype(np.float32).tofile(os.path.join(OUT_DIR, "int4_convrot_expected.f32"))

    with open(os.path.join(OUT_DIR, "int4_meta.json"), "w") as f:
        json.dump({"rows": rows, "cols": cols, "group_size": GROUP_SIZE}, f)

    err = float(np.abs(expected.numpy() - inp).mean())
    print(f"wrote int4 convrot fixtures to {OUT_DIR}")
    print(f"  convrot mean|err| = {err:.6f}")
    print(f"  scale range [{float(scales.min()):.6g}, {float(scales.max()):.6g}]")


if __name__ == "__main__":
    main()
