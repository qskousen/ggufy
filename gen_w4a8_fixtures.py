#!/usr/bin/env python3
"""Generate ASYM_W4A8_INT8 test fixtures from comfy_kitchen, so the Zig quantizer and
dequantizer are checked against what ComfyUI actually produces and loads.

Run with the ComfyUI virtualenv that provides comfy_kitchen and torch.

Unlike convrot_w4a4, the packed nibbles are unsigned codebook indices from 0 to 15 rather
than signed values, and the decode rounds codebook[i] * s_rel to a whole number before
scaling by s_channel. The fixtures pin both.

Input is the weight slice already committed as convrot_expected.f32, so this format and
int4 are measured on the same data.

Outputs, into src/test_fixtures:
    w4a8_weight.u8       packed 4-bit codebook indices, [rows, cols/2]
    w4a8_s_rel.u8        per-group scale as fp8 e4m3, [rows, cols/16]
    w4a8_s_channel.f32   per-row scale, [rows]
    w4a8_codebook.f32    the 16 levels
    w4a8_expected.f32    decoded weight, un-rotated, [rows, cols]
    w4a8_meta.json       shapes and group sizes
"""
import json
import os

import numpy as np
import torch

from comfy_kitchen.backends.eager.w4a8_int8 import (
    _FIXED_LUT,
    dequantize_w4a8_int8_weight,
    quantize_w4a8_int8_weight,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "test_fixtures")
GROUP_SIZE = 16
CONVROT_GROUPSIZE = 256
ROWS = 16
COLS = 6144


def main():
    src = os.path.join(OUT_DIR, "convrot_expected.f32")
    weight = torch.from_numpy(
        np.fromfile(src, dtype=np.float32).reshape(ROWS, COLS).copy()
    )

    packed, s_rel, s_channel, correction, codebook = quantize_w4a8_int8_weight(
        weight,
        group_size=GROUP_SIZE,
        convrot_groupsize=CONVROT_GROUPSIZE,
        symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
        codebook=True,
    )
    assert correction is None, "symmetric+codebook path must not emit a correction tensor"
    assert tuple(packed.shape) == (ROWS, COLS // 2), packed.shape
    assert tuple(s_rel.shape) == (ROWS, COLS // GROUP_SIZE), s_rel.shape
    assert tuple(s_channel.shape) == (ROWS,), s_channel.shape
    assert tuple(codebook.shape) == (16,), codebook.shape

    # Did comfy pick the fixed table or fit one? ggufy always writes the fixed table, so a
    # fitted codebook here would make the fixture untestable against it.
    fixed = torch.tensor(_FIXED_LUT, dtype=torch.float32)
    is_fixed = bool(torch.equal(codebook.cpu().float(), fixed))

    expected = dequantize_w4a8_int8_weight(
        packed,
        s_rel,
        s_channel,
        codebook=codebook,
        group_size=GROUP_SIZE,
        convrot_groupsize=CONVROT_GROUPSIZE,
        output_dtype=torch.float32,
    )

    def dump(name, arr):
        path = os.path.join(OUT_DIR, name)
        arr.tofile(path)
        print(f"  {name:24s} {arr.dtype!s:8s} {list(arr.shape)} -> {arr.nbytes} bytes")

    print(f"writing fixtures to {OUT_DIR}")
    # s_rel is fp8, so write raw bytes and let the Zig side decode them.
    dump("w4a8_weight.u8", packed.cpu().numpy().view(np.uint8))
    dump("w4a8_s_rel.u8", s_rel.cpu().view(torch.uint8).numpy())
    dump("w4a8_s_channel.f32", s_channel.cpu().float().numpy())
    dump("w4a8_codebook.f32", codebook.cpu().float().numpy())
    dump("w4a8_expected.f32", expected.cpu().float().numpy())

    meta = {
        "rows": ROWS,
        "cols": COLS,
        "group_size": GROUP_SIZE,
        "convrot_groupsize": CONVROT_GROUPSIZE,
        "fixed_lut": is_fixed,
    }
    with open(os.path.join(OUT_DIR, "w4a8_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"  w4a8_meta.json           {meta}")

    err = (expected - weight).norm() / weight.norm()
    print(f"\nreference relative L2 error: {err:.6f}  (fixed_lut={is_fixed})")
    if not is_fixed:
        print("WARNING: comfy fitted a per-tensor codebook for this input; ggufy always")
        print("         writes the frozen LUT, so the quantize test will not match.")


if __name__ == "__main__":
    main()
