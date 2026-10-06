#!/usr/bin/env python3
"""Generate W6A8_INT8 test fixtures from comfy_kitchen, so the Zig quantizer and dequantizer
are checked against what ComfyUI actually produces and loads.

Run with the ComfyUI virtualenv that provides comfy_kitchen (0.2.37 or later) and torch.

W6A8 is AsymW4A8Int8Layout at bits=6: uniform codes q+32 in [1, 63], no codebook, and a
row of 3K/4 bytes (a K/2-byte nibble plane, then a K/4-byte plane of each code's top two
bits). The fp8 group scale goes through a grid-aware search over its e4m3 neighbours.

Two inputs:
    main  the weight slice already committed as convrot_expected.f32, so this format, W4A8
          and int4 are measured on the same data
    edge  a small synthetic weight with all-zero groups, an all-zero row and one dominant
          outlier, which sends s_rel to 0 and to subnormals and drives the scale search
          into both of its clamps

The main slice is also quantized with the rotation skipped (prefix w6a8_norot_, weight,
s_rel and s_channel only). The refit rounds w / scale, so a last-bit difference in the
rotation can flip a code and move a scale byte, and comfy_kitchen itself moves a few bytes
between an f32 and an f64 rotation. Without the rotation the Zig side must match exactly.

Outputs, into src/test_fixtures, for each of main and edge (prefix w6a8_ and w6a8_edge_):
    weight.u8       packed codes, [rows, 3*cols/4]
    s_rel.u8        per-group scale as fp8 e4m3, [rows, cols/16]
    s_channel.f32   per-row scale, [rows]
    expected.f32    decoded weight, un-rotated, [rows, cols]
    meta.json       shapes and group sizes
The edge input itself is written as w6a8_edge_input.f32.
"""
import json
import os

import numpy as np
import torch

from comfy_kitchen.backends.eager.w4a8_int8 import (
    _quantize_rotated_w4a8_int8_weight,
    dequantize_w4a8_int8_weight,
    quantize_w4a8_int8_weight,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "test_fixtures")
GROUP_SIZE = 16
CONVROT_GROUPSIZE = 256
BITS = 6


def edge_weight():
    rows, cols = 8, 512
    gen = torch.Generator().manual_seed(1234)
    w = torch.randn(rows, cols, generator=gen) * 0.02
    w[1, :] = 0.0  # every group zero
    w[2, 256:] = 0.0  # a whole rotation group zero, so its scale groups are zero after rotating
    w[3, 7] = 40.0  # one outlier dwarfs the rest of the row: tiny and zero s_rel bytes
    w[4, :] *= 1e-6  # far below the fp8 range relative to nothing, scales still positive
    w[5, ::17] = 3.0  # sparse spikes
    return w


def emit_norot(prefix, weight):
    rows, cols = weight.shape
    packed, s_rel, s_channel, correction, codebook = _quantize_rotated_w4a8_int8_weight(
        weight,
        group_size=GROUP_SIZE,
        symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
        codebook=False,
        bits=BITS,
    )
    assert correction is None and codebook is None
    assert tuple(packed.shape) == (rows, cols * 3 // 4), packed.shape
    for name, arr in (
        ("weight.u8", packed.cpu().numpy().view(np.uint8)),
        ("s_rel.u8", s_rel.cpu().view(torch.uint8).numpy()),
        ("s_channel.f32", s_channel.cpu().float().numpy()),
    ):
        arr.tofile(os.path.join(OUT_DIR, prefix + name))
        print(f"  {prefix + name:28s} {arr.dtype!s:8s} {list(arr.shape)} -> {arr.nbytes} bytes")


def emit(prefix, weight):
    rows, cols = weight.shape
    packed, s_rel, s_channel, correction, codebook = quantize_w4a8_int8_weight(
        weight,
        group_size=GROUP_SIZE,
        convrot_groupsize=CONVROT_GROUPSIZE,
        symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
        bits=BITS,
    )
    assert correction is None and codebook is None, "6-bit storage has no correction or codebook"
    assert tuple(packed.shape) == (rows, cols * 3 // 4), packed.shape
    assert tuple(s_rel.shape) == (rows, cols // GROUP_SIZE), s_rel.shape
    assert tuple(s_channel.shape) == (rows,), s_channel.shape

    expected = dequantize_w4a8_int8_weight(
        packed,
        s_rel,
        s_channel,
        group_size=GROUP_SIZE,
        convrot_groupsize=CONVROT_GROUPSIZE,
        output_dtype=torch.float32,
    )

    def dump(name, arr):
        path = os.path.join(OUT_DIR, prefix + name)
        arr.tofile(path)
        print(f"  {prefix + name:28s} {arr.dtype!s:8s} {list(arr.shape)} -> {arr.nbytes} bytes")

    dump("weight.u8", packed.cpu().numpy().view(np.uint8))
    dump("s_rel.u8", s_rel.cpu().view(torch.uint8).numpy())
    dump("s_channel.f32", s_channel.cpu().float().numpy())
    dump("expected.f32", expected.cpu().float().numpy())

    meta = {
        "rows": rows,
        "cols": cols,
        "group_size": GROUP_SIZE,
        "convrot_groupsize": CONVROT_GROUPSIZE,
        "bits": BITS,
    }
    with open(os.path.join(OUT_DIR, prefix + "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    err = (expected - weight).norm() / weight.norm().clamp(min=1e-30)
    raw = s_rel.view(torch.uint8)
    print(f"  relative L2 error {err:.6f}, s_rel zero bytes {(raw == 0).sum().item()}")


def main():
    print(f"writing fixtures to {OUT_DIR}")
    main_w = torch.from_numpy(
        np.fromfile(os.path.join(OUT_DIR, "convrot_expected.f32"), dtype=np.float32)
        .reshape(16, 6144)
        .copy()
    )
    emit("w6a8_", main_w)
    emit_norot("w6a8_norot_", main_w)

    edge = edge_weight()
    edge.numpy().tofile(os.path.join(OUT_DIR, "w6a8_edge_input.f32"))
    emit("w6a8_edge_", edge)


if __name__ == "__main__":
    main()
