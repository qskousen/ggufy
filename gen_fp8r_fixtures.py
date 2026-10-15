#!/usr/bin/env python3
"""
Generate fp8r (BFL per-row FP8) reference fixtures from BFL's own export.

Takes the first 8 rows x 64 columns of single_blocks.0.q_proj from
black-forest-labs/flux-3-image-klein variants/fp8r/model.safetensors and
dequantizes them the way torch does: weight.float() * weight_scale[:, None].

Outputs (src/test_fixtures/):
  fp8r_weight.u8    F8_E4M3 bytes  [8 x 64]
  fp8r_scale.f32    per-row scales [8]
  fp8r_expected.f32 dequantized    [8 x 64]

Run from the project root with a torch venv:
  python3 gen_fp8r_fixtures.py /path/to/variants/fp8r/model.safetensors
"""

import os
import sys
import torch
from safetensors import safe_open

ROWS, COLS = 8, 64
OUT_DIR = "src/test_fixtures"
prefix = "single_blocks.0.q_proj"

with safe_open(sys.argv[1], "pt") as f:
    w = f.get_tensor(f"{prefix}.weight")[:ROWS, :COLS].contiguous()
    s = f.get_tensor(f"{prefix}.weight_scale")[:ROWS].contiguous()

assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.float32 and s.dim() == 1
expected = w.float() * s[:, None]

os.makedirs(OUT_DIR, exist_ok=True)
w.view(torch.uint8).numpy().tofile(f"{OUT_DIR}/fp8r_weight.u8")
s.numpy().tofile(f"{OUT_DIR}/fp8r_scale.f32")
expected.numpy().tofile(f"{OUT_DIR}/fp8r_expected.f32")
print("scales:", s.tolist())
