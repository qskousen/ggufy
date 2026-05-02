#!/usr/bin/env python3
"""
Generate reference fixtures for E2M1 / FP4, E8M0, and MXFP4-GGUF-block tests.

Outputs (all in src/test_fixtures/):
  e8m0_decode.f32                 – f32 for each of the 256 E8M0 byte values
  fp4_e2m1_decode.f32             – f32 for each of the 16 E2M1 nibble codes (1 per byte)
  fp4_e2m1_encode_inputs.f32      – sweep of f32 input values for encode testing
  fp4_e2m1_encode_expected.u8     – expected nibble byte for each input (one nibble per byte)
  mxfp4_gguf_test_blocks.bin      – raw GGUF mxfp4 blocks (17 bytes each)
  mxfp4_gguf_test_expected.f32    – expected f32 values decoded from those blocks

Run from the project root:
  venv/bin/python3 gen_quantization_fixtures.py
"""

import struct
import numpy as np
import ml_dtypes
import os

OUT_DIR = "src/test_fixtures"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# E8M0 decode table: all 256 byte values
# ---------------------------------------------------------------------------

e8m0_bytes = np.arange(256, dtype=np.uint8)
e8m0_decoded = e8m0_bytes.view(ml_dtypes.float8_e8m0fnu).astype(np.float32)

# Preserve raw f32 bit patterns (NaN for byte 255).
e8m0_decoded.view(np.uint8).reshape(-1).tofile(
    os.path.join(OUT_DIR, "e8m0_decode.f32")
)
print(f"E8M0 decode: 256 values written")


# ---------------------------------------------------------------------------
# FP4 / E2M1 decode: all 16 nibble codes
# ml_dtypes.float4_e2m1fn stores one 4-bit value per byte (upper nibble = 0).
# ---------------------------------------------------------------------------

all_nibbles = np.arange(16, dtype=np.uint8)
fp4_decoded = all_nibbles.view(ml_dtypes.float4_e2m1fn).astype(np.float32)
fp4_decoded.view(np.uint8).reshape(-1).tofile(
    os.path.join(OUT_DIR, "fp4_e2m1_decode.f32")
)
print(f"FP4/E2M1 decode: {fp4_decoded}")


# ---------------------------------------------------------------------------
# FP4 / E2M1 encode: sweep of f32 inputs
# ---------------------------------------------------------------------------

# Cover the full E2M1 range plus boundary and out-of-range values.
parts = []
# Exact representable values (both signs)
exact = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
for v in exact:
    parts.extend([v, -v])
# Midpoints between representable magnitudes
parts.extend([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0])
parts.extend([-0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -7.0])
# Overflow
parts.extend([100.0, -100.0, 1e10, -1e10])
# Zero / near-zero
parts.extend([0.0, -0.0, 0.001, -0.001])
# Dense linear sweep
parts.extend(np.linspace(-7.0, 7.0, 200, dtype=np.float32).tolist())

fp4_encode_inputs = np.array(parts, dtype=np.float32)
fp4_encode_expected = fp4_encode_inputs.astype(ml_dtypes.float4_e2m1fn).view(np.uint8)

fp4_encode_inputs.tofile(os.path.join(OUT_DIR, "fp4_e2m1_encode_inputs.f32"))
fp4_encode_expected.tofile(os.path.join(OUT_DIR, "fp4_e2m1_encode_expected.u8"))
print(f"FP4/E2M1 encode: {len(fp4_encode_inputs)} values written")


# ---------------------------------------------------------------------------
# MXFP4 GGUF block decode
#
# GGUF mxfp4 block layout (17 bytes per 32-element block):
#   byte 0     : E8M0 scale
#   bytes 1-16 : qs[0..15]
#       qs[j] low  nibble → element j        (j in 0..15)
#       qs[j] high nibble → element j + 16
#
# This is the "first half / second half" packing used by GGML, not the
# sequential OCP packing used in ComfyUI MXFP4 clusters.
# ---------------------------------------------------------------------------

fp4_lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e8m0_to_float(e: int) -> float:
    """Decode an E8M0 byte to float64. byte=255 → NaN."""
    if e == 255:
        return float("nan")
    if e == 0:
        return 2.0 ** -127
    return 2.0 ** (e - 127)


def make_gguf_mxfp4_block(scale_e8m0: int, first16: list, second16: list) -> bytes:
    """Pack a 17-byte GGUF mxfp4 block given 32 nibble codes."""
    qs = bytes([lo | (hi << 4) for lo, hi in zip(first16, second16)])
    return bytes([scale_e8m0]) + qs


def decode_gguf_mxfp4_block(block: bytes) -> list:
    scale = e8m0_to_float(block[0])
    out = []
    for j in range(16):
        out.append(fp4_lut[block[1 + j] & 0xF] * scale)  # element j
    for j in range(16):
        out.append(fp4_lut[(block[1 + j] >> 4) & 0xF] * scale)  # element j+16
    return out


blocks = b""
expected_vals = []

# Block 1: scale=127 (1.0), all 16 FP4 values appear in both halves.
#   qs[j] = j | (j << 4)  →  nibble j in low, nibble j in high
b1 = make_gguf_mxfp4_block(127, list(range(16)), list(range(16)))
blocks += b1
expected_vals += decode_gguf_mxfp4_block(b1)

# Block 2: scale=128 (2.0), nibble 2 (fp4=1.0) everywhere → all elements = 2.0
b2 = make_gguf_mxfp4_block(128, [2] * 16, [2] * 16)
blocks += b2
expected_vals += decode_gguf_mxfp4_block(b2)

# Block 3: scale=130 (4.0), nibble 5 (fp4=3.0) everywhere → all = 12.0
b3 = make_gguf_mxfp4_block(130, [5] * 16, [5] * 16)
blocks += b3
expected_vals += decode_gguf_mxfp4_block(b3)

# Block 4: scale=124 (0.125), nibble 7 (fp4=6.0) → elements = 0.75
#   nibble 15 (fp4=-6.0) → -0.75
b4 = make_gguf_mxfp4_block(124, [7] * 16, [15] * 16)
blocks += b4
expected_vals += decode_gguf_mxfp4_block(b4)

with open(os.path.join(OUT_DIR, "mxfp4_gguf_test_blocks.bin"), "wb") as f:
    f.write(blocks)

expected_arr = np.array(expected_vals, dtype=np.float32)
expected_arr.view(np.uint8).reshape(-1).tofile(
    os.path.join(OUT_DIR, "mxfp4_gguf_test_expected.f32")
)
print(
    f"MXFP4 GGUF blocks: {len(blocks)//17} blocks, "
    f"{len(expected_vals)} decoded values written"
)
print(
    "  Block 1 expected (first 8 / last 8):",
    expected_vals[:8],
    "...",
    expected_vals[-8:],
)
