#!/usr/bin/env python3
"""
Generate FP8 reference fixtures for DataTransform.zig tests.

Outputs (all in src/test_fixtures/):
  fp8_test_inputs.f32          – raw little-endian f32 values used for encode tests
  fp8_e4m3fn_encoded.u8        – expected E4M3FN bytes for each input
  fp8_e5m2_encoded.u8          – expected E5M2 bytes for each input
  fp8_e4m3fn_decode.f32        – expected f32 values when decoding bytes 0x00..0xFF
  fp8_e5m2_decode.f32          – expected f32 values when decoding bytes 0x00..0xFF

Run from the project root:
  venv/bin/python3 gen_fp8_fixtures.py
"""

import struct
import numpy as np
import ml_dtypes
import os

OUT_DIR = "src/test_fixtures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build encode test inputs: a sweep that covers all interesting regions
# ---------------------------------------------------------------------------

parts = []

# Zero and negative zero
parts.append(np.array([0.0, -0.0], dtype=np.float32))

# Exact subnormal E4M3FN values: mant * 2^-9, mant in 1..7
# (biased exp=0 in E4M3FN, smallest unit is 2^-9)
subnorm_e4m3 = np.array(
    [m * 2**-9 for m in range(1, 8)] + [-m * 2**-9 for m in range(1, 8)],
    dtype=np.float32,
)
parts.append(subnorm_e4m3)

# Exact subnormal E5M2 values: mant * 2^-16, mant in 1..3
subnorm_e5m2 = np.array(
    [m * 2**-16 for m in range(1, 4)] + [-m * 2**-16 for m in range(1, 4)],
    dtype=np.float32,
)
parts.append(subnorm_e5m2)

# All exactly representable E4M3FN normal values
# biased_exp 1..15 (unbiased -6..8), mant 0..6 (7 = NaN when exp=15)
e4m3_normals = []
for biased_exp in range(1, 16):
    for mant in range(8):
        if biased_exp == 15 and mant == 7:
            continue  # NaN encoding, skip
        val = (1.0 + mant / 8.0) * 2 ** (biased_exp - 7)
        e4m3_normals.extend([val, -val])
parts.append(np.array(e4m3_normals, dtype=np.float32))

# All exactly representable E5M2 normal values
# biased_exp 1..30 (unbiased -14..15), mant 0..3
e5m2_normals = []
for biased_exp in range(1, 31):
    for mant in range(4):
        val = (1.0 + mant / 4.0) * 2 ** (biased_exp - 15)
        e5m2_normals.extend([val, -val])
parts.append(np.array(e5m2_normals, dtype=np.float32))

# Values just around E4M3FN overflow boundary (448.0)
parts.append(np.array([447.9, 448.0, 448.1, 449.0, 500.0, 1000.0, 1e10], dtype=np.float32))
parts.append(np.array([-447.9, -448.0, -448.1, -449.0, -500.0, -1000.0, -1e10], dtype=np.float32))

# Values just around E5M2 overflow boundary (57344.0)
parts.append(np.array([57343.0, 57344.0, 57345.0, 60000.0, 1e8], dtype=np.float32))
parts.append(np.array([-57343.0, -57344.0, -57345.0, -60000.0, -1e8], dtype=np.float32))

# Dense sweep of normal values across both formats' ranges
parts.append(np.linspace(-500.0, 500.0, 5000, dtype=np.float32))
parts.append(np.linspace(-60000.0, 60000.0, 5000, dtype=np.float32))

# Small values near the subnormal/normal boundary
parts.append(np.linspace(-0.1, 0.1, 2000, dtype=np.float32))

# Special IEEE values
parts.append(np.array([np.inf, -np.inf, np.nan], dtype=np.float32))

inputs = np.concatenate(parts)

# ---------------------------------------------------------------------------
# Encode: inputs -> FP8
# ---------------------------------------------------------------------------

e4m3_encoded = inputs.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
e5m2_encoded = inputs.astype(ml_dtypes.float8_e5m2).view(np.uint8)

assert len(e4m3_encoded) == len(inputs)
assert len(e5m2_encoded) == len(inputs)

inputs.tofile(os.path.join(OUT_DIR, "fp8_test_inputs.f32"))
e4m3_encoded.tofile(os.path.join(OUT_DIR, "fp8_e4m3fn_encoded.u8"))
e5m2_encoded.tofile(os.path.join(OUT_DIR, "fp8_e5m2_encoded.u8"))

print(f"Encode fixtures: {len(inputs)} values")

# ---------------------------------------------------------------------------
# Decode: all 256 byte values -> f32
# ---------------------------------------------------------------------------

all_bytes = np.arange(256, dtype=np.uint8)

decoded_e4m3 = all_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
decoded_e5m2 = all_bytes.view(ml_dtypes.float8_e5m2).astype(np.float32)

# Store as raw f32 bytes (NaN bit-patterns preserved via view trick)
# We re-cast through uint8 view to avoid any quiet-NaN normalisation
decoded_e4m3.view(np.uint8).reshape(-1).tofile(
    os.path.join(OUT_DIR, "fp8_e4m3fn_decode.f32")
)
decoded_e5m2.view(np.uint8).reshape(-1).tofile(
    os.path.join(OUT_DIR, "fp8_e5m2_decode.f32")
)

print(f"Decode fixtures: 256 values each (E4M3FN, E5M2)")

# ---------------------------------------------------------------------------
# Sanity-check: round-trip should be lossless for exact representable values
# ---------------------------------------------------------------------------

rt_fail = 0
for v in e4m3_normals:
    enc = np.array([v], dtype=np.float32).astype(ml_dtypes.float8_e4m3fn)
    dec = enc.astype(np.float32)[0]
    if not np.isclose(v, dec, rtol=1e-3):
        print(f"  E4M3FN round-trip mismatch: {v} -> {dec}")
        rt_fail += 1

if rt_fail == 0:
    print("Round-trip sanity check passed.")
else:
    print(f"Round-trip sanity check: {rt_fail} failures.")
