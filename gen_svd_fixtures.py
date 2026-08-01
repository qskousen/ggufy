#!/usr/bin/env python3
"""Generate truncated-SVD test fixtures from torch.linalg.svd (known-good data).

Validates ggufy's randomized truncated SVD (src/LinAlg.zig, LEARNED_ROUNDING.md
Phase 1) against the authoritative LAPACK-backed decomposition.

Run with the ComfyUI venv that provides torch:
    /home/qt/genai/comfyui/nvenv/bin/python gen_svd_fixtures.py

Outputs (into src/test_fixtures/, all little-endian f64):
    svd_input.f64   W                [M × N]  row-major
    svd_s.f64       singular values  [K]      (descending)
    svd_u.f64       left  vectors    [M × K]  columns = left  singular vectors
    svd_v.f64       right vectors    [N × K]  columns = right singular vectors (V = Vhᵀ)
    svd_meta.f64    [M, N, K]        as f64

The test matrix has a large spectral gap after the top-K singular values, so the
top-K subspace is well separated and uniquely defined — the property the Phase-1
learned-rounding projection relies on.
"""
import os
import struct

import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "test_fixtures")

M, N = 64, 48
K = 6
SEED = 20260716


def write_f64(name, tensor):
    flat = tensor.detach().to(torch.float64).contiguous().view(-1).cpu().numpy()
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{flat.size}d", *flat.tolist()))
    print(f"  wrote {name}  ({flat.size} f64)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)
    r = min(M, N)

    # Random orthonormal bases via QR of Gaussian matrices.
    ua, _ = torch.linalg.qr(torch.randn(M, r, generator=g, dtype=torch.float64))
    vb, _ = torch.linalg.qr(torch.randn(N, r, generator=g, dtype=torch.float64))

    # Spectrum: a dominant block of K values, then a big gap, then a decaying tail.
    dominant = torch.tensor([100.0 * (0.9 ** i) for i in range(K)], dtype=torch.float64)
    tail = torch.tensor([0.5 * (0.9 ** i) for i in range(r - K)], dtype=torch.float64)
    spectrum = torch.cat([dominant, tail])  # strictly descending, gap at K

    W = (ua * spectrum) @ vb.T  # [M×N], planted spectrum

    # Authoritative reference decomposition.
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U[M×r], S[r], Vh[r×N]
    V = Vh.mH  # [N×r], columns = right singular vectors

    print(f"top-{K+2} singular values: {S[:K+2].tolist()}")
    print(f"gap ratio S[{K}]/S[{K-1}] = {(S[K]/S[K-1]).item():.4g}")

    write_f64("svd_input.f64", W)
    write_f64("svd_s.f64", S[:K])
    write_f64("svd_u.f64", U[:, :K].contiguous())
    write_f64("svd_v.f64", V[:, :K].contiguous())
    write_f64("svd_meta.f64", torch.tensor([M, N, K], dtype=torch.float64))
    print("done.")


if __name__ == "__main__":
    main()
