#!/usr/bin/env python3
"""Generate the golden safetensors containers for the calibration cache writer.

Build-time only, like gen_fp8_fixtures.py / gen_quantization_fixtures.py: the Zig
writer in src/CalibrationCache.zig must produce these bytes exactly, so the
container conventions (dtype spellings, entry ordering, offset arithmetic, header
padding) are pinned against the reference implementation rather than against
ourselves.

Two files, because the reference implementation stores `__metadata__` in a Rust
HashMap and emits its keys in a *per-process random* order — so a file with
metadata cannot be byte-compared against anything:

  calib_container.safetensors       tensors only     — the byte-for-byte pin
  calib_container_meta.safetensors  + __metadata__   — read-side only: our reader
                                                       must parse the reference's
                                                       metadata map whatever order
                                                       it happens to be in

Both mirror one (layer, bucket) group of a real cache.

    pip install safetensors numpy
    python3 gen_calib_fixtures.py
"""

import pathlib

import numpy as np
from safetensors.numpy import save

FIXTURES = pathlib.Path(__file__).parent / "src" / "test_fixtures"

LAYER = "blocks.0.attn.wq.weight"
BUCKET = "b0"


def key(field: str) -> str:
    return f"{LAYER}/{BUCKET}/{field}"


def emit(path: pathlib.Path, tensors: dict, metadata: dict | None) -> None:
    blob = save(tensors, metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    header_len = int.from_bytes(blob[:8], "little")
    print(f"wrote {path} ({len(blob)} bytes, header {header_len} bytes)")
    print("  " + blob[8 : 8 + header_len].decode())


def main() -> None:
    # diag is accumulated in f64 and narrowed to f32 on write; the values here are
    # exactly representable in both, so the narrowing is not what is being tested.
    tensors = {
        key("diag"): np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float64).astype(np.float32),
        key("amax"): np.array([0.5, 9.25, 1.0, 0.125], dtype=np.float32),
        key("rows"): np.arange(1, 9, dtype=np.float32).reshape(2, 4),
        key("rows_idx"): np.array([3, 11], dtype=np.int64),
        key("count"): np.array([42], dtype=np.int64),
    }

    emit(FIXTURES / "calib_container.safetensors", tensors, None)
    emit(
        FIXTURES / "calib_container_meta.safetensors",
        tensors,
        {"schema": "1", "arch": "krea2", "backend": "cpu"},
    )


if __name__ == "__main__":
    main()
