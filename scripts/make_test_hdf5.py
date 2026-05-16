"""Generate synthetic HDF5 stacks for testing.

Creates several (T, H, W) uint16 datasets with moving Gaussian blobs and noise,
roughly simulating calcium imaging frames. The variants differ in frame count
and dimensions so multi-video sync can be tested with mismatched timelines.
"""

from pathlib import Path

import h5py
import numpy as np


def make_stack(t: int, h: int, w: int, blob_sigma: float = 20.0, seed: int = 0) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    stack = np.zeros((t, h, w), dtype=np.uint16)
    rng = np.random.default_rng(seed)
    for i in range(t):
        cx = w / 2 + (w / 4) * np.cos(2 * np.pi * i / t)
        cy = h / 2 + (h / 4) * np.sin(2 * np.pi * i / t)
        blob = 8000 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * blob_sigma**2))
        noise = rng.integers(0, 500, size=(h, w))
        stack[i] = np.clip(blob + noise, 0, 65535).astype(np.uint16)
    return stack


def write_stack(out_path: Path, t: int, h: int, w: int, seed: int) -> None:
    stack = make_stack(t, h, w, seed=seed)
    chunk_h = min(h, 256)
    chunk_w = min(w, 256)
    with h5py.File(out_path, "w") as f:
        f.create_dataset(
            "data",
            data=stack,
            chunks=(1, chunk_h, chunk_w),
            compression="gzip",
        )
    print(f"Wrote {out_path} — shape {stack.shape}, dtype {stack.dtype}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "test_data"
    out_dir.mkdir(exist_ok=True)

    variants = [
        ("synthetic_stack.h5", 200, 256, 256, 0),
        ("stack_small.h5", 100, 128, 128, 1),
        ("stack_tall.h5", 300, 320, 240, 2),
    ]
    for name, t, h, w, seed in variants:
        write_stack(out_dir / name, t, h, w, seed)


if __name__ == "__main__":
    main()
