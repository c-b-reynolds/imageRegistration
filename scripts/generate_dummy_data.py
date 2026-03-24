"""
Generate dummy 2D numpy registration data for testing the pipeline.

Creates synthetic images containing random Gaussian blobs — simple enough
that a model can learn to register them, but non-trivial enough to be useful.

Usage:
    python scripts/generate_dummy_data.py
    python scripts/generate_dummy_data.py --size 64 --n-train 30 --n-val 6 --n-test 6
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def gaussian_blob(size: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def make_image(size: int, n_blobs: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_blobs):
        cx    = rng.uniform(size * 0.2, size * 0.8)
        cy    = rng.uniform(size * 0.2, size * 0.8)
        sigma = rng.uniform(size * 0.05, size * 0.15)
        amp   = rng.uniform(0.5, 1.0)
        img  += amp * gaussian_blob(size, cx, cy, sigma)
    img = np.clip(img, 0, 1)
    return img


def generate_split(out_dir: Path, n: int, size: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        n_blobs = rng.integers(2, 6)
        img = make_image(size, n_blobs, rng)
        np.save(out_dir / f"subject_{i:03d}.npy", img)
    print(f"  {out_dir}: {n} images ({size}x{size})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size",    type=int, default=64)
    p.add_argument("--n-train", type=int, default=30)
    p.add_argument("--n-val",   type=int, default=6)
    p.add_argument("--n-test",  type=int, default=6)
    p.add_argument("--root",    default="data")
    args = p.parse_args()

    root = Path(args.root)
    print(f"Generating dummy data in {root}/")

    generate_split(root / "train", args.n_train, args.size, seed=0)
    generate_split(root / "val",   args.n_val,   args.size, seed=1)
    generate_split(root / "test",  args.n_test,  args.size, seed=2)

    # Atlas = mean of all training images
    train_imgs = [np.load(p) for p in sorted((root / "train").glob("*.npy"))]
    atlas = np.mean(train_imgs, axis=0).astype(np.float32)
    np.save(root / "atlas.npy", atlas)
    print(f"  {root}/atlas.npy: mean of training images")
    print("Done.")


if __name__ == "__main__":
    main()
