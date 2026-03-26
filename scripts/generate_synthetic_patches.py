"""
Generate synthetic patch pairs for registration baseline testing.

Each pair consists of:
  - moving: image containing random geometric shapes (circles, rectangles,
            ellipses) with hard edges and clear structure
  - fixed:  a smooth non-rigid deformation of moving

Output: patches.npy of shape (N, 2, patch_size, patch_size) float32
        compatible with train_hybrid_patches.py

Usage:
    python scripts/generate_synthetic_patches.py
    python scripts/generate_synthetic_patches.py --n 1000 --size 128 --out dataset/synthetic_128.npy
    python scripts/generate_synthetic_patches.py --max-disp 12 --smooth 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


# ---------------------------------------------------------------------------
# Shape rendering
# ---------------------------------------------------------------------------

def _draw_circle(img, cx, cy, r, intensity):
    H, W = img.shape
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    img[mask] = np.maximum(img[mask], intensity)


def _draw_ellipse(img, cx, cy, rx, ry, angle, intensity):
    H, W = img.shape
    y, x = np.ogrid[:H, :W]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = (x - cx) * cos_a + (y - cy) * sin_a
    dy = -(x - cx) * sin_a + (y - cy) * cos_a
    mask = (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0
    img[mask] = np.maximum(img[mask], intensity)


def _draw_rectangle(img, x0, y0, x1, y1, intensity):
    x0, x1 = max(0, x0), min(img.shape[1], x1)
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    img[y0:y1, x0:x1] = np.maximum(img[y0:y1, x0:x1], intensity)


def _draw_triangle(img, pts, intensity):
    """Rasterize a filled triangle given 3 (x, y) vertices."""
    H, W = img.shape
    y, x = np.mgrid[:H, :W]
    # Barycentric coordinate test
    (x0, y0), (x1, y1), (x2, y2) = pts
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1:
        return
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    c = 1.0 - a - b
    mask = (a >= 0) & (b >= 0) & (c >= 0)
    img[mask] = np.maximum(img[mask], intensity)


def make_shape_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Render 4-8 random geometric shapes onto a black background.
    Returns float32 in [0, 1].
    """
    img = np.zeros((size, size), dtype=np.float32)
    n_shapes = int(rng.integers(4, 9))

    for _ in range(n_shapes):
        kind      = rng.choice(["circle", "ellipse", "rectangle", "triangle"])
        intensity = float(rng.uniform(0.4, 1.0))
        margin    = size // 8

        if kind == "circle":
            cx = int(rng.integers(margin, size - margin))
            cy = int(rng.integers(margin, size - margin))
            r  = int(rng.integers(size // 12, size // 5))
            _draw_circle(img, cx, cy, r, intensity)

        elif kind == "ellipse":
            cx    = int(rng.integers(margin, size - margin))
            cy    = int(rng.integers(margin, size - margin))
            rx    = int(rng.integers(size // 12, size // 4))
            ry    = int(rng.integers(size // 12, size // 4))
            angle = float(rng.uniform(0, np.pi))
            _draw_ellipse(img, cx, cy, rx, ry, angle, intensity)

        elif kind == "rectangle":
            x0 = int(rng.integers(0, size - size // 4))
            y0 = int(rng.integers(0, size - size // 4))
            x1 = x0 + int(rng.integers(size // 8, size // 3))
            y1 = y0 + int(rng.integers(size // 8, size // 3))
            _draw_rectangle(img, x0, y0, x1, y1, intensity)

        elif kind == "triangle":
            pts = [
                (int(rng.integers(margin, size - margin)),
                 int(rng.integers(margin, size - margin)))
                for _ in range(3)
            ]
            _draw_triangle(img, pts, intensity)

    return np.clip(img, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Deformation fields
# ---------------------------------------------------------------------------

def make_nonrigid_deformation(size: int, rng: np.random.Generator,
                               max_disp: float, smooth_sigma: float):
    """
    Smooth non-rigid displacement field (dx, dy) in pixels.
    Random Gaussian noise smoothed to produce low-frequency spatial warps.
    """
    raw_x = rng.standard_normal((size, size)).astype(np.float32)
    raw_y = rng.standard_normal((size, size)).astype(np.float32)

    dx = gaussian_filter(raw_x, sigma=smooth_sigma)
    dy = gaussian_filter(raw_y, sigma=smooth_sigma)

    peak = max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
    dx = (dx / peak * max_disp).astype(np.float32)
    dy = (dy / peak * max_disp).astype(np.float32)

    return dx, dy


def make_rigid_displacement(size: int, rng: np.random.Generator,
                             max_translation: float, max_shear: float):
    """
    Rigid-like displacement field from translation + shear (no rotation).

    Translation: uniform shift (tx, ty) in pixels.
    Shear:       x' = x + shear_x * y
                 y' = y + shear_y * x
    Both are small — intended to complement the non-rigid component.
    """
    tx = float(rng.uniform(-max_translation, max_translation))
    ty = float(rng.uniform(-max_translation, max_translation))
    shear_x = float(rng.uniform(-max_shear, max_shear))
    shear_y = float(rng.uniform(-max_shear, max_shear))

    y, x = np.mgrid[0:size, 0:size]

    dx = (tx + shear_x * y).astype(np.float32)
    dy = (ty + shear_y * x).astype(np.float32)

    return dx, dy


def make_deformation(size: int, rng: np.random.Generator,
                     max_disp: float, smooth_sigma: float,
                     max_translation: float, max_shear: float):
    """
    Combined displacement field: non-rigid + translation + shear.
    """
    dx_nr, dy_nr = make_nonrigid_deformation(size, rng, max_disp, smooth_sigma)
    dx_r,  dy_r  = make_rigid_displacement(size, rng, max_translation, max_shear)
    return dx_nr + dx_r, dy_nr + dy_r


def apply_deformation(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    Warp image by displacement field using bilinear interpolation.

    Convention: fixed(r) = moving(r + displacement(r))
    i.e. the displacement points from fixed coords back to moving coords.
    """
    size = img.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    src_x = (x + dx).ravel()
    src_y = (y + dy).ravel()
    warped = map_coordinates(img, [src_y, src_x], order=1,
                             mode="nearest", prefilter=False)
    return warped.reshape(size, size).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_patches(
    path: Path,
    n: int = 800,
    size: int = 64,
    max_disp: float = 8.0,
    smooth_sigma: float = None,
    max_translation: float = 3.0,
    max_shear: float = 0.05,
    seed: int = 42,
):
    """
    Generate N synthetic registration patch pairs.

    Args:
        path:            output .npy path
        n:               number of patch pairs
        size:            spatial size of each patch (square)
        max_disp:        peak non-rigid displacement in pixels
        smooth_sigma:    smoothing sigma for non-rigid field (default: size/4)
        max_translation: maximum translation in pixels (applied uniformly)
        max_shear:       maximum shear coefficient (dimensionless, pixels/pixel)
                         e.g. 0.05 shifts edge pixels by 0.05 * size = 3.2px
        seed:            random seed
    """
    if smooth_sigma is None:
        smooth_sigma = size / 4.0

    print(f"Generating {n} synthetic patch pairs "
          f"({size}x{size}, max_disp={max_disp:.1f}px, "
          f"translation=±{max_translation:.1f}px, "
          f"shear=±{max_shear:.3f}) -> {path}")

    rng     = np.random.default_rng(seed)
    patches = np.zeros((n, 2, size, size), dtype=np.float32)

    for i in range(n):
        moving        = make_shape_image(size, rng)
        dx, dy        = make_deformation(size, rng, max_disp, smooth_sigma,
                                         max_translation, max_shear)
        fixed         = apply_deformation(moving, dx, dy)
        patches[i, 0] = moving
        patches[i, 1] = fixed

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n}")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, patches)
    print(f"Saved {patches.shape} float32 array to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--out",       default="dataset/synthetic_patches.npy",
                        help="Output .npy path")
    parser.add_argument("--n",         type=int,   default=800,
                        help="Number of patch pairs")
    parser.add_argument("--size",      type=int,   default=64,
                        help="Patch size (square)")
    parser.add_argument("--max-disp",  type=float, default=8.0,
                        help="Maximum displacement in pixels")
    parser.add_argument("--smooth",      type=float, default=None,
                        help="Deformation smoothing sigma (default: size/4)")
    parser.add_argument("--translation", type=float, default=3.0,
                        help="Max translation in pixels")
    parser.add_argument("--shear",       type=float, default=0.05,
                        help="Max shear coefficient (pixels/pixel)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    generate_synthetic_patches(
        path            = Path(args.out),
        n               = args.n,
        size            = args.size,
        max_disp        = args.max_disp,
        smooth_sigma    = args.smooth,
        max_translation = args.translation,
        max_shear       = args.shear,
        seed            = args.seed,
    )


if __name__ == "__main__":
    main()
