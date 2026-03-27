"""
Visualize patch pairs from a patches.npy file.

Shows moving | fixed | |moving - fixed| for N random samples.

Usage:
    python scripts/inspect_patches.py --patches dataset/synthetic_patches.npy
    python scripts/inspect_patches.py --patches dataset/synthetic_patches.npy --n 16 --out figures/inspect.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--patches", required=True, help="Path to patches.npy")
    parser.add_argument("--n",       type=int, default=8,
                        help="Number of pairs to show")
    parser.add_argument("--out",     default=None,
                        help="Save figure to this path instead of displaying")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    patches = np.load(args.patches, mmap_mode="r")
    N = len(patches)
    print(f"Loaded {patches.shape}  dtype={patches.dtype}  "
          f"min={patches.min():.3f}  max={patches.max():.3f}")

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(N, size=min(args.n, N), replace=False)

    fig, axes = plt.subplots(len(indices), 3,
                             figsize=(7, 2.5 * len(indices)),
                             squeeze=False)

    for row, idx in enumerate(indices):
        moving = patches[idx, 0]
        fixed  = patches[idx, 1]
        diff   = np.abs(moving - fixed)

        for ax, img, title in zip(
            axes[row],
            [moving, fixed, diff],
            ["Moving", "Fixed", "|Moving - Fixed|"],
        ):
            vmax = 1.0 if title != "|Moving - Fixed|" else diff.max() + 1e-6
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            if row == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

        axes[row, 0].set_ylabel(f"#{idx}", fontsize=8, rotation=0,
                                labelpad=28, va="center")

    fig.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
