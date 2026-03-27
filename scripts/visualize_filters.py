"""
Visualize learned filter pairs from conv_f and conv_g in a trained GateFlow model.

The gate folds the first C and last C channels of the element-wise product:
    prod = conv_f(f) * conv_g(g)
    gate = prod[:, :C] + prod[:, C:]

For channel index c the four correlated filters are:
    conv_f[c]    conv_g[c]       ← top row of each cell
    conv_f[c+C]  conv_g[c+C]    ← bottom row of each cell

For ISA-style filters these quads should look like Gabor quadrature pairs:
same orientation/frequency but 90° phase shift between c and c+C,
with conv_f[c] ≈ conv_g[c] (both networks learn matching detectors).

Displays 25 quads arranged as a 5×5 grid of 2×2 cells (10×10 subplots total).
Each filter is shown with its own diverging colormap centred at zero.

Usage:
    python scripts/visualize_filters.py \\
        --checkpoint outputs/gate_flow_synthetic/checkpoints/best.pt
    python scripts/visualize_filters.py \\
        --checkpoint <path> --n-pairs 25 --out figures/filters.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.models import build_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalized_weights(conv) -> np.ndarray:
    """
    Return the L2-normalised filter weights actually used during the forward
    pass of an L2NormConv2d layer.

    Returns (out_ch, k, k) float32 numpy array.
    """
    w      = conv.weight.detach().cpu()          # (out_ch, 1, k, k)
    w_flat = w.view(w.shape[0], -1)              # (out_ch, k*k)
    w_norm = F.normalize(w_flat, p=2, dim=1)     # unit-norm per filter
    return w_norm.view_as(w).squeeze(1).numpy()  # (out_ch, k, k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a GateFlow checkpoint (.pt)")
    parser.add_argument("--n-pairs", type=int, default=25,
                        help="Number of filter quads to show (capped at C)")
    parser.add_argument("--out", default=None,
                        help="Save path (PNG/PDF); if omitted, display interactively")
    args = parser.parse_args()

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt     = Path(args.checkpoint)
    cfg_path = ckpt.parent.parent / "config.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    payload      = torch.load(ckpt, map_location="cpu", weights_only=False)
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model        = build_model(cfg["model"]["name"], **model_kwargs)
    model.load_state_dict(payload["model_state"])
    model.eval()

    vn = model.velocity_net
    C  = vn.C                            # half the output channels
    wf = _normalized_weights(vn.conv_f)  # (2C, k, k)
    wg = _normalized_weights(vn.conv_g)  # (2C, k, k)
    k  = wf.shape[-1]

    n_pairs  = min(args.n_pairs, C)
    n_cols_g = int(np.ceil(np.sqrt(n_pairs)))   # number of quad-columns
    n_rows_g = int(np.ceil(n_pairs / n_cols_g)) # number of quad-rows

    # Each quad occupies a 2×2 block of subplots
    fig_cols = n_cols_g * 2
    fig_rows = n_rows_g * 2

    cell_size = max(1.0, k / 6.0)   # scale subplot size to filter size
    fig, axes = plt.subplots(
        fig_rows, fig_cols,
        figsize=(fig_cols * cell_size * 1.1, fig_rows * cell_size * 1.1),
        gridspec_kw={"hspace": 0.12, "wspace": 0.08},
    )

    # ── Plot each quad ───────────────────────────────────────────────────────
    for idx in range(n_pairs):
        c      = idx
        grow   = idx // n_cols_g
        gcol   = idx % n_cols_g
        r0, c0 = grow * 2, gcol * 2

        # (filter_array, subplot_row_offset, subplot_col_offset, label)
        quads = [
            (wf[c],     0, 0, f"f[{c}]"),
            (wg[c],     0, 1, f"g[{c}]"),
            (wf[c + C], 1, 0, f"f[{c+C}]"),
            (wg[c + C], 1, 1, f"g[{c+C}]"),
        ]

        for filt, dr, dc, label in quads:
            ax   = axes[r0 + dr, c0 + dc]
            vmax = float(np.abs(filt).max()) + 1e-8
            ax.imshow(filt, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest", aspect="equal")
            ax.set_title(label, fontsize=5, pad=1)
            ax.axis("off")

    # ── Hide unused cells ────────────────────────────────────────────────────
    for idx in range(n_pairs, n_rows_g * n_cols_g):
        grow, gcol = idx // n_cols_g, idx % n_cols_g
        r0, c0     = grow * 2, gcol * 2
        for dr in range(2):
            for dc in range(2):
                axes[r0 + dr, c0 + dc].axis("off")

    # ── Draw boxes around each quad to make pairing obvious ──────────────────
    for idx in range(n_pairs):
        grow   = idx // n_cols_g
        gcol   = idx % n_cols_g
        r0, c0 = grow * 2, gcol * 2
        # Use the top-left subplot's position to draw a rectangle in figure coords
        ax_tl = axes[r0, c0]
        ax_br = axes[r0 + 1, c0 + 1]
        bbox_tl = ax_tl.get_position()
        bbox_br = ax_br.get_position()
        pad = 0.003
        rect = mpatches.FancyBboxPatch(
            (bbox_tl.x0 - pad, bbox_br.y0 - pad),
            (bbox_br.x1 - bbox_tl.x0) + 2 * pad,
            (bbox_tl.y1 - bbox_br.y0) + 2 * pad,
            transform=fig.transFigure,
            boxstyle="square,pad=0",
            linewidth=0.6, edgecolor="#888888", facecolor="none",
            clip_on=False,
        )
        fig.add_artist(rect)

    fig.suptitle(
        f"GateFlow filter quads  —  "
        f"top: (conv_f[c], conv_g[c])   bottom: (conv_f[c+C], conv_g[c+C])   "
        f"C={C}  k={k}×{k}",
        fontsize=8, y=1.005,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
