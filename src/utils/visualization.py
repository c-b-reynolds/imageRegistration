"""
Visualization utilities for paper figures.

Conventions:
    - All plot functions return (fig, axes) so the caller can save/modify.
    - Use save_fig() for consistent DPI / format handling.
    - Prefer vector formats (PDF, SVG) for paper submission.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def set_paper_style(font_size: int = 10) -> None:
    """Apply a clean, journal-appropriate matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
    })


def save_fig(fig: plt.Figure, path: str | Path, formats: Tuple[str, ...] = ("pdf", "png")) -> None:
    """Save a figure in multiple formats (default: PDF for paper + PNG for preview)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(p.with_suffix(f".{fmt}"), bbox_inches="tight")


# ---------------------------------------------------------------------------
# Registration-specific plots
# ---------------------------------------------------------------------------

def plot_registration_result(
    moving: np.ndarray,
    fixed: np.ndarray,
    warped: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    cmap: str = "gray",
    figsize: Tuple[float, float] = (10, 3),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Side-by-side: moving | fixed | warped | difference map.

    Args:
        moving/fixed/warped: 2D or 3D arrays (C, H, W) or (C, D, H, W)
        slice_idx: for 3D, which slice to show (default: middle)
        axis: slicing axis for 3D volumes (0=axial, 1=coronal, 2=sagittal)
    """
    def _slice(arr):
        arr = arr.squeeze()
        if arr.ndim == 3:
            mid = arr.shape[axis] // 2 if slice_idx is None else slice_idx
            arr = np.take(arr, mid, axis=axis)
        return arr

    m, f, w = _slice(moving), _slice(fixed), _slice(warped)
    diff = np.abs(w - f)

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for ax, img, title in zip(
        axes, [m, f, w, diff], ["Moving", "Fixed", "Warped", "|Warped - Fixed|"]
    ):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_deformation_grid(
    flow: np.ndarray,
    step: int = 10,
    slice_idx: Optional[int] = None,
    axis: int = 0,
    figsize: Tuple[float, float] = (5, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualize a displacement field as a deformed grid (2D or 3D slice).

    Args:
        flow: (ndim, *spatial) displacement field
        step: grid line spacing in pixels
    """
    if flow.ndim == 4:  # (ndim, D, H, W) -> take slice
        mid = flow.shape[1] // 2 if slice_idx is None else slice_idx
        flow = np.take(flow, mid, axis=1)
    # Now flow: (2, H, W)
    _, H, W = flow.shape

    fig, ax = plt.subplots(figsize=figsize)
    for y in range(0, H, step):
        gx = np.arange(W) + flow[1, y, :]
        gy = np.full(W, y) + flow[0, y, :]
        ax.plot(gx, gy, "b-", linewidth=0.5, alpha=0.7)
    for x in range(0, W, step):
        gx = np.full(H, x) + flow[1, :, x]
        gy = np.arange(H) + flow[0, :, x]
        ax.plot(gx, gy, "b-", linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Deformation Grid")
    fig.tight_layout()
    return fig, ax


def plot_learning_curves(
    history: dict,
    keys: Tuple[str, ...] = ("train_loss", "val_loss"),
    labels: Optional[Tuple[str, ...]] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot training/validation loss curves."""
    fig, ax = plt.subplots(figsize=figsize)
    if labels is None:
        labels = keys
    for key, label in zip(keys, labels):
        if key in history:
            ax.plot(history[key], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_metric_boxplot(
    data: dict,
    metric: str = "dice",
    method_order: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 4),
    palette: str = "Set2",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Box plot comparing a metric across methods.

    Args:
        data: {'Method A': [val1, val2, ...], 'Method B': [...]}
        metric: Metric name (for axis label)
        method_order: Display order (default: sorted)
    """
    import seaborn as sns

    methods = method_order or sorted(data)
    values = [data[m] for m in methods]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(values, labels=methods, patch_artist=True, notch=False)

    colors = sns.color_palette(palette, n_colors=len(methods))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel(metric)
    ax.set_xlabel("Method")
    fig.tight_layout()
    return fig, ax


def plot_jacobian_distribution(
    jac_dets: Sequence[np.ndarray],
    labels: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Histogram of Jacobian determinant values — good indicator of diffeomorphism quality."""
    fig, ax = plt.subplots(figsize=figsize)
    for i, jd in enumerate(jac_dets):
        label = labels[i] if labels else f"Method {i+1}"
        ax.hist(jd.ravel(), bins=100, alpha=0.6, label=label, density=True)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.0, label="J=0 (folding)")
    ax.set_xlabel("Jacobian Determinant")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    return fig, ax
