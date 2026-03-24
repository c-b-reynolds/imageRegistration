"""
Evaluation metrics for image registration.

All functions:
    - Accept numpy arrays or torch tensors
    - Return float scalars or dicts
    - Are documented with the citation they implement

For paper reporting, use evaluate_dataset() which returns a DataFrame
ready for table generation, including bootstrapped confidence intervals.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import torch


Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# Intensity-based metrics
# ---------------------------------------------------------------------------

def ncc(pred: Array, target: Array) -> float:
    """
    Global Normalized Cross-Correlation in [-1, 1]. Higher is better.
    For paper reporting, local NCC (lNCC) over a window is often more informative.
    """
    p = _to_numpy(pred).ravel().astype(np.float64)
    t = _to_numpy(target).ravel().astype(np.float64)
    num = ((p - p.mean()) * (t - t.mean())).sum()
    den = np.sqrt(((p - p.mean()) ** 2).sum() * ((t - t.mean()) ** 2).sum())
    return float(num / (den + 1e-8))


def mse(pred: Array, target: Array) -> float:
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.mean((p - t) ** 2))


def psnr(pred: Array, target: Array, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher is better."""
    mse_val = mse(pred, target)
    if mse_val < 1e-10:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse_val))


def ssim(pred: Array, target: Array, data_range: float = 1.0, win: int = 11) -> float:
    """
    Structural Similarity Index (Wang et al., 2004). Range [0, 1], higher is better.
    Uses scikit-image for correctness.
    """
    from skimage.metrics import structural_similarity
    p = _to_numpy(pred).squeeze()
    t = _to_numpy(target).squeeze()
    return float(structural_similarity(p, t, data_range=data_range, win_size=win))


# ---------------------------------------------------------------------------
# Registration-specific metrics
# ---------------------------------------------------------------------------

def dice(pred_seg: Array, target_seg: Array, labels: Optional[List[int]] = None) -> Dict[str, float]:
    """
    Per-label and mean Dice Similarity Coefficient.

    Args:
        pred_seg:   Predicted segmentation (integer labels)
        target_seg: Target (fixed) segmentation
        labels:     Labels to evaluate; if None, uses all unique labels except 0 (background)

    Returns:
        Dict with per-label DSC and 'mean_dice'
    """
    p = _to_numpy(pred_seg).astype(int).ravel()
    t = _to_numpy(target_seg).astype(int).ravel()

    if labels is None:
        labels = [l for l in np.unique(t) if l != 0]

    scores: Dict[str, float] = {}
    for label in labels:
        pm = p == label
        tm = t == label
        inter = (pm & tm).sum()
        union = pm.sum() + tm.sum()
        scores[f"dice_{label}"] = float(2 * inter / (union + 1e-8))

    scores["mean_dice"] = float(np.mean(list(scores.values()))) if scores else 0.0
    return scores


def jacobian_determinant_stats(flow: Array) -> Dict[str, float]:
    """
    Compute the Jacobian determinant of a displacement field.

    For a diffeomorphic registration the Jacobian det should be > 0 everywhere.
    Negative values indicate folding (non-physical deformations).

    Args:
        flow: Displacement field of shape (ndim, *spatial) or (1, ndim, *spatial)

    Returns:
        Dict with keys: mean, std, min, max, pct_negative (fraction of folded voxels)
    """
    f = _to_numpy(flow)
    if f.ndim == 4 and f.shape[0] == 1:
        f = f[0]   # remove batch dim
    # f: (ndim, *spatial)

    ndim = f.shape[0]
    if ndim == 3:
        D, H, W = f.shape[1], f.shape[2], f.shape[3]
        # Finite differences for Jacobian columns
        dfdx = np.gradient(f[0], axis=0)
        dfdy = np.gradient(f[1], axis=1)
        dfdz = np.gradient(f[2], axis=2)

        # Jacobian of (id + flow):
        J00 = 1 + np.gradient(f[0], axis=0)
        J01 = np.gradient(f[0], axis=1)
        J02 = np.gradient(f[0], axis=2)
        J10 = np.gradient(f[1], axis=0)
        J11 = 1 + np.gradient(f[1], axis=1)
        J12 = np.gradient(f[1], axis=2)
        J20 = np.gradient(f[2], axis=0)
        J21 = np.gradient(f[2], axis=1)
        J22 = 1 + np.gradient(f[2], axis=2)

        det = (J00 * (J11 * J22 - J12 * J21)
               - J01 * (J10 * J22 - J12 * J20)
               + J02 * (J10 * J21 - J11 * J20))

    elif ndim == 2:
        J00 = 1 + np.gradient(f[0], axis=0)
        J01 = np.gradient(f[0], axis=1)
        J10 = np.gradient(f[1], axis=0)
        J11 = 1 + np.gradient(f[1], axis=1)
        det = J00 * J11 - J01 * J10
    else:
        raise ValueError(f"Expected 2D or 3D flow, got {ndim}D")

    return {
        "jac_mean":     float(det.mean()),
        "jac_std":      float(det.std()),
        "jac_min":      float(det.min()),
        "jac_max":      float(det.max()),
        "jac_pct_neg":  float((det < 0).mean()),  # fraction of folded voxels
    }


# ---------------------------------------------------------------------------
# Full evaluation of one sample
# ---------------------------------------------------------------------------

def evaluate_sample(
    warped: Array,
    fixed: Array,
    flow: Array,
    warped_seg: Optional[Array] = None,
    fixed_seg: Optional[Array] = None,
    seg_labels: Optional[List[int]] = None,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute all requested metrics for one registration result.

    Args:
        warped:     Warped moving image
        fixed:      Fixed (target) image
        flow:       Displacement field (ndim, *spatial)
        warped_seg: Warped moving segmentation (optional)
        fixed_seg:  Fixed segmentation (optional)
        seg_labels: Segmentation labels for Dice
        metrics:    List of metric names to compute (default: all available)

    Returns:
        Flat dict of scalar metric values
    """
    if metrics is None:
        metrics = ["ncc", "ssim", "psnr", "jacobian_det"]

    results: Dict[str, float] = {}

    if "ncc" in metrics:
        results["ncc"] = ncc(warped, fixed)
    if "mse" in metrics:
        results["mse"] = mse(warped, fixed)
    if "ssim" in metrics:
        results["ssim"] = ssim(warped, fixed)
    if "psnr" in metrics:
        results["psnr"] = psnr(warped, fixed)
    if "jacobian_det" in metrics:
        results.update(jacobian_determinant_stats(flow))
    if "dice" in metrics and warped_seg is not None and fixed_seg is not None:
        results.update(dice(warped_seg, fixed_seg, labels=seg_labels))

    return results


# ---------------------------------------------------------------------------
# Dataset-level evaluation with bootstrapped confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray, n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    """
    Bootstrapped 95% confidence interval for the mean.

    Returns: {'mean', 'ci_lo', 'ci_hi', 'std'}
    """
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "std": float(values.std()),
            "ci_lo": float(lo), "ci_hi": float(hi)}


def evaluate_dataset(
    results_list: List[Dict[str, float]],
    bootstrap_n: int = 1000,
    alpha: float = 0.05,
) -> "pd.DataFrame":
    """
    Aggregate per-sample metric dicts into a summary DataFrame.

    Args:
        results_list: Output of [evaluate_sample(...) for each test sample]
        bootstrap_n:  Bootstrap resampling iterations
        alpha:        Significance level for CI

    Returns:
        pd.DataFrame with columns: metric, mean, std, ci_lo, ci_hi
        Ready to export as LaTeX table via df.to_latex()
    """
    import pandas as pd

    all_keys = set().union(*[r.keys() for r in results_list])
    rows = []
    for key in sorted(all_keys):
        vals = np.array([r[key] for r in results_list if key in r])
        stats = bootstrap_ci(vals, n=bootstrap_n, alpha=alpha)
        rows.append({"metric": key, **stats})

    return pd.DataFrame(rows).set_index("metric")
