"""
Data augmentation transforms for image registration.

All transforms accept and return a sample dict with 'moving' and 'fixed' tensors.
Spatial augmentations are applied consistently to both images (and their segmentations).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomFlip:
    """Randomly flip both images along one or more spatial axes."""

    def __init__(self, axes: Tuple[int, ...] = (0, 1, 2), p: float = 0.5):
        self.axes = axes
        self.p = p

    def __call__(self, sample: dict) -> dict:
        for ax in self.axes:
            if torch.rand(1).item() < self.p:
                dim = ax + 2  # skip batch + channel dims
                for key in ("moving", "fixed", "moving_seg", "fixed_seg"):
                    if key in sample:
                        sample[key] = sample[key].flip(dim)
        return sample


class RandomIntensityShift:
    """Independently shift and scale intensity of moving/fixed images."""

    def __init__(self, shift: float = 0.1, scale: float = 0.1):
        self.shift = shift
        self.scale = scale

    def __call__(self, sample: dict) -> dict:
        for key in ("moving", "fixed"):
            if key in sample:
                s = 1.0 + (torch.rand(1).item() * 2 - 1) * self.scale
                b = (torch.rand(1).item() * 2 - 1) * self.shift
                sample[key] = (sample[key] * s + b).clamp(0.0, 1.0)
        return sample


class RandomCrop:
    """Randomly crop a fixed spatial size from both images."""

    def __init__(self, size: Tuple[int, ...]):
        self.size = size

    def __call__(self, sample: dict) -> dict:
        spatial = sample["moving"].shape[1:]  # (D, H, W) or (H, W)
        starts = [
            torch.randint(0, max(1, s - c), (1,)).item()
            for s, c in zip(spatial, self.size)
        ]
        slices = tuple(
            slice(int(st), int(st) + c) for st, c in zip(starts, self.size)
        )
        full_slice = (slice(None),) + slices  # include channel dim

        for key in ("moving", "fixed", "moving_seg", "fixed_seg"):
            if key in sample:
                sample[key] = sample[key][full_slice]
        return sample


class NormalizeIntensity:
    """Per-sample min-max normalization."""

    def __call__(self, sample: dict) -> dict:
        for key in ("moving", "fixed"):
            if key in sample:
                t = sample[key]
                lo, hi = t.min(), t.max()
                sample[key] = (t - lo) / (hi - lo + 1e-8)
        return sample


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_transforms(cfg: dict):
    """Return (train_transform, val_transform) based on config."""
    if cfg["data"].get("augmentation", True):
        train_tf = Compose([
            RandomFlip(axes=(0, 1, 2), p=0.5),
            RandomIntensityShift(shift=0.05, scale=0.05),
        ])
    else:
        train_tf = None

    val_tf = None  # no augmentation at val/test time
    return train_tf, val_tf
