"""
Dataset for SAR patch pairs produced by sar_align/make_dataset.py.

patches.npy layout: (N, 2, H, W)  float32  values in [0, 1]
    channel 0 = image A (moving)
    channel 1 = image B (fixed)

Transforms are applied only to the training split. Val and test are returned
as-is so evaluation metrics are comparable across runs.
"""

import math
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class RandomRotation90:
    """
    Rotate both images in a pair by the same random multiple of 90 degrees
    (0, 90, 180, or 270). The same rotation is applied to both so the
    registration relationship between moving and fixed is preserved.
    Square corners are never void after rotation.
    """

    def __call__(self, sample: dict) -> dict:
        k = random.randint(0, 3)
        sample["moving"] = torch.rot90(sample["moving"], k, dims=[-2, -1])
        sample["fixed"]  = torch.rot90(sample["fixed"],  k, dims=[-2, -1])
        return sample


class SARAwareRandomErasing:
    """
    Erases a random rectangular region from an image and fills it with
    Gaussian noise whose mean and std match the nonzero pixels of that
    image, so the fill is statistically consistent with the surrounding
    SAR content rather than being uniform random noise.

    Applied independently to each image in the pair so the erased regions
    differ between moving and fixed, simulating independent occlusion or
    interference in each SAR acquisition.

    Args:
        scale:        (min, max) fraction of image area to erase
        ratio:        (min, max) aspect ratio of erased rectangle
        max_attempts: number of tries to find a valid rectangle before
                      returning the image unchanged
    """

    def __init__(
        self,
        scale: tuple = (0.02, 0.2),
        ratio: tuple = (0.3, 3.0),
        max_attempts: int = 10,
    ):
        self.scale        = scale
        self.ratio        = ratio
        self.max_attempts = max_attempts

    def _erase(self, img: torch.Tensor) -> torch.Tensor:
        """Apply erasing to a single (1, H, W) tensor."""
        _, H, W = img.shape
        area = H * W

        for _ in range(self.max_attempts):
            erase_area = random.uniform(*self.scale) * area
            log_ratio  = random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect     = math.exp(log_ratio)

            h = int(round(math.sqrt(erase_area / aspect)))
            w = int(round(math.sqrt(erase_area * aspect)))
            h = min(h, H)
            w = min(w, W)
            if h == 0 or w == 0:
                continue

            i = random.randint(0, H - h)
            j = random.randint(0, W - w)

            # Statistics of nonzero (scene content) pixels
            nonzero = img[img != 0.0]
            if nonzero.numel() == 0:
                return img
            fill_mean = nonzero.mean().item()
            fill_std  = nonzero.std().item() if nonzero.numel() > 1 else 0.0

            noise = torch.randn(1, h, w, dtype=img.dtype, device=img.device)
            noise = (noise * fill_std + fill_mean).clamp(0.0, 1.0)

            out = img.clone()
            out[:, i:i + h, j:j + w] = noise
            return out

        return img  # no valid region found — return unchanged

    def __call__(self, sample: dict) -> dict:
        sample["moving"] = self._erase(sample["moving"])
        sample["fixed"]  = self._erase(sample["fixed"])
        return sample


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchPairDataset(Dataset):
    """
    Wraps a slice of a patches array as a registration dataset.

    Each item returns:
        {"moving": (1, H, W) float32, "fixed": (1, H, W) float32}

    Args:
        patches:   (N, 2, H, W) float32 array (numpy or mmap)
        transform: optional callable applied to each sample dict at load time
    """

    def __init__(self, patches: np.ndarray, transform=None):
        self.patches   = patches
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        pair = self.patches[idx]                              # (2, H, W)
        sample = {
            "moving": torch.from_numpy(pair[0].copy()).unsqueeze(0),  # (1, H, W)
            "fixed":  torch.from_numpy(pair[1].copy()).unsqueeze(0),  # (1, H, W)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_patch_dataloaders(cfg: dict):
    """
    Split patches.npy into train/val/test DataLoaders.

    Val and test sets receive no transforms so metrics are comparable.
    Transforms are applied only during training.

    Config keys
    -----------
    data:
        patches_path   : path to patches.npy
        val_frac       : fraction for validation  (default 0.15)
        test_frac      : fraction for test        (default 0.15)
        seed           : shuffle seed             (default 42)
        num_workers    : DataLoader workers       (default 0)
    training:
        batch_size     : batch size
    augmentation:
        use_rotation   : apply random 90° rotations  (default True)
        use_erasing    : apply SAR-aware random erase (default False)
        erasing_scale  : [min, max] fraction of area (default [0.02, 0.2])
        erasing_ratio  : [min, max] aspect ratio     (default [0.3, 3.0])
    """
    patches_path = cfg["data"]["patches_path"]
    val_frac     = cfg["data"].get("val_frac",    0.15)
    test_frac    = cfg["data"].get("test_frac",   0.15)
    seed         = cfg["data"].get("seed",        42)
    batch_size   = cfg["training"]["batch_size"]
    num_workers  = cfg["data"].get("num_workers", 0)

    aug          = cfg.get("augmentation", {})
    use_rotation = aug.get("use_rotation", True)
    use_erasing  = aug.get("use_erasing",  False)
    erasing_scale = tuple(aug.get("erasing_scale", [0.02, 0.2]))
    erasing_ratio = tuple(aug.get("erasing_ratio", [0.3,  3.0]))

    # Memory-map the file — avoids loading all patches into RAM upfront
    patches = np.load(patches_path, mmap_mode="r")
    N       = len(patches)

    # Reproducible shuffle then split
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(N)

    n_test  = int(N * test_frac)
    n_val   = int(N * val_frac)
    n_train = N - n_val - n_test

    train_patches = patches[indices[:n_train]]
    val_patches   = patches[indices[n_train:n_train + n_val]]
    test_patches  = patches[indices[n_train + n_val:]]

    # Build training transforms
    train_transforms = []
    if use_rotation:
        train_transforms.append(RandomRotation90())
    if use_erasing:
        train_transforms.append(SARAwareRandomErasing(
            scale=erasing_scale, ratio=erasing_ratio,
        ))
    train_transform = Compose(train_transforms) if train_transforms else None

    train_ds = PatchPairDataset(train_patches, transform=train_transform)
    val_ds   = PatchPairDataset(val_patches)
    test_ds  = PatchPairDataset(test_patches)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )
