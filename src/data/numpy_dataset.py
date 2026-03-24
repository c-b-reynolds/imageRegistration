"""
Dataset that loads 2D grayscale images stored as .npy files.

Expected layout:
    data/
        train/  *.npy
        val/    *.npy
        test/   *.npy
        atlas.npy   (fixed image — used by NumpyAtlasDataset)

Each .npy file should be a 2D float array of shape (H, W).
Values can be any range — they are normalized to [0, 1] on load.
"""

from pathlib import Path
from typing import Callable, List, Optional
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _load_normalize(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0)   # (1, H, W)


class NumpyPairDataset(Dataset):
    """
    Randomly pairs .npy files for registration.
    Each call to __getitem__ returns a different (moving, fixed) pair.

    Args:
        root:      Directory containing .npy files.
        num_pairs: Virtual dataset length (pairs drawn deterministically per index).
        transform: Optional callable applied to the sample dict.
    """

    def __init__(self, root: str, num_pairs: int = 500,
                 transform: Optional[Callable] = None):
        self.paths: List[Path] = sorted(Path(root).glob("*.npy"))
        if len(self.paths) < 2:
            raise FileNotFoundError(f"Need at least 2 .npy files in {root}, found {len(self.paths)}")
        self.num_pairs = num_pairs
        self.transform = transform

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(idx)
        m_idx, f_idx = rng.sample(range(len(self.paths)), k=2)
        sample = {
            "moving": _load_normalize(self.paths[m_idx]),
            "fixed":  _load_normalize(self.paths[f_idx]),
            "moving_path": str(self.paths[m_idx]),
            "fixed_path":  str(self.paths[f_idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


class NumpyAtlasDataset(Dataset):
    """
    Atlas-based: one fixed atlas .npy, every subject is a moving image.

    Args:
        root:        Directory of subject .npy files.
        atlas_path:  Path to the fixed atlas .npy file.
        transform:   Optional callable applied to the sample dict.
    """

    def __init__(self, root: str, atlas_path: str,
                 transform: Optional[Callable] = None):
        self.paths: List[Path] = sorted(Path(root).glob("*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"No .npy files found in {root}")
        self.fixed  = _load_normalize(Path(atlas_path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        sample = {
            "moving": _load_normalize(self.paths[idx]),
            "fixed":  self.fixed.clone(),
            "moving_path": str(self.paths[idx]),
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def build_numpy_dataloaders(cfg: dict):
    """Build train/val/test DataLoaders for numpy data. Returns (train, val, test)."""
    root        = cfg["data"]["root"]
    atlas_path  = f"{root}/atlas.npy"
    batch_size  = cfg["training"]["batch_size"]
    num_workers = cfg["data"].get("num_workers", 0)

    train_ds = NumpyPairDataset(f"{root}/train", num_pairs=cfg["data"].get("num_pairs", 500))
    val_ds   = NumpyAtlasDataset(f"{root}/val",  atlas_path)
    test_ds  = NumpyAtlasDataset(f"{root}/test", atlas_path)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )
