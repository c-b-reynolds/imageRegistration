"""
Dataset classes for image registration.

Expected directory layout (adjust to your data):
    data/
        train/
            subject_001/
                image.nii.gz
                seg.nii.gz      (optional — segmentation for Dice evaluation)
            subject_002/
                ...
        val/
        test/

The RegistrationDataset yields (moving, fixed) pairs. For intra-subject
registration use PairDataset; for atlas-based use AtlasDataset.
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    """Load a medical image to a float32 numpy array."""
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
    except ImportError:
        raise ImportError(
            "SimpleITK is required to load medical images. "
            "Install with: pip install SimpleITK"
        )
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseRegistrationDataset(Dataset):
    """
    Abstract base — subclass and implement __len__ and __getitem__ to return
    dicts with at minimum: 'moving' and 'fixed' (torch.Tensor, float32).
    Optional keys: 'moving_seg', 'fixed_seg', 'moving_path', 'fixed_path'.
    """

    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform

    def _apply_transform(self, sample: dict) -> dict:
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ---------------------------------------------------------------------------
# Concrete datasets
# ---------------------------------------------------------------------------

class PairDataset(BaseRegistrationDataset):
    """
    Randomly pairs subjects from a split directory for deformable registration.
    Each sample yields a unique (moving, fixed) pair.

    Args:
        root:        Path to split directory (e.g., 'data/train')
        image_glob:  Glob pattern for image files within each subject folder
        seg_glob:    Glob pattern for segmentation (optional)
        normalize:   Apply min-max normalization
        transform:   Callable applied to the sample dict
        num_pairs:   Virtual dataset size (pairs are drawn randomly)
    """

    def __init__(
        self,
        root: str,
        image_glob: str = "image.nii.gz",
        seg_glob: Optional[str] = "seg.nii.gz",
        normalize: bool = True,
        transform: Optional[Callable] = None,
        num_pairs: int = 1000,
    ):
        super().__init__(transform)
        self.root = Path(root)
        self.normalize = normalize
        self.num_pairs = num_pairs

        self.image_paths: List[Path] = sorted(self.root.glob(f"**/{image_glob}"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found matching '{image_glob}' under {root}")

        self.seg_paths: Optional[List[Optional[Path]]] = None
        if seg_glob is not None:
            segs = [p.parent / seg_glob for p in self.image_paths]
            self.seg_paths = [p if p.exists() else None for p in segs]

    def __len__(self) -> int:
        return self.num_pairs

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(idx)  # deterministic per-index sampling
        m_idx, f_idx = rng.sample(range(len(self.image_paths)), k=2)

        moving = _load_image(self.image_paths[m_idx])
        fixed = _load_image(self.image_paths[f_idx])

        if self.normalize:
            moving = _normalize(moving)
            fixed = _normalize(fixed)

        sample: dict = {
            "moving": torch.from_numpy(moving).unsqueeze(0),  # (1, *spatial)
            "fixed": torch.from_numpy(fixed).unsqueeze(0),
            "moving_path": str(self.image_paths[m_idx]),
            "fixed_path": str(self.image_paths[f_idx]),
        }

        if self.seg_paths is not None:
            m_seg = self.seg_paths[m_idx]
            f_seg = self.seg_paths[f_idx]
            if m_seg is not None:
                sample["moving_seg"] = torch.from_numpy(_load_image(m_seg)).long()
            if f_seg is not None:
                sample["fixed_seg"] = torch.from_numpy(_load_image(f_seg)).long()

        return self._apply_transform(sample)


class AtlasDataset(BaseRegistrationDataset):
    """
    Atlas-based registration: one fixed atlas, each subject is a moving image.

    Args:
        root:        Subject directory
        atlas_path:  Path to atlas image
        image_glob:  Glob pattern for subject images
        seg_glob:    Glob pattern for subject segmentations (optional)
        atlas_seg:   Path to atlas segmentation (optional)
    """

    def __init__(
        self,
        root: str,
        atlas_path: str,
        image_glob: str = "image.nii.gz",
        seg_glob: Optional[str] = "seg.nii.gz",
        atlas_seg: Optional[str] = None,
        normalize: bool = True,
        transform: Optional[Callable] = None,
    ):
        super().__init__(transform)
        self.root = Path(root)
        self.normalize = normalize

        self.image_paths: List[Path] = sorted(self.root.glob(f"**/{image_glob}"))

        atlas_arr = _load_image(Path(atlas_path))
        self._fixed = torch.from_numpy(_normalize(atlas_arr) if normalize else atlas_arr).unsqueeze(0)

        self._fixed_seg: Optional[torch.Tensor] = None
        if atlas_seg is not None:
            self._fixed_seg = torch.from_numpy(_load_image(Path(atlas_seg))).long()

        self.seg_paths: Optional[List[Optional[Path]]] = None
        if seg_glob is not None:
            segs = [p.parent / seg_glob for p in self.image_paths]
            self.seg_paths = [p if p.exists() else None for p in segs]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        moving = _load_image(self.image_paths[idx])
        if self.normalize:
            moving = _normalize(moving)

        sample: dict = {
            "moving": torch.from_numpy(moving).unsqueeze(0),
            "fixed": self._fixed.clone(),
            "moving_path": str(self.image_paths[idx]),
        }

        if self._fixed_seg is not None:
            sample["fixed_seg"] = self._fixed_seg.clone()

        if self.seg_paths is not None and self.seg_paths[idx] is not None:
            sample["moving_seg"] = torch.from_numpy(
                _load_image(self.seg_paths[idx])
            ).long()

        return self._apply_transform(sample)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Tuple:
    """
    Build train/val/test DataLoaders from an OmegaConf/dict config.
    Returns (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader
    from .transforms import build_transforms

    root = cfg["data"]["root"]
    train_tf, val_tf = build_transforms(cfg)

    train_ds = AtlasDataset(f"{root}/train", atlas_path=f"{root}/atlas.nii.gz",
                            transform=train_tf)
    val_ds   = AtlasDataset(f"{root}/val",   atlas_path=f"{root}/atlas.nii.gz",
                            transform=val_tf)
    test_ds  = AtlasDataset(f"{root}/test",  atlas_path=f"{root}/atlas.nii.gz",
                            transform=val_tf)

    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )
