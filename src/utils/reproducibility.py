"""
Reproducibility utilities: seed setting, deterministic mode, environment info.
Call set_seed() at the top of every script.
"""

import os
import platform
import random
import sys

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all random seeds for full reproducibility.

    Args:
        seed:         Integer seed value (stored in config for reproduction).
        deterministic: Enable cuDNN deterministic mode. Slightly slower but
                       ensures bitwise-identical results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        except AttributeError:
            pass


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the best available device."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer in ("mps", "cuda") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def environment_info() -> dict:
    """Collect environment metadata for the Methods section of a paper."""
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numpy": np.__version__,
    }
    return info


def print_environment() -> None:
    for k, v in environment_info().items():
        print(f"  {k}: {v}")
