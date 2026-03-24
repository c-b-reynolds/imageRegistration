"""Abstract base class for all registration network architectures."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseRegistrationModel(nn.Module, ABC):
    """
    All custom architectures should inherit from this class.

    Subclasses must implement:
        forward(moving, fixed) -> dict with at least 'warped' and 'flow' keys
        get_config() -> dict of constructor kwargs (for checkpoint save/load)
    """

    @abstractmethod
    def forward(
        self, moving: torch.Tensor, fixed: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            moving: (B, C, *spatial) moving image
            fixed:  (B, C, *spatial) fixed image
        Returns:
            dict containing:
                'warped' - moving image warped to fixed space
                'flow'   - displacement field (B, ndim, *spatial)
                (optional) 'flow_sequence' - list of intermediate flows
        """

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return constructor kwargs needed to reconstruct this model."""

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        total = self.num_parameters
        return f"{self.__class__.__name__}: {total:,} trainable parameters"

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save model weights + config so the model can be fully reconstructed."""
        payload = {
            "arch": self.__class__.__name__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "BaseRegistrationModel":
        """Reconstruct a model from a checkpoint saved by save_checkpoint()."""
        payload = torch.load(path, map_location=device)
        model = cls(**payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model
