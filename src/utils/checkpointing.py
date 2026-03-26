"""Checkpoint saving and loading with best/latest model tracking."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Saves model/optimizer state for:
        - best model (lowest validation loss seen so far)
        - latest model (most recent epoch, for resuming)

    Args:
        directory: Where checkpoint files are written.
    """

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        cfg: dict,
    ) -> None:
        self._save(self.dir / "best.pt", model, optimizer, epoch, val_loss, cfg)
        # Also write a small sidecar JSON so you can inspect without loading .pt
        meta = {"epoch": epoch, "val_loss": round(val_loss, 6)}
        (self.dir / "best_meta.json").write_text(json.dumps(meta, indent=2))

    def save_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        cfg: dict,
    ) -> None:
        self._save(self.dir / "latest.pt", model, optimizer, epoch, train_loss, cfg)

    # ------------------------------------------------------------------

    @staticmethod
    def _save(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        cfg: dict,
    ) -> None:
        payload = {
            "epoch": epoch,
            "loss": loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": getattr(model, "get_config", lambda: {})(),
            "arch": model.__class__.__name__,
        }
        # Remove existing file first — avoids Windows rename-over-existing-file failure
        path.unlink(missing_ok=True)
        torch.save(payload, str(path))

    @staticmethod
    def load(
        path: str | Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint into an existing model (and optionally optimizer)."""
        payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        return payload

    def best_meta(self) -> Optional[dict]:
        p = self.dir / "best_meta.json"
        if p.exists():
            return json.loads(p.read_text())
        return None
