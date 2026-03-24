"""
Training loop with validation, early stopping, checkpointing, and metric logging.
"""

import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.checkpointing import CheckpointManager
from src.utils.logging import MetricLogger


class Trainer:
    """
    Self-contained trainer.  Call trainer.fit() to run training.

    Args:
        model:        The registration network.
        loss_fn:      RegistrationLoss or any callable returning a dict with 'total'.
        optimizer:    Torch optimizer.
        scheduler:    LR scheduler (optional).
        device:       'cuda', 'mps', or 'cpu'.
        cfg:          Full experiment config dict (used for checkpoint metadata).
        output_dir:   Root directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        cfg: dict,
        output_dir: str,
    ):
        self.model      = model.to(device)
        self.loss_fn    = loss_fn
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.device     = device
        self.cfg        = cfg
        self.output_dir = Path(output_dir)

        self.ckpt_manager = CheckpointManager(self.output_dir / "checkpoints")
        self.logger       = MetricLogger(self.output_dir / "logs", cfg)

        self._best_val_loss = float("inf")
        self._patience_ctr  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping_patience: int = 0,
        val_every: int = 1,
    ) -> Dict[str, list]:
        """Run the training loop. Returns a history dict of per-epoch metrics."""
        history: Dict[str, list] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, epoch)
            history["train_loss"].append(train_metrics["loss"])

            val_metrics: dict = {}
            if epoch % val_every == 0:
                val_metrics = self._val_epoch(val_loader, epoch)
                history["val_loss"].append(val_metrics["loss"])

                # Checkpoint best model
                if val_metrics["loss"] < self._best_val_loss:
                    self._best_val_loss = val_metrics["loss"]
                    self._patience_ctr  = 0
                    self.ckpt_manager.save_best(self.model, self.optimizer, epoch,
                                                val_metrics["loss"], self.cfg)
                else:
                    self._patience_ctr += 1

                # Early stopping
                if early_stopping_patience > 0 and self._patience_ctr >= early_stopping_patience:
                    print(f"[Epoch {epoch}] Early stopping triggered.")
                    break

            # Periodic checkpoint
            self.ckpt_manager.save_latest(self.model, self.optimizer, epoch,
                                          train_metrics["loss"], self.cfg)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", train_metrics["loss"]))
                else:
                    self.scheduler.step()

            self.logger.log_epoch(epoch, {**train_metrics, **val_metrics},
                                  lr=self.optimizer.param_groups[0]["lr"])
            elapsed = time.time() - t0
            self._print_epoch(epoch, epochs, train_metrics, val_metrics, elapsed)

        # Log hyperparameters + final metrics to the HPARAMS tab
        hparams = {
            "model":          self.cfg["model"]["name"],
            "lr":             self.cfg["training"]["lr"],
            "batch_size":     self.cfg["training"]["batch_size"],
            "optimizer":      self.cfg["training"]["optimizer"],
            "similarity":     self.cfg["loss"]["similarity"],
            "regularization": self.cfg["loss"]["regularization"],
            "reg_weight":     self.cfg["loss"]["regularization_weight"],
        }
        final_metrics = {"best_val_loss": self._best_val_loss}
        self.logger.log_hparams(hparams, final_metrics)

        self.logger.close()
        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = sim_loss = reg_loss = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Train {epoch}", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            moving = batch["moving"].to(self.device, non_blocking=True)
            fixed  = batch["fixed"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            out    = self.model(moving, fixed)
            losses = self.loss_fn(out["warped"], fixed, out["flow"])
            losses["total"].backward()

            grad_clip = self.cfg["training"].get("gradient_clip", 0.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

            bs = moving.size(0)
            total_loss += losses["total"].item() * bs
            sim_loss   += losses["similarity"].item() * bs
            reg_loss   += losses["regularization"].item() * bs
            n          += bs
            pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        return {"loss": total_loss / n, "sim": sim_loss / n, "reg": reg_loss / n}

    @torch.inference_mode()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in tqdm(loader, desc=f"  Val {epoch}", leave=False, dynamic_ncols=True):
            moving = batch["moving"].to(self.device, non_blocking=True)
            fixed  = batch["fixed"].to(self.device, non_blocking=True)

            out    = self.model(moving, fixed)
            losses = self.loss_fn(out["warped"], fixed, out["flow"])
            bs = moving.size(0)
            total_loss += losses["total"].item() * bs
            n          += bs

        return {"val_loss": total_loss / n, "loss": total_loss / n}

    @staticmethod
    def _print_epoch(epoch, total, train, val, elapsed):
        val_str = f"  val_loss={val.get('loss', float('nan')):.4f}" if val else ""
        print(f"Epoch {epoch:>4}/{total}  train_loss={train['loss']:.4f}{val_str}  [{elapsed:.1f}s]")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["training"]["optimizer"].lower()
    lr   = cfg["training"]["lr"]
    wd   = cfg["training"].get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int = 1):
    name   = cfg["training"].get("lr_scheduler", "none").lower()
    epochs = cfg["training"]["epochs"]
    lr_min = cfg["training"].get("lr_min", 0.0)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    elif name == "none":
        return None
    raise ValueError(f"Unknown scheduler '{name}'")
