"""
Metric logging to TensorBoard and/or Weights & Biases.
Wraps both backends behind a unified interface so scripts are backend-agnostic.
"""

import json
from pathlib import Path
from typing import Optional


class MetricLogger:
    """
    Unified logging interface for training metrics.

    Supports:
        - TensorBoard (always-on if torch.utils.tensorboard is available)
        - Weights & Biases (optional — set backend='wandb' or 'both' in config)
        - JSON sidecar file (always written, backend-independent)

    Args:
        log_dir: Directory for TensorBoard event files.
        cfg:     Full experiment config dict.
    """

    def __init__(self, log_dir: str | Path, cfg: dict):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        backend = cfg.get("logging", {}).get("backend", "tensorboard")

        # ---- TensorBoard ----
        self._tb = None
        if backend in ("tensorboard", "both"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[Logger] TensorBoard not available. Install tensorboard.")

        # ---- W&B ----
        self._wb = False
        if backend in ("wandb", "both"):
            try:
                import wandb
                project = cfg.get("logging", {}).get("wandb_project", "image-registration")
                entity  = cfg.get("logging", {}).get("wandb_entity", None) or None
                wandb.init(
                    project=project,
                    entity=entity,
                    name=cfg.get("experiment", {}).get("name", "run"),
                    config=cfg,
                    tags=cfg.get("experiment", {}).get("tags", []),
                    dir=str(self.log_dir.parent),
                )
                self._wb = True
            except ImportError:
                print("[Logger] wandb not available. Install wandb.")

        # ---- JSON fallback ----
        self._json_path = self.log_dir / "metrics.jsonl"
        self._json_fh = self._json_path.open("a")

    def log_epoch(self, epoch: int, metrics: dict, lr: Optional[float] = None) -> None:
        """Log a dict of scalar metrics for the given epoch."""
        if lr is not None:
            metrics = {**metrics, "lr": lr}

        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, global_step=epoch)

        if self._wb:
            import wandb
            wandb.log({"epoch": epoch, **metrics})

        row = {"epoch": epoch, **{k: v for k, v in metrics.items()
                                   if isinstance(v, (int, float))}}
        self._json_fh.write(json.dumps(row) + "\n")
        self._json_fh.flush()

    def log_scalars(self, tag_group: str, values: dict, step: int) -> None:
        if self._tb is not None:
            self._tb.add_scalars(tag_group, values, global_step=step)
        if self._wb:
            import wandb
            wandb.log({f"{tag_group}/{k}": v for k, v in values.items()}, step=step)

    def log_hparams(self, hparams: dict, final_metrics: dict) -> None:
        """
        Log hyperparameters alongside final metrics.
        Populates the HPARAMS tab in TensorBoard, enabling parallel-coordinates
        and scatter plots across runs.

        Args:
            hparams:       Flat dict of hyperparameter name -> scalar value.
            final_metrics: Flat dict of metric name -> scalar value (e.g. best val loss).
        """
        # TensorBoard requires all values to be int, float, str, or bool
        def _sanitize(d: dict) -> dict:
            return {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                    for k, v in d.items()}

        if self._tb is not None:
            self._tb.add_hparams(_sanitize(hparams), _sanitize(final_metrics))

        if self._wb:
            import wandb
            wandb.config.update(_sanitize(hparams))

        with (self.log_dir / "hparams.json").open("w") as f:
            json.dump({"hparams": hparams, "metrics": final_metrics}, f, indent=2)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wb:
            import wandb
            wandb.finish()
        self._json_fh.close()
