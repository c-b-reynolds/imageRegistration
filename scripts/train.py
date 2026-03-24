"""
Training entry point.

Usage:
    python scripts/train.py                                          # uses configs/default.yaml
    python scripts/train.py --config configs/experiments/baseline.yaml
    python scripts/train.py --config configs/experiments/baseline.yaml --seed 123
    python scripts/train.py --config configs/experiments/baseline.yaml model.base_features=32

OmegaConf-style dot-path overrides are supported as positional arguments.
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from omegaconf import OmegaConf

from src.data import build_dataloaders
from src.models import build_model
from src.training import RegistrationLoss, Trainer, build_optimizer, build_scheduler
from src.utils import get_device, print_environment, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a registration network.")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Path to experiment YAML config.")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from (latest.pt).")
    p.add_argument("overrides", nargs="*",
                   help="OmegaConf dot-path overrides, e.g. training.lr=1e-3")
    return p.parse_args()


def load_config(config_path: str, overrides: list) -> dict:
    """Merge default config <- experiment config <- CLI overrides."""
    default_cfg = OmegaConf.load("configs/default.yaml")
    exp_cfg     = OmegaConf.load(config_path) if config_path != "configs/default.yaml" else OmegaConf.create({})
    cli_cfg     = OmegaConf.from_dotlist(overrides)
    merged      = OmegaConf.merge(default_cfg, exp_cfg, cli_cfg)
    return OmegaConf.to_container(merged, resolve=True)


def main():
    args = parse_args()
    cfg  = load_config(args.config, args.overrides)

    # --- Reproducibility ---
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    print(f"\nExperiment: {cfg['experiment']['name']}")
    print(f"Device:     {device}")
    print_environment()

    # --- Output directory ---
    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save resolved config for full reproducibility
    (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))

    # --- Data ---
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # --- Model ---
    model_cfg = cfg["model"].copy()
    model_cfg.pop("name")
    model_cfg["image_size"] = cfg["data"]["image_size"]
    model = build_model(cfg["model"]["name"], **model_cfg)
    print(f"\n{model.parameter_summary()}")

    # --- Loss / Optimizer / Scheduler ---
    loss_fn   = RegistrationLoss(**cfg["loss"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # --- Resume ---
    start_epoch = 1
    if args.resume is not None:
        from src.utils import CheckpointManager
        payload = CheckpointManager.load(args.resume, model, optimizer, device=str(device))
        start_epoch = payload["epoch"] + 1
        print(f"Resumed from epoch {payload['epoch']}")

    # --- Train ---
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        cfg=cfg,
        output_dir=str(out_dir),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 0),
        val_every=cfg["training"].get("val_every", 1),
    )

    print(f"\nTraining complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
