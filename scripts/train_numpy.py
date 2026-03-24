"""
Training script for numpy datasets.
Identical to train.py but uses build_numpy_dataloaders instead of build_dataloaders.

Usage:
    python scripts/train_numpy.py --config configs/experiments/simple_cnn_numpy.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from omegaconf import OmegaConf

from src.data.numpy_dataset import build_numpy_dataloaders
from src.models import build_model
from src.training import RegistrationLoss, Trainer, build_optimizer, build_scheduler
from src.utils import get_device, print_environment, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/simple_cnn_numpy.yaml")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


def load_config(config_path: str, overrides: list) -> dict:
    default = OmegaConf.load("configs/default.yaml")
    exp     = OmegaConf.load(config_path)
    cli     = OmegaConf.from_dotlist(overrides)
    return OmegaConf.to_container(OmegaConf.merge(default, exp, cli), resolve=True)


def main():
    args = parse_args()
    cfg  = load_config(args.config, args.overrides)

    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    print(f"\nExperiment : {cfg['experiment']['name']}")
    print(f"Device     : {device}")
    print_environment()

    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))

    train_loader, val_loader, _ = build_numpy_dataloaders(cfg)

    model_cfg = {k: v for k, v in cfg["model"].items() if k != "name"}
    model_cfg["image_size"] = cfg["data"]["image_size"]
    model = build_model(cfg["model"]["name"], **model_cfg)
    print(f"\n{model.parameter_summary()}")

    loss_fn   = RegistrationLoss(**cfg["loss"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        cfg=cfg,
        output_dir=str(out_dir),
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 0),
        val_every=cfg["training"].get("val_every", 1),
    )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
