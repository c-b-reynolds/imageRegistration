"""
Train any registered model on image pair data from a patches.npy file.

The patches file must be shape (N, 2, H, W) float32 with values in [0, 1].
Channel 0 is the moving image, channel 1 is the fixed image.

Usage:
    python scripts/train_image_pairs.py --config configs/experiments/direct_hybrid_synthetic.yaml
    python scripts/train_image_pairs.py --config configs/experiments/direct_hybrid_sar.yaml --patches dataset/patches.npy

CLI flags override the config when provided:
    --patches     path to patches.npy
    --epochs      number of training epochs
    --batch-size  batch size
    --reg-weight  regularization weight
    --patience    early stopping patience (0 = disabled)
    --no-erasing  disable random erasing augmentation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from src.data.patch_dataset import build_patch_dataloaders
from src.models import build_model
from src.training import RegistrationLoss, Trainer, build_optimizer, build_scheduler
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",     required=True,
                        help="Path to a YAML config file.")
    parser.add_argument("--patches",    default=None,
                        help="Path to patches.npy — overrides config.")
    parser.add_argument("--epochs",     type=int,   default=None,
                        help="Number of training epochs — overrides config.")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="Batch size — overrides config.")
    parser.add_argument("--reg-weight", type=float, default=None,
                        help="Regularization weight — overrides config.")
    parser.add_argument("--patience",   type=int,   default=None,
                        help="Early stopping patience (0 = disabled) — overrides config.")
    parser.add_argument("--no-erasing", action="store_true",
                        help="Disable random erasing augmentation — overrides config.")
    parser.add_argument("--weights",    default=None,
                        help="Path to a checkpoint whose model weights are used to "
                             "initialise training (optimizer and epoch are NOT restored).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.patches    is not None: cfg["data"]["patches_path"]               = args.patches
    if args.epochs     is not None: cfg["training"]["epochs"]                 = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"]             = args.batch_size
    if args.reg_weight is not None: cfg["loss"]["regularization_weight"]      = args.reg_weight
    if args.patience   is not None: cfg["training"]["early_stopping_patience"] = args.patience
    if args.no_erasing:             cfg["augmentation"]["use_erasing"]        = False

    # Infer image_size from the patches file — always overrides config
    patches_path = Path(cfg["data"]["patches_path"])
    if not patches_path.exists():
        raise FileNotFoundError(
            f"Patches file not found: {patches_path}\n"
            f"Generate one with: python scripts/generate_synthetic_patches.py"
        )
    patches    = np.load(patches_path, mmap_mode="r")
    patch_size = patches.shape[-1]
    print(f"Patches: {patches.shape}  patch_size={patch_size}")

    cfg["model"]["image_size"] = [patch_size, patch_size]

    set_seed(cfg["experiment"]["seed"], deterministic=False)
    device = get_device()
    print(f"Device: {device}\n")

    # Output directory
    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))

    # Data
    train_loader, val_loader, _ = build_patch_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)}  "
          f"Val batches:   {len(val_loader)}  "
          f"Batch size:    {cfg['training']['batch_size']}")

    # Model
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], **model_kwargs)
    print(f"\n{model.parameter_summary()}\n")

    # Optionally initialise from a prior checkpoint (weights only)
    if args.weights is not None:
        payload = torch.load(args.weights, map_location=str(device), weights_only=False)
        model.load_state_dict(payload["model_state"])
        print(f"Loaded weights from: {args.weights}  "
              f"(epoch {payload.get('epoch', '?')}, loss {payload.get('loss', float('nan')):.4f})")

    # Loss
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

    aug = cfg.get("augmentation", {})
    print(f"Training for {cfg['training']['epochs']} epochs  "
          f"| model={cfg['model']['name']}  "
          f"| reg_weight={cfg['loss']['regularization_weight']}  "
          f"| erasing={aug.get('use_erasing', False)}\n")

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg["training"]["epochs"],
        early_stopping_patience=cfg["training"]["early_stopping_patience"],
        val_every=cfg["training"]["val_every"],
    )

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
