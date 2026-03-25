"""
Train HybridODERegistration on SAR-style patch pairs from patches.npy.

Generates a dummy patches.npy if one does not already exist at the path
specified in cfg["data"]["patches_path"], so this script is immediately
runnable as a test case without any real data.

Usage:
    # No config — use built-in defaults (good for quick tests)
    python scripts/train_hybrid_patches.py
    python scripts/train_hybrid_patches.py --patches path/to/patches.npy

    # With a YAML config (recommended for real training)
    python scripts/train_hybrid_patches.py --config configs/experiments/hybrid_ode_sar.yaml
    python scripts/train_hybrid_patches.py --config configs/experiments/hybrid_ode_sar.yaml \
        --patches dataset/patches.npy

    # CLI flags always override the config
    python scripts/train_hybrid_patches.py --config configs/experiments/hybrid_ode_sar.yaml \
        --reg-weight 0.05 --epochs 50
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
from src.utils import set_seed, get_device


# ---------------------------------------------------------------------------
# Dummy data generator
# ---------------------------------------------------------------------------

def _gaussian_blob(size, cx, cy, sigma):
    y, x = np.mgrid[0:size, 0:size]
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def _make_image(size, rng):
    img = np.zeros((size, size), dtype=np.float32)
    for _ in range(rng.integers(2, 6)):
        cx    = rng.uniform(size * 0.2, size * 0.8)
        cy    = rng.uniform(size * 0.2, size * 0.8)
        sigma = rng.uniform(size * 0.05, size * 0.15)
        img  += rng.uniform(0.5, 1.0) * _gaussian_blob(size, cx, cy, sigma)
    return np.clip(img, 0, 1).astype(np.float32)


def generate_dummy_patches(path: Path, n: int = 400, size: int = 64, seed: int = 0):
    """
    Create a patches.npy of shape (N, 2, size, size) from random Gaussian
    blob images. Channel 0 and channel 1 are independently drawn images of
    the same blob configuration with small random intensity perturbations,
    acting as a stand-in for two SAR acquisitions of the same scene.
    """
    print(f"Generating {n} dummy patch pairs ({size}x{size}) -> {path}")
    rng     = np.random.default_rng(seed)
    patches = np.zeros((n, 2, size, size), dtype=np.float32)
    for i in range(n):
        base          = _make_image(size, rng)
        # Small intensity perturbation to simulate a second acquisition
        noise         = rng.normal(0, 0.05, base.shape).astype(np.float32)
        patches[i, 0] = base
        patches[i, 1] = np.clip(base + noise, 0, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, patches)
    print(f"  Saved {patches.shape} float32 array.")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def build_cfg(args) -> dict:
    patch_size = 64  # must match the images in patches.npy

    return {
        "experiment": {
            "name":       "hybrid_ode_patches",
            "seed":       42,
            "output_dir": "outputs",
        },
        "data": {
            "patches_path": args.patches,
            "val_frac":     0.15,
            "test_frac":    0.15,
            "seed":         42,
            "num_workers":  0,
        },
        "model": {
            "name":               "HybridODERegistration",
            # image_size is set dynamically below from the patches shape
            "in_channels":        1,
            "stem_channels":      32,
            "kernel_size":        8,
            "stride":             8,   # non-overlapping: H' = patch_size / 8
            "padding":            0,
            "pool_type":          "max",
            "stem_hidden_dim":    32,
            # embed_dim must be divisible by 4^n_stages = 4^3 = 64 for 64x64
            "embed_dim":          64,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "nhead":              4,
            "ffn_dim":            128,
            "tgt_mode":           "encoder_output",
            "dropout":            0.1,
            "method":             "rk4",
            "n_t":                5,
            "adjoint":            False,
        },
        "loss": {
            "similarity":            "ncc",
            "regularization":        "deformation_gradient",
            "similarity_weight":     1.0,
            "regularization_weight": args.reg_weight,
        },
        "training": {
            "epochs":                    args.epochs,
            "batch_size":                args.batch_size,
            "optimizer":                 "adam",
            "lr":                        1e-4,
            "weight_decay":              1e-5,
            "lr_scheduler":              "cosine",
            "lr_min":                    1e-6,
            "gradient_clip":             1.0,
            "early_stopping_patience":   args.patience,
            "val_every":                 1,
        },
        "augmentation": {
            "use_rotation":  True,
            "use_erasing":   not args.no_erasing,
            "erasing_scale": [0.02, 0.2],
            "erasing_ratio": [0.3,  3.0],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",     default=None,
                        help="Path to a YAML config file. If omitted, built-in defaults are used.")
    parser.add_argument("--patches",    default=None,
                        help="Path to patches.npy — overrides config. Generated if absent.")
    parser.add_argument("--reg-weight", type=float, default=None,
                        help="Deformation gradient regularization weight — overrides config.")
    parser.add_argument("--epochs",     type=int,   default=None,
                        help="Number of training epochs — overrides config.")
    parser.add_argument("--batch-size", type=int,   default=None,
                        help="Batch size — overrides config.")
    parser.add_argument("--patience",   type=int,   default=None,
                        help="Early stopping patience (0 = disabled) — overrides config.")
    parser.add_argument("--no-erasing", action="store_true",
                        help="Disable SAR-aware random erasing — overrides config.")
    args = parser.parse_args()

    # Load config: YAML if provided, otherwise built-in defaults
    if args.config is not None:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = build_cfg(args)

    # CLI flags override config values when explicitly provided
    if args.patches    is not None:
        cfg["data"]["patches_path"] = args.patches
    if args.reg_weight is not None:
        cfg["loss"]["regularization_weight"] = args.reg_weight
    if args.epochs     is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.patience   is not None:
        cfg["training"]["early_stopping_patience"] = args.patience
    if args.no_erasing:
        cfg["augmentation"]["use_erasing"] = False

    # Generate dummy data if needed
    patches_path = Path(cfg["data"]["patches_path"])
    if not patches_path.exists():
        generate_dummy_patches(patches_path)

    # Infer patch size from the file — always overrides whatever is in config
    patches    = np.load(patches_path, mmap_mode="r")
    patch_size = patches.shape[-1]
    print(f"Patches: {patches.shape}  patch_size={patch_size}")

    cfg["model"]["image_size"] = (patch_size, patch_size)

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
          f"Val batches: {len(val_loader)}  "
          f"Batch size: {cfg['training']['batch_size']}")

    # Model
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], **model_kwargs)
    print(f"\n{model.parameter_summary()}\n")

    # Loss — NCC similarity + deformation gradient regularization on phi
    loss_fn   = RegistrationLoss(**cfg["loss"])

    # Optimiser + scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Train
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        cfg=cfg,
        output_dir=str(out_dir),
    )

    print(f"Training for {cfg['training']['epochs']} epochs  "
          f"| reg_weight={cfg['loss']['regularization_weight']}  "
          f"| rotation=True  "
          f"| erasing={cfg['augmentation']['use_erasing']}\n")

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
