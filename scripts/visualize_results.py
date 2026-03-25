"""
Visualize registration results for a trained HybridODERegistration model on
SAR patch-pair data (patches.npy produced by sar_align/make_dataset.py).

Saves per-sample figures:
  - Moving | Fixed | Warped | |Warped - Fixed|
  - Deformation field magnitude (phi)
  - Image trajectory strip (t=0 → t=1)

Usage:
    python scripts/visualize_results.py --checkpoint outputs/hybrid_ode_sar/checkpoints/best.pt
    python scripts/visualize_results.py --checkpoint <path> --split test --n 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.data.patch_dataset import build_patch_dataloaders
from src.models import build_model
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a checkpoint .pt file saved by train_hybrid_patches.py")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Which data split to visualize")
    p.add_argument("--n", type=int, default=5,
                   help="Number of samples to visualize")
    return p.parse_args()


@torch.inference_mode()
def main():
    args   = parse_args()
    ckpt   = Path(args.checkpoint)
    device = get_device()

    # Config lives two levels up from the checkpoint file:
    #   outputs/<name>/checkpoints/best.pt  →  outputs/<name>/config.yaml
    cfg_path = ckpt.parent.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"], deterministic=False)

    # Load model from checkpoint
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    model_kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    model = build_model(cfg["model"]["name"], **model_kwargs).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    print(f"Loaded {cfg['model']['name']}  "
          f"epoch={payload.get('epoch', '?')}  "
          f"val_loss={payload.get('loss', float('nan')):.4f}")

    # Data — no augmentation on val/test
    _, val_loader, test_loader = build_patch_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader

    # Output directory
    out_dir = ckpt.parent.parent / "figures" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for batch in loader:
        if n_saved >= args.n:
            break

        moving = batch["moving"].to(device)  # (B, 1, H, W)
        fixed  = batch["fixed"].to(device)
        out    = model(moving, fixed)

        # Visualize first sample in the batch
        moving_np = moving[0, 0].cpu().numpy()
        fixed_np  = fixed[0, 0].cpu().numpy()
        warped_np = out["warped"][0, 0].cpu().numpy()
        diff_np   = np.abs(warped_np - fixed_np)

        # --- Registration panel: Moving | Fixed | Warped | |Warped - Fixed| ---
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        for ax, img, title in zip(
            axes,
            [moving_np, fixed_np, warped_np, diff_np],
            ["Moving", "Fixed", "Warped", "|Warped - Fixed|"],
        ):
            vmax = 1.0 if title != "|Warped - Fixed|" else diff_np.max() + 1e-6
            ax.imshow(img, cmap="gray", vmin=0, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{n_saved:03d}_registration.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Deformation / velocity field magnitude ---
        if "phi" in out:
            phi = out["phi"][0].cpu().numpy()  # (2, H, W)  normalized [-1,1] coords
            H, W = phi.shape[1], phi.shape[2]

            is_eulerian = cfg["model"]["name"] == "EulerianHybridODERegistration"

            if is_eulerian:
                # phi is the instantaneous velocity field — magnitude in pixel units
                disp_mag = np.sqrt(
                    (phi[0] * (W - 1) / 2.0) ** 2 +
                    (phi[1] * (H - 1) / 2.0) ** 2
                )
                field_title = "Velocity magnitude (pixels/unit time)"
            else:
                # phi is an integrated position field — subtract identity to get displacement
                dx = phi[0] * (W - 1) / 2.0
                dy = phi[1] * (H - 1) / 2.0
                xs = np.linspace(-1, 1, W) * (W - 1) / 2.0
                ys = np.linspace(-1, 1, H) * (H - 1) / 2.0
                grid_x, grid_y = np.meshgrid(xs, ys)
                disp_mag = np.sqrt((dx - grid_x) ** 2 + (dy - grid_y) ** 2)
                field_title = "Displacement magnitude (pixels)"

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            im = ax.imshow(disp_mag, cmap="viridis")
            ax.set_title(field_title, fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{n_saved:03d}_deformation.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- Trajectory strip: image evolution t=0 → t=1 ---
        if "trajectory" in out:
            traj  = out["trajectory"][:, 0, 0].cpu().numpy()  # (n_t, H, W)
            n_t   = traj.shape[0]
            indices = [int(j * (n_t - 1) / min(5, n_t - 1)) for j in range(min(6, n_t))]
            fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
            if len(indices) == 1:
                axes = [axes]
            for ax, t_idx in zip(axes, indices):
                t_val = t_idx / max(n_t - 1, 1)
                ax.imshow(traj[t_idx], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"t={t_val:.2f}", fontsize=9)
                ax.axis("off")
            fig.suptitle("Image transport  f(r, t)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / f"sample_{n_saved:03d}_trajectory.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved sample {n_saved:03d}")
        n_saved += 1

    print(f"\nFigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
