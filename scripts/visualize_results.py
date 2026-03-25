"""
Visualize registration results for a trained model on the validation or test set.

Saves side-by-side figures: Moving | Fixed | Warped | |Warped - Fixed|
For NeuralODERegistration, also saves the time evolution (trajectory strip).

Usage:
    python scripts/visualize_results.py --checkpoint results/neural_ode_numpy/checkpoints/best.pt
    python scripts/visualize_results.py --checkpoint <path> --split val --n 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

from src.models import build_model
from src.utils import CheckpointManager, get_device, set_seed, save_fig, set_paper_style


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n", type=int, default=5, help="Number of samples to visualize")
    return p.parse_args()


@torch.inference_mode()
def main():
    args   = parse_args()
    ckpt   = Path(args.checkpoint)
    device = get_device()

    # Load config and model
    cfg_path = ckpt.parent.parent / "config.yaml"
    cfg      = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    set_seed(cfg["experiment"]["seed"])
    set_paper_style()

    payload = torch.load(ckpt, map_location=device)
    model   = build_model(payload["arch"], **payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    print(f"Loaded {payload['arch']} (epoch {payload['epoch']}, val_loss={payload['loss']:.4f})")

    # Data
    root = cfg["data"]["root"]
    use_numpy = cfg["data"].get("format") == "numpy" or Path(f"{root}/atlas.npy").exists()
    if use_numpy:
        from src.data.numpy_dataset import build_numpy_dataloaders
        _, val_loader, test_loader = build_numpy_dataloaders(cfg)
    else:
        from src.data import build_dataloaders
        _, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader

    # Output directory
    out_dir = ckpt.parent.parent / "figures" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, batch in enumerate(loader):
        if i >= args.n:
            break

        moving = batch["moving"].to(device)
        fixed  = batch["fixed"].to(device)
        out    = model(moving, fixed)

        moving_np = moving[0, 0].cpu().numpy()
        fixed_np  = fixed[0, 0].cpu().numpy()
        warped_np = out["warped"][0, 0].cpu().numpy()

        # --- Main registration figure ---
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        imgs   = [moving_np, fixed_np, warped_np, abs(warped_np - fixed_np)]
        titles = ["Moving", "Fixed", "Warped", "|Warped - Fixed|"]
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout()
        save_fig(fig, out_dir / f"sample_{i:03d}_registration", formats=("png",))
        plt.close(fig)

        # --- Trajectory strip (NeuralODE only) ---
        if "trajectory" in out:
            traj  = out["trajectory"][:, 0, 0].cpu().numpy()  # (n_t, H, W)
            n_t   = traj.shape[0]
            # Show ~6 evenly spaced frames
            indices = [int(j * (n_t - 1) / 5) for j in range(6)]
            fig, axes = plt.subplots(1, len(indices), figsize=(3 * len(indices), 3))
            for ax, t_idx in zip(axes, indices):
                t_val = t_idx / (n_t - 1)
                ax.imshow(traj[t_idx], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"t={t_val:.2f}")
                ax.axis("off")
            fig.suptitle("Image evolution  f(r, t)", y=1.02)
            fig.tight_layout()
            save_fig(fig, out_dir / f"sample_{i:03d}_trajectory", formats=("png",))
            plt.close(fig)

        print(f"Saved sample {i:03d}")

    print(f"\nFigures saved to: {out_dir}")


if __name__ == "__main__":
    main()
