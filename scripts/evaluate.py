"""
Evaluate a trained model on the test set and generate paper-ready outputs.

Usage:
    python scripts/evaluate.py --checkpoint outputs/neural_ode_synthetic/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint <path> --split test --save-figures

Outputs (in checkpoint's parent directory):
    test_results.csv       - per-sample metrics
    test_summary.csv       - mean ± std / CI table (paste into LaTeX)
    figures/               - registration result images, Jacobian distributions
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from src.data import build_patch_dataloaders
from src.evaluation import evaluate_dataset, evaluate_sample
from src.models import build_model
from src.utils import (
    CheckpointManager,
    get_device,
    plot_jacobian_distribution,
    plot_registration_result,
    save_fig,
    set_paper_style,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a registration model.")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint.")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--save-figures", action="store_true",
                   help="Save registration result figures.")
    p.add_argument("--n-figures", type=int, default=5,
                   help="Number of result figures to save.")
    return p.parse_args()


@torch.inference_mode()
def main():
    args   = parse_args()
    ckpt   = Path(args.checkpoint)
    device = get_device()

    # Load config saved alongside the checkpoint
    cfg_path = ckpt.parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["experiment"]["seed"])

    # Load model
    payload    = torch.load(ckpt, map_location=device)
    model_cfg  = payload["model_config"]
    model      = build_model(payload["arch"], **model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    print(f"Loaded: {payload['arch']}  (epoch {payload['epoch']}, val_loss={payload['loss']:.4f})")

    # Data
    _, val_loader, test_loader = build_patch_dataloaders(cfg)
    loader = test_loader if args.split == "test" else val_loader

    # Evaluation config with defaults
    evaluation     = cfg.get("evaluation", {})
    req_metrics    = evaluation.get("metrics", ["ncc", "ssim", "jacobian_det"])
    bootstrap_n    = evaluation.get("bootstrap_n", 1000)
    significance_a = evaluation.get("significance_alpha", 0.05)

    # Evaluate
    all_results = []
    jac_dets    = []
    out_dir     = ckpt.parent.parent / "figures"
    if args.save_figures:
        out_dir.mkdir(exist_ok=True)
        set_paper_style()

    for i, batch in enumerate(tqdm(loader, desc=f"Evaluating ({args.split})")):
        moving = batch["moving"].to(device)
        fixed  = batch["fixed"].to(device)
        out    = model(moving, fixed)

        warped_np = out["warped"][0, 0].cpu().numpy()
        fixed_np  = fixed[0, 0].cpu().numpy()
        flow_np   = out.get("flow", out.get("phi"))[0].cpu().numpy()

        result = evaluate_sample(
            warped=warped_np,
            fixed=fixed_np,
            flow=flow_np,
            warped_seg=batch.get("moving_seg", [None])[0],
            fixed_seg=batch.get("fixed_seg",  [None])[0],
            metrics=req_metrics,
        )
        all_results.append(result)
        jac_dets.append(flow_np)

        if args.save_figures and i < args.n_figures:
            fig, _ = plot_registration_result(
                batch["moving"][0, 0].numpy(), fixed_np, warped_np
            )
            save_fig(fig, out_dir / f"result_{i:03d}", formats=("png",))
            import matplotlib.pyplot as plt
            plt.close(fig)

    # Summary table
    summary_df = evaluate_dataset(
        all_results,
        bootstrap_n=bootstrap_n,
        alpha=significance_a,
    )
    results_dir = ckpt.parent.parent
    summary_df.to_csv(results_dir / "test_summary.csv")

    per_sample_df = pd.DataFrame(all_results)
    per_sample_df.to_csv(results_dir / "test_results.csv", index=False)

    print("\n===== Test Results =====")
    print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to: {results_dir}")

    if args.save_figures:
        import numpy as np
        fig, _ = plot_jacobian_distribution([np.stack(jac_dets)])
        save_fig(fig, out_dir / "jacobian_dist")
        import matplotlib.pyplot as plt
        plt.close(fig)

    latex = summary_df[["mean", "std", "ci_lo", "ci_hi"]].to_latex(
        float_format=lambda x: f"{x:.4f}", caption="Test set evaluation metrics.",
        label="tab:results"
    )
    (results_dir / "table.tex").write_text(latex)
    print("\nLaTeX table saved to table.tex")


if __name__ == "__main__":
    main()
