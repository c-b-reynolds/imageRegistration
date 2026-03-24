"""
Ablation study runner.

Sweeps a single config parameter over a list of values, runs training for
each, evaluates on the test set, and produces a comparison table and figure.

Usage:
    python scripts/ablation.py --config configs/experiments/ablation_features.yaml

The ablation config must define:
    ablation.param:  dot-path to the parameter (e.g. 'model.base_features')
    ablation.values: list of values to sweep

Results are written to:
    results/<experiment_name>/
        ablation_summary.csv
        ablation_figure.pdf
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

from src.utils import save_fig, set_paper_style


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    return p.parse_args()


def main():
    args    = parse_args()
    exp_cfg = OmegaConf.load(args.config)
    default = OmegaConf.load("configs/default.yaml")
    cfg     = OmegaConf.to_container(OmegaConf.merge(default, exp_cfg), resolve=True)

    param  = cfg["ablation"]["param"]
    values = cfg["ablation"]["values"]
    exp    = cfg["experiment"]["name"]

    print(f"Ablation: {param} over {values}")

    summaries = []
    for val in values:
        run_name = f"{exp}_{param.split('.')[-1]}_{val}"
        override = f"{param}={val}"
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", args.config,
            f"experiment.name={run_name}",
            override,
        ]
        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Running: {' '.join(cmd)}")
        if not args.dry_run:
            ret = subprocess.run(cmd, check=True)
            # Evaluate immediately after
            ckpt = Path(cfg["experiment"]["output_dir"]) / run_name / "checkpoints" / "best.pt"
            eval_cmd = [sys.executable, "scripts/evaluate.py", "--checkpoint", str(ckpt)]
            subprocess.run(eval_cmd, check=True)
            # Read summary
            summary_path = ckpt.parent.parent / "test_summary.csv"
            if summary_path.exists():
                df = pd.read_csv(summary_path, index_col=0)
                df[param.split(".")[-1]] = val
                summaries.append(df)

    if summaries and not args.dry_run:
        combined = pd.concat(summaries)
        out = Path(cfg["experiment"]["output_dir"]) / exp / "ablation_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out)
        print(f"\nAblation summary saved to {out}")

        # Figure: metric vs param value for key metrics
        set_paper_style()
        key_metrics = ["ncc", "mean_dice", "jac_pct_neg"]
        available   = [m for m in key_metrics if m in combined.index.unique()]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
            if len(available) == 1:
                axes = [axes]
            for ax, metric in zip(axes, available):
                sub = combined.loc[metric].reset_index()
                ax.errorbar(
                    sub[param.split(".")[-1]], sub["mean"],
                    yerr=sub["std"], fmt="o-", capsize=4
                )
                ax.set_xlabel(param.split(".")[-1])
                ax.set_ylabel(metric)
            fig.tight_layout()
            save_fig(fig, out.parent / "ablation_figure")
            plt.close(fig)


if __name__ == "__main__":
    main()
