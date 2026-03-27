"""
Ablation study runner.

Sweeps a single config parameter over a list of values, runs training for
each, evaluates on the test set, and produces a comparison table and figure.

The base config must be a self-contained experiment YAML (same format as
those passed to train_image_pairs.py).  Add an `ablation` section:

    ablation:
      param:  model.features      # dot-path to the parameter
      values: [16, 32, 64]        # values to sweep

Usage:
    python scripts/ablation.py --config configs/experiments/ablation_features.yaml

Results are written to:
    <output_dir>/<experiment_name>/
        ablation_summary.csv
        ablation_figure.pdf
"""

import argparse
import copy
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.utils import save_fig, set_paper_style


def _set_dotpath(d: dict, dotpath: str, value) -> None:
    """Set a nested dict key given a dot-separated path, e.g. 'model.features'."""
    keys = dotpath.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   help="Self-contained experiment YAML with an 'ablation' section.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    param  = base_cfg["ablation"]["param"]
    values = base_cfg["ablation"]["values"]
    exp    = base_cfg["experiment"]["name"]

    print(f"Ablation: {param} over {values}")

    summaries = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for val in values:
            run_name = f"{exp}_{param.split('.')[-1]}_{val}"
            run_cfg  = copy.deepcopy(base_cfg)
            run_cfg["experiment"]["name"] = run_name
            _set_dotpath(run_cfg, param, val)

            tmp_cfg_path = Path(tmpdir) / f"{run_name}.yaml"
            tmp_cfg_path.write_text(yaml.dump(run_cfg, default_flow_style=False))

            cmd = [sys.executable, "scripts/train_image_pairs.py",
                   "--config", str(tmp_cfg_path)]
            print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Running: {' '.join(cmd)}")

            if not args.dry_run:
                subprocess.run(cmd, check=True)

                ckpt = (Path(run_cfg["experiment"]["output_dir"])
                        / run_name / "checkpoints" / "best.pt")
                eval_cmd = [sys.executable, "scripts/evaluate.py",
                            "--checkpoint", str(ckpt)]
                subprocess.run(eval_cmd, check=True)

                summary_path = ckpt.parent.parent / "test_summary.csv"
                if summary_path.exists():
                    df = pd.read_csv(summary_path, index_col=0)
                    df[param.split(".")[-1]] = val
                    summaries.append(df)

    if summaries and not args.dry_run:
        combined = pd.concat(summaries)
        out = (Path(base_cfg["experiment"]["output_dir"])
               / exp / "ablation_summary.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out)
        print(f"\nAblation summary saved to {out}")

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
