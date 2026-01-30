"""
paper_plot_compare.py

Overlay plots for multiple CSVs produced by paper_spectral_grid_bench.py.

Example:
  python paper_plot_compare.py \
    --csvs runs/cifar_s4d/s4d_rel005_s4d_cifar10_rho_sweep.csv,runs/cifar_resnet18/resnet18_rel005_resnet18_cifar10_rho_sweep.csv \
    --labels S4D,ResNet18 \
    --out_png runs/compare_drop.png
"""

import argparse
import csv
import os
from typing import Dict, List

def read_csv(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # keep numeric fields
            out = {}
            for k, v in row.items():
                if k in ["model", "dataset"]:
                    out[k] = v
                else:
                    out[k] = float(v)
            rows.append(out)
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", type=str, required=True, help="comma-separated csv paths")
    p.add_argument("--labels", type=str, default=None, help="comma-separated labels (optional)")
    p.add_argument("--metric", choices=["drop_mean","flip_mean","logit_rel_mean","margin_drop_mean"], default="drop_mean")
    p.add_argument("--out_png", type=str, required=True)
    args = p.parse_args()

    csvs = [s.strip() for s in args.csvs.split(",") if s.strip()]
    labels = None
    if args.labels is not None:
        labels = [s.strip() for s in args.labels.split(",") if s.strip()]
        assert len(labels) == len(csvs), "labels length must match csvs"

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib required for plotting") from e

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    for i, path in enumerate(csvs):
        rows = read_csv(path)
        rows = sorted(rows, key=lambda r: r["rho"])
        rhos = [r["rho"] for r in rows]
        ys = [r[args.metric] for r in rows]

        label = labels[i] if labels is not None else os.path.basename(path)
        plt.plot(rhos, ys, label=label)

    plt.xlabel("rho (1.0 = Nyquist)")
    plt.ylabel(args.metric)
    plt.title(f"Overlay: {args.metric} vs rho")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=220, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
