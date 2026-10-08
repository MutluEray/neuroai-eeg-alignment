"""Sensitivity of the ResNet-50 vs ViT-B/16 (grid readout) comparison to readout choices.

    # 1) extract the other pooling grids (restricted layers), 2) fit them, 3) PCA-500 fit, then:
    python3 sensitivity.py

Variants: pooling grid 1 / 2 (main) / 4 with 1000 PCA components, and PCA 500 with grid 2.
For each variant, each model's score is its leave-one-subject-out best layer among config.SENS_LAYERS (same layer
set for every variant), and the difference ResNet-50 - ViT-grid is bootstrapped over subjects.
Writes results/sensitivity.csv and figures/fig6_sensitivity.png.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from analyze import layer_scores, load_model_results
from stats import boot_ci, loso_best

VARIANTS = [  # (label, results folder, model-dir suffix)
    ("grid 1\nPCA 1000", "main", "_g1"),
    ("grid 2\nPCA 1000 (main)", "main", ""),
    ("grid 4\nPCA 1000", "main", "_g4"),
    ("grid 2\nPCA 500", "main_pca500", ""),
]
MODELS = ("resnet50", "vit_b16_grid")


def restricted_scores(res, layers, window):
    keep = [i for i, l in enumerate(res["layers"]) if l in layers]
    norm, _ = layer_scores(res, window)
    sc, idx = loso_best(norm[:, keep])
    return sc, [res["layers"][keep[i]] for i in idx], res["subs"]


def main():
    rows = []
    for wname, win in config.WINDOWS.items():
        for label, folder, suffix in VARIANTS:
            got = {}
            for m in MODELS:
                res = load_model_results(m + suffix, folder)
                if res is None:
                    break
                got[m] = restricted_scores(res, config.SENS_LAYERS[m], win)
            if len(got) < 2:
                print(f"[skip] {label!r}: results missing")
                continue
            common = sorted(set(got["resnet50"][2]) & set(got["vit_b16_grid"][2]))
            ia = [got["resnet50"][2].index(s) for s in common]
            ib = [got["vit_b16_grid"][2].index(s) for s in common]
            a, b = got["resnet50"][0][ia], got["vit_b16_grid"][0][ib]
            d = a - b
            lo, hi = boot_ci(d)
            rows.append(dict(window=wname, variant=label.replace("\n", " "), resnet50=a.mean(), vit_grid=b.mean(),
                             diff=d.mean(), ci_lo=lo, ci_hi=hi, n_subjects=len(common),
                             layers_resnet=sorted(set(got["resnet50"][1])), layers_vit=sorted(set(got["vit_b16_grid"][1]))))
            r = rows[-1]
            print(f"[{wname:<5}] {r['variant']:<26} ResNet {r['resnet50']:.3f}  ViT-grid {r['vit_grid']:.3f}  "
                  f"diff {r['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}]  layers {r['layers_resnet']} / {r['layers_vit']}")
    if not rows:
        raise SystemExit("No sensitivity results found -- see the commands in this file's docstring / README.")
    with open(config.RESULTS_DIR / "sensitivity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    config.FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, wname in zip(axes, config.WINDOWS):
        rr = [r for r in rows if r["window"] == wname]
        x = np.arange(len(rr))
        ax.errorbar(x, [r["diff"] for r in rr],
                    yerr=[[r["diff"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["diff"] for r in rr]],
                    fmt="o", color="k", capsize=4)
        ax.axhline(0, color="r", lw=0.8, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels([r["variant"].replace(" (main)", "\n(main)").replace(" PCA", "\nPCA") for r in rr], fontsize=7)
        ax.set_title(f"{wname} window")
    axes[0].set_ylabel("ResNet-50 minus ViT-B/16 (normalized r)\n95% CI over subjects")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig6_sensitivity.png", dpi=200)
    print("wrote results/sensitivity.csv and figures/fig6_sensitivity.png")


if __name__ == "__main__":
    main()
