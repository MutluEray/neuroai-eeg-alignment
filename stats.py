"""Statistics for the key comparisons, over SUBJECTS and over test IMAGES.

    python3 stats.py

Layer choice is made WITHOUT the test subject (leave-one-subject-out): for each subject, the model's layer is the one
with the best mean score over the other subjects, and that layer's score for the held-out subject is used.

For every comparison (mean normalized-r difference between model A and model B) it reports:
  * subject bootstrap 95% CI (resampling subjects)                      -> generalisation to new subjects
  * image bootstrap 95% CI  (resampling the 200 test images, 1000x)     -> generalisation to new images
  * two-way 95% CI (resampling subjects AND images)                     -> both
  * exact two-sided sign-flip p over subjects (floor = 2/2^S), Holm-adjusted within each window
Layer-selection uncertainty is not propagated (layers are held at their leave-one-subject-out choices).
Outputs results/stats_<window>.csv.
"""
import csv
import itertools

import numpy as np

import config
from analyze import layer_scores, load_model_results

COMPARISONS = [
    ("resnet50", "resnet50_random_bn", "trained vs BN-calibrated random ResNet-50"),
    ("resnet50", "resnet50_random", "trained vs uncalibrated random ResNet-50"),
    ("resnet50_random_bn", "resnet50_random", "BN calibration effect (random ResNet-50)"),
    ("vit_b16_grid", "resnet50_random_bn", "trained ViT (grid) vs BN-calibrated random ResNet-50"),
    ("resnet50", "vit_b16_grid", "ResNet-50 vs ViT-B/16 (matched 2x2 readout)"),
    ("resnet50", "vit_b16", "ResNet-50 vs ViT-B/16 (mean-token readout)"),
    ("vit_b16_grid", "vit_b16", "ViT grid vs mean-token readout"),
    ("resnet50", "lowlevel", "ResNet-50 vs Gabor/Fourier baseline"),
    ("vit_b16_grid", "lowlevel", "ViT grid vs Gabor/Fourier baseline"),
    ("resnet50_random_bn", "lowlevel", "BN-calibrated random ResNet-50 vs Gabor/Fourier baseline"),
    ("lowlevel", "pixels", "Gabor/Fourier baseline vs raw pixels"),
]


def loso_best(norm):
    """norm (S, L) -> per-subject score at the layer chosen on the other subjects, and the chosen layer indices."""
    S = norm.shape[0]
    out, idx = np.empty(S), np.empty(S, int)
    for s in range(S):
        i = int(np.nanargmax(np.delete(norm, s, 0).mean(0)))
        idx[s], out[s] = i, norm[s, i]
    return out, idx


def sign_flip_p(d, n_random=200000, seed=0):
    S = len(d)
    if S <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=S)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_random, S))
    null = (signs * d).mean(1)
    return float((np.abs(null) >= abs(d.mean()) - 1e-12).mean())


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def main():
    results = {m: load_model_results(m, "main") for m in config.MODELS}
    results = {m: r for m, r in results.items() if r is not None}
    for wname, win in config.WINDOWS.items():
        picked = {}
        for m, res in results.items():
            norm, _ = layer_scores(res, win)
            sc, idx = loso_best(norm)
            boot = None
            if res["boot"] is not None and wname in res["boot_windows"]:
                wi = res["boot_windows"].index(wname)
                boot = np.stack([res["boot"][i, idx[i], wi] for i in range(len(idx))])  # (S, B)
            picked[m] = dict(subs=res["subs"], score=sc, boot=boot)
        rows = []
        for a, b, label in COMPARISONS:
            if a not in picked or b not in picked:
                continue
            common = sorted(set(picked[a]["subs"]) & set(picked[b]["subs"]))
            ia = [picked[a]["subs"].index(s) for s in common]
            ib = [picked[b]["subs"].index(s) for s in common]
            d = picked[a]["score"][ia] - picked[b]["score"][ib]
            lo, hi = boot_ci(d)
            row = dict(window=wname, comparison=label, mean_a=picked[a]["score"][ia].mean(),
                       mean_b=picked[b]["score"][ib].mean(), diff=d.mean(), ci_subj_lo=lo, ci_subj_hi=hi,
                       ci_img_lo=np.nan, ci_img_hi=np.nan, ci_both_lo=np.nan, ci_both_hi=np.nan,
                       p=sign_flip_p(d), n_subjects=len(common))
            if picked[a]["boot"] is not None and picked[b]["boot"] is not None:
                D = picked[a]["boot"][ia] - picked[b]["boot"][ib]          # (S, B)
                B = D.shape[1]
                row["ci_img_lo"], row["ci_img_hi"] = np.percentile(D.mean(0), [2.5, 97.5])
                sidx = np.random.default_rng(1).integers(0, len(common), (B, len(common)))
                both = D[sidx, np.arange(B)[:, None]].mean(1)
                row["ci_both_lo"], row["ci_both_hi"] = np.percentile(both, [2.5, 97.5])
            rows.append(row)
        if rows:
            adj = holm(np.array([r["p"] for r in rows]))
            for r, p_adj in zip(rows, adj):
                r["p_holm"] = p_adj
        print(f"\n=== window '{wname}' {win} s  (layers chosen leave-one-subject-out; normalized r) ===")
        for r in rows:
            print(f"{r['comparison']:<58} {r['mean_a']:.3f} vs {r['mean_b']:.3f}  diff {r['diff']:+.3f}  "
                  f"CI subj [{r['ci_subj_lo']:+.3f},{r['ci_subj_hi']:+.3f}]  img [{r['ci_img_lo']:+.3f},{r['ci_img_hi']:+.3f}]  "
                  f"both [{r['ci_both_lo']:+.3f},{r['ci_both_hi']:+.3f}]  p_holm={r['p_holm']:.4f}")
        if rows:
            with open(config.RESULTS_DIR / f"stats_{wname}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    print("\nLayer chosen per held-out subject (window 'all'):")
    for m, res in results.items():
        _, idx = loso_best(layer_scores(res, config.WINDOWS["all"])[0])
        print(f"  {m:<20}", [res["layers"][i] for i in idx])


if __name__ == "__main__":
    main()
