"""Figures for the decoding views (run decode.py first).

    python3 make_decoding_figures.py                 # example subject = first subject in the results
    python3 make_decoding_figures.py --subject 3

Figure E  200-way identification with the encoding models, by layer and response window (80 vs 10 repetitions)
Figure F  EEG -> feature-space retrieval: layer profile, and a gallery of randomly chosen examples
Figure G  16x16 linear reconstructions from EEG time windows vs from network layers
Writes PNG + SVG into figures/final/ and decoding_captions.md.
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
import data
import make_figures as mf  # shared style, palette, depth axis, save()

DEC = config.RESULTS_DIR / "decoding"
LAB = {"resnet50_g4": "ResNet-50, ImageNet", "vit_b16_grid_g4": "ViT-B/16, ImageNet",
       "resnet50_random_bn_g4": "untrained ResNet-50", "lowlevel": "Gabor/Fourier baseline"}
COL = {"resnet50_g4": mf.C["resnet"], "vit_b16_grid_g4": mf.C["vit"],
       "resnet50_random_bn_g4": mf.C["rand"], "lowlevel": mf.C["gabor"]}
IMNET_MEAN, IMNET_STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def read(path):
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if k not in ("model", "layer", "window", "source", "detail") else v
            except ValueError:
                pass
    return rows


def profile(rows, model, window, nreps, metric):
    """Layer order of first appearance; mean and SEM over subjects."""
    by, order = {}, []
    for r in rows:
        if r["model"] == model and r["window"] == window and int(r["nreps"]) == nreps:
            if r["layer"] not in by:
                by[r["layer"]], _ = [], order.append(r["layer"])
            by[r["layer"]].append(r[metric])
    mu = np.array([np.mean(by[l]) for l in order])
    se = np.array([np.std(by[l], ddof=1) / np.sqrt(len(by[l])) if len(by[l]) > 1 else 0 for l in order])
    return order, mu, se


def _baselines(ax, rows, window, nreps, metric, scale=100, ls="--"):
    for m in ("resnet50_random_bn_g4", "lowlevel"):
        _, mu, _ = profile(rows, m, window, nreps, metric)
        if len(mu):
            ax.axhline(scale * mu.max(), color=COL[m], ls=ls, lw=1.2 if ls == "--" else 1.0)


def _legend(fig, models, extra=()):
    h = [plt.Line2D([], [], color=COL[m], lw=2 if m in ("resnet50_g4", "vit_b16_grid_g4") else 1.2,
                    ls="-" if m in ("resnet50_g4", "vit_b16_grid_g4") else "--", label=LAB[m]) for m in models] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, -0.07))


# ---------------------------------------------------------------------------------------------- Figure E
def fig_ident():
    rows = read(DEC / "ident_scores.csv")
    if not rows:
        print("[skip] Figure E: run decode.py --task ident")
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=True)
    for ax, (w, title) in zip(axes, (("early", "Early response (50-200 ms)"), ("late", "Late response (200-500 ms)"),
                                     ("all", "All (50-400 ms)"))):
        for model in ("resnet50_g4", "vit_b16_grid_g4"):
            for nreps, ls, lw in ((80, "-", 2), (10, ":", 1.8)):
                layers, mu, se = profile(rows, model, w, nreps, "top1")
                if not layers:
                    continue
                x = mf.depth_x(layers)
                ax.plot(x, mu * 100, ls, color=COL[model], lw=lw)
                if nreps == 80:
                    ax.fill_between(x, (mu - se) * 100, (mu + se) * 100, color=COL[model], alpha=0.18, lw=0)
        _baselines(ax, rows, w, 80, "top1")
        _baselines(ax, rows, w, 10, "top1", ls=":")
        ax.axhline(0.5, color="k", lw=0.8, ls="-", alpha=0.5)
        ax.text(0.99, 0.5, "chance", ha="right", va="bottom", fontsize=7, color="0.3")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("relative depth in the full network")
        ax.set_xlim(-0.03, 1.03)
    axes[0].set_ylabel("200-way identification, top-1 (%)")
    _legend(fig, [m for m in COL if any(r["model"] == m for r in rows)],
            [plt.Line2D([], [], color="0.3", lw=2, label="80 repetitions averaged"),
             plt.Line2D([], [], color="0.3", lw=1.8, ls=":", label="10 repetitions averaged")])
    fig.text(0.5, -0.13, "Observed EEG matched to EEG predicted by each layer's encoding model, 200 held-out images, mean +/- SEM over subjects "
             "(17 channels). Baselines (untrained ResNet-50, Gabor/Fourier): dashed = 80 repetitions, dotted = 10 repetitions.", ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    mf.save(fig, "figE_identification")


# ---------------------------------------------------------------------------------------------- Figure F
def fig_retrieval_profile():
    rows = read(DEC / "retrieve_scores.csv")
    if not rows:
        print("[skip] Figure F: run decode.py --task retrieve")
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharex=True)
    for ax, (metric, chance, title) in zip(axes, (("top1", 0.5, "Top-1 (%)"), ("top5", 2.5, "Top-5 (%)"))):
        for model in ("resnet50_g4", "vit_b16_grid_g4"):
            layers, mu, se = profile(rows, model, "-", 80, metric)
            if not layers:
                continue
            x = mf.depth_x(layers)
            ax.plot(x, mu * 100, "-o", color=COL[model], lw=2, ms=4)
            ax.fill_between(x, (mu - se) * 100, (mu + se) * 100, color=COL[model], alpha=0.18, lw=0)
        _baselines(ax, rows, "-", 80, metric)
        ax.axhline(chance, color="k", lw=0.8, alpha=0.5)
        ax.text(0.99, chance, "chance", ha="right", va="bottom", fontsize=7, color="0.3")
        ax.set_title(f"EEG -> feature space: retrieval {title}", loc="left", fontweight="bold", fontsize=9)
        ax.set_xlabel("relative depth in the full network")
        ax.set_xlim(-0.03, 1.03)
    _legend(fig, [m for m in COL if any(r["model"] == m for r in rows)])
    fig.tight_layout()
    mf.save(fig, "figF1_retrieval_profile")


def _thumb(path, size=112):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)))


def fig_gallery(subject, model="resnet50_g4", n_ex=8, seed=3):
    rows = read(DEC / "retrieve_scores.csv")
    sfile = DEC / "retrieval_S.npz"
    if not rows or not sfile.exists():
        print("[skip] Figure F gallery")
        return
    layers, mu, _ = profile(rows, model, "-", 80, "top1")
    if not layers:
        print(f"[skip] gallery: no retrieval results for {model}")
        return
    layer = layers[int(np.argmax(mu))]
    z = np.load(sfile)
    subs = [int(s) for s in z["subjects"]]
    subject = subject if subject in subs else subs[0]
    S = z[f"{model}|{layer}"][subs.index(subject)]
    paths, _ = data.load_stimuli("test")
    rank = (S > np.diag(S)[:, None]).sum(1)
    top3 = np.argsort(-S, axis=1)[:, :3]
    idx = np.random.default_rng(seed).choice(len(S), n_ex, replace=False)

    fig, axes = plt.subplots(n_ex, 4, figsize=(5.6, 1.35 * n_ex + 0.6))
    for i, q in enumerate(idx):
        shown = [q] + list(top3[q])
        for j, k in enumerate(shown):
            ax = axes[i, j]
            ax.imshow(_thumb(paths[k]))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            edge = "#2ca02c" if (j > 0 and k == q) else ("#222222" if j == 0 else "#bbbbbb")
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(3 if edge == "#2ca02c" else 1.2)
            if j == 0:
                ax.text(0.04, 0.04, f"true image ranked #{rank[q] + 1}", transform=ax.transAxes, fontsize=6, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.7))
    for j, t in enumerate(["viewed image", "EEG-retrieved #1", "#2", "#3"]):
        axes[0, j].set_title(t, fontsize=8)
    t1 = np.mean([r["top1"] for r in rows if r["model"] == model and r["layer"] == layer])
    t5 = np.mean([r["top5"] for r in rows if r["model"] == model and r["layer"] == layer])
    fig.suptitle(f"EEG -> {LAB[model]} {layer} feature space (sub-{subject:02d}; 8 random test images, not cherry-picked)\n"
                 f"200-way retrieval over all subjects: top-1 {100 * t1:.1f}%, top-5 {100 * t5:.1f}% (chance 0.5% / 2.5%); "
                 "green = exact image", fontsize=8, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(top=0.9)
    mf.save(fig, "figF2_retrieval_gallery")


# ---------------------------------------------------------------------------------------------- Figure G
def _to_img(vec):
    x = vec.reshape(3, 16, 16).transpose(1, 2, 0) * IMNET_STD + IMNET_MEAN
    return np.clip(x, 0, 1)


def fig_recon(subject, n_ex=8, seed=5):
    rfile, rows = DEC / "recon.npz", read(DEC / "recon_scores.csv")
    if not rfile.exists() or not rows:
        print("[skip] Figure G: run decode.py --task recon")
        return
    z = np.load(rfile)
    subs = [int(s) for s in z["subjects"]]
    si = subs.index(subject) if subject in subs else 0
    subject = subs[si]
    target = z["target"]
    idx = np.random.default_rng(seed).choice(len(target), n_ex, replace=False)

    spec = [("Stimulus (16x16)", None, "#222222")]
    spec += [(f"EEG {k.split('|')[1]}", k, "#555555") for k in z.files if k.startswith("eeg|")]
    for model, layers, color in (("resnet50_g4", ("conv1", "layer2_3", "layer3_5"), mf.C["resnet"]),
                                 ("vit_b16_grid_g4", ("block01", "block05", "block12"), mf.C["vit"])):
        for l in layers:
            if f"dnn|{model}|{l}" in z.files:
                spec.append((f"{'ResNet-50' if model.startswith('resnet') else 'ViT-B/16'} {l}", f"dnn|{model}|{l}", color))

    fig = plt.figure(figsize=(13.5, 0.62 * len(spec) + 1.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1], wspace=0.28, top=0.93, bottom=0.08)
    sub = gs[0].subgridspec(len(spec), n_ex, wspace=0.04, hspace=0.06)
    for r, (label, key, color) in enumerate(spec):
        for c, k in enumerate(idx):
            ax = fig.add_subplot(sub[r, c])
            if key is None:
                vec = target[k]
            elif key.startswith("eeg|"):
                vec = z[key][si][k]
            else:
                vec = z[key][k]
            ax.imshow(_to_img(vec), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7.5, color=color, labelpad=6)
    axb = fig.add_subplot(gs[1])
    labels, vals, errs, cols = [], [], [], []
    for label, key, color in spec[1:]:
        if key.startswith("eeg|"):
            w = key.split("|")[1]
            per_sub = [r["r"] for r in rows if r["source"] == "eeg" and r["detail"] == w]
            vals.append(np.mean(per_sub)); errs.append(np.std(per_sub, ddof=1) / np.sqrt(len(per_sub)) if len(per_sub) > 1 else 0)
        else:
            _, model, l = key.split("|")
            rr = [r for r in rows if r["source"] == model and r["detail"] == l][0]
            vals.append(rr["r"]); errs.append(rr["sem_img"])
        labels.append(label); cols.append(color)
    y = np.arange(len(labels))
    axb.barh(y, vals, xerr=errs, color=cols, alpha=0.85, capsize=2)
    axb.set_yticks(y); axb.set_yticklabels(labels, fontsize=7.5); axb.invert_yaxis()
    axb.set_xlabel("reconstruction correlation beyond the mean image")
    axb.grid(axis="y", alpha=0)
    fig.suptitle(f"Linear 16x16 reconstructions of viewed images (sub-{subject:02d}, 8 random test images)", fontsize=10, y=0.985, x=0.4)
    fig.text(0.5, 0.005, "EEG rows: decoded from the stated 100 ms window. Network rows: decoded from network features (what each layer linearly "
             "carries). Bars: EEG mean +/- SEM over subjects; networks mean +/- SEM over test images.",
             ha="center", fontsize=7, color="0.35")
    mf.save(fig, "figG_reconstructions")


CAPTIONS = """# Decoding figure captions (draft)

**Figure E. Identifying the viewed image from EEG with the encoding models.** Each layer's encoding model predicts the EEG
response to all 200 held-out images; each observed response is matched to the most similar prediction (200-way, chance
0.5%). Solid lines: 80 test repetitions averaged; dotted: 10. Dashed: untrained ResNet-50 and Gabor/Fourier baselines.

**Figure F. Decoding EEG into network feature space.** (F1) A ridge model maps EEG to each layer's leading feature
components; the viewed image is retrieved among the 200 test images by feature similarity. (F2) Randomly chosen examples;
green frames mark the exact image.

**Figure G. What can be linearly recovered about the image?** 16x16 RGB reconstructions from EEG in four equal-length
(100 ms) time windows (upper rows) and from network layers (remaining rows). Networks show what each layer linearly carries
(an upper reference); EEG recovers only coarse luminance and colour layout. Temporal structure is analysed in the encoding
results (layer x time maps), not here.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1, help="subject used for the example galleries")
    a = ap.parse_args()
    fig_ident()
    fig_retrieval_profile()
    fig_gallery(a.subject)
    fig_recon(a.subject)
    (mf.OUT / "decoding_captions.md").write_text(CAPTIONS)
    print("wrote", mf.OUT / "decoding_captions.md")


if __name__ == "__main__":
    main()
