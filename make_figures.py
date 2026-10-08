"""Final, website-ready figures (PNG 300 dpi + editable SVG) and captions.

    python3 make_figures.py          # -> figures/final/*.png, *.svg, captions.md

Figure A  layer profiles by response window (+ peak-over-time), best readout (grid 4), with non-learned baselines
Figure B  layer x time heatmaps (when does each layer predict EEG?)
Figure C  effect sizes (forest plot) with 95% CIs over subjects AND test images
Figure D  readout sensitivity (pooling grid 1/2/4, PCA 500) for ResNet-50 vs ViT-B/16
Uses results/main (grid-4 folders if present, else grid-2), results/stats_*.csv and results/sensitivity.csv.
"""
import csv
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from analyze import layer_scores, load_model_results, normalize, peak_scores

OUT = config.FIG_DIR / "final"
C = dict(resnet="#0072B2", vit="#D55E00", rand="#6B6B6B", gabor="#CC79A7", pix="#009E73")
WIN_COL = dict(all="#222222", early="#56B4E9", late="#E69F00")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "savefig.dpi": 300, "svg.fonttype": "none",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


RESNET_ORDER = (["conv1"] + [f"layer{li}_{bi}" for li, n in ((1, 3), (2, 4), (3, 6), (4, 3)) for bi in range(n)]
                + ["avgpool", "fc"])


def depth_x(layers):
    """True relative depth in the FULL network: ViT block n -> (n-1)/11; ResNet layers by position among 19 stages."""
    out = []
    for l in layers:
        if l.startswith("block"):
            out.append((int(l[5:]) - 1) / 11)
        elif l in RESNET_ORDER:
            out.append(RESNET_ORDER.index(l) / (len(RESNET_ORDER) - 1))
        else:
            out.append(np.nan)
    return np.array(out)


def _load(preferred, fallback):
    r = load_model_results(preferred, "main")
    if r is not None:
        return r, preferred
    r = load_model_results(fallback, "main")
    if r is not None:
        print(f"[note] {preferred} not found, using {fallback}")
    return r, fallback


def _ro(name):
    return "4x4 grid" if name.endswith("_g4") else "2x2 grid"


def _best(res, getter):
    v = getter(res).mean(0)
    return float(v.max())


# ---------------------------------------------------------------------------------------------- Figure A
def fig_profiles():
    rn, rn_name = _load("resnet50_g4", "resnet50")
    vt, vt_name = _load("vit_b16_grid_g4", "vit_b16_grid")
    rb, rb_name = _load("resnet50_random_bn_g4", "resnet50_random_bn")
    gb = load_model_results("lowlevel", "main")
    px = load_model_results("pixels", "main")
    if rn is None or vt is None:
        print("[skip] Figure A: ResNet/ViT results missing")
        return
    panels = [
        ("Early response (50-200 ms)", "normalized r", lambda r: layer_scores(r, config.WINDOWS["early"])[0]),
        ("Late response (200-500 ms)", "normalized r", lambda r: layer_scores(r, config.WINDOWS["late"])[0]),
        ("Peak over time", "raw r (channel mean)", peak_scores),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (title, ylab, getter) in zip(axes, panels):
        for res, color, dy in ((rn, C["resnet"], 7), (vt, C["vit"], -13)):
            v = getter(res)
            mu, se = v.mean(0), v.std(0, ddof=1) / np.sqrt(len(v))
            x = depth_x(res["layers"])
            ax.fill_between(x, mu - se, mu + se, color=color, alpha=0.2, lw=0)
            ax.plot(x, mu, color=color, lw=2)
            b = int(np.argmax(mu))
            ax.plot(x[b], mu[b], "o", color=color, ms=6, mec="white", mew=1, zorder=5)
            ax.annotate(res["layers"][b], (x[b], mu[b]), xytext=(0, dy), textcoords="offset points",
                        ha="center", fontsize=7, color=color,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
        for res, color, lab in ((rb, C["rand"], "untrained ResNet-50 (best layer)"),
                                (gb, C["gabor"], "Gabor/Fourier baseline")):
            if res is None:
                continue
            y = _best(res, getter)
            ax.axhline(y, color=color, ls="--", lw=1.2)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylab)
        ax.set_xlabel("relative depth in the full network")
        ax.set_xlim(-0.03, 1.03)
        ax.margins(y=0.12)
    handles = [plt.Line2D([], [], color=C["resnet"], lw=2, label=f"ResNet-50, ImageNet ({_ro(rn_name)} readout)"),
               plt.Line2D([], [], color=C["vit"], lw=2, label=f"ViT-B/16, ImageNet ({_ro(vt_name)} readout)"),
               plt.Line2D([], [], color=C["rand"], lw=1.2, ls="--", label=f"untrained ResNet-50 ({_ro(rb_name)} readout)"),
               plt.Line2D([], [], color=C["gabor"], lw=1.2, ls="--", label="Gabor/Fourier baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
    foot = f"Mean +/- SEM over {len(rn['subs'])} subjects. ResNet-50 curve ends at layer3_5 (layer4, avgpool, fc not extracted at this readout); ViT-B/16: 8 of 12 blocks. "
    if px is not None:
        foot += "Raw-pixel baseline (off scale): " + " / ".join(f"{_best(px, g):.2f}" for _, _, g in panels) + "."
    fig.text(0.5, -0.13, foot, ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    save(fig, "figA_layer_profiles_by_window")


# ---------------------------------------------------------------------------------------------- Figure B
def _time_map(res):
    rn_ = normalize(res["r"], res["rel"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(rn_, axis=(0, 2))  # (L, T)


def fig_heatmaps():
    rn, rn_name = _load("resnet50_g4", "resnet50")
    vt, vt_name = _load("vit_b16_grid_g4", "vit_b16_grid")
    rb, rb_name = _load("resnet50_random_bn_g4", "resnet50_random_bn")
    items = [(f"ResNet-50, ImageNet\n({_ro(rn_name)} readout)", rn), (f"ViT-B/16, ImageNet\n({_ro(vt_name)} readout)", vt),
             (f"Untrained ResNet-50\n({_ro(rb_name)} readout)", rb)]
    items = [(t, r) for t, r in items if r is not None]
    if not items:
        print("[skip] Figure B")
        return
    maps = [_time_map(r) for _, r in items]
    vmax = np.nanpercentile(np.concatenate([m.ravel() for m in maps]), 99)
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items) + 0.8, 4.2), squeeze=False,
                             gridspec_kw={"wspace": 0.42})
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = None
    for ax, (title, res), M in zip(axes[0], items, maps):
        t = res["times"] * 1000
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                       extent=[t[0], t[-1], -0.5, len(res["layers"]) - 0.5], interpolation="nearest")
        for xv in (0, 200):
            ax.axvline(xv, color="white", ls=":", lw=0.9)
        step = 1 if len(res["layers"]) <= 14 else 2
        ax.set_yticks(range(0, len(res["layers"]), step))
        ax.set_yticklabels([res["layers"][i] for i in range(0, len(res["layers"]), step)], fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("time from stimulus onset (ms)")
        ax.grid(False)
    axes[0][0].set_ylabel("layer (input -> output)")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cb.set_label("noise-ceiling-normalized r")
    fig.text(0.5, -0.04, "Mean over subjects and 17 channels. Dotted lines: stimulus onset and the next image in the 5 Hz stream (200 ms). Grey: EEG reliability below threshold (excluded).",
             ha="center", fontsize=7, color="0.35")
    save(fig, "figB_layer_by_time")


# ---------------------------------------------------------------------------------------------- Figure C
ROWS = [
    ("trained ResNet-50  vs  untrained (BN-calibrated)", "trained vs BN-calibrated random ResNet-50"),
    ("trained ResNet-50  vs  Gabor/Fourier baseline", "ResNet-50 vs Gabor/Fourier baseline"),
    ("untrained ResNet-50  vs  Gabor/Fourier baseline", "BN-calibrated random ResNet-50 vs Gabor/Fourier baseline"),
    ("ResNet-50  vs  ViT-B/16 (matched 2x2 readout)", "ResNet-50 vs ViT-B/16 (matched 2x2 readout)"),
    ("ViT-B/16: 2x2 readout  vs  mean-token readout", "ViT grid vs mean-token readout"),
]


def _read_csv(p):
    return list(csv.DictReader(open(p))) if p.exists() else []


def fig_effects():
    stats = {w: {r["comparison"]: r for r in _read_csv(config.RESULTS_DIR / f"stats_{w}.csv")} for w in config.WINDOWS}
    if not any(stats.values()):
        print("[skip] Figure C: run stats.py first")
        return
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    offs = {"early": -0.24, "late": 0.0, "all": 0.24}
    for i, (label, key) in enumerate(ROWS):
        for w, off in offs.items():
            r = stats[w].get(key)
            if r is None:
                continue
            d, lo, hi = float(r["diff"]), float(r["ci_both_lo"]), float(r["ci_both_hi"])
            ax.errorbar(d, i + off, xerr=[[d - lo], [hi - d]], fmt="o", color=WIN_COL[w], ms=5, capsize=2.5, lw=1.4,
                        label={"early": "early 50-200 ms", "late": "late 200-500 ms", "all": "all 50-400 ms"}[w] if i == 0 else None)
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(2.5, color="0.75", lw=0.8)  # learned-vs-not comparisons above, architecture/readout below
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS])
    ax.invert_yaxis()
    ax.set_xlabel("difference in normalized r  (95% CI over subjects and test images)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0)
    ax.text(0, 1.02, "Effect sizes (2x2 readout; each model at its leave-one-subject-out best layer)", transform=ax.transAxes,
            fontsize=8, color="0.35")
    fig.tight_layout()
    save(fig, "figC_effect_sizes")


# ---------------------------------------------------------------------------------------------- Figure D
def fig_readout():
    rows = _read_csv(config.RESULTS_DIR / "sensitivity.csv")
    if not rows:
        print("[skip] Figure D: run sensitivity.py first")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), sharey=False)
    for ax, (w, title) in zip(axes, (("early", "Early (50-200 ms)"), ("late", "Late (200-500 ms)"), ("all", "All (50-400 ms)"))):
        rr = [r for r in rows if r["window"] == w]
        grid = [(int(re.search(r"grid (\d+)", r["variant"]).group(1)), r) for r in rr if "PCA 1000" in r["variant"]]
        grid.sort(key=lambda t: t[0])
        x = [g for g, _ in grid]
        for key, color, lab in (("resnet50", C["resnet"], "ResNet-50"), ("vit_grid", C["vit"], "ViT-B/16")):
            ax.plot(x, [float(r[key]) for _, r in grid], "-o", color=color, lw=2, ms=5, label=lab)
            p5 = [r for r in rr if "PCA 500" in r["variant"]]
            if p5:
                ax.plot([2.08], [float(p5[0][key])], "o", mfc="white", mec=color, ms=6, label=None)
        ys = [float(r[k]) for _, r in grid for k in ("resnet50", "vit_grid")]
        mid, half = (max(ys) + min(ys)) / 2, max((max(ys) - min(ys)) / 2 * 1.5, 0.05)
        ax.set_ylim(mid - half, mid + half)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4])
        ax.set_xticklabels(["1x1", "2x2", "4x4"])
        ax.set_xlabel("spatial pooling grid")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.minorticks_off()
    axes[0].set_ylabel("normalized r (best layer, LOSO)")
    axes[0].legend(frameon=False)
    fig.text(0.5, -0.04, "Layers restricted to a common set around each model's peak. Hollow markers: PCA 500 instead of 1000 (2x2 grid).",
             ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    save(fig, "figD_readout_sensitivity")


CAPTIONS = """# Figure captions (draft)

**Figure A. Predicting EEG from deep-network layers.** For each layer of ImageNet-trained ResNet-50 and ViT-B/16, a ridge
regression maps activations to the EEG response to the same image (THINGS-EEG2, 9 subjects, 17 channels); scores are
Pearson r on 200 held-out test images, divided by the square root of the EEG split-half reliability (early and late panels)
or raw r maximised over time (right). Dots mark each model's best layer. Dashed lines: best layer of an untrained ResNet-50
with calibrated BatchNorm and of a non-learned Gabor-energy/Fourier-power baseline. Early layers give the sharpest early
peak; later layers give the strongest sustained (late) prediction.

**Figure B. When does each layer predict the EEG?** Noise-ceiling-normalized predictivity as a function of layer and time.
Dotted lines: stimulus onset and the onset of the next image in the 5 Hz stream.

**Figure C. Effect sizes.** Differences in normalized r between model pairs, each at its leave-one-subject-out best layer.
Error bars are 95% CIs from a two-way bootstrap over subjects and test images. Trained networks exceed untrained and
low-level baselines, most clearly after 200 ms; an untrained network is close to a Gabor/Fourier baseline; ResNet-50 and
ViT-B/16 differ little once the readout is matched.

**Figure D. Readout sensitivity.** Finer spatial pooling raises both models' scores; the ResNet-50 vs ViT-B/16 difference is
small throughout and changes sign in the early window at the finest grid. PCA size has negligible effect.
"""


def main():
    fig_profiles()
    fig_heatmaps()
    fig_effects()
    fig_readout()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "captions.md").write_text(CAPTIONS)
    print("wrote", OUT / "captions.md")


if __name__ == "__main__":
    main()
