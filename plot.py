"""Make all figures from results/main (and results/shuffle if present).   python3 plot.py"""
import math
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from analyze import layer_scores, load_model_results, normalize

STYLE = {
    "resnet50": dict(color="tab:blue", label="ResNet-50 (ImageNet)"),
    "vit_b16": dict(color="tab:orange", label="ViT-B/16, mean-token (ImageNet)"),
    "vit_b16_grid": dict(color="tab:red", label="ViT-B/16, 2x2 grid (ImageNet)"),
    "resnet50_random": dict(color="0.65", label="ResNet-50 random init (default BN)", ls="--"),
    "resnet50_random_bn": dict(color="0.25", label="ResNet-50 random init (BN-calibrated)", ls="--"),
    "pixels": dict(color="tab:green", label="raw-pixel baseline (best resolution)", ls="-."),
    "lowlevel": dict(color="tab:purple", label="Gabor/Fourier baseline (best layer)", ls="-."),
}


def _kw(m):
    return {k: v for k, v in STYLE[m].items() if k != "label"}


def _load_all(name="main"):
    return {m: r for m in config.MODELS if (r := load_model_results(m, name)) is not None}


def _draw(ax, all_res, getter):
    """Layer curves vs relative depth (mean +/- SEM over subjects); non-network baselines as horizontal bands."""
    for m, res in all_res.items():
        v = getter(res)
        mu = v.mean(0)
        se = v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros_like(mu)
        if m in config.BASELINES:
            b = int(np.argmax(mu))
            ax.axhline(mu[b], label=STYLE[m]["label"], lw=1.5, **_kw(m))
            ax.axhspan(mu[b] - se[b], mu[b] + se[b], color=STYLE[m]["color"], alpha=0.15)
            continue
        x = np.linspace(0, 1, len(mu))
        ax.plot(x, mu, label=STYLE[m]["label"], lw=2, **_kw(m))
        ax.fill_between(x, mu - se, mu + se, color=STYLE[m]["color"], alpha=0.2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("relative layer depth (0 = input side, 1 = output)")


def fig_layer_curves(all_res, shuffle=None):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharex=True)
    _draw(axes[0], all_res, lambda r: layer_scores(r)[0])
    _draw(axes[1], all_res, lambda r: layer_scores(r)[1])
    if shuffle:
        first = True
        for m, res in shuffle.items():
            if m in config.BASELINES:
                continue
            raw = layer_scores(res)[1]
            axes[1].plot(np.linspace(0, 1, raw.shape[1]), raw.mean(0), color="k", lw=0.8, ls=":",
                         label="shuffled-EEG control" if first else None)
            first = False
    axes[0].set_title(f"Noise-ceiling-normalized predictivity\n(window {config.WINDOW[0]}-{config.WINDOW[1]} s; mean +/- SEM over subjects)")
    axes[1].set_title("Raw predictivity (Pearson r, held-out test images)")
    axes[0].set_ylabel("normalized r")
    axes[1].set_ylabel("r")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig1_layer_depth_vs_predictivity.png", dpi=200)
    plt.close(fig)


def fig_early_late(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True, sharex=True)
    for ax, w in zip(axes, ("early", "late")):
        _draw(ax, all_res, lambda r, w=w: layer_scores(r, config.WINDOWS[w])[0])
        ax.set_title(f"{w} window ({config.WINDOWS[w][0] * 1000:.0f}-{config.WINDOWS[w][1] * 1000:.0f} ms)")
    axes[0].set_ylabel("normalized r")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig5_early_vs_late_window.png", dpi=200)
    plt.close(fig)


def _layer_time_map(res):
    rn = normalize(res["r"], res["rel"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(rn, axis=(0, 2))  # (L, T)


def fig_heatmaps(all_res):
    ms = [m for m in all_res if m not in config.BASELINES]
    ncol = min(3, len(ms))
    nrow = math.ceil(len(ms) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), squeeze=False)
    maps = {m: _layer_time_map(all_res[m]) for m in ms}
    vmax = np.nanpercentile(np.concatenate([v.ravel() for v in maps.values()]), 99)
    im = None
    for k, ax in enumerate(axes.ravel()):
        if k >= len(ms):
            ax.axis("off")
            continue
        m = ms[k]
        res = all_res[m]
        t = res["times"]
        im = ax.imshow(maps[m], aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                       extent=[t[0], t[-1], -0.5, len(res["layers"]) - 0.5])
        ax.set_title(STYLE[m]["label"], fontsize=8)
        ax.set_xlabel("time from stimulus onset (s)")
        step = max(1, len(res["layers"]) // 8)
        ax.set_yticks(range(0, len(res["layers"]), step))
        ax.set_yticklabels([res["layers"][i] for i in range(0, len(res["layers"]), step)], fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="normalized r", shrink=0.9)
    fig.savefig(config.FIG_DIR / "fig2_layer_by_time_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ceiling_and_best(all_res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    first = next(iter(all_res.values()))
    t = first["times"]
    rel_ch = first["rel"].mean(1)  # (S, T)
    for row in rel_ch:
        axes[0].plot(t, row, color="0.7", lw=0.8)
    axes[0].plot(t, rel_ch.mean(0), color="k", lw=2)
    axes[0].axhline(config.REL_MIN, color="r", ls=":", lw=1)
    axes[0].set_title("EEG split-half reliability (mean over channels)\nthin = subjects, red = exclusion threshold")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("reliability")

    for i, (m, res) in enumerate(all_res.items()):
        norm, _ = layer_scores(res)
        best = norm.mean(0).argmax()
        v = norm[:, best]
        axes[1].bar(i, v.mean(), color=STYLE[m]["color"], alpha=0.8)
        axes[1].scatter(np.full(len(v), i), v, color="k", s=8, zorder=3)
        axes[1].text(i, v.max() * 1.03, res["layers"][best], ha="center", fontsize=6, rotation=45)
    axes[1].set_xticks(range(len(all_res)))
    axes[1].set_xticklabels([STYLE[m]["label"].replace(" (", "\n(").replace(", ", ",\n") for m in all_res], fontsize=5.5)
    axes[1].set_title("Best layer per model (dots = subjects; layer picked on the same data -> see stats.py for LOSO)", fontsize=8)
    axes[1].set_ylabel("normalized r")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig3_noise_ceiling_and_best_layer.png", dpi=200)
    plt.close(fig)


def fig_peak_latency(all_res):
    fig, ax = plt.subplots(figsize=(5.8, 4))
    for m, res in all_res.items():
        if m in config.BASELINES:
            continue
        mp = _layer_time_map(res)
        t = res["times"]
        ok = t >= 0.0
        peak = t[ok][np.argmax(np.where(np.isnan(mp[:, ok]), -np.inf, mp[:, ok]), axis=1)]
        ax.plot(np.linspace(0, 1, len(peak)), peak * 1000, marker="o", ms=3, label=STYLE[m]["label"], **_kw(m))
    ax.set_xlabel("relative layer depth"); ax.set_ylabel("time of peak predictivity (ms)")
    ax.set_title("Latency of peak EEG predictivity vs layer depth\n(10 ms sampling: differences of 1-2 bins are not meaningful)", fontsize=9)
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "fig4_peak_latency_vs_depth.png", dpi=200)
    plt.close(fig)


def main():
    config.FIG_DIR.mkdir(exist_ok=True)
    all_res = _load_all("main")
    if not all_res:
        raise SystemExit("No results found in results/main -- run fit.py first.")
    shuffle = _load_all("shuffle") or None
    fig_layer_curves(all_res, shuffle)
    fig_early_late(all_res)
    fig_heatmaps(all_res)
    fig_ceiling_and_best(all_res)
    fig_peak_latency(all_res)
    print("figures written to", config.FIG_DIR)


if __name__ == "__main__":
    main()
