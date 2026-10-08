"""Aggregate results: noise-ceiling-normalized predictivity per layer / model / subject / time window.

    python3 analyze.py              # writes results/summary_main.csv and prints best layer per model
    python3 analyze.py --results shuffle
"""
import argparse
import csv
import warnings

import numpy as np

import config


def load_model_results(model, name="main"):
    d = config.RESULTS_DIR / name / model
    files = sorted(d.glob("sub-*.npz"))
    if not files:
        return None
    zs = [np.load(f, allow_pickle=True) for f in files]
    return dict(
        subs=[int(f.stem.split("-")[1]) for f in files],
        layers=[str(l) for l in zs[0]["layers"]],
        r=np.stack([z["r"] for z in zs]),        # (S, L, C, T)
        rel=np.stack([z["rel"] for z in zs]),    # (S, C, T)
        times=zs[0]["times"],
        ch=[str(c) for c in zs[0]["ch_names"]],
        boot=np.stack([z["boot"] for z in zs]) if "boot" in zs[0].files else None,   # (S, L, W, B) image-bootstrap scores
        boot_windows=[str(w) for w in zs[0]["boot_windows"]] if "boot_windows" in zs[0].files else None,
    )


def normalize(r, rel):
    """r / sqrt(reliability); points with reliability < REL_MIN -> NaN. r: (S,L,C,T), rel: (S,C,T)."""
    rb = rel[:, None]
    return np.where(rb >= config.REL_MIN, r / np.sqrt(np.maximum(rb, 1e-6)), np.nan)


def window_mask(times, window=None):
    w = window or config.WINDOW
    return (times >= w[0]) & (times <= w[1])


def layer_scores(res, window=None):
    """Returns (norm, raw), each (S, L): mean over channels and the window's time points."""
    w = window_mask(res["times"], window)
    rn = normalize(res["r"], res["rel"])[..., w]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm = np.nanmean(rn, axis=(2, 3))
    raw = res["r"][..., w].mean(axis=(2, 3))
    return norm, raw


def peak_scores(res):
    """(S, L): raw Pearson r, averaged over channels, maximised over post-stimulus time (t >= 0) -- the 'peak over time'
    metric used by several EEG-DNN papers (note: those correlate against split-half targets, so absolute values differ)."""
    m = res["times"] >= 0
    return res["r"][..., m].mean(axis=2).max(axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="main")
    ap.add_argument("--models", nargs="+", default=None, help="subset / extra model folders (e.g. resnet50_g4); writes summary_<results>_custom.csv")
    ap.add_argument("--table", action="store_true", help="print the per-layer profile (normalized r: all/early/late; peak-over-time raw r)")
    a = ap.parse_args()
    rows = []
    for model in (a.models or config.MODELS):
        res = load_model_results(model, a.results)
        if res is None:
            continue
        scores = {w: layer_scores(res, win) for w, win in config.WINDOWS.items()}
        norm, raw = scores["all"]
        S, L = norm.shape
        peak = peak_scores(res)
        sem = lambda x: x.std(0, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.zeros(x.shape[1])  # noqa: E731
        for i, l in enumerate(res["layers"]):
            row = dict(model=model, layer=l, depth=i / max(L - 1, 1), n_subjects=S, raw_mean=raw[:, i].mean(),
                       peak_r_mean=peak[:, i].mean())
            for w, (n_w, _) in scores.items():
                row[f"norm_{w}_mean"] = n_w[:, i].mean()
                row[f"norm_{w}_sem"] = sem(n_w)[i]
            rows.append(row)
        for w, (n_w, r_w) in scores.items():
            best = int(np.nanargmax(n_w.mean(0)))
            print(f"{model:<16} [{w:<5}] best layer: {res['layers'][best]:<10} normalized r = {n_w.mean(0)[best]:.3f} "
                  f"(raw {r_w.mean(0)[best]:.3f}), {S} subjects")
        if a.table:
            print(f"\n  {model}: per-layer profile (mean over subjects)")
            print(f"  {'layer':<12}{'all':>8}{'early':>8}{'late':>8}{'peak r':>9}")
            for i, l in enumerate(res["layers"]):
                print(f"  {l:<12}{scores['all'][0][:, i].mean():8.3f}{scores['early'][0][:, i].mean():8.3f}"
                      f"{scores['late'][0][:, i].mean():8.3f}{peak[:, i].mean():9.3f}")
            print()
        bp = int(np.argmax(peak.mean(0)))
        print(f"{model:<20} [peak ] best layer: {res['layers'][bp]:<10} peak-over-time raw r = {peak.mean(0)[bp]:.3f}")
    out = config.RESULTS_DIR / (f"summary_{a.results}_custom.csv" if a.models else f"summary_{a.results}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
