"""Decoding views of the EEG <-> DNN relationship (complements the encoding analysis in fit.py).

    python3 decode.py                       # all three tasks, all subjects (~10-15 min)
    python3 decode.py --task ident          # or: retrieve, recon
    python3 decode.py --subjects 1 2 3

Tasks (all use the held-out 200 test images; 200-way chance = 0.5% top-1, 2.5% top-5):
  ident     Identification with the ENCODING models: predict EEG for all 200 test images from a layer's features,
            then match each observed EEG response to the closest prediction. Per layer, response window and number
            of averaged repetitions (10 vs 80).
  retrieve  EEG -> layer-feature decoding: ridge from the EEG response to a layer's (PCA) features, then retrieve the
            viewed image among the 200 test images by feature similarity. Saves similarity matrices for galleries.
  recon     Linear 16x16 RGB 'reconstructions' of the viewed image from (a) EEG in four equal-length (100 ms) time windows and
            (b) network layers (what each layer linearly carries). Honest, low-resolution illustration.

Outputs in results/decoding/: ident_scores.csv, retrieve_scores.csv, recon_scores.csv, retrieval_S.npz, recon.npz.
Uses the 17-channel preprocessed EEG, so numbers are NOT comparable with 63-channel deep decoders in the literature.
"""
import argparse
import csv
import json
import time

import numpy as np

import analyze
import config
import data
import fit

OUT = config.RESULTS_DIR / "decoding"

SOURCES = {  # models/layers used for ident + retrieve (grid-4 readouts); missing ones are skipped
    "resnet50_g4": ["conv1", "layer1_2", "layer2_3", "layer3_1", "layer3_3", "layer3_5"],
    "vit_b16_grid_g4": ["block01", "block03", "block05", "block07", "block09", "block12"],
    "resnet50_random_bn_g4": ["conv1", "layer1_2", "layer2_3", "layer3_1", "layer3_3", "layer3_5"],
    "lowlevel": ["gabor_all"],
}
RECON_SOURCES = {
    "resnet50_g4": ["conv1", "layer2_3", "layer3_5"],
    "vit_b16_grid_g4": ["block01", "block05", "block12"],
}
# four EQUAL-length (100 ms = 10 samples) windows so that window length does not confound the comparison
RECON_WINDOWS = {"50-150 ms": (0.045, 0.145), "150-250 ms": (0.145, 0.245),
                 "250-350 ms": (0.245, 0.345), "350-450 ms": (0.345, 0.445)}
IDENT_REPS = (10, 80)   # number of test repetitions averaged before matching
N_DRAWS = 5             # random repetition subsets averaged for reps < 80
K_PCS = 256             # leading feature PCs used as decoding targets in 'retrieve'


# ----------------------------------------------------------------------------- helpers
def available(model, layers):
    p = config.FEATURE_DIR / model / "meta.json"
    if not p.exists():
        print(f"[skip] {model}: no features found")
        return []
    have = json.load(open(p))["layers"]
    return [l for l in layers if l in have]


def concept_split():
    """Same concept-level fit/validation split as fit.py."""
    _, cid = data.load_stimuli("train")
    rng = np.random.default_rng(config.SEED)
    concepts = np.unique(cid)
    val_c = rng.choice(concepts, int(round(config.VAL_FRACTION * len(concepts))), replace=False)
    val = np.isin(cid, val_c)
    return np.where(~val)[0], np.where(val)[0]


def load_all(subs):
    D, times, ch = {}, None, None
    for s in subs:
        tr, times, ch = data.load_eeg(s, "train")
        te, _, _ = data.load_eeg(s, "test")
        rng = np.random.default_rng(1000 + s)
        reps = {}
        for r in IDENT_REPS:
            if r >= te.shape[1]:
                reps[r] = [te.mean(1).reshape(len(te), -1)]
            else:
                reps[r] = [te[:, rng.permutation(te.shape[1])[:r]].mean(1).reshape(len(te), -1) for _ in range(N_DRAWS)]
        D[s] = dict(Ytr=tr.mean(1).reshape(len(tr), -1), Yte=te.mean(1).reshape(len(te), -1), reps=reps)
        del tr, te
        print(f"loaded sub-{s:02d}")
    return D, times, ch


def sim_matrix(query, gallery):
    """Cosine similarity between rows after removing the mean across items (kills the common ERP component)."""
    q = query - query.mean(0)
    g = gallery - gallery.mean(0)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)
    return q @ g.T


def retrieval_metrics(S):
    """S[i, j]: similarity of query i to candidate j; the correct candidate for query i is j = i."""
    n = len(S)
    rank = (S > np.diag(S)[:, None]).sum(1)  # 0 = best
    return dict(top1=float((rank == 0).mean()), top5=float((rank < 5).mean()), pct=float(1 - rank.mean() / (n - 1))), rank


def flat_window_mask(times, n_ch, window):
    return np.tile(analyze.window_mask(times, window), n_ch)  # flatten order is (channel, time)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


# ----------------------------------------------------------------------------- task 1: identification via encoders
def task_ident(D, subs, times, n_ch, tr_idx, val_idx):
    masks = {w: flat_window_mask(times, n_ch, win) for w, win in config.WINDOWS.items()}
    rows = []
    for model, layers in SOURCES.items():
        for layer in available(model, layers):
            t0 = time.time()
            Ztr, Zte = fit.get_pca(model, layer)
            prep_tr, prep_full = fit.RidgePrep(Ztr[tr_idx]), fit.RidgePrep(Ztr)
            for s in subs:
                _, _, _, P = fit.fit_layer_subject(prep_tr, prep_full, Ztr, Zte, D[s]["Ytr"], D[s]["Yte"], tr_idx, val_idx)
                for w, mask in masks.items():
                    for r, obs_list in D[s]["reps"].items():
                        ms = [retrieval_metrics(sim_matrix(o[:, mask], P[:, mask]))[0] for o in obs_list]
                        rows.append(dict(model=model, layer=layer, window=w, nreps=r, subject=s,
                                         top1=np.mean([m["top1"] for m in ms]), top5=np.mean([m["top5"] for m in ms]),
                                         pct=np.mean([m["pct"] for m in ms])))
            sel = [r for r in rows if r["model"] == model and r["layer"] == layer and r["window"] == "all" and r["nreps"] == max(IDENT_REPS)]
            print(f"[ident] {model:<22}{layer:<10} top-1 (all window, {max(IDENT_REPS)} reps) = "
                  f"{100 * np.mean([r['top1'] for r in sel]):5.1f}%   ({time.time() - t0:.0f}s)")
    write_csv(OUT / "ident_scores.csv", rows)


# ----------------------------------------------------------------------------- task 2: EEG -> features, retrieval
def task_retrieve(D, subs, tr_idx, val_idx):
    targets = {}
    for model, layers in SOURCES.items():
        for layer in available(model, layers):
            Ztr, Zte = fit.get_pca(model, layer)
            k = min(K_PCS, Ztr.shape[1])
            targets[(model, layer)] = (Ztr[:, :k].copy(), Zte[:, :k].copy())
    rows, Sall = [], {key: [] for key in targets}
    for s in subs:
        t0 = time.time()
        Ytr, Yte = D[s]["Ytr"], D[s]["Yte"]
        mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
        Xtr, Xte = ((Ytr - mu) / sd).astype(np.float32), ((Yte - mu) / sd).astype(np.float32)
        prep_tr, prep_full = fit.RidgePrep(Xtr[tr_idx]), fit.RidgePrep(Xtr)
        for (model, layer), (Ttr, Tte) in targets.items():
            _, _, _, P = fit.fit_layer_subject(prep_tr, prep_full, Xtr, Xte, Ttr, Tte, tr_idx, val_idx)
            S = sim_matrix(P, Tte)
            m, _ = retrieval_metrics(S)
            rows.append(dict(model=model, layer=layer, window="-", nreps=max(IDENT_REPS), subject=s, **m))
            Sall[(model, layer)].append(S.astype(np.float32))
        del prep_tr, prep_full
        print(f"[retrieve] sub-{s:02d} done ({time.time() - t0:.0f}s)")
    for (model, layer) in targets:
        sel = [r for r in rows if r["model"] == model and r["layer"] == layer]
        print(f"[retrieve] {model:<22}{layer:<10} top-1 = {100 * np.mean([r['top1'] for r in sel]):5.1f}%  "
              f"top-5 = {100 * np.mean([r['top5'] for r in sel]):5.1f}%")
    write_csv(OUT / "retrieve_scores.csv", rows)
    np.savez_compressed(OUT / "retrieval_S.npz", subjects=np.array(subs),
                        **{f"{m}|{l}": np.stack(v) for (m, l), v in Sall.items()})


# ----------------------------------------------------------------------------- task 3: 16x16 reconstructions
def load_pixels16():
    d = config.FEATURE_DIR / "pixels"
    sc = json.load(open(d / "meta.json"))["scales"]["rgb_16x16"]
    return (np.load(d / "rgb_16x16_train.npy").astype(np.float32) * sc,
            np.load(d / "rgb_16x16_test.npy").astype(np.float32) * sc)


def resid_corr(P, T, mean_img):
    """Per-image Pearson r between predicted and true pixels AFTER subtracting the training mean image."""
    a, b = P - mean_img, T - mean_img
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def task_recon(D, subs, times, n_ch, tr_idx, val_idx):
    PIXtr, PIXte = load_pixels16()
    mean_img = PIXtr.mean(0)
    rows, store = [], {}

    def add(source, detail, subject, r):
        rows.append(dict(source=source, detail=detail, subject=subject, r=float(r.mean()),
                         sem_img=float(r.std(ddof=1) / np.sqrt(len(r)))))

    # (a) EEG windows -> pixels
    for w, win in RECON_WINDOWS.items():
        mask = flat_window_mask(times, n_ch, win)
        preds = []
        for s in subs:
            Ytr, Yte = D[s]["Ytr"][:, mask], D[s]["Yte"][:, mask]
            mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-6
            Xtr, Xte = ((Ytr - mu) / sd).astype(np.float32), ((Yte - mu) / sd).astype(np.float32)
            prep_tr, prep_full = fit.RidgePrep(Xtr[tr_idx]), fit.RidgePrep(Xtr)
            _, _, _, P = fit.fit_layer_subject(prep_tr, prep_full, Xtr, Xte, PIXtr, PIXte, tr_idx, val_idx)
            preds.append(P.astype(np.float32))
            add("eeg", w, s, resid_corr(P, PIXte, mean_img))
        store[f"eeg|{w}"] = np.stack(preds)
        print(f"[recon] EEG {w:<11} mean r = {np.mean([r['r'] for r in rows if r['detail'] == w]):.3f}")

    # (b) network layers -> pixels (same for every subject)
    for model, layers in RECON_SOURCES.items():
        for layer in available(model, layers):
            Ztr, Zte = fit.get_pca(model, layer)
            prep_tr, prep_full = fit.RidgePrep(Ztr[tr_idx]), fit.RidgePrep(Ztr)
            _, _, _, P = fit.fit_layer_subject(prep_tr, prep_full, Ztr, Zte, PIXtr, PIXte, tr_idx, val_idx)
            store[f"dnn|{model}|{layer}"] = P.astype(np.float32)
            add(model, layer, -1, resid_corr(P, PIXte, mean_img))
            print(f"[recon] {model:<18}{layer:<10} mean r = {rows[-1]['r']:.3f}")
    write_csv(OUT / "recon_scores.csv", rows)
    np.savez_compressed(OUT / "recon.npz", subjects=np.array(subs), target=PIXte, mean_img=mean_img, **store)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", default=["ident", "retrieve", "recon"], choices=["ident", "retrieve", "recon"])
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    a = ap.parse_args()
    subs = a.subjects or data.find_subjects()
    OUT.mkdir(parents=True, exist_ok=True)
    tr_idx, val_idx = concept_split()
    D, times, ch = load_all(subs)
    n_ch = len(ch)
    if "ident" in a.task:
        task_ident(D, subs, times, n_ch, tr_idx, val_idx)
    if "retrieve" in a.task:
        task_retrieve(D, subs, tr_idx, val_idx)
    if "recon" in a.task:
        task_recon(D, subs, times, n_ch, tr_idx, val_idx)


if __name__ == "__main__":
    main()
