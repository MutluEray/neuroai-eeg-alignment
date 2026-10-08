"""Ridge-regress each model layer's (PCA-reduced) activations onto EEG, score on the held-out 200 test images.

    python3 fit.py                                  # all models, all available subjects
    python3 fit.py --models resnet50 --subjects 1 2 # subset
    python3 fit.py --shuffle                        # negative control: training EEG rows permuted (-> results/shuffle)
    python3 fit.py --pca 500 --models resnet50 vit_b16_grid --layers layer3_5 block07   # sensitivity (-> results/main_pca500)

Protocol (per subject, model, layer):
  * train targets  = 16,540 training images x (17 ch x 100 t), 4 repetitions averaged
  * features       = layer activations -> centered -> PCA (<=PCA_COMPONENTS comps, fitted on train images only)
  * ridge penalty  = chosen on a held-out 10% of training CONCEPTS (no concept appears in both fit and validation)
  * final fit on all training images; score = Pearson r across the 200 test images per (channel, time point)
  * noise ceiling  = split-half reliability (Spearman-Brown corrected) of the 80-repetition test ERPs
  * image bootstrap: N_BOOT resamples of the 200 test images (identical resamples for every model/subject) ->
                     normalized layer scores per time window, stored as `boot` for paired image-level CIs (stats.py)
Output: results/<main|shuffle>[_pcaN]/<model>/sub-XX.npz
"""
import argparse
import json
import time
import warnings

import numpy as np

import analyze
import config
import data


# ----------------------------------------------------------------------------- stats helpers
def corr_cols(P, Y):
    """Pearson r between matching columns of P and Y (rows = observations)."""
    Pc = P - P.mean(0)
    Yc = Y - Y.mean(0)
    num = (Pc * Yc).sum(0)
    den = np.sqrt((Pc ** 2).sum(0) * (Yc ** 2).sum(0))
    return num / np.maximum(den, 1e-12)


def split_half_reliability(test_eeg, n_splits=50, seed=0):
    """test_eeg: (images, reps, ch, time). Returns Spearman-Brown corrected split-half reliability (ch, time)."""
    rng = np.random.default_rng(seed)
    n = test_eeg.shape[1]
    rs = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a = test_eeg[:, perm[: n // 2]].mean(1)
        b = test_eeg[:, perm[n // 2:]].mean(1)
        c, t = a.shape[1:]
        rs.append(corr_cols(a.reshape(len(a), -1), b.reshape(len(b), -1)).reshape(c, t))
    r = np.mean(rs, 0)
    return np.clip(2 * r / (1 + r), 0, 1)


def make_boot_weights(n=200):
    """(N_BOOT, n) resampling weights (rows sum to 1); identical for every model/subject (fixed seed)."""
    rng = np.random.default_rng(config.BOOT_SEED)
    idx = rng.integers(0, n, (config.N_BOOT, n))
    W = np.stack([np.bincount(i, minlength=n) for i in idx]).astype(np.float64)
    return W / n


def boot_scores(P, Yte, rel, times, W):
    """Image-bootstrap of the normalized layer score. P, Yte: (n_img, C*T); rel: (C, T); W: (B, n_img).
    Returns (n_windows, B): mean over channels and window time points of r / sqrt(reliability)."""
    P = P.astype(np.float64)
    Y = Yte.astype(np.float64)
    mp, my = W @ P, W @ Y
    cov = W @ (P * Y) - mp * my
    vp = W @ (P * P) - mp ** 2
    vy = W @ (Y * Y) - my ** 2
    r = (cov / np.sqrt(np.maximum(vp * vy, 1e-24))).reshape(len(W), *rel.shape)   # (B, C, T)
    rn = np.where(rel >= config.REL_MIN, r / np.sqrt(np.maximum(rel, 1e-6)), np.nan)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for win in config.WINDOWS.values():
            m = analyze.window_mask(times, win)
            out.append(np.nanmean(rn[:, :, m], axis=(1, 2)))
    return np.stack(out).astype(np.float32)


# ----------------------------------------------------------------------------- features
def _meta(model):
    return json.load(open(config.FEATURE_DIR / model / "meta.json"))


def get_pca(model, layer):
    """Cached, subject-independent PCA scores (fit on the 16,540 training images only)."""
    k = config.PCA_COMPONENTS
    cache = config.FEATURE_DIR / model / ("pca" if k == 1000 else f"pca{k}") / f"{layer}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["train"], z["test"]
    d = config.FEATURE_DIR / model
    Xtr = np.load(d / f"{layer}_train.npy").astype(np.float32)
    Xte = np.load(d / f"{layer}_test.npy").astype(np.float32)
    mu = Xtr.mean(0)
    Xtr -= mu
    Xte -= mu
    if Xtr.shape[1] > k:
        import torch  # lazy import
        _, _, V = torch.pca_lowrank(torch.from_numpy(Xtr), q=k, center=False, niter=4)
        V = V.numpy()
        Xtr, Xte = Xtr @ V, Xte @ V
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, train=Xtr, test=Xte)
    return Xtr, Xte


# ----------------------------------------------------------------------------- ridge
class RidgePrep:
    """Eigendecomposition of the (centered) Gram matrix X^T X so ridge for many alphas is cheap.
    alpha is relative: alpha_abs = alpha_rel * mean eigenvalue."""

    def __init__(self, X):
        self.xm = X.mean(0)
        self.Xc = (X - self.xm).astype(np.float64)
        lam, V = np.linalg.eigh(self.Xc.T @ self.Xc)
        self.lam = np.clip(lam, 0, None)
        self.V = V
        self.scale = max(self.lam.mean(), 1e-12)

    def coef_basis(self, Y):
        """Y: (n, outputs) raw targets. Returns (B, ym) with B = V^T X^T (Y - ym); X is centered so ym drops out."""
        return self.V.T @ (self.Xc.T @ Y), Y.mean(0)

    def predict_all_alphas(self, Xnew, B, ym, alphas_rel):
        Xp = (Xnew - self.xm) @ self.V
        for a in alphas_rel:
            yield Xp @ (B / (self.lam + a * self.scale)[:, None]) + ym

    def predict(self, Xnew, B, ym, alpha_rel):
        return next(self.predict_all_alphas(Xnew, B, ym, [alpha_rel]))


def fit_layer_subject(prep_tr, prep_full, Ztr, Zte, Ytr, Yte, tr_idx, val_idx):
    """Returns (test r per output, chosen alpha index, validation r, test predictions)."""
    alphas = config.ALPHA_REL_GRID
    # 1) choose alpha on held-out concepts
    B, ym = prep_tr.coef_basis(Ytr[tr_idx])
    Yval = Ytr[val_idx]
    val_r = [corr_cols(P, Yval).mean() for P in prep_tr.predict_all_alphas(Ztr[val_idx], B, ym, alphas)]
    ai = int(np.argmax(val_r))
    # 2) refit on all training images, score on test
    B, ym = prep_full.coef_basis(Ytr)
    P = prep_full.predict(Zte, B, ym, alphas[ai])
    return corr_cols(P, Yte), ai, float(val_r[ai]), P


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--subjects", nargs="+", type=int, default=None)
    ap.add_argument("--layers", nargs="+", default=None, help="restrict to these layer names")
    ap.add_argument("--shuffle", action="store_true", help="negative control: permute training EEG across images")
    ap.add_argument("--pca", type=int, default=None, help="number of PCA components (sensitivity analysis)")
    a = ap.parse_args()
    if a.pca:
        config.PCA_COMPONENTS = a.pca

    subs = a.subjects or data.find_subjects()
    print("subjects:", subs, "| PCA components:", config.PCA_COMPONENTS)
    out_name = ("shuffle" if a.shuffle else "main") + ("" if config.PCA_COMPONENTS == 1000 else f"_pca{config.PCA_COMPONENTS}")

    _, cid = data.load_stimuli("train")
    rng = np.random.default_rng(config.SEED)
    concepts = np.unique(cid)
    val_c = rng.choice(concepts, int(round(config.VAL_FRACTION * len(concepts))), replace=False)
    val_idx = np.where(np.isin(cid, val_c))[0]
    tr_idx = np.where(~np.isin(cid, val_c))[0]
    print(f"fit/val split: {len(tr_idx)} / {len(val_idx)} training images (split by concept)")
    Wboot = make_boot_weights(data.N_TEST)

    Ytr, Yte, REL, times, ch = {}, {}, {}, None, None
    for s in subs:
        tr, times, ch = data.load_eeg(s, "train")
        te, _, _ = data.load_eeg(s, "test")
        Y = tr.mean(1).reshape(len(tr), -1)
        if a.shuffle:
            Y = Y[np.random.default_rng(config.SEED + s).permutation(len(Y))]
        Ytr[s] = Y
        Yte[s] = te.mean(1).reshape(len(te), -1)
        REL[s] = split_half_reliability(te, config.N_SPLITS_NC, seed=s)
        del tr, te
        print(f"loaded sub-{s:02d}")
    n_ch, n_t = len(ch), len(times)
    win_mask = analyze.window_mask(times)

    for model in a.models:
        layers = [l for l in _meta(model)["layers"] if not a.layers or l in a.layers]
        if not layers:
            print(f"[{model}] no matching layers, skipping")
            continue
        R = {s: [] for s in subs}
        AI = {s: [] for s in subs}
        VR = {s: [] for s in subs}
        BT = {s: [] for s in subs}
        for layer in layers:
            t0 = time.time()
            Ztr, Zte = get_pca(model, layer)
            prep_tr, prep_full = RidgePrep(Ztr[tr_idx]), RidgePrep(Ztr)
            for s in subs:
                r, ai, vr, P = fit_layer_subject(prep_tr, prep_full, Ztr, Zte, Ytr[s], Yte[s], tr_idx, val_idx)
                R[s].append(r.reshape(n_ch, n_t)); AI[s].append(ai); VR[s].append(vr)
                BT[s].append(boot_scores(P, Yte[s], REL[s], times, Wboot))
            print(f"[{model}] {layer:<10} dim={Ztr.shape[1]:<5} {time.time() - t0:5.1f}s  "
                  f"mean test r (window, all subj) = {np.mean([R[s][-1][:, win_mask].mean() for s in subs]):.4f}")
        out = config.RESULTS_DIR / out_name / model
        out.mkdir(parents=True, exist_ok=True)
        for s in subs:
            np.savez(out / f"sub-{s:02d}.npz", layers=np.array(layers), r=np.stack(R[s]), rel=REL[s],
                     alpha_idx=np.array(AI[s]), val_r=np.array(VR[s]), times=times, ch_names=np.array(ch),
                     boot=np.stack(BT[s]), boot_windows=np.array(list(config.WINDOWS)))
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
