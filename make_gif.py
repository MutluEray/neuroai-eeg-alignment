"""Animated GIF of the decoding: EEG-based 16x16 reconstructions building up over time (needs decode.py --task recon).

    python3 make_gif.py                    # sub-01, 10 random test images
    python3 make_gif.py --subject 3 --n 12 --fps 2

For each test image: the stimulus, a time cursor sweeping 50-450 ms, and the linear EEG reconstruction from each 100 ms
window appearing as the cursor passes it. Right column: the best network layer (ResNet-50 layer3_5) for reference.
Images are random (fixed seed), not cherry-picked. Writes figures/final/decoding.gif (and .mp4 if imageio-ffmpeg is installed).
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
import data

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
WIN = ["50-150 ms", "150-250 ms", "250-350 ms", "350-450 ms"]


def img(v):
    return np.clip(v.reshape(3, 16, 16).transpose(1, 2, 0) * STD + MEAN, 0, 1)


def frame(stim, eeg, dnn, upto, k, n, subject):
    fig = plt.figure(figsize=(8, 3.6), dpi=90)
    gs = fig.add_gridspec(2, 6, height_ratios=[4, 1], hspace=0.25, wspace=0.12, left=0.03, right=0.97, top=0.82, bottom=0.08)
    panels = [("Viewed image", stim)] + [(w, eeg[i] if i < upto else None) for i, w in enumerate(WIN)] + [("ResNet-50\nlayer3_5", dnn)]
    for j, (t, v) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#bbbbbb")
        if v is None:
            ax.set_facecolor("#f2f2f2")
        else:
            ax.imshow(img(v), interpolation="nearest")
        ax.set_title(t, fontsize=8, color="#222" if v is not None else "#aaa")
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(50, 450); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("time after stimulus onset (ms)", fontsize=8)
    for i in range(4):
        ax.axvspan(50 + 100 * i, 150 + 100 * i, color="#0072B2" if i < upto else "#eeeeee", alpha=0.35 if i < upto else 1)
    ax.tick_params(labelsize=7)
    fig.suptitle(f"Linear reconstruction of a viewed image from EEG (sub-{subject:02d}, 17 channels) - image {k + 1}/{n}", fontsize=9)
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    z = np.load(config.RESULTS_DIR / "decoding" / "recon.npz")
    subs = [int(s) for s in z["subjects"]]
    si = subs.index(a.subject) if a.subject in subs else 0
    sub = subs[si]
    idx = np.random.default_rng(a.seed).choice(len(z["target"]), a.n, replace=False)
    frames, durs = [], []
    for k, q in enumerate(idx):
        eeg = [z[f"eeg|{w}"][si][q] for w in WIN]
        for upto in range(0, 5):
            frames.append(frame(z["target"][q], eeg, z["dnn|resnet50_g4|layer3_5"][q] if upto == 4 else None, upto, k, a.n, sub))
            durs.append(int(1000 / a.fps) if upto < 4 else 1800)
    out = config.FIG_DIR / "final" if hasattr(config, "FIG_DIR") else config.RESULTS_DIR.parent / "figures" / "final"
    out.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out / "decoding.gif", save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    print("wrote", out / "decoding.gif")
    try:
        import imageio
        w = imageio.get_writer(out / "decoding.mp4", fps=a.fps * 2, macro_block_size=1)
        for f, d in zip(frames, durs):
            for _ in range(max(1, round(d / 500))):
                w.append_data(np.asarray(f))
        w.close()
        print("wrote", out / "decoding.mp4")
    except Exception as e:
        print("mp4 skipped (pip install imageio imageio-ffmpeg to enable):", type(e).__name__)


if __name__ == "__main__":
    main()
