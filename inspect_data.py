"""Run this FIRST. Prints what is on disk, checks shapes/ordering, and reports EEG split-half reliability.

    python3 inspect_data.py            # uses first available subject
    python3 inspect_data.py --sub 3
"""
import argparse

import numpy as np

import config
import data
from fit import split_half_reliability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=None)
    a = ap.parse_args()

    print("DATA_ROOT:", config.DATA_ROOT)
    print("EEG_DIR exists:", config.EEG_DIR.exists(), "| STIM_DIR exists:", config.STIM_DIR.exists())
    if config.EEG_DIR.exists():
        print("\nTop-level entries in EEG_DIR:")
        for p in sorted(config.EEG_DIR.iterdir())[:40]:
            print("  ", p.name, "(OK)" if p.exists() else "(BROKEN LINK)")

    subs = data.find_subjects()
    print("\nSubjects found:", subs)
    if not subs:
        print("No subjects found -> send me the printout above and I'll adapt data.py.")
        return
    sub = a.sub or subs[0]

    train, times, ch = data.load_eeg(sub, "train")
    test, _, _ = data.load_eeg(sub, "test")
    print(f"\nsub-{sub:02d} train {train.shape}  test {test.shape}")
    print("channels:", ch)
    print(f"times: {times[0]:.3f} .. {times[-1]:.3f} s  ({len(times)} points)")
    print("NaNs  train/test:", np.isnan(train).sum(), np.isnan(test).sum())
    print("std   train/test:", float(train.std()), float(test.std()))

    rel = split_half_reliability(test, n_splits=20, seed=0)
    k = np.unravel_index(np.nanargmax(rel), rel.shape)
    print(f"\nSplit-half reliability (SB-corrected) of test ERPs: max {rel.max():.2f} at {ch[k[0]]}, {times[k[1]]:.2f}s;"
          f" mean over window {rel[:, (times >= config.WINDOW[0]) & (times <= config.WINDOW[1])].mean():.2f}")
    print("(expect clearly > 0.3 over occipital channels ~100-250 ms; ~0 everywhere => something is wrong)")

    for split in ("train", "test"):
        try:
            paths, cid = data.load_stimuli(split)
            print(f"\nstimuli {split}: {len(paths)} images, {len(np.unique(cid))} concepts; first: {paths[0].parent.name}/{paths[0].name}")
        except Exception as e:  # noqa: BLE001
            print(f"\nstimuli {split}: PROBLEM -> {e}")

    mp = config.STIM_DIR / "image_metadata.npy"
    if mp.exists():
        m = np.load(mp, allow_pickle=True).item()
        print("\nimage_metadata.npy keys:", list(m.keys()))


if __name__ == "__main__":
    main()
