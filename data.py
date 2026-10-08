"""Loading preprocessed THINGS-EEG2 (NEMAR nm000232) EEG + stimulus lists in a consistent order."""
import os
import re
import zipfile
from pathlib import Path

import numpy as np

import config

IMG_EXT = {".jpg", ".jpeg", ".png"}
N_TRAIN, N_TEST = 16540, 200


# ----------------------------------------------------------------------------- EEG
SKIP_DIRS = {"training_images", "test_images", "__pycache__"}


def _walk(root):
    """Yield all files under root, not descending into the (huge) image folders."""
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for f in fn:
            yield Path(dp) / f


def _sub_num(path: Path):
    try:
        rel = str(path.relative_to(config.EEG_DIR))
    except ValueError:
        rel = str(path)
    m = re.search(r"sub-?0*(\d+)", rel, re.I)
    return int(m.group(1)) if m else None


def _eeg_candidates():
    if not config.EEG_DIR.exists():
        raise FileNotFoundError(f"EEG_DIR not found: {config.EEG_DIR}\nSet THINGS_EEG2_ROOT or edit config.py")
    return [p for p in _walk(config.EEG_DIR) if p.suffix == ".npy"]


def _eeg_zips():
    return [p for p in _walk(config.EEG_DIR) if p.suffix == ".zip" and re.match(r"sub-?\d+", p.name, re.I)]


def find_subjects():
    subs = set()
    for p in _eeg_candidates():
        n = _sub_num(p)
        if n is not None and p.exists() and "train" in p.name.lower():
            subs.add(n)
    for z in _eeg_zips():
        n = _sub_num(z)
        if n is not None:
            subs.add(n)
    return sorted(subs)


def _maybe_unzip(sub):
    for z in _eeg_zips():
        if _sub_num(z) == sub:
            out = config.EEG_DIR / "_unzipped" / f"sub-{sub:02d}"
            if not out.exists():
                print(f"[data] unzipping {z.name} -> {out}")
                out.mkdir(parents=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(out)


def _find_eeg_file(sub, part):
    """part in {'train','test'}"""
    def look():
        return [p for p in _eeg_candidates() if _sub_num(p) == sub and part in p.name.lower()]

    cands = look()
    if not any(p.exists() for p in cands):
        _maybe_unzip(sub)
        cands = look()
    good = [p for p in cands if p.exists()]
    if not good and cands:
        raise FileNotFoundError(f"Found {cands[0]} but it is a broken link (git-annex content not fetched?).")
    if not good:
        raise FileNotFoundError(f"No '{part}' .npy for sub-{sub:02d} under {config.EEG_DIR}. Run `python3 inspect_data.py`.")
    if len(good) > 1:
        raise RuntimeError(f"Ambiguous files for sub-{sub:02d} {part}: {good}")
    return good[0]


def load_eeg(sub, part):
    """Returns (eeg, times, ch_names). part='train' -> (16540, 4, 17, 100); part='test' -> (200, 80, 17, 100)."""
    p = _find_eeg_file(sub, part)
    d = np.load(p, allow_pickle=True)
    if d.dtype == object and d.shape == ():
        d = d.item()
    if isinstance(d, dict):
        eeg = d["preprocessed_eeg_data"]
        times = np.asarray(d.get("times", np.linspace(-0.2, 0.8, eeg.shape[-1])))
        ch = list(d.get("ch_names", [f"ch{i}" for i in range(eeg.shape[-2])]))
    else:  # bare array
        eeg = d
        times = np.linspace(-0.2, 0.8, eeg.shape[-1])
        ch = [f"ch{i}" for i in range(eeg.shape[-2])]
    eeg = np.asarray(eeg, dtype=np.float32)
    exp = N_TRAIN if part == "train" else N_TEST
    if eeg.ndim != 4 or eeg.shape[0] != exp:
        raise ValueError(f"{p}: unexpected shape {eeg.shape}, expected ({exp}, reps, ch, time)")
    return eeg, times, ch


# ----------------------------------------------------------------------------- stimuli
def ensure_stimuli():
    for name in ("training_images", "test_images"):
        d, z = config.STIM_DIR / name, config.STIM_DIR / f"{name}.zip"
        if not d.exists() and z.exists():
            print(f"[data] extracting {z} (one-time)")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(d)


def _list_images(folder: Path):
    paths = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith(".")]
    return sorted(paths, key=lambda p: (p.parent.name, p.name))


def load_stimuli(split):
    """Returns (paths, concept_ids) for split in {'train','test'}, in the SAME order as the EEG condition axis.

    Order = image_metadata.npy order if available (authors' own ordering), else sorted (concept folder, filename).
    """
    ensure_stimuli()
    folder = config.STIM_DIR / ("training_images" if split == "train" else "test_images")
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing. Download stimuli/training_images.zip and test_images.zip from NEMAR.")
    paths = _list_images(folder)
    expected = N_TRAIN if split == "train" else N_TEST
    if len(paths) != expected:
        raise ValueError(f"{folder}: found {len(paths)} images, expected {expected}")

    meta_p = config.STIM_DIR / "image_metadata.npy"
    if meta_p.exists():
        try:
            meta = np.load(meta_p, allow_pickle=True).item()
            key = f"{split}_img_files"
            if key in meta:
                by_name = {p.name: p for p in paths}
                ordered = [by_name[Path(str(f)).name] for f in meta[key]]
                if len(ordered) == expected:
                    if [p.name for p in ordered] != [p.name for p in paths]:
                        print(f"[data] {split}: using image_metadata.npy order (differs from sorted order)")
                    paths = ordered
        except Exception as e:  # noqa: BLE001
            print(f"[data] could not use image_metadata.npy ({e}); using sorted order")

    names = [p.parent.name for p in paths]
    _, concept_ids = np.unique(names, return_inverse=True)
    return paths, concept_ids
