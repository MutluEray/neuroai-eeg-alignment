"""Extract layer activations (or low-level baseline features) for the THINGS-EEG2 train/test images.

    python3 extract.py                                    # all models in config.MODELS
    python3 extract.py --models resnet50_random_bn lowlevel
    python3 extract.py --limit 64                         # smoke test (writes to features/_smoke)
    # sensitivity analysis: other pooling grids, restricted to a few layers (writes features/<model>_g<N>)
    python3 extract.py --grid 1 --models resnet50 vit_b16_grid --layers layer2_3 layer3_1 layer3_3 layer3_4 layer3_5 block03 block05 block07 block09 block12

Output: features/<model>/<layer>_{train,test}.npy  (float16, divided by a per-layer scale stored in meta.json)
 - resnet50 / resnet50_random: conv maps average-pooled to POOL_GRID x POOL_GRID (random: default BatchNorm stats)
 - resnet50_random_bn: untrained ResNet-50 (same seed) whose BatchNorm statistics are calibrated on training images
 - vit_b16:       mean over patch tokens (no spatial layout), plus final CLS and logits
 - vit_b16_grid:  patch tokens reshaped to 14x14 and pooled to POOL_GRID x POOL_GRID (spatial readout matched to ResNet)
 - pixels:        raw image downsampled to 4x4, 8x8, 16x16, 32x32 RGB
 - lowlevel:      Gabor energy (8 orientations, 4 scales, pooled to 4x4) + log Fourier power in radial x orientation bins
"""
import argparse
import json
import os
import time

# python.org builds of Python on macOS often lack root certificates -> SSL error when torchvision downloads weights.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

import config
import data

DEFAULT_GRID = config.POOL_GRID
PIXEL_LAYERS = {"rgb_4x4": 4, "rgb_8x8": 8, "rgb_16x16": 16, "rgb_32x32": 32}


class ImageList(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = T.Compose([
            T.Resize(config.RESIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(config.CROP),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def _loader(paths):
    return DataLoader(ImageList(paths), batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS, persistent_workers=config.NUM_WORKERS > 0)


# ----------------------------------------------------------------------------- networks
def build_model(name):
    if name == "resnet50":
        m = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
    elif name in ("resnet50_random", "resnet50_random_bn"):
        torch.manual_seed(config.SEED)  # identical random weights for both random controls
        m = tv.models.resnet50(weights=None)
    elif name in ("vit_b16", "vit_b16_grid"):
        m = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(name)
    return m.eval()


def calibrate_bn(model, device, n_images):
    """Set BatchNorm running statistics from n training images (cumulative average), then return to eval mode.
    Weights stay random; only the normalization statistics are calibrated so activations are sensibly scaled."""
    paths, _ = data.load_stimuli("train")
    sel = np.sort(np.random.default_rng(config.SEED).choice(len(paths), n_images, replace=False))
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # None -> cumulative moving average over all calibration batches
    model.train()
    with torch.no_grad():
        for x in tqdm(_loader([paths[i] for i in sel]), desc="BN calibration"):
            model(x.to(device))
    model.eval()


def attach_hooks(model, name, store):
    """Registers hooks; returns ordered list of layer names. Hooks write CPU float32 (B, D) tensors into `store`."""
    g = config.POOL_GRID
    order = []

    def make(layer, kind="auto"):
        def hook(mod, inp, out):
            x = out.detach().float().cpu()  # pool on CPU: MPS adaptive pooling needs divisible sizes
            if kind == "cls":
                x = x[:, 0]
            elif kind == "grid":  # ViT tokens (B, 1+N, D) -> patch grid -> pooled
                p = x[:, 1:]
                B, N, D = p.shape
                s = int(round(N ** 0.5))
                x = F.adaptive_avg_pool2d(p.transpose(1, 2).reshape(B, D, s, s), (g, g))
            elif x.ndim == 4:
                x = F.adaptive_avg_pool2d(x, (g, g))
            elif x.ndim == 3:
                x = x[:, 1:].mean(1)  # ViT: mean over patch tokens
            store[layer] = x.flatten(1)
        return hook

    if name.startswith("resnet"):
        model.relu.register_forward_hook(make("conv1"));  order.append("conv1")
        for li in range(1, 5):
            for bi, blk in enumerate(getattr(model, f"layer{li}")):
                lname = f"layer{li}_{bi}"
                blk.register_forward_hook(make(lname)); order.append(lname)
        model.avgpool.register_forward_hook(make("avgpool")); order.append("avgpool")
        model.fc.register_forward_hook(make("fc")); order.append("fc")
    elif name == "vit_b16_grid":
        for i, blk in enumerate(model.encoder.layers):
            lname = f"block{i + 1:02d}"
            blk.register_forward_hook(make(lname, kind="grid")); order.append(lname)
    else:  # vit_b16
        for i, blk in enumerate(model.encoder.layers):
            lname = f"block{i + 1:02d}"
            blk.register_forward_hook(make(lname)); order.append(lname)
        model.encoder.ln.register_forward_hook(make("final_cls", kind="cls")); order.append("final_cls")
        model.heads.register_forward_hook(make("logits")); order.append("logits")
    return order


# ----------------------------------------------------------------------------- low-level (non-learned) features
class LowLevel:
    """Gabor energy model (complex-cell-like) + Fourier power spectrum summary. No learning, no ImageNet."""
    SCALES = [1, 2, 4, 8]      # image downsampling factors -> effective wavelengths 6, 12, 24, 48 px
    N_ORI = 8
    WAVELENGTH = 6.0
    KSIZE = 19
    N_RAD, N_ANG = 6, 8

    def __init__(self, device):
        self.device = device
        sigma = 0.56 * self.WAVELENGTH
        r = self.KSIZE // 2
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
        env = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        filt = []
        for i in range(self.N_ORI):
            th = np.pi * i / self.N_ORI
            xr = xx * np.cos(th) + yy * np.sin(th)
            even = env * np.cos(2 * np.pi * xr / self.WAVELENGTH)
            even = even - env * even.sum() / env.sum()   # zero DC response
            odd = env * np.sin(2 * np.pi * xr / self.WAVELENGTH)
            filt += [even / np.linalg.norm(even), odd / np.linalg.norm(odd)]
        self.W = torch.tensor(np.stack(filt)[:, None], dtype=torch.float32, device=device)  # (16, 1, k, k)

        n = config.CROP
        f = np.fft.fftfreq(n) * n  # cycles per image
        FY, FX = np.meshgrid(f, f, indexing="ij")
        rad = np.sqrt(FX ** 2 + FY ** 2)
        ang = np.mod(np.arctan2(FY, FX), np.pi)
        edges = np.logspace(np.log10(2), np.log10(n / 2), self.N_RAD + 1)
        rb = np.digitize(rad, edges) - 1
        ob = np.minimum((ang / np.pi * self.N_ANG).astype(int), self.N_ANG - 1)
        valid = ((rb >= 0) & (rb < self.N_RAD)).ravel()
        bin_id = (rb * self.N_ANG + ob).ravel()[valid]
        self.valid_idx = torch.from_numpy(np.where(valid)[0])
        self.bin_ids = torch.from_numpy(bin_id).long()
        self.counts = torch.bincount(self.bin_ids, minlength=self.N_RAD * self.N_ANG).float().clamp(min=1)
        self.hann = torch.from_numpy(np.outer(np.hanning(n), np.hanning(n))).float()

    layer_names = ["gabor_s1", "gabor_s2", "gabor_s3", "gabor_s4", "gabor_all", "fft_power"]

    def fft_power(self, g):  # g: (B, H, W) on CPU
        g = (g - g.mean((1, 2), keepdim=True)) * self.hann
        P = (torch.fft.fft2(g).abs() ** 2).reshape(len(g), -1)[:, self.valid_idx]
        out = torch.zeros(len(g), self.N_RAD * self.N_ANG).index_add_(1, self.bin_ids, P) / self.counts
        return torch.log(out + 1e-6)

    def __call__(self, x):  # x: (B, 3, H, W) normalized RGB on CPU
        gray = x.mean(1, keepdim=True)
        feats = {}
        with torch.no_grad():
            gd = gray.to(self.device)
            for si, s in enumerate(self.SCALES):
                xs = F.avg_pool2d(gd, s) if s > 1 else gd
                out = F.conv2d(xs, self.W, padding=self.KSIZE // 2)
                e = torch.sqrt(out[:, 0::2] ** 2 + out[:, 1::2] ** 2 + 1e-8).cpu()
                feats[f"gabor_s{si + 1}"] = F.adaptive_avg_pool2d(e, (4, 4)).flatten(1)
            feats["gabor_all"] = torch.cat([feats[f"gabor_s{i + 1}"] for i in range(len(self.SCALES))], 1)
            feats["fft_power"] = self.fft_power(gray[:, 0])
        return {k: v.numpy() for k, v in feats.items()}


# ----------------------------------------------------------------------------- drivers
def _save_split(buf, layers, split, out_dir, meta):
    for l in layers:
        X = np.concatenate(buf[l])
        if split == "train":  # one scale per layer, shared by train and test (irrelevant to ridge, avoids fp16 overflow)
            meta["scales"][l] = float(np.sqrt((X.astype(np.float64) ** 2).mean()) + 1e-12)
            meta["dims"][l] = int(X.shape[1])
        np.save(out_dir / f"{l}_{split}.npy", (X / meta["scales"][l]).astype(np.float16))


def _out_dir(tag, limit):
    d = config.FEATURE_DIR / ("_smoke" if limit else "") / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_model(name, limit, device, keep):
    tag = name if config.POOL_GRID == DEFAULT_GRID else f"{name}_g{config.POOL_GRID}"
    out_dir = _out_dir(tag, limit)
    model = build_model(name).to(device)
    if name == "resnet50_random_bn":
        calibrate_bn(model, device, config.BN_CALIB_IMAGES if not limit else min(limit, 64))
    store = {}
    layers = attach_hooks(model, name, store)
    if keep:
        layers = [l for l in layers if l in keep]
    if not layers:
        print(f"[{tag}] no layers selected, skipping")
        return
    meta = {"layers": layers, "scales": {}, "dims": {}}

    for split in ("train", "test"):
        paths, _ = data.load_stimuli(split)
        if limit:
            paths = paths[:limit]
        buf = {l: [] for l in layers}
        t0 = time.time()
        with torch.no_grad():
            for x in tqdm(_loader(paths), desc=f"{tag}/{split}"):
                store.clear()
                model(x.to(device))
                for l in layers:
                    buf[l].append(store[l].numpy())
        _save_split(buf, layers, split, out_dir, meta)
        print(f"[{tag}] {split}: {len(paths)} images in {time.time() - t0:.0f}s")
        del buf
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)


def run_pixels(limit):
    out_dir = _out_dir("pixels", limit)
    layers = list(PIXEL_LAYERS)
    meta = {"layers": layers, "scales": {}, "dims": {}}
    for split in ("train", "test"):
        paths, _ = data.load_stimuli(split)
        if limit:
            paths = paths[:limit]
        buf = {l: [] for l in layers}
        t0 = time.time()
        for x in tqdm(_loader(paths), desc=f"pixels/{split}"):
            for l, g in PIXEL_LAYERS.items():
                buf[l].append(F.adaptive_avg_pool2d(x, (g, g)).flatten(1).numpy())
        _save_split(buf, layers, split, out_dir, meta)
        print(f"[pixels] {split}: {len(paths)} images in {time.time() - t0:.0f}s")
        del buf
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)


def run_lowlevel(limit, device):
    out_dir = _out_dir("lowlevel", limit)
    ll = LowLevel(device)
    layers = ll.layer_names
    meta = {"layers": layers, "scales": {}, "dims": {}}
    for split in ("train", "test"):
        paths, _ = data.load_stimuli(split)
        if limit:
            paths = paths[:limit]
        buf = {l: [] for l in layers}
        t0 = time.time()
        for x in tqdm(_loader(paths), desc=f"lowlevel/{split}"):
            feats = ll(x)
            for l in layers:
                buf[l].append(feats[l])
        _save_split(buf, layers, split, out_dir, meta)
        print(f"[lowlevel] {split}: {len(paths)} images in {time.time() - t0:.0f}s")
        del buf
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only first N images per split")
    ap.add_argument("--grid", type=int, default=None, help="spatial pooling grid (sensitivity analysis; default 2)")
    ap.add_argument("--layers", nargs="+", default=None, help="only extract these layer names (network models)")
    a = ap.parse_args()
    if a.grid:
        config.POOL_GRID = a.grid
    device = os.environ.get("DEVICE") or ("mps" if torch.backends.mps.is_available()
                                          else "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| pool grid:", config.POOL_GRID)
    for m in a.models:
        if m == "pixels":
            run_pixels(a.limit)
        elif m == "lowlevel":
            run_lowlevel(a.limit, device)
        else:
            run_model(m, a.limit, device, set(a.layers) if a.layers else None)


if __name__ == "__main__":
    main()
