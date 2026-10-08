"""Central configuration. Edit here (or set THINGS_EEG2_ROOT) -- nothing else should hard-code paths."""
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _default_root():
    env = os.environ.get("THINGS_EEG2_ROOT")
    if env:
        return Path(env).expanduser()
    for cand in (ROOT / "things_eeg2", ROOT / "data" / "nm000232"):
        if cand.exists():
            return cand
    return ROOT / "things_eeg2"


def _find_dir(root, name, preferred):
    """Use `preferred` if it exists; else a directory called `name` under root; else root itself (flat layout)."""
    if preferred.exists():
        return preferred
    if root.exists():
        hits = [p for p in root.rglob(name) if p.is_dir()]
        if hits:
            return min(hits, key=lambda p: len(p.parts))
    return root


# Supports both the full NEMAR tree (derivatives/preprocessed_eeg, stimuli/) and a flat folder
# containing sub-XX.zip, training_images.zip, test_images.zip, image_metadata.npy (your current layout).
DATA_ROOT = _default_root()
EEG_DIR = _find_dir(DATA_ROOT, "preprocessed_eeg", DATA_ROOT / "derivatives" / "preprocessed_eeg")
STIM_DIR = _find_dir(DATA_ROOT, "stimuli", DATA_ROOT / "stimuli")

FEATURE_DIR = ROOT / "features"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# ---- models (fixed, bounded comparison) ----
# resnet50 / vit_b16: torchvision IMAGENET1K_V1 | vit_b16_grid: same ViT, patch tokens pooled to POOL_GRID x POOL_GRID (matches ResNet readout)
# resnet50_random: untrained control (default BN stats) | resnet50_random_bn: untrained but BN statistics calibrated on images
# pixels: raw-pixel baselines at 4 resolutions | lowlevel: Gabor energy (4 scales) + Fourier-power baselines (no learning)
MODELS = ["resnet50", "vit_b16", "vit_b16_grid", "resnet50_random", "resnet50_random_bn", "pixels", "lowlevel"]
BASELINES = ["pixels", "lowlevel"]  # non-network references: drawn as horizontal bands, no depth axis

# ---- feature extraction ----
RESIZE = 256
CROP = 224
BATCH_SIZE = 64
NUM_WORKERS = 4
POOL_GRID = 2          # conv feature maps are adaptively average-pooled to POOL_GRID x POOL_GRID

BN_CALIB_IMAGES = 2048  # training images used to calibrate BatchNorm statistics of the random-init control

# ---- regression ----
PCA_COMPONENTS = 1000  # fitted on training images only (no EEG involved)
# ridge penalty is expressed RELATIVE to the mean eigenvalue of X^T X, so it is comparable across layers
ALPHA_REL_GRID = np.logspace(-2, 5, 15)
VAL_FRACTION = 0.10    # fraction of training CONCEPTS held out to choose the ridge penalty
SEED = 0

# ---- scoring / noise ceiling ----
N_SPLITS_NC = 50       # random split-halves for the noise ceiling
REL_MIN = 0.10         # (channel, time) points with split-half reliability below this are excluded from normalized summaries
WINDOW = (0.05, 0.40)  # seconds post-stimulus used for the summary score per layer
WINDOWS = {"all": WINDOW, "early": (0.05, 0.20), "late": (0.20, 0.50)}

# ---- image-level bootstrap (resampling of the 200 test images, same resamples for every model/subject) ----
N_BOOT = 1000
BOOT_SEED = 123

# ---- sensitivity analysis (readout grid 1/2/4, PCA 500/1000), restricted to layers around each model's peak ----
SENS_LAYERS = {
    "resnet50": ["layer2_3", "layer3_1", "layer3_3", "layer3_4", "layer3_5"],
    "vit_b16_grid": ["block03", "block05", "block07", "block09", "block12"],
}
