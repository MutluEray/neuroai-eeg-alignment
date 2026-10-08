# Do deep vision models predict human EEG? A Brain-Score-style layer analysis on THINGS-EEG2

Adapts the Brain-Score logic (which models, and which layers, are most "brain-like"?) from primate neural recordings
to human scalp EEG. For each layer of a pretrained vision model, a cross-validated ridge map from layer activations to
the EEG response to the same images is fitted, and held-out predictivity is reported, normalized by the EEG's own noise
ceiling. Controls: untrained networks, non-learned low-level baselines, a shuffled-EEG negative control, and a
readout-sensitivity analysis.

> **Status:** analysis complete (9 subjects). `python3 make_report.py` regenerates tables in `results/RESULTS.md`.

## Data
THINGS-EEG2 (Gifford et al., 2022; CC-BY 4.0; NEMAR `nm000232`): 10 subjects (9 used here; sub-08 not downloaded),
16,540 training images (4 repetitions) + 200 test images (80 repetitions), 17 occipito-parietal channels,
100 time points (-200 to 790 ms) from the authors' preprocessed epochs. Place the unzipped subject folders and the
stimulus folders under `./things_eeg2` (or set `THINGS_EEG2_ROOT`).

## Method
- **Models** (fixed, bounded): torchvision ResNet-50 and ViT-B/16, both ImageNet-1k supervised.
  - ViT is read out two ways: *mean-token* (mean over patch tokens) and *grid* (patch tokens pooled to 2x2, matching the ResNet readout).
  - Controls: untrained ResNet-50 with default BatchNorm statistics, and with BatchNorm statistics calibrated on 2,048 training images.
  - Non-learned baselines: raw RGB pixels (4x4 ... 32x32) and a Gabor-energy (8 orientations x 4 scales) + Fourier-power model.
- **Layers**: ResNet conv1, every bottleneck block, avgpool, fc; ViT every block, final CLS, logits. Conv maps pooled to 2x2, then PCA (1000 components, fitted on training images only).
- **Regression**: ridge to 17 channels x 100 time points; penalty chosen on held-out *concepts*; final fit on all training images.
- **Score**: Pearson r over the 200 held-out test images per (channel, time), divided by sqrt(split-half reliability) of the 80-rep test ERPs; points with reliability < 0.1 excluded. Windows: all 50-400 ms, early 50-200 ms, late 200-500 ms.
- **Layer choice for comparisons** is leave-one-subject-out (the layer for each subject is picked on the other subjects).
- **Statistics**: paired differences over subjects (bootstrap CI, exact sign-flip p, Holm), and image-level bootstrap of the 200 test images (1,000 resamples, same resamples for all models), plus a two-way bootstrap over both.

Analysis history, for transparency: the ViT *grid* readout, the pixel baseline and the early/late split were added
after seeing that the first ResNet-vs-ViT gap could be a readout artifact; they were not part of the initial plan.

## Results
Held-out 200 images, 9 subjects, 17 channels. Normalized r = r / sqrt(EEG split-half reliability). "Best layer" is chosen
leave-one-subject-out. Intervals are 95% two-way bootstrap CIs over subjects and test images.

### 1. Learned features predict EEG far better than untrained or low-level models, most clearly after 200 ms
Difference in normalized r (2x2 readout):

| comparison | all (50-400 ms) | early (50-200 ms) | late (200-500 ms) |
|---|---|---|---|
| trained ResNet-50 minus untrained ResNet-50 (BN-calibrated) | +0.206 [+0.183, +0.227] | +0.166 [+0.142, +0.189] | +0.241 [+0.210, +0.268] |
| trained ResNet-50 minus Gabor/Fourier baseline | +0.196 [+0.170, +0.219] | +0.140 [+0.119, +0.161] | +0.221 [+0.191, +0.251] |
| untrained ResNet-50 minus Gabor/Fourier baseline | -0.010 [-0.033, +0.013] | -0.025 [-0.052, +0.002] | -0.019 [-0.045, +0.004] |
| ResNet-50 minus ViT-B/16 (matched 2x2 readout) | +0.021 [+0.013, +0.029] | +0.022 [+0.012, +0.033] | +0.010 [-0.001, +0.021] |
| ViT-B/16: 2x2 readout minus mean-token readout | +0.050 [+0.038, +0.062] | +0.092 [+0.075, +0.108] | +0.023 [+0.014, +0.032] |

- An untrained ResNet-50 is statistically indistinguishable from a Gabor-energy + Fourier-power baseline: its predictivity is about what low-level spatial structure provides.
- At the sharp early peak the learned advantage is modest in raw r (about 0.71 for the best trained layers vs about 0.64 for the best untrained layer and 0.61 for the Gabor/Fourier baseline); the larger gap is in sustained and late predictivity.

### 2. The depth profile depends on the metric
- Early window and peak-over-time: best at early-to-mid layers (ResNet-50 layer2_3, ViT-B/16 block03-05). At the 4x4 readout the ResNet early-window curve is a plateau (0.697-0.698) from layer2_3 to layer3_5.
- Late window: predictivity rises with depth (ResNet-50 layer3_5, 0.538; ViT block09, 0.533); the ViT declines at the last block.
- The same dissociation appears in identification accuracy (below).

### 3. ResNet-50 and ViT-B/16 are comparable; readout matters more than architecture
- Finer spatial pooling raises both models (normalized r, all window: ResNet 0.552 / 0.599 / 0.617 and ViT 0.529 / 0.578 / 0.600 for 1x1 / 2x2 / 4x4 grids); 4x4 had not plateaued. PCA 500 vs 1000 changes scores by about 0.003.
- ResNet minus ViT is +0.017 to +0.023 overall and +0.005 to +0.010 late across readouts; in the early window it shrinks from +0.036 (1x1) to +0.022 (2x2) and reverses to -0.006 (4x4). No consistent architecture advantage.

### 4. Decoding views (illustrative; chance 0.5% top-1 / 2.5% top-5 in 200-way tasks)
| | retrieval from EEG (top-1 / top-5) | identification via encoders (top-1, all window) |
|---|---|---|
| ResNet-50, layer3_5 | 51.1% / 82.2% | 69.2% |
| ViT-B/16, best layer (block07 / block05) | 47.1% / 79.9% | 64.8% |
| untrained ResNet-50, best layer | 13.8% / 35.0% | 32.7% |
| Gabor/Fourier baseline | 10.6% / 24.7% | 22.0% |

Test responses are averaged over 80 repetitions (identification with 10 repetitions is roughly half as accurate; see Figure E). Linear
16x16 reconstructions from EEG correlate with the viewed image beyond the mean image at r = 0.27 / 0.30 / 0.24 / 0.12 for 100 ms windows starting at 50 / 150 / 250 / 350 ms,
versus 0.68-0.87 for network layers: EEG carries coarse luminance and colour layout, not object detail.

Figures (in `figures/final/`): A layer profiles by window, B layer x time maps, C effect sizes, D readout sensitivity, E identification, F1/F2 retrieval, G reconstructions.

## Relation to prior work
- Gifford et al. (2022) introduced THINGS-EEG2 and encoding models for it. Related work reports that how ViT tokens are read out strongly affects scores (Baghel et al., 2026, arXiv:2609.26512, preprint). This project adds untrained and low-level references, noise-ceiling normalization, an image-level bootstrap and readout sensitivity.
- Random-weight CNNs are known to predict early visual responses well but higher areas less well; the untrained-network results here fit that pattern.
- EEG decoding methods on THINGS-EEG2 (e.g. NICE, ATM and later work) use 63 channels and deep encoders and report much higher retrieval accuracy; the decoding views here use 17 channels and linear models and are not comparable.

## Limitations
- Only 200 test images (one per concept) and 9 subjects: with 9 subjects the smallest possible exact p-value is 2/512, so p-values saturate; effect sizes and CIs are more informative. Layer-selection uncertainty is not propagated into the CIs.
- RSVP at 5 Hz: responses to neighbouring images overlap in each epoch; late-window predictivity reflects the current image's later processing mixed with these overlaps.
- 17 occipito-parietal channels only; peak-latency differences across layers are within 1-2 time bins (10 ms) and are not interpreted.
- Layer depth is compared on a relative scale; ResNet stages and ViT blocks are not equivalent units, and the ViT grid curve has no logits layer.
- The baselines are linear read-outs of simple features and are floors, not complete models of low-level vision.

## Run order
```bash
pip install -r requirements.txt
python3 inspect_data.py                       # verify files, shapes, image order, EEG reliability
python3 extract.py                            # all models and baselines (features/, ~10-15 GB incl. PCA caches)
python3 fit.py                                # all subjects/models (also stores image-bootstrap scores)
python3 fit.py --shuffle                      # negative control
python3 analyze.py && python3 stats.py && python3 plot.py
python3 make_figures.py                       # final website figures -> figures/final/ (PNG + SVG + captions.md)

# decoding views (identification, EEG -> feature retrieval, 16x16 reconstructions); needs the grid-4 features/fits above
python3 decode.py
python3 make_decoding_figures.py --subject 1

# sensitivity (readout grid 1/4, PCA 500) on a few layers around each peak:
L=(layer2_3 layer3_1 layer3_3 layer3_4 layer3_5 block03 block05 block07 block09 block12)   # array: works in bash and zsh
python3 extract.py --grid 1 --models resnet50 vit_b16_grid --layers "${L[@]}"
python3 extract.py --grid 4 --models resnet50 vit_b16_grid --layers "${L[@]}"
python3 fit.py --models resnet50_g1 resnet50_g4 vit_b16_grid_g1 vit_b16_grid_g4 --layers "${L[@]}"
python3 fit.py --pca 500 --models resnet50 vit_b16_grid --layers "${L[@]}"
python3 sensitivity.py && python3 make_report.py
```

## References
- Schrimpf et al. (2018). Brain-Score: Which artificial neural network for object recognition is most brain-like? bioRxiv.
- Schrimpf et al. (2020). Integrative benchmarking to advance neurally mechanistic models of human intelligence. Neuron.
- Yamins et al. (2014). Performance-optimized hierarchical models predict neural responses in higher visual cortex. PNAS.
- Cichy et al. (2016). Comparison of deep neural networks to spatio-temporal cortical dynamics of human visual object recognition. Sci Rep.
- Gifford, Dwivedi, Roig & Cichy (2022). A large and rich EEG dataset for modeling human visual object recognition. NeuroImage 264:119754.
- Hebart et al. (2019). THINGS: A database of 1,854 object concepts and more than 26,000 naturalistic object images. PLoS ONE.
- Baghel et al. (2026). EEG encoding-model comparison of CNNs and vision transformers on THINGS-EEG2. arXiv:2609.26512 (preprint).
- Song et al. (2024), NICE; Li et al. (2024), ATM: EEG-to-image retrieval on THINGS-EEG2.
- Lu & Golomb (2025). Human EEG and artificial neural networks reveal disentangled representations and processing timelines of object real-world size and depth in natural images. eLife.

*(Verify the Schrimpf, Yamins, Cichy, Hebart, Baghel, Song and Li entries (authors, titles, venues) before publishing; Gifford 2022 and Lu & Golomb 2025 were checked.)*
