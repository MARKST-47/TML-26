# Task 4: Watermark Forgery

**CISPA Helmholtz Center for Information Security: Trustworthy Machine Learning, SS2026**
Team `team_LXIV` (CMS team ID: `atml_team034`)

---

## Task Overview

We are given 200 clean target images and 8 sets of 25 source images, each set
(`WM_1`…`WM_8`) watermarked by a different, undisclosed scheme. The goal is to
**forge** the watermark from each source set onto a fixed block of clean targets,
so the forged images are detected as watermarked while remaining visually clean.

Each source set maps to a fixed block of targets:

| Source | Targets | Source | Targets |
| ------ | ------- | ------ | ------- |
| WM_1   | 1–25    | WM_5   | 101–125 |
| WM_2   | 26–50   | WM_6   | 126–150 |
| WM_3   | 51–75   | WM_7   | 151–175 |
| WM_4   | 76–100  | WM_8   | 176–200 |

**Scoring** (per image, then averaged):
`S = S_det × S_qlt`, where
`S_det = max(0, 2·(BitAccuracy − 0.5))` and `S_qlt = exp(−8·LPIPS)`.
Because the score is a **product**, a batch must win _both_ detection and quality, 
zero detection yields zero regardless of quality.

---

## Approach

Our central finding: the detector grades the **decoded message bits**, not mere
watermark presence. Naive averaging reproduces presence but not the message,
capping the detection score near zero. The only way to score is to reproduce each
scheme's actual message. We tackled each batch by its scheme type:

**1. Classical schemes: identify decoder, recover message, re-embed.**

- `WM_1` was identified as **dwtDct** (16-bit message) and `WM_2` as
  **RivaGAN** (32-bit), both via the `imwatermark` library (`identify.py`).
- For each, we decode all 25 sources, take the majority-vote message, and
  **re-embed it** into the clean targets with the genuine encoder (`reembed.py`).
  This reproduces the exact message bits (bit-accuracy 0.91 for WM_1, 0.99 for WM_2).

**2. SERUM-family pixel leak: fixed-pattern transplant.**

- `WM_5` is a SERUM-style diffusion watermark whose fixed pattern partially
  **leaks into pixel space**. We extract it as the averaged edge-preserving
  residual over the 25 sources (`source − bilateral(source)`) and transplant it
  onto the targets at strength β=2.0 (`serum_probe.py`, `serum_transplant.py`).

**3. Quality optimisation on cracked batches.**

- Since re-embedded watermarks retain large detection margin, we blend each
  watermarked image back toward clean: `f = α·w + (1−α)·c`. The product
  `S_det × S_qlt` peaks at α=0.85 for WM_1 and α=0.50 for WM_2.

**4. Remaining batches: averaged residual baseline.**

- `WM_3, 4, 6, 7, 8` are latent/token-domain watermarks (see elimination below).
  They do not respond to pixel-space forging, so they receive a low-strength
  averaged residual (α=0.4) to avoid quality loss.

**Systematic elimination of WM_3/4/6/7/8.** We ruled out, using a clean-target
control for every probe: dwtDct, dwtDctSvd, RivaGAN (lengths 8–48), TrustMark
(Q/P/B variants), blind-watermark, Tree-Ring FFT, LSB/bit-plane analysis, QIM
quantization fingerprinting, block-DCT/Fourier-phase consistency, a split-half DCT
filter (traced to selection artifact), a VGG-based learned surrogate, a pre-trained
ConvNeXt watermark forger (Meta WmForger, scored ~0.46 but wrong watermark family),
latent-space probes with a Stable-Diffusion VAE, BitMark detection (official code,
Infinity tokeniser, z-scores near zero), and QuantLoss provenance detection
(LlamaGen tokeniser, no gap between sources and clean targets). These consistent
negatives indicate the remaining watermarks live in a generative model's latent or
token space and require model-specific trained detectors to forge.

---

## Leaderboard Progression

| Build                    | Description                                          | Score     |
| ------------------------ | ---------------------------------------------------- | --------- |
| `forge_classical.py`     | WM_1/WM_2 re-embed; WM_3–8 avg α=0.5                | 0.423     |
| `forge_classical_serum.py` | + WM_5 SERUM transplant β=1.0; avg α=0.4           | 0.457     |
| `forge_final.py`         | + quality-blend WM_1(0.85)/WM_2(0.50); β=2.0        | **0.463** |

---

## Repository Structure

```
forge_final.py             Best submission builder (0.463). Produces submission.zip.
forge_classical_serum.py   Intermediate build: adds SERUM transplant (0.457).
forge_classical.py         Initial build: classical re-embed only (0.423).
identify.py                Scheme identification: probes decoders against each batch.
reembed.py                 WM_1/WM_2 message recovery and re-embedding.
serum_probe.py             Detects WM_5's fixed pixel-space pattern.
serum_transplant.py        Extracts and transplants WM_5's watermark pattern.
submission.py              Leaderboard submission script (API key not included).
task_template.py           Course-provided target-mapping template.
README.md                  This file.
```

`Dataset/`, submission `.zip` files, and the Python environment are not tracked.

---

## Reproducing the Best Result

**1. Environment.** Python 3.10+. Install dependencies:

```bash
pip install numpy pillow opencv-python-headless imwatermark onnxruntime
```

**2. Data layout.** Place the provided dataset in a `Dataset/` folder beside the
scripts:

```
Dataset/
├── clean_targets/           (1.png … 200.png)
└── watermarked_sources/
    ├── WM_1/ … WM_8/        (25 images each)
```

**3. Build the submission.**

```bash
python forge_final.py
```

This reads from `Dataset/`, writes forged images to `submission_temp/`, and
produces `submission.zip` (200 images) ready for leaderboard upload.

**4. Submit to leaderboard.**

```bash
python submission.py
```

Set your API key in `submission.py` before running.

All paths are relative to the repository root; no absolute paths or
environment-specific configuration are required.
