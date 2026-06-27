# Task 4: Watermark Forgery Attack

**CISPA Helmholtz Center for Information Security: Trustworthy Machine Learning, SS2026**
Team `team_LXIV` (CMS team ID: `atml_team034`)

---

## Task Overview

We are given 200 clean target images and 8 sets of 25 source images, each set
(`WM_1`…`WM_8`) watermarked by a different, undisclosed scheme. The goal is to
**forge** the watermark from each source set onto a fixed block of clean targets,
so the forged images are detected as watermarked while remaining visually clean.

Each source set maps to a fixed block of targets:

| Source | Targets   | Source | Targets    |
|--------|-----------|--------|------------|
| WM_1   | 1–25      | WM_5   | 101–125    |
| WM_2   | 26–50     | WM_6   | 126–150    |
| WM_3   | 51–75     | WM_7   | 151–175    |
| WM_4   | 76–100    | WM_8   | 176–200    |

**Scoring** (per image, then averaged):
`S = S_det × S_qlt`, where
`S_det = max(0, 2·(BitAccuracy − 0.5))` and `S_qlt = exp(−8·LPIPS)`.
Because the score is a **product**, a batch must win *both* detection and quality zero detection yields zero regardless of quality.

---

## Approach and Flow

Our central finding: the detector grades the **decoded message bits**, not mere
watermark presence. Naive averaging reproduces presence but not the message, and
therefore plateaus. The only way to score is to reproduce each scheme's actual
message. We tackled each batch by its scheme type:

**1. Classical schemes: identify decoder, recover message, re-embed.**
- `WM_1` was identified as **dwtDct** (length-16 message) and `WM_2` as
  **RivaGAN** (length-32), both via the `imwatermark` library.
- For each, we decode all 25 sources, take the majority-vote message, and
  **re-embed it** into the clean targets with the real encoder. This reproduces
  the exact message bits (bit-accuracy 0.91 / 0.98).

**2. SERUM-family pixel leak: fixed-pattern transplant.**
- `WM_5` is a SERUM-style diffusion watermark whose fixed pattern partially
  **leaks into pixel space**. We extract it as the averaged edge-preserving
  residual over the 25 sources (`source − bilateral(source)`) and transplant it
  onto the targets — a WMCopier-style copy attack.

**3. Remaining batches: averaged residual baseline.**
- `WM_3, 4, 6, 7, 8` are latent/token-domain watermarks (confirmed below). They do
  not respond to pixel-space forging, so they receive the baseline averaged
  residual at low strength to avoid quality loss.

**Why the rest could not be forged (systematic elimination).** We ruled out, for
`WM_3/4/6/7/8`: dwtDct, dwtDctSvd, RivaGAN, blind-watermark, Tree-Ring FFT,
LSB/bit-plane, TrustMark, a trained CNN surrogate, and a latent-space probe with a
standard SD VAE. Per the SERUM paper (Kociszewski et al., ICLR'26), detection
requires the model's **trained detector** operating in the LDM latent space, an
artifact not recoverable from 25 sample images, which is consistent with all of
our negative results.

---

## Performance Log

| Build                          | Method added                              | Leaderboard |
|--------------------------------|-------------------------------------------|-------------|
| naive blend                    | global average blend                      | 0.159       |
| averaging                      | mean residual, all batches                | 0.189       |
| + WM_1 re-embed                | dwtDct message recovery                   | 0.192       |
| + WM_2 re-embed                | RivaGAN message recovery                  | 0.327       |
| low-alpha averaging            | tuned residual strength                   | 0.423       |
| **+ WM_5 SERUM transplant**    | **fixed-pattern pixel transplant**        | **0.457**   |

Final: **0.457** (rank ~6). WM_1, WM_2, and WM_5 carry the detectable signal;
the remaining batches contribute baseline quality only.

---

## Directory Structure

```
build_hybrid5.py     Best submission builder (0.457). Produces submission.zip.
identify.py          Scheme identification: probes decoders against each batch.
reembed.py           WM_1/WM_2 message recovery and re-embedding.
serum_probe.py       Detects WM_5's fixed pixel-space pattern (SNR test).
serum_transplant.py  Extracts and transplants WM_5's watermark with verification.
classical_probe.py   Rules out TrustMark across the uncracked batches.
latent_probe.py      Latent-space (SD VAE) diagnostic for SERUM-family batches.
task_template.py     Course-provided target-mapping template.
README.md            This file.
```

*Note:* `Dataset/`, the submission `.zip` files, and the local Python environment
are intentionally not tracked (see `.gitignore`).

---

## Reproducing the Best Result

**1. Environment.** Python 3.10+. Install dependencies:

```bash
pip install numpy pillow opencv-python-headless imwatermark onnxruntime
```

**2. Data layout.** Place the provided dataset in a `Dataset/` folder beside the
scripts, with this structure:

```
Dataset/
├── clean_targets/        (1.png … 200.png)
└── watermarked_sources/
    ├── WM_1/ … WM_8/      (25 images each)
```

**3. Build the submission.** From the repository root:

```bash
python build_hybrid5.py
```

This reads from `Dataset/`, writes forged images to `submission_temp/`, and
produces **`submission.zip`** (200 images) in the repository root: ready for
leaderboard upload.

All paths are relative to the repository root; no absolute paths or
environment-specific configuration are required.
