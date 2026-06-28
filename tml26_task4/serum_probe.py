"""
SERUM-hypothesis probe (from the SprintML lecture, slide 52):
  watermark injection eta' = sqrt(1-a)*eta + sqrt(a)*W, W FIXED per user/batch.
If WM_3-8 are SERUM-style, each batch's 25 sources share a FIXED additive
pattern W that clean images lack. Test: extract per-source high-freq residual,
average across the 25, and measure how CONSISTENT that pattern is (SNR) vs noise.
High consistency => a real fixed W we can transplant (WMCopier in image space).

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python serum_probe.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import cv2

DATASET = Path("Dataset"); SRC = DATASET/"watermarked_sources"; CLEAN = DATASET/"clean_targets"
CATS = [("WM_3",51,75),("WM_4",76,100),("WM_5",101,125),
        ("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]


def residual(img):
    """high-freq residual = img - bilateral(img); isolates watermark-like signal,
    edge-preserving so we don't capture content edges."""
    f = img.astype(np.float32)
    den = cv2.bilateralFilter(f, d=7, sigmaColor=50, sigmaSpace=50)
    return f - den


def consistency(stack):
    """How fixed is the pattern across the 25? mean^2 / var, averaged over pixels.
    A true fixed W -> high (signal repeats); random noise -> ~0 (cancels)."""
    mean = stack.mean(0)
    var = stack.var(0) + 1e-6
    snr = (mean**2 / var)
    return float(snr.mean()), float(np.abs(mean).mean())


print(f"{'batch':6} {'src_SNR':>9} {'cln_SNR':>9} {'src|mean|':>9} {'cln|mean|':>9}  verdict")
print("-"*64)
for wm, lo, hi in CATS:
    sp = sorted((SRC/wm).glob("*.png"))
    # use a common size: resize clean targets to source size for residual compare
    src_res = []
    ref = cv2.imread(str(sp[0]), cv2.IMREAD_COLOR)
    H, W = ref.shape[:2]
    for p in sp:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        src_res.append(residual(im))
    src_stack = np.stack(src_res)

    cln_res = []
    for n in range(lo, hi+1):
        im = cv2.imread(str(CLEAN/f"{n}.png"), cv2.IMREAD_COLOR)
        if im is None: continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H))
        cln_res.append(residual(im))
    cln_stack = np.stack(cln_res)

    s_snr, s_mean = consistency(src_stack)
    c_snr, c_mean = consistency(cln_stack)
    # a fixed watermark: sources show HIGHER consistency than clean baseline
    ratio = s_snr / (c_snr + 1e-9)
    flag = "  <- FIXED W?" if ratio > 1.5 and s_mean > c_mean*1.2 else ""
    print(f"{wm:6} {s_snr:9.4f} {c_snr:9.4f} {s_mean:9.4f} {c_mean:9.4f}{flag}  (ratio {ratio:.2f})")
print("-"*64)
print("If src_SNR >> cln_SNR: sources share a FIXED pattern clean images lack")
print("=> SERUM-style additive W survives to image space => transplantable.")
print("If src ~ cln: watermark is latent-only (needs the LDM encoder, not CPU-forgeable).")
