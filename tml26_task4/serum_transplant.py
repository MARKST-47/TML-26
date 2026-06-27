"""
SERUM/WMCopier transplant for WM_5, WM_6 (strong fixed W) + WM_4 (marginal).
Extract each batch's fixed pattern W = mean over sources of edge-preserving
residual, transplant onto clean targets at strength BETA, then VERIFY the forged
targets carry the same W as real sources (correlation), so we don't fool ourselves.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python serum_transplant.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import cv2

DATASET = Path("Dataset"); SRC = DATASET/"watermarked_sources"; CLEAN = DATASET/"clean_targets"
CANDS = [("WM_4",76,100),("WM_5",101,125),("WM_6",126,150)]
BETA = 1.0  # transplant strength; W is already at natural amplitude


def residual(img):
    f = img.astype(np.float32)
    return f - cv2.bilateralFilter(f, d=7, sigmaColor=50, sigmaSpace=50)


def extract_W(wm):
    sp = sorted((SRC/wm).glob("*.png"))
    ref = cv2.imread(str(sp[0]), cv2.IMREAD_COLOR); H,Wd = ref.shape[:2]
    res = []
    for p in sp:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im.shape[:2] != (H,Wd): im = cv2.resize(im,(Wd,H))
        res.append(residual(im))
    return np.stack(res).mean(0), (H,Wd)  # fixed W estimate


def corr(a, b):
    a = a.ravel()-a.mean(); b = b.ravel()-b.mean()
    d = (np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/d) if d>0 else 0.0


print(f"{'batch':6} {'W_corr_src':>11} {'W_corr_forged':>13} {'rmse':>7}  verdict")
print("-"*56)
for wm, lo, hi in CANDS:
    W, (H,Wd) = extract_W(wm)
    # baseline: how well does W correlate with a real source's residual?
    sp = sorted((SRC/wm).glob("*.png"))
    src_corrs = []
    for p in sp[:8]:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im.shape[:2]!=(H,Wd): im=cv2.resize(im,(Wd,H))
        src_corrs.append(corr(W, residual(im)))
    src_corr = np.mean(src_corrs)

    # transplant onto clean targets, then check forged residual correlates with W
    fwd_corrs, rmses = [], []
    for n in range(lo, min(lo+8,hi+1)):
        c = cv2.imread(str(CLEAN/f"{n}.png"), cv2.IMREAD_COLOR)
        ch,cw = c.shape[:2]
        Wr = cv2.resize(W,(cw,ch)) if (ch,cw)!=(H,Wd) else W
        forged = np.clip(c.astype(np.float32)+BETA*Wr,0,255).astype(np.uint8)
        fwd_corrs.append(corr(Wr, residual(forged)))
        rmses.append(float(np.sqrt(np.mean((c.astype(float)-forged.astype(float))**2))))
    fwd_corr = np.mean(fwd_corrs); rmse = np.mean(rmses)

    # verdict: forged should carry W (high fwd_corr) like real sources do (src_corr)
    ok = fwd_corr > 0.3 and src_corr > 0.3
    verdict = "TRANSPLANT OK" if ok else "weak/unclear"
    print(f"{wm:6} {src_corr:11.3f} {fwd_corr:13.3f} {rmse:7.2f}  {verdict}")
print("-"*56)
print("src_corr high = W is really in the sources. fwd_corr high = transplant")
print("reproduces it. Both high => worth submitting these batches. Then leaderbd decides.")
