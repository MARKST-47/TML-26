"""
Hybrid v8 = v7 + WM_5 BETA bumped 1.0 -> 2.0 (offline proxy_product 0.815 -> 0.875).
  WM_1 dwtDct re-embed, blend 0.85 toward clean   (quality-optimized, from v7)
  WM_2 RivaGAN re-embed, blend 0.50 toward clean   (quality-optimized, from v7)
  WM_5 SERUM transplant at BETA=2.0                (the one change vs v7)
  WM_3,4,6,7,8 averaged at AVG_ALPHA=0.4
Isolated test: only WM_5's BETA differs from v7, so any leaderboard move is WM_5.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python build_hybrid8.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import zipfile
import numpy as np
import cv2
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder

DATASET=Path("Dataset"); SRC=DATASET/"watermarked_sources"; CLEAN=DATASET/"clean_targets"
OUT=Path("submission_temp"); OUT.mkdir(exist_ok=True); ZIP="submission.zip"
AVG_ALPHA=0.4; BETA=2.0            # <-- only change from v7 (was 1.0)
BLEND_WM1=0.85; BLEND_WM2=0.50

WatermarkEncoder.loadModel(); WatermarkDecoder.loadModel()

def lb(p): return cv2.imread(str(p),cv2.IMREAD_COLOR)
def dec(img,m,l):
    try: return np.array(WatermarkDecoder("bits",l).decode(img,m),dtype=np.int32)
    except: return None
def recover(wm,m,l):
    rows=[dec(lb(p),m,l) for p in sorted((SRC/wm).glob("*.png"))]
    rows=np.stack([r for r in rows if r is not None and len(r)==l])
    return (rows.mean(0)>0.5).astype(int)
def avg_resid(wm,lo,hi):
    sp=sorted((SRC/wm).glob("*.png"))
    mw=np.mean([np.asarray(Image.open(p).convert("RGB"),np.float32) for p in sp],0)
    mc=np.mean([np.asarray(Image.open(CLEAN/f"{n}.png").convert("RGB"),np.float32) for n in range(lo,hi+1)],0)
    return mw-mc
def serum_W(wm):
    sp=sorted((SRC/wm).glob("*.png"))
    ref=lb(sp[0]); H,Wd=ref.shape[:2]; res=[]
    for p in sp:
        im=lb(p).astype(np.float32)
        if im.shape[:2]!=(H,Wd): im=cv2.resize(im,(Wd,H))
        res.append(im - cv2.bilateralFilter(im,7,50,50))
    return np.stack(res).mean(0)

# WM_1: re-embed + blend 0.85
m1=recover("WM_1","dwtDct",16); e1=WatermarkEncoder(); e1.set_watermark("bits",m1.tolist())
for n in range(1,26):
    c=lb(CLEAN/f"{n}.png")
    try: w=e1.encode(c,"dwtDct")
    except: w=c
    f=np.clip(BLEND_WM1*w.astype(float)+(1-BLEND_WM1)*c.astype(float),0,255).astype(np.uint8)
    cv2.imwrite(str(OUT/f"{n}.png"),f)
# WM_2: re-embed + blend 0.50
m2=recover("WM_2","rivaGan",32); e2=WatermarkEncoder(); e2.set_watermark("bits",m2.tolist())
for n in range(26,51):
    c=lb(CLEAN/f"{n}.png")
    try: w=e2.encode(c,"rivaGan")
    except: w=c
    f=np.clip(BLEND_WM2*w.astype(float)+(1-BLEND_WM2)*c.astype(float),0,255).astype(np.uint8)
    cv2.imwrite(str(OUT/f"{n}.png"),f)
# WM_3,4 average
for wm,lo,hi in [("WM_3",51,75),("WM_4",76,100)]:
    d=avg_resid(wm,lo,hi)
    for n in range(lo,hi+1):
        c=np.asarray(Image.open(CLEAN/f"{n}.png").convert("RGB"),np.float32)
        Image.fromarray(np.clip(c+AVG_ALPHA*d,0,255).astype(np.uint8)).save(OUT/f"{n}.png")
# WM_5 transplant at BETA=2.0
W5=serum_W("WM_5")
for n in range(101,126):
    c=lb(CLEAN/f"{n}.png").astype(np.float32); ch,cw=c.shape[:2]
    Wr=cv2.resize(W5,(cw,ch)) if W5.shape[:2]!=(ch,cw) else W5
    cv2.imwrite(str(OUT/f"{n}.png"),np.clip(c+BETA*Wr,0,255).astype(np.uint8))
# WM_6,7,8 average
for wm,lo,hi in [("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]:
    d=avg_resid(wm,lo,hi)
    for n in range(lo,hi+1):
        c=np.asarray(Image.open(CLEAN/f"{n}.png").convert("RGB"),np.float32)
        Image.fromarray(np.clip(c+AVG_ALPHA*d,0,255).astype(np.uint8)).save(OUT/f"{n}.png")

n=len(list(OUT.glob("*.png")))
print(f"Forged {n} images. v8: WM_5 BETA={BETA} (vs v7's 1.0), rest = v7.")
if n==200:
    with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as zf:
        for i in range(1,201): zf.write(OUT/f"{i}.png",arcname=f"{i}.png")
    print(f"Saved {ZIP} — ready. Compare to v7 (0.457).")
else: print("[WARN] not 200, no zip.")
