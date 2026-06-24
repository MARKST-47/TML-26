"""
Hybrid v3: WM_1 (dwtDct/16) + WM_2 (RivaGAN/32) re-embedded UNCHANGED.
WM_3-8 averaged at REDUCED strength (AVG_ALPHA) -> better LPIPS, since their
detection is ~0 either way. Verifies the two re-embeds before zipping.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python build_hybrid3.py
"""
from pathlib import Path
import zipfile
import numpy as np
import cv2
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder

DATASET=Path("Dataset"); SRC=DATASET/"watermarked_sources"; CLEAN=DATASET/"clean_targets"
OUT=Path("submission_temp"); OUT.mkdir(exist_ok=True); ZIP="submission.zip"

AVG_ALPHA=0.5   # averaged batches strength (was 1.0). Lower = better quality.

WatermarkEncoder.loadModel(); WatermarkDecoder.loadModel()

PLAN={
 "WM_1":(1,25,"reembed","dwtDct",16),
 "WM_2":(26,50,"reembed","rivaGan",32),
 "WM_3":(51,75,"average",None,None),
 "WM_4":(76,100,"average",None,None),
 "WM_5":(101,125,"average",None,None),
 "WM_6":(126,150,"average",None,None),
 "WM_7":(151,175,"average",None,None),
 "WM_8":(176,200,"average",None,None),
}

def lb(p): return cv2.imread(str(p),cv2.IMREAD_COLOR)
def dec(img,m,l):
    try: return np.array(WatermarkDecoder("bits",l).decode(img,m),dtype=np.int32)
    except: return None
def ac(a,b):
    if a is None or b is None: return 0.0
    n=min(len(a),len(b)); return float((a[:n]==b[:n]).mean()) if n else 0.0
def recover(wm,m,l):
    rows=[dec(lb(p),m,l) for p in sorted((SRC/wm).glob("*.png"))]
    rows=np.stack([r for r in rows if r is not None and len(r)==l])
    return (rows.mean(0)>0.5).astype(int),rows
def avg_resid(wm,lo,hi):
    sp=sorted((SRC/wm).glob("*.png"))
    mw=np.mean([np.asarray(Image.open(p).convert("RGB"),np.float32) for p in sp],0)
    mc=np.mean([np.asarray(Image.open(CLEAN/f"{n}.png").convert("RGB"),np.float32) for n in range(lo,hi+1)],0)
    return mw-mc

print(f"AVG_ALPHA={AVG_ALPHA}\n{'batch':6} {'mode':8} {'bitacc':>8} {'rmse':>7}")
print("-"*36)
for wm,(lo,hi,mode,m,l) in PLAN.items():
    rmses=[]; accs=[]
    if mode=="reembed":
        msg,src=recover(wm,m,l)
        enc=WatermarkEncoder(); enc.set_watermark("bits",msg.tolist())
        for k,n in enumerate(range(lo,hi+1)):
            c=lb(CLEAN/f"{n}.png")
            try: f=enc.encode(c,m)
            except: f=c
            cv2.imwrite(str(OUT/f"{n}.png"),f)
            accs.append(ac(dec(f,m,l),src[k%len(src)]))
            rmses.append(float(np.sqrt(np.mean((c.astype(float)-f.astype(float))**2))))
        print(f"{wm:6} {mode:8} {np.mean(accs):8.3f} {np.mean(rmses):7.2f}")
    else:
        d=avg_resid(wm,lo,hi)
        for n in range(lo,hi+1):
            c=np.asarray(Image.open(CLEAN/f"{n}.png").convert("RGB"),np.float32)
            f=np.clip(c+AVG_ALPHA*d,0,255).astype(np.uint8)
            Image.fromarray(f).save(OUT/f"{n}.png")
            rmses.append(float(np.sqrt(np.mean((c-f)**2))))
        print(f"{wm:6} {mode:8} {'(avg)':>8} {np.mean(rmses):7.2f}")

n=len(list(OUT.glob("*.png")))
print(f"\nForged {n} images.")
if n==200:
    with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as zf:
        for i in range(1,201): zf.write(OUT/f"{i}.png",arcname=f"{i}.png")
    print(f"Saved {ZIP} — ready.")
else: print("[WARN] not 200, no zip.")
