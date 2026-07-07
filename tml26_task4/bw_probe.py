"""
Probe WM_3-8 against blind-watermark (guofei9987 DWT-DCT-SVD).
Extraction needs wm_shape (bit length), so we sweep candidate lengths and
measure cross-source agreement. High agree + mixed bits => this scheme.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bw_probe.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from blind_watermark import WaterMark

SRC = Path("Dataset/watermarked_sources")
# unknown batches (WM_1=dwtDct, WM_2=RivaGAN already cracked); include them as controls
CATS = ["WM_3","WM_4","WM_5","WM_6","WM_7","WM_8"]
LENGTHS = [16, 20, 32, 48, 64, 100, 128, 256]


def extract_bits(path, length):
    try:
        bwm = WaterMark(password_img=1, password_wm=1)
        bits = bwm.extract(filename=str(path), wm_shape=length, mode='bit')
        return np.array(bits, dtype=np.int32)
    except Exception:
        return None


def agree(rows):
    return float((np.abs(rows.mean(0) - 0.5) * 2.0).mean())


print(f"{'batch':6} {'best_len':>8} {'self_agree':>10} {'ones':>10}")
print("-" * 40)
for wm in CATS:
    paths = sorted((SRC / wm).glob("*.png"))
    best = (-1.0, None, None)
    for length in LENGTHS:
        rows = [extract_bits(p, length) for p in paths]
        rows = [r for r in rows if r is not None and len(r) == length]
        if len(rows) < len(paths) // 2:
            continue
        rows = np.stack(rows)
        ag = agree(rows)
        if ag > best[0]:
            best = (ag, length, (rows.mean(0) > 0.5).astype(int))
    ag, length, msg = best
    if msg is None:
        print(f"{wm:6} {'-':>8} {'extract-fail':>10}")
        continue
    flag = "  <- HIT" if (ag > 0.85 and 0 < msg.sum() < length) else \
           ("  <- weak" if ag > 0.6 else "")
    print(f"{wm:6} {length:>8} {ag:10.3f} {int(msg.sum()):>4}/{length}{flag}")
print("-" * 40)
print("HIT = high agree + mixed bits (not 0/all) => blind-watermark scheme => forgeable.")
