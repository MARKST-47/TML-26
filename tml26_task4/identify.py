"""
Scheme identification via decoder agreement.

All 25 sources in a batch share ONE message. So the CORRECT (decoder, bit-length)
makes all 25 decode to (nearly) the same bitstring; a WRONG guess yields bits that
disagree image-to-image. We score per-bit agreement across the 25 sources:
    agreement = mean over bit positions of |fraction_of_1s - 0.5| * 2
    (1.0 = every source agrees on every bit; ~0 = random/no scheme match)

A batch that lights up (agreement near 1.0) is a classic invisible-watermark
scheme, and the majority-vote bits ARE the recovered message -> we can re-embed.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python identify.py
"""
from pathlib import Path
import numpy as np
import cv2
from imwatermark import WatermarkDecoder

SRC = Path("Dataset/watermarked_sources")
CATS = ["WM_1", "WM_2", "WM_3", "WM_4", "WM_5", "WM_6", "WM_7", "WM_8"]
METHODS = ["dwtDct", "dwtDctSvd"]
LENGTHS = [32, 48, 64, 256]  # common invisible-watermark payload sizes


def load_bgr(p):
    """imwatermark expects an OpenCV BGR uint8 array."""
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)  # already BGR
    return img


def decode_bits(img, method, length):
    try:
        dec = WatermarkDecoder("bits", length)
        bits = dec.decode(img, method)
        return np.array(bits, dtype=np.int32)
    except Exception:
        return None


def agreement(bit_rows):
    """bit_rows: (n_sources, length). Returns mean per-bit consensus in [0,1]."""
    frac1 = bit_rows.mean(axis=0)            # fraction of sources with bit=1
    consensus = np.abs(frac1 - 0.5) * 2.0     # 1 if unanimous, 0 if 50/50
    return float(consensus.mean())


print("Scanning decoders x bit-lengths for cross-source agreement...\n")
print(f"{'batch':6} {'method':10} {'len':>4} {'agreement':>10}")
print("-" * 36)

best = {}
for wm in CATS:
    paths = sorted((SRC / wm).glob("*.png"))
    imgs = [load_bgr(p) for p in paths]
    best_score, best_combo, best_bits = -1.0, None, None
    for method in METHODS:
        for length in LENGTHS:
            rows = []
            for im in imgs:
                b = decode_bits(im, method, length)
                if b is not None and len(b) == length:
                    rows.append(b)
            if len(rows) < len(imgs) // 2:
                continue
            rows = np.stack(rows)
            sc = agreement(rows)
            tag = "  <-- strong" if sc > 0.85 else ("  <- maybe" if sc > 0.7 else "")
            print(f"{wm:6} {method:10} {length:>4} {sc:>10.3f}{tag}")
            if sc > best_score:
                msg = (rows.mean(axis=0) > 0.5).astype(int)
                best_score, best_combo, best_bits = sc, (method, length), msg
    best[wm] = (best_combo, best_score, best_bits)
    print()

print("=" * 50)
print("SUMMARY (best guess per batch):")
for wm in CATS:
    combo, sc, bits = best[wm]
    verdict = "CLASSIC scheme -> forgeable" if sc > 0.85 else \
              ("possible" if sc > 0.7 else "NOT classic (diffusion/other?)")
    print(f"{wm}: {combo} agree={sc:.3f}  {verdict}")
    if sc > 0.85 and bits is not None:
        bitstr = "".join(map(str, bits.tolist()))
        print(f"      message[{len(bits)}b] = {bitstr}")
