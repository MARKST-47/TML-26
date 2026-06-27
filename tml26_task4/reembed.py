"""
Re-embedding attack (the path past 0.19).

Averaging reproduces watermark PRESENCE but not the MESSAGE BITS (proved by
bitacc_test.py: best averaging = 0.375 bit-acc = S_det 0). The fix: recover each
batch's message with the real DECODER, then stamp it onto the clean targets with
the matching ENCODER. We verify bit-accuracy locally before any submission.

This script does WM_1 (confirmed dwtDct/48). It:
  1. recovers WM_1's message (majority vote over 25 sources),
  2. encodes it onto each clean target 1..25,
  3. decodes the result and reports bit-accuracy + an LPIPS-proxy quality number.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reembed.py
"""
from pathlib import Path
import numpy as np
import cv2
from imwatermark import WatermarkEncoder, WatermarkDecoder

DATASET = Path("Dataset")
SRC = DATASET / "watermarked_sources"
CLEAN = DATASET / "clean_targets"
OUT = Path("reembed_out")
OUT.mkdir(exist_ok=True)

METHOD = "dwtDct"
LENGTH = 48
WM, LO, HI = "WM_1", 1, 25


def load_bgr(p):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)


def decode(img_bgr):
    return np.array(WatermarkDecoder("bits", LENGTH).decode(img_bgr, METHOD), dtype=np.int32)


def bit_acc(a, b):
    n = min(len(a), len(b))
    return float((a[:n] == b[:n]).mean()) if n else 0.0


def quality_proxy(orig_bgr, forged_bgr):
    d = orig_bgr.astype(np.float64) - forged_bgr.astype(np.float64)
    rmse = np.sqrt(np.mean(d ** 2))
    lpips = (rmse / 255.0) * 2.5         # same rough proxy as before
    return rmse, float(np.exp(-8.0 * lpips))


# 1) recover the message
src_paths = sorted((SRC / WM).glob("*.png"))
rows = np.stack([decode(load_bgr(p)) for p in src_paths])
message = (rows.mean(0) > 0.5).astype(int)
print(f"{WM}: recovered message, source self-agreement = "
      f"{np.mean([bit_acc(r, message) for r in rows]):.3f}")
msg_str = "".join(map(str, message.tolist()))
print(f"message = {msg_str}\n")

# 2) encode onto each clean target, 3) verify
enc = WatermarkEncoder()
enc.set_watermark("bits", message.tolist())

print(f"{'img':>4} {'bit_acc':>8} {'rmse':>7} {'S_qlt':>7} {'S_final(proxy)':>15}")
print("-" * 48)
accs, qlts, finals = [], [], []
for n in range(LO, HI + 1):
    clean = load_bgr(CLEAN / f"{n}.png")
    forged = enc.encode(clean, METHOD)
    cv2.imwrite(str(OUT / f"{n}.png"), forged)

    ba = bit_acc(decode(forged), message)
    rmse, sqlt = quality_proxy(clean, forged)
    sdet = max(0.0, 2 * (ba - 0.5))
    sfin = sdet * sqlt
    accs.append(ba); qlts.append(sqlt); finals.append(sfin)
    print(f"{n:>4} {ba:8.3f} {rmse:7.2f} {sqlt:7.3f} {sfin:15.3f}")

print("-" * 48)
print(f"mean bit_acc={np.mean(accs):.3f}  mean S_qlt={np.mean(qlts):.3f}  "
      f"mean S_final(proxy)={np.mean(finals):.3f}")
print("\nIf bit_acc ~0.95+ and S_final >> 0.19: re-embedding works -> extend to all")
print("8 batches (confirm each one's method/length first), then build the full zip.")
