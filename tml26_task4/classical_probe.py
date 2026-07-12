"""
Probe the uncracked batches (WM_3,4,6,7,8) with the CLASSICAL decoders SERUM's
paper names as baselines: TrustMark and Stable Signature (HiDDeN extractor).
Logic (same as what cracked WM_1/WM_2): if a batch is a classical scheme, its 25
sources decode to a CONSISTENT message. High bit-agreement across the 25 = that
decoder matches = we can re-embed = real crack.

Runs on GPU via Condor. Installs trustmark + downloads HiDDeN extractor.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
from PIL import Image
import subprocess, sys

DATA = Path("/home/atml_team034/tml26_task4/Dataset")
SRC = DATA/"watermarked_sources"
BATCHES = [("WM_3",51,75),("WM_4",76,100),("WM_6",126,150),
           ("WM_7",151,175),("WM_8",176,200)]
# include WM_1 as a positive control (we KNOW it's classical dwtDct)
CONTROL = [("WM_1",1,25),("WM_2",26,50)]

def consistency(bits_list):
    """given list of decoded bit arrays, how consistent are they? (max=1.0)
    For a real classical watermark, all 25 sources share the message -> high."""
    bits_list = [b for b in bits_list if b is not None]
    if len(bits_list) < 5: return None, 0
    L = min(len(b) for b in bits_list)
    arr = np.stack([b[:L] for b in bits_list])
    # majority vote per bit, then fraction of sources agreeing with the vote
    vote = (arr.mean(0) > 0.5).astype(int)
    agree = (arr == vote).mean()
    return float(agree), L

# ---------- TrustMark ----------
print("="*55); print("TRUSTMARK DECODER")
try:
    subprocess.run([sys.executable,"-m","pip","install","--quiet","trustmark"],
                   check=True, timeout=600)
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, model_type='Q')
    for wm,lo,hi in CONTROL+ [b for b in BATCHES]:
        bl=[]
        for p in sorted((SRC/wm).glob("*.png")):
            try:
                wm_str, present, _ = tm.decode(Image.open(p).convert("RGB"))
                if present:
                    bl.append(np.array([int(c) for c in wm_str if c in "01"]))
            except Exception:
                pass
        agree, L = consistency(bl)
        if agree is None:
            print(f"  {wm}: no consistent decode (likely not TrustMark)")
        else:
            flag = "  <-- TRUSTMARK MATCH!" if agree>0.85 else ""
            print(f"  {wm}: bit-agreement={agree:.3f} (len {L}, {len(bl)}/25 decoded){flag}")
except Exception as e:
    print("  TrustMark unavailable:", repr(e)[:120])

print("="*55)
print("Interpretation: WM_1/WM_2 are our controls (known classical).")
print("If a mystery batch shows agreement >0.85 like the controls,")
print("that decoder matches it -> we re-embed -> real crack.")
print("If all mystery batches are ~0.5 (random), they're NOT this scheme.")
