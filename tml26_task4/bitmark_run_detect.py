"""
(THE TEST): run BitMark detection on WM_3/4/6/7/8 sources vs clean controls.
Encode each image -> bits (proven working), flatten, run WatermarkDetector.detect()
with candidate green lists. A batch is BitMark if sources z >> 4 while clean z ~ 0.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, subprocess, types
from pathlib import Path

INF = Path("/home/atml_team034/tml26_task4/BitMark/Infinity")
BM = Path("/home/atml_team034/tml26_task4/BitMark")
VAE_CKPT = "/home/atml_team034/tml26_task4/weights/Infinity/infinity_vae_d32reg.pth"
DATA = Path("/home/atml_team034/tml26_task4/Dataset")

subprocess.run([sys.executable,"-m","pip","install","--quiet",
    "einops","safetensors","imageio","omegaconf","huggingface-hub","timm","pillow",
    "pydantic==1.10.13","opencv-python-headless","pandas","scipy","tokenizers","transformers"],
    check=True)

import torch, numpy as np
class _AnyModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"): raise AttributeError(name)
        return lambda *a, **k: None
for mn in ["flash_attn","flash_attn.flash_attn_interface","flash_attn.bert_padding",
           "flash_attn.layers","flash_attn.ops"]:
    m=_AnyModule(mn); m.__file__="<stub>"; m.__path__=[]; sys.modules[mn]=m

sys.path.insert(0, str(INF))
sys.path.insert(0, str(BM))
from infinity.models.bsq_vae.vae import vae_model
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from extended_watermark_processor import WatermarkDetector
from PIL import Image
from torchvision.transforms.functional import to_tensor

dev=torch.device("cuda")
vae=vae_model(VAE_CKPT,"dynamic",32,2**32,patch_size=16,
              encoder_ch_mult=[1,2,4,4,4],decoder_ch_mult=[1,2,4,4,4],test_mode=True).to(dev).eval()
print("[OK] VAE loaded")

hd=h_div_w_templates[np.argmin(np.abs(h_div_w_templates-1.0))]; pn="0.06M"
scales=dynamic_resolution_h_w[hd][pn]["scales"]; scale_schedule=[(1,h,w) for (t,h,w) in scales]
tgt_h,tgt_w=dynamic_resolution_h_w[hd][pn]["pixel"]

def encode_bits(path):
    pil=Image.open(path).convert("RGB"); w,h=pil.size
    if w/h<=tgt_w/tgt_h: rw,rh=tgt_w,int(tgt_w/(w/h))
    else: rh,rw=tgt_h,int((w/h)*tgt_h)
    pil=pil.resize((rw,rh),resample=Image.LANCZOS); arr=np.array(pil)
    cy=(arr.shape[0]-tgt_h)//2; cx=(arr.shape[1]-tgt_w)//2
    im=to_tensor(arr[cy:cy+tgt_h,cx:cx+tgt_w]); im=im.add(im).add_(-1)
    with torch.no_grad():
        out=vae.encode(im.unsqueeze(0).to(dev),scale_schedule=scale_schedule)
    bits=out[3]  # all_bit_indices per scale
    cat=torch.cat([t.view(-1,t.shape[-1]) for t in bits],dim=0)
    return cat.flatten()

def get_detector(green):
    return WatermarkDetector(vocab=[0,1],gamma=0.5,delta=2.0,device=dev,
                             z_threshold=4.0,ignore_repeated_ngrams=False,green_list=green)

CANDIDATES=["00,11","01,10"]  # 2-bit pattern candidates (context_width=1)
BATCHES=[("WM_3",51,75),("WM_4",76,100),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]

for green in CANDIDATES:
    print(f"\n{'='*55}\nGREEN LIST = '{green}'")
    det=get_detector(green)
    print(f"{'batch':6} {'src_z':>8} {'clean_z':>8}  verdict")
    print("-"*40)
    for wm,lo,hi in BATCHES:
        src_paths=sorted((DATA/'watermarked_sources'/wm).glob("*.png"))[:8]
        src_z=[]
        for p in src_paths:
            try:
                r=det.detect(tokenized_text=encode_bits(p)); src_z.append(r["z_score"])
            except Exception as e:
                print("  detect err:",repr(e)[:80]); break
        cln_z=[]
        for n in range(lo,lo+8):
            try:
                r=det.detect(tokenized_text=encode_bits(DATA/'clean_targets'/f"{n}.png")); cln_z.append(r["z_score"])
            except: pass
        if src_z:
            sz=np.mean(src_z); cz=np.mean(cln_z) if cln_z else 0
            flag="** BITMARK! **" if sz>4 and sz>cz+2 else ""
            print(f"{wm:6} {sz:8.2f} {cz:8.2f}  {flag}")
print("\nDONE. src_z>4 AND src>>clean = that batch is BitMark with that green list.")
