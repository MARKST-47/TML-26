"""
QuantLoss test (corrected config): LlamaGen VQ, ORIGINAL encoder (no finetuning).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, subprocess
from pathlib import Path

REPO = Path("/home/atml_team034/tml26_task4/DataProvenanceIAR")
DATA = Path("/home/atml_team034/tml26_task4/Dataset")

subprocess.run([sys.executable,"-m","pip","install","--quiet",
    "numpy","scipy","scikit-learn","pyyaml","pillow","tqdm",
    "huggingface_hub","omegaconf","timm"], check=False)
subprocess.run([sys.executable,"-m","pip","install","--quiet","-e",str(REPO)], check=False)

import torch, numpy as np
from PIL import Image

sys.path.insert(0, str(REPO))
from dataprov.models.llamagen import LlamaGenIAR
from dataprov.signals import provenance_signals
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    "name":"llamagen","model":"llamagen","image_size":384,"value_range":"pm1",
    "hf_repo":"FoundationVision/LlamaGen",
    "vq_ckpt":"vq_ds16_c2i.pt","gpt_ckpt":"c2i_XL_384.pt",
    "vq_model":"VQ-16","codebook_size":16384,"codebook_embed_dim":8,
    "gpt_model":"GPT-XL","gpt_type":"c2i","num_classes":1000,"cls_token_num":1,
    "downsample_size":16,"token_optim":False,
    "encoder":"original","hf_encoder_repo":"","load_generator":False,
})
dev = torch.device("cuda")
print("Building LlamaGen VQ (original encoder, auto-download)...")
model = LlamaGenIAR(cfg, device=dev)
print("[OK] LlamaGen VQ loaded.")

def load_img(p, size=384):
    im = Image.open(p).convert("RGB").resize((size,size), Image.LANCZOS)
    return torch.tensor(np.array(im),dtype=torch.float32).permute(2,0,1)/127.5 - 1.0

BATCHES=[("WM_3",51,75),("WM_4",76,100),("WM_6",126,150),("WM_7",151,175),("WM_8",176,200)]
print(f"\n{'batch':6} {'src_QL':>10} {'clean_QL':>10}  verdict")
print("-"*44)
for wm,lo,hi in BATCHES:
    sp = sorted((DATA/'watermarked_sources'/wm).glob("*.png"))[:8]
    si = torch.stack([load_img(p) for p in sp]).to(dev)
    ci = torch.stack([load_img(DATA/'clean_targets'/f"{n}.png") for n in range(lo,lo+8)]).to(dev)
    ss = provenance_signals(model, si); cs = provenance_signals(model, ci)
    sql=float(np.mean(ss["quant_loss"])); cql=float(np.mean(cs["quant_loss"]))
    flag="** GENERATED (IAR)! **" if sql < cql*0.6 else ""
    print(f"{wm:6} {sql:10.5f} {cql:10.5f}  {flag}")
print("-"*44)
print("src_QL << clean_QL => LlamaGen-generated => forge via round-trip.")
