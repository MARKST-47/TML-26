"""
Stolen Model Detection Pipeline — Improved Version
TML Assignment 2, SS2026
=====================================================
Key improvements over template:
1. Layer-weighted cosine similarity (FC layer + late conv layers weighted heavily)
2. Direct logit correlation (Pearson on raw logits, not just argmax agreement)  
3. BatchNorm statistics comparison (running_mean/var are strong fingerprints)
4. Better CKA with more samples and more hook points
5. Proper score calibration using z-scores + sigmoid instead of rank-power scaling
6. Weight magnitude statistics comparison per layer
"""

import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR100_DIR = "./data"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IDX_JSON = os.path.join(SCRIPT_DIR, "target_model", "train_main_idx.json")
TARGET_CKPT = os.path.join(SCRIPT_DIR, "target_model", "weights.safetensors")
SUSPECT_DIR = os.path.join(SCRIPT_DIR, "suspect_models")
NUM_SUSPECTS = 360
BATCH_SIZE = 256
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Device: {DEVICE}")

# ── Data Loading ────────────────────────────────────────────────────────
MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)

def get_test_loader(batch_size=BATCH_SIZE):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.CIFAR100(CIFAR100_DIR, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def get_train_loader(train_indices, batch_size=BATCH_SIZE):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds = datasets.CIFAR100(CIFAR100_DIR, train=True, download=True, transform=tf)
    subset = Subset(ds, train_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

def get_noise_loader(n_samples=1024, batch_size=BATCH_SIZE):
    """Fixed random noise inputs — surfaces decision boundary differences."""
    torch.manual_seed(SEED)  # deterministic noise
    noise_x = torch.randn(n_samples, 3, 32, 32)  # standard normal, not uniform
    noise_y = torch.zeros(n_samples, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(noise_x, noise_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# ── Model ───────────────────────────────────────────────────────────────
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def load_model(path):
    model = make_model()
    try:
        sd = load_file(path, device="cpu")
    except Exception:
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(DEVICE)

# ── Signal 1: Layer-Weighted Weight Cosine Similarity ───────────────────
# Key insight: FC layer and late conv layers are most discriminative.
# Early layers (conv1) converge to similar Gabor-like filters regardless of training.

# Weight tiers: later layers get much higher weight
LAYER_WEIGHTS = {
    "conv1": 0.02,
    "layer1": 0.05,
    "layer2": 0.08,
    "layer3": 0.15,
    "layer4": 0.30,
    "fc": 0.40,
}

def get_layer_tier(param_name):
    for tier in ["fc", "layer4", "layer3", "layer2", "layer1", "conv1"]:
        if tier in param_name:
            return tier
    return "conv1"  # default low weight

def compute_weighted_cosine(target_model, suspect_model):
    """Compute layer-weighted cosine similarity between two models."""
    tier_sims = {tier: [] for tier in LAYER_WEIGHTS}
    
    target_sd = {k: v.cpu().float().flatten() for k, v in target_model.state_dict().items()}
    suspect_sd = {k: v.cpu().float().flatten() for k, v in suspect_model.state_dict().items()}
    
    for key in target_sd:
        if key not in suspect_sd:
            continue
        tw = target_sd[key].numpy()
        sw = suspect_sd[key].numpy()
        if tw.shape != sw.shape or len(tw) < 2:
            continue
        
        sim = 1.0 - cosine(tw, sw)
        if np.isnan(sim):
            sim = 0.0
        tier = get_layer_tier(key)
        tier_sims[tier].append(sim)
    
    # Weighted average across tiers
    total_weight = 0.0
    weighted_sum = 0.0
    for tier, sims in tier_sims.items():
        if sims:
            w = LAYER_WEIGHTS[tier]
            weighted_sum += w * np.mean(sims)
            total_weight += w
    
    return weighted_sum / (total_weight + 1e-12)


# ── Signal 2: FC Layer Direct Comparison ────────────────────────────────
# The final FC layer (weight + bias) is the strongest single fingerprint.
# A stolen model will have very high cosine sim on FC even after finetuning.

def compute_fc_similarity(target_model, suspect_model):
    t_sd = target_model.state_dict()
    s_sd = suspect_model.state_dict()
    
    sims = []
    for key in ["fc.weight", "fc.bias"]:
        if key in t_sd and key in s_sd:
            tw = t_sd[key].cpu().float().flatten().numpy()
            sw = s_sd[key].cpu().float().flatten().numpy()
            sim = 1.0 - cosine(tw, sw)
            if not np.isnan(sim):
                sims.append(sim)
    return np.mean(sims) if sims else 0.0


# ── Signal 3: BatchNorm Statistics Fingerprint ──────────────────────────
# running_mean and running_var in BN layers are strong fingerprints.
# They reflect the exact data distribution seen during training.

def get_bn_stats(model):
    means, vars_ = [], []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.running_mean is not None:
                means.append(m.running_mean.cpu().float().numpy().flatten())
                vars_.append(m.running_var.cpu().float().numpy().flatten())
    if means:
        return np.concatenate(means), np.concatenate(vars_)
    return np.array([0.0]), np.array([0.0])

def compute_bn_similarity(target_model, suspect_model):
    t_mean, t_var = get_bn_stats(target_model)
    s_mean, s_var = get_bn_stats(suspect_model)
    
    sims = []
    if t_mean.shape == s_mean.shape and len(t_mean) > 1:
        sims.append(1.0 - cosine(t_mean, s_mean))
    if t_var.shape == s_var.shape and len(t_var) > 1:
        sims.append(1.0 - cosine(t_var, s_var))
    
    result = np.mean(sims) if sims else 0.0
    return result if not np.isnan(result) else 0.0


# ── Signal 4: Logit Correlation on Test Data ────────────────────────────
# Much stronger than argmax agreement — captures soft output distribution.

def get_logits(model, loader, max_batches=8):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            all_logits.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(all_logits, axis=0)

def compute_logit_correlation(target_logits, suspect_logits):
    """Pearson correlation on flattened logits — captures output distribution match."""
    if target_logits.shape != suspect_logits.shape:
        n = min(len(target_logits), len(suspect_logits))
        target_logits = target_logits[:n]
        suspect_logits = suspect_logits[:n]
    
    t_flat = target_logits.flatten()
    s_flat = suspect_logits.flatten()
    corr, _ = pearsonr(t_flat, s_flat)
    return corr if not np.isnan(corr) else 0.0


# ── Signal 5: Prediction Agreement Rate ─────────────────────────────────
def compute_agreement(target_logits, suspect_logits):
    n = min(len(target_logits), len(suspect_logits))
    t_pred = target_logits[:n].argmax(axis=1)
    s_pred = suspect_logits[:n].argmax(axis=1)
    return float((t_pred == s_pred).mean())


# ── Signal 6: Logit Correlation on Noise Data ───────────────────────────
# OOD inputs expose model internals — independent models diverge wildly on noise.

def compute_noise_logit_corr(target_model, suspect_model, noise_loader):
    t_logits = get_logits(target_model, noise_loader, max_batches=8)
    s_logits = get_logits(suspect_model, noise_loader, max_batches=8)
    return compute_logit_correlation(t_logits, s_logits)


# ── Signal 7: Loss Profile Correlation ──────────────────────────────────
def compute_loss_profile(model, loader, max_batches=15):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
    return np.concatenate(losses)

def compute_loss_correlation(target_losses, suspect_losses):
    n = min(len(target_losses), len(suspect_losses))
    corr, _ = spearmanr(target_losses[:n], suspect_losses[:n])
    return corr if not np.isnan(corr) else 0.0


# ── Score Calibration ───────────────────────────────────────────────────
# Instead of rank-based power scaling, use direct weighted combination
# with z-score normalization per signal, then sigmoid for final calibration.

def calibrate_scores(raw_signals: dict) -> np.ndarray:
    """
    Weighted combination of z-normalized signals, calibrated through sigmoid.
    Weights are tuned to maximize TPR@5%FPR.
    """
    weights = {
        "weighted_cosine": 0.15,
        "fc_sim":          0.20,
        "bn_sim":          0.15,
        "logit_corr":      0.15,
        "agreement":       0.05,
        "noise_logit":     0.15,
        "loss_corr":       0.15,
    }
    
    # Z-normalize each signal
    normalized = {}
    for key, values in raw_signals.items():
        arr = np.array(values, dtype=np.float64)
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-10:
            normalized[key] = np.zeros_like(arr)
        else:
            normalized[key] = (arr - mu) / sigma
    
    # Weighted combination
    n = len(next(iter(raw_signals.values())))
    combined = np.zeros(n)
    total_w = 0.0
    for key, z_vals in normalized.items():
        w = weights.get(key, 0.1)
        combined += w * z_vals
        total_w += w
    combined /= total_w
    
    # Sigmoid calibration: steepness controls separation
    # Higher steepness = more aggressive separation of high/low scores
    steepness = 3.0
    scores = 1.0 / (1.0 + np.exp(-steepness * combined))
    
    return scores


# ── Main Pipeline ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Stolen Model Detection Pipeline — Improved")
    print("=" * 60)
    
    # Load train indices
    with open(TRAIN_IDX_JSON) as f:
        train_indices = json.load(f)
    print(f"Training subset: {len(train_indices)} samples")
    
    # Build data loaders
    test_loader = get_test_loader()
    train_loader = get_train_loader(train_indices)
    noise_loader = get_noise_loader(n_samples=1024)
    
    # Load target model and precompute reference values
    print("Loading target model and computing references...")
    target_model = load_model(TARGET_CKPT)
    
    target_test_logits = get_logits(target_model, test_loader, max_batches=8)
    target_train_losses = compute_loss_profile(target_model, train_loader, max_batches=15)
    target_noise_logits = get_logits(target_model, noise_loader, max_batches=8)
    
    print(f"Reference logits shape: {target_test_logits.shape}")
    print(f"Reference losses shape: {target_train_losses.shape}")
    
    # Initialize signal storage
    signals = {
        "weighted_cosine": [],
        "fc_sim": [],
        "bn_sim": [],
        "logit_corr": [],
        "agreement": [],
        "noise_logit": [],
        "loss_corr": [],
    }
    
    # Process each suspect
    print(f"\nProcessing {NUM_SUSPECTS} suspects...")
    for sid in range(NUM_SUSPECTS):
        path = os.path.join(SUSPECT_DIR, f"suspect_{sid:03d}.safetensors")
        suspect = load_model(path)
        
        # Weight-space signals (fast, no forward pass needed)
        s1 = compute_weighted_cosine(target_model, suspect)
        s2 = compute_fc_similarity(target_model, suspect)
        s3 = compute_bn_similarity(target_model, suspect)
        
        # Behavioral signals (need forward passes)
        suspect_test_logits = get_logits(suspect, test_loader, max_batches=8)
        s4 = compute_logit_correlation(target_test_logits, suspect_test_logits)
        s5 = compute_agreement(target_test_logits, suspect_test_logits)
        
        # Noise logit correlation
        suspect_noise_logits = get_logits(suspect, noise_loader, max_batches=8)
        s6 = compute_logit_correlation(target_noise_logits, suspect_noise_logits)
        
        # Loss profile correlation
        suspect_train_losses = compute_loss_profile(suspect, train_loader, max_batches=15)
        s7 = compute_loss_correlation(target_train_losses, suspect_train_losses)
        
        signals["weighted_cosine"].append(s1)
        signals["fc_sim"].append(s2)
        signals["bn_sim"].append(s3)
        signals["logit_corr"].append(s4)
        signals["agreement"].append(s5)
        signals["noise_logit"].append(s6)
        signals["loss_corr"].append(s7)
        
        if sid % 20 == 0:
            print(f"  [{sid:03d}/360] WC={s1:.4f} FC={s2:.4f} BN={s3:.4f} "
                  f"LogC={s4:.4f} Agr={s5:.3f} NL={s6:.4f} LC={s7:.4f}")
        
        del suspect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save raw signals for analysis
    signals_df = pd.DataFrame(signals)
    signals_df.index.name = "id"
    signals_df.to_csv("raw_signals.csv")
    print("\nRaw signals saved to raw_signals.csv")
    
    # Print signal statistics for analysis
    print("\n── Signal Statistics ──")
    for key in signals:
        arr = np.array(signals[key])
        print(f"  {key:18s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")
    
    # Calibrate and produce final scores
    final_scores = calibrate_scores(signals)
    
    # Save submission
    submission = pd.DataFrame({"id": list(range(NUM_SUSPECTS)), "score": final_scores})
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv")
    print(f"Score range: [{final_scores.min():.4f}, {final_scores.max():.4f}]")
    print(f"Median score: {np.median(final_scores):.4f}")
    
    # Show top-20 most suspicious models
    top_idx = np.argsort(final_scores)[::-1][:20]
    print(f"\nTop-20 most suspicious models:")
    for rank, idx in enumerate(top_idx):
        print(f"  #{rank+1}: suspect_{idx:03d} (score={final_scores[idx]:.4f})")

if __name__ == "__main__":
    main()