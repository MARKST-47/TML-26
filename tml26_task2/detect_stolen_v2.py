"""
Stolen Model Detection Pipeline — Improved Version v2
TML Assignment 2, SS2026

Changes over v1 (detect_stolen.py):
1. Added layer4 CKA — catches fine-tuned suspects where weights drift but features don't
2. Added noise agreement rate — ModelDiff signal, strongest for distilled/extracted models
3. Replaced loss_corr (train only) with loss_gap (train_corr - test_corr) — cleaner membership signal
4. Replaced top-1 agreement with top-3 agreement — more granular, fewer false negatives
5. Rebalanced weights accordingly: bn_sim and cka_layer4 promoted, fc_sim kept
6. Raw signals output renamed to raw_signals_v2.csv
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

# Config
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

# Data Loading
MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)

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
    """Fixed Gaussian noise — surfaces decision boundary geometry."""
    torch.manual_seed(SEED)
    noise_x = torch.randn(n_samples, 3, 32, 32)
    noise_y = torch.zeros(n_samples, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(noise_x, noise_y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

# Model
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

# Signal 1: Layer-Weighted Weight Cosine Similarity
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
    return "conv1"

def compute_weighted_cosine(target_model, suspect_model):
    tier_sims = {tier: [] for tier in LAYER_WEIGHTS}
    target_sd  = {k: v.cpu().float().flatten() for k, v in target_model.state_dict().items()}
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
        tier_sims[get_layer_tier(key)].append(sim)

    total_weight, weighted_sum = 0.0, 0.0
    for tier, sims in tier_sims.items():
        if sims:
            w = LAYER_WEIGHTS[tier]
            weighted_sum += w * np.mean(sims)
            total_weight += w
    return weighted_sum / (total_weight + 1e-12)

# Signal 2: FC Layer Direct Comparison
def compute_fc_similarity(target_model, suspect_model):
    t_sd, s_sd = target_model.state_dict(), suspect_model.state_dict()
    sims = []
    for key in ["fc.weight", "fc.bias"]:
        if key in t_sd and key in s_sd:
            tw = t_sd[key].cpu().float().flatten().numpy()
            sw = s_sd[key].cpu().float().flatten().numpy()
            sim = 1.0 - cosine(tw, sw)
            if not np.isnan(sim):
                sims.append(sim)
    return float(np.mean(sims)) if sims else 0.0

# Signal 3: BatchNorm Statistics Fingerprint
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
    result = float(np.mean(sims)) if sims else 0.0
    return result if not np.isnan(result) else 0.0

# Signal 4: Logit Correlation on Test Data
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
    if target_logits.shape != suspect_logits.shape:
        n = min(len(target_logits), len(suspect_logits))
        target_logits  = target_logits[:n]
        suspect_logits = suspect_logits[:n]
    corr, _ = pearsonr(target_logits.flatten(), suspect_logits.flatten())
    return corr if not np.isnan(corr) else 0.0

# Signal 5: Top-3 Agreement Rate
# Replaces top-1 agreement. Checks whether any of the suspect's top-3
# predicted classes overlap with target's top-3. Far more sensitive than
# argmax-only, catching near-stolen models where rank ordering matches
# even when the single top prediction differs by noise.
def compute_top3_agreement(target_logits, suspect_logits):
    n = min(len(target_logits), len(suspect_logits))
    t_top3 = np.argsort(target_logits[:n],  axis=1)[:, -3:]
    s_top3 = np.argsort(suspect_logits[:n], axis=1)[:, -3:]
    matches = [len(set(t) & set(s)) > 0 for t, s in zip(t_top3, s_top3)]
    return float(np.mean(matches))

# Signal 6: Logit Correlation on Noise Data
def compute_noise_logit_corr(target_noise_logits, suspect_noise_logits):
    return compute_logit_correlation(target_noise_logits, suspect_noise_logits)

# Signal 7: Noise Prediction Agreement Rate (ModelDiff) ───────────────
# NEW: Argmax agreement on Gaussian noise inputs.
# Two independently trained models disagree sharply on OOD noise.
# A stolen model inherits the exact decision boundary and agrees at
# rates far above chance — even on inputs neither model was trained on.
# This is the core signal from Shah et al. (ModelDiff, ICML 2023).
# Reuses already-computed noise logits so zero extra forward passes.
def compute_noise_agreement(target_noise_logits, suspect_noise_logits):
    n = min(len(target_noise_logits), len(suspect_noise_logits))
    t_pred = target_noise_logits[:n].argmax(axis=1)
    s_pred = suspect_noise_logits[:n].argmax(axis=1)
    return float((t_pred == s_pred).mean())

# Signal 8: Train–Test Loss Gap (Dataset Inference)
# NEW: Replaces train-only loss correlation.
# Computes Spearman correlation of per-sample losses separately on
# training data and test data, then returns (train_corr - test_corr).
# A stolen model memorizes the same training samples as the target,
# so its train loss profile correlates strongly. An independent model
# shows similar (low) correlation on both splits — no meaningful gap.
def compute_loss_profile(model, loader, max_batches=15):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(DEVICE), y.to(DEVICE)
            losses.append(F.cross_entropy(model(x), y, reduction="none").cpu().numpy())
    return np.concatenate(losses)

def compute_loss_gap(target_train_losses, target_test_losses,
                     suspect_train_losses, suspect_test_losses):
    n_train = min(len(target_train_losses), len(suspect_train_losses))
    n_test  = min(len(target_test_losses),  len(suspect_test_losses))
    train_corr, _ = spearmanr(target_train_losses[:n_train], suspect_train_losses[:n_train])
    test_corr,  _ = spearmanr(target_test_losses[:n_test],   suspect_test_losses[:n_test])
    train_corr = train_corr if not np.isnan(train_corr) else 0.0
    test_corr  = test_corr  if not np.isnan(test_corr)  else 0.0
    return float(train_corr - test_corr)

# Signal 9: Layer4 CKA ────────────────────────────────────────────────
# NEW: Centered Kernel Alignment on layer4 (last residual block) activations.
# Fine-tuning diverges weights but preserves feature-space geometry at deep
# layers because the learned semantic representations are inherited.
# CKA is invariant to orthogonal rotation and scaling, so it catches
# fine-tuned suspects that weight cosine similarity underscores.
# Target activations are computed once before the loop.
def collect_layer4_activations(model, loader, max_batches=8):
    activations = []

    def hook(module, inp, out):
        # Global average pool to collapse spatial dims: (B, 512)
        pooled = out.mean(dim=(2, 3))
        activations.append(pooled.detach().cpu().float().numpy())

    handle = model.layer4[-1].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            model(x.to(DEVICE))
    handle.remove()
    return np.concatenate(activations, axis=0)

def linear_cka(X, Y):
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    num   = np.linalg.norm(Y.T @ X) ** 2
    denom = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(num / (denom + 1e-12))

def compute_layer4_cka(target_acts, suspect_model, loader, max_batches=8):
    suspect_acts = collect_layer4_activations(suspect_model, loader, max_batches)
    n   = min(len(target_acts), len(suspect_acts))
    val = linear_cka(target_acts[:n], suspect_acts[:n])
    return val if not np.isnan(val) else 0.0

# Score Calibration
def calibrate_scores(raw_signals: dict) -> np.ndarray:
    """
    Z-normalize each signal, take weighted sum, pass through sigmoid.
    Weights reflect each signal's discriminative power and coverage
    across attack types (direct copy, fine-tune, distill).
    """
    weights = {
        "weighted_cosine": 0.12,
        "fc_sim":          0.13,
        "bn_sim":          0.18,   # promoted: strongest for direct copies
        "logit_corr":      0.10,
        "top3_agreement":  0.08,   # replaces agreement (was 0.05)
        "noise_logit":     0.08,
        "noise_agreement": 0.12,   # ModelDiff core signal
        "loss_gap":        0.10,   # replaces loss_corr
        "cka_layer4":      0.15,   # catches fine-tuned suspects
    }

    normalized = {}
    for key, values in raw_signals.items():
        arr = np.array(values, dtype=np.float64)
        mu, sigma = arr.mean(), arr.std()
        normalized[key] = np.zeros_like(arr) if sigma < 1e-10 else (arr - mu) / sigma

    n = len(next(iter(raw_signals.values())))
    combined, total_w = np.zeros(n), 0.0
    for key, z_vals in normalized.items():
        w = weights.get(key, 0.1)
        combined += w * z_vals
        total_w  += w
    combined /= total_w

    steepness = 3.0
    return 1.0 / (1.0 + np.exp(-steepness * combined))

# Main Pipeline
def main():
    print("=" * 60)
    print("Stolen Model Detection Pipeline — Improved v2")
    print("=" * 60)

    with open(TRAIN_IDX_JSON) as f:
        train_indices = json.load(f)
    print(f"Training subset: {len(train_indices)} samples")

    test_loader  = get_test_loader()
    train_loader = get_train_loader(train_indices)
    noise_loader = get_noise_loader(n_samples=1024)

    print("Loading target model and computing references...")
    target_model = load_model(TARGET_CKPT)

    # Pre-compute all target reference values once
    target_test_logits   = get_logits(target_model, test_loader,  max_batches=8)
    target_noise_logits  = get_logits(target_model, noise_loader, max_batches=8)
    target_train_losses  = compute_loss_profile(target_model, train_loader, max_batches=15)
    target_test_losses   = compute_loss_profile(target_model, test_loader,  max_batches=8)
    target_layer4_acts   = collect_layer4_activations(target_model, test_loader, max_batches=8)

    print(f"Test logits shape:   {target_test_logits.shape}")
    print(f"Train losses shape:  {target_train_losses.shape}")
    print(f"Layer4 acts shape:   {target_layer4_acts.shape}")

    signals = {
        "weighted_cosine": [],
        "fc_sim":          [],
        "bn_sim":          [],
        "logit_corr":      [],
        "top3_agreement":  [],
        "noise_logit":     [],
        "noise_agreement": [],
        "loss_gap":        [],
        "cka_layer4":      [],
    }

    print(f"\nProcessing {NUM_SUSPECTS} suspects...")
    for sid in range(NUM_SUSPECTS):
        path    = os.path.join(SUSPECT_DIR, f"suspect_{sid:03d}.safetensors")
        suspect = load_model(path)

        # Weight-space signals (no forward pass)
        s1 = compute_weighted_cosine(target_model, suspect)
        s2 = compute_fc_similarity(target_model, suspect)
        s3 = compute_bn_similarity(target_model, suspect)

        # Behavioral signals on test data
        suspect_test_logits = get_logits(suspect, test_loader, max_batches=8)
        s4 = compute_logit_correlation(target_test_logits, suspect_test_logits)
        s5 = compute_top3_agreement(target_test_logits, suspect_test_logits)

        # Behavioral signals on noise (reuses logits — no extra pass for noise_agreement)
        suspect_noise_logits = get_logits(suspect, noise_loader, max_batches=8)
        s6 = compute_noise_logit_corr(target_noise_logits, suspect_noise_logits)
        s7 = compute_noise_agreement(target_noise_logits, suspect_noise_logits)

        # Loss gap signal
        suspect_train_losses = compute_loss_profile(suspect, train_loader, max_batches=15)
        suspect_test_losses  = compute_loss_profile(suspect, test_loader,  max_batches=8)
        s8 = compute_loss_gap(target_train_losses, target_test_losses,
                              suspect_train_losses, suspect_test_losses)

        # Layer4 CKA
        s9 = compute_layer4_cka(target_layer4_acts, suspect, test_loader, max_batches=8)

        signals["weighted_cosine"].append(s1)
        signals["fc_sim"].append(s2)
        signals["bn_sim"].append(s3)
        signals["logit_corr"].append(s4)
        signals["top3_agreement"].append(s5)
        signals["noise_logit"].append(s6)
        signals["noise_agreement"].append(s7)
        signals["loss_gap"].append(s8)
        signals["cka_layer4"].append(s9)

        if sid % 20 == 0:
            print(f"  [{sid:03d}/360] WC={s1:.4f} FC={s2:.4f} BN={s3:.4f} "
                  f"LogC={s4:.4f} T3={s5:.3f} NL={s6:.4f} "
                  f"NA={s7:.3f} Gap={s8:.4f} CKA={s9:.4f}")

        del suspect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save raw signals
    signals_df = pd.DataFrame(signals)
    signals_df.index.name = "id"
    signals_df.to_csv("raw_signals_v2.csv")
    print("\nRaw signals saved to raw_signals_v2.csv")

    print("\n── Signal Statistics ──")
    for key in signals:
        arr = np.array(signals[key])
        print(f"  {key:18s}: mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"min={arr.min():.4f}  max={arr.max():.4f}")

    final_scores = calibrate_scores(signals)

    submission = pd.DataFrame({"id": list(range(NUM_SUSPECTS)), "score": final_scores})
    submission.to_csv("submission.csv", index=False)
    print(f"\nSubmission saved to submission.csv")
    print(f"Score range: [{final_scores.min():.4f}, {final_scores.max():.4f}]")
    print(f"Median score: {np.median(final_scores):.4f}")

    top_idx = np.argsort(final_scores)[::-1][:20]
    print(f"\nTop-20 most suspicious models:")
    for rank, idx in enumerate(top_idx):
        print(f"  #{rank+1}: suspect_{idx:03d} (score={final_scores[idx]:.4f})")

if __name__ == "__main__":
    main()