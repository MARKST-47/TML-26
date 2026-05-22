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
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")

# Global configuration constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR100_DIR = "./data"          # Path to CIFAR-100 dataset root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Absolute paths to data targets
TRAIN_IDX_JSON = os.path.join(SCRIPT_DIR, "target_model", "train_main_idx.json")
TARGET_CKPT    = os.path.join(SCRIPT_DIR, "target_model", "weights.safetensors") 
SUSPECT_DIR    = os.path.join(SCRIPT_DIR, "suspect_models")
NUM_SUSPECTS = 360
BATCH_SIZE = 256
PROBE_BATCHES = 4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Executing pipeline on device: {DEVICE}")

# CUSTOM TRANSFORM: EXACT BIASED RANDOM CROPPING
class ExactBiasedCrop(object):
    # As specified in the task description, this transform applies a precise cropping strategy with systematic bias and jitter to replicate the target model's augmentation behavior.
    def __init__(self, size=32, padding=4, bias_x=0.5, bias_y=-0.25, jitter=0.25):
        self.size = size
        self.padding = padding
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter

    def __call__(self, img):
        # Apply reflection padding
        img_padded = F.pad(img, [self.padding]*4, mode='reflect')
        _, h, w = img_padded.shape
        
        # Calculate base center coordinates
        center_x = (w - self.size) // 2
        center_y = (h - self.size) // 2
        
        # Inject systematic bias and randomized jitter
        shift_x = int(self.bias_x * self.padding + np.random.uniform(-self.jitter, self.jitter) * self.padding)
        shift_y = int(self.bias_y * self.padding + np.random.uniform(-self.jitter, self.jitter) * self.padding)
        
        x1 = int(np.clip(center_x + shift_x, 0, w - self.size))
        y1 = int(np.clip(center_y + shift_y, 0, h - self.size))
        
        return img_padded[:, y1:y1+self.size, x1:x1+self.size]

# DATA CORES & LOADERS
def get_cifar100_loaders(train_indices: list, batch_size: int = BATCH_SIZE):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)

    # Base tensor transform to preserve image structure before custom padding operations
    base_transforms = transforms.Compose([transforms.ToTensor()])
    
    full_train = datasets.CIFAR100(CIFAR100_DIR, train=True, download=True, transform=base_transforms)
    full_test  = datasets.CIFAR100(CIFAR100_DIR, train=False, download=True, transform=base_transforms)

    # Apply the target model's precise augmentation configurations manually
    train_subset = Subset(full_train, train_indices)
    
    # Packaged pipeline for evaluation normalization
    norm_transform = transforms.Normalize(mean, std)

    def collate_train(batch):
        xs, ys = zip(*batch)
        cropper = ExactBiasedCrop()
        flipper = transforms.RandomHorizontalFlip(p=0.5)
        processed_xs = []
        for x in xs:
            x_aug = flipper(cropper(x))
            processed_xs.append(norm_transform(x_aug))
        return torch.stack(processed_xs), torch.tensor(ys)

    def collate_test(batch):
        xs, ys = zip(*batch)
        processed_xs = [norm_transform(x) for x in xs]
        return torch.stack(processed_xs), torch.tensor(ys)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False, 
                              collate_fn=collate_train, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(full_test, batch_size=batch_size, shuffle=False, 
                             collate_fn=collate_test, num_workers=2, pin_memory=True)
    return train_loader, test_loader

def get_probe_loaders(batch_size: int = BATCH_SIZE):
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    
    clean_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    clean_ds = datasets.CIFAR100(CIFAR100_DIR, train=False, download=True, transform=clean_tf)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # OOD Dataset generation using random uniform noise to surface decision boundary choices
    noise_xs = torch.rand(batch_size * PROBE_BATCHES, 3, 32, 32)
    noise_ys = torch.randint(0, 100, (batch_size * PROBE_BATCHES,))
    for i in range(len(noise_xs)):
        noise_xs[i] = transforms.Normalize(mean, std)(noise_xs[i])
    noise_ds = torch.utils.data.TensorDataset(noise_xs, noise_ys)
    noise_loader = DataLoader(noise_ds, batch_size=batch_size, shuffle=False)
    
    return clean_loader, noise_loader

# MODEL ARCHITECTURE CONSTRUCTION
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def load_verified_model(path: str):
    model = make_model()
    try:
        state_dict = load_file(path, device="cpu")
    except Exception:
        # Fallback handling for traditional PyTorch pth weights format if needed
        state_dict = torch.load(path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    
    state_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(DEVICE)

# PERMUTATION RE-ALIGNMENT ENGINE (ANTI-MUTATION SHIELD)
def align_and_extract_weights(model: nn.Module) -> dict:
    """
    Normalizes weight-space checks against channel sorting mutations 
    by extracting parameter vectors ordered by L2 filter norms.
    """
    aligned_vectors = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name or param.dim() < 2:
                aligned_vectors[name] = param.detach().cpu().flatten().float()
                continue
            
            # Sort filters by their absolute energy signature to counter indexing mutations
            matrix = param.detach().cpu().float()
            norms = torch.norm(matrix.view(matrix.size(0), -1), dim=1)
            sort_idx = torch.argsort(norms)
            aligned_vectors[name] = matrix[sort_idx].flatten()
    return aligned_vectors

def get_bn_fingerprint(model: nn.Module) -> np.ndarray:
    fingerprint = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.running_mean is not None:
                fingerprint.append(m.running_mean.cpu().float().numpy().flatten())
                fingerprint.append(m.running_var.cpu().float().numpy().flatten())
    return np.concatenate(fingerprint) if fingerprint else np.array([0.0])

# SIGNALS MATHEMATICAL MATRIX
def compute_weight_similarity(target_w, suspect_w, target_bn, suspect_bn) -> float:
    sims = []
    common_keys = [k for k in target_w if k in suspect_w]
    
    for k in common_keys:
        tw = target_w[k].numpy()
        sw = suspect_w[k].numpy()
        if tw.shape == sw.shape:
            sims.append(1.0 - cosine(tw, sw))
            
    if len(target_bn) == len(suspect_bn):
        sims.append(1.0 - cosine(target_bn, suspect_bn))
        
    return float(np.mean(sims)) if sims else 0.0

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    fro_product = np.linalg.norm(Y.T @ X) ** 2
    norm_factor = np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro")
    return float(fro_product / (norm_factor + 1e-12))

def collect_layer_activations(model: nn.Module, loader: DataLoader) -> dict:
    activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, inp, out):
            pooled = out.mean(dim=(2, 3)) if out.dim() == 4 else out
            activations.setdefault(name, []).append(pooled.detach().cpu().float().numpy())
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or "layer4" in name:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            if idx >= PROBE_BATCHES: break
            model(x.to(DEVICE))

    for h in hooks: h.remove()
    return {k: np.concatenate(v, axis=0) for k, v in activations.items()}

def compute_cka_signal(target_model, suspect_model, loader) -> float:
    t_acts = collect_layer_activations(target_model, loader)
    s_acts = collect_layer_activations(suspect_model, loader)
    keys = [k for k in t_acts if k in s_acts]
    
    scores = []
    for k in keys:
        if t_acts[k].shape == s_acts[k].shape:
            scores.append(linear_cka(t_acts[k], s_acts[k]))
    return float(np.mean(scores)) if scores else 0.0

def compute_loss_fingerprint(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if idx >= 15: break  # Standardize sample bounds to manage overhead
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            losses.append(F.cross_entropy(logits, y, reduction="none").cpu().numpy())
    return np.concatenate(losses)

def compute_behavioral_logits(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for idx, (x, _) in enumerate(loader):
            if idx >= PROBE_BATCHES: break
            outputs.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(outputs, axis=0)

# METRIC ENSEMBLE ENGINE
SIGNAL_WEIGHTS = {
    "weight_space": 0.35,
    "cka_clean":    0.20,
    "cka_noise":    0.15,
    "membership":   0.15,
    "behavioral":   0.15
}

def rank_aggregation(matrix_dict: dict) -> np.ndarray:
    n_samples = len(next(iter(matrix_dict.values())))
    master_ranks = np.zeros(n_samples)
    
    for metric_name, raw_values in matrix_dict.items():
        weight = SIGNAL_WEIGHTS.get(metric_name, 0.1)
        # Assign continuous rank scores safely
        assigned_ranks = rankdata(raw_values, method="average")
        master_ranks += weight * (assigned_ranks / n_samples)
        
    # Non-linear scaling optimization designed to prioritize true positives at 5% FPR
    pushed_scores = np.power(master_ranks, 2.5)
    return (pushed_scores - pushed_scores.min()) / (pushed_scores.max() - pushed_scores.min() + 1e-12)

# PIPELINE EXECUTION ENTRYPOINT
def main():
    print("Initializing Multi-Signal Forgery Verification System...")
    
    with open(TRAIN_IDX_JSON, "r") as f:
        train_indices = json.load(f)
    print(f"Indices file loaded cleanly. Evaluated set size: {len(train_indices)} elements")

    train_loader, test_loader = get_cifar100_loaders(train_indices)
    clean_probe_loader, noise_probe_loader = get_probe_loaders()

    print("Extracting baseline references from pristine Target Model...")
    target_model = load_verified_model(TARGET_CKPT)
    target_w = align_and_extract_weights(target_model)
    target_bn = get_bn_fingerprint(target_model)
    
    target_train_losses = compute_loss_fingerprint(target_model, train_loader)
    target_clean_logits = compute_behavioral_logits(target_model, clean_probe_loader)

    # Initialize tracking records
    suspect_ids = list(range(NUM_SUSPECTS))
    metrics_log = {k: [] for k in SIGNAL_WEIGHTS.keys()}

    print(f"\nProcessing loop engaged over {NUM_SUSPECTS} suspect verification candidates...")
    for s_id in suspect_ids:
        filename = f"suspect_{s_id:03d}.safetensors"
        path = os.path.join(SUSPECT_DIR, filename)

        suspect_model = load_verified_model(path)

        # Signal 1: Weight Matrix Similarity Matching
        sw = align_and_extract_weights(suspect_model)
        s_bn = get_bn_fingerprint(suspect_model)
        w_sim = compute_weight_similarity(target_w, sw, target_bn, s_bn)
        metrics_log["weight_space"].append(w_sim)

        # Signal 2 & 3: Multi-Context Activation Parity (CKA)
        cka_c = compute_cka_signal(target_model, suspect_model, clean_probe_loader)
        cka_n = compute_cka_signal(target_model, suspect_model, noise_probe_loader)
        metrics_log["cka_clean"].append(cka_c)
        metrics_log["cka_noise"].append(cka_n)

        # Signal 4: Dataset Membership Inference Rank Agreement
        suspect_train_losses = compute_loss_fingerprint(suspect_model, train_loader)
        corr_val, _ = spearmanr(target_train_losses, suspect_train_losses)
        metrics_log["membership"].append(corr_val if not np.isnan(corr_val) else 0.0)

        # Signal 5: Direct Behavioral Softmax Signature Match
        suspect_clean_logits = compute_behavioral_logits(suspect_model, clean_probe_loader)
        t_preds = target_clean_logits.argmax(axis=1)
        s_preds = suspect_clean_logits.argmax(axis=1)
        ag_rate = float((t_preds == s_preds).mean())
        metrics_log["behavioral"].append(ag_rate)

        if s_id % 40 == 0:
            print(f"[ID {s_id:03d}] metrics tracking: W={w_sim:.3f} | CKA-C={cka_c:.3f} | MIA-Corr={corr_val:.3f}")

        del suspect_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate calibrated ensemble probabilities
    final_scores = rank_aggregation(metrics_log)

    # Standardize output profiles to fulfill submission requirements
    submission_df = pd.DataFrame({
        "id": suspect_ids,
        "score": final_scores
    })
    
    submission_df.to_csv("submission.csv", index=None)
    print("\n[SUCCESS] Matrix operations complete. Submission footprint saved to: submission.csv")

if __name__ == "__main__":
    main()