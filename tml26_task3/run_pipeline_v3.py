"""
TML Task 3 — Adversarial Robustness Training (v3)
Team: atml_team034
Approach: TRADES + AWP with standard ResNet18 (only fc replaced), 
          lower LR + AWP gamma for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
NUM_CLASSES = 9
BATCH_SIZE = 128
EPOCHS = 200
LR = 0.05             # Lower than v2 for stability
WEIGHT_DECAY = 5e-4
BETA = 6.0             # TRADES KL weight
AWP_GAMMA = 0.005      # Lower than v2 for stability
PGD_STEPS = 7
PGD_STEP_SIZE = 2/255
PGD_EPSILON = 8/255
CHECKPOINT_EVERY = 10
VAL_SPLIT = 2000

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Data loading with augmentation
# ──────────────────────────────────────────────
dataset_path = os.path.join(SCRIPT_DIR, "train.npz")
print(f"Loading dataset from: {dataset_path}", flush=True)
data = np.load(dataset_path)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

full_dataset = TensorDataset(images, labels)
train_dataset, val_dataset = random_split(
    full_dataset, [len(full_dataset) - VAL_SPLIT, VAL_SPLIT],
    generator=torch.Generator().manual_seed(42)
)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False,
                        num_workers=2, pin_memory=True)

print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}", flush=True)
print(f"Image shape: {images.shape}, Label range: {labels.min().item()}-{labels.max().item()}", flush=True)

# ──────────────────────────────────────────────
# Model — Standard ResNet18, ONLY fc replaced
# ──────────────────────────────────────────────
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Sanity check
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32).to(device))
    assert out.shape == (1, NUM_CLASSES), f"Bad output shape: {out.shape}"
print(f"Model output shape: {out.shape} ✓", flush=True)

# ──────────────────────────────────────────────
# TRADES adversarial example generation
# ──────────────────────────────────────────────
def trades_generate_adv(model, x_natural, step_size=PGD_STEP_SIZE,
                        epsilon=PGD_EPSILON, perturb_steps=PGD_STEPS):
    model.eval()
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.no_grad():
        logits_natural = model(x_natural)

    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        loss_kl = F.kl_div(
            F.log_softmax(model(x_adv), dim=1),
            F.softmax(logits_natural, dim=1),
            reduction='batchmean'
        )
        grad = torch.autograd.grad(loss_kl, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()

# ──────────────────────────────────────────────
# AWP (Adversarial Weight Perturbation)
# ──────────────────────────────────────────────
class AWP:
    def __init__(self, model, gamma=AWP_GAMMA):
        self.model = model
        self.gamma = gamma
        self.backup = {}

    def perturb(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and "weight" in name:
                    self.backup[name] = param.data.clone()
                    w_norm = torch.norm(param.data)
                    g_norm = torch.norm(param.grad)
                    if w_norm > 0 and g_norm > 0:
                        param.data.add_(self.gamma * w_norm * (param.grad / g_norm))

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup.clear()

# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total

def evaluate_pgd(model, loader, epsilon=8/255, step_size=2/255, steps=20):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = x.detach() + torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total

# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
awp = AWP(model, gamma=AWP_GAMMA)

print(f"\n{'='*70}", flush=True)
print(f"Starting TRADES + AWP training | Standard ResNet18 | {EPOCHS} epochs", flush=True)
print(f"PGD: eps={PGD_EPSILON:.4f}, steps={PGD_STEPS}, step_size={PGD_STEP_SIZE:.4f}", flush=True)
print(f"LR={LR}, BETA={BETA}, AWP_GAMMA={AWP_GAMMA}", flush=True)
print(f"{'='*70}\n", flush=True)

best_score = 0.0
epoch_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    t0 = time.time()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x = train_transform(x)

        # --- Step 1: Generate TRADES adversarial examples ---
        x_adv = trades_generate_adv(model, x)

        # --- Step 2: Compute proxy loss + AWP perturbation ---
        model.train()
        optimizer.zero_grad()
        logits_nat = model(x)
        logits_adv = model(x_adv)
        loss_ce = F.cross_entropy(logits_nat, y)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_nat.detach(), dim=1),
            reduction='batchmean'
        )
        loss_proxy = loss_ce + BETA * loss_kl
        loss_proxy.backward()

        # --- Step 3: AWP weight perturbation ---
        awp.perturb()

        # --- Step 4: True loss on perturbed weights ---
        optimizer.zero_grad()
        logits_nat_p = model(x)
        logits_adv_p = model(x_adv)
        loss_ce_p = F.cross_entropy(logits_nat_p, y)
        loss_kl_p = F.kl_div(
            F.log_softmax(logits_adv_p, dim=1),
            F.softmax(logits_nat_p.detach(), dim=1),
            reduction='batchmean'
        )
        loss_final = loss_ce_p + BETA * loss_kl_p
        loss_final.backward()

        # --- Step 5: Restore weights + SGD step ---
        awp.restore()
        optimizer.step()
        train_loss += loss_final.item()

    scheduler.step()
    epoch_time = time.time() - t0
    avg_loss = train_loss / len(train_loader)

    print(f"Epoch [{epoch:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.5f} | Time: {epoch_time:.1f}s", flush=True)

    if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
        clean_acc = evaluate_clean(model, val_loader)
        robust_acc = evaluate_pgd(model, val_loader)
        score = 0.5 * clean_acc + 0.5 * robust_acc
        print(f"  → Val Clean: {clean_acc:.4f} | Val PGD-20: {robust_acc:.4f} | "
              f"Score: {score:.4f}", flush=True)

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        if score > best_score:
            best_score = score
            best_path = os.path.join(SCRIPT_DIR, "model_best_v3.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best! Score={score:.4f} saved to model_best_v3.pt", flush=True)

torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_v3.pt"))
total_time = time.time() - epoch_start
print(f"\n{'='*70}", flush=True)
print(f"Training complete in {total_time/3600:.1f}h | Best score: {best_score:.4f}", flush=True)
print(f"Final model → model_v3.pt | Best model → model_best_v3.pt", flush=True)
print(f"{'='*70}", flush=True)