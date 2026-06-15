"""
TML Task 3 — Adversarial Robustness Training (v6)
Team: atml_team034

Philosophy: Mark's code scored 0.607. Don't fix what works.
Only add: checkpointing (to catch the best epoch before robust overfitting)
         + 150 epochs (Mark used 120, loss was still dropping)
         + train on ALL 50k samples (no val split — Mark didn't split either)
         + evaluate on train subset every 10 epochs just for logging

Everything else is identical to Mark's recipe:
  - Pretrained ResNet18 (ImageNet)
  - TRADES BETA=4.0
  - 7 PGD steps, eps=8/255, step_size=2/255
  - Label smoothing 0.1
  - Warmup 5 epochs + cosine LR
  - AMP + Nesterov SGD
  - Batch size 256
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ──────────────────────────────────────────────
# Config — IDENTICAL to Mark's proven settings
# ──────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 150           # Mark used 120; a bit more room
BETA = 4.0             # Mark's value — proven
EPSILON = 8 / 255
STEP_SIZE = 2 / 255
PGD_STEPS = 7          # Mark's value — proven
NUM_CLASSES = 9
WARMUP_EPOCHS = 5
LABEL_SMOOTHING = 0.1
CHECKPOINT_EVERY = 5   # Save more frequently to catch the sweet spot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints_v6")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Data — ALL 50k samples, no val split (like Mark)
# ──────────────────────────────────────────────
dataset_path = os.path.join(SCRIPT_DIR, "train.npz")
print(f"Loading: {dataset_path}", flush=True)
raw = np.load(dataset_path)
imgs = raw["images"]
lbls = torch.from_numpy(raw["labels"]).long()
print(f"Samples: {len(lbls)} | Labels: {lbls.min()}-{lbls.max()}", flush=True)


def augment(img: torch.Tensor) -> torch.Tensor:
    """RandomCrop(32, padding=4) + RandomHorizontalFlip."""
    img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
    top = torch.randint(0, 9, (1,)).item()
    left = torch.randint(0, 9, (1,)).item()
    img = img[:, top:top + 32, left:left + 32]
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)
    return img


class CIFAR9Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: torch.Tensor, train: bool = True):
        self.images = images
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx].astype(np.float32)) / 255.0
        if self.train:
            img = augment(img)
        return img, self.labels[idx]


train_loader = DataLoader(
    CIFAR9Dataset(imgs, lbls, train=True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
    pin_memory=True, drop_last=True, persistent_workers=True,
)

# Eval loader: no augmentation, subset of 5k for speed
eval_indices = np.random.RandomState(42).choice(len(lbls), 5000, replace=False)
eval_loader = DataLoader(
    CIFAR9Dataset(imgs[eval_indices], lbls[eval_indices], train=False),
    batch_size=256, shuffle=False, num_workers=2, pin_memory=True,
)

print(f"Train: {len(lbls)} (full dataset, no split)", flush=True)
print(f"Eval subset: {len(eval_indices)} samples", flush=True)
print(f"Batches/epoch: {len(train_loader)}", flush=True)

# ──────────────────────────────────────────────
# Model — Pretrained ResNet18, only fc replaced
# ──────────────────────────────────────────────
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

model.eval()
with torch.no_grad():
    _check = model(torch.randn(2, 3, 32, 32, device=device))
    assert _check.shape == (2, NUM_CLASSES), f"Bad output shape: {_check.shape}"
print(f"Output shape: {list(_check.shape)} ✓", flush=True)

# ──────────────────────────────────────────────
# AMP
# ──────────────────────────────────────────────
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ──────────────────────────────────────────────
# TRADES PGD — identical to Mark's implementation
# ──────────────────────────────────────────────
def trades_pgd(model, x_nat):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x_adv = x_nat.detach().clone() + 0.001 * torch.randn_like(x_nat)
    x_adv = x_adv.clamp(0.0, 1.0)

    with torch.no_grad():
        logits_nat = model(x_nat).float().detach()
    probs_nat = F.softmax(logits_nat, dim=1)

    for _ in range(PGD_STEPS):
        x_adv.requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_adv = model(x_adv)
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv.float(), dim=1),
            probs_nat,
            reduction="batchmean",
        )
        loss_kl.backward()
        with torch.no_grad():
            x_adv = x_adv + STEP_SIZE * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_nat + EPSILON), x_nat - EPSILON)
            x_adv = x_adv.clamp(0.0, 1.0)
        x_adv = x_adv.detach()

    for p in model.parameters():
        p.requires_grad_(True)
    return x_adv


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_clean(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def evaluate_pgd(model, loader, steps=20):
    """PGD-20 eval (CE-based, not KL — standard evaluation protocol)."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Random start
        x_adv = x.detach() + torch.empty_like(x).uniform_(-EPSILON, EPSILON)
        x_adv = x_adv.clamp(0.0, 1.0)
        for _ in range(steps):
            x_adv.requires_grad_(True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + STEP_SIZE * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - EPSILON), x + EPSILON)
            x_adv = x_adv.clamp(0.0, 1.0)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                correct += (model(x_adv).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ──────────────────────────────────────────────
# Optimizer + LR — identical to Mark's
# ──────────────────────────────────────────────
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9,
    weight_decay=5e-4, nesterov=True,
)


def lr_lambda(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
print(f"\n{'='*70}", flush=True)
print(f"v6: Mark's recipe + checkpointing", flush=True)
print(f"ResNet18 (pretrained) · TRADES β={BETA} · ε={EPSILON:.4f} · "
      f"PGD×{PGD_STEPS} · {EPOCHS} epochs", flush=True)
print(f"batch={BATCH_SIZE} · warmup={WARMUP_EPOCHS}ep · "
      f"label_smooth={LABEL_SMOOTHING} · AMP={use_amp}", flush=True)
print(f"Checkpointing every {CHECKPOINT_EVERY} epochs", flush=True)
print(f"{'='*70}\n", flush=True)

best_score = 0.0
epoch_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        x_adv = trades_pgd(model, x)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_nat = model(x)
            logits_adv = model(x_adv)
            loss_ce = F.cross_entropy(logits_nat, y, label_smoothing=LABEL_SMOOTHING)
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv.float(), dim=1),
                F.softmax(logits_nat.detach().float(), dim=1),
                reduction="batchmean",
            )
            loss = loss_ce + BETA * loss_kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    scheduler.step()
    epoch_time = time.time() - t0
    avg_loss = epoch_loss / len(train_loader)
    lr = scheduler.get_last_lr()[0]

    print(f"Epoch [{epoch:3d}/{EPOCHS}] | Loss: {avg_loss:.4f} | "
          f"LR: {lr:.5f} | Time: {epoch_time:.1f}s", flush=True)

    # Evaluate + checkpoint
    if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
        clean_acc = evaluate_clean(model, eval_loader)
        robust_acc = evaluate_pgd(model, eval_loader)
        score = 0.5 * clean_acc + 0.5 * robust_acc
        print(f"  → Clean: {clean_acc:.4f} | PGD-20: {robust_acc:.4f} | "
              f"Score: {score:.4f}", flush=True)

        # Save checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(),
                       os.path.join(SCRIPT_DIR, "model_best_v6.pt"))
            print(f"  ★ New best! Score={score:.4f} → model_best_v6.pt", flush=True)

# Save final
torch.save(model.state_dict(), os.path.join(SCRIPT_DIR, "model_v6_final.pt"))

total_time = time.time() - epoch_start
print(f"\n{'='*70}", flush=True)
print(f"Training complete in {total_time/3600:.1f}h", flush=True)
print(f"Best score: {best_score:.4f} → model_best_v6.pt", flush=True)
print(f"Submit with MODEL_NAME = 'resnet18'", flush=True)
print(f"{'='*70}", flush=True)