import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # fastest cuDNN kernels for fixed 32×32
 
BATCH_SIZE = 256       
EPOCHS = 120
BETA = 4.0       
EPSILON = 8 / 255
STEP_SIZE = 2 / 255
PGD_STEPS = 7
NUM_CLASSES = 9
WARMUP_EPOCHS = 5         # linear warmup; important with pretrained weights
LABEL_SMOOTHING = 0.1       # applied only to CE term

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "train.npz")
print(f"Loading: {dataset_path}")
raw = np.load(dataset_path)
imgs = raw["images"]                       # uint8 numpy, (N, 3, 32, 32)
lbls = torch.from_numpy(raw["labels"]).long()
print(f"Samples: {len(lbls)} | Labels: {lbls.min()}–{lbls.max()}")
 
 
def augment(img: torch.Tensor) -> torch.Tensor:
    """RandomCrop(32, padding=4) + RandomHorizontalFlip — pure tensor ops."""
    img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
    top = torch.randint(0, 9, (1,)).item()   # 40−32+1 = 9 valid positions
    left = torch.randint(0, 9, (1,)).item()
    img = img[:, top:top + 32, left:left + 32]
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)
    return img
 
 
class CIFAR9Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: torch.Tensor):
        self.images = images
        self.labels = labels
 
    def __len__(self) -> int:
        return len(self.labels)
 
    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx].astype(np.float32)) / 255.0
        return augment(img), self.labels[idx]
 
 
loader = DataLoader(
    CIFAR9Dataset(imgs, lbls),
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 4,
    pin_memory = True,
    drop_last = True,
    persistent_workers = True,
)
print(f"Batches/epoch: {len(loader)}")
 
model = resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)
 
# Sanity check 
model.eval()
with torch.no_grad():
    _check = model(torch.randn(2, 3, 32, 32, device=device))
assert _check.shape == (2, NUM_CLASSES), f"Bad output shape: {_check.shape}"
print(f"Output shape: {list(_check.shape)}")
 
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
 
def trades_pgd(model: nn.Module, x_nat: torch.Tensor) -> torch.Tensor:
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
        # KL in fp32 — prevents log-domain underflow inside fp16 autocast
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
 
 
optimizer = torch.optim.SGD(
    model.parameters(),
    lr           = 0.1,
    momentum     = 0.9,
    weight_decay = 5e-4,
    nesterov     = True,
)
 
def lr_lambda(epoch: int) -> float:
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS         
    progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1.0 + math.cos(math.pi * progress))  
 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
 
print(f"\n── ResNet18 · TRADES β={BETA} · ε={EPSILON:.4f} · "
      f"PGD×{PGD_STEPS} · pretrained · epochs={EPOCHS} · AMP={use_amp}")
print(f"── batch={BATCH_SIZE} · warmup={WARMUP_EPOCHS}ep · "
      f"label_smooth={LABEL_SMOOTHING}\n")
 
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Step 1 — generate adversarial examples (weights frozen internally)
        x_adv = trades_pgd(model, x)
        # Step 2 — TRADES loss on natural + adversarial inputs
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_nat = model(x)
            logits_adv = model(x_adv)
            # Label smoothing on the clean CE term — free +0.5–1% clean acc
            loss_ce = F.cross_entropy(logits_nat, y, label_smoothing=LABEL_SMOOTHING)
            # Robustness term: KL( f(x) ∥ f(x_adv) ) in fp32
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
    if epoch % 10 == 0 or epoch == 1:
        avg = epoch_loss / len(loader)
        lr  = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss={avg:.4f}  lr={lr:.5f}")
 

print("\nRunning post-training evaluation (PGD-20, no restarts)...")
 
eval_loader = DataLoader(
    CIFAR9Dataset(imgs, lbls),   
    batch_size  = 256,
    shuffle     = False,
    num_workers = 4,
    pin_memory  = True,
)
 
def pgd_eval(model, x, y, steps=20):
    """Standard PGD for evaluation: maximises CE (not KL)."""
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-EPSILON, EPSILON)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + STEP_SIZE * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + EPSILON), x - EPSILON)
            x_adv = x_adv.clamp(0.0, 1.0)
        x_adv = x_adv.detach()
    for p in model.parameters(): p.requires_grad_(True)
    return x_adv
 
model.eval()
n_clean = n_robust = n_total = 0
MAX_EVAL_SAMPLES = 10_000   
 
for x, y in eval_loader:
    if n_total >= MAX_EVAL_SAMPLES:
        break
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        n_clean  += (model(x).argmax(1) == y).sum().item()
    x_adv = pgd_eval(model, x, y, steps=20)
    with torch.no_grad():
        n_robust += (model(x_adv).argmax(1) == y).sum().item()
    n_total += len(y)
 
clean_acc = n_clean / n_total
robust_acc = n_robust / n_total
print(f"\n  Samples evaluated : {n_total}")
print(f"Clean accuracy    : {clean_acc:.3f}  ({clean_acc*100:.1f}%)")
print(f"Robust accuracy   : {robust_acc:.3f}  ({robust_acc*100:.1f}%)")
print(f"Est. unified score: {(clean_acc + robust_acc) / 2:.3f}")
 
save_path = os.path.join(SCRIPT_DIR, "model_trades_pgd.pt")
torch.save(model.state_dict(), save_path)
print(f"\n State dict → {save_path}")
print("Set MODEL_NAME = 'resnet18' in submission.py")

