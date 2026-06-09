import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    # Fastest cuDNN kernel selection for fixed input size (32×32)
    torch.backends.cudnn.benchmark = True
 
# Hyperparameters 
BATCH_SIZE  = 256        
EPOCHS      = 100
BETA        = 6.0        # TRADES trade-off coefficient (higher → more robust, lower clean)
EPSILON     = 8 / 255  # L-inf perturbation budget (standard CIFAR setting)
STEP_SIZE   = 2 / 255  # PGD step size (= EPSILON / 4, industry standard)
PGD_STEPS   = 7         
NUM_CLASSES = 9
 
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(SCRIPT_DIR, "train.npz")
print(f"Loading dataset from: {dataset_path}")
raw  = np.load(dataset_path)
imgs = raw["images"]                               
lbls = torch.from_numpy(raw["labels"]).long()
print(f"Dataset : {len(lbls)} samples | Label range : {lbls.min()}–{lbls.max()}")
 
 
def augment(img: torch.Tensor) -> torch.Tensor:
    """
    RandomCrop(32, padding=4) + RandomHorizontalFlip on a C×H×W float tensor.
    """
    # Reflect-pad spatial dims by 4 → 40×40
    img = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
    # Random 32×32 crop  (valid top/left range: 0..8 inclusive, i.e. 40-32=8)
    top  = torch.randint(0, 9, (1,)).item()
    left = torch.randint(0, 9, (1,)).item()
    img  = img[:, top : top + 32, left : left + 32]
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        img = img.flip(-1)
    return img
 
 
class CIFAR9Dataset(Dataset):
    """Converts uint8 numpy images on-the-fly to save CPU RAM."""
    def __init__(self, images: np.ndarray, labels: torch.Tensor):
        self.images = images
        self.labels = labels
        
    def __len__(self) -> int:
        return len(self.labels)
 
    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.images[idx].astype(np.float32)) / 255.0
        img = augment(img)
        return img, self.labels[idx]
 
 
loader = DataLoader(
    CIFAR9Dataset(imgs, lbls),
    batch_size        = BATCH_SIZE,
    shuffle           = True,
    num_workers       = 4,      
    pin_memory        = True,   # faster host→device transfers
    drop_last         = True,
    persistent_workers= True,   # avoids worker re-spawn cost every epoch
)
print(f"Batches per epoch : {len(loader)}")
 
# Oonly model.fc is replaced.
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)
 
# Check output shape before spending hours training
model.eval()
with torch.no_grad():
    _dummy = model(torch.randn(2, 3, 32, 32, device=device))
assert _dummy.shape == (2, NUM_CLASSES), f"Wrong output shape: {_dummy.shape}"
print(f"Output shape verified: {list(_dummy.shape)}")
 
# Automatic Mixed Precision 
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
 
def trades_pgd(model: nn.Module, x_nat: torch.Tensor) -> torch.Tensor:
    model.eval()
    # Freeze model weights (we only want ∂loss/∂x_adv, not ∂loss/∂θ)
    for p in model.parameters():
        p.requires_grad_(False)
    # Random initialisation within the epsilon ball
    x_adv = x_nat.detach().clone() + 0.001 * torch.randn_like(x_nat)
    x_adv = x_adv.clamp(0.0, 1.0)
    # Cache natural logits once in fp32 for stable KL computation
    with torch.no_grad():
        logits_nat = model(x_nat).float().detach()
    probs_nat = F.softmax(logits_nat, dim=1)  # target distribution (fixed)
    for _ in range(PGD_STEPS):
        x_adv.requires_grad_(True)
        # Forward pass under AMP (fp16 where safe)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_adv = model(x_adv)
        # KL in fp32 to avoid log-domain underflow
        loss_kl = F.kl_div(
            F.log_softmax(logits_adv.float(), dim=1),
            probs_nat,
            reduction="batchmean",
        )
        loss_kl.backward()
        # Signed gradient step + projection onto L-inf ball
        with torch.no_grad():
            x_adv = x_adv + STEP_SIZE * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_nat + EPSILON), x_nat - EPSILON)
            x_adv = x_adv.clamp(0.0, 1.0)
        x_adv = x_adv.detach()
    # Restore weight gradients for the main training step
    for p in model.parameters():
        p.requires_grad_(True)
    return x_adv
 
 
optimizer = torch.optim.SGD(
    model.parameters(),
    lr           = 0.1,
    momentum     = 0.9,
    weight_decay = 5e-4,
    nesterov     = True,   # Nesterov momentum
)
# Cosine annealing decays LR from 0.1 → ~0; prevents robust overfitting
# in the absence of AWP by naturally shrinking updates late in training.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
# Training Loop 
print(f"\n TRADES Training:")
print(f"   Epochs={EPOCHS} | β={BETA} | ε={EPSILON:.4f} | "
      f"α={STEP_SIZE:.4f} | PGD steps={PGD_STEPS}")
print(f"   Batch size={BATCH_SIZE} | AMP={use_amp}")
 
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
 
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Generate adversarial examples (weights frozen internally)
        x_adv = trades_pgd(model, x)
        # Compute TRADES loss on perturbed and natural inputs
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_nat = model(x)
            logits_adv = model(x_adv)
            # Clean cross-entropy term
            loss_ce = F.cross_entropy(logits_nat, y)
            # Robustness regularisation: KL( f(x) || f(x_adv) )
            # fp32 cast prevents log-domain underflow in fp16 AMP context
            loss_kl = F.kl_div(
                F.log_softmax(logits_adv.float(), dim=1),
                F.softmax(logits_nat.detach().float(), dim=1),
                reduction="batchmean",
            )
            # TRADES objective
            loss = loss_ce + BETA * loss_kl
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    scheduler.step()
    if epoch % 10 == 0 or epoch == 1:
        avg  = epoch_loss / len(loader)
        lr_n = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss={avg:.4f}  lr={lr_n:.5f}")
 
save_path = os.path.join(SCRIPT_DIR, "model_trades.pt")
torch.save(model.state_dict(), save_path)
print(f"\n State dict saved → {save_path}")
