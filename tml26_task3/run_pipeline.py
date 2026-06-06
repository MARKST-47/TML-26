import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The dataset is provided as a .npz file (compressed numpy archive)
data = np.load("train.npz")
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

print("Dataset size:", len(dataset))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

NUM_CLASSES = 9

model = resnet50(weights=None)

# Overwrite the first layer: standard ResNet50 downsamples 32x32 images aggressively.
# We change the 7x7 conv to 3x3 (stride 1) and bypass the MaxPool layer entirely.
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(device)

model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32).to(device))
print("Output shape:", out.shape)

def trades_loss_generation(model, x_natural, step_size=2/255, epsilon=8/255, perturb_steps=10):
    """Generates adversarial examples maximizing the KL-divergence (TRADES method)."""
    model.eval()
    # Random initialization
    x_adv = x_natural.clone().detach() + 0.001 * torch.randn(x_natural.shape).to(device)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    with torch.no_grad():
        logits_natural = model(x_natural)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(
                F.log_softmax(model(x_adv), dim=1),
                F.softmax(logits_natural, dim=1),
                reduction='batchmean'
            )
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        # Projection step onto the L_inf epsilon-ball
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0) 
    return x_adv.detach()


class AdvWeightPerturb:
    """Implements Adversarial Weight Perturbation (AWP) to flatten the loss landscape."""
    def __init__(self, model, proxy_optimizer, gamma=0.01):
        self.model = model
        self.proxy_optimizer = proxy_optimizer
        self.gamma = gamma
        self.backup = {}

    def perturb(self):
        """Modifies weights in the direction that maximizes the adversarial loss."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None and "weight" in name:
                    self.backup[name] = param.data.clone()
                    norm = torch.norm(param.data)
                    grad_norm = torch.norm(param.grad)
                    if norm != 0 and grad_norm != 0:
                        # Perturb proportional to the weight scale
                        r_at = self.gamma * norm * (param.grad / grad_norm)
                        param.data.add_(r_at)

    def restore(self):
        """Restores original weights before performing the true SGD optimization step."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup.clear()

EPOCHS = 200
BETA = 6.0  # TRADES tradeoff parameter coefficients (balances clean vs robust)
AWP_GAMMA = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
proxy_optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # Dummy proxy step for AWP
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
awp_adversary = AdvWeightPerturb(model, proxy_optimizer, gamma=AWP_GAMMA)

print("\nStarting Training (TRADES + AWP)...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        # Step A: Generate adversarial examples based on current weights
        x_adv = trades_loss_generation(model, x, step_size=2/255, epsilon=8/255, perturb_steps=10)
        # Step B: Calculate proxy gradients to find adversarial weight space directions
        model.train()
        proxy_optimizer.zero_grad()
        logits_nat = model(x)
        logits_adv = model(x_adv)
        loss_ce = F.cross_entropy(logits_nat, y)
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1), reduction='batchmean')
        loss_trades = loss_ce + BETA * loss_kl
        # Step C: Perturb weights via AWP engine
        loss_trades.backward()
        awp_adversary.perturb()
        # Step D: Recalculate true robust loss on the perturbed weights
        optimizer.zero_grad()
        logits_nat_perturbed = model(x)
        logits_adv_perturbed = model(x_adv)
        loss_ce_p = F.cross_entropy(logits_nat_perturbed, y)
        loss_kl_p = F.kl_div(F.log_softmax(logits_adv_perturbed, dim=1), F.softmax(logits_nat_perturbed, dim=1), reduction='batchmean')
        loss_final = loss_ce_p + BETA * loss_kl_p
        # Step E: Propagate loss, restore original weights, update parameters
        loss_final.backward()
        awp_adversary.restore()
        optimizer.step()
        train_loss += loss_final.item()
    scheduler.step()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{EPOCHS}] | Average TRADES Loss: {train_loss / len(loader):.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

torch.save(model.state_dict(), "model.pt")
print("\nTraining completed successfully. Model state saved to 'model.pt'.")