import subprocess
import sys

# Installation
try:
    from xgboost import XGBClassifier

    USE_XGB = True
except ImportError:
    print("XGBoost not found, attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
        from xgboost import XGBClassifier

        USE_XGB = True
        print("XGBoost installed successfully.")
    except Exception as e:
        USE_XGB = False
        print(f"Installation failed: {e}. Falling back to GradientBoostingClassifier.")

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import csv
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve

BASE = Path(__file__).parent
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]


# Dataset
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids, self.imgs, self.labels = [], [], []
        self.transform = transform

    def __getitem__(self, i):
        img = self.transform(self.imgs[i]) if self.transform else self.imgs[i]
        return self.ids[i], img, self.labels[i]

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, i):
        id_, img, label = super().__getitem__(i)
        return id_, img, label, self.membership[i]


# Model loader
def load_resnet(path, device):
    m = resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc = torch.nn.Linear(512, 9)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


# Feature extraction
def log_odds(logits, label):
    p = F.softmax(logits, dim=-1)[label].clamp(1e-7, 1 - 1e-7)
    return (p / (1 - p)).log().item()


def get_entropy(logits):
    p = F.softmax(logits, dim=-1)
    return (-p * p.log()).sum().item()


def get_margin(logits):
    p = F.softmax(logits, dim=-1)
    sorted_p = p.sort(descending=True).values
    return (sorted_p[0] - sorted_p[1]).item()


tta_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ]
)


@torch.no_grad()
def extract_features(img_tensor, label, target, ref, shadows, device, n_tta=32):
    # Higher TTA (32) is essential for TPR@5%FPR stability
    imgs = torch.stack([tta_tf(img_tensor) for _ in range(n_tta)]).to(device)

    t_logits = target(imgs).mean(0)
    phi_t = log_odds(t_logits, label)

    r_logits = ref(imgs).mean(0)
    phi_r = log_odds(r_logits, label)

    shadow_phis = []
    for s in shadows:
        s_logits = s(imgs).mean(0)
        shadow_phis.append(log_odds(s_logits, label))

    shadow_mean = np.mean(shadow_phis)
    shadow_std = np.std(shadow_phis) + 1e-8
    lira_z = (phi_t - shadow_mean) / shadow_std

    # Return only the most robust features
    return [
        phi_t,
        phi_t - phi_r,  # Reference signal
        lira_z,  # LiRA signal
        get_entropy(t_logits),
        label,  # Add the class ID as a feature
    ]


def tpr_at_fpr(scores, labels, target_fpr=0.05):
    """Local validation metric — same as leaderboard."""
    labels = np.array(labels)
    scores = np.array(scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    # find TPR at closest FPR <= target_fpr
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

print("Loading datasets...")
pub_ds = torch.load(BASE / "pub.pt", weights_only=False)
priv_ds = torch.load(BASE / "priv.pt", weights_only=False)
pub_ds.transform = transform
priv_ds.transform = transform

print("Loading target model...")
target = load_resnet(BASE / "model.pt", device)

print("Loading reference model...")
ref = load_resnet(BASE / "reference.pt", device)

print("Loading shadow models...")
shadows = []
for i in range(1, 17):  # adjust range to however many you have
    p = BASE / f"shadow_{i}.pt"
    if p.exists():
        shadows.append(load_resnet(p, device))
print(f"  Loaded {len(shadows)} shadow models")


# Extract features for pub.pt
print("\nExtracting features for pub.pt...")
pub_features, pub_labels, pub_ids = [], [], []

for i in range(len(pub_ds)):
    curr_id, img_tensor, label, membership = pub_ds[i]
    feats = extract_features(img_tensor, label, target, ref, shadows, device)
    pub_features.append(feats)
    pub_labels.append(int(membership))
    pub_ids.append(curr_id)
    if i % 200 == 0:
        print(f"  pub.pt {i}/{len(pub_ds)}")

X_pub = np.array(pub_features)
y_pub = np.array(pub_labels)
print(f"pub.pt: {X_pub.shape}, members={y_pub.sum()}, non-members={(1 - y_pub).sum()}")


scaler = StandardScaler()
X_pub_scaled = scaler.fit_transform(X_pub)

# 2. VALIDATION SPLIT (Crucial to catch overfitting)
X_train, X_val, y_train, y_val = train_test_split(
    X_pub_scaled, y_pub, test_size=0.2, random_state=42, stratify=y_pub
)

# 3. XGBoost - High Regularization
clf = XGBClassifier(
    n_estimators=100,
    max_depth=2,  # Shallow trees generalize better
    learning_rate=0.05,
    subsample=0.6,  # Use only 60% of data to prevent memorization
    reg_lambda=15,  # High L2 penalty
    eval_metric="logloss",
)

clf.fit(X_train, y_train)

# Check the REAL local score
val_scores = clf.predict_proba(X_val)[:, 1]
real_local_score = tpr_at_fpr(val_scores, y_val)
print(f"\nREALISTIC Local Score (Unseen Pub Data): {real_local_score:.4f}")

# 4. Final Fit on all data for submission
clf.fit(X_pub_scaled, y_pub)

# Local validation on pub.pt
pub_pred_scores = clf.predict_proba(X_pub_scaled)[:, 1]
local_score = tpr_at_fpr(pub_pred_scores, y_pub, target_fpr=0.05)
print(f"\nLocal TPR@5%FPR on pub.pt: {local_score:.4f}")
print("(Note: this is in-sample so it's optimistic — but should be >> 0.05)\n")


# Extract features for priv.pt
print("Extracting features for priv.pt...")
priv_features, priv_ids = [], []

for i in range(len(priv_ds)):
    curr_id, img_tensor, label, _ = priv_ds[i]
    feats = extract_features(img_tensor, label, target, ref, shadows, device)
    priv_features.append(feats)
    priv_ids.append(curr_id)
    if i % 200 == 0:
        print(f"  priv.pt {i}/{len(priv_ds)}")

X_priv = np.array(priv_features)
X_priv_scaled = scaler.transform(X_priv)


priv_scores = clf.predict_proba(X_priv_scaled)[:, 1]

print("Writing submission.csv...")
with open(BASE / "submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "score"])
    for cid, cs in zip(priv_ids, priv_scores):
        w.writerow([cid, float(cs)])

print(f"Done. {len(priv_ids)} rows written.")
