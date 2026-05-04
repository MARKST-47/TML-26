import subprocess
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import csv
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve


# DYNAMIC INSTALLATION
def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        return __import__(import_name)
    except ImportError:
        print(f"{package} not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return __import__(import_name)


xgb_mod = install_and_import("xgboost", "xgboost")
XGBClassifier = xgb_mod.XGBClassifier
optuna = install_and_import("optuna")

# CONFIG & DATASET
BASE = Path(__file__).parent
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]


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

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# FEATURE EXTRACTION
def log_odds(logits, label):
    p = F.softmax(logits, dim=-1)[label].clamp(1e-7, 1 - 1e-7)
    return (p / (1 - p)).log().item()


def get_entropy(logits):
    p = F.softmax(logits, dim=-1)
    return (-p * p.log()).sum().item()


tta_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ]
)


@torch.no_grad()
def extract_features(
    img_tensor,
    label,
    target,
    ref,
    shadows,
    shadow_indices,
    device,
    sample_idx=None,
    n_tta=32,
):
    imgs = torch.stack([tta_tf(img_tensor) for _ in range(n_tta)]).to(device)

    # Target & Reference log-odds
    t_logits = target(imgs).mean(0)
    phi_t = log_odds(t_logits, label)
    r_logits = ref(imgs).mean(0)
    phi_r = log_odds(r_logits, label)

    # Collect shadow phis
    all_shadow_phis = []
    out_phis = []
    for s, s_idx in zip(shadows, shadow_indices):
        phi_s = log_odds(s(imgs).mean(0), label)
        all_shadow_phis.append(phi_s)
        if sample_idx is not None and s_idx is not None:
            if sample_idx not in s_idx:
                out_phis.append(phi_s)
        else:
            out_phis.append(phi_s)

    # Use "OUT" models to calculate the null distribution (consistent for pub and priv)
    mu_out = np.mean(out_phis)
    sigma_out = np.std(out_phis) + 1e-8

    # This is the "honest" LiRA Z-score
    lira_z = (phi_t - mu_out) / sigma_out

    return [
        phi_t,
        phi_t - phi_r,
        lira_z,  # Core LiRA signal
        get_entropy(t_logits),
        phi_t - mu_out,  # Raw difference
        label,
    ]


# PER-CLASS NORMALIZATION
def normalize_per_class(X, labels_col, class_stats):
    X_norm = X.copy()
    for c, (mu, sigma) in class_stats.items():
        mask = labels_col == c
        X_norm[mask, :-1] = (X[mask, :-1] - mu) / sigma
    return X_norm


def tpr_at_fpr(scores, labels, target_fpr=0.05):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[max(idx, 0)])


# SETUP
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


def load_resnet(path, device):
    m = resnet18(weights=None)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    m.fc = torch.nn.Linear(512, 9)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()


print("Loading target model...")
target = load_resnet(BASE / "model.pt", device)

print("Loading reference model...")
ref = load_resnet(BASE / "reference.pt", device)

print("Loading shadow models and indices...")
shadows = []
shadow_indices = []
for i in range(1, 17):
    p = BASE / f"shadow_{i}.pt"
    if p.exists():
        shadows.append(load_resnet(p, device))
        idx_path = BASE / f"shadow_{i}_indices.pt"
        if idx_path.exists():
            shadow_indices.append(set(torch.load(idx_path)))
        else:
            shadow_indices.append(None)
print(f"  Loaded {len(shadows)} shadow models")
print(
    f"  Indices available for {sum(x is not None for x in shadow_indices)}/{len(shadows)} shadows"
)


# EXTRACT FEATURES FOR PUB.PT
print("\nExtracting features for pub.pt...")
pub_features = []
for i in range(len(pub_ds)):
    pub_features.append(
        extract_features(
            pub_ds[i][1],
            pub_ds[i][2],
            target,
            ref,
            shadows,
            shadow_indices,
            device,
            sample_idx=i,
        )
    )
    if i % 200 == 0:
        print(f"  pub.pt {i}/{len(pub_ds)}")

X_pub = np.array(pub_features)
y_pub = np.array([int(pub_ds[i][3]) for i in range(len(pub_ds))])
print(f"pub.pt: {X_pub.shape}, members={y_pub.sum()}, non-members={(1 - y_pub).sum()}")


# PER-CLASS NORMALIZATION USING PUB NON-MEMBERS
labels_pub = X_pub[:, -1].astype(int)
class_stats = {}
for c in np.unique(labels_pub):
    nonmem_mask = (labels_pub == c) & (y_pub == 0)
    if nonmem_mask.sum() > 5:
        class_stats[c] = (
            X_pub[nonmem_mask, :-1].mean(axis=0),
            X_pub[nonmem_mask, :-1].std(axis=0) + 1e-8,
        )

X_pub_cn = normalize_per_class(X_pub, labels_pub, class_stats)
scaler = StandardScaler()
X_pub_scaled = scaler.fit_transform(X_pub_cn)


# OPTUNA HYPERPARAMETER SEARCH
X_train, X_val, y_train, y_val = train_test_split(
    X_pub_scaled, y_pub, test_size=0.2, random_state=42, stratify=y_pub
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    params = {
        "n_estimators": 100,
        "max_depth": trial.suggest_int("max_depth", 2, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.03, log=True),
        "subsample": trial.suggest_float("subsample", 0.55, 0.70),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 15.0, 30.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        "eval_metric": "logloss",
        "random_state": 42,
    }
    clf_trial = XGBClassifier(**params)
    clf_trial.fit(X_train, y_train)
    val_scores = clf_trial.predict_proba(X_val)[:, 1]
    return tpr_at_fpr(val_scores, y_val, target_fpr=0.05)


print("Starting Optuna optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200, show_progress_bar=True)
print(f"Best val TPR@5%FPR: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")


# FINAL TRAIN ON ALL PUB DATA
clf = XGBClassifier(
    **study.best_params, n_estimators=100, eval_metric="logloss", random_state=42
)
clf.fit(X_pub_scaled, y_pub)

pub_pred_scores = clf.predict_proba(X_pub_scaled)[:, 1]
print(
    f"In-sample TPR@5%FPR on pub.pt: {tpr_at_fpr(pub_pred_scores, y_pub):.4f} (optimistic)"
)


# EXTRACT FEATURES FOR PRIV.PT
print("\nExtracting features for priv.pt...")
priv_features = []
priv_ids = []
for i in range(len(priv_ds)):
    curr_id, img, label, _ = priv_ds[i]
    priv_features.append(
        extract_features(
            img, label, target, ref, shadows, shadow_indices, device, sample_idx=None
        )
    )
    priv_ids.append(curr_id)
    if i % 200 == 0:
        print(f"  priv.pt {i}/{len(priv_ds)}")

X_priv = np.array(priv_features)
labels_priv = X_priv[:, -1].astype(int)
X_priv_cn = normalize_per_class(X_priv, labels_priv, class_stats)
X_priv_scaled = scaler.transform(X_priv_cn)

priv_scores = clf.predict_proba(X_priv_scaled)[:, 1]


# SUBMISSION
with open(BASE / "submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "score"])
    for cid, cs in zip(priv_ids, priv_scores):
        w.writerow([cid, float(cs)])
print(f"Done. {len(priv_ids)} rows written to submission.csv")
