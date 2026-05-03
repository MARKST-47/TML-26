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


# FEATURE EXTRACTION WITH TTA VARIANCE
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
def extract_features(img_tensor, label, target, ref, shadows, device, n_tta=32):
    imgs = torch.stack([tta_tf(img_tensor) for _ in range(n_tta)]).to(device)

    # Target — Per-augmentation stability
    t_logits_batch = target(imgs)
    t_logits_mean = t_logits_batch.mean(0)
    phi_t = log_odds(t_logits_mean, label)
    phi_t_per_aug = [log_odds(t_logits_batch[k], label) for k in range(n_tta)]
    phi_t_std = np.std(phi_t_per_aug)

    # Reference — Per-augmentation stability
    r_logits_batch = ref(imgs)
    r_logits_mean = r_logits_batch.mean(0)
    phi_r = log_odds(r_logits_mean, label)
    phi_r_per_aug = [log_odds(r_logits_batch[k], label) for k in range(n_tta)]
    phi_r_std = np.std(phi_r_per_aug)

    # Shadows for LiRA — now with per-shadow TTA variance
    shadow_phis = []
    shadow_stds = []
    for s in shadows:
        s_logits_batch = s(imgs)  # [n_tta, 9] — reuse full batch
        s_phis = [log_odds(s_logits_batch[k], label) for k in range(n_tta)]
        shadow_phis.append(np.mean(s_phis))
        shadow_stds.append(np.std(s_phis))

    shadow_mean = np.mean(shadow_phis)
    shadow_std = np.std(shadow_phis) + 1e-8
    lira_z = (phi_t - shadow_mean) / shadow_std
    shadow_std_mean = np.mean(shadow_stds)  # avg shadow instability

    return [
        phi_t,
        phi_t - phi_r,
        lira_z,
        get_entropy(t_logits_mean),
        phi_t_std,
        phi_r_std,
        phi_t_std - phi_r_std,
        shadow_std_mean,
        phi_t_std - shadow_std_mean,
        label,
    ]


# PER-CLASS NORMALIZATION LOGIC
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


# MAIN EXECUTION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [transforms.Resize(32), transforms.Normalize(mean=MEAN, std=STD)]
)

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


target = load_resnet(BASE / "model.pt", device)
ref = load_resnet(BASE / "reference.pt", device)
shadows = [
    load_resnet(BASE / f"shadow_{i}.pt", device)
    for i in range(1, 17)
    if (BASE / f"shadow_{i}.pt").exists()
]

# Feature Extraction
print("Extracting features (this will take time with N_TTA=32)...")
pub_features = []
for i in range(len(pub_ds)):
    pub_features.append(
        extract_features(pub_ds[i][1], pub_ds[i][2], target, ref, shadows, device)
    )
    if i % 200 == 0:
        print(f"  pub.pt {i}/{len(pub_ds)}")
X_pub = np.array(pub_features)
y_pub = np.array([int(pub_ds[i][3]) for i in range(len(pub_ds))])

# Compute Class Stats using Pub Non-Members
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


def objective(trial):
    params = {
        "n_estimators": 100,
        "max_depth": trial.suggest_int("max_depth", 2, 3),  # was 2-5
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.008, 0.03, log=True
        ),  # was 0.01-0.15
        "subsample": trial.suggest_float("subsample", 0.55, 0.70),  # was 0.5-0.9
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 15.0, 30.0),  # was 1-30
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),  # was 0-5
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),  # was 1-10
        "eval_metric": "logloss",
        "random_state": 42,
    }
    clf_trial = XGBClassifier(**params)
    clf_trial.fit(X_train, y_train)
    val_scores = clf_trial.predict_proba(X_val)[:, 1]
    return tpr_at_fpr(val_scores, y_val, target_fpr=0.05)


print("Starting Optuna optimization...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)  # was 100
print(f"Best params: {study.best_params}")

# Final Train
clf = XGBClassifier(
    **study.best_params, n_estimators=100, eval_metric="logloss", random_state=42
)
clf.fit(X_pub_scaled, y_pub)

# PRIVATE PREDICTION
print("Predicting for priv.pt...")
priv_features = []
priv_ids = []
for i in range(len(priv_ds)):
    curr_id, img, label, _ = priv_ds[i]
    priv_features.append(extract_features(img, label, target, ref, shadows, device))
    priv_ids.append(curr_id)

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
print("Done. submission.csv is ready.")
