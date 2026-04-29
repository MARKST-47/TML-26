import os
import sys
import torch
import requests
import csv
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"  # DONOT CHANGE
API_KEY = "14cdd947fec2bbe735ed8001c9154ce6"
TASK_ID = "01-mia"  # DONOT CHANGE


# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose(
    [
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model.")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Loading all shadow models.")
shadows = []
for i in range(1, 6):
    s = resnet18(weights=None)
    s.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    s.maxpool = torch.nn.Identity()
    s.fc = torch.nn.Linear(512, 9)
    s.load_state_dict(torch.load(BASE / f"shadow_{i}.pt", map_location=device))
    s.to(device)
    s.eval()
    shadows.append(s)

print("Calculating RMIA scores...")
priv_ds.membership = [-1] * len(priv_ds)  # Replace None with -1
loader = torch.utils.data.DataLoader(priv_ds, batch_size=256, shuffle=False)

all_ids = []
all_scores = []

with torch.no_grad():
    for ids, imgs, labels, _ in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Horizontal flipped image for Test time augmentation
        imgs_flipped = torch.flip(imgs, dims=[3])

        # Target Model Probabilities (Average of original and flipped)
        p_t_orig = F.softmax(model(imgs), dim=1)[torch.arange(len(labels)), labels]
        p_t_flip = F.softmax(model(imgs_flipped), dim=1)[
            torch.arange(len(labels)), labels
        ]
        p_target = (p_t_orig + p_t_flip) / 2.0

        # Shadow Model Probabilities (Average across all models and flips)
        p_shadow_sum = 0
        for s in shadows:
            p_s_orig = F.softmax(s(imgs), dim=1)[torch.arange(len(labels)), labels]
            p_s_flip = F.softmax(s(imgs_flipped), dim=1)[
                torch.arange(len(labels)), labels
            ]
            p_shadow_sum += (p_s_orig + p_s_flip) / 2.0

        p_shadow_avg = p_shadow_sum / len(shadows)

        # Compute Log Ratio (RMIA Score) with larger epsilon to handle zeros
        eps = 1e-7
        score = torch.log(p_target + eps) - torch.log(p_shadow_avg + eps)

        all_ids.extend(ids.tolist())
        all_scores.extend(score.cpu().numpy().tolist())

# Normalize scores to [0, 1]
all_scores = np.array(all_scores)
min_s, max_s = np.min(all_scores), np.max(all_scores)
normalized_scores = (all_scores - min_s) / (max_s - min_s + 1e-8)

print("Creating submission...")
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "score"])
    for curr_id, curr_score in zip(all_ids, normalized_scores):
        writer.writerow([curr_id, curr_score])

print("Saved:", OUTPUT_CSV)


# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
