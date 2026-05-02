import os
import sys
import torch
import requests
import csv
import random
import argparse
from scipy.stats import norm
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
REF_PATH = BASE / "reference.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"  # DONOT CHANGE
API_KEY = "14cdd947fec2bbe735ed8001c9154ce6"
TASK_ID = "01-mia"  # DONOT CHANGE


# # dataset classes
# class TaskDataset(Dataset):
#     def __init__(self, transform=None):
#         self.ids = []
#         self.imgs = []
#         self.labels = []
#         self.transform = transform

#     def __getitem__(self, index):
#         id_ = self.ids[index]
#         img = self.imgs[index]
#         if self.transform is not None:
#             img = self.transform(img)
#         label = self.labels[index]
#         return id_, img, label

#     def __len__(self):
#         return len(self.ids)


# class MembershipDataset(TaskDataset):
#     def __init__(self, transform=None):
#         super().__init__(transform)
#         self.membership = []

#     def __getitem__(self, index):
#         id_, img, label = super().__getitem__(index)
#         return id_, img, label, self.membership[index]


# # load datasets
# print("Loading datasets...")
# pub_ds = torch.load(PUB_PATH, weights_only=False)
# priv_ds = torch.load(PRIV_PATH, weights_only=False)


# # normalization (same as training)
# MEAN = [0.7406, 0.5331, 0.7059]
# STD = [0.1491, 0.1864, 0.1301]

# transform = transforms.Compose(
#     [
#         transforms.Resize(32),
#         transforms.Normalize(mean=MEAN, std=STD),
#     ]
# )

# pub_ds.transform = transform
# priv_ds.transform = transform


# # load model
# print("Loading model.")
# model = resnet18(weights=None)
# model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
# model.maxpool = torch.nn.Identity()
# model.fc = torch.nn.Linear(512, 9)

# model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# print("Loading reference model...")
# ref_model = resnet18(weights=None)
# ref_model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
# ref_model.maxpool = torch.nn.Identity()
# ref_model.fc = torch.nn.Linear(512, 9)
# ref_model.load_state_dict(torch.load(BASE / "reference.pt", map_location="cpu"))
# ref_model.to(device)
# ref_model.eval()


# def log_odds(logits, label):
#     """phi(x) = log(p_y / (1 - p_y)) — correct statistic for reference attack."""
#     p = F.softmax(logits, dim=-1)[label].clamp(1e-7, 1 - 1e-7)
#     return (p / (1 - p)).log().item()


# tta_tf = transforms.Compose(
#     [
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomCrop(32, padding=4),
#     ]
# )


# def score_sample(img_tensor, label, n_tta=8):
#     """Returns reference-model attack score for one sample."""
#     imgs = torch.stack([tta_tf(img_tensor) for _ in range(n_tta)]).to(device)
#     with torch.no_grad():
#         phi_t = log_odds(model(imgs).mean(0), label)
#         phi_r = log_odds(ref_model(imgs).mean(0), label)
#     return phi_t - phi_r


# print("Scoring priv.pt with reference model attack...")
# all_ids, all_scores = [], []

# for i in range(len(priv_ds)):
#     curr_id, img_tensor, label, _ = priv_ds[i]
#     s = score_sample(img_tensor, label, n_tta=8)
#     all_ids.append(curr_id)
#     all_scores.append(s)
#     if i % 200 == 0:
#         print(f"  {i}/{len(priv_ds)}")

# all_scores = np.array(all_scores)
# lo, hi = all_scores.min(), all_scores.max()
# normalized = (all_scores - lo) / (hi - lo + 1e-8)


# with open(OUTPUT_CSV, "w", newline="") as f:
#     w = csv.writer(f)
#     w.writerow(["id", "score"])
#     for cid, cs in zip(all_ids, normalized):
#         w.writerow([cid, float(cs)])

# print("Saved:", OUTPUT_CSV)

if not OUTPUT_CSV.exists():
    raise FileNotFoundError("Run train_meta.py first to generate submission.csv")

print("Loaded pre-computed scores from train_meta.py")


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
