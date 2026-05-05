# TML 2026 - Assignment 1: Membership Inference Attack

## How to Reproduce Our Best Result

Our best leaderboard score (0.0580 TPR@5%FPR) uses an XGBoost meta-classifier trained on features extracted from a target model, a reference model, and 16 shadow models.

### Prerequisites

- Access to the UdS HPC cluster with HTCondor
- Team credentials (atml_team034)
- VPN connection to the university network

### Step 1: Connect to the cluster

```bash
ssh atml_team034@conduit2.hpc.uni-saarland.de
cd ~/tml26_task1
```

### Step 2: Download the dataset (if not already present)

```bash
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/pub.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/priv.pt"
wget "https://huggingface.co/datasets/SprintML/tml26_task1/resolve/main/model.pt"
```

### Step 3: Train shadow models

```bash
condor_submit -i mia.sub
# Inside the container:
pip install scikit-learn
python train_shadow.py
```

This trains 16 shadow models on random 50% splits of pub.pt (30 epochs each). Output: `shadow_1.pt` through `shadow_16.pt`.
This trains 16 shadow models on random 50% splits of pub.pt (30 epochs each) and saves their training indices. Output: `shadow_1.pt` through `shadow_16.pt` and `shadow_1_indices.pt` through `shadow_16_indices.pt`

### Step 4: Train the reference model

```bash
python train_reference.py
```

This trains a reference model exclusively on non-members from pub.pt (50 epochs). Output: `reference.pt`.

### Step 5: Run the meta-classifier attack

```bash
pip install xgboost optuna
python train_meta.py
```

This script:
1. Extracts features from pub.pt and priv.pt using 32 test-time augmentations
2. Runs Optuna hyperparameter search (200 trials) for XGBoost
3. Trains the final XGBoost classifier on all pub.pt data
4. Predicts membership scores for priv.pt
5. Saves `submission.csv`

### Step 6: Submit to the leaderboard

```bash
python task_template.py
```

### File Overview

| File | Description |
|------|-------------|
| `train_shadow.py` | Trains 16 shadow models on random splits of pub.pt |
| `train_reference.py` | Trains a reference model on non-members only |
| `train_meta.py` | Extracts features, trains XGBoost with Optuna, generates submission.csv |
| `task_template.py` | Submits submission.csv to the leaderboard server |
| `mia.sub` | HTCondor submit file for GPU jobs |
