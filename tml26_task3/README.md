# Task 3: Adversarial ML & Robustness

CISPA Helmholtz Center for Information Security — Trustworthy Machine Learning, SS2026

---

## Task Overview

Train an adversarially robust ResNet classifier using a 9-class, $32 \times 32$ image dataset (train.npz) to withstand unseen adversarial perturbations. Output is a saved model state dictionary (model.pt), evaluated on a balanced performance metric: $\text{Score} = 0.5 \times \text{clean accuracy} + 0.5 \times \text{robust accuracy}$.

## Directory Structure

```
tml26_task3/
├── run_pipeline.py           # Cluster bootstrap and execution wrapper
├── pipeline.sub              # HTCondor job configuration
├── submission.py             # Leaderboard submission script
├── models/
```

## Reproducing the Best Result
