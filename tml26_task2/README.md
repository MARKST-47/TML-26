# Task 2: Stolen Model Detection

CISPA Helmholtz Center for Information Security — Trustworthy Machine Learning, SS2026

---

## Task Overview

Identify which models among 360 suspect ResNet-18 classifiers were stolen from a victim model, using white-box access to both the target and all suspect weights. Output is a continuous stealing confidence score per suspect (higher = more likely stolen), evaluated at **TPR @ 5% FPR**.

The evaluation covers three theft types:

1. **Direct copies and parameter mutations** — weight scrambling, scaling, layer shifting
2. **Fine-tuning** — weights drift but deep feature geometry is preserved
3. **Knowledge distillation / black-box extraction** — no shared weights, but decision boundary is inherited

---

## Performance Log

| Version | File                      | Strategy                                                                   |  TPR@5%FPR   |
| :------ | :------------------------ | :------------------------------------------------------------------------- | :----------: |
| v1      | `detect_stolen.py`        | Weight cosines, BN stats, FC similarity, top-1 agreement                   |   `0.5741`   |
| **v2**  | **`detect_stolen_v2.py`** | **+ CKA, ModelDiff noise agreement, train–test loss gap, top-3 agreement** | **`0.5926`** |

Best verified submission: `results/submission_5926.csv`

---

## Signal Overview

| Signal            | Weight v1 | Weight v2 | What it detects                                                                            |
| :---------------- | :-------: | :-------: | :----------------------------------------------------------------------------------------- |
| `weighted_cosine` |   0.15    |   0.12    | Layer-weighted cosine across all parameters (`layer4`=0.30, `fc`=0.40)                     |
| `fc_sim`          |   0.20    |   0.13    | Cosine similarity on FC weight + bias directly                                             |
| `bn_sim`          |   0.15    |   0.18    | Cosine over BatchNorm running mean/variance — strong fingerprint for direct copies         |
| `logit_corr`      |   0.15    |   0.10    | Pearson correlation of raw test-set logits                                                 |
| `top3_agreement`  |   0.05    |   0.08    | Set intersection of top-3 predicted classes — catches fine-tuned models where top-1 shifts |
| `noise_logit`     |   0.15    |   0.08    | Pearson correlation of logits on fixed Gaussian noise                                      |
| `noise_agreement` |     —     |   0.12    | Argmax agreement on noise (ModelDiff) — inherited decision boundary signal                 |
| `loss_gap`        |   0.15    |   0.10    | `train_spearman − test_spearman` loss correlation — membership inference signal            |
| `cka_layer4`      |     —     |   0.15    | Linear CKA on layer4 activations — invariant to weight rotation, catches fine-tuning       |

---

## Directory Structure

```
tml26_task2/
├── detect_stolen_v2.py       # Active production pipeline (best score: 0.5926)
├── detect_stolen.py          # Baseline v1 pipeline (score: 0.5741)
├── task_template.py          # Original course skeleton
├── run_pipeline.py           # Cluster bootstrap and execution wrapper
├── pipeline.sub              # HTCondor job configuration
├── submission.py             # Leaderboard submission script
├── target_model/
│   ├── weights.safetensors   # Target ResNet-18 weights
│   └── train_main_idx.json   # Training sample indices
├── suspect_models/
│   ├── suspect_000.safetensors
│   ├── suspect_001.safetensors
│   └── ... (360 total)
├── data/                     # CIFAR-100 auto-downloaded here
└── results/
    └── submission_5926.csv   # Verified best submission artifact
    └── submission_5740.csv
```

---

## Reproducing the Best Result

### 1. Prerequisites

- University VPN access: `vpn.uni-saarland.de`
- Cluster credentials: `atml_team034`
- All suspect model `.safetensors` files downloaded to `suspect_models/`
- Target model files in `target_model/`

### 2. SSH into the Cluster

```bash
ssh atml_team034@conduit.hpc.uni-saarland.de
cd ~/tml26_task2
```

### 3. Submit the Pipeline Job

This spins up a Docker container, resolves environment paths, and runs the full detection loop over all 360 suspects:

```bash
condor_submit pipeline.sub
```

**Resource allocation:** 2 CPUs, 16 GB RAM, 1 GPU — expected runtime ~5 hours.

### 4. Monitor Execution

```bash
condor_q
tail -f runlogs/pipeline.$(ClusterId).$(ProcId).out
```

### 5. Submit to Leaderboard

Submission must be run from the **login node** (compute nodes have no external internet access).

**Option A — Submit the freshly generated result:**

```bash
python3 submission.py
```

**Option B — Resubmit the verified best checkpoint directly:**

```bash
python3 -c "
import requests
with open('results/submission_5926.csv', 'rb') as f:
    resp = requests.post(
        'http://34.63.153.158/submit/19-stolen-model-detection',
        headers={'X-API-Key': '14cdd947fec2bbe735ed8001c9154ce6'},
        files={'file': ('submission.csv', f, 'csv')},
        timeout=(10, 120)
    )
print(resp.json())
"

```
