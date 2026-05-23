# TML 2026 - Assignment 2: Stolen Model Detection

## How to Reproduce Our Best Result

Our best leaderboard score (0.5741 TPR@5%FPR) uses a multi-signal detection pipeline combining 7 weight-space and behavioral signals with z-score normalization and sigmoid calibration.

### Prerequisites

- Access to the UdS HPC cluster with HTCondor
- Team credentials (atml_team034)
- VPN connection to the university network

### Step 1: Connect to the cluster

```bash
ssh atml_team034@conduit.hpc.uni-saarland.de
cd ~/tml26_task2
```

### Step 2: Ensure data is downloaded

The suspect models and target model should already be present:

```bash
ls target_model/weights.safetensors
ls suspect_models/suspect_000.safetensors
```

If not, run the download script:

```bash
condor_submit download.sub
```

### Step 3: Run the detection pipeline

```bash
condor_submit pipeline.sub
```

This submits a GPU job that:
1. Installs dependencies inside the Docker container
2. Loads the target model and computes reference signals (logits, losses, BN stats)
3. Iterates over all 360 suspect models computing 7 similarity signals per model
4. Calibrates scores using z-normalization and sigmoid scaling
5. Saves `raw_signals.csv` and `submission.csv`

Runtime: ~5 hours on a Tesla P100-16GB.

### Step 4: Monitor the job

```bash
condor_q
tail -f runlogs/pipeline.*.out
```

### Step 5: Submit to the leaderboard

```bash
python3 -c "
import requests
resp = requests.post(
    'http://34.63.153.158/submit/19-stolen-model-detection',
    headers={'X-API-Key': 'YOUR_API_KEY_HERE'},
    files={'file': ('submission.csv', open('submission.csv','rb'), 'csv')},
    timeout=(10,120)
)
print(resp.json())
"
```

### File Overview

| File | Description |
|------|-------------|
| `detect_stolen.py` | Main detection pipeline: computes 7 signals and produces submission.csv |
| `task_template.py` | Original baseline pipeline |
| `run_pipeline.py` | Entry point for HTCondor, installs deps and calls detect_stolen.main() |
| `pipeline.sub` | HTCondor submit file for GPU jobs |
| `submission.py` | Submission script with API key and server config |
| `raw_signals.csv` | Per-model signal values (7 columns × 360 rows) for analysis |
| `submission.csv` | Final submission file (id, score) |

### Signal Descriptions

| Signal | Weight | Description |
|--------|--------|-------------|
| weighted_cosine | 0.15 | Layer-weighted cosine similarity (FC=0.40, layer4=0.30, early layers~0) |
| fc_sim | 0.20 | Cosine similarity on fc.weight and fc.bias only |
| bn_sim | 0.15 | Cosine similarity on concatenated BatchNorm running_mean/var |
| logit_corr | 0.15 | Pearson correlation of flattened logits on 2048 test samples |
| agreement | 0.05 | Fraction of matching argmax predictions on test data |
| noise_logit | 0.15 | Pearson correlation of logits on 1024 fixed random noise inputs |
| loss_corr | 0.15 | Spearman correlation of per-sample training losses |
