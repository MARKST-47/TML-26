# TML Task 3: Adversarial Robustness

**Team:** team_LXIV (atml_team034)  
**Best Leaderboard Score:** 0.6114  
**Architecture:** ResNet18 (pretrained, ImageNet)  
**Approach:** TRADES adversarial training

## Recreating the Best Result

### Prerequisites
- HTCondor cluster with GPU access
- Docker image: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`
- Dataset: `train.npz` (50,000 images, 9 classes): place in this directory

### Steps

1. **Submit the training job:**
   ```bash
   condor_submit pipeline_v5.sub
   ```

2. **Monitor training:**
   ```bash
   tail -f runlogs/pipeline_v5.*.out
   ```

3. **Submit the best model to the leaderboard:**
   
   Edit `submission.py`:
   - Set `MODEL_PATH` to the path of `model_best_v6.pt`
   - Set `MODEL_NAME = "resnet18"`
   - Set your API key
   
   ```bash
   python3 submission.py
   ```

### Training Configuration (run_pipeline_v5.py)

| Hyperparameter | Value |
|---|---|
| Architecture | ResNet18 (ImageNet pretrained) |
| Training method | TRADES |
| BETA (tradeoff) | 4.0 |
| PGD steps | 7 |
| Epsilon | 8/255 |
| Step size | 2/255 |
| Batch size | 256 |
| Epochs | 150 |
| Optimizer | SGD (Nesterov, momentum=0.9) |
| Weight decay | 5e-4 |
| LR schedule | Warmup (5ep) + Cosine annealing |
| Peak LR | 0.1 |
| Label smoothing | 0.1 |
| AMP | Enabled |
| Data augmentation | RandomCrop(32, pad=4, reflect) + HFlip |
| Checkpointing | Every 5 epochs, best model by PGD-20 score |

### File Structure

- `run_pipeline_v5.py` : Training script (best configuration)
- `pipeline_v5.sub` : HTCondor submit file
- `submission.py` : Leaderboard submission script
- `train.npz` : Training dataset
- `model_best_v6.pt` : Best model checkpoint (output)

### Expected Output

- Training time: ~2 hours on Tesla P100-16GB
- Best unified score (0.5 × clean + 0.5 × robust): ~0.61
