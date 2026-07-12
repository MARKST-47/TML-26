# TML-26: Trustworthy Machine Learning

This repository contains the projects and assignments for the **Trustworthy Machine Learning (Summer 2026)** course at CISPA Helmholtz Center for Information Security. The primary focus of these tasks is to analyze and improve the privacy, robustness, and fairness of modern machine learning systems.

## 📂 Repository Structure

| Task       | Topic                                      | Status       |
| :--------- | :----------------------------------------- | :----------- |
| **Task 1** | Privacy: Membership Inference Attack (MIA) | ✅ Completed |
| **Task 2** | Model Stealing: Stolen Model Detection     | ✅ Completed |
| **Task 3** | Adversarial ML & Robustness                | ✅ Completed |
| **Task 4** | Watermark Forgery Attack                   | ✅ Completed |

---

## 🛡️ Task 1: Membership Inference Attack (MIA)

### Overview

The objective of this task is to perform a Membership Inference Attack to determine whether specific data samples were part of a target model's training dataset. We are provided with a pretrained **ResNet-18** model and two datasets: `pub.pt` (labeled) and `priv.pt` (unlabeled). The challenge lies in distinguishing between members and non-members drawn from the same underlying distribution without explicit indicators.

### Key Methodology

- **Likelihood Ratio Attack (LiRA):** We implemented a LiRA approach using 16 shadow models to estimate the "OUT" distribution for each sample.
- **Meta-Classifier:** We trained an **XGBoost** model on the public dataset using features such as log-odds transformations, entropy, and prediction stability across Test-Time Augmentations (TTA).
- **Calibration:** Features were normalized on a per-class basis to account for varying model confidence levels across different image categories.

### 🚀 Reproducibility

For specific instructions on how to recreate our best leaderboard result, please refer to the detailed [Task 1 README](./tml26_task1/README.md).

## 🛡️ Task 2: Stolen Model Detection

### Overview

The goal of this task is to protect intellectual property by detecting stolen versions of a victim model among **360 suspect verification candidates**. The evaluation framework evaluates defenses against varied adversary actions, including direct checkpoint replication, heavy post-theft fine-tuning, and black-box dataset distillation/extraction. The final scoring metric evaluates the system's **True Positive Rate at a strict 5% False Positive Rate threshold (TPR@FPR=0.05)**.

### Key Methodology & Multi-Signal Architecture

Our production framework (`v2`) operates a calibrated ensemble across 9 distinct structural and behavioral signals to handle variations in model forgery:

- **Weight-Space Analysis (Anti-Mutation Layer):**
  - **Layer-Weighted Cosine Similarity:** Computes weight alignment across specific neural tiers, assigning higher structural importance to deep representations (`layer4` weight: 0.30, `fc` weight: 0.40).
  - **BatchNorm Statistics Fingerprinting:** Computes cosine similarities over frozen running mean and variance vectors to flag identical base model replicas.
- **Behavioral OOD Probing & Decision Boundary Geometry:**
  - **ModelDiff Boundary Matching (Shah et al., ICML 2023):** Feeds fixed Gaussian noise inputs to expose decision boundary profiles. Two independent models disagree sharply on Out-Of-Distribution (OOD) noise, whereas a distilled model inherits specific boundary layout bugs, driving up agreement.
  - **Top-3 Softmax Agreement Rate:** Replaces rigid argmax matching with a granular top-3 overlapping set intersection, catching fine-tuned variants where classification sequences remain stable despite soft logit shifting.
- **Representation & Membership Tracking:**
  - **Centered Kernel Alignment (CKA):** Tracks linear CKA over intermediate `layer4` activations. Because CKA is invariant to orthogonal rotation and scaling, it successfully exposes heavily fine-tuned candidates where raw weights have drifted but internal semantic representations remain identical.
  - **Dataset Inference Loss Gap:** Computes Spearman rank correlations over per-sample cross-entropy losses across the target's specific training index. It extracts the differential gap between training and testing data splits to catch memorization overlap.

### 🚀 Reproducibility

To replicate our top-performing pipeline, please refer to the detailed [Task 2 README](./tml26_task2/README.md).

## 🛡️ Task 3: Adversarial ML & Robustness

### Overview

The objective is to train an adversarially robust model capable of defending against unknown server-side adversarial attacks while preserving high generalization performance on clean data. We are provided with a compressed dataset (train.npz) containing $32 \times 32$ images spanning 9 distinct classes, to be trained using a customized ResNet architecture. The challenge is in navigating the fundamental trade-off between standard accuracy and adversarial robustness to maximize a balanced scoring metric ($0.5 \times \text{clean} + 0.5 \times \text{robust}$) without prior knowledge of the evaluation attack vector.

### Key Methodology

- **TRADES Optimization:** Utilized the TRADES loss function with a tuned regularization parameter (beta = 4.0) to analytically balance the fundamental trade-off between clean classification performance and robust accuracy.
- **Adversarial Training Inner Loop:** Employed a 7-step Projected Gradient Descent (PGD) attack loop with an operating radius of epsilon = 8/255 and a step size of 2/255, freezing model weights during the perturbation step to eliminate redundant backward passes.
- **Pretrained Initialization & Warmup:** Leveraged ImageNet pretrained weights to provide a strong structural feature foundation, combined with a 5-epoch linear learning rate warmup to protect the pretrained weight basins from early gradient disruption.
- **Regularization & Optimization Stack:** Combined Automatic Mixed Precision (AMP) and Nesterov SGD with label smoothing (0.1) and basic spatial augmentations (Random Crop and Horizontal Flip) to maximize clean generalization metrics.
- **Robust Overfitting Mitigation:** Implemented a high-frequency checkpointing engine evaluated against a local PGD-20 validation benchmark every 5 epochs, isolating the peak-performing model before the onset of robust validation decay.

### 🚀 Reproducibility

To replicate our top-performing pipeline, please refer to the detailed [Task 3 README](./tml26_task3/README.md).

## 🛡️ Task 4: Watermark Forgery Attack

### Overview

The objective is to forge invisible watermarks onto clean target images such that they deceive an unknown server side detector while maintaining high perceptual fidelity. We are provided with a dataset containing 200 clean target images and multiple sets of watermarked source images across 8 distinct watermark families (WM1​ to WM8​). The core challenge lies in navigating the competitive trade-off between detection alignment (Sdet​) and structural preservation (Sqlt​) to maximize a balanced final score (Sfinal ​= Sdet ​× Sqlt​) under strict, non-linear distortion constraints.

### Key Methodology

To break through the architectural limits of spatial-domain attacks, we transitioned from flat pixel-level averages to a **Multi-Domain Steganographic Alignment** framework. Our approach maps each watermark family directly to its specific embedding domain while enforcing strict amplitude calibration bounds to maximize the joint extraction-distortion objective:

- **Cryptographic Bit-Plane Reconstruction (WM_1, WM_2):** Extracts exact watermark bit messages through population-wide majority voting (`dwtDct` 16-bit for WM_1; `RivaGAN` 32-bit for WM_2). We apply native library encoders to re-embed the tokens into clean targets, then surgically blend them back toward the pristine frames (**alpha_WM1 = 0.85**, **alpha_WM2 = 0.50**) to minimize the LPIPS penalty without losing detection margin.
- **Joint Integer RGB Space Solver (WM_5):** Identifies a critical leak where floating-point chrominance adjustments (Cb and Cr) are entirely destroyed by float-to-uint8 rounding during PNG serialization. We resolve this by implementing a joint spatial integer solver that scales channel residuals alongside binary LSB bit-plane injections, pre-calculating integer coordinate adjustments directly in the target disk domain.
- **Frequency & Coherence Alignment (WM_3, WM_4, WM_6):** Bypasses spatial phase-cancellation on structural watermarks by operating directly inside transform sub-bands:
  - **WM_3:** Injects a calibrated multi-channel residual map concurrently across all three YCbCr space coordinates.
  - **WM_4:** Synthesizes a localized geometric template isolated from cross-image Fourier Phase Coherence grids.
  - **WM_6:** Matches mid-frequency coefficient distributions across tiled 8x8 block Discrete Cosine Transform (DCT) profiles.
- **Amplitude-Calibrated Noise Windows (WM_7, WM_8):** Models the remaining unmapped black-box categories using localized high-pass variance residuals, damping the noise footprint aggressively (**alpha = 0.015**) to satisfy server distribution gates while preserving structural quality.

### 🚀 Reproducibility

To replicate our top-performing pipeline, please refer to the detailed [Task 4 README](./tml26_task4/README.md).

## ✍️ Team Information

- **Team ID:** team034
- **Course:** Trustworthy Machine Learning, 2026
- **Supervisors:** Adam Dziedzic and Franziska Boenisch
