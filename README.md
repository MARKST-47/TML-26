# TML-26: Trustworthy Machine Learning

This repository contains the projects and assignments for the **Trustworthy Machine Learning (Summer 2026)** course at CISPA Helmholtz Center for Information Security. The primary focus of these tasks is to analyze and improve the privacy, robustness, and fairness of modern machine learning systems.

## 📂 Repository Structure

| Task | Topic | Status |
| :--- | :--- | :--- |
| **Task 1** | Privacy: Membership Inference Attack (MIA) | ✅ Completed | 
---

## 🛡️ Task 1: Membership Inference Attack (MIA)

### Overview
The objective of this task is to perform a Membership Inference Attack to determine whether specific data samples were part of a target model's training dataset. We are provided with a pretrained **ResNet-18** model and two datasets: `pub.pt` (labeled) and `priv.pt` (unlabeled). The challenge lies in distinguishing between members and non-members drawn from the same underlying distribution without explicit indicators.

### Key Methodology
*   **Likelihood Ratio Attack (LiRA):** We implemented a LiRA approach using 16 shadow models to estimate the "OUT" distribution for each sample.
*   **Meta-Classifier:** We trained an **XGBoost** model on the public dataset using features such as log-odds transformations, entropy, and prediction stability across Test-Time Augmentations (TTA).
*   **Calibration:** Features were normalized on a per-class basis to account for varying model confidence levels across different image categories.

### 🚀 Reproducibility
For specific instructions on how to recreate our best leaderboard result, please refer to the detailed [Task 1 README](./task1/README.md).

## ✍️ Team Information
*   **Team ID:** team034
*   **Course:** Trustworthy Machine Learning, 2026
*   **Supervisors:** Adam Dziedzic and Franziska Boenisch