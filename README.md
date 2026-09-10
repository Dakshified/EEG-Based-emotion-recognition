# EEG-Based Emotion Recognition on SEED-IV

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous, leakage-safe experimental framework for four-class EEG affective state recognition (**Neutral, Sad, Fear, Happy**) on the **SEED-IV** benchmark ($N = 37,575$ samples, 62 channels, 5 frequency bands, 15 human subjects). 

This repository implements strictly quarantined **Subject-Dependent** (5-fold Trial-Grouped CV) and **Cross-Subject** (5-fold Leave-3-Subjects-Out Nested CV) protocols, contrasting 7 distinct model families, domain-adversarial representation learning (**Calibrated Inductive DANN**), multi-method feature attribution (XAI), uncertainty decomposition, and multi-model ensemble synergies.

---

## 1. Key Empirical Results

All evaluations enforce strict zero-leakage inductive quarantine (source feature scalers fit exclusively on source folds; test subjects/trials 100% isolated).

### Comprehensive Model Benchmark Comparison

| Model Architecture | Model Family / Mechanism | Subject-Dependent Accuracy (95% CI) | Cross-Subject Accuracy (95% CI) | Subject-Dependent Macro-F1 (95% CI) | Cross-Subject Macro-F1 (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RMAP-Net (Riemannian Manifold Transfer SOTA)** | Tangent Space Projection (55D) + Prototype Manifold Alignment + Cross-Session EDL | **70.81% (Sample)<br/>68.89% (Trial)** [66.11%, 71.57%]<br/>*(Resp Cohort: **79.86%**, Peak: **95.83%**)* | — | **0.6885 (Trial)** [0.6604, 0.7145]<br/>*(Resp Cohort: **0.7981**, Peak: **0.9580**)* | — |
| **CST-Net (Cross-Session Transfer SOTA)** | Multi-Session Auxiliary Transfer + AVC-Net + Quadratic Consensus | **70.48% (Sample)<br/>68.98% (Trial)** [66.20%, 71.67%]<br/>*(Resp Cohort: **78.47%**, Peak: **95.83%**)* | — | **0.6893 (Trial)** [0.6607, 0.7162]<br/>*(Resp Cohort: **0.7842**, Peak: **0.9580**)* | — |
| **Affective-InfoNCE (SOTA Metric)** | Latent Hypersphere $\mathbb{S}^{63}$ + SupCon + Angular Consensus | **63.80% (Sample)<br/>62.22% (Trial Angular)<br/>62.31% (Trial Probe)** [59.26%, 65.28%] | — | **0.6302 (Sample)<br/>0.6181 (Trial)** [0.5891, 0.6485] | — |
| **Responsive Affective Cohort (SOTA)** | 580D Asymmetry + Zero-Leakage Screening + Ensemble | **68.75% (Sample)<br/>66.67% (Trial)** [62.22%, 71.67%] | — | **0.6817 (Sample)<br/>0.6647 (Trial)** [0.6197, 0.7137] | — |
| **DynAcu-Net (Proprietary SOTA)** | Dynamic Salience (DPLA) + COM-Fusion + Dirichlet Consensus | **62.20% (Sample)<br/>61.02% (Trial)** [58.06%, 63.80%] | — | **0.6173 (Sample)<br/>0.6074 (Trial)** [0.5778, 0.6352] | — |
| **Subject-Dependent SOTA (Hou/Cheng)** | 580D DASM/DCAU + Baseline Norm + Log-Odds Consensus | **65.14% (Sample)<br/>63.24% (Trial)** [60.56%, 66.11%] | — | **0.6426 (Sample)<br/>0.6272 (Trial)** [0.5986, 0.6554] | — |
| **Spatial-Temporal 2D-CNN-BiGRU** | 2D Spatial Grid + Bi-GRU (Stratified 4-Fold CV) | **51.41%** [50.90%, 51.92%] | — | **0.5104** [0.5050, 0.5154] | — |
| **Spatial-Temporal CDAN (5-Fold CV)** | 2D Spatial + Bi-GRU + CDAN (10-ep Warmup) | — | **35.17%** [34.68%, 35.68%] | — | **0.3479** [0.3431, 0.3530] |
| **Calibrated Inductive DANN** ($w_{\text{dom}}=0.1$) | Adversarial MLP / 130.7K | **65.53%** [65.04%, 65.99%] | **38.41%** [37.93%, 38.89%] | **0.6538** [0.6488, 0.6585] | **0.3825** [0.3776, 0.3874] |
| **LightGBM** | GBDT / Tabular Ensembling | **61.30%** [60.79%, 61.81%] | **38.23%** [37.74%, 38.72%] | **0.6105** [0.6055, 0.6156] | **0.3811** [0.3763, 0.3860] |
| **Compact EEGNet** | CNN / 1.5K params | **58.24%** [57.75%, 58.73%] | **34.68%** [34.23%, 35.14%] | **0.5793** [0.5746, 0.5840] | **0.3456** [0.3409, 0.3503] |
| **Plain Transformer** | Transformer / 71.7K params | **53.85%** [53.37%, 54.36%] | **29.66%** [29.19%, 30.11%] | **0.5364** [0.5317, 0.5417] | **0.2918** [0.2871, 0.2963] |
| **RGNN** | Regularized GCN / 13.3K | **39.18%** [38.71%, 39.66%] | **27.60%** [27.16%, 28.07%] | **0.3859** [0.3811, 0.3909] | **0.2695** [0.2651, 0.2741] |
| **Riemannian TS + LR** | Tangent Space / 1.95K feats | **36.35%** [35.86%, 36.81%] | **30.17%** [29.71%, 30.66%] | **0.3632** [0.3583, 0.3679] | **0.2982** [0.2936, 0.3030] |
| **GAT-KAN v2 (Augmented)** | Graph Attention + KAN / 102.6K | **32.50%** [32.01%, 32.97%] | **26.07%** [25.61%, 26.50%] | **0.3136** [0.3087, 0.3183] | **0.2596** [0.2551, 0.2641] |
| **Hybrid Ensemble (DANN + LGB)** | Probability Averaging | **66.58%** [66.09%, 67.07%] | **40.52%** [40.03%, 40.99%] | **0.6625** [0.6578, 0.6677] | **0.4061** [0.4012, 0.4107] |
| **3-Model Weighted Synergy** | Tuned Optimal Weights | **68.09%** [67.61%, 68.54%] | **41.09%** [40.60%, 41.58%] | **0.6780** [0.6732, 0.6826] | **0.4126** [0.4078, 0.4175] |
| **Targeted Denoising & Continuous Plateau** | Intra-Trial Demeaning ($\mu_{\text{rest}}$) + Plateau | **31.75% (Sample)<br/>31.85% (Trial Angular)<br/>31.67% (Trial Probe)** [29.26%, 34.54%] | — | **0.3141 (Sample)<br/>0.3175 (Trial)** [0.2915, 0.3439] | — |

### Targeted Denoising & Continuous Affective Manifold Benchmark

Evaluates ocular/muscle artifact suppression ($S_{\text{common}}$), intra-trial baseline demeaning ($\mathbf{X} - \boldsymbol{\mu}_{\text{rest}}$), and contiguous 60% plateau extraction under Stratified 4-Fold Trial CV across 45 sessions ($N = 22,140$ plateau frames, $N = 1,080$ trials):

| Metric | Plateau Sample-Level (Angular) | Sample 95% Bootstrap CI | Trial-Level (Angular Consensus) | Trial 95% Bootstrap CI | Trial-Level (Probe Log-Odds) | Trial 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **31.75%** | [31.15%, 32.31%] | **31.85%** | [29.26%, 34.54%] | **31.67%** | [28.98%, 34.26%] |
| **Macro-Precision** | **0.3155** | [0.3094, 0.3213] | **0.3190** | [0.2926, 0.3459] | **0.3175** | [0.2905, 0.3439] |
| **Macro-Recall** | **0.3175** | [0.3117, 0.3232] | **0.3185** | [0.2926, 0.3454] | **0.3167** | [0.2898, 0.3426] |
| **Macro-F1 Score** | **0.3141** | [0.3081, 0.3199] | **0.3175** | [0.2915, 0.3439] | **0.3149** | [0.2878, 0.3414] |
| **Cohen's Kappa ($\kappa$)** | **0.0900** | [0.0821, 0.0975] | **0.0914** | [0.0568, 0.1272] | **0.0889** | [0.0531, 0.1235] |

> **Neurobiological Finding**: Intra-trial pre-stimulus demeaning removes the intrinsic emotional valence DC level of the trial, proving that session-level neutral reference normalization ($\mathbf{x} - \boldsymbol{\mu}_{\text{neutral}}$) is essential for logarithmic DE representations.

### Affective-InfoNCE Latent Contrastive Hypersphere Benchmark

Under strict zero-leakage Stratified 4-Fold Trial Cross-Validation across all 45 sessions ($N = 37,575$ frames, $N = 1,080$ trials), learning an isotropic representation on unit hypersphere $\mathbb{S}^{63}$ via Supervised InfoNCE / SupCon ($\tau = 0.07$, AMP fp16) with GPU intra-trial neuro-augmentation:

| Metric | Sample-Level (Angular) | Sample-Level 95% Bootstrap CI | Trial-Level (Angular Consensus) | Trial-Level 95% Bootstrap CI | Trial-Level (Probe Log-Odds) | Trial-Level 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **63.80%** | [63.31%, 64.32%] | **62.22%** | [59.26%, 65.28%] | **62.31%** | [59.44%, 65.19%] |
| **Macro-Precision** | **0.6298** | [0.6250, 0.6350] | **0.6188** | [0.5898, 0.6501] | **0.6195** | [0.5906, 0.6503] |
| **Macro-Recall** | **0.6326** | [0.6279, 0.6375] | **0.6222** | [0.5936, 0.6524] | **0.6231** | [0.5953, 0.6520] |
| **Macro-F1 Score** | **0.6302** | [0.6254, 0.6352] | **0.6181** | [0.5891, 0.6485] | **0.6191** | [0.5903, 0.6483] |
| **Cohen's Kappa ($\kappa$)** | **0.5152** | [0.5088, 0.5221] | **0.4963** | [0.4574, 0.5374] | **0.4975** | [0.4592, 0.5359] |
| **Macro ROC-AUC** | — | — | — | — | **0.8152** | — |

> **Affective-InfoNCE Top Performing Subjects**:
> - Subject 15: **85.67% Frame Accuracy**, **83.33% Trial Angular Consensus** (60/72 trials, Macro-F1: 0.8316, $\kappa=0.7778$)
> - Subject 02: **79.28% Frame Accuracy**, **76.39% Trial Angular Consensus** (55/72 trials, Macro-F1: 0.7506, $\kappa=0.6852$)
> - Subject 01: **67.11% Frame Accuracy**, **70.83% Trial Probe Consensus** (51/72 trials, Macro-F1: 0.7050, $\kappa=0.6111$)

### Physiological Responsive Cohort SOTA Benchmark (BCI Illiteracy Screening)

Under strict zero-leakage Stratified 4-Fold Trial Cross-Validation across the physiologically validated Responsive Cohort ($N = 12,525$ frames, $N = 360$ trials, Subjects 2, 4, 10, 13, 15), screening decisions are executed **strictly within the 18 training trials** per fold (internal 3-fold CV $\ge 50\%$):

| Cohort Group | Evaluated Subjects | Total Evaluated Trials | Sample Accuracy (95% CI) | Trial Consensus Accuracy (95% CI) | Trial Macro-F1 (95% CI) | Trial Cohen's $\kappa$ | Trial Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Responsive Affective Cohort** | **5** | **360** | **68.75%** [67.94%, 69.58%] | **66.67%** [62.22%, 71.67%] | **0.6647** [0.6197, 0.7137] | **0.5556** | **0.8701** |
| **Non-Responsive Cohort** | **10** | **720** | **57.76%** | **57.22%** | **0.5668** | **0.4296** | **0.7911** |
| **Complete Population** | **15** | **1,080** | **61.43%** | **60.37%** | **0.5996** | **0.4716** | **0.8181** |

> **Architecture Comparison on Responsive Cohort**:
> - **Calibrated Deep MLP**: **70.63% Sample Accuracy**, **68.06% Trial Consensus Accuracy**, **0.6766 Trial Macro-F1**
> - **Hybrid Ensemble (0.5 MLP + 0.5 LGB)**: **68.75% Sample Accuracy**, **66.67% Trial Consensus Accuracy**, **0.6647 Trial Macro-F1**
> - **Regularized LightGBM**: **65.21% Sample Accuracy**, **62.50% Trial Consensus Accuracy**, **0.6229 Trial Macro-F1**

### DynAcu-Net (Dynamic Latency-Anchored Cortical-Ocular Network) SOTA Benchmark

Under strict zero-leakage Stratified 4-Fold Trial Cross-Validation across all 45 sessions ($N = 23,040$ salient frames, $N = 1,080$ trials), combining **Dynamic Latency Anchoring (DPLA)**, **Bidirectional Cross-Attention Cortical-Ocular Manifold Fusion (COM-Fusion)**, and **Evidential Dirichlet Consensus (EAD)**:

| Evaluation Level | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro-Recall (95% CI) | Cohen's $\kappa$ (95% CI) | Macro ROC-AUC | Mean Cortical Gate ($\gamma$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frame-Level (Salient Frames)** | **62.20%** [61.57%, 62.86%] | **0.6173** [0.6110, 0.6237] | **0.6192** [0.6132, 0.6256] | **0.4942** [0.4862, 0.5029] | **0.8419** | **0.5426** (54.3% EEG / 45.7% Eye) |
| **Trial Dirichlet Consensus (EAD)** | **61.02%** [58.06%, 63.80%] | **0.6074** [0.5778, 0.6352] | **0.6102** [0.5808, 0.6371] | **0.4802** [0.4405, 0.5177] | **0.8407** | — |

> **DynAcu-Net Peak Responding Subjects & Sessions**:
> - Subject 15 Session 2: **92.93% Frame Accuracy**, **87.50% Trial Consensus** (21/24 trials correct)
> - Subject 14 Session 3: **88.69% Frame Accuracy**, **83.33% Trial Consensus** (20/24 trials correct)
> - Subject 15 Session 3: **85.32% Frame Accuracy**, **83.33% Trial Consensus** (20/24 trials correct)
> - Subject 07 Session 2: **75.83% Frame Accuracy**, **79.17% Trial Consensus** (19/24 trials correct)
> - Subject 15 Overall Mean: **81.64% Frame Accuracy**, **79.17% Trial Consensus** (57/72 trials, $\kappa=0.7222$)
> - Subject 02 Overall Mean: **78.32% Frame Accuracy**, **75.00% Trial Consensus** (54/72 trials, $\kappa=0.6667$)

### Subject-Dependent Trial Consensus SOTA Benchmark (Hou et al. 2023 & Cheng et al. 2021)

Under strict zero-leakage Stratified 4-Fold Trial Cross-Validation across all 45 sessions ($N = 37,575$ frames, $N = 1,080$ trials), combining **580D Differential Asymmetry (DASM) & Caudality (DCAU)** with **Session Neutral Baseline Reference Normalization ($\mu_{\text{neutral}}$)** and **Log-Odds Consensus Aggregation**:

| Model Architecture | Sample-Level Accuracy (95% CI) | Sample Macro-F1 (95% CI) | Trial-Consensus Accuracy (95% CI) | Trial Macro-F1 (95% CI) | Trial Cohen's $\kappa$ | Trial Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Calibrated Deep MLP** | **65.14%** [64.71%, 65.64%] | **0.6426** [0.6382, 0.6476] | **63.24%** [60.56%, 66.11%] | **0.6272** [0.5986, 0.6554] | **0.5099** | **0.8156** |
| **Hybrid Ensemble (LGB+MLP)** | **61.54%** [61.09%, 62.10%] | **0.6065** [0.6018, 0.6119] | **60.83%** [58.15%, 63.70%] | **0.6036** [0.5767, 0.6310] | 0.4778 | **0.8199** |
| **LightGBM Classifier** | **54.33%** [53.85%, 54.88%] | **0.5364** [0.5316, 0.5420] | **52.22%** [49.35%, 55.37%] | **0.5195** [0.4908, 0.5506] | 0.3630 | 0.7285 |

> **Peak Responding Subjects & Sessions**:
> - Subject 14 Session 3: **95.13% Sample Accuracy**, **91.67% Trial Consensus** (22/24 trials)
> - Subject 15 Session 2: **94.47% Sample Accuracy**, **87.50% Trial Consensus** (21/24 trials)
> - Subject 02 Session 2: **89.30% Sample Accuracy**, **83.33% Trial Consensus** (20/24 trials)
> - Subject 15 Overall Mean: **86.63% Sample Accuracy**, **84.72% Trial Consensus** (61/72 trials)
> - Subject 02 Overall Mean: **75.49% Sample Accuracy**, **72.22% Trial Consensus** (52/72 trials)

> **Top Intra-Session Performances (Spatial-Temporal 2D-CNN-BiGRU, Stratified 4-Fold CV)**:
> - Subject 07 Session 3: **77.73%** Accuracy, **0.7716** Macro-F1 ($N=750$)
> - Subject 15 Session 2: **76.45%** Accuracy, **0.7551** Macro-F1 ($N=760$)
> - Subject 15 Session 3: **74.67%** Accuracy, **0.7162** Macro-F1 ($N=750$)
> - Subject 14 Session 3: **74.00%** Accuracy, **0.7252** Macro-F1 ($N=750$)
> - Subject 06 Session 2: **70.13%** Accuracy, **0.6986** Macro-F1 ($N=760$)
> - Subject 01 Session 3: **67.73%** Accuracy, **0.6641** Macro-F1 ($N=750$)
> - Subject 06 Session 3: **66.80%** Accuracy, **0.6856** Macro-F1 ($N=750$)

> **3-Model Synergy Optimal Weights**:
> - *Subject-Dependent*: $\alpha_{\text{DANN}}=0.44, \alpha_{\text{LGB}}=0.34, \alpha_{\text{EEGNet}}=0.22 \implies \mathbf{68.09\%}$ Accuracy ($+2.56\%$ over standalone DANN).
> - *Cross-Subject*: $\alpha_{\text{DANN}}=0.04, \alpha_{\text{LGB}}=0.54, \alpha_{\text{EEGNet}}=0.42 \implies \mathbf{41.09\%}$ Accuracy ($+2.68\%$ over standalone DANN).

### Literature Paper Replication Benchmark (Sample-Level Shuffled Split)

To investigate why many SEED-IV publications report **95%+ accuracy**, we implemented the exact protocol commonly found in literature (random 80/20 sample shuffling per subject across all sessions):

| Model Architecture | Partitioning Protocol | Accuracy (95% CI) | Macro-F1 (95% CI) | Macro ROC-AUC (95% CI) | Cohen's $\kappa$ (95% CI) | Status / Generalization Target |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **LightGBM** | Sample-Level Shuffled 80/20 | **99.99%** [99.96%, 100.00%] | **0.9999** [0.9996, 1.0000] | **1.0000** [1.0000, 1.0000] | **0.9998** [0.9995, 1.0000] | Frame Interpolation / Trial Memorization |
| **Shallow MLP** | Sample-Level Shuffled 80/20 | **100.00%** [100.00%, 100.00%] | **1.0000** [1.0000, 1.0000] | **1.0000** [1.0000, 1.0000] | **1.0000** [1.0000, 1.0000] | Frame Interpolation / Trial Memorization |

### Temporal Autocorrelation Leakage-Free Frame Shuffling Benchmark ($\pm 8$s Exclusion Buffer)

To mathematically dissect the source of this near-100% classification accuracy, we evaluated intra-trial frame-level classification under a **quarantined $\pm 8$-second temporal exclusion buffer** ($\min |t_{\text{train}} - t_{\text{test}}| \ge 8.0\text{ s}$ within every continuous trial). This guarantees $0.00\%$ moving-average smoothing overlap ($\text{supp}(w_t) \cap \text{supp}(w_{\text{test}}) = \emptyset$).

Evaluated across all 15 subjects ($N = 37,575$ frames, 5-fold temporal block CV, 580D Cortical Asymmetry space):

| Partitioning Protocol | Model Architecture | Frame Accuracy (95% CI) | Macro-F1 Score (95% CI) | Cohen's $\kappa$ (95% CI) | Primary Error / Generalization Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Unbuffered Shuffled ($0\text{s}$ Buffer)** | **Shallow MLP** | **100.00%** [100.00%, 100.00%] | **1.0000** [1.0000, 1.0000] | **1.0000** [1.0000, 1.0000] | Moving-Average Filter Autocorrelation ($\rho > 0.95$) |
| **Unbuffered Shuffled ($0\text{s}$ Buffer)** | **Regularized LightGBM** | **100.00%** [100.00%, 100.00%] | **1.0000** [1.0000, 1.0000] | **1.0000** [1.0000, 1.0000] | Moving-Average Filter Autocorrelation ($\rho > 0.95$) |
| **Buffered Leak-Free Split ($\pm 8\text{s}$ Buffer)** | **Shallow MLP** | **99.46%** [99.38%, 99.53%] | **0.9943** [0.9934, 0.9951] | **0.9927** [0.9917, 0.9937] | **Stimulus Identity Memorization (Movie Clip Snooping)** |
| **Buffered Leak-Free Split ($\pm 8\text{s}$ Buffer)** | **Regularized LightGBM** | **99.10%** [99.01%, 99.19%] | **0.9906** [0.9896, 0.9915] | **0.9880** [0.9867, 0.9891] | **Stimulus Identity Memorization (Movie Clip Snooping)** |
| **Strict Whole-Trial Quarantine** | **Calibrated Deep MLP** | **65.14%** [64.71%, 65.64%] | **0.6426** [0.6382, 0.6476] | **0.5099** (Trial: **63.24%**) | **True Affective Generalization to Unseen Stimuli** |

> **Crucial Scientific Insight (Two-Stage Data Leakage Proof)**:
> 1. **Stage 1 (Moving-Average Autocorrelation)**: Unbuffered random shuffling allows models to interpolate between adjacent frames sharing moving-average kernel support. Enforcing $|t_{\text{train}} - t_{\text{test}}| \ge 8.0\text{ s}$ eliminates this correlation entirely.
> 2. **Stage 2 (Stimulus Identity Memorization)**: Even when temporal autocorrelation is eliminated ($99.46\%$), models achieve near-perfect classification because frames within the *same* continuous movie trial share identical tonic neural fingerprints (audio-visual sensory processing, narrative arc, luminance). The model memorizes *which movie was playing* rather than emotional valence.
> 3. **Conclusion**: Any intra-trial splitting (random or buffered) produces invalid affective BCI benchmarks. Only **Strict Whole-Trial Quarantine** (where entire trials are held out) evaluates true emotional generalizability.

> **Notice on Few-Shot Calibration**: The Few-Shot Calibration experiment (`evaluate_few_shot_calibration.py`) is INCOMPLETE -- a confirmed statistical inconsistency in the confidence interval computation was found and the experiment was halted pending investigation. Results are not yet valid and are excluded pending a fix.

---

## 2. Dataset Acquisition & Regeneration

> **Dataset Exclusion Notice**: Due to GitHub file size limits, the preprocessed binary dataset (`seed_iv_processed.npz`, ~84 MB) and raw Differential Entropy MATLAB feature files (`eeg_feature_smooth/`, ~335 MB) are excluded from the repository via `.gitignore`.

### To obtain and prepare the data:

1. **Request the SEED-IV Dataset**: Download the official SEED-IV dataset from the [BCMI Lab, Shanghai Jiao Tong University](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html).
2. **Place Raw Features**: Extract the `eeg_feature_smooth/` folder into the root directory of this repository:
   ```
   eri/
   ├── eeg_feature_smooth/
   │   ├── 1/ (1_20150507.mat ... 15_20150508.mat)
   │   ├── 2/
   │   └── 3/
   ```
3. **Compile and Verify**: Run the data loader to generate the unified `seed_iv_processed.npz`:
   ```bash
   python load_seed_iv.py
   python verify_seed_iv.py
   ```
   This generates a verified array containing 37,575 samples (62 channels x 5 frequency bands = 310 features) with aligned trial, session, subject, and emotion labels.

---

## 3. Installation & Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/Dakshified/EEG-Based-emotion-recognition.git
cd EEG-Based-emotion-recognition
```

### 2. Create and activate a Python virtual environment
```bash
# Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS:
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. GPU Verification
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 4. Repository Structure & Execution Guide

```
├── checkpoints/                 # Saved PyTorch model checkpoints (.pt)
│   └── dann_final/              # 10 verified Calibrated DANN fold models
├── figures/                     # Publication-quality 300 DPI evaluation figures
│   ├── ablations/               # DANN loss weight and lambda ablation curves
│   ├── adtc_net/                # ADTC-Net responsive cohort bar chart, 1,080-trial CM & ROC curves
│   ├── avc_net/                 # AVC-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── baselines/               # Baseline ROC, PR, Confusion Matrices, Friedman ranks
│   ├── buffered_shuffle/        # Unbuffered vs buffered vs trial quarantine leakage divergence plot
│   ├── coma_net/                # COMA-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── csec_net/                # CSEC-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── csec_refined/            # CSEC-Refined per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── cst_net/                 # CST-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── ds_gat_literature/       # DS-GAT literature benchmark figures
│   ├── dynacu_net/              # DynAcu-Net trial consensus accuracy, gating dynamics, and 1,080-trial CM
│   ├── explainability/          # Integrated Gradients & Occlusion attribution maps
│   ├── final_model/             # Calibrated DANN diagnostic figures
│   ├── geodesic_attribution/    # GEA Riemannian topomaps, band heatmaps & uncertainty decomposition
│   ├── hou_rfpn/                # Hou et al. (IEEE TIM 2023) RFPN per-subject accuracy, CM & ROC curves
│   ├── paper_replication/       # Literature sample-level replication ROC, CM & bar charts
│   ├── psec_net/                # PSEC-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── responsive_cohort/       # Responsive vs Non-Responsive trial accuracy & Responsive cohort CM
│   ├── responsible_ai/          # ECE reliability diagrams & selective abstention curves
│   ├── rmap_net/                # RMAP-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── salient_windows/         # Salience window energy dynamics, CM & accuracy lift charts
│   ├── sota_pipeline/           # 45-session accuracy distribution & pooled confusion matrix
│   ├── st_gode/                 # ST-GODE per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── topk_consensus/          # TopK-Quadratic-Net per-subject accuracy, 1,080-trial CM & ROC curves
│   ├── treh_net_replication/    # TREH-Net literature replication per-subject accuracy, CM & ROC curves
│   └── trial_consensus/         # Trial vs sample accuracy, log-odds dynamics, and consensus CM
├── granger_cache/               # Causal functional connectivity matrices
├── reference reseach papers/    # Reviewed academic literature
├── spatial_mapping.py           # Canonical 62-channel to 9x9 2D spatial grid transformation
├── temporal_dataset.py          # Trial-quarantined sliding sequence generator (T=8, stride=2)
├── model_spatial_temporal_cdan.py # Spatial 2D-CNN + Temporal Bi-GRU + 512D CDAN Discriminator
├── random_sampling/
│   ├── train_treh_net_literature.py # TREH-Net: Topological-Riemannian Evidential Hybrid Network
│   ├── train_random_sampling_literature.py # Conventional literature random 80/20 frame shuffling benchmark
│   └── results/                 # treh_net_replication_results.json, literature_replication_results.json
├── xai/
│   ├── geodesic_evidential_attribution.py # Geodesic Evidential Attribution (GEA) Riemannian XAI framework
│   └── results/                 # gea_attribution_metrics.json
├── train_rmap_net_sota.py       # Riemannian Manifold Alignment & Prototype-Guided Evidential Network (RMAP-Net)
├── train_cross_session_transfer_sota.py # Cross-Session Transfer Learning Network (CST-Net)
├── train_avc_net_sota.py        # Valence-Aware Climax & Margin-Gated Evidential Attention Network
├── train_topk_quadratic_sota.py # Top-K Climax Extraction & Quadratic Evidential Consensus Network
├── train_psec_net_sota.py       # Peak-Decisive Super-Evidential Consensus Network (PSEC-Net)
├── train_csec_refined_sota.py   # Refined Climax-Sharpened Evidential Consensus Network (CSEC-Refined)
├── train_csec_net_sota.py       # Climax-Sharpened Evidential Consensus Network (CSEC-Net)
├── train_adtc_net_sota.py       # Adaptive Dynamic Temperature-Calibrated Evidential Network (ADTC-Net)
├── train_coma_net_sota.py       # Cortical-Ocular Manifold Alignment Network (COMA-Net)
├── train_st_gode_sota.py        # Spatio-Temporal Graph Neural ODE with Evidential Dirichlet Consensus
├── train_target_high_acc_sota.py # Session-Level 4-Fold Trial-Quarantined SOTA RFPN Benchmark
├── train_hou_rfpn_sota.py       # Hou et al. (IEEE TIM 2023) 4-Matrix S2D Residual Feature Pyramid Network
├── train_ds_gat_literature_sota.py # Dynamical Spectral Graph Attention Network (DS-GAT)
├── train_responsive_cohort_sota.py # Physiological Responsive Cohort SOTA & BCI Illiteracy Screening
├── train_dynacu_net_sota.py     # Proprietary DynAcu-Net SOTA (DPLA + COM-Fusion + Dirichlet Consensus)
├── train_trial_consensus_sota.py # Subject-Dependent 580D asymmetry + baseline normalization + trial consensus
├── train_sota_hou_pipeline.py   # SOTA Hou et al. (2023) 15-ch spatial asymmetry + Space-to-Depth pipeline
├── train_upgraded_benchmarks.py # Dual-regime unified benchmark trainer & bootstrap CI evaluator
├── evaluate_buffered_frame_shuffle.py # Temporal autocorrelation leakage-free frame shuffle benchmark
├── evaluate_salient_windows_benchmark.py # Intra-trial salience extraction & transition filtering benchmark
├── evaluate_paper_replication_benchmark.py # Literature replication benchmark (sample-level 80/20)
├── train_final_dann.py          # Primary DANN model training pipeline (10 folds)
├── train_gat_kan_v2.py          # Proposed GAT-KAN v2 architecture
├── baseline_*.py                # 6 comparative baseline implementations
├── evaluate_dann_explainability.py   # XAI attribution & faithfulness suite
├── evaluate_dann_responsible_ai.py   # ECE, selective abstention & MC-Dropout uncertainty
├── evaluate_7models_significance.py  # Friedman omnibus & Holm-Bonferroni tests
├── evaluate_three_model_ensemble.py  # 3-model weighted synergy pipeline
└── requirements.txt             # Pinned project dependencies
```

### Running Experiments

* **Run Geodesic Evidential Attribution (GEA) Riemannian XAI Pipeline**:
  ```bash
  python -u xai/geodesic_evidential_attribution.py --device cuda
  ```
* **Run TREH-Net (Topological-Riemannian Evidential Hybrid Network | 95%–97% Window)**:
  ```bash
  python -u random_sampling/train_treh_net_literature.py --device cuda
  ```
* **Run Conventional Literature Random Sampling Replication Benchmark (95%–97% Window)**:
  ```bash
  python -u random_sampling/train_random_sampling_literature.py --device cuda
  ```
* **Run Riemannian Manifold Alignment & Prototype Evidential Network (RMAP-Net)**:
  ```bash
  python -u train_rmap_net_sota.py --device cuda
  ```
* **Run Cross-Session Transfer Learning Network (CST-Net)**:
  ```bash
  python -u train_cross_session_transfer_sota.py --device cuda
  ```
* **Run Valence-Aware Climax & Margin-Gated Evidential Attention Network (AVC-Net)**:
  ```bash
  python -u train_avc_net_sota.py --device cuda
  ```
* **Run Top-K Climax Extraction & Quadratic Evidential Consensus Network (TopK-Quadratic-Net)**:
  ```bash
  python -u train_topk_quadratic_sota.py --device cuda
  ```
* **Run Peak-Decisive Super-Evidential Consensus Network (PSEC-Net)**:
  ```bash
  python -u train_psec_net_sota.py --device cuda
  ```
* **Run Refined Climax-Sharpened Evidential Consensus Network (CSEC-Refined)**:
  ```bash
  python -u train_csec_refined_sota.py --device cuda
  ```
* **Run Climax-Sharpened Evidential Consensus Network (CSEC-Net)**:
  ```bash
  python -u train_csec_net_sota.py --device cuda
  ```
* **Run Adaptive Dynamic Temperature-Calibrated Evidential Network (ADTC-Net)**:
  ```bash
  python -u train_adtc_net_sota.py --device cuda
  ```
* **Run Cortical-Ocular Manifold Alignment Network (COMA-Net)**:
  ```bash
  python -u train_coma_net_sota.py --device cuda
  ```
* **Run Spatio-Temporal Graph Neural ODE with Evidential Dirichlet Consensus (ST-GODE)**:
  ```bash
  python -u train_st_gode_sota.py --device cuda
  ```
* **Run Hou et al. (IEEE TIM 2023) RFPN SOTA Benchmark (Strict 70/30 Trial Quarantine)**:
  ```bash
  python -u train_hou_rfpn_sota.py --device cuda
  ```
* **Run Temporal Autocorrelation Leakage-Free Frame Shuffle Benchmark**:
  ```bash
  python -u evaluate_buffered_frame_shuffle.py --device cuda
  ```
* **Run Targeted Denoising & Continuous Affective Manifold Benchmark**:
  ```bash
  python -u train_denoised_continuous_sota.py --device cuda
  ```
* **Run Affective-InfoNCE Latent Hypersphere Benchmark (SupCon + GPU Neuro-Augmentation)**:
  ```bash
  python -u train_affective_infonce_sota.py --device cuda
  ```
* **Run Physiological Responsive Cohort SOTA Benchmark (BCI Illiteracy Screening)**:
  ```bash
  python -u train_responsive_cohort_sota.py --device cuda
  ```
* **Run DynAcu-Net SOTA Benchmark (Proprietary DPLA + COM-Fusion + Dirichlet Consensus)**:
  ```bash
  python train_dynacu_net_sota.py --device cuda
  ```
* **Run Subject-Dependent Trial Consensus SOTA Benchmark (Hou et al. 2023 & Cheng et al. 2021)**:
  ```bash
  python train_trial_consensus_sota.py
  ```
* **Run SOTA Asymmetry Spatial Tensor & Baseline Calibration Pipeline (Hou et al. 2023)**:
  ```bash
  python train_sota_hou_pipeline.py
  ```
* **Run Affective Salience Window Extraction Benchmark**:
  ```bash
  python evaluate_salient_windows_benchmark.py
  ```
* **Run Literature Paper Replication Benchmark (95%+ Comparison Protocol)**:
  ```bash
  python evaluate_paper_replication_benchmark.py
  ```
* **Run Upgraded Spatial-Temporal 2D-CNN-BiGRU & CDAN Benchmarks**:
  ```bash
  # Run full dual-regime benchmarks (Regime A Intra-Session + Regime B CDAN Cross-Subject):
  python train_upgraded_benchmarks.py --mode all

  # Or run Regime A (Subject-Dependent / Intra-Session) only:
  python train_upgraded_benchmarks.py --mode regime_a

  # Or run Regime B (Transductive CDAN Cross-Subject) only:
  python train_upgraded_benchmarks.py --mode regime_b
  ```
* **Train Primary Calibrated DANN Model**:
  ```bash
  python train_final_dann.py
  ```
* **Run Explainability & Feature Attribution**:
  ```bash
  python evaluate_dann_explainability.py
  ```
* **Run Responsible AI & Uncertainty Decomposition**:
  ```bash
  python evaluate_dann_responsible_ai.py
  ```
* **Run 7-Model Statistical Significance Suite**:
  ```bash
  python evaluate_7models_significance.py
  ```
* **Run 3-Model Hybrid Ensemble Evaluation**:
  ```bash
  python evaluate_three_model_ensemble.py
  ```

---

## 5. Key Documentation & Artifacts

* **Technical Project Walkthrough**: `walkthrough.md` — Comprehensive documentation covering theoretical formulations, data quarantine audits, baseline comparisons, ablation studies, explainability axioms, ensemble synergies, literature replication analyses, affective salience extraction, SOTA asymmetry spatial modeling, DynAcu-Net cortical-ocular fusion, and Responsive Cohort BCI illiteracy screening.
* **Structured Results**:
  * `cst_net_results.json` / `cst_net_results.csv`
  * `avc_net_results.json` / `avc_net_results.csv`
  * `topk_consensus_results.json` / `topk_consensus_results.csv`
  * `psec_net_results.json` / `psec_net_results.csv`
  * `csec_refined_results.json` / `csec_refined_results.csv`
  * `csec_net_results.json` / `csec_net_results.csv`
  * `adtc_net_results.json` / `adtc_net_results.csv`
  * `coma_net_results.json` / `coma_net_results.csv`
  * `st_gode_results.json` / `st_gode_results.csv`
  * `target_high_acc_results.json` / `target_high_acc_results.csv`
  * `hou_rfpn_sota_results.json` / `hou_rfpn_sota_results.csv`
  * `buffered_frame_shuffle_results.json` / `buffered_frame_shuffle_results.csv`
  * `denoised_continuous_results.json` / `denoised_continuous_results.csv`
  * `affective_infonce_results.json` / `affective_infonce_results.csv`
  * `responsive_cohort_results.json` / `responsive_cohort_results.csv`
  * `dynacu_net_results.json` / `dynacu_net_results.csv`
  * `trial_consensus_sota_results.json` / `trial_consensus_sota_results.csv`
  * `sota_hou_pipeline_results.json` / `sota_hou_pipeline_results.csv`
  * `salient_window_benchmark_results.json` / `salient_window_benchmark_results.csv`
  * `paper_replication_benchmark_results.json` / `paper_replication_benchmark_results.csv`
  * `dann_final_results.json` / `dann_final_results.csv`
  * `three_model_ensemble_results.json` / `three_model_ensemble_results.csv`
  * `hybrid_ensemble_results.json` / `hybrid_ensemble_results.csv`
  * `statistical_significance_results.json`
  * `dann_ablation_results.json`

---

## 6. Citation & Attribution

If you find this codebase or benchmark methodology helpful in your research, please cite:

```bibtex
@misc{eeg_emotion_seed_iv_2026,
  author = {Daksh and Contributors},
  title = {Leakage-Safe EEG-Based Emotion Recognition and Domain Adaptation on SEED-IV},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Dakshified/EEG-Based-emotion-recognition}}
}
```
