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

> **Scientific Insight**: In SEED-IV, DE features undergo moving-average temporal smoothing across consecutive 1-second frames within each continuous trial ($\rho > 0.99$). When samples are randomly shuffled, adjacent seconds of the *same* trial are split into train and test sets, enabling classifiers to achieve near-100% accuracy via temporal interpolation. Under rigorous **trial-quarantined** evaluation (where test trials and subjects are strictly out-of-sample), true state-of-the-art performance is **68.09%** (Subject-Dependent) and **41.09%** (Cross-Subject).

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
├── figures/                     # 162 publication-quality 300 DPI evaluation figures
│   ├── ablations/               # DANN loss weight and lambda ablation curves
│   ├── baselines/               # Baseline ROC, PR, Confusion Matrices, Friedman ranks
│   ├── dynacu_net/              # DynAcu-Net trial consensus accuracy, gating dynamics, and 1,080-trial CM
│   ├── explainability/          # Integrated Gradients & Occlusion attribution maps
│   ├── final_model/             # Calibrated DANN diagnostic figures
│   ├── paper_replication/       # Literature sample-level replication ROC, CM & bar charts
│   ├── responsible_ai/          # ECE reliability diagrams & selective abstention curves
│   ├── salient_windows/         # Salience window energy dynamics, CM & accuracy lift charts
│   ├── sota_pipeline/           # 45-session accuracy distribution & pooled confusion matrix
│   └── trial_consensus/         # Trial vs sample accuracy, log-odds dynamics, and consensus CM
├── granger_cache/               # Causal functional connectivity matrices
├── reference reseach papers/    # Reviewed academic literature
├── spatial_mapping.py           # Canonical 62-channel to 9x9 2D spatial grid transformation
├── temporal_dataset.py          # Trial-quarantined sliding sequence generator (T=8, stride=2)
├── model_spatial_temporal_cdan.py # Spatial 2D-CNN + Temporal Bi-GRU + 512D CDAN Discriminator
├── train_dynacu_net_sota.py     # Proprietary DynAcu-Net SOTA (DPLA + COM-Fusion + Dirichlet Consensus)
├── train_trial_consensus_sota.py # Subject-Dependent 580D asymmetry + baseline normalization + trial consensus
├── train_sota_hou_pipeline.py   # SOTA Hou et al. (2023) 15-ch spatial asymmetry + Space-to-Depth pipeline
├── train_upgraded_benchmarks.py # Dual-regime unified benchmark trainer & bootstrap CI evaluator
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

* **Technical Project Walkthrough**: `walkthrough.md` — Comprehensive documentation covering theoretical formulations, data quarantine audits, baseline comparisons, ablation studies, explainability axioms, ensemble synergies, literature replication analyses, affective salience extraction, SOTA asymmetry spatial modeling, and DynAcu-Net cortical-ocular fusion.
* **Structured Results**:
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
