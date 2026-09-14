# Contributing to the SEED-IV Affective Computing Framework

Thank you for your interest in contributing to our open-source research codebase for **EEG-Based Emotion Recognition on SEED-IV**! We welcome contributions ranging from novel geometric manifold architectures and domain adaptation techniques to bug fixes, documentation enhancements, and additional benchmark datasets.

---

## 1. Core Scientific Principles: The Zero-Leakage Charter

This project adheres to rigorous mathematical integrity and reproducibility standards. Affective computing models deployed for real-world brain-computer interfaces (BCIs) must generalize to **unseen continuous video stimuli**, not memorize temporal autocorrelation or tonic clip identities.

All contributions introducing new models or evaluation protocols must strictly comply with the **Zero-Leakage Charter**:

1. **Mutual Whole-Trial Quarantine**:
   Every out-of-sample evaluation must enforce complete whole-trial isolation:
   ```python
   assert len(set(train_trial_ids).intersection(set(test_trial_ids))) == 0, \
       "FATAL: Cross-trial data leakage detected! Test trials must be strictly held out."
   ```
2. **Strictly Inductive Preprocessing**:
   Any normalizers, feature scalers, or dimensionality reduction transformers (e.g., `StandardScaler`, `PCA`, `Covariances`) must be fitted **exclusively on training trial folds** and applied transformatively to test folds.
3. **Training-Fold Baseline Insulation**:
   Neutral reference vectors ($\boldsymbol{\mu}_{\text{neutral}}$) or resting-state baselines must be computed strictly from training neutral trials within the fold.
4. **Literature Replication Separation**:
   Any experiments replicating conventional sample-level random frame shuffling must reside strictly within `random_sampling/` and be explicitly labeled as literature baseline replications.

---

## 2. Development Environment Setup

### Prerequisites:
- Python 3.10 or higher (Python 3.11 recommended)
- NVIDIA GPU with CUDA 11.8+ or 12.x support (CPU fallback supported)
- Git

### Step-by-Step Installation:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dakshified/EEG-Based-emotion-recognition.git
   cd EEG-Based-emotion-recognition
   ```

2. **Create and Activate a Virtual Environment**:
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

3. **Install PyTorch with CUDA Support**:
   Install the appropriate PyTorch build for your CUDA version (e.g., CUDA 12.4):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify Installation**:
   ```bash
   python -c "import torch, sklearn, pyriemann; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
   ```

---

## 3. Dataset Preparation

To execute benchmarks requiring raw features:
1. Request and download the **SEED-IV** dataset from Shanghai Jiao Tong University's [BCMI Lab](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html).
2. Extract the `eeg_feature_smooth/` directory into the repository root:
   ```
   eri/
   ├── eeg_feature_smooth/
   │   ├── 1/ (1_20150507.mat ... 15_20150508.mat)
   │   ├── 2/
   │   └── 3/
   ```
3. Compile and verify the processed dataset:
   ```bash
   python load_seed_iv.py
   python verify_seed_iv.py
   ```
   This generates `seed_iv_processed.npz` containing 37,575 1-second frames across 62 channels $\times$ 5 frequency bands.

---

## 4. Running the Test & Benchmark Suites

Before submitting a Pull Request, verify that all core pipelines execute without errors:

### 1. Literature Replication & Advanced Evaluation Suite:
```bash
# Run TREH-Net 95.64% baseline
python -u random_sampling/train_treh_net_literature.py --device cuda

# Run TREH-Net Advanced Evaluation (Ablation, t-SNE, ECE Reliability Diagrams)
python -u evaluation/evaluate_treh_advanced_suite.py --device cuda
```

### 2. Zero-Leakage Quarantined SOTA:
```bash
# Run RMAP-Net (Riemannian Manifold Alignment & Prototype Transfer)
python -u train_rmap_net_sota.py --device cuda

# Run CST-Net (Cross-Session Multi-Source Auxiliary Transfer)
python -u train_cross_session_transfer_sota.py --device cuda
```

### 3. Native Riemannian Explainable AI:
```bash
# Run Geodesic Evidential Attribution (GEA)
python -u xai/geodesic_evidential_attribution.py --device cuda
```

---

## 5. Repository Layout & Contribution Boundaries

```
eri/
├── demo/                    # Interactive Jupyter walkthrough notebooks
├── evaluation/              # Advanced evaluation suites, ablation studies & ECE calibration
│   └── results/             # Structured benchmark JSON metrics
├── figures/                 # 300 DPI publication figures organized by topic
│   ├── evaluation/          # t-SNE manifolds, calibration curves, ablation bars
│   ├── geodesic_attribution/# GEA topomaps, band matrices, uncertainty dynamics
│   ├── rmap_net/            # RMAP-Net confusion matrices & ROC curves
│   └── treh_net_replication/# TREH-Net replication figures
├── random_sampling/         # Literature replication benchmarks & data leakage tests
├── xai/                     # Riemannian XAI & Geodesic Evidential Attribution
├── train_rmap_net_sota.py   # RMAP-Net SOTA implementation
├── train_cross_session_transfer_sota.py # CST-Net SOTA implementation
├── train_avc_net_sota.py    # AVC-Net SOTA implementation
├── CONTRIBUTING.md          # Open-source collaboration guidelines
└── requirements.txt         # Project dependencies
```

---

## 6. Pull Request Submission Checklist

When opening a Pull Request:
- [ ] Code follows PEP 8 formatting standards.
- [ ] Programmatic assertions for zero-leakage trial quarantine are present and active.
- [ ] Any new metrics are exported in structured JSON format to `results/` or `evaluation/results/`.
- [ ] Any new figures are exported at 300 DPI in `figures/`.
- [ ] Documentation (`README.md`, `walkthrough.md`) is updated with mathematical rationale and benchmark comparisons.
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat:` for new architectures or feature pipelines.
  - `fix:` for bug fixes.
  - `docs:` for documentation updates.
  - `perf:` for performance optimizations.
  - `test:` for evaluation suites or sanity checks.

Thank you for advancing open, rigorous, and leak-free affective neurocomputing!
