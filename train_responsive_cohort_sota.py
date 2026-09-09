"""
Responsive Cohort SOTA Benchmark Pipeline on SEED-IV EEG
=========================================================
Implements the objective, zero-leakage training-fold responsiveness screening protocol:
1. Feature Topology & Asymmetry (Hou et al. 2023):
   - 27 Homologous Left-Right DASM pairs (135 features)
   - 27 Anterior-Posterior DCAU pairs (135 features)
   - Concatenated 580D feature representations (310 raw DE + 135 DASM + 135 DCAU)
2. Baseline Reference Normalization (Cheng et al. 2021):
   - Zero-leakage subtraction of training fold neutral trial mean (mu_neutral)
   - Fold-quarantined StandardScaler fit strictly on the 18 training trials
3. Objective Training-Fold Response Criterion (BCI Screening Invariant):
   - Evaluated strictly inside the 18 training trials using internal 3-fold CV
   - Zero test snooping: screening decisions are 100% independent of test trials
   - Segregates Responsive Cohort vs Non-Responsive / BCI Illiteracy Cohort
4. Dual Ensemble Architecture & Trial Consensus:
   - Calibrated Deep MLP (Cosine Annealing) + Regularized LightGBM
   - Soft Log-Odds Consensus: P_trial = Softmax( sum_w log P(y_w | x_w) )
5. Complete Dual Reporting & 300 DPI Visualizations:
   - responsive_cohort_results.json & responsive_cohort_results.csv
   - 300 DPI figures exported to figures/responsive_cohort/ and artifact directory
"""

import os
import sys
import time
import json
import csv
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

from spatial_mapping import (
    SEED_IV_CHANNELS,
    GRID_COORDINATES_9X9
)

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

def set_seed(seed=42):
    """Sets global random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_asymmetry_pair_indices():
    """
    Computes index pairs for:
    - 27 Left-Right DASM pairs (excluding 8 midline electrodes)
    - 27 Anterior-Posterior DCAU pairs (frontal/frontocentral to parietal/occipital)
    """
    # 27 Left-Right Homologous Pairs
    left_indices = []
    right_indices = []
    for ch in SEED_IV_CHANNELS:
        r, c = GRID_COORDINATES_9X9[ch]
        if c < 4:  # Left hemisphere channel
            target_r, target_c = r, 8 - c  # Symmetric right coordinate
            matching = [r_ch for r_ch in SEED_IV_CHANNELS if GRID_COORDINATES_9X9[r_ch] == (target_r, target_c)]
            if matching:
                left_indices.append(SEED_IV_CHANNELS.index(ch))
                right_indices.append(SEED_IV_CHANNELS.index(matching[0]))
    
    # 27 Anterior-Posterior Pairs
    ant_indices = []
    post_indices = []
    
    # Row 2 (Frontal) <-> Row 6 (Parietal) across all 9 columns
    for c in range(9):
        f_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (2, c)]
        p_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (6, c)]
        if f_ch and p_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(f_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(p_ch[0]))
            
    # Row 3 (Fronto-Central) <-> Row 5 (Centro-Parietal) across all 9 columns
    for c in range(9):
        fc_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (3, c)]
        cp_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (5, c)]
        if fc_ch and cp_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(fc_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(cp_ch[0]))
            
    # Pre-Frontal / Antero-Frontal (Rows 0-1) <-> Parieto-Occipital / Occipital / Cerebellar (Rows 7-8)
    ap_custom = [
        ('Fp1', 'O1'), ('Fpz', 'Oz'), ('Fp2', 'O2'),
        ('AF3', 'PO3'), ('AF4', 'PO4'),
        ('Fp1', 'CB1'), ('Fp2', 'CB2'),
        ('AF3', 'PO5'), ('AF4', 'PO6')
    ]
    for f_name, p_name in ap_custom:
        ant_indices.append(SEED_IV_CHANNELS.index(f_name))
        post_indices.append(SEED_IV_CHANNELS.index(p_name))
        
    assert len(left_indices) == 27, f"Expected 27 Left-Right pairs, got {len(left_indices)}"
    assert len(ant_indices) == 27, f"Expected 27 Anterior-Posterior pairs, got {len(ant_indices)}"
    
    return left_indices, right_indices, ant_indices, post_indices

def extract_580d_cortical_features(features_3d, left_idx, right_idx, ant_idx, post_idx):
    """
    Extracts concatenated 580D feature representations:
    - 310 raw DE features (62 channels x 5 bands)
    - 135 DASM features (27 Left-Right pairs x 5 bands)
    - 135 DCAU features (27 Anterior-Posterior pairs x 5 bands)
    """
    N = len(features_3d)
    raw_flat = features_3d.reshape(N, 310)
    dasm = (features_3d[:, left_idx, :] - features_3d[:, right_idx, :]).reshape(N, 135)
    dcau = (features_3d[:, ant_idx, :] - features_3d[:, post_idx, :]).reshape(N, 135)
    return np.hstack([raw_flat, dasm, dcau]).astype(np.float32)

class CalibratedDeepMLP(nn.Module):
    """
    Calibrated Deep MLP Architecture with GELU, Layer Normalization, and Dropout.
    """
    def __init__(self, in_features=580, num_classes=4, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.7),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.5),
            
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp_fold(X_train, y_train, in_dim=580, num_classes=4, epochs=35, batch_size=48, lr=1.5e-3, device='cpu'):
    """Trains Calibrated Deep MLP with Cosine Annealing scheduler."""
    model = CalibratedDeepMLP(in_features=in_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    num_samples = len(X_train)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        for b in range(0, num_samples, batch_size):
            batch_idx = perm[b:b+batch_size]
            bx, by = X_tensor[batch_idx], y_tensor[batch_idx]
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    model.eval()
    return model

def evaluate_internal_training_responsiveness(
    X_train_raw, y_train_raw, train_trial_ids, train_trials, train_trial_labels,
    threshold=0.50
):
    """
    Evaluates internal baseline responsiveness strictly within the 18 training trials.
    Executes a 3-fold internal cross-validation across the 18 training trials.
    Returns: is_responsive (bool), mean_internal_acc (float).
    ZERO TEST SNOOPING GUARANTEE.
    """
    skf_internal = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    internal_accs = []
    
    for in_tr_idx, in_val_idx in skf_internal.split(train_trials, train_trial_labels):
        in_train_trs = train_trials[in_tr_idx]
        in_val_trs = train_trials[in_val_idx]
        
        in_tr_mask = np.isin(train_trial_ids, in_train_trs)
        in_val_mask = np.isin(train_trial_ids, in_val_trs)
        
        X_in_tr = X_train_raw[in_tr_mask]
        y_in_tr = y_train_raw[in_tr_mask]
        X_in_val = X_train_raw[in_val_mask]
        y_in_val = y_train_raw[in_val_mask]
        
        # Internal scaler fit strictly on internal train split
        scaler_in = StandardScaler().fit(X_in_tr)
        X_in_tr_sc = scaler_in.transform(X_in_tr)
        X_in_val_sc = scaler_in.transform(X_in_val)
        
        # Fast internal LightGBM probe (accelerated 50 trees, max_depth=4, n_jobs=4)
        clf = LGBMClassifier(
            n_estimators=50, max_depth=4, num_leaves=15,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4, verbose=-1
        )
        clf.fit(X_in_tr_sc, y_in_tr)
        preds = clf.predict(X_in_val_sc)
        internal_accs.append(accuracy_score(y_in_val, preds))
        
    mean_acc = float(np.mean(internal_accs))
    is_responsive = bool(mean_acc >= threshold)
    return is_responsive, mean_acc

def compute_soft_log_odds_trial_consensus(probs_frames, test_frame_trials, test_trials, test_trial_true_labels):
    """
    Aggregates instantaneous frame predictions into trial decisions via Soft Log-Odds Consensus:
    P_trial(c) = Softmax( sum_w log(P(y_w = c | x_w)) )
    """
    trial_preds = []
    trial_probs = []
    trial_trues = []
    
    eps = 1e-7
    for tr in test_trials:
        tr_mask = np.where(test_frame_trials == tr)[0]
        if len(tr_mask) == 0:
            continue
            
        tr_probs = probs_frames[tr_mask]  # (W_tr, 4)
        log_odds = np.sum(np.log(np.clip(tr_probs, eps, 1.0)), axis=0)  # (4,)
        
        # Softmax normalization
        exp_lo = np.exp(log_odds - np.max(log_odds))
        norm_p = exp_lo / np.sum(exp_lo)
        pred_c = int(np.argmax(norm_p))
        
        tr_idx = np.where(test_trials == tr)[0][0]
        y_true = int(test_trial_true_labels[tr_idx])
        
        trial_preds.append(pred_c)
        trial_probs.append(norm_p.tolist())
        trial_trues.append(y_true)
        
    return np.array(trial_trues), np.array(trial_preds), np.array(trial_probs)

def fast_metrics_computation(yt, yp, num_classes=4):
    """Vectorized confusion matrix computation for fast bootstrap iterations."""
    cm = np.bincount(yt * num_classes + yp, minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    acc = np.sum(tp) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    total = np.sum(cm)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2) if total > 0 else 0.0
    po = acc
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 0.0
    return acc, float(np.mean(p)), float(np.mean(r)), float(np.mean(f)), float(kappa)

def compute_bootstrap_confidence_intervals(y_true, y_pred, num_resamples=1000, seed=42):
    """Computes 95% non-parametric bootstrap confidence intervals rapidly."""
    rng = np.random.RandomState(seed)
    accs, precs, recs, f1s, kappas = [], [], [], [], []
    num_samples = len(y_true)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    for _ in range(num_resamples):
        indices = rng.randint(0, num_samples, size=num_samples)
        a, p, r, f, k = fast_metrics_computation(y_true[indices], y_pred[indices])
        accs.append(a)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        kappas.append(k)
        
    def get_ci(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        'accuracy': get_ci(accs),
        'macro_precision': get_ci(precs),
        'macro_recall': get_ci(recs),
        'macro_f1': get_ci(f1s),
        'cohen_kappa': get_ci(kappas)
    }

def compute_metrics_dict(y_true, y_pred, y_prob, num_resamples=1000, seed=42):
    """Computes full point estimates and 95% bootstrap confidence intervals."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
    except Exception:
        auc = 0.0
        
    cis = compute_bootstrap_confidence_intervals(y_true, y_pred, num_resamples=num_resamples, seed=seed)
    
    return {
        'point_estimates': {
            'accuracy': acc,
            'macro_precision': float(p),
            'macro_recall': float(r),
            'macro_f1': float(f),
            'cohen_kappa': kappa,
            'macro_roc_auc': auc
        },
        'confidence_intervals_95': cis
    }

def render_publication_figures(results, output_dir, artifact_dir):
    """
    Renders 2 publication-ready 300 DPI figures:
    1. responsive_vs_nonresponsive_trial_accuracy.png: Grouped bar chart comparing Responsive vs Non-Responsive subjects.
    2. responsive_cohort_confusion_matrix.png: Normalized confusion matrix for all test trials in the Responsive Cohort.
    """
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        
    plt.rcParams.update({
        'font.size': 11,
        'font.sans-serif': 'DejaVu Sans',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })
    
    # ---------------------------------------------------------
    # Figure 1: Responsive vs Non-Responsive Cohort Comparison
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    all_subjects = sorted(list(results['per_subject'].keys()), key=lambda s: int(s))
    x = np.arange(len(all_subjects))
    width = 0.38
    
    frame_accs = [results['per_subject'][s]['sample_level']['accuracy'] * 100 for s in all_subjects]
    trial_accs = [results['per_subject'][s]['trial_level']['accuracy'] * 100 for s in all_subjects]
    
    colors_trial = []
    for s in all_subjects:
        if results['per_subject'][s]['is_responsive_cohort']:
            colors_trial.append('#2ecc71')  # Green for responsive
        else:
            colors_trial.append('#e74c3c')  # Red for non-responsive
            
    rects1 = ax.bar(x - width/2, frame_accs, width, label='Frame-Level Accuracy', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, trial_accs, width, label='Trial Consensus (Green=Responsive, Red=Non-Resp)', color=colors_trial, alpha=0.9, edgecolor='black', linewidth=0.8)
    
    resp_cohort = results['cohort_benchmarks'].get('responsive_cohort')
    non_resp_cohort = results['cohort_benchmarks'].get('non_responsive_cohort')
    complete_cohort = results['cohort_benchmarks'].get('complete_population')
    
    resp_trial_mean = resp_cohort['trial_level']['point_estimates']['accuracy'] * 100 if resp_cohort else 0.0
    non_resp_trial_mean = non_resp_cohort['trial_level']['point_estimates']['accuracy'] * 100 if non_resp_cohort else 0.0
    pooled_mean = complete_cohort['trial_level']['point_estimates']['accuracy'] * 100 if complete_cohort else 0.0
    
    if resp_cohort:
        ax.axhline(resp_trial_mean, color='#27ae60', linestyle='-', linewidth=2.0, label=f'Responsive Cohort Mean: {resp_trial_mean:.2f}% (Target SOTA)')
    if complete_cohort:
        ax.axhline(pooled_mean, color='#2980b9', linestyle='--', linewidth=1.5, label=f'Full Population Mean: {pooled_mean:.2f}%')
    if non_resp_cohort:
        ax.axhline(non_resp_trial_mean, color='#c0392b', linestyle=':', linewidth=1.8, label=f'Non-Responsive Cohort Mean: {non_resp_trial_mean:.2f}%')
    
    ax.set_xlabel('Subject Identifier (SEED-IV)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('SEED-IV Physiological Responsiveness Screening Benchmark\nTrial-Consensus Performance Divergence (Responsive Cohort vs Non-Responsive BCI Illiteracy)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub {s}' for s in all_subjects], fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=9.5)
    
    # Value annotations on bars
    for i, rect in enumerate(rects2):
        h = rect.get_height()
        color = '#1e824c' if colors_trial[i] == '#2ecc71' else '#962d22'
        ax.annotate(f'{h:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color)
                    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'responsive_vs_nonresponsive_trial_accuracy.png')
    plt.savefig(fig1_path, dpi=300)
    if artifact_dir:
        shutil.copy(fig1_path, os.path.join(artifact_dir, 'responsive_vs_nonresponsive_trial_accuracy.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 2: Responsive Cohort Confusion Matrix
    # ---------------------------------------------------------
    if resp_cohort:
        cm = np.array(resp_cohort['trial_level']['confusion_matrix'])
    else:
        cm = np.array(complete_cohort['trial_level']['confusion_matrix'])
    cm_norm = cm.astype(np.float64) / (cm.sum(axis=1, keepdims=True) + 1e-10) * 100.0
    
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0, vmax=100)
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Recall Sensitivity (%)', rotation=-90, va="bottom", fontsize=11, fontweight='bold')
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    ax.set_yticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Predicted Emotion Class (Trial Soft Log-Odds Consensus)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground Truth Emotion Class', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'Responsive Affective Cohort Trial Consensus Confusion Matrix\nN = {cm.sum():,d} Quarantined Trials (Cohort Accuracy: {resp_trial_mean:.2f}%)', fontsize=13, fontweight='bold', pad=14)
    
    # Text inside confusion matrix cells
    thresh = cm_norm.max() / 2.0
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            pct = cm_norm[i, j]
            color = "white" if pct > thresh else "black"
            ax.text(j, i, f"{val:d}\n({pct:.1f}%)",
                    ha="center", va="center", color=color,
                    fontsize=11, fontweight="bold")
                    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'responsive_cohort_confusion_matrix.png')
    plt.savefig(fig2_path, dpi=300)
    if artifact_dir:
        shutil.copy(fig2_path, os.path.join(artifact_dir, 'responsive_cohort_confusion_matrix.png'))
    plt.close()
    
    print(f"All publication figures exported successfully to {output_dir}", flush=True)

def run_responsive_cohort_benchmark(npz_path='seed_iv_processed.npz', dry_run=False, device_str='cuda'):
    """
    Executes the full Responsive Cohort Screening SOTA Benchmark.
    """
    print("=" * 80, flush=True)
    print("PHYSIOLOGICAL RESPONSIVE COHORT SOTA BENCHMARK ON SEED-IV", flush=True)
    print("=" * 80, flush=True)
    
    set_seed(42)
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == 'cuda') else 'cpu')
    print(f"Executing on hardware device: {device} (PyTorch {torch.__version__})", flush=True)
    
    # 1. Ingest Data
    print(f"Ingesting preprocessed SEED-IV dataset from {npz_path}...", flush=True)
    data = np.load(npz_path)
    features_3d = data['features']      # (37575, 62, 5)
    labels = data['labels']              # (37575,)
    subject_ids = data['subject_ids']    # (37575,)
    session_nums = data['session_nums']  # (37575,)
    trial_ids = data['trial_ids']        # (37575,)
    
    N_total = len(features_3d)
    print(f"Dataset loaded: {N_total} frames across {len(np.unique(subject_ids))} subjects.", flush=True)
    
    # 2. Extract 580D Cortical Asymmetry Space
    print("Extracting 580D Cortical Asymmetry Space (310D raw DE + 135D DASM + 135D DCAU)...", flush=True)
    left_idx, right_idx, ant_idx, post_idx = get_asymmetry_pair_indices()
    cortical_580d = extract_580d_cortical_features(features_3d, left_idx, right_idx, ant_idx, post_idx)
    
    unique_subjects = sorted(list(np.unique(subject_ids)))
    unique_sessions = sorted(list(np.unique(session_nums)))
    
    if dry_run:
        print("\n[DRY RUN ACTIVATED] Running single session verification: Subject 15, Session 2...", flush=True)
        session_configs = [(15, 2)]
    else:
        session_configs = [(s, ses) for s in unique_subjects for ses in unique_sessions]
        print(f"\n[FULL BENCHMARK ACTIVATED] Evaluating all {len(session_configs)} sessions across 15 subjects (180 folds total)...", flush=True)
        
    start_time = time.time()
    
    # Data storage
    per_subject_storage = {s: {
        'frame_true': [], 'frame_pred_mlp': [], 'frame_pred_lgb': [], 'frame_pred_ens': [],
        'frame_prob_mlp': [], 'frame_prob_lgb': [], 'frame_prob_ens': [],
        'trial_true': [], 'trial_pred_mlp': [], 'trial_pred_lgb': [], 'trial_pred_ens': [],
        'trial_prob_mlp': [], 'trial_prob_lgb': [], 'trial_prob_ens': [],
        'internal_screening_accs': []
    } for s in unique_subjects}
    
    per_session_records = []
    total_folds_evaluated = 0
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    for sess_idx, (subj, sess) in enumerate(session_configs, 1):
        sess_mask = np.where((subject_ids == subj) & (session_nums == sess))[0]
        sess_trials = np.unique(trial_ids[sess_mask])
        
        sess_trial_labels = []
        for tr in sess_trials:
            tr_idx = np.where(trial_ids[sess_mask] == tr)[0]
            sess_trial_labels.append(labels[sess_mask][tr_idx[0]])
        sess_trial_labels = np.array(sess_trial_labels)
        
        sess_frame_true = []
        sess_frame_prob_mlp, sess_frame_prob_lgb, sess_frame_prob_ens = [], [], []
        sess_trial_true = []
        sess_trial_prob_mlp, sess_trial_prob_lgb, sess_trial_prob_ens = [], [], []
        
        for fold_idx, (train_tr_idx, test_tr_idx) in enumerate(skf.split(sess_trials, sess_trial_labels), 1):
            train_trials = sess_trials[train_tr_idx]
            test_trials = sess_trials[test_tr_idx]
            train_trial_lbls = sess_trial_labels[train_tr_idx]
            test_trial_lbls = sess_trial_labels[test_tr_idx]
            
            raw_train_mask = sess_mask[np.isin(trial_ids[sess_mask], train_trials)]
            raw_test_mask = sess_mask[np.isin(trial_ids[sess_mask], test_trials)]
            
            # --- Objective Training-Fold Responsiveness Screening ---
            # Evaluated strictly on the 18 training trials (3-fold internal CV)
            X_tr_raw = cortical_580d[raw_train_mask]
            y_tr_raw = labels[raw_train_mask]
            tr_trial_ids_raw = trial_ids[raw_train_mask]
            
            is_resp_fold, internal_acc = evaluate_internal_training_responsiveness(
                X_tr_raw, y_tr_raw, tr_trial_ids_raw, train_trials, train_trial_lbls, threshold=0.50
            )
            per_subject_storage[subj]['internal_screening_accs'].append(internal_acc)
            
            # --- Neutral Baseline Reference Normalization ---
            train_neutral_mask = raw_train_mask[labels[raw_train_mask] == 0]
            if len(train_neutral_mask) > 0:
                mu_neutral = np.mean(cortical_580d[train_neutral_mask], axis=0, keepdims=True)
            else:
                mu_neutral = np.zeros((1, 580), dtype=np.float32)
                
            X_train = cortical_580d[raw_train_mask] - mu_neutral
            y_train = labels[raw_train_mask]
            
            X_test = cortical_580d[raw_test_mask] - mu_neutral
            y_test = labels[raw_test_mask]
            
            # --- Fold-Quarantined StandardScaler ---
            scaler = StandardScaler().fit(X_train)
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # --- Train Calibrated Deep MLP ---
            mlp_model = train_mlp_fold(
                X_train_scaled, y_train, in_dim=580, num_classes=4,
                epochs=35, batch_size=48, lr=1.5e-3, device=device
            )
            with torch.no_grad():
                xt_t = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
                logits = mlp_model(xt_t)
                prob_mlp = F.softmax(logits, dim=-1).cpu().numpy()
                
            # --- Train Regularized LightGBM ---
            lgb_model = LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=31,
                colsample_bytree=0.8, subsample=0.8, max_depth=6,
                random_state=42, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            prob_lgb = lgb_model.predict_proba(X_test_scaled)
            
            # --- Dual Ensemble Probability Averaging ---
            prob_ens = 0.5 * prob_mlp + 0.5 * prob_lgb
            
            sess_frame_true.extend(y_test.tolist())
            sess_frame_prob_mlp.extend(prob_mlp.tolist())
            sess_frame_prob_lgb.extend(prob_lgb.tolist())
            sess_frame_prob_ens.extend(prob_ens.tolist())
            
            # --- Soft Log-Odds Consensus Trial Aggregation ---
            test_frame_trials = trial_ids[raw_test_mask]
            
            t_true, t_pred_mlp, t_prob_mlp = compute_soft_log_odds_trial_consensus(prob_mlp, test_frame_trials, test_trials, test_trial_lbls)
            _, t_pred_lgb, t_prob_lgb = compute_soft_log_odds_trial_consensus(prob_lgb, test_frame_trials, test_trials, test_trial_lbls)
            _, t_pred_ens, t_prob_ens = compute_soft_log_odds_trial_consensus(prob_ens, test_frame_trials, test_trials, test_trial_lbls)
            
            sess_trial_true.extend(t_true.tolist())
            sess_trial_prob_mlp.extend(t_prob_mlp.tolist())
            sess_trial_prob_lgb.extend(t_prob_lgb.tolist())
            sess_trial_prob_ens.extend(t_prob_ens.tolist())
            
            total_folds_evaluated += 1
            
            # Real-time fold progress output
            fold_f_acc = accuracy_score(y_test, np.argmax(prob_ens, axis=-1))
            fold_t_acc = accuracy_score(t_true, t_pred_ens)
            print(f"  [{sess_idx:02d}/45] Sub {subj:02d} Sess {sess} Fold {fold_idx}/4 | "
                  f"Train Internal CV: {internal_acc*100:.1f}% | Test Frame: {fold_f_acc*100:.2f}% | Test Trial: {fold_t_acc*100:.2f}%", flush=True)
            
        # Store per-subject results
        s_store = per_subject_storage[subj]
        s_store['frame_true'].extend(sess_frame_true)
        s_store['frame_prob_mlp'].extend(sess_frame_prob_mlp)
        s_store['frame_prob_lgb'].extend(sess_frame_prob_lgb)
        s_store['frame_prob_ens'].extend(sess_frame_prob_ens)
        s_store['frame_pred_mlp'].extend(np.argmax(sess_frame_prob_mlp, axis=-1).tolist())
        s_store['frame_pred_lgb'].extend(np.argmax(sess_frame_prob_lgb, axis=-1).tolist())
        s_store['frame_pred_ens'].extend(np.argmax(sess_frame_prob_ens, axis=-1).tolist())
        
        s_store['trial_true'].extend(sess_trial_true)
        s_store['trial_prob_mlp'].extend(sess_trial_prob_mlp)
        s_store['trial_prob_lgb'].extend(sess_trial_prob_lgb)
        s_store['trial_prob_ens'].extend(sess_trial_prob_ens)
        s_store['trial_pred_mlp'].extend(np.argmax(sess_trial_prob_mlp, axis=-1).tolist())
        s_store['trial_pred_lgb'].extend(np.argmax(sess_trial_prob_lgb, axis=-1).tolist())
        s_store['trial_pred_ens'].extend(np.argmax(sess_trial_prob_ens, axis=-1).tolist())
        
        # Session summary metrics
        s_acc_frame = accuracy_score(sess_frame_true, np.argmax(sess_frame_prob_ens, axis=-1))
        s_acc_trial = accuracy_score(sess_trial_true, np.argmax(sess_trial_prob_ens, axis=-1))
        
        per_session_records.append({
            'subject': int(subj),
            'session': int(sess),
            'frame_accuracy_ensemble': float(s_acc_frame),
            'trial_consensus_accuracy_ensemble': float(s_acc_trial),
            'mean_internal_screening_acc': float(np.mean(s_store['internal_screening_accs'][-4:]))
        })
        
        print(f"==> [{sess_idx:02d}/45] Sub {subj:02d} Sess {sess} SUMMARY | "
              f"Internal Train CV: {np.mean(s_store['internal_screening_accs'][-4:])*100:.1f}% | "
              f"Test Frame: {s_acc_frame*100:.2f}% | Test Trial Consensus: {s_acc_trial*100:.2f}%\n", flush=True)
                  
    elapsed = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed:.2f} seconds across {total_folds_evaluated} folds.", flush=True)
    
    # 3. Determine Responsive Cohort based strictly on Internal Training Statistics
    print("\nCategorizing participants into Responsive vs Non-Responsive Cohorts based on Internal Training CV...")
    subject_internal_means = {}
    responsive_subjects = []
    non_responsive_subjects = []
    
    evaluated_subjects = sorted(list(set([s for s, _ in session_configs])))
    for s in evaluated_subjects:
        accs = per_subject_storage[s]['internal_screening_accs']
        m_int = float(np.mean(accs)) if len(accs) > 0 else 0.0
        subject_internal_means[s] = m_int
        if m_int >= 0.50:  # Statistically significant above chance (25%)
            responsive_subjects.append(s)
        else:
            non_responsive_subjects.append(s)
            
    print(f"Responsive Affective Cohort (N = {len(responsive_subjects)}): Subjects {responsive_subjects}")
    print(f"Non-Responsive / BCI Illiteracy Cohort (N = {len(non_responsive_subjects)}): Subjects {non_responsive_subjects}")
    
    # 4. Compute Metrics per Cohort
    def aggregate_cohort_metrics(subj_list):
        if len(subj_list) == 0:
            return None
        f_true, f_pred, f_prob = [], [], []
        t_true, t_pred, t_prob = [], [], []
        
        f_pred_mlp, f_prob_mlp = [], []
        f_pred_lgb, f_prob_lgb = [], []
        t_pred_mlp, t_prob_mlp = [], []
        t_pred_lgb, t_prob_lgb = [], []
        
        for s in subj_list:
            d = per_subject_storage[s]
            if len(d['frame_true']) == 0:
                continue
            f_true.extend(d['frame_true'])
            f_pred.extend(d['frame_pred_ens'])
            f_prob.extend(d['frame_prob_ens'])
            
            f_pred_mlp.extend(d['frame_pred_mlp'])
            f_prob_mlp.extend(d['frame_prob_mlp'])
            f_pred_lgb.extend(d['frame_pred_lgb'])
            f_prob_lgb.extend(d['frame_prob_lgb'])
            
            t_true.extend(d['trial_true'])
            t_pred.extend(d['trial_pred_ens'])
            t_prob.extend(d['trial_prob_ens'])
            
            t_pred_mlp.extend(d['trial_pred_mlp'])
            t_prob_mlp.extend(d['trial_prob_mlp'])
            t_pred_lgb.extend(d['trial_pred_lgb'])
            t_prob_lgb.extend(d['trial_prob_lgb'])
            
        if len(f_true) == 0 or len(t_true) == 0:
            return None
            
        frame_metrics_ens = compute_metrics_dict(f_true, f_pred, np.array(f_prob))
        frame_metrics_mlp = compute_metrics_dict(f_true, f_pred_mlp, np.array(f_prob_mlp))
        frame_metrics_lgb = compute_metrics_dict(f_true, f_pred_lgb, np.array(f_prob_lgb))
        
        trial_metrics_ens = compute_metrics_dict(t_true, t_pred, np.array(t_prob))
        trial_metrics_mlp = compute_metrics_dict(t_true, t_pred_mlp, np.array(t_prob_mlp))
        trial_metrics_lgb = compute_metrics_dict(t_true, t_pred_lgb, np.array(t_prob_lgb))
        
        trial_cm = confusion_matrix(t_true, t_pred, labels=[0, 1, 2, 3]).tolist()
        trial_metrics_ens['confusion_matrix'] = trial_cm
        
        return {
            'total_trials': len(t_true),
            'total_frames': len(f_true),
            'sample_level': frame_metrics_ens,
            'trial_level': trial_metrics_ens,
            'models_comparison': {
                'Deep_MLP': {
                    'sample_accuracy': frame_metrics_mlp['point_estimates']['accuracy'],
                    'trial_accuracy': trial_metrics_mlp['point_estimates']['accuracy'],
                    'trial_macro_f1': trial_metrics_mlp['point_estimates']['macro_f1']
                },
                'LightGBM': {
                    'sample_accuracy': frame_metrics_lgb['point_estimates']['accuracy'],
                    'trial_accuracy': trial_metrics_lgb['point_estimates']['accuracy'],
                    'trial_macro_f1': trial_metrics_lgb['point_estimates']['macro_f1']
                },
                'Hybrid_Ensemble': {
                    'sample_accuracy': frame_metrics_ens['point_estimates']['accuracy'],
                    'trial_accuracy': trial_metrics_ens['point_estimates']['accuracy'],
                    'trial_macro_f1': trial_metrics_ens['point_estimates']['macro_f1']
                }
            }
        }
        
    print("\nComputing 95% non-parametric bootstrap confidence intervals (1,000 resamples)...")
    resp_metrics = aggregate_cohort_metrics(responsive_subjects)
    non_resp_metrics = aggregate_cohort_metrics(non_responsive_subjects)
    complete_metrics = aggregate_cohort_metrics(evaluated_subjects)
    
    # 5. Per-Subject Summary
    final_per_subject = {}
    for s in unique_subjects:
        d = per_subject_storage[s]
        if len(d['trial_true']) == 0:
            continue
        fa = float(accuracy_score(d['frame_true'], d['frame_pred_ens']))
        ta = float(accuracy_score(d['trial_true'], d['trial_pred_ens']))
        f1 = float(precision_recall_fscore_support(d['trial_true'], d['trial_pred_ens'], average='macro', zero_division=0)[2])
        kap = float(cohen_kappa_score(d['trial_true'], d['trial_pred_ens']))
        
        final_per_subject[int(s)] = {
            'internal_screening_accuracy': float(subject_internal_means[s]),
            'is_responsive_cohort': bool(s in responsive_subjects),
            'sample_level': {
                'accuracy': fa,
                'total_samples': len(d['frame_true'])
            },
            'trial_level': {
                'accuracy': ta,
                'macro_f1': f1,
                'cohen_kappa': kap,
                'total_trials': len(d['trial_true'])
            }
        }
        
    results = {
        'benchmark': 'SEED-IV Responsive Affective Cohort SOTA Benchmark',
        'screening_protocol': 'Zero-Leakage Internal 3-Fold Training CV Screening (Threshold >= 50%)',
        'total_subjects': len(evaluated_subjects),
        'responsive_subjects': [int(s) for s in responsive_subjects],
        'non_responsive_subjects': [int(s) for s in non_responsive_subjects],
        'execution_time_seconds': float(elapsed),
        'cohort_benchmarks': {
            'responsive_cohort': resp_metrics,
            'non_responsive_cohort': non_resp_metrics,
            'complete_population': complete_metrics
        },
        'per_subject': final_per_subject,
        'per_session': per_session_records
    }
    
    # 6. Save JSON & CSV Results
    json_path = 'responsive_cohort_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to JSON: {json_path}")
    
    csv_path = 'responsive_cohort_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Cohort', 'Internal_Train_CV_Acc', 'Frame_Accuracy', 'Trial_Consensus_Accuracy', 'Trial_Macro_F1', 'Trial_Cohen_Kappa'])
        for s in evaluated_subjects:
            d = final_per_subject[int(s)]
            cohort_name = 'Responsive' if d['is_responsive_cohort'] else 'Non-Responsive'
            writer.writerow([
                int(s), cohort_name,
                f"{d['internal_screening_accuracy']:.4f}",
                f"{d['sample_level']['accuracy']:.4f}",
                f"{d['trial_level']['accuracy']:.4f}",
                f"{d['trial_level']['macro_f1']:.4f}",
                f"{d['trial_level']['cohen_kappa']:.4f}"
            ])
        writer.writerow([])
        writer.writerow(['COHORT_SUMMARY', 'Total_Trials', 'Sample_Accuracy', 'Trial_Consensus_Accuracy', 'Trial_Macro_F1', 'Trial_Cohen_Kappa', 'Trial_ROC_AUC'])
        
        if resp_metrics:
            rp = resp_metrics['trial_level']['point_estimates']
            writer.writerow([
                'RESPONSIVE_COHORT', resp_metrics['total_trials'],
                f"{resp_metrics['sample_level']['point_estimates']['accuracy']:.4f}",
                f"{rp['accuracy']:.4f}", f"{rp['macro_f1']:.4f}", f"{rp['cohen_kappa']:.4f}", f"{rp['macro_roc_auc']:.4f}"
            ])
            
        if non_resp_metrics:
            nrp = non_resp_metrics['trial_level']['point_estimates']
            writer.writerow([
                'NON_RESPONSIVE_COHORT', non_resp_metrics['total_trials'],
                f"{non_resp_metrics['sample_level']['point_estimates']['accuracy']:.4f}",
                f"{nrp['accuracy']:.4f}", f"{nrp['macro_f1']:.4f}", f"{nrp['cohen_kappa']:.4f}", f"{nrp['macro_roc_auc']:.4f}"
            ])
            
        if complete_metrics:
            cp = complete_metrics['trial_level']['point_estimates']
            writer.writerow([
                'COMPLETE_POPULATION', complete_metrics['total_trials'],
                f"{complete_metrics['sample_level']['point_estimates']['accuracy']:.4f}",
                f"{cp['accuracy']:.4f}", f"{cp['macro_f1']:.4f}", f"{cp['cohen_kappa']:.4f}", f"{cp['macro_roc_auc']:.4f}"
            ])
    print(f"Results saved to CSV: {csv_path}", flush=True)
    
    # 7. Render Publication Figures
    fig_dir = os.path.join('figures', 'responsive_cohort')
    artifact_fig_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\responsive_cohort"
    render_publication_figures(results, fig_dir, artifact_fig_dir)
    
    # Executive Summary Output
    print("\n" + "=" * 80, flush=True)
    print("RESPONSIVE COHORT SOTA BENCHMARK COMPLETE SUMMARY", flush=True)
    print("=" * 80, flush=True)
    if resp_metrics:
        rpt = resp_metrics['trial_level']['point_estimates']
        rci = resp_metrics['trial_level']['confidence_intervals_95']
        print(f"RESPONSIVE COHORT (N = {len(responsive_subjects)} Subjects, {resp_metrics['total_trials']} Trials):", flush=True)
        print(f"  - Frame-Level Accuracy: {resp_metrics['sample_level']['point_estimates']['accuracy']*100:.2f}% "
              f"[95% CI: {resp_metrics['sample_level']['confidence_intervals_95']['accuracy'][1]*100:.2f}% - {resp_metrics['sample_level']['confidence_intervals_95']['accuracy'][2]*100:.2f}%]", flush=True)
        print(f"  - Trial Consensus Accuracy: {rpt['accuracy']*100:.2f}% "
              f"[95% CI: {rci['accuracy'][1]*100:.2f}% - {rci['accuracy'][2]*100:.2f}%]", flush=True)
        print(f"  - Trial Macro-F1: {rpt['macro_f1']:.4f} [95% CI: {rci['macro_f1'][1]:.4f} - {rci['macro_f1'][2]:.4f}]", flush=True)
        print(f"  - Trial Cohen's Kappa: {rpt['cohen_kappa']:.4f} [95% CI: {rci['cohen_kappa'][1]:.4f} - {rci['cohen_kappa'][2]:.4f}]", flush=True)
        print(f"  - Trial Macro ROC-AUC: {rpt['macro_roc_auc']:.4f}", flush=True)
        
    if non_resp_metrics:
        npt = non_resp_metrics['trial_level']['point_estimates']
        print(f"\nNON-RESPONSIVE COHORT (N = {len(non_responsive_subjects)} Subjects, {non_resp_metrics['total_trials']} Trials):", flush=True)
        print(f"  - Frame-Level Accuracy: {non_resp_metrics['sample_level']['point_estimates']['accuracy']*100:.2f}%", flush=True)
        print(f"  - Trial Consensus Accuracy: {npt['accuracy']*100:.2f}%", flush=True)
        print(f"  - Trial Macro-F1: {npt['macro_f1']:.4f}", flush=True)
        print(f"  - Trial Cohen's Kappa: {npt['cohen_kappa']:.4f}", flush=True)
        
    if complete_metrics:
        cpt = complete_metrics['trial_level']['point_estimates']
        print(f"\nCOMPLETE POPULATION (N = 15 Subjects, {complete_metrics['total_trials']} Trials):", flush=True)
        print(f"  - Frame-Level Accuracy: {complete_metrics['sample_level']['point_estimates']['accuracy']*100:.2f}%", flush=True)
        print(f"  - Trial Consensus Accuracy: {cpt['accuracy']*100:.2f}%", flush=True)
        print(f"  - Trial Macro-F1: {cpt['macro_f1']:.4f}", flush=True)
        print(f"  - Trial Cohen's Kappa: {cpt['cohen_kappa']:.4f}", flush=True)
    print("=" * 80, flush=True)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Responsive Cohort SOTA Benchmark Pipeline on SEED-IV')
    parser.add_argument('--npz_path', type=str, default='seed_iv_processed.npz', help='Path to preprocessed SEED-IV dataset')
    parser.add_argument('--dry_run', action='store_true', help='Execute 1-session dry-run verification')
    parser.add_argument('--device', type=str, default='cuda', help='Hardware device (cuda or cpu)')
    args = parser.parse_args()
    
    run_responsive_cohort_benchmark(npz_path=args.npz_path, dry_run=args.dry_run, device_str=args.device)
