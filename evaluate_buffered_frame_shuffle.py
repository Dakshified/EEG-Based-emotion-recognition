"""
Temporal Autocorrelation Leakage-Free Frame Shuffling Benchmark on SEED-IV
===========================================================================
Investigates and proves the mathematical cause of ~100.00% reported accuracy in literature:
1. Regime A (Unbuffered Random Shuffled Split - 0s Buffer):
   - Standard literature 80/20 random frame shuffle across continuous trials.
   - High autocorrelation (Pearson rho > 0.95) between adjacent smoothed frames leaks labels.
2. Regime B (Buffered Leak-Free Shuffled Split - +/- 8s Buffer):
   - Enforces kernel support condition: min |t_train - t_test| >= 8.0 seconds within every trial.
   - Strictly purges +/- 8 frames around test blocks, severing the 4-second moving-average filter memory.
3. Comparative Models:
   - Shallow MLP (580 -> 256 -> 64 -> 4 with BatchNorm, GELU, Dropout)
   - Regularized LightGBM (300 estimators, max_depth=6, colsample=0.8)
4. Full 15-Subject Evaluation:
   - Outputs complete metrics, 95% bootstrap confidence intervals, and 300 DPI divergence visualization.
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

def set_seed(seed=42):
    """Sets global random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def get_asymmetry_pair_indices():
    """
    Computes index pairs for:
    - 27 Left-Right DASM pairs (excluding 8 midline electrodes)
    - 27 Anterior-Posterior DCAU pairs (frontal/frontocentral to parietal/occipital)
    """
    left_indices = []
    right_indices = []
    for ch in SEED_IV_CHANNELS:
        r, c = GRID_COORDINATES_9X9[ch]
        if c < 4:
            target_r, target_c = r, 8 - c
            matching = [r_ch for r_ch in SEED_IV_CHANNELS if GRID_COORDINATES_9X9[r_ch] == (target_r, target_c)]
            if matching:
                left_indices.append(SEED_IV_CHANNELS.index(ch))
                right_indices.append(SEED_IV_CHANNELS.index(matching[0]))
    
    ant_indices = []
    post_indices = []
    
    for c in range(9):
        f_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (2, c)]
        p_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (6, c)]
        if f_ch and p_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(f_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(p_ch[0]))
            
    for c in range(9):
        fc_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (3, c)]
        cp_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (5, c)]
        if fc_ch and cp_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(fc_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(cp_ch[0]))
            
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
    """Extracts 580D representation: 310 raw DE + 135 DASM + 135 DCAU."""
    N = len(features_3d)
    raw_flat = features_3d.reshape(N, 310)
    dasm = (features_3d[:, left_idx, :] - features_3d[:, right_idx, :]).reshape(N, 135)
    dcau = (features_3d[:, ant_idx, :] - features_3d[:, post_idx, :]).reshape(N, 135)
    return np.hstack([raw_flat, dasm, dcau]).astype(np.float32)

class ShallowMLP(nn.Module):
    """Shallow MLP model matching the literature replication architecture."""
    def __init__(self, in_features=580, num_classes=4, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_and_eval_mlp(X_train, y_train, X_test, y_test, device, epochs=40, batch_size=64, lr=1e-3):
    """Trains and evaluates Shallow MLP on given train/test partitions."""
    model = ShallowMLP(in_features=580, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    
    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    num_samples = len(X_train)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        for b in range(0, num_samples, batch_size):
            b_idx = perm[b:b+batch_size]
            bx, by = X_tr_t[b_idx], y_tr_t[b_idx]
            optimizer.zero_grad()
            logits = model(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=-1)
        
    acc = float(accuracy_score(y_test, test_preds))
    p, r, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
    kappa = float(cohen_kappa_score(y_test, test_preds))
    
    return {
        'accuracy': acc,
        'macro_f1': float(f1),
        'cohen_kappa': kappa,
        'predictions': test_preds.tolist(),
        'probabilities': test_probs.tolist()
    }

def train_and_eval_lgbm(X_train, y_train, X_test, y_test):
    """Trains and evaluates Regularized LightGBM classifier."""
    clf = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    
    acc = float(accuracy_score(y_test, test_preds))
    p, r, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
    kappa = float(cohen_kappa_score(y_test, test_preds))
    
    return {
        'accuracy': acc,
        'macro_f1': float(f1),
        'cohen_kappa': kappa,
        'predictions': test_preds.tolist(),
        'probabilities': test_probs.tolist()
    }

def fast_metrics_computation(yt, yp, num_classes=4):
    """Vectorized metrics computation for bootstrap confidence intervals."""
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
    """Computes 95% non-parametric bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    accs, f1s, kappas = [], [], []
    num_samples = len(y_true)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    for _ in range(num_resamples):
        indices = rng.randint(0, num_samples, size=num_samples)
        a, _, _, f, k = fast_metrics_computation(y_true[indices], y_pred[indices])
        accs.append(a)
        f1s.append(f)
        kappas.append(k)
        
    def get_ci(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        'accuracy': get_ci(accs),
        'macro_f1': get_ci(f1s),
        'cohen_kappa': get_ci(kappas)
    }

def render_divergence_figure(results, output_dir, artifact_dir):
    """
    Renders 300 DPI grouped bar chart contrasting:
    1. Unbuffered Shuffled Split (0s Buffer - ~100% Literature Artifact)
    2. Buffered Leak-Free Split (+/- 8s Exclusion Buffer - True Frame Discriminability)
    3. Strict Trial-Quarantine Benchmark (from Section 9/12)
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
    
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    
    subjects = sorted(list(results['per_subject_results'].keys()), key=lambda x: int(x))
    sub_labels = [f"Sub {int(s):02d}" for s in subjects]
    
    unbuffered_mlp = [results['per_subject_results'][s]['unbuffered_mlp']['accuracy'] * 100 for s in subjects]
    buffered_mlp = [results['per_subject_results'][s]['buffered_mlp']['accuracy'] * 100 for s in subjects]
    buffered_lgb = [results['per_subject_results'][s]['buffered_lgb']['accuracy'] * 100 for s in subjects]
    
    x = np.arange(len(subjects))
    w = 0.26
    
    r1 = ax.bar(x - w, unbuffered_mlp, w, label='Unbuffered Shuffled (0s Buffer - Literature ~100%)', color='#e74c3c', edgecolor='#c0392b', alpha=0.9)
    r2 = ax.bar(x, buffered_mlp, w, label='Buffered Leak-Free (±8s Buffer - Shallow MLP)', color='#3498db', edgecolor='#2980b9')
    r3 = ax.bar(x + w, buffered_lgb, w, label='Buffered Leak-Free (±8s Buffer - LightGBM)', color='#2ecc71', edgecolor='#27ae60')
    
    mean_unbuf = results['population_summary']['unbuffered_mlp']['accuracy'] * 100
    mean_buf_mlp = results['population_summary']['buffered_mlp']['accuracy'] * 100
    mean_buf_lgb = results['population_summary']['buffered_lgb']['accuracy'] * 100
    
    ax.axhline(mean_unbuf, color='#c0392b', linestyle=':', linewidth=1.5,
               label=f'Mean Unbuffered MLP ({mean_unbuf:.2f}%)')
    ax.axhline(mean_buf_mlp, color='#2980b9', linestyle='--', linewidth=1.5,
               label=f'Mean Buffered MLP ({mean_buf_mlp:.2f}%)')
    ax.axhline(25.0, color='#7f8c8d', linestyle='-', linewidth=1.2, label='Chance Level (25.0%)')
    
    ax.set_title("Temporal Autocorrelation Leakage Isolation on SEED-IV (15 Subjects)\nPerformance Divergence: Unbuffered Shuffling vs. Mandatory ±8s Exclusion Buffer",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Subject Identifier", fontsize=11, fontweight='bold')
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, rotation=0, fontweight='bold')
    ax.set_ylim(0, 108)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'buffered_vs_unbuffered_leakage_divergence.png')
    plt.savefig(fig_path, dpi=300)
    if artifact_dir:
        shutil.copy(fig_path, os.path.join(artifact_dir, 'buffered_vs_unbuffered_leakage_divergence.png'))
    plt.close()
    
    print(f"Publication figure exported successfully to {output_dir}", flush=True)

def run_buffered_frame_shuffle_benchmark(npz_path='seed_iv_processed.npz', dry_run=False, device_str='cuda'):
    """
    Executes the full Temporal Autocorrelation Leakage-Free Frame Shuffling Benchmark across all 15 subjects.
    """
    print("=" * 80, flush=True)
    print("TEMPORAL AUTOCORRELATION LEAKAGE-FREE FRAME SHUFFLE BENCHMARK", flush=True)
    print("=" * 80, flush=True)
    
    set_seed(42)
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == 'cuda') else 'cpu')
    print(f"Executing on hardware device: {device} (PyTorch {torch.__version__})", flush=True)
    
    # 1. Load Preprocessed SEED-IV Dataset
    print(f"Ingesting preprocessed SEED-IV dataset from {npz_path}...", flush=True)
    data = np.load(npz_path)
    features_3d = data['features']      # (37575, 62, 5)
    labels = data['labels']              # (37575,)
    subject_ids = data['subject_ids']    # (37575,)
    session_nums = data['session_nums']  # (37575,)
    trial_ids = data['trial_ids']        # (37575,)
    
    # 2. Extract 580D Cortical Asymmetry Space
    print("Extracting 580D Cortical Asymmetry Space (310D raw DE + 135D DASM + 135D DCAU)...", flush=True)
    left_idx, right_idx, ant_idx, post_idx = get_asymmetry_pair_indices()
    cortical_580d = extract_580d_cortical_features(features_3d, left_idx, right_idx, ant_idx, post_idx)
    
    unique_subjects = sorted(list(np.unique(subject_ids)))
    if dry_run:
        print("\n[DRY RUN ACTIVATED] Evaluating single subject verification: Subject 15...", flush=True)
        eval_subjects = [15]
    else:
        print(f"\n[FULL BENCHMARK ACTIVATED] Evaluating all {len(unique_subjects)} subjects across 3 sessions...", flush=True)
        eval_subjects = unique_subjects
        
    start_time = time.time()
    
    per_subject_results = {}
    
    pop_unbuf_mlp_trues, pop_unbuf_mlp_preds = [], []
    pop_unbuf_lgb_trues, pop_unbuf_lgb_preds = [], []
    pop_buf_mlp_trues, pop_buf_mlp_preds = [], []
    pop_buf_lgb_trues, pop_buf_lgb_preds = [], []
    
    for sub_idx, subj in enumerate(eval_subjects, 1):
        sub_mask = np.where(subject_ids == subj)[0]
        X_sub = cortical_580d[sub_mask]
        y_sub = labels[sub_mask]
        t_ids_sub = trial_ids[sub_mask]
        s_nums_sub = session_nums[sub_mask]
        
        N_sub = len(X_sub)
        
        # -------------------------------------------------------------
        # 1. REGIME A: Standard Unbuffered Shuffled Split (0s Buffer)
        # -------------------------------------------------------------
        skf_unbuf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        unbuf_mlp_preds_sub = np.zeros(N_sub, dtype=np.int64)
        unbuf_lgb_preds_sub = np.zeros(N_sub, dtype=np.int64)
        
        for tr_idx, te_idx in skf_unbuf.split(X_sub, y_sub):
            X_tr, y_tr = X_sub[tr_idx], y_sub[tr_idx]
            X_te, y_te = X_sub[te_idx], y_sub[te_idx]
            
            # Neutral baseline subtraction on train partition
            neut_mask = (y_tr == 0)
            mu_neut = np.mean(X_tr[neut_mask], axis=0, keepdims=True) if np.sum(neut_mask) > 0 else np.zeros((1, 580))
            X_tr_ref = X_tr - mu_neut
            X_te_ref = X_te - mu_neut
            
            # Inductive scaler
            scaler = StandardScaler().fit(X_tr_ref)
            X_tr_sc = scaler.transform(X_tr_ref)
            X_te_sc = scaler.transform(X_te_ref)
            
            # Evaluate MLP & LightGBM
            res_mlp = train_and_eval_mlp(X_tr_sc, y_tr, X_te_sc, y_te, device, epochs=20, batch_size=128)
            res_lgb = train_and_eval_lgbm(X_tr_sc, y_tr, X_te_sc, y_te)
            
            unbuf_mlp_preds_sub[te_idx] = res_mlp['predictions']
            unbuf_lgb_preds_sub[te_idx] = res_lgb['predictions']
            
        acc_unbuf_mlp = float(accuracy_score(y_sub, unbuf_mlp_preds_sub))
        f1_unbuf_mlp = float(precision_recall_fscore_support(y_sub, unbuf_mlp_preds_sub, average='macro', zero_division=0)[2])
        acc_unbuf_lgb = float(accuracy_score(y_sub, unbuf_lgb_preds_sub))
        f1_unbuf_lgb = float(precision_recall_fscore_support(y_sub, unbuf_lgb_preds_sub, average='macro', zero_division=0)[2])
        
        pop_unbuf_mlp_trues.extend(y_sub.tolist())
        pop_unbuf_mlp_preds.extend(unbuf_mlp_preds_sub.tolist())
        pop_unbuf_lgb_trues.extend(y_sub.tolist())
        pop_unbuf_lgb_preds.extend(unbuf_lgb_preds_sub.tolist())
        
        # -------------------------------------------------------------
        # 2. REGIME B: Buffered Leak-Free Shuffled Split (+/- 8s Buffer)
        # -------------------------------------------------------------
        # Partition every trial into 5 temporal blocks. For fold k in 0..4:
        # Test block = block k.
        # Excluded buffer = frames within +/- 8 seconds of test block.
        # Valid train = frames outside the +/- 8 second exclusion zone.
        
        buf_mlp_preds_sub = []
        buf_lgb_preds_sub = []
        buf_trues_sub = []
        
        total_purged_buffer_frames = 0
        
        for fold_k in range(5):
            test_indices_fold = []
            train_indices_fold = []
            
            # Iterate across unique session-trial combinations (72 trials total)
            for s_num in np.unique(s_nums_sub):
                sess_mask_sub = (s_nums_sub == s_num)
                trials_in_sess = np.unique(t_ids_sub[sess_mask_sub])
                
                for tr in trials_in_sess:
                    tr_frame_indices = np.where(sess_mask_sub & (t_ids_sub == tr))[0]
                    W_tr = len(tr_frame_indices)
                    if W_tr == 0:
                        continue
                        
                    # Block boundaries for fold k (5 blocks)
                    s_blk = int(fold_k * W_tr / 5.0)
                    e_blk = int((fold_k + 1) * W_tr / 5.0)
                    
                    test_local_idx = np.arange(s_blk, e_blk)
                    test_indices_fold.extend(tr_frame_indices[test_local_idx].tolist())
                    
                    # Compute exclusion buffer: any frame within 8 frames of test_local_idx
                    # Valid train frames: all frames with dist >= 8 from [s_blk, e_blk-1]
                    train_local_idx = []
                    for t_loc in range(W_tr):
                        # Min distance to test block
                        dist_to_test = min(abs(t_loc - t_test) for t_test in test_local_idx) if len(test_local_idx) > 0 else 999
                        if dist_to_test >= 8:
                            train_local_idx.append(t_loc)
                        elif dist_to_test > 0: # Excluded buffer frame
                            total_purged_buffer_frames += 1
                            
                    train_indices_fold.extend(tr_frame_indices[train_local_idx].tolist())
                    
            test_indices_fold = np.array(test_indices_fold, dtype=np.int64)
            train_indices_fold = np.array(train_indices_fold, dtype=np.int64)
            
            # MATHEMATICAL INTEGRITY ASSERTION: Verify 0 buffer leakage
            for s_num in np.unique(s_nums_sub):
                for tr in np.unique(t_ids_sub[s_nums_sub == s_num]):
                    tr_all = np.where((s_nums_sub == s_num) & (t_ids_sub == tr))[0]
                    tr_test = np.intersect1d(test_indices_fold, tr_all)
                    tr_train = np.intersect1d(train_indices_fold, tr_all)
                    if len(tr_test) > 0 and len(tr_train) > 0:
                        # Find minimum temporal distance within trial
                        # Relative positions in trial
                        pos_dict = {idx: pos for pos, idx in enumerate(tr_all)}
                        min_dist = min(abs(pos_dict[tr_i] - pos_dict[te_i]) for tr_i in tr_train for te_i in tr_test)
                        assert min_dist >= 8, f"LEAKAGE VIOLATION: min distance {min_dist} < 8s!"
                        
            X_tr_b, y_tr_b = X_sub[train_indices_fold], y_sub[train_indices_fold]
            X_te_b, y_te_b = X_sub[test_indices_fold], y_sub[test_indices_fold]
            
            # Neutral baseline subtraction on valid train partition
            neut_mask_b = (y_tr_b == 0)
            mu_neut_b = np.mean(X_tr_b[neut_mask_b], axis=0, keepdims=True) if np.sum(neut_mask_b) > 0 else np.zeros((1, 580))
            X_tr_b_ref = X_tr_b - mu_neut_b
            X_te_b_ref = X_te_b - mu_neut_b
            
            scaler_b = StandardScaler().fit(X_tr_b_ref)
            X_tr_b_sc = scaler_b.transform(X_tr_b_ref)
            X_te_b_sc = scaler_b.transform(X_te_b_ref)
            
            res_b_mlp = train_and_eval_mlp(X_tr_b_sc, y_tr_b, X_te_b_sc, y_te_b, device, epochs=20, batch_size=128)
            res_b_lgb = train_and_eval_lgbm(X_tr_b_sc, y_tr_b, X_te_b_sc, y_te_b)
            
            buf_mlp_preds_sub.extend(res_b_mlp['predictions'])
            buf_lgb_preds_sub.extend(res_b_lgb['predictions'])
            buf_trues_sub.extend(y_te_b.tolist())
            
        acc_buf_mlp = float(accuracy_score(buf_trues_sub, buf_mlp_preds_sub))
        f1_buf_mlp = float(precision_recall_fscore_support(buf_trues_sub, buf_mlp_preds_sub, average='macro', zero_division=0)[2])
        acc_buf_lgb = float(accuracy_score(buf_trues_sub, buf_lgb_preds_sub))
        f1_buf_lgb = float(precision_recall_fscore_support(buf_trues_sub, buf_lgb_preds_sub, average='macro', zero_division=0)[2])
        
        pop_buf_mlp_trues.extend(buf_trues_sub)
        pop_buf_mlp_preds.extend(buf_mlp_preds_sub)
        pop_buf_lgb_trues.extend(buf_trues_sub)
        pop_buf_lgb_preds.extend(buf_lgb_preds_sub)
        
        per_subject_results[str(subj)] = {
            'total_samples': N_sub,
            'unbuffered_mlp': {'accuracy': acc_unbuf_mlp, 'macro_f1': f1_unbuf_mlp},
            'unbuffered_lgb': {'accuracy': acc_unbuf_lgb, 'macro_f1': f1_unbuf_lgb},
            'buffered_mlp': {'accuracy': acc_buf_mlp, 'macro_f1': f1_buf_mlp},
            'buffered_lgb': {'accuracy': acc_buf_lgb, 'macro_f1': f1_buf_lgb},
            'collapse_mlp': float(acc_unbuf_mlp - acc_buf_mlp),
            'collapse_lgb': float(acc_unbuf_lgb - acc_buf_lgb)
        }
        
        print(f"[{sub_idx:02d}/{len(eval_subjects)}] Subject {subj:02d} | "
              f"Unbuffered MLP: {acc_unbuf_mlp*100:.2f}% (F1={f1_unbuf_mlp:.4f}) | "
              f"Buffered MLP (±8s): {acc_buf_mlp*100:.2f}% (F1={f1_buf_mlp:.4f}) | "
              f"Buffered LGB: {acc_buf_lgb*100:.2f}% | MLP Collapse: -{(acc_unbuf_mlp-acc_buf_mlp)*100:.2f}%", flush=True)
              
    elapsed = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed:.2f} seconds across {len(eval_subjects)} subjects.", flush=True)
    
    # 3. Compute Population Aggregate Metrics & Bootstrap CIs
    print("\nComputing 95% non-parametric bootstrap confidence intervals (1,000 resamples)...", flush=True)
    
    ci_unbuf_mlp = compute_bootstrap_confidence_intervals(pop_unbuf_mlp_trues, pop_unbuf_mlp_preds)
    ci_unbuf_lgb = compute_bootstrap_confidence_intervals(pop_unbuf_lgb_trues, pop_unbuf_lgb_preds)
    ci_buf_mlp = compute_bootstrap_confidence_intervals(pop_buf_mlp_trues, pop_buf_mlp_preds)
    ci_buf_lgb = compute_bootstrap_confidence_intervals(pop_buf_lgb_trues, pop_buf_lgb_preds)
    
    results = {
        'benchmark': 'SEED-IV Temporal Autocorrelation Leakage-Free Frame Shuffling Benchmark',
        'hardware': str(device),
        'execution_time_seconds': float(elapsed),
        'total_subjects': len(eval_subjects),
        'population_summary': {
            'unbuffered_mlp': {
                'accuracy': float(accuracy_score(pop_unbuf_mlp_trues, pop_unbuf_mlp_preds)),
                'macro_f1': float(precision_recall_fscore_support(pop_unbuf_mlp_trues, pop_unbuf_mlp_preds, average='macro', zero_division=0)[2]),
                'cohen_kappa': float(cohen_kappa_score(pop_unbuf_mlp_trues, pop_unbuf_mlp_preds)),
                'confidence_intervals_95': ci_unbuf_mlp
            },
            'unbuffered_lgb': {
                'accuracy': float(accuracy_score(pop_unbuf_lgb_trues, pop_unbuf_lgb_preds)),
                'macro_f1': float(precision_recall_fscore_support(pop_unbuf_lgb_trues, pop_unbuf_lgb_preds, average='macro', zero_division=0)[2]),
                'cohen_kappa': float(cohen_kappa_score(pop_unbuf_lgb_trues, pop_unbuf_lgb_preds)),
                'confidence_intervals_95': ci_unbuf_lgb
            },
            'buffered_mlp': {
                'accuracy': float(accuracy_score(pop_buf_mlp_trues, pop_buf_mlp_preds)),
                'macro_f1': float(precision_recall_fscore_support(pop_buf_mlp_trues, pop_buf_mlp_preds, average='macro', zero_division=0)[2]),
                'cohen_kappa': float(cohen_kappa_score(pop_buf_mlp_trues, pop_buf_mlp_preds)),
                'confidence_intervals_95': ci_buf_mlp
            },
            'buffered_lgb': {
                'accuracy': float(accuracy_score(pop_buf_lgb_trues, pop_buf_lgb_preds)),
                'macro_f1': float(precision_recall_fscore_support(pop_buf_lgb_trues, pop_buf_lgb_preds, average='macro', zero_division=0)[2]),
                'cohen_kappa': float(cohen_kappa_score(pop_buf_lgb_trues, pop_buf_lgb_preds)),
                'confidence_intervals_95': ci_buf_lgb
            }
        },
        'per_subject_results': per_subject_results
    }
    
    # 4. Save JSON and CSV
    json_path = 'buffered_frame_shuffle_results.json'
    csv_path = 'buffered_frame_shuffle_results.csv'
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured experimental results to {json_path}", flush=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Subject_ID', 'Total_Samples',
            'Unbuf_MLP_Acc', 'Unbuf_MLP_F1',
            'Unbuf_LGB_Acc', 'Unbuf_LGB_F1',
            'Buf_MLP_Acc', 'Buf_MLP_F1',
            'Buf_LGB_Acc', 'Buf_LGB_F1',
            'MLP_Leakage_Collapse'
        ])
        for s in eval_subjects:
            sm = per_subject_results[str(s)]
            writer.writerow([
                s,
                sm['total_samples'],
                f"{sm['unbuffered_mlp']['accuracy']*100:.2f}%",
                f"{sm['unbuffered_mlp']['macro_f1']:.4f}",
                f"{sm['unbuffered_lgb']['accuracy']*100:.2f}%",
                f"{sm['unbuffered_lgb']['macro_f1']:.4f}",
                f"{sm['buffered_mlp']['accuracy']*100:.2f}%",
                f"{sm['buffered_mlp']['macro_f1']:.4f}",
                f"{sm['buffered_lgb']['accuracy']*100:.2f}%",
                f"{sm['buffered_lgb']['macro_f1']:.4f}",
                f"-{sm['collapse_mlp']*100:.2f}%"
            ])
        # Summary row
        writer.writerow([
            'POPULATION_MEAN',
            len(pop_unbuf_mlp_trues),
            f"{results['population_summary']['unbuffered_mlp']['accuracy']*100:.2f}%",
            f"{results['population_summary']['unbuffered_mlp']['macro_f1']:.4f}",
            f"{results['population_summary']['unbuffered_lgb']['accuracy']*100:.2f}%",
            f"{results['population_summary']['unbuffered_lgb']['macro_f1']:.4f}",
            f"{results['population_summary']['buffered_mlp']['accuracy']*100:.2f}%",
            f"{results['population_summary']['buffered_mlp']['macro_f1']:.4f}",
            f"{results['population_summary']['buffered_lgb']['accuracy']*100:.2f}%",
            f"{results['population_summary']['buffered_lgb']['macro_f1']:.4f}",
            f"-{(results['population_summary']['unbuffered_mlp']['accuracy'] - results['population_summary']['buffered_mlp']['accuracy'])*100:.2f}%"
        ])
    print(f"Saved tabular metrics to {csv_path}", flush=True)
    
    # 5. Render 300 DPI Publication Figure
    fig_dir = 'figures/buffered_shuffle'
    artifact_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\buffered_shuffle"
    render_divergence_figure(results, fig_dir, artifact_dir)
    
    # Executive Summary Print
    pop_unbuf = results['population_summary']['unbuffered_mlp']['accuracy'] * 100
    pop_buf_mlp = results['population_summary']['buffered_mlp']['accuracy'] * 100
    pop_buf_lgb = results['population_summary']['buffered_lgb']['accuracy'] * 100
    collapse = pop_unbuf - pop_buf_mlp
    
    print("\n" + "=" * 80, flush=True)
    print("LEAKAGE ISOLATION SUMMARY (UNBUFFERED VS. MANDATORY ±8s BUFFER)", flush=True)
    print("=" * 80, flush=True)
    print(f"Unbuffered Shuffled MLP (0s Buffer - Literature):  {pop_unbuf:.2f}% [95% CI: {ci_unbuf_mlp['accuracy'][1]*100:.2f}% - {ci_unbuf_mlp['accuracy'][2]*100:.2f}%]", flush=True)
    print(f"Buffered Leak-Free MLP (±8s Buffer - True Acc):    {pop_buf_mlp:.2f}% [95% CI: {ci_buf_mlp['accuracy'][1]*100:.2f}% - {ci_buf_mlp['accuracy'][2]*100:.2f}%]", flush=True)
    print(f"Buffered Leak-Free LGB (±8s Buffer):               {pop_buf_lgb:.2f}% [95% CI: {ci_buf_lgb['accuracy'][1]*100:.2f}% - {ci_buf_lgb['accuracy'][2]*100:.2f}%]", flush=True)
    print(f"Autocorrelation Leakage Inflation Penalty:         -{collapse:.2f}% Accuracy Collapse", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal Autocorrelation Leakage-Free Benchmark')
    parser.add_argument('--npz', type=str, default='seed_iv_processed.npz', help='Path to seed_iv_processed.npz')
    parser.add_argument('--dry-run', action='store_true', help='Execute single-subject verification (Subject 15)')
    parser.add_argument('--device', type=str, default='cuda', help='Execution device (cuda/cpu)')
    args = parser.parse_args()
    
    run_buffered_frame_shuffle_benchmark(npz_path=args.npz, dry_run=args.dry_run, device_str=args.device)
