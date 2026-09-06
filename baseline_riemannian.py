import os
import time
import csv
import json
import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# =========================================================================
# GPU BATCHED RIEMANNIAN GEOMETRY OPERATIONS
# =========================================================================

def build_spd_matrices_gpu(features, device, eps=1e-4):
    """
    Builds (N, 62, 62) SPD covariance matrices from (N, 62, 5) DE features on GPU.
    P_i = (X_i @ X_i.T) / 5.0 + eps * I_62
    """
    X_torch = torch.tensor(features, dtype=torch.float32, device=device)
    eye = torch.eye(62, dtype=torch.float32, device=device).unsqueeze(0)
    P = (X_torch @ X_torch.transpose(-1, -2)) / 5.0 + eps * eye
    return P

def project_tangent_space_gpu(P_train_gpu, P_test_gpu, n_channels=62):
    """
    Computes Log-Euclidean geometric mean on P_train_gpu and projects both
    training and test SPD matrices into Riemannian tangent space in batched form on CUDA.
    Zero data leakage: Cref is computed strictly from training matrices.
    """
    # 1. Compute matrix logarithms for all training SPD matrices using batched eigh
    evals, evecs = torch.linalg.eigh(P_train_gpu)
    log_evals = torch.log(torch.clamp(evals, min=1e-12))
    log_P_train = evecs @ torch.diag_embed(log_evals) @ evecs.transpose(-1, -2)
    
    # 2. Log-Euclidean mean: matrix exponential of mean of logarithms
    mean_log_P = torch.mean(log_P_train, dim=0)
    m_evals, m_evecs = torch.linalg.eigh(mean_log_P)
    Cref_invsqrt = m_evecs @ torch.diag_embed(torch.exp(-0.5 * m_evals)) @ m_evecs.T
    
    # 3. Project matrices into tangent space: S = Cref^{-1/2} @ P @ Cref^{-1/2}
    def project_batch(P_batch):
        S = Cref_invsqrt.unsqueeze(0) @ P_batch @ Cref_invsqrt.unsqueeze(0)
        s_evals, s_evecs = torch.linalg.eigh(S)
        s_log_evals = torch.log(torch.clamp(s_evals, min=1e-12))
        log_S = s_evecs @ torch.diag_embed(s_log_evals) @ s_evecs.transpose(-1, -2)
        
        # Upper-triangular vectorization with sqrt(2) weighting on off-diagonals
        triu_idx = torch.triu_indices(n_channels, n_channels, device=P_batch.device)
        row_idx, col_idx = triu_idx[0], triu_idx[1]
        weights = torch.where(row_idx == col_idx, 1.0, float(np.sqrt(2.0)))
        feats = log_S[:, row_idx, col_idx] * weights.unsqueeze(0)
        return feats
        
    feats_train = project_batch(P_train_gpu)
    feats_test = project_batch(P_test_gpu)
    
    return feats_train.cpu().numpy(), feats_test.cpu().numpy()

# =========================================================================
# EVALUATION & METRICS HELPERS
# =========================================================================

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000):
    """Computes 95% confidence intervals using bootstrap resampling."""
    accs, precs, recs, f1s, aucs, kappas = [], [], [], [], [], []
    num_samples = len(y_true)
    
    for _ in range(num_resamples):
        indices = np.random.choice(num_samples, num_samples, replace=True)
        y_t_res = y_true[indices]
        y_p_res = y_pred[indices]
        y_prob_res = y_prob[indices]
        
        accs.append(np.mean(y_t_res == y_p_res))
        
        p, r, f, _ = precision_recall_fscore_support(y_t_res, y_p_res, average='macro', zero_division=0)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        
        try:
            auc_val = roc_auc_score(y_t_res, y_prob_res, average='macro', multi_class='ovr')
            aucs.append(auc_val)
        except Exception:
            aucs.append(0.5)
            
        kappas.append(cohen_kappa_score(y_t_res, y_p_res))
            
    ci_acc = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    ci_prec = (float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5)))
    ci_rec = (float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5)))
    ci_f1 = (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))
    ci_auc = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    ci_kappa = (float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5)))
    
    return ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa

def save_evaluation_plots(y_true, y_pred, y_prob, prefix, figures_dir):
    """Generates and saves Confusion Matrix, ROC, PR, and Calibration diagrams (300 DPI)."""
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Confusion Matrix (Raw + Row-Normalized)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Normalized Ratio')
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})", ha='center', va='center', fontsize=9,
                     color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.xticks(range(4), CLASS_NAMES)
    plt.yticks(range(4), CLASS_NAMES)
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    plt.title(f'Confusion Matrix - {prefix}', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 2. ROC Curves per Class
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[c]} (AUC={roc_auc:.3f})', color=EMOTION_COLORS[c], lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'ROC Curves - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_roc.png"), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, c])
        plt.plot(recall, precision, label=f'{CLASS_NAMES[c]}', color=EMOTION_COLORS[c], lw=2)
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title(f'Precision-Recall Curves - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower left', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_pr.png"), dpi=300)
    plt.close()
    
    # 4. Calibration Diagram
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob[:, c], n_bins=5)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{CLASS_NAMES[c]}', color=EMOTION_COLORS[c], lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Mean Predicted Probability', fontweight='bold')
    plt.ylabel('Fraction of Positives', fontweight='bold')
    plt.title(f'Calibration Diagram - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_calibration.png"), dpi=300)
    plt.close()

# =========================================================================
# EXPERIMENT RUNNERS (SUBJECT-DEPENDENT & CROSS-SUBJECT)
# =========================================================================

C_CANDIDATES = [0.01, 0.1, 1.0]

def run_subject_dependent(P_all_gpu, labels, subject_ids, session_nums, trial_ids, figures_dir):
    """Subject-dependent protocol: pooled data, 5-fold nested CV grouped by trial."""
    print("=" * 80, flush=True)
    print("RUNNING RIEMANNIAN + LOGISTIC REGRESSION: Subject-Dependent Protocol", flush=True)
    print("=" * 80, flush=True)
    
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys = np.unique(trial_keys)
    num_trials = len(unique_trial_keys)
    num_samples = len(labels)
    
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    fold_metrics_list = []
    
    for fold, (train_trial_idx, test_trial_idx) in enumerate(kf_outer.split(np.arange(num_trials))):
        t_fold_start = time.time()
        print(f"\n--- Outer Fold {fold+1}/5 ---", flush=True)
        
        train_trials_set = set(unique_trial_keys[train_trial_idx])
        test_trials_set = set(unique_trial_keys[test_trial_idx])
        
        train_indices = np.array([idx for idx in range(num_samples) if trial_keys[idx] in train_trials_set])
        test_indices = np.array([idx for idx in range(num_samples) if trial_keys[idx] in test_trials_set])
        
        print(f"  Outer Fold {fold+1}: {len(train_indices)} train samples, {len(test_indices)} test samples.", flush=True)
        
        # 1. Project train and test SPD matrices into tangent space using train-local mean
        t_proj = time.time()
        P_train_gpu = P_all_gpu[train_indices]
        P_test_gpu = P_all_gpu[test_indices]
        
        X_train_ts, X_test_ts = project_tangent_space_gpu(P_train_gpu, P_test_gpu, n_channels=62)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  GPU Tangent Space projection completed in {time.time()-t_proj:.2f}s (Features: {X_train_ts.shape[1]}).", flush=True)
        
        # 2. Fit StandardScaler on outer training fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_ts)
        X_test_scaled = scaler.transform(X_test_ts)
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        
        # 3. Inner CV for C tuning across outer training trials
        train_trial_keys_sub = unique_trial_keys[train_trial_idx]
        kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
        
        best_score = -1.0
        best_c = None
        
        print("  [INNER CV] Tuning C parameter across [0.01, 0.1, 1.0]...", flush=True)
        for c_val in C_CANDIDATES:
            inner_accs = []
            for in_tr_t_idx, in_val_t_idx in kf_inner.split(np.arange(len(train_trial_keys_sub))):
                in_tr_set = set(train_trial_keys_sub[in_tr_t_idx])
                in_val_set = set(train_trial_keys_sub[in_val_t_idx])
                
                in_tr_idx = np.array([i for i in range(len(train_indices)) if trial_keys[train_indices[i]] in in_tr_set])
                in_val_idx = np.array([i for i in range(len(train_indices)) if trial_keys[train_indices[i]] in in_val_set])
                
                in_scaler = StandardScaler()
                X_in_tr = in_scaler.fit_transform(X_train_ts[in_tr_idx])
                X_in_val = in_scaler.transform(X_train_ts[in_val_idx])
                y_in_tr = y_train[in_tr_idx]
                y_in_val = y_train[in_val_idx]
                
                clf = LogisticRegression(C=c_val, max_iter=500, random_state=42, solver='lbfgs')
                clf.fit(X_in_tr, y_in_tr)
                preds_val = clf.predict(X_in_val)
                inner_accs.append(np.mean(preds_val == y_in_val))
                
            mean_acc = np.mean(inner_accs)
            print(f"    C={c_val:<4} => Inner Val Acc: {mean_acc:.4f}", flush=True)
            if mean_acc > best_score:
                best_score = mean_acc
                best_c = c_val
                
        print(f"  [BEST HYPERPARAMETER] Selected C={best_c} (Inner Acc: {best_score:.4f})", flush=True)
        
        # 4. Fit outer classifier with best C
        model = LogisticRegression(C=best_c, max_iter=1000, random_state=42, solver='lbfgs')
        model.fit(X_train_scaled, y_train)
        
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
        
        fold_acc = np.mean(preds == y_test)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        try:
            fold_auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
        except Exception:
            fold_auc = 0.5
        fold_kappa = cohen_kappa_score(y_test, preds)
        
        t_fold = time.time() - t_fold_start
        print(f"  [OUTER FOLD COMPLETE] Fold {fold+1}/5 => Acc: {fold_acc:.4f} | Macro-F1: {fold_f1:.4f} | Macro-AUC: {fold_auc:.4f} | Kappa: {fold_kappa:.4f} ({t_fold:.2f}s)", flush=True)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)
        fold_metrics_list.append({
            'accuracy': float(fold_acc), 
            'precision': float(fold_prec), 
            'recall': float(fold_rec), 
            'f1': float(fold_f1), 
            'auc': float(fold_auc),
            'kappa': float(fold_kappa)
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    # Overall metrics
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    # Save evaluation diagrams
    prefix = "riemannian_subject_dependent"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Save out-of-fold sample predictions for verified downstream statistical testing
    np.savez_compressed(
        "oof_preds_riemannian_subject_dependent.npz",
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all
    )
    
    results = {
        "protocol": "subject-dependent",
        "accuracy": overall_acc,
        "accuracy_ci": ci_acc,
        "precision": float(overall_prec),
        "precision_ci": ci_prec,
        "recall": float(overall_rec),
        "recall_ci": ci_rec,
        "f1": float(overall_f1),
        "f1_ci": ci_f1,
        "auc": overall_auc,
        "auc_ci": ci_auc,
        "kappa": overall_kappa,
        "kappa_ci": ci_kappa,
        "fold_metrics": fold_metrics_list
    }
    
    print(f"\n>>> [SUBJECT-DEPENDENT COMPLETE] Acc: {overall_acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]) | Macro-F1: {overall_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]) | Kappa: {overall_kappa:.4f}\n", flush=True)
    return results

def run_cross_subject(P_all_gpu, labels, subject_ids, figures_dir):
    """Cross-subject protocol: 5-fold Leave-3-Subjects-Out nested CV."""
    print("=" * 80, flush=True)
    print("RUNNING RIEMANNIAN + LOGISTIC REGRESSION: Cross-Subject Protocol (Leave-3-Subjects-Out)", flush=True)
    print("=" * 80, flush=True)
    
    # 15 subjects partitioned into 5 folds of 3 subjects each
    subject_folds = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    fold_metrics_list = []
    
    for fold, test_subs in enumerate(subject_folds):
        t_fold_start = time.time()
        train_subs = [s for s in range(1, 16) if s not in test_subs]
        print(f"\n--- Outer Fold {fold+1}/5 (Test Subjects: {test_subs}) ---", flush=True)
        
        # Verify strict isolation
        assert len(set(train_subs).intersection(set(test_subs))) == 0, "Data leakage detected across subject boundaries!"
        
        train_indices = np.where(np.isin(subject_ids, train_subs))[0]
        test_indices = np.where(np.isin(subject_ids, test_subs))[0]
        
        print(f"  Outer Fold {fold+1}: {len(train_indices)} train samples (12 subjects), {len(test_indices)} test samples (3 subjects).", flush=True)
        
        # 1. Project train and test SPD matrices into tangent space using train-local mean
        t_proj = time.time()
        P_train_gpu = P_all_gpu[train_indices]
        P_test_gpu = P_all_gpu[test_indices]
        
        X_train_ts, X_test_ts = project_tangent_space_gpu(P_train_gpu, P_test_gpu, n_channels=62)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  GPU Tangent Space projection completed in {time.time()-t_proj:.2f}s (Features: {X_train_ts.shape[1]}).", flush=True)
        
        # 2. Fit StandardScaler on outer training fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_ts)
        X_test_scaled = scaler.transform(X_test_ts)
        y_train = labels[train_indices]
        y_test = labels[test_indices]
        
        # 3. Inner CV across the 12 train subjects (3 inner folds of 4 subjects each)
        inner_subject_splits = [
            train_subs[0:4],
            train_subs[4:8],
            train_subs[8:12]
        ]
        
        best_score = -1.0
        best_c = None
        
        print("  [INNER CV] Tuning C parameter across [0.01, 0.1, 1.0]...", flush=True)
        for c_val in C_CANDIDATES:
            inner_accs = []
            for in_val_subs in inner_subject_splits:
                in_tr_subs = [s for s in train_subs if s not in in_val_subs]
                
                in_tr_mask = np.isin(subject_ids[train_indices], in_tr_subs)
                in_val_mask = np.isin(subject_ids[train_indices], in_val_subs)
                
                in_scaler = StandardScaler()
                X_in_tr = in_scaler.fit_transform(X_train_ts[in_tr_mask])
                X_in_val = in_scaler.transform(X_train_ts[in_val_mask])
                y_in_tr = y_train[in_tr_mask]
                y_in_val = y_train[in_val_mask]
                
                clf = LogisticRegression(C=c_val, max_iter=500, random_state=42, solver='lbfgs')
                clf.fit(X_in_tr, y_in_tr)
                preds_val = clf.predict(X_in_val)
                inner_accs.append(np.mean(preds_val == y_in_val))
                
            mean_acc = np.mean(inner_accs)
            print(f"    C={c_val:<4} => Inner Val Acc: {mean_acc:.4f}", flush=True)
            if mean_acc > best_score:
                best_score = mean_acc
                best_c = c_val
                
        print(f"  [BEST HYPERPARAMETER] Selected C={best_c} (Inner Acc: {best_score:.4f})", flush=True)
        
        # 4. Fit outer classifier with best C
        model = LogisticRegression(C=best_c, max_iter=1000, random_state=42, solver='lbfgs')
        model.fit(X_train_scaled, y_train)
        
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
        
        fold_acc = np.mean(preds == y_test)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        try:
            fold_auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
        except Exception:
            fold_auc = 0.5
        fold_kappa = cohen_kappa_score(y_test, preds)
        
        t_fold = time.time() - t_fold_start
        print(f"  [OUTER FOLD COMPLETE] Fold {fold+1}/5 => Acc: {fold_acc:.4f} | Macro-F1: {fold_f1:.4f} | Macro-AUC: {fold_auc:.4f} | Kappa: {fold_kappa:.4f} ({t_fold:.2f}s)", flush=True)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)
        fold_metrics_list.append({
            'accuracy': float(fold_acc), 
            'precision': float(fold_prec), 
            'recall': float(fold_rec), 
            'f1': float(fold_f1), 
            'auc': float(fold_auc),
            'kappa': float(fold_kappa)
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    # Overall metrics
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    # Save evaluation diagrams
    prefix = "riemannian_cross_subject"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Save out-of-fold sample predictions for verified downstream statistical testing
    np.savez_compressed(
        "oof_preds_riemannian_cross_subject.npz",
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all
    )
    
    results = {
        "protocol": "cross-subject",
        "accuracy": overall_acc,
        "accuracy_ci": ci_acc,
        "precision": float(overall_prec),
        "precision_ci": ci_prec,
        "recall": float(overall_rec),
        "recall_ci": ci_rec,
        "f1": float(overall_f1),
        "f1_ci": ci_f1,
        "auc": overall_auc,
        "auc_ci": ci_auc,
        "kappa": overall_kappa,
        "kappa_ci": ci_kappa,
        "fold_metrics": fold_metrics_list
    }
    
    print(f"\n>>> [CROSS-SUBJECT COMPLETE] Acc: {overall_acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]) | Macro-F1: {overall_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]) | Kappa: {overall_kappa:.4f}\n", flush=True)
    return results

# =========================================================================
# MAIN PIPELINE ENTRYPOINT
# =========================================================================

def main():
    t_start = time.time()
    figures_dir = os.path.join("figures", "baselines")
    os.makedirs(figures_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("RIEMANNIAN TANGENT SPACE + LOGISTIC REGRESSION BASELINE (GPU)", flush=True)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)
    
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at '{dataset_path}'")
        
    print(f"Loading SEED-IV dataset from '{dataset_path}'...", flush=True)
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    num_samples = len(labels)
    
    print(f"Dataset loaded: {num_samples} samples across 15 subjects.", flush=True)
    
    # 1. Build all SPD matrices on GPU
    t_spd = time.time()
    print("Generating SPD covariance matrices from DE features on GPU...", flush=True)
    P_all_gpu = build_spd_matrices_gpu(features, device=device, eps=1e-4)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"Generated {num_samples} SPD matrices of shape {P_all_gpu.shape} in {time.time()-t_spd:.2f}s", flush=True)
    
    # 2. Run Subject-Dependent Protocol
    res_dep = run_subject_dependent(P_all_gpu, labels, subject_ids, session_nums, trial_ids, figures_dir)
    
    # 3. Run Cross-Subject Protocol
    res_cross = run_cross_subject(P_all_gpu, labels, subject_ids, figures_dir)
    
    # 4. Save structured results (JSON & CSV)
    all_results = {
        "subject_dependent": res_dep,
        "cross_subject": res_cross
    }
    
    json_path = "riemannian_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved structured JSON results to '{json_path}'", flush=True)
    
    csv_path = "riemannian_baseline_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Protocol", "Accuracy", "Accuracy_95CI_Lower", "Accuracy_95CI_Upper", 
            "Macro_Precision", "Macro_Precision_95CI_Lower", "Macro_Precision_95CI_Upper",
            "Macro_Recall", "Macro_Recall_95CI_Lower", "Macro_Recall_95CI_Upper",
            "Macro_F1", "Macro_F1_95CI_Lower", "Macro_F1_95CI_Upper",
            "Macro_AUC", "Macro_AUC_95CI_Lower", "Macro_AUC_95CI_Upper",
            "Cohen_Kappa", "Cohen_Kappa_95CI_Lower", "Cohen_Kappa_95CI_Upper"
        ])
        for p_key, r in [("subject_dependent", res_dep), ("cross_subject", res_cross)]:
            writer.writerow([
                r["protocol"],
                f"{r['accuracy']:.4f}", f"{r['accuracy_ci'][0]:.4f}", f"{r['accuracy_ci'][1]:.4f}",
                f"{r['precision']:.4f}", f"{r['precision_ci'][0]:.4f}", f"{r['precision_ci'][1]:.4f}",
                f"{r['recall']:.4f}", f"{r['recall_ci'][0]:.4f}", f"{r['recall_ci'][1]:.4f}",
                f"{r['f1']:.4f}", f"{r['f1_ci'][0]:.4f}", f"{r['f1_ci'][1]:.4f}",
                f"{r['auc']:.4f}", f"{r['auc_ci'][0]:.4f}", f"{r['auc_ci'][1]:.4f}",
                f"{r['kappa']:.4f}", f"{r['kappa_ci'][0]:.4f}", f"{r['kappa_ci'][1]:.4f}"
            ])
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)
    
    # 5. Print Final Summary Table
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY: RIEMANNIAN TANGENT SPACE + LOGISTIC REGRESSION RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Protocol':<20} | {'Accuracy (95% CI)':<25} | {'Macro-F1 (95% CI)':<25} | {'Cohen Kappa (95% CI)':<25}", flush=True)
    print("-" * 105, flush=True)
    for p_key, r in [("subject_dependent", res_dep), ("cross_subject", res_cross)]:
        acc_str = f"{r['accuracy']:.4f} [{r['accuracy_ci'][0]:.4f}, {r['accuracy_ci'][1]:.4f}]"
        f1_str = f"{r['f1']:.4f} [{r['f1_ci'][0]:.4f}, {r['f1_ci'][1]:.4f}]"
        kappa_str = f"{r['kappa']:.4f} [{r['kappa_ci'][0]:.4f}, {r['kappa_ci'][1]:.4f}]"
        print(f"{r['protocol']:<20} | {acc_str:<25} | {f1_str:<25} | {kappa_str:<25}", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Riemannian baseline pipeline runtime: {(time.time() - t_start)/60:.2f} minutes.", flush=True)

if __name__ == "__main__":
    main()
