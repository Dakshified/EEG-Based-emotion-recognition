import os
import time
import csv
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# =========================================================================
# STANDARDIZATION & METRICS HELPERS
# =========================================================================

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0 # Prevent division by zero
        
    def transform(self, x):
        return (x - self.mean) / self.std
        
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000):
    """Computes 95% confidence intervals using bootstrap resampling."""
    accs, precs, recs, f1s, aucs = [], [], [], [], []
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
            
    ci_acc = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    ci_prec = (float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5)))
    ci_rec = (float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5)))
    ci_f1 = (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))
    ci_auc = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    
    return ci_acc, ci_prec, ci_rec, ci_f1, ci_auc

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

PARAM_GRID = [
    {'learning_rate': 0.05, 'n_estimators': 100, 'num_leaves': 31},
    {'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 31},
    {'learning_rate': 0.10, 'n_estimators': 100, 'num_leaves': 31},
    {'learning_rate': 0.10, 'n_estimators': 200, 'num_leaves': 31},
]

def run_subject_dependent(X, y, subject_ids, session_nums, trial_ids, figures_dir):
    """Subject-dependent protocol: pooled data, 5-fold nested CV grouped by trial."""
    print("=" * 80, flush=True)
    print("RUNNING LIGHTGBM: Subject-Dependent Protocol (Pooled, 5-Fold Nested CV)", flush=True)
    print("=" * 80, flush=True)
    
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys = np.unique(trial_keys)
    num_trials = len(unique_trial_keys)
    
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
        
        train_indices = np.array([idx for idx in range(len(y)) if trial_keys[idx] in train_trials_set])
        test_indices = np.array([idx for idx in range(len(y)) if trial_keys[idx] in test_trials_set])
        
        # Scaling fit on training fold only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_indices])
        y_train = y[train_indices]
        X_test = scaler.transform(X[test_indices])
        y_test = y[test_indices]
        
        # Inner CV for hyperparameter tuning
        train_trial_keys_sub = unique_trial_keys[train_trial_idx]
        kf_inner = KFold(n_splits=3, shuffle=True, random_state=42)
        
        best_score = -1.0
        best_params = None
        
        print("  [INNER CV] Tuning hyperparameters across 4 candidate combinations...", flush=True)
        for param_idx, params in enumerate(PARAM_GRID):
            inner_accs = []
            for in_tr_t_idx, in_val_t_idx in kf_inner.split(np.arange(len(train_trial_keys_sub))):
                in_tr_set = set(train_trial_keys_sub[in_tr_t_idx])
                in_val_set = set(train_trial_keys_sub[in_val_t_idx])
                
                in_tr_idx = np.array([i for i in range(len(train_indices)) if trial_keys[train_indices[i]] in in_tr_set])
                in_val_idx = np.array([i for i in range(len(train_indices)) if trial_keys[train_indices[i]] in in_val_set])
                
                # Inner scaling
                in_scaler = StandardScaler()
                X_in_tr = in_scaler.fit_transform(X_train[in_tr_idx])
                y_in_tr = y_train[in_tr_idx]
                X_in_val = in_scaler.transform(X_train[in_val_idx])
                y_in_val = y_train[in_val_idx]
                
                clf = lgb.LGBMClassifier(
                    **params,
                    objective='multiclass',
                    num_class=4,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                clf.fit(X_in_tr, y_in_tr)
                preds_val = clf.predict(X_in_val)
                inner_accs.append(np.mean(preds_val == y_in_val))
                
            mean_acc = np.mean(inner_accs)
            print(f"    Grid {param_idx+1}/4: {params} => Inner Val Acc: {mean_acc:.4f}", flush=True)
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = params
                
        print(f"  [BEST PARAMS] Selected: {best_params} (Inner Acc: {best_score:.4f})", flush=True)
        
        # Train outer model with best params on full outer training split
        model = lgb.LGBMClassifier(
            **best_params,
            objective='multiclass',
            num_class=4,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Outer test evaluation
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        fold_acc = np.mean(preds == y_test)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        try:
            fold_auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
        except Exception:
            fold_auc = 0.5
            
        t_fold = time.time() - t_fold_start
        print(f"  [OUTER FOLD COMPLETE] Fold {fold+1}/5 => Acc: {fold_acc:.4f} | Macro-F1: {fold_f1:.4f} | Macro-AUC: {fold_auc:.4f} ({t_fold:.2f}s)", flush=True)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)
        fold_metrics_list.append({'accuracy': float(fold_acc), 'precision': float(fold_prec), 'recall': float(fold_rec), 'f1': float(fold_f1), 'auc': float(fold_auc)})
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    # Overall metrics
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    # Save evaluation diagrams
    prefix = "lightgbm_subject_dependent"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Save out-of-fold sample predictions for verified downstream statistical testing
    np.savez_compressed(
        "oof_preds_lightgbm_subject_dependent.npz",
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
        "fold_metrics": fold_metrics_list
    }
    
    print(f"\n>>> [SUBJECT-DEPENDENT COMPLETE] Acc: {overall_acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]) | Macro-F1: {overall_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])\n", flush=True)
    return results

def run_cross_subject(X, y, subject_ids, figures_dir):
    """Cross-subject protocol: 5-fold Leave-3-Subjects-Out nested CV."""
    print("=" * 80, flush=True)
    print("RUNNING LIGHTGBM: Cross-Subject Protocol (Leave-3-Subjects-Out, 5 Folds)", flush=True)
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
        
        # Outer standardization fit on train subjects only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_indices])
        y_train = y[train_indices]
        X_test = scaler.transform(X[test_indices])
        y_test = y[test_indices]
        
        # Inner CV across the 12 train subjects (3 inner folds of 4 subjects each)
        inner_subject_splits = [
            train_subs[0:4],
            train_subs[4:8],
            train_subs[8:12]
        ]
        
        best_score = -1.0
        best_params = None
        
        print("  [INNER CV] Tuning hyperparameters across 4 candidate combinations...", flush=True)
        for param_idx, params in enumerate(PARAM_GRID):
            inner_accs = []
            for in_val_subs in inner_subject_splits:
                in_tr_subs = [s for s in train_subs if s not in in_val_subs]
                
                in_tr_mask = np.isin(subject_ids[train_indices], in_tr_subs)
                in_val_mask = np.isin(subject_ids[train_indices], in_val_subs)
                
                in_scaler = StandardScaler()
                X_in_tr = in_scaler.fit_transform(X_train[in_tr_mask])
                y_in_tr = y_train[in_tr_mask]
                X_in_val = in_scaler.transform(X_train[in_val_mask])
                y_in_val = y_train[in_val_mask]
                
                clf = lgb.LGBMClassifier(
                    **params,
                    objective='multiclass',
                    num_class=4,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                clf.fit(X_in_tr, y_in_tr)
                preds_val = clf.predict(X_in_val)
                inner_accs.append(np.mean(preds_val == y_in_val))
                
            mean_acc = np.mean(inner_accs)
            print(f"    Grid {param_idx+1}/4: {params} => Inner Val Acc: {mean_acc:.4f}", flush=True)
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = params
                
        print(f"  [BEST PARAMS] Selected: {best_params} (Inner Acc: {best_score:.4f})", flush=True)
        
        # Train outer model with best params on full 12 train subjects
        model = lgb.LGBMClassifier(
            **best_params,
            objective='multiclass',
            num_class=4,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Outer test evaluation on 3 unseen test subjects
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        
        fold_acc = np.mean(preds == y_test)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        try:
            fold_auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
        except Exception:
            fold_auc = 0.5
            
        t_fold = time.time() - t_fold_start
        print(f"  [OUTER FOLD COMPLETE] Fold {fold+1}/5 => Acc: {fold_acc:.4f} | Macro-F1: {fold_f1:.4f} | Macro-AUC: {fold_auc:.4f} ({t_fold:.2f}s)", flush=True)
        
        y_true_all.extend(y_test)
        y_pred_all.extend(preds)
        y_prob_all.extend(probs)
        fold_metrics_list.append({'accuracy': float(fold_acc), 'precision': float(fold_prec), 'recall': float(fold_rec), 'f1': float(fold_f1), 'auc': float(fold_auc)})
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    # Overall metrics
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    # Save evaluation diagrams
    prefix = "lightgbm_cross_subject"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Save out-of-fold sample predictions for verified downstream statistical testing
    np.savez_compressed(
        "oof_preds_lightgbm_cross_subject.npz",
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
        "fold_metrics": fold_metrics_list
    }
    
    print(f"\n>>> [CROSS-SUBJECT COMPLETE] Acc: {overall_acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]) | Macro-F1: {overall_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])\n", flush=True)
    return results

# =========================================================================
# MAIN ENTRYPOINT
# =========================================================================

def main():
    t_start = time.time()
    figures_dir = os.path.join("figures", "baselines")
    os.makedirs(figures_dir, exist_ok=True)
    
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at '{dataset_path}'")
        
    print("=" * 80, flush=True)
    print("LIGHTGBM BASELINE TRAINING & EVALUATION (DE FEATURES)", flush=True)
    print("=" * 80, flush=True)
    print(f"Loading SEED-IV dataset from '{dataset_path}'...", flush=True)
    
    data = np.load(dataset_path)
    features_3d = data["features"]       # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    
    # Flatten 62 channels x 5 frequency bands = 310 features
    num_samples = len(labels)
    X = features_3d.reshape(num_samples, 62 * 5)
    y = labels
    print(f"Dataset loaded: {num_samples} samples, {X.shape[1]} DE features (62 channels x 5 bands).", flush=True)
    
    # 1. Subject-Dependent Protocol
    res_dep = run_subject_dependent(X, y, subject_ids, session_nums, trial_ids, figures_dir)
    
    # 2. Cross-Subject Protocol
    res_cross = run_cross_subject(X, y, subject_ids, figures_dir)
    
    # Save structured results (JSON & CSV)
    all_results = {
        "subject_dependent": res_dep,
        "cross_subject": res_cross
    }
    
    json_path = "lightgbm_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved structured JSON results to '{json_path}'", flush=True)
    
    csv_path = "lightgbm_baseline_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Protocol", "Accuracy", "Accuracy_95CI_Lower", "Accuracy_95CI_Upper", 
                         "Macro_Precision", "Macro_Precision_95CI_Lower", "Macro_Precision_95CI_Upper",
                         "Macro_Recall", "Macro_Recall_95CI_Lower", "Macro_Recall_95CI_Upper",
                         "Macro_F1", "Macro_F1_95CI_Lower", "Macro_F1_95CI_Upper",
                         "Macro_AUC", "Macro_AUC_95CI_Lower", "Macro_AUC_95CI_Upper"])
        for p_key, r in [("subject_dependent", res_dep), ("cross_subject", res_cross)]:
            writer.writerow([
                r["protocol"],
                f"{r['accuracy']:.4f}", f"{r['accuracy_ci'][0]:.4f}", f"{r['accuracy_ci'][1]:.4f}",
                f"{r['precision']:.4f}", f"{r['precision_ci'][0]:.4f}", f"{r['precision_ci'][1]:.4f}",
                f"{r['recall']:.4f}", f"{r['recall_ci'][0]:.4f}", f"{r['recall_ci'][1]:.4f}",
                f"{r['f1']:.4f}", f"{r['f1_ci'][0]:.4f}", f"{r['f1_ci'][1]:.4f}",
                f"{r['auc']:.4f}", f"{r['auc_ci'][0]:.4f}", f"{r['auc_ci'][1]:.4f}"
            ])
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)
    
    # Print Final Summary Table
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY: LIGHTGBM BASELINE RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Protocol':<20} | {'Accuracy (95% CI)':<25} | {'Macro-F1 (95% CI)':<25} | {'Macro-AUC (95% CI)':<25}", flush=True)
    print("-" * 105, flush=True)
    for p_key, r in [("subject_dependent", res_dep), ("cross_subject", res_cross)]:
        acc_str = f"{r['accuracy']:.4f} [{r['accuracy_ci'][0]:.4f}, {r['accuracy_ci'][1]:.4f}]"
        f1_str = f"{r['f1']:.4f} [{r['f1_ci'][0]:.4f}, {r['f1_ci'][1]:.4f}]"
        auc_str = f"{r['auc']:.4f} [{r['auc_ci'][0]:.4f}, {r['auc_ci'][1]:.4f}]"
        print(f"{r['protocol']:<20} | {acc_str:<25} | {f1_str:<25} | {auc_str:<25}", flush=True)
    print("=" * 80, flush=True)
    print(f"Total LightGBM pipeline runtime: {(time.time() - t_start)/60:.2f} minutes.", flush=True)

if __name__ == "__main__":
    main()
