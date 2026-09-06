import os
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve
from scipy.stats import chi2
import matplotlib.pyplot as plt

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# =========================================================================
# 1. DANN ARCHITECTURE DEFINITION
# =========================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANN(nn.Module):
    def __init__(self, in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=10, dropout=0.2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        self.grl = GradientReversal(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_domains)
        )
    def forward(self, x, alpha=0.0):
        self.grl.alpha = alpha
        feat = self.feature_extractor(x)
        class_out = self.class_classifier(feat)
        domain_feat = self.grl(feat)
        domain_out = self.domain_classifier(domain_feat)
        return class_out, domain_out

# =========================================================================
# 2. EVALUATION & METRICS HELPERS
# =========================================================================

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000, seed=42):
    """Computes 95% confidence intervals using bootstrap resampling."""
    rng = np.random.RandomState(seed)
    accs, precs, recs, f1s, aucs, kappas = [], [], [], [], [], []
    num_samples = len(y_true)
    
    for _ in range(num_resamples):
        indices = rng.choice(num_samples, num_samples, replace=True)
        y_t_res = y_true[indices]
        y_p_res = y_pred[indices]
        y_prob_res = y_prob[indices]
        
        accs.append(accuracy_score(y_t_res, y_p_res))
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

def evaluate_metrics(y_true, y_pred, y_prob, compute_ci=True):
    """Calculates accuracy, macro precision/recall/F1, macro AUC, and Cohen's Kappa."""
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    try:
        auc_val = float(roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr'))
    except Exception:
        auc_val = 0.5
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    res = {
        'accuracy': acc,
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc_val),
        'kappa': float(kappa)
    }
    
    if compute_ci:
        ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, 1000)
        res['accuracy_ci'] = ci_acc
        res['precision_ci'] = ci_prec
        res['recall_ci'] = ci_rec
        res['f1_ci'] = ci_f1
        res['auc_ci'] = ci_auc
        res['kappa_ci'] = ci_kappa
        
    return res

def compute_error_overlap_and_mcnemar(y_true, y_pred_m1, y_pred_m2, name_m1="DANN", name_m2="LightGBM"):
    """
    Computes 2x2 error contingency matrix, rescue rates, oracle upper bound, and McNemar's test.
    """
    correct_m1 = (y_pred_m1 == y_true)
    correct_m2 = (y_pred_m2 == y_true)
    
    n_11 = int(np.sum(correct_m1 & correct_m2))        # Both correct
    n_10 = int(np.sum(correct_m1 & (~correct_m2)))     # M1 correct, M2 wrong
    n_01 = int(np.sum((~correct_m1) & correct_m2))     # M2 correct, M1 wrong
    n_00 = int(np.sum((~correct_m1) & (~correct_m2)))   # Both wrong
    
    total = len(y_true)
    assert n_11 + n_10 + n_01 + n_00 == total
    
    # Rescue rates
    m1_errors = n_01 + n_00
    m2_errors = n_10 + n_00
    
    rescue_rate_m2_on_m1 = float(n_01 / m1_errors) if m1_errors > 0 else 0.0
    rescue_rate_m1_on_m2 = float(n_10 / m2_errors) if m2_errors > 0 else 0.0
    
    # Oracle Upper Bound (correct if EITHER model is correct)
    oracle_correct = n_11 + n_10 + n_01
    oracle_accuracy = float(oracle_correct / total)
    
    # McNemar's Test with continuity correction
    b = n_10
    c = n_01
    if b + c > 0:
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        p_val = float(1.0 - chi2.cdf(stat, df=1))
    else:
        stat = 0.0
        p_val = 1.0
        
    return {
        'contingency_matrix': {
            'both_correct_n11': n_11,
            f'{name_m1}_only_correct_n10': n_10,
            f'{name_m2}_only_correct_n01': n_01,
            'both_wrong_n00': n_00,
            'total_samples': total
        },
        'rescue_rates': {
            f'{name_m2}_rescues_{name_m1}_errors_pct': rescue_rate_m2_on_m1 * 100.0,
            f'{name_m2}_rescued_samples': n_01,
            f'total_{name_m1}_errors': m1_errors,
            f'{name_m1}_rescues_{name_m2}_errors_pct': rescue_rate_m1_on_m2 * 100.0,
            f'{name_m1}_rescued_samples': n_10,
            f'total_{name_m2}_errors': m2_errors
        },
        'oracle_upper_bound': {
            'oracle_correct_samples': oracle_correct,
            'oracle_accuracy_pct': oracle_accuracy * 100.0
        },
        'mcnemar_test': {
            'statistic': stat,
            'p_value': p_val,
            'is_significant_p05': p_val < 0.05,
            'is_significant_p001': p_val < 0.001
        }
    }

def save_evaluation_plots(y_true, y_pred, y_prob, prefix, figures_dir):
    """Generates and saves Confusion Matrix, ROC, PR, and Calibration diagrams (300 DPI)."""
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Confusion Matrix
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
    
    # 2. ROC Curves
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
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob[:, c], n_bins=10)
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
# 3. MAIN EVALUATION PIPELINE
# =========================================================================

def extract_dann_logits_and_scaled_probs(proto_name, optimal_temperature, device):
    """
    Loads trained DANN checkpoints and extracts unnormalized test logits and temperature-scaled probabilities.
    """
    dataset_path = "seed_iv_processed.npz"
    data = np.load(dataset_path)
    features_flat = data['features'].reshape(data['features'].shape[0], -1)
    labels = data['labels']
    subject_ids = data['subject_ids']
    session_nums = data['session_nums']
    trial_ids = data['trial_ids']
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys = np.unique(trial_keys)
    num_trials = len(unique_trial_keys)
    
    checkpoints_dir = os.path.join("checkpoints", "dann_final")
    
    if proto_name == 'cross_subject':
        subject_folds = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ]
        fold_configs = [{'fold': i+1, 'test_subs': subs, 'n_domains': 10} for i, subs in enumerate(subject_folds)]
    else:
        from sklearn.model_selection import KFold
        kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_configs = []
        for fold, (train_trial_idx, test_trial_idx) in enumerate(kf_outer.split(np.arange(num_trials))):
            outer_train_trials = unique_trial_keys[train_trial_idx]
            kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
            inner_train_idx, inner_val_idx = next(kf_inner.split(np.arange(len(outer_train_trials))))
            fold_configs.append({
                'fold': fold + 1,
                'train_trials': outer_train_trials[inner_train_idx],
                'val_trials': outer_train_trials[inner_val_idx],
                'test_trials': unique_trial_keys[test_trial_idx],
                'n_domains': 15
            })
            
    test_logits_list = []
    test_labels_list = []
    
    for f_cfg in fold_configs:
        fold_num = f_cfg['fold']
        if proto_name == 'cross_subject':
            test_subs = f_cfg['test_subs']
            train_pool_subs = [s for s in range(1, 16) if s not in test_subs]
            train_subs = train_pool_subs[:-2]
            train_mask = np.isin(subject_ids, train_subs)
            test_mask = np.isin(subject_ids, test_subs)
        else:
            train_mask = np.isin(trial_keys, list(f_cfg['train_trials']))
            test_mask = np.isin(trial_keys, list(f_cfg['test_trials']))
            
        X_train_raw = features_flat[train_mask]
        X_test_raw = features_flat[test_mask]
        y_test = labels[test_mask]
        
        # Scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)
        
        # Load DANN Checkpoint
        ckpt_name = f"dann_final_{proto_name}_fold{fold_num}.pt"
        ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        
        model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=f_cfg['n_domains'], dropout=0.2).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32)), batch_size=512, shuffle=False)
            for bx in test_loader:
                bx = bx[0].to(device)
                logits, _ = model(bx, alpha=0.0)
                test_logits_list.append(logits.cpu().numpy())
                
        test_labels_list.extend(y_test)
        
    test_logits_arr = np.concatenate(test_logits_list, axis=0)
    test_labels_arr = np.array(test_labels_list)
    
    # Softmax raw
    exp_logits = np.exp(test_logits_arr - np.max(test_logits_arr, axis=1, keepdims=True))
    raw_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Softmax temperature-scaled
    scaled_logits = test_logits_arr / optimal_temperature
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    temp_scaled_probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
    
    return raw_probs, temp_scaled_probs, test_labels_arr

def main():
    print("=" * 80, flush=True)
    print("STEP 1: HYBRID ENSEMBLE EVALUATION (DANN + LIGHTGBM)", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    
    figures_dir = os.path.join("figures", "ensemble")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Load optimal temperatures
    rai_path = "dann_responsible_ai_results.json"
    assert os.path.exists(rai_path), f"Missing {rai_path}"
    with open(rai_path, "r") as f:
        rai_data = json.load(f)
    temp_sd = rai_data['temperature_scaling']['optimal_temperatures']['subject_dependent']
    temp_cs = rai_data['temperature_scaling']['optimal_temperatures']['cross_subject']
    print(f"Optimal Temperatures Loaded: SD T*={temp_sd:.4f}, CS T*={temp_cs:.4f}", flush=True)
    
    ensemble_results = {}
    csv_rows = []
    
    for proto_key, proto_title, opt_t in [
        ('subject_dependent', 'Subject-Dependent Protocol', temp_sd),
        ('cross_subject', 'Cross-Subject Protocol', temp_cs)
    ]:
        print("\n" + "#" * 80, flush=True)
        print(f"EVALUATING PROTOCOL: {proto_title.upper()}", flush=True)
        print("#" * 80, flush=True)
        
        # 1. Load DANN saved predictions
        dann_file = f"oof_preds_dann_final_{proto_key}.npz"
        assert os.path.exists(dann_file), f"Missing {dann_file}"
        dann_data = np.load(dann_file)
        y_true_dann = dann_data['y_true']
        y_prob_dann_raw = dann_data['y_prob']
        y_pred_dann = dann_data['y_pred']
        
        # 2. Load LightGBM saved predictions
        lgb_file = f"oof_preds_lightgbm_{proto_key}.npz"
        assert os.path.exists(lgb_file), f"Missing {lgb_file}"
        lgb_data = np.load(lgb_file)
        y_true_lgb = lgb_data['y_true']
        y_prob_lgb = lgb_data['y_prob']
        y_pred_lgb = lgb_data['y_pred']
        
        # 3. Strict Fold & Sample Alignment Check
        print("  [ALIGNMENT CHECK] Verifying sample-by-sample ground-truth match...", flush=True)
        assert len(y_true_dann) == len(y_true_lgb) == 37575, f"Sample count mismatch: DANN={len(y_true_dann)}, LGB={len(y_true_lgb)}"
        assert np.array_equal(y_true_dann, y_true_lgb), "FATAL: Ground truth mismatch between DANN and LightGBM out-of-fold arrays!"
        y_true = y_true_dann
        print(f"  [ALIGNMENT VERIFIED] 100% exact match across all {len(y_true)} samples!", flush=True)
        
        # 4. Extract temperature-scaled DANN probabilities
        print(f"  [TEMPERATURE SCALING] Applying T*={opt_t:.4f} to DANN logits...", flush=True)
        raw_probs_extracted, temp_scaled_probs_dann, labels_extracted = extract_dann_logits_and_scaled_probs(proto_key, opt_t, device)
        assert np.array_equal(y_true, labels_extracted), "Extracted logits ground truth mismatch!"
        
        # 5. Baseline Reproduction Verification
        dann_metrics = evaluate_metrics(y_true, y_pred_dann, y_prob_dann_raw, compute_ci=False)
        lgb_metrics = evaluate_metrics(y_true, y_pred_lgb, y_prob_lgb, compute_ci=False)
        
        print(f"\n  [STANDALONE BENCHMARK REPRODUCTION]")
        print(f"    Standalone DANN:     Acc={dann_metrics['accuracy']*100:.4f}% | Macro-F1={dann_metrics['f1']:.4f} | AUC={dann_metrics['auc']:.4f} | Kappa={dann_metrics['kappa']:.4f}")
        print(f"    Standalone LightGBM: Acc={lgb_metrics['accuracy']*100:.4f}% | Macro-F1={lgb_metrics['f1']:.4f} | AUC={lgb_metrics['auc']:.4f} | Kappa={lgb_metrics['kappa']:.4f}")
        
        # Expected targets check:
        if proto_key == 'subject_dependent':
            assert abs(dann_metrics['accuracy'] - 0.6553) < 0.001, f"DANN SD mismatch: {dann_metrics['accuracy']}"
            assert abs(lgb_metrics['accuracy'] - 0.6130) < 0.001, f"LGBM SD mismatch: {lgb_metrics['accuracy']}"
        else:
            assert abs(dann_metrics['accuracy'] - 0.3841) < 0.001, f"DANN CS mismatch: {dann_metrics['accuracy']}"
            assert abs(lgb_metrics['accuracy'] - 0.3823) < 0.001, f"LGBM CS mismatch: {lgb_metrics['accuracy']}"
        print("  [REPRODUCTION VERIFIED] Both models reproduced exact validated benchmark figures within 0.001 tolerance.", flush=True)
        
        # 6. ENSEMBLE VARIANT A: Raw Softmax DANN + LightGBM (0.5 / 0.5)
        print("\n  [EVALUATING ENSEMBLE VARIANT A: Raw Softmax DANN + LightGBM (50/50)]", flush=True)
        prob_ens_A = 0.5 * y_prob_dann_raw + 0.5 * y_prob_lgb
        pred_ens_A = np.argmax(prob_ens_A, axis=1)
        metrics_ens_A = evaluate_metrics(y_true, pred_ens_A, prob_ens_A, compute_ci=True)
        
        print(f"    Ensemble Variant A => Acc: {metrics_ens_A['accuracy']*100:.2f}% (95% CI: [{metrics_ens_A['accuracy_ci'][0]*100:.2f}%, {metrics_ens_A['accuracy_ci'][1]*100:.2f}%])")
        print(f"                          Macro-F1: {metrics_ens_A['f1']:.4f} (95% CI: [{metrics_ens_A['f1_ci'][0]:.4f}, {metrics_ens_A['f1_ci'][1]:.4f}])")
        print(f"                          Macro-AUC: {metrics_ens_A['auc']:.4f} | Kappa: {metrics_ens_A['kappa']:.4f}")
        
        # 7. ENSEMBLE VARIANT B: Temperature-Scaled DANN + LightGBM (0.5 / 0.5)
        print("\n  [EVALUATING ENSEMBLE VARIANT B: Temperature-Scaled DANN + LightGBM (50/50)]", flush=True)
        prob_ens_B = 0.5 * temp_scaled_probs_dann + 0.5 * y_prob_lgb
        pred_ens_B = np.argmax(prob_ens_B, axis=1)
        metrics_ens_B = evaluate_metrics(y_true, pred_ens_B, prob_ens_B, compute_ci=True)
        
        print(f"    Ensemble Variant B => Acc: {metrics_ens_B['accuracy']*100:.2f}% (95% CI: [{metrics_ens_B['accuracy_ci'][0]*100:.2f}%, {metrics_ens_B['accuracy_ci'][1]*100:.2f}%])")
        print(f"                          Macro-F1: {metrics_ens_B['f1']:.4f} (95% CI: [{metrics_ens_B['f1_ci'][0]:.4f}, {metrics_ens_B['f1_ci'][1]:.4f}])")
        print(f"                          Macro-AUC: {metrics_ens_B['auc']:.4f} | Kappa: {metrics_ens_B['kappa']:.4f}")
        
        # 8. DIAGNOSTIC ERROR OVERLAP & MCNEMAR TEST
        print("\n  [ERROR OVERLAP & COMPLEMENTARY SIGNAL ANALYSIS]", flush=True)
        overlap_diag = compute_error_overlap_and_mcnemar(y_true, y_pred_dann, y_pred_lgb, "DANN", "LightGBM")
        c_mat = overlap_diag['contingency_matrix']
        r_rates = overlap_diag['rescue_rates']
        oracle = overlap_diag['oracle_upper_bound']
        mcnemar = overlap_diag['mcnemar_test']
        
        print(f"    Contingency Matrix (N={c_mat['total_samples']}):")
        print(f"      Both Correct (N11):           {c_mat['both_correct_n11']:,} ({c_mat['both_correct_n11']/len(y_true)*100:.2f}%)")
        print(f"      DANN Only Correct (N10):      {c_mat['DANN_only_correct_n10']:,} ({c_mat['DANN_only_correct_n10']/len(y_true)*100:.2f}%)")
        print(f"      LightGBM Only Correct (N01):  {c_mat['LightGBM_only_correct_n01']:,} ({c_mat['LightGBM_only_correct_n01']/len(y_true)*100:.2f}%)")
        print(f"      Both Incorrect (N00):         {c_mat['both_wrong_n00']:,} ({c_mat['both_wrong_n00']/len(y_true)*100:.2f}%)")
        print(f"    Rescue Rates:")
        print(f"      LightGBM rescues DANN errors: {r_rates['LightGBM_rescues_DANN_errors_pct']:.2f}% ({r_rates['LightGBM_rescued_samples']:,} / {r_rates['total_DANN_errors']:,})")
        print(f"      DANN rescues LightGBM errors: {r_rates['DANN_rescues_LightGBM_errors_pct']:.2f}% ({r_rates['DANN_rescued_samples']:,} / {r_rates['total_LightGBM_errors']:,})")
        print(f"    Theoretical Oracle Upper Bound: {oracle['oracle_accuracy_pct']:.2f}% ({oracle['oracle_correct_samples']:,} / {len(y_true):,})")
        print(f"    McNemar's Test: chi2={mcnemar['statistic']:.4f}, p={mcnemar['p_value']:.4e} (Significant: {mcnemar['is_significant_p05']})")
        
        # Save evaluation diagrams for the primary ensemble (Variant B / Variant A)
        prefix = f"hybrid_ensemble_{proto_key}"
        save_evaluation_plots(y_true, pred_ens_B, prob_ens_B, prefix, figures_dir)
        print(f"  [SAVED] Evaluation figures for {prefix} to '{figures_dir}'", flush=True)
        
        # Collect results
        ensemble_results[proto_key] = {
            'protocol': proto_key,
            'optimal_temperature': opt_t,
            'standalone_dann': dann_metrics,
            'standalone_lightgbm': lgb_metrics,
            'ensemble_variant_A_raw': metrics_ens_A,
            'ensemble_variant_B_calibrated': metrics_ens_B,
            'error_overlap_diagnostics': overlap_diag
        }
        
        csv_rows.append({
            'Protocol': proto_key,
            'Model': 'Standalone DANN',
            'Accuracy': f"{dann_metrics['accuracy']*100:.2f}%",
            'Macro-F1': f"{dann_metrics['f1']:.4f}",
            'Macro-AUC': f"{dann_metrics['auc']:.4f}",
            'Kappa': f"{dann_metrics['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{oracle['oracle_accuracy_pct']:.2f}%"
        })
        csv_rows.append({
            'Protocol': proto_key,
            'Model': 'Standalone LightGBM',
            'Accuracy': f"{lgb_metrics['accuracy']*100:.2f}%",
            'Macro-F1': f"{lgb_metrics['f1']:.4f}",
            'Macro-AUC': f"{lgb_metrics['auc']:.4f}",
            'Kappa': f"{lgb_metrics['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{oracle['oracle_accuracy_pct']:.2f}%"
        })
        csv_rows.append({
            'Protocol': proto_key,
            'Model': 'Hybrid Ensemble (Variant A: Raw)',
            'Accuracy': f"{metrics_ens_A['accuracy']*100:.2f}%",
            'Macro-F1': f"{metrics_ens_A['f1']:.4f}",
            'Macro-AUC': f"{metrics_ens_A['auc']:.4f}",
            'Kappa': f"{metrics_ens_A['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{oracle['oracle_accuracy_pct']:.2f}%"
        })
        csv_rows.append({
            'Protocol': proto_key,
            'Model': 'Hybrid Ensemble (Variant B: Calibrated)',
            'Accuracy': f"{metrics_ens_B['accuracy']*100:.2f}%",
            'Macro-F1': f"{metrics_ens_B['f1']:.4f}",
            'Macro-AUC': f"{metrics_ens_B['auc']:.4f}",
            'Kappa': f"{metrics_ens_B['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{oracle['oracle_accuracy_pct']:.2f}%"
        })
        
    # Save structured JSON
    json_path = "hybrid_ensemble_results.json"
    with open(json_path, "w") as f:
        json.dump(ensemble_results, f, indent=4)
    print(f"\nSaved structured JSON results to '{json_path}'", flush=True)
    
    # Save structured CSV
    csv_path = "hybrid_ensemble_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['Protocol', 'Model', 'Accuracy', 'Macro-F1', 'Macro-AUC', 'Kappa', 'Oracle_Upper_Bound'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)

if __name__ == '__main__':
    main()
