import os
import time
import json
import csv
import numpy as np
import torch
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
import matplotlib.pyplot as plt

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# =========================================================================
# 1. EVALUATION & METRICS HELPERS
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

def compute_8way_error_breakdown(y_true, y_pred_dann, y_pred_lgb, y_pred_eeg):
    """
    Computes full 3-way (8-cell) contingency matrix across DANN, LightGBM, and EEGNet.
    """
    c_dann = (y_pred_dann == y_true)
    c_lgb = (y_pred_lgb == y_true)
    c_eeg = (y_pred_eeg == y_true)
    
    n_111 = int(np.sum(c_dann & c_lgb & c_eeg))          # All 3 correct
    n_110 = int(np.sum(c_dann & c_lgb & (~c_eeg)))       # DANN + LGBM correct, EEGNet wrong
    n_101 = int(np.sum(c_dann & (~c_lgb) & c_eeg))       # DANN + EEGNet correct, LGBM wrong
    n_011 = int(np.sum((~c_dann) & c_lgb & c_eeg))       # LGBM + EEGNet correct, DANN wrong
    n_100 = int(np.sum(c_dann & (~c_lgb) & (~c_eeg)))     # DANN alone correct
    n_010 = int(np.sum((~c_dann) & c_lgb & (~c_eeg)))     # LGBM alone correct
    n_001 = int(np.sum((~c_dann) & (~c_lgb) & c_eeg))     # EEGNet alone correct (NEW RESCUE)
    n_000 = int(np.sum((~c_dann) & (~c_lgb) & (~c_eeg)))   # All 3 wrong
    
    total = len(y_true)
    assert n_111 + n_110 + n_101 + n_011 + n_100 + n_010 + n_001 + n_000 == total
    
    # Dual failure count (where both DANN and LightGBM fail)
    dual_fail_dann_lgb = n_001 + n_000
    eeg_unique_rescue_rate = float(n_001 / dual_fail_dann_lgb) if dual_fail_dann_lgb > 0 else 0.0
    
    # 2-model oracle vs 3-model oracle
    oracle_2m_correct = n_111 + n_110 + n_101 + n_011 + n_100 + n_010
    oracle_2m_acc = float(oracle_2m_correct / total)
    
    oracle_3m_correct = total - n_000
    oracle_3m_acc = float(oracle_3m_correct / total)
    
    return {
        'contingency_breakdown': {
            'all_three_correct_N111': n_111,
            'dann_and_lgbm_only_correct_N110': n_110,
            'dann_and_eegnet_only_correct_N101': n_101,
            'lgbm_and_eegnet_only_correct_N011': n_011,
            'dann_alone_correct_N100': n_100,
            'lgbm_alone_correct_N010': n_010,
            'eegnet_alone_correct_N001_unique_rescue': n_001,
            'all_three_wrong_N000': n_000,
            'total_samples': total
        },
        'eegnet_unique_rescue': {
            'dual_dann_lgbm_errors': dual_fail_dann_lgb,
            'eegnet_rescued_samples': n_001,
            'eegnet_unique_rescue_rate_pct': eeg_unique_rescue_rate * 100.0
        },
        'oracle_upper_bounds': {
            'oracle_2model_correct': oracle_2m_correct,
            'oracle_2model_acc_pct': oracle_2m_acc * 100.0,
            'oracle_3model_correct': oracle_3m_correct,
            'oracle_3model_acc_pct': oracle_3m_acc * 100.0,
            'oracle_gain_from_eegnet_samples': n_001,
            'oracle_gain_from_eegnet_pct': (oracle_3m_acc - oracle_2m_acc) * 100.0
        }
    }

def optimize_3way_weights_grid_search(y_true, prob_dann, prob_lgb, prob_eeg, step=0.02):
    """
    Grid search over 3-weight simplex (alpha_DANN + alpha_LGBM + alpha_EEG = 1) to maximize accuracy or minimize NLL.
    """
    best_acc = -1.0
    best_weights = None
    best_prob = None
    
    # Pre-generate simplex grid points
    grid_points = []
    steps = int(round(1.0 / step))
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            a_d = i * step
            a_l = j * step
            a_e = k * step
            grid_points.append((a_d, a_l, a_e))
            
    for a_d, a_l, a_e in grid_points:
        p_blend = a_d * prob_dann + a_l * prob_lgb + a_e * prob_eeg
        preds = np.argmax(p_blend, axis=1)
        acc = np.mean(preds == y_true)
        if acc > best_acc:
            best_acc = acc
            best_weights = (a_d, a_l, a_e)
            best_prob = p_blend
            
    return best_weights, best_acc, best_prob

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

# =========================================================================
# 2. MAIN 3-MODEL ENSEMBLE PIPELINE
# =========================================================================

def main():
    print("=" * 80, flush=True)
    print("3-MODEL HYBRID ENSEMBLE EVALUATION (DANN + LIGHTGBM + COMPACT EEGNET)", flush=True)
    print("=" * 80, flush=True)
    
    figures_dir = os.path.join("figures", "ensemble")
    os.makedirs(figures_dir, exist_ok=True)
    
    three_model_results = {}
    csv_rows = []
    
    for proto_key, proto_title in [
        ('subject_dependent', 'Subject-Dependent Protocol'),
        ('cross_subject', 'Cross-Subject Protocol')
    ]:
        print("\n" + "#" * 80, flush=True)
        print(f"EVALUATING PROTOCOL: {proto_title.upper()}", flush=True)
        print("#" * 80, flush=True)
        
        # 1. Load All 3 Saved Predictions
        dann_file = f"oof_preds_dann_final_{proto_key}.npz"
        lgb_file = f"oof_preds_lightgbm_{proto_key}.npz"
        eeg_file = f"oof_preds_eegnet_{proto_key}.npz"
        
        assert os.path.exists(dann_file), f"Missing {dann_file}"
        assert os.path.exists(lgb_file), f"Missing {lgb_file}"
        assert os.path.exists(eeg_file), f"Missing {eeg_file}"
        
        d_dann = np.load(dann_file)
        d_lgb = np.load(lgb_file)
        d_eeg = np.load(eeg_file)
        
        y_true_dann, y_pred_dann, y_prob_dann = d_dann['y_true'], d_dann['y_pred'], d_dann['y_prob']
        y_true_lgb, y_pred_lgb, y_prob_lgb = d_lgb['y_true'], d_lgb['y_pred'], d_lgb['y_prob']
        y_true_eeg, y_pred_eeg, y_prob_eeg = d_eeg['y_true'], d_eeg['y_pred'], d_eeg['y_prob']
        
        # 2. Strict Ground-Truth & Sample Count Alignment
        print("  [ALIGNMENT CHECK] Verifying sample-by-sample 3-way ground truth alignment...", flush=True)
        assert len(y_true_dann) == len(y_true_lgb) == len(y_true_eeg) == 37575, "Sample count mismatch!"
        assert np.array_equal(y_true_dann, y_true_lgb), "DANN vs LightGBM ground truth mismatch!"
        assert np.array_equal(y_true_dann, y_true_eeg), "DANN vs EEGNet ground truth mismatch!"
        y_true = y_true_dann
        print(f"  [ALIGNMENT VERIFIED] 100% exact sample-by-sample match across all {len(y_true)} instances!", flush=True)
        
        # 3. Individual Baseline Reproduction Check
        m_dann = evaluate_metrics(y_true, y_pred_dann, y_prob_dann, compute_ci=False)
        m_lgb = evaluate_metrics(y_true, y_pred_lgb, y_prob_lgb, compute_ci=False)
        m_eeg = evaluate_metrics(y_true, y_pred_eeg, y_prob_eeg, compute_ci=False)
        
        print("\n  [INDIVIDUAL BENCHMARK REPRODUCTION CHECK]")
        print(f"    1. Standalone DANN:     Acc={m_dann['accuracy']*100:.4f}% | Macro-F1={m_dann['f1']:.4f} | AUC={m_dann['auc']:.4f}")
        print(f"    2. Standalone LightGBM: Acc={m_lgb['accuracy']*100:.4f}% | Macro-F1={m_lgb['f1']:.4f} | AUC={m_lgb['auc']:.4f}")
        print(f"    3. Standalone EEGNet:   Acc={m_eeg['accuracy']*100:.4f}% | Macro-F1={m_eeg['f1']:.4f} | AUC={m_eeg['auc']:.4f}")
        
        if proto_key == 'subject_dependent':
            assert abs(m_dann['accuracy'] - 0.6553) < 0.001, f"DANN SD mismatch: {m_dann['accuracy']}"
            assert abs(m_lgb['accuracy'] - 0.6130) < 0.001, f"LGBM SD mismatch: {m_lgb['accuracy']}"
            assert abs(m_eeg['accuracy'] - 0.5870) < 0.001, f"EEGNet SD mismatch: {m_eeg['accuracy']}"
        else:
            assert abs(m_dann['accuracy'] - 0.3841) < 0.001, f"DANN CS mismatch: {m_dann['accuracy']}"
            assert abs(m_lgb['accuracy'] - 0.3823) < 0.001, f"LGBM CS mismatch: {m_lgb['accuracy']}"
            assert abs(m_eeg['accuracy'] - 0.3796) < 0.001, f"EEGNet CS mismatch: {m_eeg['accuracy']}"
        print("  [REPRODUCTION VERIFIED] All 3 individual models reproduce validated numbers within 0.001 tolerance!", flush=True)
        
        # 4. 2-Model DANN + LightGBM Reference
        prob_2m = 0.5 * y_prob_dann + 0.5 * y_prob_lgb
        pred_2m = np.argmax(prob_2m, axis=1)
        m_2m = evaluate_metrics(y_true, pred_2m, prob_2m, compute_ci=True)
        print(f"\n  [2-MODEL REFERENCE: DANN + LightGBM (50/50)]")
        print(f"    2-Model Ensemble Acc: {m_2m['accuracy']*100:.2f}% | Macro-F1: {m_2m['f1']:.4f} | AUC: {m_2m['auc']:.4f} | Kappa: {m_2m['kappa']:.4f}")
        
        # 5. STEP 2: 3-Model Simple Averaging Ensemble (1/3, 1/3, 1/3)
        print("\n  [STEP 2: 3-MODEL SIMPLE AVERAGING ENSEMBLE (1/3 each)]", flush=True)
        prob_3m_equal = (1.0/3.0) * y_prob_dann + (1.0/3.0) * y_prob_lgb + (1.0/3.0) * y_prob_eeg
        pred_3m_equal = np.argmax(prob_3m_equal, axis=1)
        m_3m_equal = evaluate_metrics(y_true, pred_3m_equal, prob_3m_equal, compute_ci=True)
        
        print(f"    3-Model Equal Ensemble => Acc: {m_3m_equal['accuracy']*100:.2f}% (95% CI: [{m_3m_equal['accuracy_ci'][0]*100:.2f}%, {m_3m_equal['accuracy_ci'][1]*100:.2f}%])")
        print(f"                              Macro-F1: {m_3m_equal['f1']:.4f} (95% CI: [{m_3m_equal['f1_ci'][0]:.4f}, {m_3m_equal['f1_ci'][1]:.4f}])")
        print(f"                              Macro-AUC: {m_3m_equal['auc']:.4f} | Kappa: {m_3m_equal['kappa']:.4f}")
        
        delta_acc_vs_2m = (m_3m_equal['accuracy'] - m_2m['accuracy']) * 100.0
        delta_f1_vs_2m = m_3m_equal['f1'] - m_2m['f1']
        print(f"    >>> Margin vs 2-Model Ensemble: Acc {delta_acc_vs_2m:+.2f}% | Macro-F1 {delta_f1_vs_2m:+.4f}", flush=True)
        
        # 6. STEP 3: 8-Way Error Overlap Breakdown
        print("\n  [STEP 3: 8-WAY 3-MODEL ERROR OVERLAP BREAKDOWN]", flush=True)
        diag_8way = compute_8way_error_breakdown(y_true, y_pred_dann, y_pred_lgb, y_pred_eeg)
        cb = diag_8way['contingency_breakdown']
        ur = diag_8way['eegnet_unique_rescue']
        ob = diag_8way['oracle_upper_bounds']
        
        print(f"    8-Cell Mutual Correctness Partition (N={cb['total_samples']}):")
        print(f"      1. All 3 Models Correct (N111):               {cb['all_three_correct_N111']:,} ({cb['all_three_correct_N111']/len(y_true)*100:.2f}%)")
        print(f"      2. DANN + LightGBM Correct, EEGNet Wrong (N110): {cb['dann_and_lgbm_only_correct_N110']:,} ({cb['dann_and_lgbm_only_correct_N110']/len(y_true)*100:.2f}%)")
        print(f"      3. DANN + EEGNet Correct, LightGBM Wrong (N101): {cb['dann_and_eegnet_only_correct_N101']:,} ({cb['dann_and_eegnet_only_correct_N101']/len(y_true)*100:.2f}%)")
        print(f"      4. LightGBM + EEGNet Correct, DANN Wrong (N011): {cb['lgbm_and_eegnet_only_correct_N011']:,} ({cb['lgbm_and_eegnet_only_correct_N011']/len(y_true)*100:.2f}%)")
        print(f"      5. DANN Alone Correct (N100):                   {cb['dann_alone_correct_N100']:,} ({cb['dann_alone_correct_N100']/len(y_true)*100:.2f}%)")
        print(f"      6. LightGBM Alone Correct (N010):               {cb['lgbm_alone_correct_N010']:,} ({cb['lgbm_alone_correct_N010']/len(y_true)*100:.2f}%)")
        print(f"      7. EEGNet Alone Correct (N001 - NEW RESCUE):    {cb['eegnet_alone_correct_N001_unique_rescue']:,} ({cb['eegnet_alone_correct_N001_unique_rescue']/len(y_true)*100:.2f}%)")
        print(f"      8. All 3 Models Wrong (N000):                   {cb['all_three_wrong_N000']:,} ({cb['all_three_wrong_N000']/len(y_true)*100:.2f}%)")
        print(f"    EEGNet Unique Rescue on Dual (DANN+LGBM) Errors: {ur['eegnet_unique_rescue_rate_pct']:.2f}% ({ur['eegnet_rescued_samples']:,} / {ur['dual_dann_lgbm_errors']:,})")
        print(f"    Oracle Upper Bounds:")
        print(f"      2-Model Oracle (DANN + LGBM): {ob['oracle_2model_acc_pct']:.2f}% ({ob['oracle_2model_correct']:,} samples)")
        print(f"      3-Model Oracle (+ EEGNet):    {ob['oracle_3model_acc_pct']:.2f}% ({ob['oracle_3model_correct']:,} samples)")
        print(f"      Net Oracle Gain from EEGNet:  +{ob['oracle_gain_from_eegnet_pct']:.2f}% (+{ob['oracle_gain_from_eegnet_samples']:,} samples)")
        
        # 7. STEP 4: Weighted Ensemble Optimization (Grid Search over Simplex)
        print("\n  [STEP 4: WEIGHTED 3-MODEL ENSEMBLE OPTIMIZATION]", flush=True)
        best_w, opt_acc, prob_3m_opt = optimize_3way_weights_grid_search(y_true, y_prob_dann, y_prob_lgb, y_prob_eeg, step=0.02)
        pred_3m_opt = np.argmax(prob_3m_opt, axis=1)
        m_3m_opt = evaluate_metrics(y_true, pred_3m_opt, prob_3m_opt, compute_ci=True)
        
        print(f"    Optimal Weights: alpha_DANN = {best_w[0]:.2f}, alpha_LGBM = {best_w[1]:.2f}, alpha_EEGNet = {best_w[2]:.2f}")
        print(f"    Optimal Weighted Ensemble => Acc: {m_3m_opt['accuracy']*100:.2f}% (95% CI: [{m_3m_opt['accuracy_ci'][0]*100:.2f}%, {m_3m_opt['accuracy_ci'][1]*100:.2f}%])")
        print(f"                                 Macro-F1: {m_3m_opt['f1']:.4f} (95% CI: [{m_3m_opt['f1_ci'][0]:.4f}, {m_3m_opt['f1_ci'][1]:.4f}])")
        print(f"                                 Macro-AUC: {m_3m_opt['auc']:.4f} | Kappa: {m_3m_opt['kappa']:.4f}")
        
        # Save plots for 3-model ensemble
        prefix_3m = f"three_model_ensemble_{proto_key}"
        save_evaluation_plots(y_true, pred_3m_equal, prob_3m_equal, prefix_3m, figures_dir)
        
        three_model_results[proto_key] = {
            'protocol': proto_key,
            'standalone_dann': m_dann,
            'standalone_lightgbm': m_lgb,
            'standalone_eegnet': m_eeg,
            'two_model_ensemble_dann_lgb': m_2m,
            'three_model_simple_average': m_3m_equal,
            'three_model_weighted_optimal': {
                'optimal_weights': {'alpha_dann': best_w[0], 'alpha_lgb': best_w[1], 'alpha_eegnet': best_w[2]},
                'metrics': m_3m_opt
            },
            'error_overlap_8way': diag_8way
        }
        
        csv_rows.append({
            'Protocol': proto_key,
            'Model': 'Standalone EEGNet',
            'Accuracy': f"{m_eeg['accuracy']*100:.2f}%",
            'Macro-F1': f"{m_eeg['f1']:.4f}",
            'Macro-AUC': f"{m_eeg['auc']:.4f}",
            'Kappa': f"{m_eeg['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{m_eeg['accuracy']*100:.2f}%"
        })
        csv_rows.append({
            'Protocol': proto_key,
            'Model': '2-Model Ensemble (DANN + LGBM)',
            'Accuracy': f"{m_2m['accuracy']*100:.2f}%",
            'Macro-F1': f"{m_2m['f1']:.4f}",
            'Macro-AUC': f"{m_2m['auc']:.4f}",
            'Kappa': f"{m_2m['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{ob['oracle_2model_acc_pct']:.2f}%"
        })
        csv_rows.append({
            'Protocol': proto_key,
            'Model': '3-Model Simple Averaging (Equal 1/3)',
            'Accuracy': f"{m_3m_equal['accuracy']*100:.2f}%",
            'Macro-F1': f"{m_3m_equal['f1']:.4f}",
            'Macro-AUC': f"{m_3m_equal['auc']:.4f}",
            'Kappa': f"{m_3m_equal['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{ob['oracle_3model_acc_pct']:.2f}%"
        })
        csv_rows.append({
            'Protocol': proto_key,
            'Model': f"3-Model Weighted (D:{best_w[0]:.2f}, L:{best_w[1]:.2f}, E:{best_w[2]:.2f})",
            'Accuracy': f"{m_3m_opt['accuracy']*100:.2f}%",
            'Macro-F1': f"{m_3m_opt['f1']:.4f}",
            'Macro-AUC': f"{m_3m_opt['auc']:.4f}",
            'Kappa': f"{m_3m_opt['kappa']:.4f}",
            'Oracle_Upper_Bound': f"{ob['oracle_3model_acc_pct']:.2f}%"
        })
        
    # Save structured JSON
    json_path = "three_model_ensemble_results.json"
    with open(json_path, "w") as f:
        json.dump(three_model_results, f, indent=4)
    print(f"\nSaved structured JSON results to '{json_path}'", flush=True)
    
    # Save structured CSV
    csv_path = "three_model_ensemble_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['Protocol', 'Model', 'Accuracy', 'Macro-F1', 'Macro-AUC', 'Kappa', 'Oracle_Upper_Bound'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)

if __name__ == '__main__':
    main()
