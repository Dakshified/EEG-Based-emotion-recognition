import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']

# =========================================================================
# 1. DANN ARCHITECTURE
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
# 2. CALIBRATION METRICS HELPERS
# =========================================================================

def compute_ece_and_bins(y_true, y_prob, n_bins=10):
    """Computes Expected Calibration Error (ECE), MCE, Brier, NLL, and bin statistics."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            abs_diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)
            count = int(np.sum(in_bin))
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = 0.0
            abs_diff = 0.0
            count = 0
            
        bin_stats.append({
            'bin_lower': float(bin_lower),
            'bin_upper': float(bin_upper),
            'bin_center': float((bin_lower + bin_upper) / 2.0),
            'count': count,
            'prop': float(prop_in_bin),
            'accuracy': float(accuracy_in_bin),
            'confidence': float(avg_confidence_in_bin),
            'gap': float(abs_diff)
        })
        
    y_true_onehot = np.zeros_like(y_prob)
    for i, t in enumerate(y_true):
        y_true_onehot[i, t] = 1.0
    brier = float(np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1)))
    
    eps = 1e-12
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    nll = float(log_loss(y_true, y_prob_clipped))
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'brier_score': brier,
        'nll': nll,
        'bin_stats': bin_stats
    }

def softmax_with_temp(logits, temperature):
    """Applies temperature scaling to logits and computes softmax probabilities."""
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def fit_temperature(val_logits, val_labels):
    """Finds optimal temperature T > 0 minimizing NLL on validation set."""
    def nll_obj(T):
        probs = softmax_with_temp(val_logits, T)
        eps = 1e-12
        probs_clipped = np.clip(probs, eps, 1.0 - eps)
        return log_loss(val_labels, probs_clipped)
        
    res = minimize_scalar(nll_obj, bounds=(0.01, 50.0), method='bounded')
    return float(res.x)

# =========================================================================
# 3. RELIABILITY PLOT COMPARISON (BEFORE VS AFTER)
# =========================================================================

def plot_temperature_scaling_comparison(uncalib_dict, calib_dict, optimal_temps, output_path):
    """Generates 4-panel comparison figure (Uncalibrated vs Temperature Scaled)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    
    proto_keys = ['subject_dependent', 'cross_subject']
    proto_titles = ['Subject-Dependent Protocol', 'Cross-Subject Protocol']
    
    for row_idx, (p_key, p_title) in enumerate(zip(proto_keys, proto_titles)):
        opt_T = optimal_temps[p_key]
        
        # Column 0: Uncalibrated
        ax_uncal = axes[row_idx, 0]
        uncal_res = uncalib_dict[p_key]
        _plot_single_reliability(ax_uncal, uncal_res, f"{p_title}\nUncalibrated (T=1.000)", '#e74c3c')
        
        # Column 1: Temperature Scaled
        ax_cal = axes[row_idx, 1]
        cal_res = calib_dict[p_key]
        _plot_single_reliability(ax_cal, cal_res, f"{p_title}\nTemperature Scaled (T={opt_T:.3f})", '#27ae60')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Temperature scaled reliability comparison to '{output_path}'", flush=True)

def _plot_single_reliability(ax, res, title, bar_color):
    bin_stats = res['bin_stats']
    centers = [b['bin_center'] for b in bin_stats]
    accs = [b['accuracy'] for b in bin_stats]
    confs = [b['confidence'] for b in bin_stats]
    widths = [b['bin_upper'] - b['bin_lower'] for b in bin_stats]
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Perfect Calibration')
    ax.bar(centers, accs, width=[w * 0.9 for w in widths], color=bar_color, alpha=0.75, 
           edgecolor='black', linewidth=0.8, label='Empirical Accuracy')
    
    for b, acc, conf in zip(bin_stats, accs, confs):
        if b['count'] > 0:
            gap_low = min(acc, conf)
            gap_high = max(acc, conf)
            ax.fill_between([b['bin_lower'], b['bin_upper']], gap_low, gap_high, 
                            color='#e74c3c' if gap_high == conf else '#2980b9', alpha=0.25, hatch='//')
            
    ece_val = res['ece'] * 100
    mce_val = res['mce'] * 100
    brier_val = res['brier_score']
    nll_val = res['nll']
    
    text_str = (f"ECE: {ece_val:.2f}%\n"
                f"MCE: {mce_val:.2f}%\n"
                f"Brier Score: {brier_val:.4f}\n"
                f"NLL: {nll_val:.3f}")
    
    ax.text(0.05, 0.92, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))
    
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Confidence (Mean Predicted Probability)', fontsize=10.5, fontweight='bold')
    ax.set_ylabel('Empirical Accuracy', fontsize=10.5, fontweight='bold')
    ax.set_title(title, fontsize=11.5, fontweight='bold', pad=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True, fontsize=9.5)

# =========================================================================
# 4. MAIN TEMPERATURE SCALING PIPELINE
# =========================================================================

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("STEP 1: TEMPERATURE SCALING EVALUATION FOR CALIBRATED INDUCTIVE DANN", flush=True)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)
    
    figures_dir = os.path.join("figures", "responsible_ai")
    checkpoints_dir = os.path.join("checkpoints", "dann_final")
    os.makedirs(figures_dir, exist_ok=True)
    
    dataset_path = "seed_iv_processed.npz"
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    num_samples = len(labels)
    features_flat = features.reshape(num_samples, -1)
    
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys = np.unique(trial_keys)
    num_trials = len(unique_trial_keys)
    
    protocols_config = {
        'cross_subject': [
            {'fold': 1, 'test_subs': [1, 2, 3], 'n_domains': 10},
            {'fold': 2, 'test_subs': [4, 5, 6], 'n_domains': 10},
            {'fold': 3, 'test_subs': [7, 8, 9], 'n_domains': 10},
            {'fold': 4, 'test_subs': [10, 11, 12], 'n_domains': 10},
            {'fold': 5, 'test_subs': [13, 14, 15], 'n_domains': 10}
        ]
    }
    
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    sd_splits = []
    for fold, (train_trial_idx, test_trial_idx) in enumerate(kf_outer.split(np.arange(num_trials))):
        outer_train_trials = unique_trial_keys[train_trial_idx]
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(kf_inner.split(np.arange(len(outer_train_trials))))
        
        sd_splits.append({
            'fold': fold + 1,
            'train_trials': outer_train_trials[inner_train_idx],
            'val_trials': outer_train_trials[inner_val_idx],
            'test_trials': unique_trial_keys[test_trial_idx],
            'n_domains': 15
        })
    protocols_config['subject_dependent'] = sd_splits
    
    uncalib_results = {}
    calib_results = {}
    optimal_temperatures = {}
    
    for proto_name, fold_configs in protocols_config.items():
        print("\n" + "#" * 80, flush=True)
        print(f"PROCESSING PROTOCOL: {proto_name.upper()} TEMPERATURE SCALING", flush=True)
        print("#" * 80, flush=True)
        
        val_logits_all = []
        val_labels_all = []
        
        test_logits_all = []
        test_labels_all = []
        
        for f_cfg in fold_configs:
            fold_num = f_cfg['fold']
            if proto_name == 'cross_subject':
                test_subs = f_cfg['test_subs']
                train_pool_subs = [s for s in range(1, 16) if s not in test_subs]
                val_subs = train_pool_subs[-2:]
                train_subs = train_pool_subs[:-2]
                
                train_mask = np.isin(subject_ids, train_subs)
                val_mask = np.isin(subject_ids, val_subs)
                test_mask = np.isin(subject_ids, test_subs)
            else:
                train_mask = np.isin(trial_keys, list(f_cfg['train_trials']))
                val_mask = np.isin(trial_keys, list(f_cfg['val_trials']))
                test_mask = np.isin(trial_keys, list(f_cfg['test_trials']))
                
            X_train_raw = features_flat[train_mask]
            X_val_raw = features_flat[val_mask]
            y_val = labels[val_mask]
            X_test_raw = features_flat[test_mask]
            y_test = labels[test_mask]
            
            scaler = StandardScaler()
            scaler.fit(X_train_raw)
            X_val_scaled = scaler.transform(X_val_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            ckpt_name = f"dann_final_{proto_name}_fold{fold_num}.pt"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            
            model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=f_cfg['n_domains'], dropout=0.2).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            # Validation logits extraction
            with torch.no_grad():
                val_loader = DataLoader(TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32)), batch_size=512, shuffle=False)
                val_fold_logits = []
                for bx in val_loader:
                    bx = bx[0].to(device)
                    c_out, _ = model(bx, alpha=0.0)
                    val_fold_logits.append(c_out.cpu().numpy())
                val_logits_all.append(np.vstack(val_fold_logits))
                val_labels_all.append(y_val)
                
                # Test logits extraction
                test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32)), batch_size=512, shuffle=False)
                test_fold_logits = []
                for bx in test_loader:
                    bx = bx[0].to(device)
                    c_out, _ = model(bx, alpha=0.0)
                    test_fold_logits.append(c_out.cpu().numpy())
                test_logits_all.append(np.vstack(test_fold_logits))
                test_labels_all.append(y_test)
                
        val_logits_arr = np.vstack(val_logits_all)
        val_labels_arr = np.concatenate(val_labels_all)
        test_logits_arr = np.vstack(test_logits_all)
        test_labels_arr = np.concatenate(test_labels_all)
        
        # 1. Fit Optimal Temperature on Validation Splits
        T_opt = fit_temperature(val_logits_arr, val_labels_arr)
        optimal_temperatures[proto_name] = T_opt
        print(f"  Optimal Temperature T fitted on validation set: T = {T_opt:.4f}", flush=True)
        
        # 2. Uncalibrated Test Metrics (T=1.0)
        uncal_probs = softmax_with_temp(test_logits_arr, 1.0)
        uncal_stats = compute_ece_and_bins(test_labels_arr, uncal_probs, n_bins=10)
        uncalib_results[proto_name] = uncal_stats
        
        # 3. Calibrated Test Metrics (T=T_opt)
        cal_probs = softmax_with_temp(test_logits_arr, T_opt)
        cal_stats = compute_ece_and_bins(test_labels_arr, cal_probs, n_bins=10)
        calib_results[proto_name] = cal_stats
        
        # 4. Verify Accuracy Invariance
        uncal_preds = np.argmax(uncal_probs, axis=1)
        cal_preds = np.argmax(cal_probs, axis=1)
        acc_diff = np.sum(uncal_preds != cal_preds)
        print(f"  Accuracy Invariance Verification: {acc_diff} predictions changed (Expected: 0).", flush=True)
        acc_val = np.mean(test_labels_arr == cal_preds)
        print(f"  Test Accuracy: {acc_val*100:.2f}% (Identical before and after scaling).", flush=True)
        
        print(f"  ECE Before: {uncal_stats['ece']*100:.2f}%  -->  ECE After: {cal_stats['ece']*100:.2f}%  (Improvement: -{(uncal_stats['ece'] - cal_stats['ece'])*100:.2f}%)", flush=True)
        print(f"  MCE Before: {uncal_stats['mce']*100:.2f}%  -->  MCE After: {cal_stats['mce']*100:.2f}%", flush=True)
        print(f"  Brier Before: {uncal_stats['brier_score']:.4f}  -->  Brier After: {cal_stats['brier_score']:.4f}", flush=True)
        print(f"  NLL Before: {uncal_stats['nll']:.3f}  -->  NLL After: {cal_stats['nll']:.3f}", flush=True)
        
    # 5. Generate Comparison Figure
    fig_comp_path = os.path.join(figures_dir, "dann_temperature_scaled_reliability.png")
    plot_temperature_scaling_comparison(uncalib_results, calib_results, optimal_temperatures, fig_comp_path)
    
    # 6. Update dann_responsible_ai_results.json
    json_path = "dann_responsible_ai_results.json"
    with open(json_path, "r") as f:
        full_json = json.load(f)
        
    full_json['temperature_scaling'] = {
        'optimal_temperatures': optimal_temperatures,
        'uncalibrated': uncalib_results,
        'calibrated': calib_results
    }
    
    with open(json_path, "w") as f:
        json.dump(full_json, f, indent=4)
    print(f"\n[Updated] Responsible AI results JSON with temperature scaling in '{json_path}'", flush=True)
    print("=" * 80, flush=True)
    print(f"Temperature scaling phase complete in {(time.time() - t_start):.2f} seconds.", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
