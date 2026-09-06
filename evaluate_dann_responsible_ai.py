import os
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    brier_score_loss,
    log_loss
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
CLASS_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

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

def enable_mc_dropout(model):
    """
    Enables dropout layers during inference while keeping BatchNorm in eval mode
    to preserve running statistics.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

# =========================================================================
# 2. CALIBRATION & ECE COMPUTATION
# =========================================================================

def compute_ece_and_bins(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    using equal-width binning on predicted confidence.
    """
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
        
    # Multi-class Brier Score (mean squared error over one-hot vectors)
    n_samples, n_classes = y_prob.shape
    y_true_onehot = np.zeros_like(y_prob)
    for i, t in enumerate(y_true):
        y_true_onehot[i, t] = 1.0
    brier = float(np.mean(np.sum((y_prob - y_true_onehot)**2, axis=1)))
    
    # Negative Log-Likelihood (NLL) / Cross-Entropy Loss
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

# =========================================================================
# 3. ACCURACY-VS-REJECTION (ABSTENTION) CURVES
# =========================================================================

def compute_abstention_curve(y_true, y_prob, rejection_rates=None):
    """
    Computes selective classification metrics as low-confidence samples are rejected.
    """
    if rejection_rates is None:
        rejection_rates = np.arange(0.0, 0.75, 0.05)
        
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    n_total = len(y_true)
    
    sorted_indices = np.argsort(confidences) # Ascending order: least confident first
    
    results = []
    for r in rejection_rates:
        n_reject = int(np.floor(r * n_total))
        retained_indices = sorted_indices[n_reject:]
        
        y_t_ret = y_true[retained_indices]
        y_p_ret = predictions[retained_indices]
        
        acc = float(np.mean(y_t_ret == y_p_ret))
        prec, rec, f1, _ = precision_recall_fscore_support(y_t_ret, y_p_ret, average='macro', zero_division=0)
        
        results.append({
            'rejection_rate': float(r),
            'retained_fraction': float(1.0 - r),
            'retained_count': int(len(retained_indices)),
            'accuracy': acc,
            'macro_f1': float(f1),
            'macro_precision': float(prec),
            'macro_recall': float(rec),
            'min_confidence': float(confidences[retained_indices].min()) if len(retained_indices) > 0 else 1.0,
            'mean_confidence': float(confidences[retained_indices].mean()) if len(retained_indices) > 0 else 1.0
        })
        
    return results

# =========================================================================
# 4. MONTE CARLO DROPOUT UNCERTAINTY DECOMPOSITION
# =========================================================================

def compute_mc_dropout_uncertainty(model, X_tensor, device, T=30, batch_size=512):
    """
    Executes T stochastic forward passes with Monte Carlo Dropout to decompose
    predictive uncertainty into Aleatoric (expected data entropy) and Epistemic (mutual info).
    """
    enable_mc_dropout(model)
    num_samples = len(X_tensor)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)
    
    # Storage for all T probability predictions: (T, N, C)
    all_pass_probs = np.zeros((T, num_samples, 4), dtype=np.float32)
    
    with torch.no_grad():
        for t in range(T):
            probs_list = []
            for bx in loader:
                bx = bx[0].to(device)
                logits, _ = model(bx, alpha=0.0)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(probs)
            all_pass_probs[t] = np.vstack(probs_list)
            
    # Predictive Mean: \bar{p}(x) = 1/T \sum_t p_t(x)
    mean_probs = np.mean(all_pass_probs, axis=0) # (N, 4)
    
    # Entropy helper: - \sum p * ln(p)
    eps = 1e-12
    mean_probs_clipped = np.clip(mean_probs, eps, 1.0)
    total_entropy = -np.sum(mean_probs_clipped * np.log(mean_probs_clipped), axis=1) # (N,)
    
    # Aleatoric Uncertainty: 1/T \sum_t H[p_t(x)]
    all_pass_clipped = np.clip(all_pass_probs, eps, 1.0)
    pass_entropies = -np.sum(all_pass_clipped * np.log(all_pass_clipped), axis=2) # (T, N)
    aleatoric = np.mean(pass_entropies, axis=0) # (N,)
    
    # Epistemic Uncertainty: Total Entropy - Aleatoric (Mutual Information I(y, w | x))
    epistemic = np.maximum(0.0, total_entropy - aleatoric) # (N,)
    
    return {
        'mean_probs': mean_probs,
        'total_entropy': total_entropy,
        'aleatoric': aleatoric,
        'epistemic': epistemic
    }

# =========================================================================
# 5. HIGH-RESOLUTION PLOTTING HELPERS (300 DPI)
# =========================================================================

def plot_reliability_diagrams(calib_results_dict, output_path):
    """
    Plots 10-bin Reliability Diagrams for Subject-Dependent and Cross-Subject protocols.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol', '#2980b9'),
        ('cross_subject', 'Cross-Subject Protocol', '#8e44ad')
    ]
    
    for ax, (proto_key, proto_title, color) in zip(axes, proto_configs):
        res = calib_results_dict[proto_key]
        bin_stats = res['bin_stats']
        
        centers = [b['bin_center'] for b in bin_stats]
        accs = [b['accuracy'] for b in bin_stats]
        confs = [b['confidence'] for b in bin_stats]
        counts = [b['count'] for b in bin_stats]
        widths = [b['bin_upper'] - b['bin_lower'] for b in bin_stats]
        
        # Plot perfect calibration diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Perfect Calibration')
        
        # Plot calibration bars
        bars = ax.bar(centers, accs, width=[w * 0.9 for w in widths], color=color, alpha=0.75, 
                      edgecolor='black', linewidth=0.8, label='Empirical Accuracy')
        
        # Highlight gaps with shaded hatch
        for i, (b, acc, conf) in enumerate(zip(bin_stats, accs, confs)):
            if b['count'] > 0:
                gap_low = min(acc, conf)
                gap_high = max(acc, conf)
                ax.fill_between([b['bin_lower'], b['bin_upper']], gap_low, gap_high, 
                                color='#e74c3c', alpha=0.25, hatch='//')
                
        # Annotate ECE, MCE, Brier
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
        ax.set_xlabel('Confidence (Mean Predicted Softmax Probability)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Empirical Accuracy', fontsize=11, fontweight='bold')
        ax.set_title(f'Reliability Diagram\n{proto_title}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', frameon=True, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Reliability diagrams to '{output_path}'", flush=True)

def plot_accuracy_vs_rejection(abstention_dict, output_path):
    """
    Plots Accuracy and Macro-F1 vs Rejection Rate curves with clinical threshold callouts.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol', 0.6553, [(0.70, '70% Target'), (0.75, '75% Target')]),
        ('cross_subject', 'Cross-Subject Protocol', 0.3841, [(0.45, '45% Target'), (0.50, '50% Target')])
    ]
    
    for ax, (proto_key, proto_title, baseline_acc, targets) in zip(axes, proto_configs):
        curve = abstention_dict[proto_key]
        rejections = [pt['rejection_rate'] * 100 for pt in curve]
        accuracies = [pt['accuracy'] * 100 for pt in curve]
        macro_f1s = [pt['macro_f1'] * 100 for pt in curve]
        
        ax.plot(rejections, accuracies, 'o-', color='#27ae60', lw=2.5, label='Retained Accuracy (%)')
        ax.plot(rejections, macro_f1s, 's--', color='#2980b9', lw=2.0, label='Retained Macro-F1 (%)')
        
        # Baseline reference line
        ax.axhline(baseline_acc * 100, color='gray', linestyle=':', lw=1.5, label=f'Full Data Baseline ({baseline_acc*100:.1f}%)')
        
        # Clinical target thresholds
        colors_targets = ['#e67e22', '#c0392b']
        for (tgt, tgt_label), t_col in zip(targets, colors_targets):
            ax.axhline(tgt * 100, color=t_col, linestyle='--', lw=1.2, alpha=0.8)
            # Find first rejection point crossing target
            cross_pt = next((pt for pt in curve if pt['accuracy'] >= tgt), None)
            if cross_pt is not None:
                cross_r = cross_pt['rejection_rate'] * 100
                cross_acc = cross_pt['accuracy'] * 100
                ax.plot(cross_r, cross_acc, marker='*', markersize=12, color=t_col)
                
                # Offset text intelligently based on x position
                if cross_r > 50:
                    x_off = cross_r - 26
                    y_off = cross_acc - 2.5
                else:
                    x_off = cross_r + 4
                    y_off = cross_acc - 4.5
                    
                ax.annotate(f"{tgt_label}: Reject {cross_r:.0f}%\n(Acc: {cross_acc:.1f}%)",
                            xy=(cross_r, cross_acc),
                            xytext=(x_off, y_off),
                            arrowprops=dict(facecolor=t_col, shrink=0.08, width=1, headwidth=6),
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor=t_col, alpha=0.9))
                
        ax.set_xlim(-2, 72)
        ax.set_xlabel('Rejection Rate (%) [Least Confident Abstention]', fontsize=11, fontweight='bold')
        ax.set_ylabel('Performance on Retained Samples (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Accuracy & F1 vs. Rejection\n{proto_title}', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right' if proto_key == 'cross_subject' else 'lower right', frameon=True, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Accuracy vs Rejection curves to '{output_path}'", flush=True)

def plot_subject_fairness(fairness_dict, output_path):
    """
    Plots 15-subject performance bar charts and demographic disparity metrics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol', '#3498db'),
        ('cross_subject', 'Cross-Subject Protocol', '#9b59b6')
    ]
    
    for ax, (proto_key, proto_title, bar_color) in zip(axes, proto_configs):
        sub_data = fairness_dict[proto_key]
        subjects = [f"S{s['subject_id']}" for s in sub_data]
        accs = [s['accuracy'] * 100 for s in sub_data]
        f1s = [s['macro_f1'] * 100 for s in sub_data]
        
        x = np.arange(len(subjects))
        width = 0.38
        
        rects1 = ax.bar(x - width/2, accs, width, label='Accuracy (%)', color=bar_color, alpha=0.85, edgecolor='black', linewidth=0.7)
        rects2 = ax.bar(x + width/2, f1s, width, label='Macro-F1 (%)', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.7)
        
        # Mean reference line & std band
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        min_acc = np.min(accs)
        max_acc = np.max(accs)
        gap = max_acc - min_acc
        cv = (std_acc / mean_acc) * 100
        
        ax.axhline(mean_acc, color='#e74c3c', linestyle='-', lw=1.8, label=f'Mean Acc: {mean_acc:.1f}%')
        ax.fill_between([-0.5, 14.5], mean_acc - std_acc, mean_acc + std_acc, color='#e74c3c', alpha=0.12, label=r'$\pm 1\sigma$ Band')
        
        # Disparity summary text box
        disp_text = (f"Max Subject: {max_acc:.1f}%\n"
                     f"Min Subject: {min_acc:.1f}%\n"
                     f"Disparity Gap: {gap:.1f}%\n"
                     f"Coeff. of Var: {cv:.1f}%")
        ax.text(0.03, 0.95, disp_text, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#bdc3c7'))
        
        ax.set_xticks(x)
        ax.set_xticklabels(subjects, fontsize=9, fontweight='bold')
        ax.set_xlim(-0.6, 14.6)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Subject Identifier', fontsize=11, fontweight='bold')
        ax.set_ylabel('Performance (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Subject-Level Fairness & Performance Distribution\n{proto_title}', fontsize=11.5, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        ax.legend(loc='lower right', frameon=True, fontsize=9)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Subject fairness disparity plots to '{output_path}'", flush=True)

def plot_uncertainty_decomposition(fairness_dict, output_path):
    """
    Plots subject-level Aleatoric vs Epistemic uncertainty decomposition and diagnostic scatter plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol', 0),
        ('cross_subject', 'Cross-Subject Protocol', 1)
    ]
    
    for proto_key, proto_title, col_idx in proto_configs:
        sub_data = fairness_dict[proto_key]
        subjects = [f"S{s['subject_id']}" for s in sub_data]
        accs = np.array([s['accuracy'] * 100 for s in sub_data])
        aleatoric = np.array([s['mean_aleatoric'] for s in sub_data])
        epistemic = np.array([s['mean_epistemic'] for s in sub_data])
        
        # 1. Stacked Bar Chart (Row 0)
        ax_bar = axes[0, col_idx]
        x = np.arange(len(subjects))
        width = 0.6
        
        p1 = ax_bar.bar(x, aleatoric, width, label='Aleatoric (Data Noise)', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.7)
        p2 = ax_bar.bar(x, epistemic, width, bottom=aleatoric, label='Epistemic (Model Deficit)', color='#e67e22', alpha=0.85, edgecolor='black', linewidth=0.7)
        
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(subjects, fontsize=9, fontweight='bold')
        ax_bar.set_xlabel('Subject Identifier', fontsize=10.5, fontweight='bold')
        ax_bar.set_ylabel('Predictive Uncertainty (Nats)', fontsize=10.5, fontweight='bold')
        ax_bar.set_title(f'Uncertainty Decomposition by Subject\n{proto_title}', fontsize=11.5, fontweight='bold', pad=8)
        ax_bar.grid(True, linestyle='--', alpha=0.4, axis='y')
        ax_bar.legend(loc='upper right', frameon=True, fontsize=9.5)
        
        # 2. Scatter Plot: Accuracy vs Epistemic & Aleatoric (Row 1)
        ax_scat = axes[1, col_idx]
        
        ax_scat.scatter(epistemic, accs, s=80, color='#e67e22', edgecolor='black', label='Epistemic vs. Accuracy', alpha=0.9, zorder=3)
        ax_scat.scatter(aleatoric, accs, s=80, color='#3498db', edgecolor='black', marker='^', label='Aleatoric vs. Accuracy', alpha=0.9, zorder=3)
        
        for s_idx, (e_val, a_val, acc_val) in enumerate(zip(epistemic, aleatoric, accs)):
            ax_scat.annotate(f"S{s_idx+1}", (e_val, acc_val), textcoords="offset points", xytext=(4, 4), fontsize=8, color='#d35400')
            
        # Fit trendlines
        if len(epistemic) > 1:
            z_e = np.polyfit(epistemic, accs, 1)
            p_e = np.poly1d(z_e)
            x_e_line = np.linspace(epistemic.min(), epistemic.max(), 50)
            ax_scat.plot(x_e_line, p_e(x_e_line), color='#e67e22', linestyle='--', lw=1.5, alpha=0.8)
            
            z_a = np.polyfit(aleatoric, accs, 1)
            p_a = np.poly1d(z_a)
            x_a_line = np.linspace(aleatoric.min(), aleatoric.max(), 50)
            ax_scat.plot(x_a_line, p_a(x_a_line), color='#3498db', linestyle=':', lw=1.5, alpha=0.8)
            
        ax_scat.set_xlabel('Uncertainty Magnitude (Nats)', fontsize=10.5, fontweight='bold')
        ax_scat.set_ylabel('Subject Accuracy (%)', fontsize=10.5, fontweight='bold')
        ax_scat.set_title(f'Accuracy vs. Uncertainty Driver\n{proto_title}', fontsize=11.5, fontweight='bold', pad=8)
        ax_scat.grid(True, linestyle='--', alpha=0.4)
        ax_scat.legend(loc='upper right', frameon=True, fontsize=9.5)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Aleatoric vs Epistemic uncertainty plots to '{output_path}'", flush=True)

# =========================================================================
# 6. MAIN EVALUATION PIPELINE
# =========================================================================

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("RESPONSIBLE AI, CALIBRATION & UNCERTAINTY DECOMPOSITION EVALUATION", flush=True)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)
    
    figures_dir = os.path.join("figures", "responsible_ai")
    checkpoints_dir = os.path.join("checkpoints", "dann_final")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Load Dataset
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_path}' not found.")
        
    print(f"Loading SEED-IV dataset from '{dataset_path}'...", flush=True)
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    num_samples = len(labels)
    features_flat = features.reshape(num_samples, -1)
    
    # 2. Define Protocols & Splits (Identical to train_final_dann.py)
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
        sd_splits.append({
            'fold': fold + 1,
            'train_trials': unique_trial_keys[train_trial_idx],
            'test_trials': unique_trial_keys[test_trial_idx],
            'n_domains': 15
        })
    protocols_config['subject_dependent'] = sd_splits
    
    # Result containers
    calibration_results = {}
    abstention_results = {}
    fairness_results = {}
    subject_csv_rows = []
    
    for proto_name, fold_configs in protocols_config.items():
        print("\n" + "#" * 80, flush=True)
        print(f"EVALUATING PROTOCOL: {proto_name.upper()} (Zero Retraining / 10 Checkpoints)", flush=True)
        print("#" * 80, flush=True)
        
        all_y_true = []
        all_y_pred = []
        all_y_prob = []
        all_subjects = []
        all_total_entropy = []
        all_aleatoric = []
        all_epistemic = []
        
        for f_cfg in fold_configs:
            fold_num = f_cfg['fold']
            if proto_name == 'cross_subject':
                test_subs = f_cfg['test_subs']
                train_subs = [s for s in range(1, 16) if s not in test_subs][:-2]
                train_mask = np.isin(subject_ids, train_subs)
                test_mask = np.isin(subject_ids, test_subs)
            else:
                train_mask = np.isin(trial_keys, list(f_cfg['train_trials']))
                test_mask = np.isin(trial_keys, list(f_cfg['test_trials']))
                
            X_train_raw = features_flat[train_mask]
            X_test_raw = features_flat[test_mask]
            y_test = labels[test_mask]
            s_test = subject_ids[test_mask]
            
            # Scaler
            scaler = StandardScaler()
            scaler.fit(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            # Load model checkpoint
            ckpt_name = f"dann_final_{proto_name}_fold{fold_num}.pt"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            
            model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=f_cfg['n_domains'], dropout=0.2).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            
            # Standard Evaluation
            model.eval()
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=512, shuffle=False)
            
            fold_probs_list = []
            with torch.no_grad():
                for bx in test_loader:
                    bx = bx[0].to(device)
                    c_out, _ = model(bx, alpha=0.0)
                    probs = torch.softmax(c_out, dim=1).cpu().numpy()
                    fold_probs_list.append(probs)
            fold_probs = np.vstack(fold_probs_list)
            fold_preds = np.argmax(fold_probs, axis=1)
            
            # Monte Carlo Dropout Uncertainty (T=30 stochastic passes)
            mc_res = compute_mc_dropout_uncertainty(model, X_test_tensor, device, T=30, batch_size=512)
            
            all_y_true.extend(y_test)
            all_y_pred.extend(fold_preds)
            all_y_prob.extend(fold_probs)
            all_subjects.extend(s_test)
            all_total_entropy.extend(mc_res['total_entropy'])
            all_aleatoric.extend(mc_res['aleatoric'])
            all_epistemic.extend(mc_res['epistemic'])
            
            print(f"  Fold {fold_num}/5 complete: {len(y_test)} test samples evaluated (Accuracy: {np.mean(y_test == fold_preds)*100:.2f}%)", flush=True)
            
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob)
        all_subjects = np.array(all_subjects)
        all_total_entropy = np.array(all_total_entropy)
        all_aleatoric = np.array(all_aleatoric)
        all_epistemic = np.array(all_epistemic)
        
        # 1. Calibration Metrics
        calib_stats = compute_ece_and_bins(all_y_true, all_y_prob, n_bins=10)
        calibration_results[proto_name] = calib_stats
        print(f"\n  [{proto_name.upper()}] Calibration: ECE = {calib_stats['ece']*100:.2f}%, MCE = {calib_stats['mce']*100:.2f}%, Brier = {calib_stats['brier_score']:.4f}, NLL = {calib_stats['nll']:.3f}", flush=True)
        
        # 2. Abstention Curve
        abst_curve = compute_abstention_curve(all_y_true, all_y_prob)
        abstention_results[proto_name] = abst_curve
        print(f"  [{proto_name.upper()}] Abstention Curve computed (0% to 70% rejection)", flush=True)
        
        # 3. Subject-Level Breakdown & Uncertainty Decomposition
        subject_breakdown = []
        for sub_id in range(1, 16):
            sub_mask = (all_subjects == sub_id)
            sub_yt = all_y_true[sub_mask]
            sub_yp = all_y_pred[sub_mask]
            sub_yprob = all_y_prob[sub_mask]
            sub_tot_ent = all_total_entropy[sub_mask]
            sub_ale = all_aleatoric[sub_mask]
            sub_epi = all_epistemic[sub_mask]
            
            sub_acc = float(np.mean(sub_yt == sub_yp))
            sub_p, sub_r, sub_f1, _ = precision_recall_fscore_support(sub_yt, sub_yp, average='macro', zero_division=0)
            sub_conf = float(np.mean(np.max(sub_yprob, axis=1)))
            sub_mean_tot = float(np.mean(sub_tot_ent))
            sub_mean_ale = float(np.mean(sub_ale))
            sub_mean_epi = float(np.mean(sub_epi))
            sub_epi_ratio = float(sub_mean_epi / (sub_mean_ale + 1e-12))
            
            # Primary Driver Diagnosis
            # If accuracy is below protocol average: check if epistemic ratio is high or aleatoric is high
            proto_avg_acc = float(np.mean(all_y_true == all_y_pred))
            if sub_acc < proto_avg_acc:
                if sub_mean_epi > np.mean(all_epistemic) and sub_epi_ratio > 0.08:
                    primary_driver = "Model Limitation (High Epistemic)"
                else:
                    primary_driver = "Data Limitation (High Aleatoric)"
            else:
                primary_driver = "High Performing / Robust"
                
            sub_entry = {
                'subject_id': int(sub_id),
                'sample_count': int(np.sum(sub_mask)),
                'accuracy': sub_acc,
                'macro_f1': float(sub_f1),
                'mean_confidence': sub_conf,
                'mean_total_entropy': sub_mean_tot,
                'mean_aleatoric': sub_mean_ale,
                'mean_epistemic': sub_mean_epi,
                'epistemic_ratio': sub_epi_ratio,
                'primary_driver': primary_driver
            }
            subject_breakdown.append(sub_entry)
            
            # Add to CSV row collection
            subject_csv_rows.append([
                sub_id,
                proto_name,
                int(np.sum(sub_mask)),
                f"{sub_acc:.4f}",
                f"{float(sub_f1):.4f}",
                f"{sub_conf:.4f}",
                f"{sub_mean_tot:.4f}",
                f"{sub_mean_ale:.4f}",
                f"{sub_mean_epi:.4f}",
                f"{sub_epi_ratio:.4f}",
                primary_driver
            ])
            
        fairness_results[proto_name] = subject_breakdown
        
        # Disparity summary
        sub_accs = [s['accuracy'] for s in subject_breakdown]
        min_acc = np.min(sub_accs)
        max_acc = np.max(sub_accs)
        mean_acc = np.mean(sub_accs)
        std_acc = np.std(sub_accs)
        gap = max_acc - min_acc
        cv = std_acc / mean_acc
        print(f"  [{proto_name.upper()}] Subject Fairness: Mean={mean_acc*100:.2f}%, Range=[{min_acc*100:.2f}%, {max_acc*100:.2f}%], Gap={gap*100:.2f}%, CV={cv*100:.2f}%", flush=True)
        
    # 4. Generate Visualizations (300 DPI)
    print("\n" + "=" * 80, flush=True)
    print("GENERATING 300 DPI RESPONSIBLE AI VISUALIZATIONS", flush=True)
    print("=" * 80, flush=True)
    
    fig_rel = os.path.join(figures_dir, "dann_reliability_diagrams.png")
    plot_reliability_diagrams(calibration_results, fig_rel)
    
    fig_abst = os.path.join(figures_dir, "dann_accuracy_vs_rejection_curves.png")
    plot_accuracy_vs_rejection(abstention_results, fig_abst)
    
    fig_fair = os.path.join(figures_dir, "dann_subject_fairness_disparity.png")
    plot_subject_fairness(fairness_results, fig_fair)
    
    fig_unc = os.path.join(figures_dir, "dann_aleatoric_vs_epistemic_uncertainty.png")
    plot_uncertainty_decomposition(fairness_results, fig_unc)
    
    # 5. Save Structured Outputs (JSON & CSV)
    json_path = "dann_responsible_ai_results.json"
    full_output = {
        'calibration': calibration_results,
        'abstention': abstention_results,
        'subject_fairness_uncertainty': fairness_results
    }
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=4)
    print(f"\n[Saved] Complete Responsible AI JSON to '{json_path}'", flush=True)
    
    csv_path = "dann_subject_fairness_uncertainty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Subject_ID", "Protocol", "Sample_Count", "Accuracy", "Macro_F1",
            "Mean_Confidence", "Total_Entropy", "Aleatoric_Uncertainty", "Epistemic_Uncertainty",
            "Epistemic_Aleatoric_Ratio", "Primary_Driver"
        ])
        for row in subject_csv_rows:
            writer.writerow(row)
    print(f"[Saved] Complete Subject Fairness & Uncertainty CSV to '{csv_path}'", flush=True)
    
    print("=" * 80, flush=True)
    print(f"Responsible AI evaluation completed successfully in {(time.time() - t_start):.2f} seconds.", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
