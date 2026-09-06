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
from sklearn.metrics import precision_recall_fscore_support
import lightgbm as lgb
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
# 2. INFERENCE HELPERS
# =========================================================================

def evaluate_dann_on_array(model, X_arr, device, batch_size=512):
    """Batched DANN evaluation on GPU."""
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X_arr, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    preds_list = []
    with torch.no_grad():
        for bx in loader:
            bx = bx[0].to(device)
            c_out, _ = model(bx, alpha=0.0)
            preds_list.append(c_out.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds_list)

def evaluate_lgbm_on_array(model, X_arr):
    """LightGBM evaluation on CPU."""
    return model.predict(X_arr)

# =========================================================================
# 3. HIGH-RESOLUTION PLOTTING (300 DPI)
# =========================================================================

def plot_channel_dropout_robustness(robustness_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol'),
        ('cross_subject', 'Cross-Subject Protocol')
    ]
    
    for ax, (proto_key, proto_title) in zip(axes, proto_configs):
        p_data = robustness_results[proto_key]['channel_dropout']
        k_levels = [entry['level'] for entry in p_data['dann']]
        pct_failed = [f"k={k}\n({k/62*100:.0f}%)" for k in k_levels]
        
        # DANN
        dann_acc = np.array([e['mean_accuracy'] * 100 for e in p_data['dann']])
        dann_acc_std = np.array([e['std_accuracy'] * 100 for e in p_data['dann']])
        dann_f1 = np.array([e['mean_macro_f1'] * 100 for e in p_data['dann']])
        
        # LightGBM
        lgb_acc = np.array([e['mean_accuracy'] * 100 for e in p_data['lightgbm']])
        lgb_acc_std = np.array([e['std_accuracy'] * 100 for e in p_data['lightgbm']])
        lgb_f1 = np.array([e['mean_macro_f1'] * 100 for e in p_data['lightgbm']])
        
        x = np.arange(len(k_levels))
        
        # Plot DANN
        ax.plot(x, dann_acc, 'o-', color='#27ae60', lw=2.5, label=f"Calibrated DANN (Clean: {dann_acc[0]:.1f}%)")
        ax.fill_between(x, dann_acc - dann_acc_std, dann_acc + dann_acc_std, color='#27ae60', alpha=0.15)
        
        # Plot LightGBM
        ax.plot(x, lgb_acc, 's--', color='#2980b9', lw=2.2, label=f"LightGBM (Clean: {lgb_acc[0]:.1f}%)")
        ax.fill_between(x, lgb_acc - lgb_acc_std, lgb_acc + lgb_acc_std, color='#2980b9', alpha=0.15)
        
        # Determine dynamic y-limits to give plenty of room for annotations
        y_min = min(dann_acc.min() - dann_acc_std.max(), lgb_acc.min() - lgb_acc_std.max())
        y_max = max(dann_acc.max() + dann_acc_std.max(), lgb_acc.max() + lgb_acc_std.max())
        ax.set_ylim(max(0, y_min - 6.5), min(100, y_max + 7.0))
        
        # Annotate Retention at max corruption (k=15)
        dann_ret_15 = p_data['dann'][-1]['retention_pct']
        lgb_ret_15 = p_data['lightgbm'][-1]['retention_pct']
        
        # Adaptive positioning: place higher model above, lower model below
        if dann_acc[-1] >= lgb_acc[-1]:
            dann_ytext = dann_acc[-1] + 3.2
            lgb_ytext = lgb_acc[-1] - 4.2
        else:
            dann_ytext = dann_acc[-1] - 4.2
            lgb_ytext = lgb_acc[-1] + 3.2
            
        ax.annotate(f"DANN k=15: {dann_acc[-1]:.1f}%\n(Ret: {dann_ret_15:.1f}%)",
                    xy=(x[-1], dann_acc[-1]), xytext=(x[-1] - 1.3, dann_ytext),
                    arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1, headwidth=5),
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#27ae60', alpha=0.9))
                    
        ax.annotate(f"LGBM k=15: {lgb_acc[-1]:.1f}%\n(Ret: {lgb_ret_15:.1f}%)",
                    xy=(x[-1], lgb_acc[-1]), xytext=(x[-1] - 1.3, lgb_ytext),
                    arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=1, headwidth=5),
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ebf5fb', edgecolor='#2980b9', alpha=0.9))
                    
        ax.set_xticks(x)
        ax.set_xticklabels(pct_failed, fontsize=9.5, fontweight='bold')
        ax.set_xlabel('Number of Failed Channels (k out of 62)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'Test Accuracy (%) [Mean $\pm$ 1$\sigma$ across 20 trials]', fontsize=10.5, fontweight='bold')
        ax.set_title(f'Channel Dropout Robustness (Electrode Failure)\n{proto_title}', fontsize=11.5, fontweight='bold', pad=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower left', frameon=True, fontsize=9.5)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Channel dropout robustness figure to '{output_path}'", flush=True)

def plot_gaussian_noise_robustness(robustness_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol'),
        ('cross_subject', 'Cross-Subject Protocol')
    ]
    
    for ax, (proto_key, proto_title) in zip(axes, proto_configs):
        p_data = robustness_results[proto_key]['gaussian_noise']
        sigmas = [entry['level'] for entry in p_data['dann']]
        sig_labels = [f"$\\sigma$={s:.2f}" if s > 0 else "Clean ($\\sigma$=0)" for s in sigmas]
        
        # DANN
        dann_acc = np.array([e['mean_accuracy'] * 100 for e in p_data['dann']])
        dann_acc_std = np.array([e['std_accuracy'] * 100 for e in p_data['dann']])
        
        # LightGBM
        lgb_acc = np.array([e['mean_accuracy'] * 100 for e in p_data['lightgbm']])
        lgb_acc_std = np.array([e['std_accuracy'] * 100 for e in p_data['lightgbm']])
        
        x = np.arange(len(sigmas))
        
        # Plot DANN
        ax.plot(x, dann_acc, 'o-', color='#27ae60', lw=2.5, label=f"Calibrated DANN (Clean: {dann_acc[0]:.1f}%)")
        ax.fill_between(x, dann_acc - dann_acc_std, dann_acc + dann_acc_std, color='#27ae60', alpha=0.15)
        
        # Plot LightGBM
        ax.plot(x, lgb_acc, 's--', color='#2980b9', lw=2.2, label=f"LightGBM (Clean: {lgb_acc[0]:.1f}%)")
        ax.fill_between(x, lgb_acc - lgb_acc_std, lgb_acc + lgb_acc_std, color='#2980b9', alpha=0.15)
        
        # Determine dynamic y-limits
        y_min = min(dann_acc.min() - dann_acc_std.max(), lgb_acc.min() - lgb_acc_std.max())
        y_max = max(dann_acc.max() + dann_acc_std.max(), lgb_acc.max() + lgb_acc_std.max())
        ax.set_ylim(max(0, y_min - 6.5), min(100, y_max + 7.0))
        
        # Annotate Retention at sigma=1.0
        dann_ret_1 = p_data['dann'][-1]['retention_pct']
        lgb_ret_1 = p_data['lightgbm'][-1]['retention_pct']
        
        # Adaptive positioning: place higher model above, lower model below
        if dann_acc[-1] >= lgb_acc[-1]:
            dann_ytext = dann_acc[-1] + 3.2
            lgb_ytext = lgb_acc[-1] - 4.2
        else:
            dann_ytext = dann_acc[-1] - 4.2
            lgb_ytext = lgb_acc[-1] + 3.2
            
        ax.annotate(f"DANN $\\sigma$=1.0: {dann_acc[-1]:.1f}%\n(Ret: {dann_ret_1:.1f}%)",
                    xy=(x[-1], dann_acc[-1]), xytext=(x[-1] - 1.2, dann_ytext),
                    arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=1, headwidth=5),
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', edgecolor='#27ae60', alpha=0.9))
                    
        ax.annotate(f"LGBM $\\sigma$=1.0: {lgb_acc[-1]:.1f}%\n(Ret: {lgb_ret_1:.1f}%)",
                    xy=(x[-1], lgb_acc[-1]), xytext=(x[-1] - 1.2, lgb_ytext),
                    arrowprops=dict(facecolor='#2980b9', shrink=0.08, width=1, headwidth=5),
                    fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ebf5fb', edgecolor='#2980b9', alpha=0.9))
                    
        ax.set_xticks(x)
        ax.set_xticklabels(sig_labels, fontsize=9.5, fontweight='bold')
        ax.set_xlabel('Gaussian Noise Std Dev (relative to feature std)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'Test Accuracy (%) [Mean $\pm$ 1$\sigma$ across 20 trials]', fontsize=10.5, fontweight='bold')
        ax.set_title(f'Gaussian Noise Robustness (Additive Sensor Noise)\n{proto_title}', fontsize=11.5, fontweight='bold', pad=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower left', frameon=True, fontsize=9.5)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Gaussian noise robustness figure to '{output_path}'", flush=True)

def plot_robustness_summary_heatmap(robustness_results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    proto_configs = [
        ('subject_dependent', 'Subject-Dependent Protocol'),
        ('cross_subject', 'Cross-Subject Protocol')
    ]
    
    for ax, (proto_key, proto_title) in zip(axes, proto_configs):
        p_data = robustness_results[proto_key]
        
        # Columns: [k=3, k=6, k=9, k=12, k=15, σ=0.10, σ=0.25, σ=0.50, σ=1.00]
        col_labels = ['k=3', 'k=6', 'k=9', 'k=12', 'k=15', 'σ=0.10', 'σ=0.25', 'σ=0.50', 'σ=1.00']
        row_labels = ['Calibrated DANN', 'LightGBM']
        
        matrix = np.zeros((2, 9))
        
        # Channel dropout retentions (skip k=0)
        for i, m_key in enumerate(['dann', 'lightgbm']):
            for j in range(1, 6):
                matrix[i, j-1] = p_data['channel_dropout'][m_key][j]['retention_pct']
                
        # Gaussian noise retentions (skip sigma=0)
        for i, m_key in enumerate(['dann', 'lightgbm']):
            for j in range(1, 5):
                matrix[i, 5 + (j-1)] = p_data['gaussian_noise'][m_key][j]['retention_pct']
                
        im = ax.imshow(matrix, cmap='YlGnBu', vmin=50, vmax=100)
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.18, fraction=0.05)
        cbar.set_label('Relative Accuracy Retention (%)', fontsize=10, fontweight='bold')
        
        for r in range(2):
            for c in range(9):
                val = matrix[r, c]
                ax.text(c, r, f"{val:.1f}%", ha='center', va='center', fontsize=9.5, fontweight='bold',
                        color='white' if val < 75 or val > 92 else 'black')
                
        ax.set_xticks(range(9))
        ax.set_xticklabels(col_labels, fontsize=9.5, fontweight='bold')
        ax.set_yticks(range(2))
        ax.set_yticklabels(row_labels, fontsize=10.5, fontweight='bold')
        ax.set_title(f'Robustness Retention Heatmap\n{proto_title}', fontsize=11.5, fontweight='bold', pad=10)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] Robustness summary heatmap to '{output_path}'", flush=True)

# =========================================================================
# 4. MAIN ROBUSTNESS EVALUATION PIPELINE
# =========================================================================

def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("STEP 2: ROBUSTNESS PHASE (CHANNEL DROPOUT & GAUSSIAN NOISE STRESS TESTS)", flush=True)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)
    
    figures_dir = os.path.join("figures", "robustness")
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
        sd_splits.append({
            'fold': fold + 1,
            'train_trials': unique_trial_keys[train_trial_idx],
            'test_trials': unique_trial_keys[test_trial_idx],
            'n_domains': 15
        })
    protocols_config['subject_dependent'] = sd_splits
    
    # Stress test levels
    channel_dropout_levels = [0, 3, 6, 9, 12, 15]
    gaussian_noise_levels = [0.0, 0.1, 0.25, 0.5, 1.0]
    num_trials = 20
    
    robustness_results = {
        'cross_subject': {'channel_dropout': {'dann': [], 'lightgbm': []}, 'gaussian_noise': {'dann': [], 'lightgbm': []}},
        'subject_dependent': {'channel_dropout': {'dann': [], 'lightgbm': []}, 'gaussian_noise': {'dann': [], 'lightgbm': []}}
    }
    
    csv_rows = []
    
    for proto_name, fold_configs in protocols_config.items():
        print("\n" + "#" * 80, flush=True)
        print(f"EVALUATING ROBUSTNESS: {proto_name.upper()} (DANN vs. LightGBM)", flush=True)
        print("#" * 80, flush=True)
        
        # Load all 5 folds data & models into memory
        fold_data = []
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
            y_train = labels[train_mask]
            X_test_raw = features_flat[test_mask]
            y_test = labels[test_mask]
            
            # Scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            # 1. DANN Model Checkpoint
            ckpt_name = f"dann_final_{proto_name}_fold{fold_num}.pt"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            
            dann_model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=f_cfg['n_domains'], dropout=0.2).to(device)
            dann_model.load_state_dict(checkpoint['state_dict'])
            dann_model.eval()
            
            # 2. Train LightGBM outer model with validated hyperparameters
            lgb_params = {'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 31}
            lgb_model = lgb.LGBMClassifier(**lgb_params, objective='multiclass', num_class=4, random_state=42, n_jobs=-1, verbose=-1)
            lgb_model.fit(X_train_scaled, y_train)
            
            fold_data.append({
                'fold': fold_num,
                'X_test_scaled': X_test_scaled,
                'y_test': y_test,
                'dann_model': dann_model,
                'lgb_model': lgb_model
            })
            print(f"  Fold {fold_num}/5 loaded: DANN checkpoint + trained LightGBM model.", flush=True)
            
        # ---------------------------------------------------------------------
        # EXPERIMENT A: CHANNEL DROPOUT CORRUPTION
        # ---------------------------------------------------------------------
        print(f"\n  [Experiment A: Channel Dropout] Evaluating k in {channel_dropout_levels} (20 paired trials per level)...", flush=True)
        clean_dann_acc, clean_lgb_acc = None, None
        
        for k in channel_dropout_levels:
            dann_trial_accs, dann_trial_f1s = [], []
            lgb_trial_accs, lgb_trial_f1s = [], []
            
            n_eval_trials = 1 if k == 0 else num_trials
            
            for t in range(n_eval_trials):
                rng = np.random.RandomState(42 + 1000 * k + t)
                
                if k == 0:
                    corrupted_channels = np.array([], dtype=int)
                else:
                    corrupted_channels = rng.choice(62, size=k, replace=False)
                    
                all_y_true = []
                all_dann_preds = []
                all_lgb_preds = []
                
                for f_entry in fold_data:
                    X_test = f_entry['X_test_scaled'].copy()
                    y_test = f_entry['y_test']
                    
                    if k > 0:
                        for c in corrupted_channels:
                            feat_indices = [c * 5 + b for b in range(5)]
                            X_test[:, feat_indices] = 0.0 # Exactly 0.0 in standardized space
                            
                    # Batched DANN evaluation (GPU)
                    d_preds = evaluate_dann_on_array(f_entry['dann_model'], X_test, device)
                    # LightGBM evaluation (CPU)
                    l_preds = evaluate_lgbm_on_array(f_entry['lgb_model'], X_test)
                    
                    all_y_true.extend(y_test)
                    all_dann_preds.extend(d_preds)
                    all_lgb_preds.extend(l_preds)
                    
                y_true_arr = np.array(all_y_true)
                d_preds_arr = np.array(all_dann_preds)
                l_preds_arr = np.array(all_lgb_preds)
                
                d_acc = np.mean(y_true_arr == d_preds_arr)
                _, _, d_f1, _ = precision_recall_fscore_support(y_true_arr, d_preds_arr, average='macro', zero_division=0)
                
                l_acc = np.mean(y_true_arr == l_preds_arr)
                _, _, l_f1, _ = precision_recall_fscore_support(y_true_arr, l_preds_arr, average='macro', zero_division=0)
                
                dann_trial_accs.append(d_acc)
                dann_trial_f1s.append(d_f1)
                lgb_trial_accs.append(l_acc)
                lgb_trial_f1s.append(l_f1)
                
            mean_d_acc, std_d_acc = float(np.mean(dann_trial_accs)), float(np.std(dann_trial_accs))
            mean_d_f1, std_d_f1 = float(np.mean(dann_trial_f1s)), float(np.std(dann_trial_f1s))
            mean_l_acc, std_l_acc = float(np.mean(lgb_trial_accs)), float(np.std(lgb_trial_accs))
            mean_l_f1, std_l_f1 = float(np.mean(lgb_trial_f1s)), float(np.std(lgb_trial_f1s))
            
            if k == 0:
                clean_dann_acc = mean_d_acc
                clean_lgb_acc = mean_l_acc
                
            ret_d = (mean_d_acc / clean_dann_acc) * 100
            ret_l = (mean_l_acc / clean_lgb_acc) * 100
            deg_d = (clean_dann_acc - mean_d_acc) * 100
            deg_l = (clean_lgb_acc - mean_l_acc) * 100
            
            robustness_results[proto_name]['channel_dropout']['dann'].append({
                'level': k,
                'mean_accuracy': mean_d_acc, 'std_accuracy': std_d_acc,
                'mean_macro_f1': mean_d_f1, 'std_macro_f1': std_d_f1,
                'retention_pct': ret_d, 'degradation_pct': deg_d
            })
            robustness_results[proto_name]['channel_dropout']['lightgbm'].append({
                'level': k,
                'mean_accuracy': mean_l_acc, 'std_accuracy': std_l_acc,
                'mean_macro_f1': mean_l_f1, 'std_macro_f1': std_l_f1,
                'retention_pct': ret_l, 'degradation_pct': deg_l
            })
            
            csv_rows.append([proto_name, 'channel_dropout', f"k={k}", 'Calibrated DANN', f"{mean_d_acc:.4f}", f"{std_d_acc:.4f}", f"{mean_d_f1:.4f}", f"{std_d_f1:.4f}", f"{ret_d:.2f}%", f"{deg_d:.2f}%"])
            csv_rows.append([proto_name, 'channel_dropout', f"k={k}", 'LightGBM', f"{mean_l_acc:.4f}", f"{std_l_acc:.4f}", f"{mean_l_f1:.4f}", f"{std_l_f1:.4f}", f"{ret_l:.2f}%", f"{deg_l:.2f}%"])
            
            print(f"    k={k:2d} ({k/62*100:4.1f}% failed) | DANN: {mean_d_acc*100:5.2f}% +/- {std_d_acc*100:4.2f}% (Ret: {ret_d:5.1f}%) | LGBM: {mean_l_acc*100:5.2f}% +/- {std_l_acc*100:4.2f}% (Ret: {ret_l:5.1f}%)", flush=True)
            
        # ---------------------------------------------------------------------
        # EXPERIMENT B: GAUSSIAN NOISE INJECTION
        # ---------------------------------------------------------------------
        print(f"\n  [Experiment B: Gaussian Noise] Evaluating sigma in {gaussian_noise_levels} (20 paired trials per level)...", flush=True)
        
        for sigma in gaussian_noise_levels:
            dann_trial_accs, dann_trial_f1s = [], []
            lgb_trial_accs, lgb_trial_f1s = [], []
            
            n_eval_trials = 1 if sigma == 0.0 else num_trials
            
            for t in range(n_eval_trials):
                rng = np.random.RandomState(42 + 5000 * int(sigma * 100) + t)
                
                all_y_true = []
                all_dann_preds = []
                all_lgb_preds = []
                
                for f_entry in fold_data:
                    X_test = f_entry['X_test_scaled'].copy()
                    y_test = f_entry['y_test']
                    
                    if sigma > 0.0:
                        noise = rng.normal(0.0, sigma, size=X_test.shape)
                        X_test = X_test + noise
                        
                    # Batched DANN evaluation (GPU)
                    d_preds = evaluate_dann_on_array(f_entry['dann_model'], X_test, device)
                    # LightGBM evaluation (CPU)
                    l_preds = evaluate_lgbm_on_array(f_entry['lgb_model'], X_test)
                    
                    all_y_true.extend(y_test)
                    all_dann_preds.extend(d_preds)
                    all_lgb_preds.extend(l_preds)
                    
                y_true_arr = np.array(all_y_true)
                d_preds_arr = np.array(all_dann_preds)
                l_preds_arr = np.array(all_lgb_preds)
                
                d_acc = np.mean(y_true_arr == d_preds_arr)
                _, _, d_f1, _ = precision_recall_fscore_support(y_true_arr, d_preds_arr, average='macro', zero_division=0)
                
                l_acc = np.mean(y_true_arr == l_preds_arr)
                _, _, l_f1, _ = precision_recall_fscore_support(y_true_arr, l_preds_arr, average='macro', zero_division=0)
                
                dann_trial_accs.append(d_acc)
                dann_trial_f1s.append(d_f1)
                lgb_trial_accs.append(l_acc)
                lgb_trial_f1s.append(l_f1)
                
            mean_d_acc, std_d_acc = float(np.mean(dann_trial_accs)), float(np.std(dann_trial_accs))
            mean_d_f1, std_d_f1 = float(np.mean(dann_trial_f1s)), float(np.std(dann_trial_f1s))
            mean_l_acc, std_l_acc = float(np.mean(lgb_trial_accs)), float(np.std(lgb_trial_accs))
            mean_l_f1, std_l_f1 = float(np.mean(lgb_trial_f1s)), float(np.std(lgb_trial_f1s))
            
            ret_d = (mean_d_acc / clean_dann_acc) * 100
            ret_l = (mean_l_acc / clean_lgb_acc) * 100
            deg_d = (clean_dann_acc - mean_d_acc) * 100
            deg_l = (clean_lgb_acc - mean_l_acc) * 100
            
            robustness_results[proto_name]['gaussian_noise']['dann'].append({
                'level': sigma,
                'mean_accuracy': mean_d_acc, 'std_accuracy': std_d_acc,
                'mean_macro_f1': mean_d_f1, 'std_macro_f1': std_d_f1,
                'retention_pct': ret_d, 'degradation_pct': deg_d
            })
            robustness_results[proto_name]['gaussian_noise']['lightgbm'].append({
                'level': sigma,
                'mean_accuracy': mean_l_acc, 'std_accuracy': std_l_acc,
                'mean_macro_f1': mean_l_f1, 'std_macro_f1': std_l_f1,
                'retention_pct': ret_l, 'degradation_pct': deg_l
            })
            
            csv_rows.append([proto_name, 'gaussian_noise', f"sigma={sigma:.2f}", 'Calibrated DANN', f"{mean_d_acc:.4f}", f"{std_d_acc:.4f}", f"{mean_d_f1:.4f}", f"{std_d_f1:.4f}", f"{ret_d:.2f}%", f"{deg_d:.2f}%"])
            csv_rows.append([proto_name, 'gaussian_noise', f"sigma={sigma:.2f}", 'LightGBM', f"{mean_l_acc:.4f}", f"{std_l_acc:.4f}", f"{mean_l_f1:.4f}", f"{std_l_f1:.4f}", f"{ret_l:.2f}%", f"{deg_l:.2f}%"])
            
            print(f"    sigma={sigma:4.2f} | DANN: {mean_d_acc*100:5.2f}% +/- {std_d_acc*100:4.2f}% (Ret: {ret_d:5.1f}%) | LGBM: {mean_l_acc*100:5.2f}% +/- {std_l_acc*100:4.2f}% (Ret: {ret_l:5.1f}%)", flush=True)
            
    # 5. Generate 300 DPI Visualizations
    print("\n" + "=" * 80, flush=True)
    print("GENERATING 300 DPI ROBUSTNESS VISUALIZATIONS", flush=True)
    print("=" * 80, flush=True)
    
    fig_drop_path = os.path.join(figures_dir, "dann_vs_lightgbm_channel_dropout_robustness.png")
    plot_channel_dropout_robustness(robustness_results, fig_drop_path)
    
    fig_noise_path = os.path.join(figures_dir, "dann_vs_lightgbm_gaussian_noise_robustness.png")
    plot_gaussian_noise_robustness(robustness_results, fig_noise_path)
    
    fig_heat_path = os.path.join(figures_dir, "dann_robustness_summary_heatmap.png")
    plot_robustness_summary_heatmap(robustness_results, fig_heat_path)
    
    # 6. Save Structured JSON and CSV
    json_path = "dann_robustness_results.json"
    with open(json_path, "w") as f:
        json.dump(robustness_results, f, indent=4)
    print(f"\n[Saved] Complete Robustness JSON to '{json_path}'", flush=True)
    
    csv_path = "dann_robustness_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Protocol", "Corruption_Type", "Corruption_Level", "Model",
            "Mean_Accuracy", "Std_Accuracy", "Mean_Macro_F1", "Std_Macro_F1",
            "Retention_Pct", "Degradation_Pct"
        ])
        for row in csv_rows:
            writer.writerow(row)
    print(f"[Saved] Complete Robustness CSV to '{csv_path}'", flush=True)
    
    print("=" * 80, flush=True)
    print(f"Robustness phase completed successfully in {(time.time() - t_start):.2f} seconds.", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
