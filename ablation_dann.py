import os
import time
import copy
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
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

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

def reset_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================================
# GRADIENT REVERSAL LAYER & INDUCTIVE DANN ARCHITECTURE
# =========================================================================

class GradientReversalFunction(Function):
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
        
    def forward(self, x, alpha=1.0):
        self.grl.alpha = alpha
        feat = self.feature_extractor(x)
        class_out = self.class_classifier(feat)
        domain_feat = self.grl(feat)
        domain_out = self.domain_classifier(domain_feat)
        return class_out, domain_out

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

# =========================================================================
# SINGLE-FOLD TRAINING HELPER
# =========================================================================

def train_and_evaluate_fold(X_train_raw, y_train, d_train, X_val_raw, y_val, X_test_raw, y_test, 
                            n_domains, device, max_epochs=100, patience=15, batch_size=128,
                            w_dom=1.0, use_ganin_schedule=True, fixed_lambda=1.0):
    """
    Trains Inductive DANN with parameterized domain weight and GRL lambda schedule.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                      torch.tensor(y_train, dtype=torch.long),
                      torch.tensor(d_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), 
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    
    model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=n_domains, dropout=0.2).to(device)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    
    best_val_acc = -1.0
    best_epoch = -1
    best_weights = None
    epochs_no_improve = 0
    epoch_times = []
    
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        
        # Schedule determination
        if use_ganin_schedule:
            p = float(epoch - 1) / max_epochs
            lambda_p = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        else:
            lambda_p = float(fixed_lambda)
            
        # Training
        model.train()
        train_class_loss, train_class_correct = 0.0, 0
        train_domain_loss, train_domain_correct = 0.0, 0
        train_total = 0
        
        for bx, by, bd in train_loader:
            bx, by, bd = bx.to(device), by.to(device), bd.to(device)
            optimizer.zero_grad()
            
            c_out, d_out = model(bx, alpha=lambda_p)
            loss_c = criterion_class(c_out, by)
            
            if w_dom > 0.0:
                loss_d = criterion_domain(d_out, bd)
                total_loss = loss_c + w_dom * loss_d
            else:
                loss_d = torch.tensor(0.0, device=device)
                total_loss = loss_c
                
            total_loss.backward()
            optimizer.step()
            
            n_b = len(by)
            train_total += n_b
            train_class_loss += loss_c.item() * n_b
            train_domain_loss += loss_d.item() * n_b
            train_class_correct += (c_out.argmax(dim=1) == by).sum().item()
            train_domain_correct += (d_out.argmax(dim=1) == bd).sum().item()
            
        scheduler.step()
        
        tr_c_loss = train_class_loss / train_total
        tr_c_acc = train_class_correct / train_total
        tr_d_loss = train_domain_loss / train_total
        tr_d_acc = train_domain_correct / train_total
        
        # Validation (Emotion task only on validation set)
        model.eval()
        val_class_loss, val_class_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                c_out, _ = model(bx, alpha=0.0)
                loss_c = criterion_class(c_out, by)
                val_class_loss += loss_c.item() * len(by)
                val_class_correct += (c_out.argmax(dim=1) == by).sum().item()
                val_total += len(by)
                
        val_c_loss = val_class_loss / val_total
        val_c_acc = val_class_correct / val_total
        
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)
        
        if val_c_acc > best_val_acc:
            best_val_acc = val_c_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            mark = '*'
        else:
            epochs_no_improve += 1
            mark = ''
            
        if epoch % 5 == 0 or epoch == 1 or mark == '*':
            tr_c_str = f"{tr_c_loss:.3f}/{tr_c_acc*100:.1f}%"
            tr_d_str = f"{tr_d_loss:.3f}/{tr_d_acc*100:.1f}%"
            val_c_str = f"{val_c_loss:.3f}/{val_c_acc*100:.1f}% {mark}"
            print(f"    Epoch [{epoch:03d}/{max_epochs:03d}] ({t_epoch:.2f}s) | Lam: {lambda_p:.3f} | Train Emo: {tr_c_str} | Dom Discrim: {tr_d_str} | Val Emo: {val_c_str}", flush=True)
            
        if epochs_no_improve >= patience:
            print(f"    --> Early stopped at epoch {epoch} (Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch})", flush=True)
            break
            
    # Evaluate best model on test set
    model.load_state_dict(best_weights)
    model.eval()
    
    test_preds, test_probs, test_targets = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            c_out, _ = model(bx, alpha=0.0)
            probs = torch.softmax(c_out, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(by.numpy())
            
    y_test_arr = np.array(test_targets)
    y_pred_arr = np.array(test_preds)
    y_prob_arr = np.array(test_probs)
    
    acc = np.mean(y_test_arr == y_pred_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_arr, y_pred_arr, average='macro', zero_division=0)
    auc_val = roc_auc_score(y_test_arr, y_prob_arr, average='macro', multi_class='ovr')
    kappa_val = cohen_kappa_score(y_test_arr, y_pred_arr)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc_val),
        'kappa': float(kappa_val),
        'best_epoch': int(best_epoch),
        'best_val_acc': float(best_val_acc),
        'avg_epoch_time': float(np.mean(epoch_times)),
        'y_true': y_test_arr,
        'y_pred': y_pred_arr,
        'y_prob': y_prob_arr
    }

# =========================================================================
# PROTOCOL RUNNERS
# =========================================================================

def run_cross_subject(features_flat, labels, subject_ids, device, w_dom=1.0, use_ganin_schedule=True, fixed_lambda=1.0):
    """Cross-subject protocol: 5-fold Leave-3-Subjects-Out nested CV with 10 source domains."""
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
        train_pool_subs = [s for s in range(1, 16) if s not in test_subs]
        val_subs = train_pool_subs[-2:]
        train_subs = train_pool_subs[:-2]
        
        print(f"\n--- Outer Fold {fold+1}/5 (Test Subjects: {test_subs}, Val Subjects: {val_subs}, Train Subjects: {train_subs}) ---", flush=True)
        
        assert len(set(train_subs).intersection(set(val_subs))) == 0
        assert len(set(train_subs).intersection(set(test_subs))) == 0
        assert len(set(val_subs).intersection(set(test_subs))) == 0
        
        train_mask = np.isin(subject_ids, train_subs)
        val_mask = np.isin(subject_ids, val_subs)
        test_mask = np.isin(subject_ids, test_subs)
        
        X_train = features_flat[train_mask]
        y_train = labels[train_mask]
        s_train = subject_ids[train_mask]
        
        sub_to_domain = {sub: idx for idx, sub in enumerate(train_subs)}
        d_train = np.array([sub_to_domain[s] for s in s_train])
        
        X_val = features_flat[val_mask]
        y_val = labels[val_mask]
        
        X_test = features_flat[test_mask]
        y_test = labels[test_mask]
        
        print(f"  Fold {fold+1}: Train={len(y_train)} samples, Val={len(y_val)} samples, Test={len(y_test)} samples.", flush=True)
        
        res = train_and_evaluate_fold(
            X_train, y_train, d_train, X_val, y_val, X_test, y_test, len(train_subs), device,
            w_dom=w_dom, use_ganin_schedule=use_ganin_schedule, fixed_lambda=fixed_lambda
        )
        fold_time = time.time() - t_fold_start
        
        print(f"  >>> Fold {fold+1} Complete ({fold_time:.1f}s) | Test Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | AUC: {res['auc']:.4f} | Kappa: {res['kappa']:.4f}", flush=True)
        
        y_true_all.extend(res['y_true'])
        y_pred_all.extend(res['y_pred'])
        y_prob_all.extend(res['y_prob'])
        
        fold_metrics_list.append({
            'fold': fold + 1,
            'test_subjects': test_subs,
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
            'kappa': res['kappa'],
            'best_epoch': res['best_epoch'],
            'best_val_acc': res['best_val_acc'],
            'fold_time_sec': fold_time
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    return {
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
        "fold_metrics": fold_metrics_list,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all
    }

def run_subject_dependent(features_flat, labels, subject_ids, session_nums, trial_ids, device, w_dom=1.0, use_ganin_schedule=True, fixed_lambda=1.0):
    """Subject-dependent protocol: pooled data, 5-fold nested CV grouped by trial with 15 subject domains."""
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
        
        outer_train_trials = unique_trial_keys[train_trial_idx]
        outer_test_trials = unique_trial_keys[test_trial_idx]
        
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(kf_inner.split(np.arange(len(outer_train_trials))))
        
        train_trials_set = set(outer_train_trials[inner_train_idx])
        val_trials_set = set(outer_train_trials[inner_val_idx])
        test_trials_set = set(outer_test_trials)
        
        assert len(train_trials_set.intersection(val_trials_set)) == 0
        assert len(train_trials_set.intersection(test_trials_set)) == 0
        assert len(val_trials_set.intersection(test_trials_set)) == 0
        
        train_mask = np.isin(trial_keys, list(train_trials_set))
        val_mask = np.isin(trial_keys, list(val_trials_set))
        test_mask = np.isin(trial_keys, list(test_trials_set))
        
        X_train = features_flat[train_mask]
        y_train = labels[train_mask]
        s_train = subject_ids[train_mask]
        d_train = s_train - 1
        
        X_val = features_flat[val_mask]
        y_val = labels[val_mask]
        X_test = features_flat[test_mask]
        y_test = labels[test_mask]
        
        print(f"  Fold {fold+1}: Train={len(y_train)} samples, Val={len(y_val)} samples, Test={len(y_test)} samples.", flush=True)
        
        res = train_and_evaluate_fold(
            X_train, y_train, d_train, X_val, y_val, X_test, y_test, 15, device,
            w_dom=w_dom, use_ganin_schedule=use_ganin_schedule, fixed_lambda=fixed_lambda
        )
        fold_time = time.time() - t_fold_start
        
        print(f"  >>> Fold {fold+1} Complete ({fold_time:.1f}s) | Test Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | AUC: {res['auc']:.4f} | Kappa: {res['kappa']:.4f}", flush=True)
        
        y_true_all.extend(res['y_true'])
        y_pred_all.extend(res['y_pred'])
        y_prob_all.extend(res['y_prob'])
        
        fold_metrics_list.append({
            'fold': fold + 1,
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
            'kappa': res['kappa'],
            'best_epoch': res['best_epoch'],
            'best_val_acc': res['best_val_acc'],
            'fold_time_sec': fold_time
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    return {
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
        "fold_metrics": fold_metrics_list,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_prob": y_prob_all
    }

# =========================================================================
# PLOTTING AND VISUALIZATION HELPERS
# =========================================================================

def generate_ablation_plots(ablation_data, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Comparative Bar Charts (Accuracy and Macro-F1 across all 5 configurations)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    config_keys = list(ablation_data.keys())
    config_labels = [ablation_data[k]['name'] for k in config_keys]
    
    for idx, (proto_key, proto_title) in enumerate([("subject_dependent", "Subject-Dependent Protocol"), ("cross_subject", "Cross-Subject Protocol")]):
        ax = axes[idx]
        x = np.arange(len(config_keys))
        width = 0.35
        
        accs = [ablation_data[k][proto_key]['accuracy'] * 100 for k in config_keys]
        acc_cis_low = [ablation_data[k][proto_key]['accuracy_ci'][0] * 100 for k in config_keys]
        acc_cis_high = [ablation_data[k][proto_key]['accuracy_ci'][1] * 100 for k in config_keys]
        acc_err = [np.array(accs) - np.array(acc_cis_low), np.array(acc_cis_high) - np.array(accs)]
        
        f1s = [ablation_data[k][proto_key]['f1'] * 100 for k in config_keys]
        f1_cis_low = [ablation_data[k][proto_key]['f1_ci'][0] * 100 for k in config_keys]
        f1_cis_high = [ablation_data[k][proto_key]['f1_ci'][1] * 100 for k in config_keys]
        f1_err = [np.array(f1s) - np.array(f1_cis_low), np.array(f1_cis_high) - np.array(f1s)]
        
        rects1 = ax.bar(x - width/2, accs, width, yerr=acc_err, capsize=4, label='Accuracy (%)', color='#2b5c8f', alpha=0.9, edgecolor='black')
        rects2 = ax.bar(x + width/2, f1s, width, yerr=f1_err, capsize=4, label='Macro-F1 (%)', color='#27ae60', alpha=0.9, edgecolor='black')
        
        ax.set_title(proto_title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=25, ha='right', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', frameon=True)
        ax.set_ylabel('Score (%)', fontweight='bold', fontsize=11)
        
        for r in rects1:
            h = r.get_height()
            ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        for r in rects2:
            h = r.get_height()
            ax.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    plt.suptitle('Inductive DANN Ablation Study: Performance Comparison', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    chart_path = os.path.join(figures_dir, "dann_ablation_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparative bar chart to '{chart_path}'", flush=True)
    
    # 2. Domain Loss Weight Sensitivity Curve (w_dom = 0.0, 0.1, 0.5, 1.0)
    sensitivity_keys = [
        ("ablation_no_domain", 0.0),
        ("ablation_reduced_w_dom_0.1", 0.1),
        ("ablation_moderate_w_dom_0.5", 0.5),
        ("baseline_reference", 1.0)
    ]
    w_vals = [k[1] for k in sensitivity_keys]
    
    plt.figure(figsize=(9, 5))
    
    sd_accs = [ablation_data[k[0]]['subject_dependent']['accuracy'] * 100 for k in sensitivity_keys]
    cs_accs = [ablation_data[k[0]]['cross_subject']['accuracy'] * 100 for k in sensitivity_keys]
    
    sd_f1s = [ablation_data[k[0]]['subject_dependent']['f1'] * 100 for k in sensitivity_keys]
    cs_f1s = [ablation_data[k[0]]['cross_subject']['f1'] * 100 for k in sensitivity_keys]
    
    plt.plot(w_vals, sd_accs, 'o-', label='Subject-Dependent Acc (%)', color='#2980b9', lw=2.5, markersize=8)
    plt.plot(w_vals, sd_f1s, 's--', label='Subject-Dependent Macro-F1 (%)', color='#3498db', lw=2, markersize=7)
    plt.plot(w_vals, cs_accs, 'o-', label='Cross-Subject Acc (%)', color='#c0392b', lw=2.5, markersize=8)
    plt.plot(w_vals, cs_f1s, 's--', label='Cross-Subject Macro-F1 (%)', color='#e74c3c', lw=2, markersize=7)
    
    for x_val, y_val in zip(w_vals, sd_accs):
        plt.annotate(f'{y_val:.2f}%', (x_val, y_val), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9, fontweight='bold', color='#2980b9')
    for x_val, y_val in zip(w_vals, cs_accs):
        plt.annotate(f'{y_val:.2f}%', (x_val, y_val), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9, fontweight='bold', color='#c0392b')
        
    plt.xlabel(r'Domain Adversarial Loss Weight ($w_{\mathrm{dom}}$)', fontweight='bold', fontsize=11)
    plt.ylabel('Score (%)', fontweight='bold', fontsize=11)
    plt.title('Inductive DANN: Domain Adversarial Weight Sensitivity Analysis', fontsize=13, fontweight='bold')
    plt.xticks(w_vals, [r'$w_{\mathrm{dom}}=0.0$ (None)', r'$w_{\mathrm{dom}}=0.1$', r'$w_{\mathrm{dom}}=0.5$', r'$w_{\mathrm{dom}}=1.0$ (Baseline)'])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, loc='center right')
    plt.tight_layout()
    sens_path = os.path.join(figures_dir, "dann_ablation_loss_weight_sensitivity.png")
    plt.savefig(sens_path, dpi=300)
    plt.close()
    print(f"Saved sensitivity curve to '{sens_path}'", flush=True)

# =========================================================================
# MAIN ENTRYPOINT
# =========================================================================

def main():
    t_pipeline_start = time.time()
    figures_dir = os.path.join("figures", "ablations")
    os.makedirs(figures_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("INDUCTIVE DANN COMPREHENSIVE ABLATION STUDY (GPU)", flush=True)
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
    
    features_flat = features.reshape(num_samples, -1)
    print(f"Dataset loaded: {num_samples} samples across 15 subjects (Input features: {features_flat.shape[1]}).", flush=True)
    
    # Define the 5 ablation configurations
    ablation_configs = [
        {
            "id": "baseline_reference",
            "name": "Full DANN (Baseline)",
            "w_dom": 1.0,
            "use_ganin_schedule": True,
            "fixed_lambda": 1.0,
            "desc": "Original Validated Baseline (w_dom=1.0, Ganin dynamic schedule)"
        },
        {
            "id": "ablation_no_domain",
            "name": "No Domain Adv (w_dom=0)",
            "w_dom": 0.0,
            "use_ganin_schedule": True,
            "fixed_lambda": 1.0,
            "desc": "No Domain Adversarial component (Pure Emotion Classification MLP)"
        },
        {
            "id": "ablation_fixed_lambda",
            "name": "Fixed Lambda (lambda=1.0)",
            "w_dom": 1.0,
            "use_ganin_schedule": False,
            "fixed_lambda": 1.0,
            "desc": "Constant Lambda=1.0 (No Ganin progressive schedule)"
        },
        {
            "id": "ablation_reduced_w_dom_0.1",
            "name": "Reduced Weight (w_dom=0.1)",
            "w_dom": 0.1,
            "use_ganin_schedule": True,
            "fixed_lambda": 1.0,
            "desc": "Reduced Domain Loss Weight (0.1x of baseline)"
        },
        {
            "id": "ablation_moderate_w_dom_0.5",
            "name": "Moderate Weight (w_dom=0.5)",
            "w_dom": 0.5,
            "use_ganin_schedule": True,
            "fixed_lambda": 1.0,
            "desc": "Moderate Domain Loss Weight (0.5x of baseline)"
        }
    ]
    
    all_ablation_results = {}
    
    for cfg_idx, cfg in enumerate(ablation_configs):
        print("\n" + "#" * 80, flush=True)
        print(f"[{cfg_idx+1}/{len(ablation_configs)}] RUNNING ABLATION CONFIGURATION: {cfg['name']}", flush=True)
        print(f"Description: {cfg['desc']}", flush=True)
        print(f"Parameters: w_dom={cfg['w_dom']}, use_ganin_schedule={cfg['use_ganin_schedule']}, fixed_lambda={cfg['fixed_lambda']}", flush=True)
        print("#" * 80, flush=True)
        
        # Reset seeds before each configuration to ensure identical initial conditions
        reset_seeds(42)
        
        t_cfg_start = time.time()
        
        # 1. Run Cross-Subject Protocol
        print(f"\n>>> Running Cross-Subject Protocol for {cfg['name']}...", flush=True)
        res_cross = run_cross_subject(
            features_flat, labels, subject_ids, device,
            w_dom=cfg['w_dom'], use_ganin_schedule=cfg['use_ganin_schedule'], fixed_lambda=cfg['fixed_lambda']
        )
        
        # 2. Run Subject-Dependent Protocol
        print(f"\n>>> Running Subject-Dependent Protocol for {cfg['name']}...", flush=True)
        res_dep = run_subject_dependent(
            features_flat, labels, subject_ids, session_nums, trial_ids, device,
            w_dom=cfg['w_dom'], use_ganin_schedule=cfg['use_ganin_schedule'], fixed_lambda=cfg['fixed_lambda']
        )
        
        cfg_time = time.time() - t_cfg_start
        print(f"\n=== Configuration '{cfg['name']}' Finished in {cfg_time/60:.2f} min ===", flush=True)
        print(f"  Cross-Subject:      Acc = {res_cross['accuracy']*100:.2f}% | Macro-F1 = {res_cross['f1']*100:.2f}% | Kappa = {res_cross['kappa']:.4f}", flush=True)
        print(f"  Subject-Dependent:  Acc = {res_dep['accuracy']*100:.2f}% | Macro-F1 = {res_dep['f1']*100:.2f}% | Kappa = {res_dep['kappa']:.4f}", flush=True)
        
        # Strip large raw arrays from JSON payload to keep it clean, but preserve summary metrics
        cross_summary = {k: v for k, v in res_cross.items() if k not in ['y_true', 'y_pred', 'y_prob']}
        dep_summary = {k: v for k, v in res_dep.items() if k not in ['y_true', 'y_pred', 'y_prob']}
        
        all_ablation_results[cfg['id']] = {
            "id": cfg['id'],
            "name": cfg['name'],
            "desc": cfg['desc'],
            "hyperparameters": {
                "w_dom": cfg['w_dom'],
                "use_ganin_schedule": cfg['use_ganin_schedule'],
                "fixed_lambda": cfg['fixed_lambda']
            },
            "runtime_sec": cfg_time,
            "cross_subject": cross_summary,
            "subject_dependent": dep_summary
        }
        
        # Checkpoint incremental results to disk after every completed ablation configuration
        with open("dann_ablation_results.json", "w") as f:
            json.dump(all_ablation_results, f, indent=4)
            
    # Save structured CSV
    csv_path = "dann_ablation_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Ablation_ID", "Configuration", "Protocol", "w_dom", "Schedule", 
            "Accuracy", "Accuracy_95CI_Lower", "Accuracy_95CI_Upper",
            "Macro_Precision", "Macro_Precision_95CI_Lower", "Macro_Precision_95CI_Upper",
            "Macro_Recall", "Macro_Recall_95CI_Lower", "Macro_Recall_95CI_Upper",
            "Macro_F1", "Macro_F1_95CI_Lower", "Macro_F1_95CI_Upper",
            "Macro_AUC", "Macro_AUC_95CI_Lower", "Macro_AUC_95CI_Upper",
            "Cohen_Kappa", "Cohen_Kappa_95CI_Lower", "Cohen_Kappa_95CI_Upper"
        ])
        for cfg_id, r_data in all_ablation_results.items():
            for p_key in ["cross_subject", "subject_dependent"]:
                p_res = r_data[p_key]
                sched_str = "Ganin Dynamic" if r_data["hyperparameters"]["use_ganin_schedule"] else f"Fixed {r_data['hyperparameters']['fixed_lambda']}"
                writer.writerow([
                    cfg_id, r_data["name"], p_res["protocol"], r_data["hyperparameters"]["w_dom"], sched_str,
                    f"{p_res['accuracy']:.4f}", f"{p_res['accuracy_ci'][0]:.4f}", f"{p_res['accuracy_ci'][1]:.4f}",
                    f"{p_res['precision']:.4f}", f"{p_res['precision_ci'][0]:.4f}", f"{p_res['precision_ci'][1]:.4f}",
                    f"{p_res['recall']:.4f}", f"{p_res['recall_ci'][0]:.4f}", f"{p_res['recall_ci'][1]:.4f}",
                    f"{p_res['f1']:.4f}", f"{p_res['f1_ci'][0]:.4f}", f"{p_res['f1_ci'][1]:.4f}",
                    f"{p_res['auc']:.4f}", f"{p_res['auc_ci'][0]:.4f}", f"{p_res['auc_ci'][1]:.4f}",
                    f"{p_res['kappa']:.4f}", f"{p_res['kappa_ci'][0]:.4f}", f"{p_res['kappa_ci'][1]:.4f}"
                ])
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)
    
    # Generate publication figures
    generate_ablation_plots(all_ablation_results, figures_dir)
    
    total_time = (time.time() - t_pipeline_start) / 60
    print("\n" + "=" * 80, flush=True)
    print(f"ALL 5 ABLATION CONFIGURATIONS COMPLETE in {total_time:.2f} minutes.", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    main()
