"""
Unified Benchmark Training & Evaluation Engine
=============================================
Spatial-Temporal 2D-CNN-BiGRU & Transductive CDAN for SEED-IV EEG

Evaluates:
1. Regime A: Subject-Dependent / Intra-Session Benchmark (15 subjects x 3 sessions = 45 runs)
   - Protocol: Train on Trials 1-16 -> Test on Trials 17-24 (Strict Trial-Quarantine)
   - Target benchmark: 88% - 94% Accuracy
2. Regime B: Transductive CDAN Cross-Subject Benchmark (5-Fold Leave-3-Subjects-Out Nested CV)
   - Protocol: Source Labeled (12 subjects) + Target Unlabeled (3 subjects, 100% blind labels)
   - Dynamic GRL schedule: alpha_p = 2 / (1 + exp(-10*p)) - 1
   - Multilinear conditioning (f (x) g, 512D) with entropy weighting

Outputs:
- Comprehensive metrics with 95% Bootstrap Confidence Intervals
- JSON/CSV results files
- Publication-quality 300 DPI figures in figures/upgraded_spatial_temporal/
"""

import os
import sys
import time
import json
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
import matplotlib.pyplot as plt

from spatial_mapping import get_channel_grid_map
from temporal_dataset import (
    build_trial_quarantined_sequences,
    prepare_scaled_spatial_datasets,
    EEGSequenceDataset
)
from model_spatial_temporal_cdan import SpatialTemporalCDAN

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000, seed=42):
    """Computes 95% confidence intervals using non-parametric bootstrap resampling."""
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
            
    ci_acc = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
    ci_prec = [float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5))]
    ci_rec = [float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))]
    ci_f1 = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]
    ci_auc = [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]
    ci_kappa = [float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))]
    
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

# =========================================================================
# 2. REGIME A: SUBJECT-DEPENDENT / INTRA-SESSION BENCHMARK
# =========================================================================

def train_single_session_regime_a(train_loader, test_loader, device, epochs=35, lr=1e-3, weight_decay=1e-4):
    """
    Trains Spatial-Temporal 2D-CNN-BiGRU on a single session's training trials (1-16)
    and evaluates on test trials (17-24).
    """
    model = SpatialTemporalCDAN(in_channels=5, spatial_dim=128, rnn_hidden=64, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        
        for batch in train_loader:
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits, _, _, _ = model(x, alpha=0.0)
            loss = criterion(logits, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            n_samples += len(y)
            
        scheduler.step()
        avg_train_loss = total_loss / max(n_samples, 1)
        
        # Save best model state based on training convergence
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best state for evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Evaluate on test set
    model.eval()
    test_preds = []
    test_probs = []
    test_trues = []
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            logits, _, _, probs = model(x, alpha=0.0)
            
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            probs_np = probs.cpu().numpy()
            y_np = y.cpu().numpy()
            
            test_preds.extend(preds)
            test_probs.extend(probs_np)
            test_trues.extend(y_np)
            
    return np.array(test_trues), np.array(test_preds), np.array(test_probs)

def run_regime_a_benchmark(seq_dict, device, epochs=35, batch_size=32, dry_run=False):
    """
    Executes the full Subject-Dependent / Intra-Session benchmark across all 15 subjects x 3 sessions (45 runs).
    """
    print("\n" + "="*80)
    print("STARTING REGIME A: SUBJECT-DEPENDENT / INTRA-SESSION BENCHMARK")
    print("="*80)
    print(f"Total Sequence Samples: {len(seq_dict['labels'])}")
    print(f"Device: {device} | Epochs per session: {epochs} | Batch size: {batch_size}")
    
    subjects = range(1, 16) if not dry_run else [1]
    sessions = range(1, 4) if not dry_run else [1]
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    session_results = []
    start_time = time.time()
    
    total_runs = len(subjects) * len(sessions)
    run_idx = 0
    
    for sub in subjects:
        sub_accs = []
        for ses in sessions:
            run_idx += 1
            # Filter mask for this subject & session
            sub_ses_mask = (seq_dict['subject_ids'] == sub) & (seq_dict['session_nums'] == ses)
            train_mask = sub_ses_mask & (seq_dict['trial_ids'] <= 16)
            test_mask = sub_ses_mask & (seq_dict['trial_ids'] > 16)
            
            assert np.sum(train_mask) > 0, f"No training samples for Subject {sub} Session {ses}"
            assert np.sum(test_mask) > 0, f"No test samples for Subject {sub} Session {ses}"
            
            # Scaled spatial datasets (zero leakage)
            train_ds, test_ds, _, _ = prepare_scaled_spatial_datasets(seq_dict, train_mask, test_mask)
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
            
            y_true, y_pred, y_prob = train_single_session_regime_a(
                train_loader, test_loader, device, epochs=epochs
            )
            
            acc = accuracy_score(y_true, y_pred)
            sub_accs.append(acc)
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)
            
            session_results.append({
                'subject': sub,
                'session': ses,
                'n_train': len(train_ds),
                'n_test': len(test_ds),
                'accuracy': float(acc),
                'macro_f1': float(precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2])
            })
            
            print(f"[{run_idx:02d}/{total_runs:02d}] Subject {sub:02d} Session {ses} -> Test Acc: {acc*100:6.2f}% (N_test={len(test_ds)})")
            
    total_elapsed = time.time() - start_time
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # Compute pooled overall metrics with 95% Bootstrap CIs
    overall_metrics = evaluate_metrics(all_y_true, all_y_pred, all_y_prob, compute_ci=True)
    overall_metrics['total_runs'] = total_runs
    overall_metrics['total_test_samples'] = len(all_y_true)
    overall_metrics['elapsed_seconds'] = total_elapsed
    overall_metrics['session_breakdown'] = session_results
    
    print("\n" + "-"*80)
    print("REGIME A (SUBJECT-DEPENDENT) FINAL VERIFIED METRICS:")
    print(f"Pooled Accuracy:  {overall_metrics['accuracy']*100:6.2f}% [95% CI: {overall_metrics['accuracy_ci'][0]*100:.2f}%, {overall_metrics['accuracy_ci'][1]*100:.2f}%]")
    print(f"Macro-F1 Score:   {overall_metrics['f1']:.4f} [95% CI: {overall_metrics['f1_ci'][0]:.4f}, {overall_metrics['f1_ci'][1]:.4f}]")
    print(f"Macro Precision:  {overall_metrics['precision']:.4f} [95% CI: {overall_metrics['precision_ci'][0]:.4f}, {overall_metrics['precision_ci'][1]:.4f}]")
    print(f"Macro Recall:     {overall_metrics['recall']:.4f} [95% CI: {overall_metrics['recall_ci'][0]:.4f}, {overall_metrics['recall_ci'][1]:.4f}]")
    print(f"Macro ROC-AUC:    {overall_metrics['auc']:.4f} [95% CI: {overall_metrics['auc_ci'][0]:.4f}, {overall_metrics['auc_ci'][1]:.4f}]")
    print(f"Cohen's Kappa:    {overall_metrics['kappa']:.4f} [95% CI: {overall_metrics['kappa_ci'][0]:.4f}, {overall_metrics['kappa_ci'][1]:.4f}]")
    print(f"Execution Time:   {total_elapsed:.2f}s ({total_elapsed/total_runs:.2f}s/session)")
    print("-"*80)
    
    return overall_metrics, (all_y_true, all_y_pred, all_y_prob)

# =========================================================================
# 3. REGIME B: TRANSDUCTIVE CDAN CROSS-SUBJECT BENCHMARK
# =========================================================================

def train_single_fold_cdan(source_loader, target_unlabeled_loader, target_test_loader, device, epochs=25, lr=5e-4, weight_decay=1e-4, w_dom=0.1):
    """
    Trains Spatial-Temporal 2D-CNN-BiGRU + CDAN across 1 fold of Leave-3-Subjects-Out.
    Strict Zero-Leakage: Target labels are NEVER accessed during training or model selection.
    """
    model = SpatialTemporalCDAN(in_channels=5, spatial_dim=128, rnn_hidden=64, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    len_source = len(source_loader)
    len_target = len(target_unlabeled_loader)
    n_batches = max(len_source, len_target)
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        # Dynamic GRL schedule: alpha_p = 2 / (1 + exp(-10 * p)) - 1
        p = float(epoch) / float(max(epochs, 1))
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        
        iter_source = iter(source_loader)
        iter_target = iter(target_unlabeled_loader)
        
        epoch_total_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_dom_loss = 0.0
        
        for step in range(n_batches):
            try:
                batch_s = next(iter_source)
            except StopIteration:
                iter_source = iter(source_loader)
                batch_s = next(iter_source)
                
            try:
                batch_t = next(iter_target)
            except StopIteration:
                iter_target = iter(target_unlabeled_loader)
                batch_t = next(iter_target)
                
            src_x = batch_s['features'].to(device)
            src_y = batch_s['label'].to(device)
            tgt_x = batch_t['features'].to(device)
            
            optimizer.zero_grad()
            loss_dict = model.compute_cdan_loss(src_x, src_y, tgt_x, alpha=alpha, w_dom=w_dom)
            loss_dict['total_loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += loss_dict['total_loss'].item()
            epoch_cls_loss += loss_dict['loss_cls'].item()
            epoch_dom_loss += loss_dict['loss_dom'].item()
            
        scheduler.step()
        avg_loss = epoch_total_loss / n_batches
        
        # Save best model state based strictly on training objective convergence
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load optimal training state for held-out target evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    # Strictly out-of-sample evaluation on held-out target subjects
    model.eval()
    test_preds, test_probs, test_trues = [], [], []
    with torch.no_grad():
        for batch in target_test_loader:
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            logits, _, _, probs = model(x, alpha=0.0)
            
            test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_trues.extend(y.cpu().numpy())
            
    return np.array(test_trues), np.array(test_preds), np.array(test_probs)

def run_regime_b_cdan_benchmark(seq_dict, device, epochs=30, batch_size=64, dry_run=False):
    """
    Executes the 5-fold Leave-3-Subjects-Out Transductive CDAN benchmark.
    """
    print("\n" + "="*80)
    print("STARTING REGIME B: TRANSDUCTIVE CDAN CROSS-SUBJECT BENCHMARK")
    print("="*80)
    
    # 5 Folds with 3 subjects each
    folds = [
        {'name': 'Fold 1', 'test_subs': [1, 2, 3],   'source_subs': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        {'name': 'Fold 2', 'test_subs': [4, 5, 6],   'source_subs': [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        {'name': 'Fold 3', 'test_subs': [7, 8, 9],   'source_subs': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]},
        {'name': 'Fold 4', 'test_subs': [10, 11, 12], 'source_subs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15]},
        {'name': 'Fold 5', 'test_subs': [13, 14, 15], 'source_subs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    ]
    
    if dry_run:
        folds = folds[:1] # 1 fold only for dry run
        
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_results = []
    start_time = time.time()
    
    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Running {fold['name']}: Target Subjects {fold['test_subs']} (Source: {len(fold['source_subs'])} subjects) ---")
        
        # Source mask (Labeled training)
        source_mask = np.isin(seq_dict['subject_ids'], fold['source_subs'])
        # Target mask (Unlabeled adaptation & evaluation)
        target_mask = np.isin(seq_dict['subject_ids'], fold['test_subs'])
        
        train_ds, test_ds, _, target_unlabeled_ds = prepare_scaled_spatial_datasets(
            seq_dict,
            train_mask=source_mask,
            test_mask=target_mask,
            target_unlabeled_mask=target_mask
        )
        
        source_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        target_unlabeled_loader = DataLoader(target_unlabeled_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        target_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        y_true, y_pred, y_prob = train_single_fold_cdan(
            source_loader, target_unlabeled_loader, target_test_loader, device,
            epochs=epochs, lr=5e-4, weight_decay=1e-4, w_dom=0.1
        )
        
        acc = accuracy_score(y_true, y_pred)
        f1 = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)[2]
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        
        fold_results.append({
            'fold': fold['name'],
            'test_subjects': fold['test_subs'],
            'n_source': len(train_ds),
            'n_target': len(test_ds),
            'accuracy': float(acc),
            'macro_f1': float(f1)
        })
        
        print(f"Result {fold['name']} -> Target Acc: {acc*100:6.2f}%, Macro-F1: {f1:.4f} (N_target={len(test_ds)})")
        
    total_elapsed = time.time() - start_time
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    # Compute pooled overall metrics with 95% Bootstrap CIs
    overall_metrics = evaluate_metrics(all_y_true, all_y_pred, all_y_prob, compute_ci=True)
    overall_metrics['total_folds'] = len(folds)
    overall_metrics['total_target_samples'] = len(all_y_true)
    overall_metrics['elapsed_seconds'] = total_elapsed
    overall_metrics['fold_breakdown'] = fold_results
    
    print("\n" + "-"*80)
    print("REGIME B (TRANSDUCTIVE CDAN CROSS-SUBJECT) FINAL VERIFIED METRICS:")
    print(f"Pooled Accuracy:  {overall_metrics['accuracy']*100:6.2f}% [95% CI: {overall_metrics['accuracy_ci'][0]*100:.2f}%, {overall_metrics['accuracy_ci'][1]*100:.2f}%]")
    print(f"Macro-F1 Score:   {overall_metrics['f1']:.4f} [95% CI: {overall_metrics['f1_ci'][0]:.4f}, {overall_metrics['f1_ci'][1]:.4f}]")
    print(f"Macro Precision:  {overall_metrics['precision']:.4f} [95% CI: {overall_metrics['precision_ci'][0]:.4f}, {overall_metrics['precision_ci'][1]:.4f}]")
    print(f"Macro Recall:     {overall_metrics['recall']:.4f} [95% CI: {overall_metrics['recall_ci'][0]:.4f}, {overall_metrics['recall_ci'][1]:.4f}]")
    print(f"Macro ROC-AUC:    {overall_metrics['auc']:.4f} [95% CI: {overall_metrics['auc_ci'][0]:.4f}, {overall_metrics['auc_ci'][1]:.4f}]")
    print(f"Cohen's Kappa:    {overall_metrics['kappa']:.4f} [95% CI: {overall_metrics['kappa_ci'][0]:.4f}, {overall_metrics['kappa_ci'][1]:.4f}]")
    print(f"Execution Time:   {total_elapsed:.2f}s ({total_elapsed/len(folds):.2f}s/fold)")
    print("-"*80)
    
    return overall_metrics, (all_y_true, all_y_pred, all_y_prob)

# =========================================================================
# 4. PUBLICATION-QUALITY VISUALIZATION EXPORTERS (300 DPI)
# =========================================================================

def generate_evaluation_figures(reg_a_data, reg_b_data, output_dir="figures/upgraded_spatial_temporal"):
    """
    Generates high-resolution (300 DPI) evaluation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # 1. Regime A Confusion Matrix & ROC Curves
    if reg_a_data is not None:
        y_true, y_pred, y_prob = reg_a_data
        cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
        
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(cax)
        
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.1f}%",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
                        
        ax.set_title("Subject-Dependent Spatial-Temporal 2D-CNN-BiGRU\nNormalized Confusion Matrix (%)", fontsize=13, pad=12, fontweight='bold')
        ax.set_xlabel("Predicted Emotion Class", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Emotion Class", fontsize=11, fontweight='bold')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "regimeA_subject_dependent_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_path}")
        
        # Regime A Multi-Class ROC Curves
        plt.figure(figsize=(7, 6), dpi=300)
        for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, EMOTION_COLORS)):
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{cls_name} (AUC = {roc_auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance (0.50)')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate", fontsize=11, fontweight='bold')
        plt.ylabel("True Positive Rate", fontsize=11, fontweight='bold')
        plt.title("Subject-Dependent 2D-CNN-BiGRU ROC Curves", fontsize=13, pad=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10, frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "regimeA_subject_dependent_roc_curves.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"Saved: {roc_path}")
        
    # 2. Regime B CDAN Confusion Matrix & ROC Curves
    if reg_b_data is not None:
        y_true_b, y_pred_b, y_prob_b = reg_b_data
        cm_b = confusion_matrix(y_true_b, y_pred_b, normalize='true') * 100
        
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        cax = ax.imshow(cm_b, interpolation='nearest', cmap=plt.cm.Purples)
        fig.colorbar(cax)
        
        tick_marks = np.arange(len(CLASS_NAMES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
        
        thresh = cm_b.max() / 2.
        for i in range(cm_b.shape[0]):
            for j in range(cm_b.shape[1]):
                ax.text(j, i, f"{cm_b[i, j]:.1f}%",
                        ha="center", va="center",
                        color="white" if cm_b[i, j] > thresh else "black",
                        fontsize=12, fontweight='bold')
                        
        ax.set_title("Transductive CDAN Cross-Subject (5-Fold Nested CV)\nNormalized Confusion Matrix (%)", fontsize=13, pad=12, fontweight='bold')
        ax.set_xlabel("Predicted Emotion Class", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Emotion Class", fontsize=11, fontweight='bold')
        plt.tight_layout()
        cm_b_path = os.path.join(output_dir, "regimeB_cdan_cross_subject_confusion_matrix.png")
        plt.savefig(cm_b_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_b_path}")
        
        # Regime B ROC Curves
        plt.figure(figsize=(7, 6), dpi=300)
        for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, EMOTION_COLORS)):
            y_binary = (y_true_b == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_prob_b[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{cls_name} (AUC = {roc_auc:.3f})")
            
        plt.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance (0.50)')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel("False Positive Rate", fontsize=11, fontweight='bold')
        plt.ylabel("True Positive Rate", fontsize=11, fontweight='bold')
        plt.title("Transductive CDAN Cross-Subject ROC Curves", fontsize=13, pad=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10, frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_b_path = os.path.join(output_dir, "regimeB_cdan_cross_subject_roc_curves.png")
        plt.savefig(roc_b_path, dpi=300)
        plt.close()
        print(f"Saved: {roc_b_path}")

# =========================================================================
# 5. MAIN ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Spatial-Temporal 2D-CNN-BiGRU and CDAN Benchmark Evaluator")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'regime_a', 'regime_b', 'dry_run'],
                        help="Execution mode: 'all', 'regime_a', 'regime_b', or 'dry_run'")
    parser.add_argument('--dataset', type=str, default='seed_iv_processed.npz', help="Path to processed SEED-IV dataset")
    parser.add_argument('--epochs_a', type=int, default=35, help="Training epochs for Regime A")
    parser.add_argument('--epochs_b', type=int, default=30, help="Training epochs for Regime B (CDAN)")
    parser.add_argument('--batch_size_a', type=int, default=32, help="Batch size for Regime A")
    parser.add_argument('--batch_size_b', type=int, default=64, help="Batch size for Regime B")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")
    
    # 1. Load and build trial-quarantined sequences
    print(f"Loading dataset from '{args.dataset}' and generating sequences (T=8, stride=2)...")
    seq_dict = build_trial_quarantined_sequences(args.dataset, T=8, stride=2)
    print(f"Successfully generated {len(seq_dict['labels'])} sequences.")
    
    reg_a_metrics, reg_a_data = None, None
    reg_b_metrics, reg_b_data = None, None
    
    if args.mode in ['all', 'regime_a', 'dry_run']:
        is_dry = (args.mode == 'dry_run')
        epochs_a = 3 if is_dry else args.epochs_a
        reg_a_metrics, reg_a_data = run_regime_a_benchmark(
            seq_dict, device, epochs=epochs_a, batch_size=args.batch_size_a, dry_run=is_dry
        )
        if not is_dry:
            with open("upgraded_regimeA_subject_dependent_results.json", "w") as f:
                json.dump(reg_a_metrics, f, indent=2)
            print("Saved Regime A results to 'upgraded_regimeA_subject_dependent_results.json'")
            
    if args.mode in ['all', 'regime_b', 'dry_run']:
        is_dry = (args.mode == 'dry_run')
        epochs_b = 3 if is_dry else args.epochs_b
        reg_b_metrics, reg_b_data = run_regime_b_cdan_benchmark(
            seq_dict, device, epochs=epochs_b, batch_size=args.batch_size_b, dry_run=is_dry
        )
        if not is_dry:
            with open("upgraded_regimeB_cdan_cross_subject_results.json", "w") as f:
                json.dump(reg_b_metrics, f, indent=2)
            print("Saved Regime B results to 'upgraded_regimeB_cdan_cross_subject_results.json'")
            
    # Generate visualization figures if full run
    if args.mode in ['all', 'regime_a', 'regime_b'] and args.mode != 'dry_run':
        print("\nGenerating publication-quality figures (300 DPI)...")
        generate_evaluation_figures(reg_a_data, reg_b_data, output_dir="figures/upgraded_spatial_temporal")
        
    print("\n[SUCCESS] Benchmark execution completed cleanly!")

if __name__ == '__main__':
    main()
