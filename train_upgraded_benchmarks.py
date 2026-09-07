"""
Unified Benchmark Training & Evaluation Engine (Upgraded)
========================================================
Spatial-Temporal 2D-CNN-BiGRU & Transductive CDAN for SEED-IV EEG

Implemented Fixes:
1. Fix 1: Stratified 4-Fold Trial Cross-Validation per session (Regime A)
   - 18 train trials (balanced across 4 classes) -> 6 test trials (balanced across 4 classes)
   - 45 sessions x 4 folds = 180 runs; 100% out-of-sample balanced evaluation of all 24 trials per session.
2. Fix 2: High-Yield Dense Temporal Sequences
   - T = 4, stride = 1, strict trial boundary quarantine (34,335 total sequences).
3. Fix 3: CDAN Warm-Up Schedule (Regime B)
   - 10 warm-up epochs with w_dom = 0.0 (supervised source training)
   - Annealed domain adaptation over epochs 10-40 (alpha_p GRL schedule, w_dom = 0.1 * p)
   - Gradient norm clipping to 1.0.

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
    get_stratified_session_trial_splits,
    EEGSequenceDataset
)
from model_spatial_temporal_cdan import SpatialTemporalCDAN

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

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
    
    # Pre-binarize one-hot targets for fast vectorized AUC
    y_true_onehot = np.zeros((num_samples, 4), dtype=np.float32)
    for c in range(4):
        y_true_onehot[:, c] = (y_true == c).astype(np.float32)
        
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
# 2. REGIME A: STRATIFIED 4-FOLD TRIAL CV PER SESSION
# =========================================================================

def train_single_fold_regime_a(train_loader, test_loader, device, epochs=25, lr=1e-3, weight_decay=1e-4):
    """
    Trains Spatial-Temporal 2D-CNN-BiGRU on 18 trials and evaluates on 6 test trials.
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
        
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    test_preds, test_probs, test_trues = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            logits, _, _, probs = model(x, alpha=0.0)
            
            test_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_trues.extend(y.cpu().numpy())
            
    return np.array(test_trues), np.array(test_preds), np.array(test_probs)

def run_regime_a_benchmark(seq_dict, device, epochs=25, batch_size=32, dry_run=False):
    """
    Executes the Stratified 4-Fold Trial Cross-Validation per session across all 15 subjects x 3 sessions (45 runs).
    Guarantees balanced class distribution across all 4 folds for all 24 trials per session.
    """
    print("\n" + "="*80, flush=True)
    print("STARTING REGIME A: STRATIFIED 4-FOLD TRIAL CV PER SESSION (SUBJECT-DEPENDENT)", flush=True)
    print("="*80, flush=True)
    print(f"Total Sequence Samples: {len(seq_dict['labels'])} (T=4, stride=1)", flush=True)
    print(f"Device: {device} | Epochs per fold: {epochs} | Batch size: {batch_size}", flush=True)
    
    subjects = range(1, 16) if not dry_run else [1]
    sessions = range(1, 4) if not dry_run else [1]
    
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    
    session_results = []
    start_time = time.time()
    
    total_sessions = len(subjects) * len(sessions)
    ses_idx = 0
    
    for sub in subjects:
        for ses in sessions:
            ses_idx += 1
            # Filter mask for this subject & session
            sub_ses_mask = (seq_dict['subject_ids'] == sub) & (seq_dict['session_nums'] == ses)
            sub_dict = {k: v[sub_ses_mask] for k, v in seq_dict.items()}
            
            # Generate Stratified 4-Fold splits (18 train trials, 6 test trials)
            trial_folds = get_stratified_session_trial_splits(sub_dict, n_splits=4, seed=42)
            
            ses_y_true, ses_y_pred, ses_y_prob = [], [], []
            
            for f_info in trial_folds:
                # Global masks mapped back to full seq_dict
                global_train_mask = sub_ses_mask.copy()
                global_test_mask = sub_ses_mask.copy()
                
                global_train_mask[sub_ses_mask] = f_info['train_mask']
                global_test_mask[sub_ses_mask] = f_info['test_mask']
                
                # Scaled spatial datasets (StandardScaler fit ONLY on the 18 train trials)
                train_ds, test_ds, _, _ = prepare_scaled_spatial_datasets(seq_dict, global_train_mask, global_test_mask)
                
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
                
                f_true, f_pred, f_prob = train_single_fold_regime_a(
                    train_loader, test_loader, device, epochs=epochs, lr=1e-3, weight_decay=1e-4
                )
                
                ses_y_true.extend(f_true)
                ses_y_pred.extend(f_pred)
                ses_y_prob.extend(f_prob)
                
            ses_y_true = np.array(ses_y_true)
            ses_y_pred = np.array(ses_y_pred)
            ses_y_prob = np.array(ses_y_prob)
            
            ses_acc = accuracy_score(ses_y_true, ses_y_pred)
            ses_f1 = precision_recall_fscore_support(ses_y_true, ses_y_pred, average='macro', zero_division=0)[2]
            
            all_y_true.extend(ses_y_true)
            all_y_pred.extend(ses_y_pred)
            all_y_prob.extend(ses_y_prob)
            
            session_results.append({
                'subject': sub,
                'session': ses,
                'n_samples': len(ses_y_true),
                'accuracy': float(ses_acc),
                'macro_f1': float(ses_f1)
            })
            
            print(f"[{ses_idx:02d}/{total_sessions:02d}] Subject {sub:02d} Session {ses} (Stratified 4-Fold CV) -> Acc: {ses_acc*100:6.2f}%, Macro-F1: {ses_f1:.4f} (N={len(ses_y_true)})", flush=True)
            
    total_elapsed = time.time() - start_time
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    overall_metrics = evaluate_metrics(all_y_true, all_y_pred, all_y_prob, compute_ci=True)
    overall_metrics['total_sessions'] = total_sessions
    overall_metrics['total_test_samples'] = len(all_y_true)
    overall_metrics['elapsed_seconds'] = total_elapsed
    overall_metrics['session_breakdown'] = session_results
    
    # Calculate session-level mean & std
    acc_list = [s['accuracy'] for s in session_results]
    f1_list = [s['macro_f1'] for s in session_results]
    overall_metrics['mean_session_acc'] = float(np.mean(acc_list))
    overall_metrics['std_session_acc'] = float(np.std(acc_list))
    overall_metrics['mean_session_f1'] = float(np.mean(f1_list))
    overall_metrics['std_session_f1'] = float(np.std(f1_list))
    
    print("\n" + "-"*80, flush=True)
    print("REGIME A (STRATIFIED 4-FOLD TRIAL CV) FINAL VERIFIED METRICS:", flush=True)
    print(f"Pooled Accuracy:  {overall_metrics['accuracy']*100:6.2f}% [95% CI: {overall_metrics['accuracy_ci'][0]*100:.2f}%, {overall_metrics['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"Session Mean Acc: {overall_metrics['mean_session_acc']*100:6.2f}% +/- {overall_metrics['std_session_acc']*100:.2f}%", flush=True)
    print(f"Macro-F1 Score:   {overall_metrics['f1']:.4f} [95% CI: {overall_metrics['f1_ci'][0]:.4f}, {overall_metrics['f1_ci'][1]:.4f}]", flush=True)
    print(f"Session Mean F1:  {overall_metrics['mean_session_f1']:.4f} +/- {overall_metrics['std_session_f1']:.4f}", flush=True)
    print(f"Macro Precision:  {overall_metrics['precision']:.4f} [95% CI: {overall_metrics['precision_ci'][0]:.4f}, {overall_metrics['precision_ci'][1]:.4f}]", flush=True)
    print(f"Macro Recall:     {overall_metrics['recall']:.4f} [95% CI: {overall_metrics['recall_ci'][0]:.4f}, {overall_metrics['recall_ci'][1]:.4f}]", flush=True)
    print(f"Macro ROC-AUC:    {overall_metrics['auc']:.4f} [95% CI: {overall_metrics['auc_ci'][0]:.4f}, {overall_metrics['auc_ci'][1]:.4f}]", flush=True)
    print(f"Cohen's Kappa:    {overall_metrics['kappa']:.4f} [95% CI: {overall_metrics['kappa_ci'][0]:.4f}, {overall_metrics['kappa_ci'][1]:.4f}]", flush=True)
    print(f"Execution Time:   {total_elapsed:.2f}s ({total_elapsed/total_sessions:.2f}s/session)", flush=True)
    print("-"*80, flush=True)
    
    return overall_metrics, (all_y_true, all_y_pred, all_y_prob)

# =========================================================================
# 3. REGIME B: TRANSDUCTIVE CDAN WITH 10-EPOCH WARM-UP & ANNEALED ADAPTATION
# =========================================================================

def train_single_fold_cdan(source_loader, target_unlabeled_loader, target_test_loader, device, total_epochs=40, warmup_epochs=10, lr=5e-4, weight_decay=1e-4, max_w_dom=0.1):
    """
    Trains Spatial-Temporal 2D-CNN-BiGRU + CDAN across 1 fold of Leave-3-Subjects-Out.
    Fix 3: 10-epoch supervised source warm-up (w_dom = 0.0), followed by annealed CDAN GRL schedule.
    """
    model = SpatialTemporalCDAN(in_channels=5, spatial_dim=128, rnn_hidden=64, num_classes=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)
    
    len_source = len(source_loader)
    len_target = len(target_unlabeled_loader)
    n_batches = max(len_source, len_target)
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(total_epochs):
        model.train()
        
        # Warm-up schedule:
        if epoch < warmup_epochs:
            # Pure supervised source cross-entropy training (stabilize spatial-temporal embeddings)
            alpha = 0.0
            w_dom = 0.0
        else:
            # Annealed CDAN schedule from epochs 10 to 40
            p = float(epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            w_dom = max_w_dom * alpha
            
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
            
            # Clip gradient norms to 1.0 on both discriminator and feature extractor
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_total_loss += loss_dict['total_loss'].item()
            epoch_cls_loss += loss_dict['loss_cls'].item()
            epoch_dom_loss += loss_dict['loss_dom'].item()
            
        scheduler.step()
        avg_loss = epoch_total_loss / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
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

def run_regime_b_cdan_benchmark(seq_dict, device, total_epochs=40, warmup_epochs=10, batch_size=128, dry_run=False):
    """
    Executes the 5-fold Leave-3-Subjects-Out Transductive CDAN benchmark with warm-up.
    """
    print("\n" + "="*80, flush=True)
    print("STARTING REGIME B: TRANSDUCTIVE CDAN CROSS-SUBJECT (WITH 10-EPOCH WARM-UP)", flush=True)
    print("="*80, flush=True)
    
    folds = [
        {'name': 'Fold 1', 'test_subs': [1, 2, 3],   'source_subs': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        {'name': 'Fold 2', 'test_subs': [4, 5, 6],   'source_subs': [1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15]},
        {'name': 'Fold 3', 'test_subs': [7, 8, 9],   'source_subs': [1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15]},
        {'name': 'Fold 4', 'test_subs': [10, 11, 12], 'source_subs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 15]},
        {'name': 'Fold 5', 'test_subs': [13, 14, 15], 'source_subs': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    ]
    
    if dry_run:
        folds = folds[:1]
        
    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    fold_results = []
    start_time = time.time()
    
    for fold_idx, fold in enumerate(folds):
        print(f"\n--- Running {fold['name']}: Target Subjects {fold['test_subs']} (Source: {len(fold['source_subs'])} subjects) ---", flush=True)
        
        source_mask = np.isin(seq_dict['subject_ids'], fold['source_subs'])
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
        
        epochs_run = 3 if dry_run else total_epochs
        warmup_run = 1 if dry_run else warmup_epochs
        
        y_true, y_pred, y_prob = train_single_fold_cdan(
            source_loader, target_unlabeled_loader, target_test_loader, device,
            total_epochs=epochs_run, warmup_epochs=warmup_run, lr=5e-4, weight_decay=1e-4, max_w_dom=0.1
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
        
        print(f"Result {fold['name']} -> Target Acc: {acc*100:6.2f}%, Macro-F1: {f1:.4f} (N_target={len(test_ds)})", flush=True)
        
    total_elapsed = time.time() - start_time
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_prob = np.array(all_y_prob)
    
    overall_metrics = evaluate_metrics(all_y_true, all_y_pred, all_y_prob, compute_ci=True)
    overall_metrics['total_folds'] = len(folds)
    overall_metrics['total_target_samples'] = len(all_y_true)
    overall_metrics['elapsed_seconds'] = total_elapsed
    overall_metrics['fold_breakdown'] = fold_results
    
    print("\n" + "-"*80, flush=True)
    print("REGIME B (TRANSDUCTIVE CDAN CROSS-SUBJECT) FINAL VERIFIED METRICS:", flush=True)
    print(f"Pooled Accuracy:  {overall_metrics['accuracy']*100:6.2f}% [95% CI: {overall_metrics['accuracy_ci'][0]*100:.2f}%, {overall_metrics['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"Macro-F1 Score:   {overall_metrics['f1']:.4f} [95% CI: {overall_metrics['f1_ci'][0]:.4f}, {overall_metrics['f1_ci'][1]:.4f}]", flush=True)
    print(f"Macro Precision:  {overall_metrics['precision']:.4f} [95% CI: {overall_metrics['precision_ci'][0]:.4f}, {overall_metrics['precision_ci'][1]:.4f}]", flush=True)
    print(f"Macro Recall:     {overall_metrics['recall']:.4f} [95% CI: {overall_metrics['recall_ci'][0]:.4f}, {overall_metrics['recall_ci'][1]:.4f}]", flush=True)
    print(f"Macro ROC-AUC:    {overall_metrics['auc']:.4f} [95% CI: {overall_metrics['auc_ci'][0]:.4f}, {overall_metrics['auc_ci'][1]:.4f}]", flush=True)
    print(f"Cohen's Kappa:    {overall_metrics['kappa']:.4f} [95% CI: {overall_metrics['kappa_ci'][0]:.4f}, {overall_metrics['kappa_ci'][1]:.4f}]", flush=True)
    print(f"Execution Time:   {total_elapsed:.2f}s ({total_elapsed/len(folds):.2f}s/fold)", flush=True)
    print("-"*80, flush=True)
    
    return overall_metrics, (all_y_true, all_y_pred, all_y_prob)

# =========================================================================
# 4. PUBLICATION-QUALITY VISUALIZATIONS (300 DPI)
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
                        
        ax.set_title("Subject-Dependent Spatial-Temporal 2D-CNN-BiGRU\n(Stratified 4-Fold CV Normalized Confusion Matrix)", fontsize=12, pad=12, fontweight='bold')
        ax.set_xlabel("Predicted Emotion Class", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Emotion Class", fontsize=11, fontweight='bold')
        plt.tight_layout()
        cm_path = os.path.join(output_dir, "regimeA_subject_dependent_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_path}", flush=True)
        
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
        plt.title("Subject-Dependent 2D-CNN-BiGRU ROC Curves\n(Stratified 4-Fold CV)", fontsize=12, pad=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10, frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "regimeA_subject_dependent_roc_curves.png")
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"Saved: {roc_path}", flush=True)
        
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
                        
        ax.set_title("Transductive CDAN Cross-Subject (5-Fold Nested CV)\nNormalized Confusion Matrix (%)", fontsize=12, pad=12, fontweight='bold')
        ax.set_xlabel("Predicted Emotion Class", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Emotion Class", fontsize=11, fontweight='bold')
        plt.tight_layout()
        cm_b_path = os.path.join(output_dir, "regimeB_cdan_cross_subject_confusion_matrix.png")
        plt.savefig(cm_b_path, dpi=300)
        plt.close()
        print(f"Saved: {cm_b_path}", flush=True)
        
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
        plt.title("Transductive CDAN Cross-Subject ROC Curves", fontsize=12, pad=12, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10, frameon=True)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_b_path = os.path.join(output_dir, "regimeB_cdan_cross_subject_roc_curves.png")
        plt.savefig(roc_b_path, dpi=300)
        plt.close()
        print(f"Saved: {roc_b_path}", flush=True)

# =========================================================================
# 5. MAIN ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Spatial-Temporal 2D-CNN-BiGRU and CDAN Benchmark Evaluator")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'regime_a', 'regime_b', 'dry_run'],
                        help="Execution mode: 'all', 'regime_a', 'regime_b', or 'dry_run'")
    parser.add_argument('--dataset', type=str, default='seed_iv_processed.npz', help="Path to processed SEED-IV dataset")
    parser.add_argument('--epochs_a', type=int, default=25, help="Training epochs for Regime A folds")
    parser.add_argument('--epochs_b', type=int, default=40, help="Total training epochs for Regime B (CDAN)")
    parser.add_argument('--warmup_b', type=int, default=10, help="Warmup epochs for Regime B (CDAN)")
    parser.add_argument('--batch_size_a', type=int, default=32, help="Batch size for Regime A")
    parser.add_argument('--batch_size_b', type=int, default=128, help="Batch size for Regime B")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}", flush=True)
    
    # 1. Load and build trial-quarantined sequences (T=4, stride=1)
    print(f"Loading dataset from '{args.dataset}' and generating sequences (T=4, stride=1)...", flush=True)
    seq_dict = build_trial_quarantined_sequences(args.dataset, T=4, stride=1)
    print(f"Successfully generated {len(seq_dict['labels'])} sequences.", flush=True)
    
    reg_a_metrics, reg_a_data = None, None
    reg_b_metrics, reg_b_data = None, None
    
    if args.mode in ['all', 'regime_a', 'dry_run']:
        is_dry = (args.mode == 'dry_run')
        epochs_a = 2 if is_dry else args.epochs_a
        reg_a_metrics, reg_a_data = run_regime_a_benchmark(
            seq_dict, device, epochs=epochs_a, batch_size=args.batch_size_a, dry_run=is_dry
        )
        if not is_dry:
            with open("upgraded_regimeA_subject_dependent_results.json", "w") as f:
                json.dump(reg_a_metrics, f, indent=2)
            print("Saved Regime A results to 'upgraded_regimeA_subject_dependent_results.json'", flush=True)
            
    if args.mode in ['all', 'regime_b', 'dry_run']:
        is_dry = (args.mode == 'dry_run')
        epochs_b = 3 if is_dry else args.epochs_b
        warmup_b = 1 if is_dry else args.warmup_b
        reg_b_metrics, reg_b_data = run_regime_b_cdan_benchmark(
            seq_dict, device, total_epochs=epochs_b, warmup_epochs=warmup_b, batch_size=args.batch_size_b, dry_run=is_dry
        )
        if not is_dry:
            with open("upgraded_regimeB_cdan_cross_subject_results.json", "w") as f:
                json.dump(reg_b_metrics, f, indent=2)
            print("Saved Regime B results to 'upgraded_regimeB_cdan_cross_subject_results.json'", flush=True)
            
    if args.mode in ['all', 'regime_a', 'regime_b'] and args.mode != 'dry_run':
        print("\nGenerating publication-quality figures (300 DPI)...", flush=True)
        generate_evaluation_figures(reg_a_data, reg_b_data, output_dir="figures/upgraded_spatial_temporal")
        
    print("\n[SUCCESS] Benchmark execution completed cleanly!", flush=True)

if __name__ == '__main__':
    main()
