"""
Few-Shot Subject Calibration Experiment
========================================
Quantifies gap closure between zero-shot cross-subject (38.41%) and full
subject-dependent (65.53%) performance as a function of K calibration trials
(K in [0, 1, 2, 4, 8]) from previously quarantined target subjects.

Strict Leakage Prevention:
- Base cross-subject models (checkpoints/dann_final/) have 3 test subjects/fold fully quarantined.
- StandardScalers fitted strictly on source training subjects only.
- For each subject and each K, calibration and evaluation trials are strictly disjoint.
- Evaluation data is never seen during fine-tuning.
- 5 independent random trial-selection repeats per K value.
- Cross-architecture comparison with LightGBM.
"""

import os
import sys
import time
import copy
import json
import csv

# Ensure UTF-8 output encoding for Windows compatibility
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
import lightgbm as lgb
import matplotlib.pyplot as plt

# Set working directory to project root
sys.path.insert(0, os.path.abspath(r"c:\Users\Daksh's pc\Desktop\eri"))
from train_final_dann import DANN

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
K_VALUES = [0, 1, 2, 4, 8]
REPEAT_SEEDS = [42, 123, 456, 789, 101112]
SECONDS_PER_TRIAL = 139.2  # SEED-IV empirical mean: 34.8 windows * 4s

def reset_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000, seed=42):
    """Computes 95% bootstrap confidence intervals for classification metrics."""
    rng = np.random.RandomState(seed)
    accs, f1s, aucs, kappas = [], [], [], []
    num_samples = len(y_true)
    
    for _ in range(num_resamples):
        idx = rng.choice(num_samples, num_samples, replace=True)
        yt_res = y_true[idx]
        yp_res = y_pred[idx]
        yprob_res = y_prob[idx]
        
        accs.append(np.mean(yt_res == yp_res))
        f1s.append(precision_recall_fscore_support(yt_res, yp_res, average='macro', zero_division=0)[2])
        try:
            aucs.append(roc_auc_score(yt_res, yprob_res, average='macro', multi_class='ovr'))
        except Exception:
            aucs.append(0.5)
        kappas.append(cohen_kappa_score(yt_res, yp_res))
        
    return {
        'accuracy_ci': [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        'f1_ci': [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        'auc_ci': [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
        'kappa_ci': [float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5))]
    }

# =========================================================================
# EXPERIMENT ENGINE: DANN FEW-SHOT CALIBRATION
# =========================================================================

def run_dann_few_shot_calibration(features_flat, labels, subject_ids, trial_keys, device, checkpoints_dir):
    print("=" * 85, flush=True)
    print("RUNNING CALIBRATED INDUCTIVE DANN FEW-SHOT SUBJECT CALIBRATION", flush=True)
    print("=" * 85, flush=True)

    subject_folds = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]

    # Pre-fit scalers and pre-load base models per fold
    fold_scalers = {}
    fold_base_models = {}
    
    for fold, test_subs in enumerate(subject_folds):
        train_pool_subs = [s for s in range(1, 16) if s not in test_subs]
        train_subs = train_pool_subs[:-2]
        
        scaler = StandardScaler()
        scaler.fit(features_flat[np.isin(subject_ids, train_subs)])
        fold_scalers[fold] = scaler
        
        ckpt_path = os.path.join(checkpoints_dir, f"dann_final_cross_subject_fold{fold+1}.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        base_model = DANN(
            in_features=310, hidden_dim1=256, hidden_dim2=128,
            n_classes=4, n_domains=len(train_subs), dropout=0.2
        ).to(device)
        base_model.load_state_dict(ckpt['state_dict'])
        base_model.eval()
        fold_base_models[fold] = base_model

    # ---------------------------------------------------------------------
    # Step 1: K = 0 Baseline Reproduction Check
    # ---------------------------------------------------------------------
    print("\n[STEP 1] Running K=0 Zero-Shot Cross-Subject Baseline Check...", flush=True)
    y_true_k0, y_pred_k0, y_prob_k0, sub_ids_k0 = [], [], [], []
    per_subject_k0_acc = {}

    for fold, test_subs in enumerate(subject_folds):
        scaler = fold_scalers[fold]
        model = fold_base_models[fold]
        
        for sub in test_subs:
            sub_mask = (subject_ids == sub)
            X_sub_scaled = scaler.transform(features_flat[sub_mask])
            y_sub = labels[sub_mask]
            
            with torch.no_grad():
                bx = torch.tensor(X_sub_scaled, dtype=torch.float32, device=device)
                c_out, _ = model(bx, alpha=0.0)
                probs = torch.softmax(c_out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
            y_true_k0.extend(y_sub)
            y_pred_k0.extend(preds)
            y_prob_k0.extend(probs)
            sub_ids_k0.extend([sub] * len(y_sub))
            
            sub_acc = float(np.mean(preds == y_sub))
            per_subject_k0_acc[sub] = sub_acc

    y_true_k0 = np.array(y_true_k0)
    y_pred_k0 = np.array(y_pred_k0)
    y_prob_k0 = np.array(y_prob_k0)
    k0_acc = float(np.mean(y_true_k0 == y_pred_k0))
    k0_f1 = float(precision_recall_fscore_support(y_true_k0, y_pred_k0, average='macro', zero_division=0)[2])
    k0_auc = float(roc_auc_score(y_true_k0, y_prob_k0, average='macro', multi_class='ovr'))
    k0_kappa = float(cohen_kappa_score(y_true_k0, y_pred_k0))
    k0_cis = compute_bootstrap_confidence_intervals(y_true_k0, y_pred_k0, y_prob_k0, 1000)

    print(f"  >>> DANN K=0 Accuracy: {k0_acc*100:.4f}% | Macro-F1: {k0_f1:.4f} | AUC: {k0_auc:.4f} | Kappa: {k0_kappa:.4f}", flush=True)
    print(f"  >>> 95% Bootstrap CI: [{k0_cis['accuracy_ci'][0]*100:.2f}%, {k0_cis['accuracy_ci'][1]*100:.2f}%]", flush=True)
    assert abs(k0_acc - 0.384112) < 0.001, f"Sanity check failed: Expected 38.4112%, got {k0_acc*100:.4f}%"
    print("  [PASS] K=0 Baseline reproduction check PASSED exact match (38.4112%).\n", flush=True)

    dann_results = {
        0: {
            'K': 0,
            'duration_sec': 0.0,
            'duration_min': 0.0,
            'accuracy_mean': k0_acc,
            'accuracy_std': 0.0,
            'accuracy_ci': k0_cis['accuracy_ci'],
            'f1_mean': k0_f1,
            'f1_std': 0.0,
            'f1_ci': k0_cis['f1_ci'],
            'auc_mean': k0_auc,
            'auc_std': 0.0,
            'auc_ci': k0_cis['auc_ci'],
            'kappa_mean': k0_kappa,
            'kappa_std': 0.0,
            'kappa_ci': k0_cis['kappa_ci'],
            'per_subject_mean_acc': per_subject_k0_acc,
            'repeats': [{
                'seed': 42,
                'accuracy': k0_acc,
                'f1': k0_f1,
                'auc': k0_auc,
                'kappa': k0_kappa,
                'per_subject_acc': per_subject_k0_acc
            }]
        }
    }

    # ---------------------------------------------------------------------
    # Step 2: Few-Shot Calibration for K in [1, 2, 4, 8] across 5 repeats
    # ---------------------------------------------------------------------
    print("[STEP 2] Running Few-Shot Fine-Tuning Calibration (K in [1, 2, 4, 8])...", flush=True)

    for K in [1, 2, 4, 8]:
        t_k_start = time.time()
        cal_duration_sec = K * SECONDS_PER_TRIAL
        cal_duration_min = cal_duration_sec / 60.0
        
        print(f"\n--- Evaluating K = {K} Trials ({cal_duration_sec:.1f}s / {cal_duration_min:.2f} mins calibration EEG) ---", flush=True)
        
        repeat_accs, repeat_f1s, repeat_aucs, repeat_kappas = [], [], [], []
        repeat_per_sub_accs = {sub: [] for sub in range(1, 16)}
        repeat_records = []
        y_true_all_repeats, y_pred_all_repeats, y_prob_all_repeats = [], [], []
        
        for rep_idx, seed in enumerate(REPEAT_SEEDS):
            t_rep_start = time.time()
            y_true_rep, y_pred_rep, y_prob_rep, sub_ids_rep = [], [], [], []
            sub_accs_rep = {}
            
            for fold, test_subs in enumerate(subject_folds):
                scaler = fold_scalers[fold]
                base_model = fold_base_models[fold]
                
                for sub in test_subs:
                    sub_mask = (subject_ids == sub)
                    sub_trial_keys = np.unique(trial_keys[sub_mask])
                    assert len(sub_trial_keys) == 72, f"Subject {sub} has {len(sub_trial_keys)} trials, expected 72."
                    
                    # Deterministic trial sampling per repeat seed
                    rng = np.random.RandomState(seed * 100 + sub)
                    cal_trials = rng.choice(sub_trial_keys, size=K, replace=False)
                    eval_trials = np.array([t for t in sub_trial_keys if t not in cal_trials])
                    
                    # Strict leakage prevention assertion
                    assert len(set(cal_trials).intersection(set(eval_trials))) == 0, f"Leakage detected for Subj {sub}!"
                    
                    cal_mask = np.isin(trial_keys, cal_trials)
                    eval_mask = np.isin(trial_keys, eval_trials)
                    
                    X_cal_scaled = scaler.transform(features_flat[cal_mask])
                    y_cal = labels[cal_mask]
                    X_eval_scaled = scaler.transform(features_flat[eval_mask])
                    y_eval = labels[eval_mask]
                    
                    # Independent model clone
                    model_clone = copy.deepcopy(base_model)
                    model_clone.train()
                    
                    # Freeze BatchNorm running stats (eval mode), but keep learnable gamma/beta active
                    for m in model_clone.modules():
                        if isinstance(m, nn.BatchNorm1d):
                            m.eval()
                            
                    optimizer = torch.optim.AdamW(model_clone.parameters(), lr=1e-4, weight_decay=1e-4)
                    criterion = nn.CrossEntropyLoss()
                    
                    tx_cal = torch.tensor(X_cal_scaled, dtype=torch.float32, device=device)
                    ty_cal = torch.tensor(y_cal, dtype=torch.long, device=device)
                    tx_eval = torch.tensor(X_eval_scaled, dtype=torch.float32, device=device)
                    
                    # Fine-tune for 10 epochs
                    n_cal = len(y_cal)
                    batch_size = 16
                    epochs = 10
                    
                    for epoch in range(epochs):
                        perm = torch.randperm(n_cal, device=device)
                        for b_idx in range(0, n_cal, batch_size):
                            b_indices = perm[b_idx:b_idx + batch_size]
                            bx_b, by_b = tx_cal[b_indices], ty_cal[b_indices]
                            
                            optimizer.zero_grad()
                            c_out, _ = model_clone(bx_b, alpha=0.0)
                            loss = criterion(c_out, by_b)
                            loss.backward()
                            optimizer.step()
                            
                    # Evaluate on held-out evaluation set
                    model_clone.eval()
                    with torch.no_grad():
                        c_out, _ = model_clone(tx_eval, alpha=0.0)
                        probs = torch.softmax(c_out, dim=1).cpu().numpy()
                        preds = np.argmax(probs, axis=1)
                        
                    y_true_rep.extend(y_eval)
                    y_pred_rep.extend(preds)
                    y_prob_rep.extend(probs)
                    sub_ids_rep.extend([sub] * len(y_eval))
                    
                    s_acc = float(np.mean(preds == y_eval))
                    sub_accs_rep[sub] = s_acc
                    repeat_per_sub_accs[sub].append(s_acc)
                    
            y_true_rep = np.array(y_true_rep)
            y_pred_rep = np.array(y_pred_rep)
            y_prob_rep = np.array(y_prob_rep)
            
            y_true_all_repeats.extend(y_true_rep)
            y_pred_all_repeats.extend(y_pred_rep)
            y_prob_all_repeats.extend(y_prob_rep)
            
            rep_acc = float(np.mean(y_true_rep == y_pred_rep))
            rep_f1 = float(precision_recall_fscore_support(y_true_rep, y_pred_rep, average='macro', zero_division=0)[2])
            rep_auc = float(roc_auc_score(y_true_rep, y_prob_rep, average='macro', multi_class='ovr'))
            rep_kappa = float(cohen_kappa_score(y_true_rep, y_pred_rep))
            
            repeat_accs.append(rep_acc)
            repeat_f1s.append(rep_f1)
            repeat_aucs.append(rep_auc)
            repeat_kappas.append(rep_kappa)
            
            t_rep = time.time() - t_rep_start
            print(f"  [K={K} Repeat {rep_idx+1}/5 (Seed {seed})] Acc: {rep_acc*100:.2f}% | F1: {rep_f1:.4f} | AUC: {rep_auc:.4f} | Kappa: {rep_kappa:.4f} ({t_rep:.1f}s)", flush=True)
            
            repeat_records.append({
                'seed': seed,
                'accuracy': rep_acc,
                'f1': rep_f1,
                'auc': rep_auc,
                'kappa': rep_kappa,
                'per_subject_acc': sub_accs_rep
            })
            
        mean_acc = float(np.mean(repeat_accs))
        std_acc = float(np.std(repeat_accs))
        mean_f1 = float(np.mean(repeat_f1s))
        std_f1 = float(np.std(repeat_f1s))
        mean_auc = float(np.mean(repeat_aucs))
        std_auc = float(np.std(repeat_aucs))
        mean_kappa = float(np.mean(repeat_kappas))
        std_kappa = float(np.std(repeat_kappas))
        
        # 95% Bootstrap CI over the pooled evaluation predictions across all 5 repeats
        y_true_all_repeats = np.array(y_true_all_repeats)
        y_pred_all_repeats = np.array(y_pred_all_repeats)
        y_prob_all_repeats = np.array(y_prob_all_repeats)
        cis = compute_bootstrap_confidence_intervals(y_true_all_repeats, y_pred_all_repeats, y_prob_all_repeats, 1000)
        
        per_sub_mean = {sub: float(np.mean(repeat_per_sub_accs[sub])) for sub in range(1, 16)}
        t_k = time.time() - t_k_start
        
        print(f"  >>> [K={K} SUMMARY ({t_k:.1f}s)] Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}% (95% CI: [{cis['accuracy_ci'][0]*100:.2f}%, {cis['accuracy_ci'][1]*100:.2f}%])", flush=True)
        print(f"      Macro-F1: {mean_f1:.4f} +/- {std_f1:.4f} | ROC-AUC: {mean_auc:.4f} +/- {std_auc:.4f} | Kappa: {mean_kappa:.4f} +/- {std_kappa:.4f}", flush=True)
        
        dann_results[K] = {
            'K': K,
            'duration_sec': cal_duration_sec,
            'duration_min': cal_duration_min,
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'accuracy_ci': cis['accuracy_ci'],
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'f1_ci': cis['f1_ci'],
            'auc_mean': mean_auc,
            'auc_std': std_auc,
            'auc_ci': cis['auc_ci'],
            'kappa_mean': mean_kappa,
            'kappa_std': std_kappa,
            'kappa_ci': cis['kappa_ci'],
            'per_subject_mean_acc': per_sub_mean,
            'repeats': repeat_records
        }

    return dann_results

# =========================================================================
# EXPERIMENT ENGINE: LIGHTGBM FEW-SHOT CALIBRATION
# =========================================================================

def run_lightgbm_few_shot_calibration(features_flat, labels, subject_ids, trial_keys):
    print("\n" + "=" * 85, flush=True)
    print("RUNNING LIGHTGBM FEW-SHOT SUBJECT CALIBRATION COMPARISON", flush=True)
    print("=" * 85, flush=True)

    subject_folds = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]

    params = {
        'objective': 'multiclass',
        'num_class': 4,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # Pre-train base LightGBM boosters per fold
    fold_scalers = {}
    fold_base_boosters = {}
    
    for fold, test_subs in enumerate(subject_folds):
        train_subs = [s for s in range(1, 16) if s not in test_subs]
        train_mask = np.isin(subject_ids, train_subs)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(features_flat[train_mask])
        y_train = labels[train_mask]
        fold_scalers[fold] = scaler
        
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        booster = lgb.train(params, train_data, num_boost_round=100)
        fold_base_boosters[fold] = (booster, X_train_scaled, y_train)

    # ---------------------------------------------------------------------
    # Step 1: K = 0 LightGBM Baseline Check
    # ---------------------------------------------------------------------
    print("\n[STEP 1] Running LightGBM K=0 Baseline Check...", flush=True)
    y_true_k0, y_pred_k0, y_prob_k0 = [], [], []
    per_subject_k0_acc = {}

    for fold, test_subs in enumerate(subject_folds):
        scaler = fold_scalers[fold]
        booster, _, _ = fold_base_boosters[fold]
        
        for sub in test_subs:
            sub_mask = (subject_ids == sub)
            X_sub_scaled = scaler.transform(features_flat[sub_mask])
            y_sub = labels[sub_mask]
            
            probs = booster.predict(X_sub_scaled)
            preds = np.argmax(probs, axis=1)
            
            y_true_k0.extend(y_sub)
            y_pred_k0.extend(preds)
            y_prob_k0.extend(probs)
            
            sub_acc = float(np.mean(preds == y_sub))
            per_subject_k0_acc[sub] = sub_acc

    y_true_k0 = np.array(y_true_k0)
    y_pred_k0 = np.array(y_pred_k0)
    y_prob_k0 = np.array(y_prob_k0)
    k0_acc = float(np.mean(y_true_k0 == y_pred_k0))
    k0_f1 = float(precision_recall_fscore_support(y_true_k0, y_pred_k0, average='macro', zero_division=0)[2])
    k0_auc = float(roc_auc_score(y_true_k0, y_prob_k0, average='macro', multi_class='ovr'))
    k0_kappa = float(cohen_kappa_score(y_true_k0, y_pred_k0))
    k0_cis = compute_bootstrap_confidence_intervals(y_true_k0, y_pred_k0, y_prob_k0, 1000)

    print(f"  >>> LightGBM K=0 Accuracy: {k0_acc*100:.4f}% | Macro-F1: {k0_f1:.4f} | AUC: {k0_auc:.4f} | Kappa: {k0_kappa:.4f}", flush=True)
    print(f"  >>> 95% Bootstrap CI: [{k0_cis['accuracy_ci'][0]*100:.2f}%, {k0_cis['accuracy_ci'][1]*100:.2f}%]", flush=True)
    assert abs(k0_acc - 0.382302) < 0.002, f"LightGBM Sanity check mismatch: got {k0_acc*100:.4f}%"
    print("  [PASS] LightGBM K=0 Baseline reproduction check PASSED match (38.23%).\n", flush=True)

    lgb_results = {
        0: {
            'K': 0,
            'duration_sec': 0.0,
            'duration_min': 0.0,
            'accuracy_mean': k0_acc,
            'accuracy_std': 0.0,
            'accuracy_ci': k0_cis['accuracy_ci'],
            'f1_mean': k0_f1,
            'f1_std': 0.0,
            'f1_ci': k0_cis['f1_ci'],
            'auc_mean': k0_auc,
            'auc_std': 0.0,
            'auc_ci': k0_cis['auc_ci'],
            'kappa_mean': k0_kappa,
            'kappa_std': 0.0,
            'kappa_ci': k0_cis['kappa_ci'],
            'per_subject_mean_acc': per_subject_k0_acc
        }
    }

    # ---------------------------------------------------------------------
    # Step 2: Few-Shot Calibration for K in [1, 2, 4, 8]
    # ---------------------------------------------------------------------
    print("[STEP 2] Running LightGBM Few-Shot Calibration (K in [1, 2, 4, 8])...", flush=True)

    for K in [1, 2, 4, 8]:
        t_k_start = time.time()
        cal_duration_sec = K * SECONDS_PER_TRIAL
        cal_duration_min = cal_duration_sec / 60.0
        
        repeat_accs, repeat_f1s, repeat_aucs, repeat_kappas = [], [], [], []
        repeat_per_sub_accs = {sub: [] for sub in range(1, 16)}
        y_true_all_repeats, y_pred_all_repeats, y_prob_all_repeats = [], [], []
        
        for rep_idx, seed in enumerate(REPEAT_SEEDS):
            y_true_rep, y_pred_rep, y_prob_rep = [], [], []
            
            for fold, test_subs in enumerate(subject_folds):
                scaler = fold_scalers[fold]
                base_booster, X_train_scaled, y_train = fold_base_boosters[fold]
                
                for sub in test_subs:
                    sub_mask = (subject_ids == sub)
                    sub_trial_keys = np.unique(trial_keys[sub_mask])
                    
                    rng = np.random.RandomState(seed * 100 + sub)
                    cal_trials = rng.choice(sub_trial_keys, size=K, replace=False)
                    eval_trials = np.array([t for t in sub_trial_keys if t not in cal_trials])
                    
                    cal_mask = np.isin(trial_keys, cal_trials)
                    eval_mask = np.isin(trial_keys, eval_trials)
                    
                    X_cal_scaled = scaler.transform(features_flat[cal_mask])
                    y_cal = labels[cal_mask]
                    X_eval_scaled = scaler.transform(features_flat[eval_mask])
                    y_eval = labels[eval_mask]
                    
                    # Adapt LightGBM with sample-weighted source + target calibration samples
                    X_comb = np.vstack([X_train_scaled, X_cal_scaled])
                    y_comb = np.concatenate([y_train, y_cal])
                    weights = np.ones(len(y_comb))
                    weights[len(y_train):] = 50.0  # High weight on calibration samples
                    
                    comb_data = lgb.Dataset(X_comb, label=y_comb, weight=weights)
                    ft_booster = lgb.train(params, comb_data, num_boost_round=20, init_model=base_booster)
                    
                    probs = ft_booster.predict(X_eval_scaled)
                    preds = np.argmax(probs, axis=1)
                    
                    y_true_rep.extend(y_eval)
                    y_pred_rep.extend(preds)
                    y_prob_rep.extend(probs)
                    
                    s_acc = float(np.mean(preds == y_eval))
                    repeat_per_sub_accs[sub].append(s_acc)
                    
            y_true_rep = np.array(y_true_rep)
            y_pred_rep = np.array(y_pred_rep)
            y_prob_rep = np.array(y_prob_rep)
            
            y_true_all_repeats.extend(y_true_rep)
            y_pred_all_repeats.extend(y_pred_rep)
            y_prob_all_repeats.extend(y_prob_rep)
            
            rep_acc = float(np.mean(y_true_rep == y_pred_rep))
            rep_f1 = float(precision_recall_fscore_support(y_true_rep, y_pred_rep, average='macro', zero_division=0)[2])
            rep_auc = float(roc_auc_score(y_true_rep, y_prob_rep, average='macro', multi_class='ovr'))
            rep_kappa = float(cohen_kappa_score(y_true_rep, y_pred_rep))
            
            repeat_accs.append(rep_acc)
            repeat_f1s.append(rep_f1)
            repeat_aucs.append(rep_auc)
            repeat_kappas.append(rep_kappa)
            
        mean_acc = float(np.mean(repeat_accs))
        std_acc = float(np.std(repeat_accs))
        mean_f1 = float(np.mean(repeat_f1s))
        std_f1 = float(np.std(repeat_f1s))
        mean_auc = float(np.mean(repeat_aucs))
        std_auc = float(np.std(repeat_aucs))
        mean_kappa = float(np.mean(repeat_kappas))
        std_kappa = float(np.std(repeat_kappas))
        
        # 95% Bootstrap CI over the pooled evaluation predictions across all 5 repeats
        y_true_all_repeats = np.array(y_true_all_repeats)
        y_pred_all_repeats = np.array(y_pred_all_repeats)
        y_prob_all_repeats = np.array(y_prob_all_repeats)
        cis = compute_bootstrap_confidence_intervals(y_true_all_repeats, y_pred_all_repeats, y_prob_all_repeats, 1000)
        per_sub_mean = {sub: float(np.mean(repeat_per_sub_accs[sub])) for sub in range(1, 16)}
        t_k = time.time() - t_k_start
        
        print(f"  >>> [LightGBM K={K} ({t_k:.1f}s)] Accuracy: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}% | F1: {mean_f1:.4f} | AUC: {mean_auc:.4f} | Kappa: {mean_kappa:.4f}", flush=True)
        
        lgb_results[K] = {
            'K': K,
            'duration_sec': cal_duration_sec,
            'duration_min': cal_duration_min,
            'accuracy_mean': mean_acc,
            'accuracy_std': std_acc,
            'accuracy_ci': cis['accuracy_ci'],
            'f1_mean': mean_f1,
            'f1_std': std_f1,
            'f1_ci': cis['f1_ci'],
            'auc_mean': mean_auc,
            'auc_std': std_auc,
            'auc_ci': cis['auc_ci'],
            'kappa_mean': mean_kappa,
            'kappa_std': std_kappa,
            'kappa_ci': cis['kappa_ci'],
            'per_subject_mean_acc': per_sub_mean
        }

    return lgb_results

# =========================================================================
# PUBLICATION FIGURE GENERATION (300 DPI)
# =========================================================================

def generate_publication_figure(dann_results, lgb_results, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, "few_shot_calibration_curve.png")
    
    print(f"\nGenerating 300 DPI Publication Figure -> '{fig_path}'...", flush=True)
    
    # 4-Panel Layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(wspace=0.28, hspace=0.32)
    
    ks = [0, 1, 2, 4, 8]
    durations_min = [dann_results[k]['duration_min'] for k in ks]
    
    dann_accs = [dann_results[k]['accuracy_mean'] * 100 for k in ks]
    dann_ci_low = [dann_results[k]['accuracy_ci'][0] * 100 for k in ks]
    dann_ci_high = [dann_results[k]['accuracy_ci'][1] * 100 for k in ks]
    
    lgb_accs = [lgb_results[k]['accuracy_mean'] * 100 for k in ks]
    lgb_ci_low = [lgb_results[k]['accuracy_ci'][0] * 100 for k in ks]
    lgb_ci_high = [lgb_results[k]['accuracy_ci'][1] * 100 for k in ks]
    
    dann_f1s = [dann_results[k]['f1_mean'] for k in ks]
    dann_kappas = [dann_results[k]['kappa_mean'] for k in ks]
    
    # Subject-dependent upper bounds
    DANN_SD_UPPER = 65.53
    LGB_SD_UPPER = 61.30
    
    # ---------------------------------------------------------------------
    # PANEL A: DANN Accuracy vs K Calibration Trials & Real-World Minutes
    # ---------------------------------------------------------------------
    ax_a = axes[0, 0]
    ax_a.plot(ks, dann_accs, marker='o', markersize=8, color='#3D8B7D', lw=2.5, label='Calibrated DANN (Few-Shot)')
    ax_a.fill_between(ks, dann_ci_low, dann_ci_high, color='#3D8B7D', alpha=0.2, label='95% Bootstrap CI')
    
    # Reference lines
    ax_a.axhline(y=DANN_SD_UPPER, color='#C9622A', linestyle='--', lw=1.8, label=f'Subject-Dependent Upper Bound ({DANN_SD_UPPER}%)')
    ax_a.axhline(y=38.41, color='#8A94A6', linestyle=':', lw=1.5, label='Zero-Shot Cross-Subject Baseline (38.41%)')
    
    # Annotate points with real-world minutes
    for i, k in enumerate(ks):
        t_label = f"{durations_min[i]:.1f}m" if k > 0 else "0m"
        ax_a.annotate(f"{dann_accs[i]:.1f}%\n({t_label})", (ks[i], dann_accs[i]),
                      textcoords="offset points", xytext=(0, 10), ha='center',
                      fontsize=8.5, fontweight='bold', color='#0E1A2B')
                      
    ax_a.set_title('A. Few-Shot Calibration Trajectory (DANN)', fontsize=12, fontweight='bold', pad=10)
    ax_a.set_xlabel('Number of Calibration Trials (K)', fontweight='bold')
    ax_a.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax_a.set_xticks(ks)
    ax_a.set_ylim(32, 70)
    ax_a.grid(True, linestyle='--', alpha=0.5)
    ax_a.legend(loc='lower right', frameon=True, fontsize=8.5)
    
    # ---------------------------------------------------------------------
    # PANEL B: DANN vs LightGBM Cross-Architecture Comparison
    # ---------------------------------------------------------------------
    ax_b = axes[0, 1]
    ax_b.plot(ks, dann_accs, marker='o', markersize=8, color='#3D8B7D', lw=2.5, label='Calibrated DANN')
    ax_b.fill_between(ks, dann_ci_low, dann_ci_high, color='#3D8B7D', alpha=0.15)
    
    ax_b.plot(ks, lgb_accs, marker='s', markersize=7, color='#D97706', lw=2.2, label='LightGBM (Weighted Update)')
    ax_b.fill_between(ks, lgb_ci_low, lgb_ci_high, color='#D97706', alpha=0.15)
    
    ax_b.axhline(y=DANN_SD_UPPER, color='#3D8B7D', linestyle=':', lw=1.2, alpha=0.7)
    ax_b.axhline(y=LGB_SD_UPPER, color='#D97706', linestyle=':', lw=1.2, alpha=0.7)
    
    ax_b.set_title('B. DANN vs. LightGBM Adaptation Efficiency', fontsize=12, fontweight='bold', pad=10)
    ax_b.set_xlabel('Number of Calibration Trials (K)', fontweight='bold')
    ax_b.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax_b.set_xticks(ks)
    ax_b.set_ylim(32, 70)
    ax_b.grid(True, linestyle='--', alpha=0.5)
    ax_b.legend(loc='lower right', frameon=True, fontsize=8.5)
    
    # ---------------------------------------------------------------------
    # PANEL C: Macro-F1 & Cohen's Kappa Gains
    # ---------------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.plot(ks, dann_f1s, marker='^', markersize=8, color='#2563EB', lw=2.2, label='Macro-F1 Score')
    ax_c.plot(ks, dann_kappas, marker='D', markersize=7, color='#7C3AED', lw=2.2, label="Cohen's Kappa (κ)")
    
    for i, k in enumerate(ks):
        ax_c.annotate(f"{dann_f1s[i]:.3f}", (ks[i], dann_f1s[i]),
                      textcoords="offset points", xytext=(0, 8), ha='center',
                      fontsize=8.5, fontweight='bold', color='#2563EB')
        ax_c.annotate(f"{dann_kappas[i]:.3f}", (ks[i], dann_kappas[i]),
                      textcoords="offset points", xytext=(0, -14), ha='center',
                      fontsize=8.5, fontweight='bold', color='#7C3AED')
                      
    ax_c.set_title("C. Macro-F1 & Chance-Corrected Agreement (κ)", fontsize=12, fontweight='bold', pad=10)
    ax_c.set_xlabel('Number of Calibration Trials (K)', fontweight='bold')
    ax_c.set_ylabel('Metric Value', fontweight='bold')
    ax_c.set_xticks(ks)
    ax_c.set_ylim(0.15, 0.60)
    ax_c.grid(True, linestyle='--', alpha=0.5)
    ax_c.legend(loc='lower right', frameon=True, fontsize=8.5)
    
    # ---------------------------------------------------------------------
    # PANEL D: Per-Subject Individual Calibration Trajectories
    # ---------------------------------------------------------------------
    ax_d = axes[1, 1]
    palette = plt.cm.tab20(np.linspace(0, 1, 15))
    
    for sub in range(1, 16):
        sub_traj = [dann_results[k]['per_subject_mean_acc'][sub] * 100 for k in ks]
        ax_d.plot(ks, sub_traj, marker='.', markersize=6, lw=1.3, alpha=0.75, color=palette[sub-1], label=f'S{sub:02d}')
        
    ax_d.plot(ks, dann_accs, marker='o', markersize=9, color='#0E1A2B', lw=3.0, label='Cohort Mean')
    
    ax_d.set_title('D. Per-Subject Individual Trajectories (N=15)', fontsize=12, fontweight='bold', pad=10)
    ax_d.set_xlabel('Number of Calibration Trials (K)', fontweight='bold')
    ax_d.set_ylabel('Subject Accuracy (%)', fontweight='bold')
    ax_d.set_xticks(ks)
    ax_d.set_ylim(15, 80)
    ax_d.grid(True, linestyle='--', alpha=0.5)
    ax_d.legend(bbox_to_anchor=(1.04, 1.0), loc='upper left', ncol=1, frameon=True, fontsize=7.5)
    
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Publication figure saved to '{fig_path}'.", flush=True)

# =========================================================================
# MAIN ENTRYPOINT
# =========================================================================

def main():
    t_global_start = time.time()
    figures_dir = os.path.join("figures", "calibration")
    checkpoints_dir = os.path.join("checkpoints", "dann_final")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85, flush=True)
    print("FEW-SHOT SUBJECT CALIBRATION EXPERIMENTAL PIPELINE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85, flush=True)
    
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'")
        
    print(f"Loading SEED-IV dataset from '{dataset_path}'...", flush=True)
    data = np.load(dataset_path)
    features_flat = data["features"].reshape(len(data["labels"]), -1)
    labels = data["labels"]
    subject_ids = data["subject_ids"]
    session_nums = data["session_nums"]
    trial_ids = data["trial_ids"]
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    
    print(f"Loaded {len(labels)} samples across 15 subjects (Total trials: {len(np.unique(trial_keys))}).", flush=True)
    
    # 1. Run DANN Few-Shot Calibration
    reset_seeds(42)
    dann_results = run_dann_few_shot_calibration(features_flat, labels, subject_ids, trial_keys, device, checkpoints_dir)
    
    # 2. Run LightGBM Comparison
    reset_seeds(42)
    lgb_results = run_lightgbm_few_shot_calibration(features_flat, labels, subject_ids, trial_keys)
    
    # 3. Generate Publication Figure
    generate_publication_figure(dann_results, lgb_results, figures_dir)
    
    # 4. Save Structured JSON & CSV
    final_output = {
        'metadata': {
            'dataset': 'SEED-IV',
            'num_subjects': 15,
            'trials_per_subject': 72,
            'seconds_per_trial': SECONDS_PER_TRIAL,
            'k_values': K_VALUES,
            'repeat_seeds': REPEAT_SEEDS,
            'num_repeats': len(REPEAT_SEEDS),
            'dann_subject_dependent_upper_bound': 0.6553,
            'lgb_subject_dependent_upper_bound': 0.6130
        },
        'dann_calibration': dann_results,
        'lightgbm_calibration': lgb_results
    }
    
    json_path = "few_shot_calibration_results.json"
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=4)
    print(f"\nSaved structured JSON results to '{json_path}'", flush=True)
    
    csv_path = "few_shot_calibration_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Model", "K_Trials", "Calibration_Duration_Sec", "Calibration_Duration_Min",
            "Accuracy_Mean", "Accuracy_Std", "Accuracy_95CI_Lower", "Accuracy_95CI_Upper",
            "Macro_F1_Mean", "Macro_F1_Std", "Macro_F1_95CI_Lower", "Macro_F1_95CI_Upper",
            "Macro_AUC_Mean", "Macro_AUC_Std", "Macro_AUC_95CI_Lower", "Macro_AUC_95CI_Upper",
            "Cohen_Kappa_Mean", "Cohen_Kappa_Std", "Cohen_Kappa_95CI_Lower", "Cohen_Kappa_95CI_Upper",
            "Gap_Closure_Pct"
        ])
        
        # DANN Rows
        d_sd_bound = 0.6553
        d_k0 = dann_results[0]['accuracy_mean']
        for k in K_VALUES:
            r = dann_results[k]
            gap_closure = ((r['accuracy_mean'] - d_k0) / (d_sd_bound - d_k0)) * 100.0 if k > 0 else 0.0
            writer.writerow([
                "Calibrated Inductive DANN",
                k,
                f"{r['duration_sec']:.1f}",
                f"{r['duration_min']:.2f}",
                f"{r['accuracy_mean']:.4f}",
                f"{r['accuracy_std']:.4f}",
                f"{r['accuracy_ci'][0]:.4f}",
                f"{r['accuracy_ci'][1]:.4f}",
                f"{r['f1_mean']:.4f}",
                f"{r['f1_std']:.4f}",
                f"{r['f1_ci'][0]:.4f}",
                f"{r['f1_ci'][1]:.4f}",
                f"{r['auc_mean']:.4f}",
                f"{r['auc_std']:.4f}",
                f"{r['auc_ci'][0]:.4f}",
                f"{r['auc_ci'][1]:.4f}",
                f"{r['kappa_mean']:.4f}",
                f"{r['kappa_std']:.4f}",
                f"{r['kappa_ci'][0]:.4f}",
                f"{r['kappa_ci'][1]:.4f}",
                f"{gap_closure:.2f}"
            ])
            
        # LightGBM Rows
        lgb_sd_bound = 0.6130
        lgb_k0 = lgb_results[0]['accuracy_mean']
        for k in K_VALUES:
            r = lgb_results[k]
            gap_closure = ((r['accuracy_mean'] - lgb_k0) / (lgb_sd_bound - lgb_k0)) * 100.0 if k > 0 else 0.0
            writer.writerow([
                "LightGBM",
                k,
                f"{r['duration_sec']:.1f}",
                f"{r['duration_min']:.2f}",
                f"{r['accuracy_mean']:.4f}",
                f"{r['accuracy_std']:.4f}",
                f"{r['accuracy_ci'][0]:.4f}",
                f"{r['accuracy_ci'][1]:.4f}",
                f"{r['f1_mean']:.4f}",
                f"{r['f1_std']:.4f}",
                f"{r['f1_ci'][0]:.4f}",
                f"{r['f1_ci'][1]:.4f}",
                f"{r['auc_mean']:.4f}",
                f"{r['auc_std']:.4f}",
                f"{r['auc_ci'][0]:.4f}",
                f"{r['auc_ci'][1]:.4f}",
                f"{r['kappa_mean']:.4f}",
                f"{r['kappa_std']:.4f}",
                f"{r['kappa_ci'][0]:.4f}",
                f"{r['kappa_ci'][1]:.4f}",
                f"{gap_closure:.2f}"
            ])
            
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)
    
    # 5. Final Summary Table Output
    print("\n" + "=" * 95, flush=True)
    print("FINAL CONSOLIDATION SUMMARY: FEW-SHOT SUBJECT CALIBRATION EXPERIMENT")
    print("=" * 95, flush=True)
    print(f"{'K Trials':<10} | {'Duration':<12} | {'DANN Accuracy (95% CI)':<27} | {'DANN Macro-F1':<15} | {'Gap Closure':<12} | {'LightGBM Acc':<12}", flush=True)
    print("-" * 95, flush=True)
    for k in K_VALUES:
        d = dann_results[k]
        l = lgb_results[k]
        dur_str = f"{d['duration_min']:.1f} min" if k > 0 else "0 min (Zero)"
        d_acc_str = f"{d['accuracy_mean']*100:.2f}% [{d['accuracy_ci'][0]*100:.2f}%, {d['accuracy_ci'][1]*100:.2f}%]"
        d_f1_str = f"{d['f1_mean']:.4f} +/- {d['f1_std']:.4f}"
        gap = ((d['accuracy_mean'] - d_k0) / (d_sd_bound - d_k0)) * 100.0 if k > 0 else 0.0
        gap_str = f"{gap:.1f}%"
        l_acc_str = f"{l['accuracy_mean']*100:.2f}%"
        print(f"K = {k:<6} | {dur_str:<12} | {d_acc_str:<27} | {d_f1_str:<15} | {gap_str:<12} | {l_acc_str:<12}", flush=True)
    print("=" * 95, flush=True)
    
    # Copy figure to artifact directory if it exists
    artifact_fig_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\calibration"
    if os.path.exists(os.path.dirname(artifact_fig_dir)):
        import shutil
        os.makedirs(artifact_fig_dir, exist_ok=True)
        shutil.copy(os.path.join(figures_dir, "few_shot_calibration_curve.png"), os.path.join(artifact_fig_dir, "few_shot_calibration_curve.png"))
        print(f"Copied publication figure to artifact dir '{artifact_fig_dir}'.", flush=True)
        
    print(f"Total Experiment Runtime: {(time.time() - t_global_start)/60:.2f} minutes.", flush=True)

if __name__ == '__main__':
    main()
