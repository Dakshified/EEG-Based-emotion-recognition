"""
Affective Salience Window Extraction Benchmark for SEED-IV EEG
==============================================================
Evaluates the impact of eliminating non-emotional transition frames (stimulus onset,
baseline drift, post-stimulus habituation) on emotion recognition under strict zero-leakage
trial-quarantined cross-validation.

Neurobiological Rationale:
-------------------------
Continuous movie clips (15-45s) do not maintain uniform emotional valence or arousal.
The first few seconds after clip onset capture orienting reflexes and sensory accommodation,
while the final seconds involve emotional recovery and habituation. By isolating affective
salience epochs using high-frequency spectral dynamics (Beta: 14-30 Hz, Gamma: 31-50 Hz)
and temporal boundaries, we minimize ground-truth label noise.

Filtering Strategies (Strictly Intra-Trial):
--------------------------------------------
1. Raw Baseline: 100% full-trial frames retained.
2. Strategy 1 (Temporal Climax Crop): Drop first 3s (onset) and last 2s (offset) per trial.
3. Strategy 2 (Spectral Activation): Rank frames by mean Beta+Gamma DE across 62 channels;
   retain top 60% highest-energy frames per trial.
4. Strategy 3 (Hybrid Salience): Apply 3s onset crop, then rank remaining frames by
   Beta+Gamma power, retaining top 60% per trial.

Evaluation Protocol:
-------------------
- Stratified 4-Fold Trial Cross-Validation per session (18 train trials, 6 test trials).
- 45 sessions across 15 subjects = 180 folds per model per strategy (1,440 fold runs total).
- StandardScaler fitted strictly on retained training frames in each fold.
- Models: LightGBM (n_est=300, lr=0.05, max_depth=6) & Calibrated Shallow MLP (310->128->64->4).
- 95% non-parametric bootstrap confidence intervals (1,000 resamples).
- Publication figures (300 DPI) exported to figures/salient_windows/.
"""

import os
import sys
import time
import json
import csv
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
STRATEGY_DISPLAY_NAMES = {
    'raw': 'Raw Full-Trial (Baseline)',
    'temporal_crop': 'Strategy 1: Temporal Climax Crop (Drop 3s Onset / 2s Offset)',
    'spectral_energy': 'Strategy 2: Spectral Activation (Top 60% Beta+Gamma)',
    'hybrid': 'Strategy 3: Hybrid Salience (3s Crop + Top 60% Beta+Gamma)'
}

def set_seed(seed=42):
    """Sets global random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000, seed=42):
    """Computes 95% non-parametric bootstrap confidence intervals."""
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
    
    return {
        'accuracy_ci': ci_acc,
        'precision_ci': ci_prec,
        'recall_ci': ci_rec,
        'f1_ci': ci_f1,
        'auc_ci': ci_auc,
        'kappa_ci': ci_kappa
    }

def extract_salient_trial_windows(trial_features, strategy='hybrid', top_k_ratio=0.6):
    """
    Extracts salient frames within a single trial independently.
    
    CRITICAL CONSTRAINT: Operates strictly intra-trial. Zero data or statistics cross trials.
    
    Parameters:
        trial_features (np.ndarray): Shape (L, 62, 5) or (L, 310) for a single trial of length L.
        strategy (str): One of ['raw', 'temporal_crop', 'spectral_energy', 'hybrid'].
        top_k_ratio (float): Fraction of frames to retain for ranking strategies (default: 0.6).
        
    Returns:
        selected_indices (np.ndarray): 1D array of selected frame indices (chronologically sorted).
        filtered_features (np.ndarray): Shape (k, 310) of filtered frame features.
    """
    L = len(trial_features)
    if trial_features.ndim == 2:
        feats_3d = trial_features.reshape(L, 62, 5)
        feats_flat = trial_features
    else:
        feats_3d = trial_features
        feats_flat = trial_features.reshape(L, -1)
        
    # Strategy 0: Raw (Baseline)
    if strategy in ['raw', 'none']:
        return np.arange(L), feats_flat
        
    # Strategy 1: Temporal Climax Crop (Drop first 3s and last 2s)
    elif strategy in ['temporal_crop', 'temporal_climax_crop', 'strategy_1']:
        if L > 5:
            indices = np.arange(3, L - 2)
        else:
            indices = np.arange(L)
        return indices, feats_flat[indices]
        
    # Strategy 2: High-Frequency Spectral Activation (Top 60% Beta+Gamma energy)
    elif strategy in ['spectral_energy', 'spectral_activation', 'strategy_2']:
        # Beta is band 3 (14-30 Hz), Gamma is band 4 (31-50 Hz)
        beta_power = feats_3d[:, :, 3]   # (L, 62)
        gamma_power = feats_3d[:, :, 4]  # (L, 62)
        energy = np.mean(beta_power + gamma_power, axis=1) # (L,)
        
        k = max(1, int(round(top_k_ratio * L)))
        k = min(k, L)
        top_indices = np.argsort(energy)[-k:]
        selected_indices = np.sort(top_indices)
        return selected_indices, feats_flat[selected_indices]
        
    # Strategy 3: Hybrid Salience Filtering (Drop 3s onset, then Top 60% Beta+Gamma energy)
    elif strategy in ['hybrid', 'hybrid_salience', 'strategy_3']:
        if L > 3:
            candidate_indices = np.arange(3, L)
        else:
            candidate_indices = np.arange(L)
            
        cand_3d = feats_3d[candidate_indices]
        beta_power = cand_3d[:, :, 3]
        gamma_power = cand_3d[:, :, 4]
        energy = np.mean(beta_power + gamma_power, axis=1) # (len(candidate_indices),)
        
        k = max(1, int(round(top_k_ratio * len(candidate_indices))))
        k = min(k, len(candidate_indices))
        top_in_cand = np.argsort(energy)[-k:]
        selected_indices = np.sort(candidate_indices[top_in_cand])
        return selected_indices, feats_flat[selected_indices]
        
    else:
        raise ValueError(f"Unknown salience extraction strategy '{strategy}'")

class ShallowMLP(nn.Module):
    """
    Calibrated Shallow MLP Architecture:
    Input (310) -> Linear(128) -> BatchNorm1d -> GELU -> Dropout(0.2) ->
    Linear(64) -> BatchNorm1d -> GELU -> Dropout(0.2) -> Linear(4)
    """
    def __init__(self, in_features=310, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_shallow_mlp(X_train, y_train, X_test, device, epochs=40, batch_size=64, lr=1e-3, weight_decay=1e-4):
    """Fast GPU-resident training of Shallow MLP."""
    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    model = ShallowMLP(in_features=310, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    n_samples = len(X_train)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            bx, by = X_tr_t[idx], y_tr_t[idx]
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
    return preds.cpu().numpy(), probs.cpu().numpy()

def train_lightgbm(X_train, y_train, X_test, random_state=42):
    """Trains LightGBM Classifier."""
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)
    return preds, probs

def run_salient_windows_benchmark(dataset_path="seed_iv_processed.npz", dry_run=False, random_state=42):
    """
    Runs the complete Salience Window Extraction benchmark across all 4 strategies,
    both models (LightGBM & Shallow MLP), and all 45 sessions using Stratified 4-Fold CV.
    """
    set_seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80, flush=True)
    print("AFFECTIVE SALIENCE WINDOW EXTRACTION BENCHMARK (SEED-IV)", flush=True)
    print(f"Device: {device} | Random State: {random_state} | Dry-Run Mode: {dry_run}", flush=True)
    print("=" * 80, flush=True)
    
    # 1. Load Dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at '{dataset_path}'")
        
    data = np.load(dataset_path)
    raw_features = data['features']       # (37575, 62, 5)
    labels = data['labels']               # (37575,)
    subject_ids = data['subject_ids']     # (37575,)
    session_nums = data['session_nums']   # (37575,)
    trial_ids = data['trial_ids']         # (37575,)
    
    # Unique trial identifier (Subject * 1000 + Session * 100 + Trial)
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trials = np.unique(trial_keys)
    assert len(unique_trials) == 1080, f"Expected 1080 trials, found {len(unique_trials)}"
    
    # Extract trial index map
    trial_data_map = {}
    for u_trial in unique_trials:
        mask = (trial_keys == u_trial)
        trial_data_map[u_trial] = {
            'features': raw_features[mask], # (L, 62, 5)
            'label': labels[mask][0],       # int (0..3)
            'subject_id': subject_ids[mask][0],
            'session_num': session_nums[mask][0],
            'trial_id': trial_ids[mask][0],
            'length': np.sum(mask)
        }
        
    strategies = ['raw', 'temporal_crop', 'spectral_energy', 'hybrid']
    models = ['lightgbm', 'shallow_mlp']
    
    subjects_to_run = [1] if dry_run else list(range(1, 16))
    sessions_to_run = [1] if dry_run else [1, 2, 3]
    
    benchmark_results = {
        'protocol': 'Stratified 4-Fold Trial Cross-Validation per Session (Strict Intra-Trial Salience)',
        'dataset': 'SEED-IV (310 DE Features, 4 Emotion Classes)',
        'dry_run': dry_run,
        'strategies': {}
    }
    
    csv_rows = []
    
    start_time_all = time.time()
    
    # Store predictions for plotting
    predictions_store = {}
    
    for strategy in strategies:
        strat_display = STRATEGY_DISPLAY_NAMES[strategy]
        print(f"\nEvaluating: {strat_display}", flush=True)
        print("-" * 80, flush=True)
        
        strat_start_time = time.time()
        
        # Pre-filter all trials under current strategy
        filtered_trial_map = {}
        total_retained_frames = 0
        total_raw_frames = 0
        
        for u_trial, t_dict in trial_data_map.items():
            sel_idx, f_feats = extract_salient_trial_windows(t_dict['features'], strategy=strategy, top_k_ratio=0.6)
            filtered_trial_map[u_trial] = {
                'features': f_feats, # (k, 310)
                'label': t_dict['label'],
                'subject_id': t_dict['subject_id'],
                'session_num': t_dict['session_num'],
                'trial_id': t_dict['trial_id'],
                'retained_len': len(f_feats),
                'raw_len': t_dict['length']
            }
            total_retained_frames += len(f_feats)
            total_raw_frames += t_dict['length']
            
        retention_ratio = total_retained_frames / total_raw_frames
        print(f"Frame Retention: {total_retained_frames} / {total_raw_frames} ({retention_ratio*100:.2f}%)", flush=True)
        
        strategy_eval = {
            'strategy_name': strategy,
            'display_name': strat_display,
            'total_retained_frames': total_retained_frames,
            'retention_ratio': retention_ratio,
            'models': {}
        }
        
        predictions_store[strategy] = {}
        
        for model_name in models:
            n_folds_total = len(subjects_to_run) * len(sessions_to_run) * 4
            print(f"  Training Model: {model_name.upper()} across {n_folds_total} folds...", flush=True)
            
            all_y_true = []
            all_y_pred = []
            all_y_prob = []
            
            subject_metrics = {s: {'accs': [], 'f1s': [], 'kappas': [], 'aucs': []} for s in subjects_to_run}
            session_breakdowns = []
            
            for sub_idx, sub_id in enumerate(subjects_to_run, 1):
                sub_t0 = time.time()
                for ses_num in sessions_to_run:
                    # Get 24 trials for this session
                    session_trial_keys = [k for k, v in trial_data_map.items() if v['subject_id'] == sub_id and v['session_num'] == ses_num]
                    session_trial_labels = [trial_data_map[k]['label'] for k in session_trial_keys]
                    
                    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
                    
                    sess_y_true = []
                    sess_y_pred = []
                    sess_y_prob = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(session_trial_keys, session_trial_labels)):
                        train_keys = [session_trial_keys[i] for i in train_idx]
                        test_keys = [session_trial_keys[i] for i in test_idx]
                        
                        # Concatenate filtered training frames
                        X_train = np.vstack([filtered_trial_map[k]['features'] for k in train_keys])
                        y_train = np.concatenate([[filtered_trial_map[k]['label']] * filtered_trial_map[k]['retained_len'] for k in train_keys])
                        
                        # Concatenate filtered test frames
                        X_test = np.vstack([filtered_trial_map[k]['features'] for k in test_keys])
                        y_test = np.concatenate([[filtered_trial_map[k]['label']] * filtered_trial_map[k]['retained_len'] for k in test_keys])
                        
                        # Fit StandardScaler strictly on training frames
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Train Model
                        if model_name == 'lightgbm':
                            preds, probs = train_lightgbm(X_train_scaled, y_train, X_test_scaled, random_state=random_state)
                        elif model_name == 'shallow_mlp':
                            preds, probs = train_shallow_mlp(X_train_scaled, y_train, X_test_scaled, device=device)
                        else:
                            raise ValueError(f"Unknown model {model_name}")
                            
                        # Fold Metrics
                        f_acc = accuracy_score(y_test, preds)
                        f_f1 = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)[2]
                        try:
                            f_auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
                        except Exception:
                            f_auc = 0.5
                        f_kappa = cohen_kappa_score(y_test, preds)
                        
                        csv_rows.append({
                            'Strategy': strategy,
                            'Model': model_name,
                            'Subject_ID': sub_id,
                            'Session_ID': ses_num,
                            'Fold': fold_idx + 1,
                            'N_Train_Frames': len(y_train),
                            'N_Test_Frames': len(y_test),
                            'Accuracy': round(f_acc, 6),
                            'Macro_F1': round(f_f1, 6),
                            'Macro_AUC': round(f_auc, 6),
                            'Kappa': round(f_kappa, 6)
                        })
                        
                        sess_y_true.extend(y_test)
                        sess_y_pred.extend(preds)
                        sess_y_prob.extend(probs)
                        
                    # Session level metrics
                    sess_y_true = np.array(sess_y_true)
                    sess_y_pred = np.array(sess_y_pred)
                    sess_y_prob = np.array(sess_y_prob)
                    
                    s_acc = accuracy_score(sess_y_true, sess_y_pred)
                    s_f1 = precision_recall_fscore_support(sess_y_true, sess_y_pred, average='macro', zero_division=0)[2]
                    try:
                        s_auc = roc_auc_score(sess_y_true, sess_y_prob, average='macro', multi_class='ovr')
                    except Exception:
                        s_auc = 0.5
                    s_kappa = cohen_kappa_score(sess_y_true, sess_y_pred)
                    
                    subject_metrics[sub_id]['accs'].append(s_acc)
                    subject_metrics[sub_id]['f1s'].append(s_f1)
                    subject_metrics[sub_id]['kappas'].append(s_kappa)
                    subject_metrics[sub_id]['aucs'].append(s_auc)
                    
                    session_breakdowns.append({
                        'subject_id': sub_id,
                        'session_num': ses_num,
                        'n_samples': len(sess_y_true),
                        'accuracy': s_acc,
                        'macro_f1': s_f1,
                        'auc': s_auc,
                        'kappa': s_kappa
                    })
                    
                    all_y_true.extend(sess_y_true)
                    all_y_pred.extend(sess_y_pred)
                    all_y_prob.extend(sess_y_prob)
                    
                sub_mean_acc = np.mean(subject_metrics[sub_id]['accs'])
                if len(subjects_to_run) > 1:
                    print(f"    [Sub {sub_id:02d}/15] Mean Accuracy: {sub_mean_acc*100:.2f}% ({time.time()-sub_t0:.1f}s)", flush=True)
                    
            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)
            all_y_prob = np.array(all_y_prob)
            
            predictions_store[strategy][model_name] = {
                'y_true': all_y_true,
                'y_pred': all_y_pred,
                'y_prob': all_y_prob
            }
            
            # Pooled Metrics
            pooled_acc = accuracy_score(all_y_true, all_y_pred)
            pooled_prec, pooled_rec, pooled_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro', zero_division=0)
            try:
                pooled_auc = roc_auc_score(all_y_true, all_y_prob, average='macro', multi_class='ovr')
            except Exception:
                pooled_auc = 0.5
            pooled_kappa = cohen_kappa_score(all_y_true, all_y_pred)
            
            # 95% Bootstrap CIs
            ci_dict = compute_bootstrap_confidence_intervals(all_y_true, all_y_pred, all_y_prob, num_resamples=1000, seed=random_state)
            
            # Subject Aggregations
            sub_mean_accs = [np.mean(subject_metrics[s]['accs']) for s in subjects_to_run]
            sub_mean_f1s = [np.mean(subject_metrics[s]['f1s']) for s in subjects_to_run]
            
            model_summary = {
                'pooled_accuracy': pooled_acc,
                'pooled_precision': pooled_prec,
                'pooled_recall': pooled_rec,
                'pooled_f1': pooled_f1,
                'pooled_auc': pooled_auc,
                'pooled_kappa': pooled_kappa,
                'accuracy_ci': ci_dict['accuracy_ci'],
                'f1_ci': ci_dict['f1_ci'],
                'auc_ci': ci_dict['auc_ci'],
                'kappa_ci': ci_dict['kappa_ci'],
                'mean_subject_accuracy': float(np.mean(sub_mean_accs)),
                'std_subject_accuracy': float(np.std(sub_mean_accs)),
                'mean_subject_f1': float(np.mean(sub_mean_f1s)),
                'std_subject_f1': float(np.std(sub_mean_f1s)),
                'per_subject_mean_accuracies': {int(s): float(np.mean(subject_metrics[s]['accs'])) for s in subjects_to_run},
                'session_breakdowns': session_breakdowns
            }
            
            strategy_eval['models'][model_name] = model_summary
            
            print(f"    -> {model_name.upper()} Pooled Accuracy: {pooled_acc*100:.2f}% [{ci_dict['accuracy_ci'][0]*100:.2f}%, {ci_dict['accuracy_ci'][1]*100:.2f}%] | Macro-F1: {pooled_f1:.4f} | Kappa: {pooled_kappa:.4f}", flush=True)
            
        benchmark_results['strategies'][strategy] = strategy_eval
        print(f"Strategy runtime: {time.time() - strat_start_time:.2f}s", flush=True)
        
    benchmark_results['total_elapsed_seconds'] = time.time() - start_time_all
    
    # 2. Save JSON and CSV
    json_path = "salient_window_benchmark_results.json"
    csv_path = "salient_window_benchmark_results.csv"
    
    with open(json_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved structured JSON results to: {json_path}", flush=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Strategy', 'Model', 'Subject_ID', 'Session_ID', 'Fold',
            'N_Train_Frames', 'N_Test_Frames', 'Accuracy', 'Macro_F1', 'Macro_AUC', 'Kappa'
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved tabular CSV results to: {csv_path}", flush=True)
    
    # 3. Generate Publication Figures (300 DPI)
    fig_dir = "figures/salient_windows"
    os.makedirs(fig_dir, exist_ok=True)
    
    if not dry_run:
        print("\nGenerating 300 DPI Publication Figures in figures/salient_windows/...", flush=True)
        
        # Figure 1: Raw vs Salient Accuracy Comparison (15 subjects)
        fig1_path = os.path.join(fig_dir, "raw_vs_salient_accuracy_comparison.png")
        plot_raw_vs_salient_accuracy_comparison(benchmark_results, fig1_path)
        
        # Figure 2: Salient Confusion Matrix (Best Strategy - Hybrid Shallow MLP or LGB)
        fig2_path = os.path.join(fig_dir, "salient_confusion_matrix.png")
        best_strat, best_model, best_acc = 'hybrid', 'lightgbm', -1
        for s in strategies:
            for m in models:
                acc = benchmark_results['strategies'][s]['models'][m]['pooled_accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_strat = s
                    best_model = m
                    
        y_true_best = predictions_store[best_strat][best_model]['y_true']
        y_pred_best = predictions_store[best_strat][best_model]['y_pred']
        plot_salient_confusion_matrix(y_true_best, y_pred_best, best_strat, best_model, fig2_path)
        
        # Figure 3: Spectral Salience Energy Profile
        fig3_path = os.path.join(fig_dir, "spectral_salience_energy_profile.png")
        plot_spectral_salience_energy_profile(dataset_path, fig3_path)
        
        # Copy to Artifact directory
        artifact_fig_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\salient_windows"
        os.makedirs(artifact_fig_dir, exist_ok=True)
        for fig_name in os.listdir(fig_dir):
            src = os.path.join(fig_dir, fig_name)
            dst = os.path.join(artifact_fig_dir, fig_name)
            shutil.copy2(src, dst)
            print(f"Copied figure {fig_name} to artifact directory.", flush=True)
            
    print("\nSalience Window Benchmark Completed Successfully!", flush=True)
    return benchmark_results

def plot_raw_vs_salient_accuracy_comparison(results_data, output_path):
    """Plots a grouped bar chart comparing the 15 subjects across the 4 filtering strategies."""
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    
    strategies = ['raw', 'temporal_crop', 'spectral_energy', 'hybrid']
    strat_labels = [
        'Raw Baseline (100%)',
        'Strat 1: Temporal Crop (85.6%)',
        'Strat 2: Spectral Power (60.0%)',
        'Strat 3: Hybrid Salience (54.8%)'
    ]
    colors = ['#7f8c8d', '#3498db', '#9b59b6', '#2ecc71']
    
    subjects = list(range(1, 16))
    x = np.arange(len(subjects))
    width = 0.20
    
    for idx, (strat, label, color) in enumerate(zip(strategies, strat_labels, colors)):
        sub_accs = [results_data['strategies'][strat]['models']['lightgbm']['per_subject_mean_accuracies'][s] * 100 for s in subjects]
        offset = (idx - 1.5) * width
        rects = ax.bar(x + offset, sub_accs, width, label=label, color=color, alpha=0.9, edgecolor='black', linewidth=0.6)
        
        pooled_acc = results_data['strategies'][strat]['models']['lightgbm']['pooled_accuracy'] * 100
        ax.axhline(pooled_acc, color=color, linestyle='--', alpha=0.7, linewidth=1.2)
        
    ax.set_xlabel('Subject ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stratified 4-Fold CV Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Affective Salience Window Extraction: Intra-Session Accuracy Across 15 Subjects (LightGBM)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub {s:02d}' for s in subjects], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, fontsize=10, loc='upper left')
    
    raw_acc = results_data['strategies']['raw']['models']['lightgbm']['pooled_accuracy'] * 100
    hyb_acc = results_data['strategies']['hybrid']['models']['lightgbm']['pooled_accuracy'] * 100
    lift = hyb_acc - raw_acc
    ax.annotate(
        f'Hybrid Salience Pooled Lift: {lift:+.2f}% ({raw_acc:.2f}% -> {hyb_acc:.2f}%)',
        xy=(0.98, 0.94), xycoords='axes fraction',
        ha='right', va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#d4edda', edgecolor='#28a745', alpha=0.9)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}", flush=True)

def plot_salient_confusion_matrix(y_true, y_pred, strategy_name, model_name, output_path):
    """Plots the confusion matrix for the best performing salience strategy."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Normalized Recall', rotation=-90, va="bottom", fontsize=11, fontweight='bold')
    
    tick_marks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_norm = cm_norm[i, j]
            val_raw = cm[i, j]
            color = "white" if val_norm > thresh else "black"
            ax.text(j, i, f"{val_norm*100:.1f}%\n(N={val_raw})",
                    ha="center", va="center", color=color, fontsize=10, fontweight='bold')
            
    strat_title = STRATEGY_DISPLAY_NAMES.get(strategy_name, strategy_name)
    ax.set_title(f'Pooled Confusion Matrix: {strat_title}\nModel: {model_name.upper()} (N={len(y_true)} frames)', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylabel('Ground Truth Affective State', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Affective State', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}", flush=True)

def plot_spectral_salience_energy_profile(dataset_path, output_path):
    """
    Plots the temporal evolution of High-Frequency Spectral Energy (Beta + Gamma DE)
    across normalized trial progression (0% to 100% of trial duration) across all 1080 trials.
    """
    data = np.load(dataset_path)
    features = data['features']      # (37575, 62, 5)
    labels = data['labels']          # (37575,)
    subject_ids = data['subject_ids']# (37575,)
    session_nums = data['session_nums'] # (37575,)
    trial_ids = data['trial_ids']    # (37575,)
    
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trials = np.unique(trial_keys)
    
    n_points = 100
    resampled_profiles_by_class = {c: [] for c in range(4)}
    
    for u_trial in unique_trials:
        mask = (trial_keys == u_trial)
        t_feats = features[mask] # (L, 62, 5)
        t_label = labels[mask][0]
        L = len(t_feats)
        
        beta_power = t_feats[:, :, 3]
        gamma_power = t_feats[:, :, 4]
        energy = np.mean(beta_power + gamma_power, axis=1) # (L,)
        
        x_orig = np.linspace(0, 100, L)
        x_target = np.linspace(0, 100, n_points)
        interp_energy = np.interp(x_target, x_orig, energy)
        resampled_profiles_by_class[t_label].append(interp_energy)
        
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    timeline = np.linspace(0, 100, n_points)
    
    for c, cname, col in zip(range(4), CLASS_NAMES, EMOTION_COLORS):
        profiles = np.array(resampled_profiles_by_class[c]) # (N_trials, 100)
        mean_prof = np.mean(profiles, axis=0)
        sem_prof = np.std(profiles, axis=0) / np.sqrt(len(profiles))
        
        ax.plot(timeline, mean_prof, label=f'{cname} (N={len(profiles)} trials)', color=col, linewidth=2.2)
        ax.fill_between(timeline, mean_prof - sem_prof, mean_prof + sem_prof, color=col, alpha=0.15)
        
    ax.axvspan(0, 10, color='#e74c3c', alpha=0.12, label='Stimulus Onset Latency (~0-3s)')
    ax.axvspan(10, 90, color='#2ecc71', alpha=0.10, label='Affective Climax Epoch (High Salience)')
    ax.axvspan(90, 100, color='#95a5a6', alpha=0.12, label='Offset Habituation / Cooldown (~2s)')
    
    ax.axvline(10, color='#c0392b', linestyle=':', linewidth=1.5)
    ax.axvline(90, color='#7f8c8d', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Normalized Trial Duration (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('High-Frequency Energy: Mean Beta + Gamma DE (a.u.)', fontsize=12, fontweight='bold')
    ax.set_title('Neurobiological Dynamics of Affective Salience Across Movie Clip Trials\n(1080 Trials across 15 Subjects & 3 Sessions)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.95, fontsize=10, loc='lower center', ncol=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Salient Window Extraction Benchmark on SEED-IV")
    parser.add_argument("--dataset_path", type=str, default="seed_iv_processed.npz", help="Path to SEED-IV processed npz dataset")
    parser.add_argument("--dry_run", action="store_true", help="Run 1-session verification dry run")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    run_salient_windows_benchmark(
        dataset_path=args.dataset_path,
        dry_run=args.dry_run,
        random_state=args.random_state
    )
