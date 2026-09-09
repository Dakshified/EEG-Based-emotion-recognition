"""
Literature Paper Replication Benchmark Evaluator for SEED-IV EEG
================================================================
Replicates the exact sample-level shuffled 80/20 train/test protocol commonly 
found in 95%+ SEED-IV literature (e.g. Ahmadzadeh et al., Cheng et al., Hou et al.).

Protocol:
- Data: Differential Entropy (DE) features from seed_iv_processed.npz (310D, 4 classes)
- Split: For each of the 15 subjects, combine all sessions (~2,505 samples)
- Partitioning: Stratified 80/20 train/test split with random sample shuffling (seed=42)
- Zero-Leakage: StandardScaler fit strictly on the 80% train split
- Models:
  1. LightGBM Classifier (n_estimators=300, lr=0.05, num_leaves=31)
  2. Shallow MLP (310 -> 128 -> 64 -> 4, BatchNorm, GELU, Dropout 0.2, Adam, Cosine Annealing, 40 epochs)
- Outputs:
  - Per-subject and pooled accuracy, macro-F1, Cohen's kappa, ROC-AUC
  - 95% non-parametric bootstrap confidence intervals (1,000 resamples)
  - Structured JSON and CSV exports
  - Publication-ready 300 DPI figures in figures/paper_replication/
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
    auc
)
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

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

class ShallowMLP(nn.Module):
    """
    Shallow MLP Architecture matching standard affective computing literature:
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

def train_shallow_mlp(X_train, y_train, X_test, y_test, device, epochs=40, batch_size=64, lr=1e-3, weight_decay=1e-4):
    """Trains Shallow MLP with Adam and Cosine Annealing scheduler."""
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = ShallowMLP(in_features=310, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_samples = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            n_samples += len(by)
            
        scheduler.step()
        avg_loss = total_loss / max(n_samples, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    y_preds, y_probs, y_trues = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            y_preds.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            y_trues.extend(by.numpy())
            
    return np.array(y_trues), np.array(y_preds), np.array(y_probs)

def run_paper_replication_benchmark(dataset_path="seed_iv_processed.npz", random_state=42):
    """
    Executes the standard literature protocol across all 15 subjects individually.
    """
    set_seed(random_state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Paper Replication Benchmark on Device: {device}", flush=True)
    
    # 1. Load Data
    data = np.load(dataset_path)
    raw_features = data['features'] # (37575, 62, 5) or (37575, 310)
    if raw_features.ndim == 3:
        features_flat = raw_features.reshape(len(raw_features), -1) # (37575, 310)
    else:
        features_flat = raw_features
        
    labels = data['labels'] # (37575,)
    subject_ids = data['subject_ids'] # (37575,)
    
    print(f"Loaded {len(labels)} samples across {len(np.unique(subject_ids))} subjects (310 features).", flush=True)
    
    lgb_subject_results = []
    mlp_subject_results = []
    
    lgb_all_y_true, lgb_all_y_pred, lgb_all_y_prob = [], [], []
    mlp_all_y_true, mlp_all_y_pred, mlp_all_y_prob = [], [], []
    
    start_time = time.time()
    
    print("\n" + "="*85, flush=True)
    print("STARTING LITERATURE REPLICATION BENCHMARK (SAMPLE-LEVEL SHUFFLED 80/20 SPLIT)", flush=True)
    print("="*85, flush=True)
    
    for sub in range(1, 16):
        sub_mask = (subject_ids == sub)
        X_sub = features_flat[sub_mask]
        y_sub = labels[sub_mask]
        
        # Stratified 80/20 train/test split with sample shuffling
        X_train, X_test, y_train, y_test = train_test_split(
            X_sub, y_sub, test_size=0.20, shuffle=True, stratify=y_sub, random_state=random_state
        )
        
        # Zero-leakage standard scaling fit exclusively on training split
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Model A: LightGBM
        lgb_model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        y_pred_lgb = lgb_model.predict(X_test_scaled)
        y_prob_lgb = lgb_model.predict_proba(X_test_scaled)
        
        lgb_acc = accuracy_score(y_test, y_pred_lgb)
        lgb_f1 = precision_recall_fscore_support(y_test, y_pred_lgb, average='macro', zero_division=0)[2]
        lgb_kappa = cohen_kappa_score(y_test, y_pred_lgb)
        
        lgb_subject_results.append({
            'subject': sub,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(lgb_acc),
            'macro_f1': float(lgb_f1),
            'kappa': float(lgb_kappa)
        })
        
        lgb_all_y_true.extend(y_test)
        lgb_all_y_pred.extend(y_pred_lgb)
        lgb_all_y_prob.extend(y_prob_lgb)
        
        # 2. Model B: Shallow MLP
        y_true_mlp, y_pred_mlp, y_prob_mlp = train_shallow_mlp(
            X_train_scaled, y_train, X_test_scaled, y_test, device,
            epochs=40, batch_size=64, lr=1e-3, weight_decay=1e-4
        )
        
        mlp_acc = accuracy_score(y_true_mlp, y_pred_mlp)
        mlp_f1 = precision_recall_fscore_support(y_true_mlp, y_pred_mlp, average='macro', zero_division=0)[2]
        mlp_kappa = cohen_kappa_score(y_true_mlp, y_pred_mlp)
        
        mlp_subject_results.append({
            'subject': sub,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(mlp_acc),
            'macro_f1': float(mlp_f1),
            'kappa': float(mlp_kappa)
        })
        
        mlp_all_y_true.extend(y_true_mlp)
        mlp_all_y_pred.extend(y_pred_mlp)
        mlp_all_y_prob.extend(y_prob_mlp)
        
        print(f"Subject {sub:02d}/15 (N_test={len(X_test):3d}) -> LightGBM: {lgb_acc*100:6.2f}% (F1: {lgb_f1:.4f}) | Shallow MLP: {mlp_acc*100:6.2f}% (F1: {mlp_f1:.4f})", flush=True)
        
    total_elapsed = time.time() - start_time
    
    # Evaluate Pooled Metrics
    lgb_all_y_true = np.array(lgb_all_y_true)
    lgb_all_y_pred = np.array(lgb_all_y_pred)
    lgb_all_y_prob = np.array(lgb_all_y_prob)
    
    mlp_all_y_true = np.array(mlp_all_y_true)
    mlp_all_y_pred = np.array(mlp_all_y_pred)
    mlp_all_y_prob = np.array(mlp_all_y_prob)
    
    lgb_pooled = evaluate_metrics(lgb_all_y_true, lgb_all_y_pred, lgb_all_y_prob, compute_ci=True)
    lgb_pooled['mean_subject_acc'] = float(np.mean([s['accuracy'] for s in lgb_subject_results]))
    lgb_pooled['std_subject_acc'] = float(np.std([s['accuracy'] for s in lgb_subject_results]))
    lgb_pooled['mean_subject_f1'] = float(np.mean([s['macro_f1'] for s in lgb_subject_results]))
    lgb_pooled['std_subject_f1'] = float(np.std([s['macro_f1'] for s in lgb_subject_results]))
    lgb_pooled['per_subject_breakdown'] = lgb_subject_results
    
    mlp_pooled = evaluate_metrics(mlp_all_y_true, mlp_all_y_pred, mlp_all_y_prob, compute_ci=True)
    mlp_pooled['mean_subject_acc'] = float(np.mean([s['accuracy'] for s in mlp_subject_results]))
    mlp_pooled['std_subject_acc'] = float(np.std([s['accuracy'] for s in mlp_subject_results]))
    mlp_pooled['mean_subject_f1'] = float(np.mean([s['macro_f1'] for s in mlp_subject_results]))
    mlp_pooled['std_subject_f1'] = float(np.std([s['macro_f1'] for s in mlp_subject_results]))
    mlp_pooled['per_subject_breakdown'] = mlp_subject_results
    
    print("\n" + "="*85, flush=True)
    print("FINAL VERIFIED LITERATURE REPLICATION RESULTS (SAMPLE-LEVEL SHUFFLED 80/20)", flush=True)
    print("="*85, flush=True)
    print(f"--- Model A: LightGBM Classifier ---", flush=True)
    print(f"Pooled Accuracy:    {lgb_pooled['accuracy']*100:6.2f}% [95% CI: {lgb_pooled['accuracy_ci'][0]*100:.2f}%, {lgb_pooled['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"Per-Subject Mean:   {lgb_pooled['mean_subject_acc']*100:6.2f}% +/- {lgb_pooled['std_subject_acc']*100:.2f}%", flush=True)
    print(f"Macro-F1 Score:     {lgb_pooled['f1']:.4f} [95% CI: {lgb_pooled['f1_ci'][0]:.4f}, {lgb_pooled['f1_ci'][1]:.4f}]", flush=True)
    print(f"Macro ROC-AUC:      {lgb_pooled['auc']:.4f} [95% CI: {lgb_pooled['auc_ci'][0]:.4f}, {lgb_pooled['auc_ci'][1]:.4f}]", flush=True)
    print(f"Cohen's Kappa:      {lgb_pooled['kappa']:.4f} [95% CI: {lgb_pooled['kappa_ci'][0]:.4f}, {lgb_pooled['kappa_ci'][1]:.4f}]", flush=True)
    
    print(f"\n--- Model B: Shallow MLP (Neural Baseline) ---", flush=True)
    print(f"Pooled Accuracy:    {mlp_pooled['accuracy']*100:6.2f}% [95% CI: {mlp_pooled['accuracy_ci'][0]*100:.2f}%, {mlp_pooled['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"Per-Subject Mean:   {mlp_pooled['mean_subject_acc']*100:6.2f}% +/- {mlp_pooled['std_subject_acc']*100:.2f}%", flush=True)
    print(f"Macro-F1 Score:     {mlp_pooled['f1']:.4f} [95% CI: {mlp_pooled['f1_ci'][0]:.4f}, {mlp_pooled['f1_ci'][1]:.4f}]", flush=True)
    print(f"Macro ROC-AUC:      {mlp_pooled['auc']:.4f} [95% CI: {mlp_pooled['auc_ci'][0]:.4f}, {mlp_pooled['auc_ci'][1]:.4f}]", flush=True)
    print(f"Cohen's Kappa:      {mlp_pooled['kappa']:.4f} [95% CI: {mlp_pooled['kappa_ci'][0]:.4f}, {mlp_pooled['kappa_ci'][1]:.4f}]", flush=True)
    print(f"\nTotal Runtime:      {total_elapsed:.2f}s", flush=True)
    print("="*85, flush=True)
    
    # Save Structured Results
    results_dict = {
        'protocol': 'Sample-Level Shuffled Stratified 80/20 Train/Test Split (Standard Literature Benchmark)',
        'dataset': 'SEED-IV (62 Channels x 5 Frequency Bands = 310 DE Features, 4 Classes)',
        'total_samples': len(labels),
        'total_test_samples': len(lgb_all_y_true),
        'elapsed_seconds': total_elapsed,
        'lightgbm': lgb_pooled,
        'shallow_mlp': mlp_pooled
    }
    
    with open("paper_replication_benchmark_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    print("Saved JSON results to 'paper_replication_benchmark_results.json'", flush=True)
    
    # Save CSV Results
    with open("paper_replication_benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Model", "N_Train", "N_Test", "Accuracy", "Macro_F1", "Cohen_Kappa"])
        for s in lgb_subject_results:
            writer.writerow([s['subject'], "LightGBM", s['n_train'], s['n_test'], f"{s['accuracy']:.4f}", f"{s['macro_f1']:.4f}", f"{s['kappa']:.4f}"])
        for s in mlp_subject_results:
            writer.writerow([s['subject'], "Shallow_MLP", s['n_train'], s['n_test'], f"{s['accuracy']:.4f}", f"{s['macro_f1']:.4f}", f"{s['kappa']:.4f}"])
        writer.writerow(["Pooled", "LightGBM", len(features_flat) - len(lgb_all_y_true), len(lgb_all_y_true), f"{lgb_pooled['accuracy']:.4f}", f"{lgb_pooled['f1']:.4f}", f"{lgb_pooled['kappa']:.4f}"])
        writer.writerow(["Pooled", "Shallow_MLP", len(features_flat) - len(mlp_all_y_true), len(mlp_all_y_true), f"{mlp_pooled['accuracy']:.4f}", f"{mlp_pooled['f1']:.4f}", f"{mlp_pooled['kappa']:.4f}"])
    print("Saved CSV results to 'paper_replication_benchmark_results.csv'", flush=True)
    
    # Generate Figures
    generate_figures(lgb_subject_results, mlp_subject_results, (lgb_all_y_true, lgb_all_y_pred, lgb_all_y_prob), (mlp_all_y_true, mlp_all_y_pred, mlp_all_y_prob))
    
    return results_dict

def generate_figures(lgb_sub, mlp_sub, lgb_data, mlp_data, output_dir="figures/paper_replication"):
    """Exports 300 DPI publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # 1. Per-Subject Accuracy Bar Chart
    subjects = np.arange(1, 16)
    lgb_accs = [s['accuracy'] * 100 for s in lgb_sub]
    mlp_accs = [s['accuracy'] * 100 for s in mlp_sub]
    
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    width = 0.38
    x = np.arange(len(subjects))
    
    rects1 = ax.bar(x - width/2, lgb_accs, width, label='LightGBM', color='#2980b9', edgecolor='#1f618d', alpha=0.9)
    rects2 = ax.bar(x + width/2, mlp_accs, width, label='Shallow MLP', color='#27ae60', edgecolor='#1e8449', alpha=0.9)
    
    # Add horizontal lines for mean accuracy
    ax.axhline(np.mean(lgb_accs), color='#2980b9', linestyle='--', lw=1.5, alpha=0.8, label=f'LightGBM Mean ({np.mean(lgb_accs):.1f}%)')
    ax.axhline(np.mean(mlp_accs), color='#27ae60', linestyle=':', lw=1.8, alpha=0.8, label=f'MLP Mean ({np.mean(mlp_accs):.1f}%)')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('SEED-IV Subject ID', fontsize=12, fontweight='bold')
    ax.set_title('SEED-IV Literature Replication Benchmark: Per-Subject Accuracy\n(Sample-Level Shuffled 80/20 Stratified Train/Test Split)', fontsize=13, pad=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub {s:02d}' for s in subjects], fontsize=10, fontweight='bold')
    ax.set_ylim([70, 103])
    ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Annotate bar values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, rotation=90)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7.5, rotation=90)
                    
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "per_subject_accuracy_bar_chart.png")
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Saved: {bar_path}", flush=True)
    
    # 2. Side-by-Side Normalized Confusion Matrices
    y_true_lgb, y_pred_lgb, y_prob_lgb = lgb_data
    y_true_mlp, y_pred_mlp, y_prob_mlp = mlp_data
    
    cm_lgb = confusion_matrix(y_true_lgb, y_pred_lgb, normalize='true') * 100
    cm_mlp = confusion_matrix(y_true_mlp, y_pred_mlp, normalize='true') * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    for ax, cm, title, cmap in zip(axes, [cm_lgb, cm_mlp], ['(A) LightGBM Classifier', '(B) Shallow MLP Network'], [plt.cm.Blues, plt.cm.Greens]):
        cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
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
                        
        ax.set_title(f"{title}\nNormalized Confusion Matrix (%)", fontsize=12, pad=10, fontweight='bold')
        ax.set_xlabel("Predicted Class", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Class", fontsize=11, fontweight='bold')
        
    plt.suptitle("SEED-IV Literature Protocol (Sample-Level 80/20 Shuffled Split)", fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "literature_replication_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}", flush=True)
    
    # 3. Multi-Class ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    for ax, y_t, y_p, title in zip(axes, [y_true_lgb, y_true_mlp], [y_prob_lgb, y_prob_mlp], ['(A) LightGBM ROC Curves', '(B) Shallow MLP ROC Curves']):
        for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, EMOTION_COLORS)):
            y_binary = (y_t == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_p[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{cls_name} (AUC = {roc_auc:.4f})")
            
        ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Random Chance (0.50)')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=11, fontweight='bold')
        ax.set_ylabel("True Positive Rate", fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, pad=10, fontweight='bold')
        ax.legend(loc="lower right", fontsize=9.5, frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.suptitle("SEED-IV Literature Replication Multi-Class ROC Curves", fontsize=13, y=1.02, fontweight='bold')
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "literature_replication_roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Literature Paper Replication Benchmark on SEED-IV")
    parser.add_argument('--dataset', type=str, default='seed_iv_processed.npz', help="Path to processed SEED-IV dataset")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    run_paper_replication_benchmark(dataset_path=args.dataset, random_state=args.seed)
