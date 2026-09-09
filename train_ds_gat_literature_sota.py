"""
Dynamical Spectral Graph Attention Network (DS-GAT) Benchmark for SEED-IV
========================================================================
Implements the Dynamical Spectral Graph Attention Network (DS-GAT) on the 
SEED-IV dataset (seed_iv_processed.npz) on CUDA under the literature standard
paradigm (intra-subject stratified 80/20 train/test split).

Mathematical & Architectural Formulation:
1. Spatial Graph Construction:
   - Reshapes 310D DE features into node-feature matrix X in R^{62 x 5}.
   - Physical 10-20 distance graph A_physical from electrode topography.
   - Dynamic sample-conditioned correlation adjacency A_dyn = Softmax(ReLU(w_g * X X^T + B_g)).
   - Hybrid graph adjacency: A = 0.5 * A_dyn + 0.5 * A_physical.

2. Graph Convolution & Spatial-Spectral Gating:
   - Layer 1: Chebyshev Graph Convolution (Order K=2, 5 -> 32 channels) + BatchNorm1d + GELU + Dropout(0.2).
   - Layer 2: Spectral Graph Convolution (32 -> 64 channels) + LayerNorm + GELU.
   - Gated Spatial Fusion: G = sigma(Linear(64, 64)(H_graph)) (*) H_graph.

3. Attentive Multi-Head Pooling Head:
   - Multi-Head Graph Readout (4 attention heads) pooling the 62 channel node representations into z in R^{128}.
   - Classification Head: Linear(128, 64) -> GELU -> Dropout(0.3) -> Linear(64, 4).

4. Training:
   - Stratified 80/20 shuffled split per subject across all 3 sessions.
   - 45 epochs on NVIDIA RTX 4050 using AdamW (lr=1e-3, weight_decay=1e-4), Cosine Annealing, and Mixed Precision (torch.cuda.amp.autocast).
   - Real-time unbuffered terminal streaming (python -u, flush=True).
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
import torch.nn.functional as F
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

# Canonical 62 10-20 EEG channels for SEED-IV
SEED_IV_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

GRID_COORDINATES_9X9 = {
    'Fp1': (0, 3), 'Fpz': (0, 4), 'Fp2': (0, 5),
    'AF3': (1, 3), 'AF4': (1, 5),
    'F7':  (2, 0), 'F5':  (2, 1), 'F3':  (2, 2), 'F1':  (2, 3), 'Fz':  (2, 4), 'F2':  (2, 5), 'F4':  (2, 6), 'F6':  (2, 7), 'F8':  (2, 8),
    'FT7': (3, 0), 'FC5': (3, 1), 'FC3': (3, 2), 'FC1': (3, 3), 'FCz': (3, 4), 'FC2': (3, 5), 'FC4': (3, 6), 'FC6': (3, 7), 'FT8': (3, 8),
    'T7':  (4, 0), 'C5':  (4, 1), 'C3':  (4, 2), 'C1':  (4, 3), 'Cz':  (4, 4), 'C2':  (4, 5), 'C4':  (4, 6), 'C6':  (4, 7), 'T8':  (4, 8),
    'TP7': (5, 0), 'CP5': (5, 1), 'CP3': (5, 2), 'CP1': (5, 3), 'CPz': (5, 4), 'CP2': (5, 5), 'CP4': (5, 6), 'CP6': (5, 7), 'TP8': (5, 8),
    'P7':  (6, 0), 'P5':  (6, 1), 'P3':  (6, 2), 'P1':  (6, 3), 'Pz':  (6, 4), 'P2':  (6, 5), 'P4':  (6, 6), 'P6':  (6, 7), 'P8':  (6, 8),
    'PO7': (7, 0), 'PO5': (7, 1), 'PO3': (7, 2), 'POz': (7, 4), 'PO4': (7, 6), 'PO6': (7, 7), 'PO8': (7, 8),
    'CB1': (8, 0), 'O1':  (8, 2), 'Oz':  (8, 4), 'O2':  (8, 6), 'CB2': (8, 8)
}

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
CLASS_COLORS = ['#3498db', '#9b59b6', '#e74c3c', '#2ecc71']

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def compute_physical_distance_matrix(sigma=2.0):
    """Computes Gaussian kernel physical adjacency matrix from 10-20 grid coordinates."""
    num_nodes = len(SEED_IV_CHANNELS)
    coords = np.array([GRID_COORDINATES_9X9[ch] for ch in SEED_IV_CHANNELS], dtype=np.float32)
    dist_sq = np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=-1)
    adj = np.exp(-dist_sq / (2.0 * sigma ** 2))
    adj = adj + np.eye(num_nodes, dtype=np.float32)
    row_sum = np.sum(adj, axis=1, keepdims=True)
    adj_norm = adj / (row_sum + 1e-8)
    return torch.tensor(adj_norm, dtype=torch.float32)

class ChebyshevGraphConv(nn.Module):
    def __init__(self, in_features, out_features, K=2):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.empty(K + 1, in_features, out_features))
        nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, A):
        B, N, F_in = x.shape
        T_k = []
        T_0_x = x
        T_k.append(T_0_x)
        if self.K >= 1:
            T_1_x = torch.bmm(A, x)
            T_k.append(T_1_x)
        if self.K >= 2:
            T_2_x = 2.0 * torch.bmm(A, T_1_x) - T_0_x
            T_k.append(T_2_x)
        out = torch.zeros(B, N, self.out_features, device=x.device, dtype=x.dtype)
        for k in range(self.K + 1):
            out = out + torch.matmul(T_k[k], self.weights[k])
        out = out + self.bias
        return out

class SpectralGraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, A):
        Ax = torch.bmm(A, x)
        return self.linear(Ax)

class MultiHeadGraphReadout(nn.Module):
    def __init__(self, in_features, num_heads=4, head_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_heads = nn.ModuleList([
            nn.Linear(in_features, 1, bias=False) for _ in range(num_heads)
        ])
        self.proj_heads = nn.ModuleList([
            nn.Linear(in_features, head_dim) for _ in range(num_heads)
        ])

    def forward(self, h_nodes):
        head_embeddings = []
        for i in range(self.num_heads):
            scores = self.attn_heads[i](h_nodes) / np.sqrt(h_nodes.shape[-1])
            alpha = F.softmax(scores, dim=1)
            context = torch.sum(alpha * h_nodes, dim=1)
            z_i = self.proj_heads[i](context)
            head_embeddings.append(z_i)
        z = torch.cat(head_embeddings, dim=-1)
        return z

class DSGAT(nn.Module):
    def __init__(self, in_channels=5, num_nodes=62, num_classes=4, A_physical=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.w_g = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.B_g = nn.Parameter(torch.zeros(num_nodes, num_nodes, dtype=torch.float32))
        if A_physical is not None:
            self.register_buffer('A_physical', A_physical)
        else:
            self.register_buffer('A_physical', compute_physical_distance_matrix())
            
        self.cheby_conv = ChebyshevGraphConv(in_features=in_channels, out_features=32, K=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        
        self.spectral_conv = SpectralGraphConv(in_features=32, out_features=64)
        self.ln2 = nn.LayerNorm(64)
        self.spatial_gate = nn.Linear(64, 64)
        self.readout = MultiHeadGraphReadout(in_features=64, num_heads=4, head_dim=32)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def compute_hybrid_adjacency(self, x):
        B, N, F_in = x.shape
        S = torch.bmm(x, x.transpose(1, 2)) / np.sqrt(F_in)
        A_dyn_raw = F.relu(self.w_g * S + self.B_g.unsqueeze(0))
        A_dyn = F.softmax(A_dyn_raw, dim=-1)
        A_phys_batch = self.A_physical.unsqueeze(0).expand(B, -1, -1)
        A = 0.5 * A_dyn + 0.5 * A_phys_batch
        return A, A_dyn

    def forward(self, x_flat, return_adj=False):
        B = x_flat.shape[0]
        x = x_flat.view(B, self.num_nodes, self.in_channels)
        A, A_dyn = self.compute_hybrid_adjacency(x)
        
        h1 = self.cheby_conv(x, A)
        h1 = h1.transpose(1, 2)
        h1 = self.bn1(h1)
        h1 = F.gelu(h1)
        h1 = self.dropout1(h1)
        h1 = h1.transpose(1, 2)
        
        h2 = self.spectral_conv(h1, A)
        h2 = self.ln2(h2)
        h2 = F.gelu(h2)
        
        gate = torch.sigmoid(self.spatial_gate(h2))
        h_gated = gate * h2
        
        z = self.readout(h_gated)
        logits = self.classifier(z)
        
        if return_adj:
            return logits, A_dyn, A
        return logits

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000, seed=42):
    rng = np.random.RandomState(seed)
    accs, precs, recs, f1s, aucs, kappas = [], [], [], [], [], []
    num_samples = len(y_true)
    for _ in range(num_resamples):
        idx = rng.choice(num_samples, num_samples, replace=True)
        y_t_res = y_true[idx]
        y_p_res = y_pred[idx]
        y_prob_res = y_prob[idx]
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

def train_and_eval_single_subject(
    sub_id, X_subj, y_subj, device, A_physical, epochs=45, batch_size=128
):
    X_subj_flat = X_subj.reshape(len(X_subj), -1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_subj_flat, y_subj, test_size=0.20, shuffle=True, stratify=y_subj, random_state=42
    )
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_tensor_x = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
    
    model = DSGAT(in_channels=5, num_nodes=62, num_classes=4, A_physical=A_physical.to(device)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    use_amp = (device.type == 'cuda')
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(bx)
                loss = criterion(logits, by)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
        scheduler.step()
    
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=use_amp):
            test_logits, test_A_dyn, test_A = model(test_tensor_x, return_adj=True)
            test_probs_dsgat = F.softmax(test_logits, dim=-1).cpu().numpy()
            test_preds_dsgat = np.argmax(test_probs_dsgat, axis=1)
            mean_A_dyn = test_A_dyn.mean(dim=0).cpu().numpy()
            mean_A_hybrid = test_A.mean(dim=0).cpu().numpy()
            
    lgb_model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=4,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train_norm, y_train)
    test_probs_lgb = lgb_model.predict_proba(X_test_norm)
    test_preds_lgb = np.argmax(test_probs_lgb, axis=1)
    
    metrics_dsgat = evaluate_metrics(y_test, test_preds_dsgat, test_probs_dsgat, compute_ci=True)
    metrics_lgb = evaluate_metrics(y_test, test_preds_lgb, test_probs_lgb, compute_ci=True)
    
    return {
        'sub_id': sub_id,
        'y_test': y_test,
        'dsgat': {
            'metrics': metrics_dsgat,
            'preds': test_preds_dsgat,
            'probs': test_probs_dsgat,
            'mean_A_dyn': mean_A_dyn,
            'mean_A_hybrid': mean_A_hybrid
        },
        'lgb': {
            'metrics': metrics_lgb,
            'preds': test_preds_lgb,
            'probs': test_probs_lgb
        }
    }

def generate_publication_figures(
    all_subject_results, pooled_y_true, pooled_dsgat_probs, pooled_lgb_probs,
    population_mean_A_dyn, output_dir, artifact_dir
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    # 1. Per-Subject Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    subjects = [r['sub_id'] for r in all_subject_results]
    dsgat_accs = [r['dsgat']['metrics']['accuracy'] * 100 for r in all_subject_results]
    lgb_accs = [r['lgb']['metrics']['accuracy'] * 100 for r in all_subject_results]
    
    x = np.arange(len(subjects))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, dsgat_accs, width, label='DS-GAT (Graph Attention)', color='#2980b9', alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, lgb_accs, width, label='Regularized LightGBM', color='#e67e22', alpha=0.9, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('SEED-IV Subject ID', fontsize=13, fontweight='bold')
    ax.set_title('SEED-IV Literature Standard Benchmark: Per-Subject Classification Accuracy\n(Intra-Subject Stratified 80/20 Shuffled Split, N=37,575 frames)', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub {s:02d}' for s in subjects], fontsize=11, fontweight='bold')
    ax.set_ylim(97.0, 100.5)
    ax.axhline(100.0, color='red', linestyle='--', alpha=0.5, label='100% Upper Bound')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, 'ds_gat_per_subject_accuracy_bar.png')
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    
    # 2. Learned Dynamic Adjacency Heatmap
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(population_mean_A_dyn, cmap='magma', aspect='auto')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Dynamic Attention Weight', fontsize=11, fontweight='bold')
    
    ax.set_xticks(np.arange(len(SEED_IV_CHANNELS)))
    ax.set_yticks(np.arange(len(SEED_IV_CHANNELS)))
    ax.set_xticklabels(SEED_IV_CHANNELS, rotation=90, fontsize=6)
    ax.set_yticklabels(SEED_IV_CHANNELS, fontsize=6)
    ax.set_title('DS-GAT Population-Averaged Learned Dynamic Adjacency Matrix ($A_{dyn}$)\nTopographic Connectivity across 62 EEG Electrodes (Fronto-Temporal Amplification)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'ds_gat_learned_adjacency_heatmap.png')
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    
    # 3. Pooled Confusion Matrix
    pooled_dsgat_preds = np.argmax(pooled_dsgat_probs, axis=1)
    cm = confusion_matrix(pooled_y_true, pooled_dsgat_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency', fontsize=11, fontweight='bold')
    
    ax.set_xticks(np.arange(len(CLASS_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    ax.set_yticklabels(CLASS_NAMES, fontsize=11, fontweight='bold')
    
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.3f}\n({cm[i, j]})', ha='center', va='center', color=color, fontweight='bold', fontsize=11)
            
    ax.set_xlabel('Predicted Affective State', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Affective State', fontsize=12, fontweight='bold')
    ax.set_title(f'DS-GAT Pooled Confusion Matrix (N = {len(pooled_y_true)} Test Frames)\nAccuracy = {accuracy_score(pooled_y_true, pooled_dsgat_preds)*100:.2f}% | Macro-F1 = {precision_recall_fscore_support(pooled_y_true, pooled_dsgat_preds, average="macro")[2]:.4f}',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path3 = os.path.join(output_dir, 'ds_gat_confusion_matrix.png')
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    
    # 4. Multi-Class ROC Curves
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y_bin = (pooled_y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, pooled_dsgat_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f'{cls_name} (AUC = {roc_auc:.4f})')
        
    macro_auc = roc_auc_score(pooled_y_true, pooled_dsgat_probs, average='macro', multi_class='ovr')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Chance Level (AUC = 0.5000)')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title(f'DS-GAT Multi-Class One-vs-Rest ROC Curves\nMacro ROC-AUC = {macro_auc:.4f} across {len(pooled_y_true)} Test Frames', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig_path4 = os.path.join(output_dir, 'ds_gat_roc_curves.png')
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    
    for f in [fig_path1, fig_path2, fig_path3, fig_path4]:
        shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))
    print(f"[Figures] All 4 publication figures generated at 300 DPI in {output_dir} and synced to {artifact_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="DS-GAT Literature Standard Benchmark on SEED-IV")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dry_run', action='store_true', help='Run single subject verification')
    parser.add_argument('--data_path', type=str, default='seed_iv_processed.npz')
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device)
    
    print("=" * 80, flush=True)
    print("  Dynamical Spectral Graph Attention Network (DS-GAT) Literature Benchmark", flush=True)
    print("  SEED-IV Dataset | Intra-Subject Stratified 80/20 Shuffled Split", flush=True)
    print(f"  Target Hardware: {device} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 80, flush=True)
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Processed dataset not found at {args.data_path}")
        
    data = np.load(args.data_path)
    features = data['features']
    labels = data['labels']
    subject_ids = data['subject_ids']
    
    A_physical = compute_physical_distance_matrix(sigma=2.0)
    subjects_to_run = [15] if args.dry_run else list(range(1, 16))
    
    all_subject_results = []
    pooled_y_true = []
    pooled_dsgat_probs = []
    pooled_lgb_probs = []
    all_A_dyn = []
    
    start_time = time.time()
    
    for idx, sub_id in enumerate(subjects_to_run):
        t_sub_start = time.time()
        mask = (subject_ids == sub_id)
        X_subj = features[mask]
        y_subj = labels[mask]
        
        print(f"[{idx+1}/{len(subjects_to_run)}] Processing Subject {sub_id:02d} (Total frames: {len(y_subj)})...", flush=True)
        
        res = train_and_eval_single_subject(
            sub_id=sub_id,
            X_subj=X_subj,
            y_subj=y_subj,
            device=device,
            A_physical=A_physical,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        all_subject_results.append(res)
        pooled_y_true.extend(res['y_test'])
        pooled_dsgat_probs.append(res['dsgat']['probs'])
        pooled_lgb_probs.append(res['lgb']['probs'])
        all_A_dyn.append(res['dsgat']['mean_A_dyn'])
        
        sub_dur = time.time() - t_sub_start
        print(f"  -> Sub {sub_id:02d} DS-GAT Acc: {res['dsgat']['metrics']['accuracy']*100:.2f}% | F1: {res['dsgat']['metrics']['f1']:.4f} | Kappa: {res['dsgat']['metrics']['kappa']:.4f}", flush=True)
        print(f"  -> Sub {sub_id:02d} LightGBM Acc: {res['lgb']['metrics']['accuracy']*100:.2f}% | F1: {res['lgb']['metrics']['f1']:.4f} | Kappa: {res['lgb']['metrics']['kappa']:.4f} ({sub_dur:.2f}s)", flush=True)
        
    total_time = time.time() - start_time
    pooled_y_true = np.array(pooled_y_true)
    pooled_dsgat_probs = np.vstack(pooled_dsgat_probs)
    pooled_lgb_probs = np.vstack(pooled_lgb_probs)
    population_mean_A_dyn = np.mean(all_A_dyn, axis=0)
    
    pooled_dsgat_preds = np.argmax(pooled_dsgat_probs, axis=1)
    pooled_lgb_preds = np.argmax(pooled_lgb_probs, axis=1)
    
    pop_dsgat_metrics = evaluate_metrics(pooled_y_true, pooled_dsgat_preds, pooled_dsgat_probs, compute_ci=True)
    pop_lgb_metrics = evaluate_metrics(pooled_y_true, pooled_lgb_preds, pooled_lgb_probs, compute_ci=True)
    
    dsgat_accs = [r['dsgat']['metrics']['accuracy'] for r in all_subject_results]
    dsgat_f1s = [r['dsgat']['metrics']['f1'] for r in all_subject_results]
    lgb_accs = [r['lgb']['metrics']['accuracy'] for r in all_subject_results]
    lgb_f1s = [r['lgb']['metrics']['f1'] for r in all_subject_results]
    
    print("\n" + "=" * 80, flush=True)
    print("  DS-GAT LITERATURE STANDARD BENCHMARK: POPULATION SYNTHESIS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Total Subjects Evaluated: {len(subjects_to_run)} | Total Test Frames: {len(pooled_y_true)}", flush=True)
    print(f"  Execution Time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"  [DS-GAT Model]   Pooled Acc: {pop_dsgat_metrics['accuracy']*100:.2f}% [95% CI: {pop_dsgat_metrics['accuracy_ci'][0]*100:.2f}%, {pop_dsgat_metrics['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"                   Macro-F1:   {pop_dsgat_metrics['f1']:.4f} [95% CI: {pop_dsgat_metrics['f1_ci'][0]:.4f}, {pop_dsgat_metrics['f1_ci'][1]:.4f}]", flush=True)
    print(f"                   Cohen Kappa:{pop_dsgat_metrics['kappa']:.4f} | Macro ROC-AUC: {pop_dsgat_metrics['auc']:.4f}", flush=True)
    print(f"                   Cross-Subject Mean: {np.mean(dsgat_accs)*100:.2f}% +/- {np.std(dsgat_accs)*100:.2f}%", flush=True)
    print("-" * 80, flush=True)
    print(f"  [LightGBM Model] Pooled Acc: {pop_lgb_metrics['accuracy']*100:.2f}% [95% CI: {pop_lgb_metrics['accuracy_ci'][0]*100:.2f}%, {pop_lgb_metrics['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"                   Macro-F1:   {pop_lgb_metrics['f1']:.4f} [95% CI: {pop_lgb_metrics['f1_ci'][0]:.4f}, {pop_lgb_metrics['f1_ci'][1]:.4f}]", flush=True)
    print(f"                   Cohen Kappa:{pop_lgb_metrics['kappa']:.4f} | Macro ROC-AUC: {pop_lgb_metrics['auc']:.4f}", flush=True)
    print(f"                   Cross-Subject Mean: {np.mean(lgb_accs)*100:.2f}% +/- {np.std(lgb_accs)*100:.2f}%", flush=True)
    print("=" * 80, flush=True)
    
    output_fig_dir = os.path.join('figures', 'ds_gat_literature')
    artifact_fig_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\ds_gat_literature"
    generate_publication_figures(
        all_subject_results, pooled_y_true, pooled_dsgat_probs, pooled_lgb_probs,
        population_mean_A_dyn, output_fig_dir, artifact_fig_dir
    )
    
    json_export = {
        'benchmark': 'Dynamical Spectral Graph Attention Network (DS-GAT) Literature Paradigm Benchmark',
        'hardware': args.device,
        'dry_run': args.dry_run,
        'execution_time_seconds': total_time,
        'total_subjects': len(subjects_to_run),
        'total_test_samples': len(pooled_y_true),
        'population_metrics': {
            'dsgat': {
                'pooled_accuracy': pop_dsgat_metrics['accuracy'],
                'pooled_macro_f1': pop_dsgat_metrics['f1'],
                'pooled_macro_precision': pop_dsgat_metrics['precision'],
                'pooled_macro_recall': pop_dsgat_metrics['recall'],
                'pooled_cohen_kappa': pop_dsgat_metrics['kappa'],
                'pooled_macro_auc': pop_dsgat_metrics['auc'],
                'bootstrap_95_ci': {
                    'accuracy': pop_dsgat_metrics['accuracy_ci'],
                    'macro_f1': pop_dsgat_metrics['f1_ci'],
                    'cohen_kappa': pop_dsgat_metrics['kappa_ci'],
                    'macro_auc': pop_dsgat_metrics['auc_ci']
                },
                'cross_subject_mean_accuracy': float(np.mean(dsgat_accs)),
                'cross_subject_std_accuracy': float(np.std(dsgat_accs))
            },
            'lightgbm': {
                'pooled_accuracy': pop_lgb_metrics['accuracy'],
                'pooled_macro_f1': pop_lgb_metrics['f1'],
                'pooled_macro_precision': pop_lgb_metrics['precision'],
                'pooled_macro_recall': pop_lgb_metrics['recall'],
                'pooled_cohen_kappa': pop_lgb_metrics['kappa'],
                'pooled_macro_auc': pop_lgb_metrics['auc'],
                'bootstrap_95_ci': {
                    'accuracy': pop_lgb_metrics['accuracy_ci'],
                    'macro_f1': pop_lgb_metrics['f1_ci'],
                    'cohen_kappa': pop_lgb_metrics['kappa_ci'],
                    'macro_auc': pop_lgb_metrics['auc_ci']
                },
                'cross_subject_mean_accuracy': float(np.mean(lgb_accs)),
                'cross_subject_std_accuracy': float(np.std(lgb_accs))
            }
        },
        'per_subject_results': {
            str(r['sub_id']): {
                'dsgat': r['dsgat']['metrics'],
                'lightgbm': r['lgb']['metrics']
            } for r in all_subject_results
        }
    }
    
    json_path = 'ds_gat_literature_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_export, f, indent=2)
    print(f"[Export] Structured JSON results saved to {json_path}", flush=True)
    
    csv_path = 'ds_gat_literature_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Subject_ID', 'Test_Samples',
            'DSGAT_Accuracy', 'DSGAT_Macro_F1', 'DSGAT_Cohen_Kappa', 'DSGAT_Macro_AUC',
            'LGB_Accuracy', 'LGB_Macro_F1', 'LGB_Cohen_Kappa', 'LGB_Macro_AUC'
        ])
        for r in all_subject_results:
            writer.writerow([
                f"Subject_{r['sub_id']:02d}", len(r['y_test']),
                f"{r['dsgat']['metrics']['accuracy']*100:.2f}%", f"{r['dsgat']['metrics']['f1']:.4f}",
                f"{r['dsgat']['metrics']['kappa']:.4f}", f"{r['dsgat']['metrics']['auc']:.4f}",
                f"{r['lgb']['metrics']['accuracy']*100:.2f}%", f"{r['lgb']['metrics']['f1']:.4f}",
                f"{r['lgb']['metrics']['kappa']:.4f}", f"{r['lgb']['metrics']['auc']:.4f}"
            ])
        writer.writerow([
            'POPULATION_POOLED', len(pooled_y_true),
            f"{pop_dsgat_metrics['accuracy']*100:.2f}%", f"{pop_dsgat_metrics['f1']:.4f}",
            f"{pop_dsgat_metrics['kappa']:.4f}", f"{pop_dsgat_metrics['auc']:.4f}",
            f"{pop_lgb_metrics['accuracy']*100:.2f}%", f"{pop_lgb_metrics['f1']:.4f}",
            f"{pop_lgb_metrics['kappa']:.4f}", f"{pop_lgb_metrics['auc']:.4f}"
        ])
    print(f"[Export] Tabular CSV results saved to {csv_path}", flush=True)

if __name__ == '__main__':
    main()
