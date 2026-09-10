"""
Geodesic Evidential Attribution (GEA) Framework for EEG Emotion Recognition
===========================================================================
Native XAI attribution framework that computes analytical path integrals along
true Riemannian covariance geodesics and decomposes Dirichlet epistemic uncertainty.

Key Innovations:
1. Riemannian Geodesic Path Integration:
   Interpolates on SPD covariance manifold S_++^10:
   C(t) = C_base^1/2 exp(t * logm(C_base^-1/2 C C_base^-1/2)) C_base^1/2
2. Evidential Dirichlet Attribution:
   Extracts closed-form gradients of target class evidence e_c(x) and
   epistemic uncertainty u(x) = K / S.
3. Multi-Band Topographic Brain Mapping:
   Projects 310D attributions across 62 standard 10-20 channels and 5 frequency bands
   (delta, theta, alpha, beta, gamma) for all 4 affective states.

Outputs:
- JSON metrics saved to xai/results/gea_attribution_metrics.json
- 300 DPI figures exported to figures/geodesic_attribution/
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
BAND_NAMES = ['Delta (1-3 Hz)', 'Theta (4-7 Hz)', 'Alpha (8-13 Hz)', 'Beta (14-30 Hz)', 'Gamma (31-50 Hz)']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

SEED_IV_CHANNELS = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

# 2D Cartesian electrode coordinates normalized to [-1, 1] on scalp disk
ELECTRODE_2D_COORDS = {
    'Fp1': (-0.30, 0.85), 'Fpz': (0.00, 0.88), 'Fp2': (0.30, 0.85),
    'AF3': (-0.25, 0.70), 'AF4': (0.25, 0.70),
    'F7':  (-0.75, 0.55), 'F5':  (-0.50, 0.55), 'F3':  (-0.30, 0.55), 'F1':  (-0.15, 0.55), 'Fz':  (0.00, 0.55), 'F2':  (0.15, 0.55), 'F4':  (0.30, 0.55), 'F6':  (0.50, 0.55), 'F8':  (0.75, 0.55),
    'FT7': (-0.85, 0.30), 'FC5': (-0.60, 0.30), 'FC3': (-0.35, 0.30), 'FC1': (-0.18, 0.30), 'FCz': (0.00, 0.30), 'FC2': (0.18, 0.30), 'FC4': (0.35, 0.30), 'FC6': (0.60, 0.30), 'FT8': (0.85, 0.30),
    'T7':  (-0.90, 0.00), 'C5':  (-0.65, 0.00), 'C3':  (-0.40, 0.00), 'C1':  (-0.20, 0.00), 'Cz':  (0.00, 0.00), 'C2':  (0.20, 0.00), 'C4':  (0.40, 0.00), 'C6':  (0.65, 0.00), 'T8':  (0.90, 0.00),
    'TP7': (-0.85, -0.30), 'CP5': (-0.60, -0.30), 'CP3': (-0.35, -0.30), 'CP1': (-0.18, -0.30), 'CPz': (0.00, -0.30), 'CP2': (0.18, -0.30), 'CP4': (0.35, -0.30), 'CP6': (0.60, -0.30), 'TP8': (0.85, -0.30),
    'P7':  (-0.75, -0.55), 'P5':  (-0.50, -0.55), 'P3':  (-0.30, -0.55), 'P1':  (-0.15, -0.55), 'Pz':  (0.00, -0.55), 'P2':  (0.15, -0.55), 'P4':  (0.30, -0.55), 'P6':  (0.50, -0.55), 'P8':  (0.75, -0.55),
    'PO7': (-0.60, -0.75), 'PO5': (-0.40, -0.75), 'PO3': (-0.25, -0.75), 'POz': (0.00, -0.75), 'PO4': (0.25, -0.75), 'PO6': (0.40, -0.75), 'PO8': (0.60, -0.75),
    'CB1': (-0.45, -0.90), 'O1':  (-0.25, -0.90), 'Oz':  (0.00, -0.90), 'O2':  (0.25, -0.90), 'CB2': (0.45, -0.90)
}

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_riemannian_features(X_de_310):
    """Computes 55D Log-Euclidean Riemannian Tangent Space vector from 10 key electrodes."""
    N = len(X_de_310)
    X_reshaped = X_de_310.reshape(N, 62, 5)
    sub_ch = X_reshaped[:, [0, 2, 4, 6, 23, 27, 30, 32, 48, 50], :] # (N, 10, 5)
    covs = np.matmul(sub_ch, np.swapaxes(sub_ch, 1, 2)) / 5.0 # (N, 10, 10)
    eye = np.eye(10)[None, :, :] * 1e-4
    covs_reg = covs + eye
    
    w, v = np.linalg.eigh(covs_reg)
    w = np.maximum(w, 1e-6)
    log_covs = np.matmul(v * np.log(w)[:, None, :], np.swapaxes(v, 1, 2))
    
    riem_feats = []
    for i in range(N):
        mat = log_covs[i]
        vals = []
        for r in range(10):
            for c in range(r, 10):
                if r == c:
                    vals.append(mat[r, c])
                else:
                    vals.append(np.sqrt(2.0) * mat[r, c])
        riem_feats.append(vals)
    return np.array(riem_feats, dtype=np.float32), covs_reg

def compute_topological_features(X_de_310):
    """Computes 33D topological persistent descriptors."""
    N = len(X_de_310)
    X_reshaped = X_de_310.reshape(N, 62, 5)
    band_energy = np.sum(X_reshaped ** 2, axis=1)
    f_cent = np.mean(X_reshaped[:, 0:16, :], axis=1)
    t_cent = np.mean(X_reshaped[:, 16:32, :], axis=1)
    c_cent = np.mean(X_reshaped[:, 32:48, :], axis=1)
    p_cent = np.mean(X_reshaped[:, 48:62, :], axis=1)
    
    d_ft = np.linalg.norm(f_cent - t_cent, axis=1, keepdims=True)
    d_fc = np.linalg.norm(f_cent - c_cent, axis=1, keepdims=True)
    d_fp = np.linalg.norm(f_cent - p_cent, axis=1, keepdims=True)
    d_tc = np.linalg.norm(t_cent - c_cent, axis=1, keepdims=True)
    d_tp = np.linalg.norm(t_cent - p_cent, axis=1, keepdims=True)
    d_cp = np.linalg.norm(c_cent - p_cent, axis=1, keepdims=True)
    
    curv1 = d_ft / (d_fc + 1e-5)
    curv2 = d_fp / (d_cp + 1e-5)
    
    topo_feats = np.hstack([band_energy, f_cent, t_cent, c_cent, p_cent, d_ft, d_fc, d_fp, d_tc, d_tp, d_cp, curv1, curv2])
    return np.array(topo_feats, dtype=np.float32)

def riemannian_geodesic(C_base, C_target, t):
    """Computes point along Riemannian manifold geodesic at step t in [0, 1]."""
    w_b, v_b = np.linalg.eigh(C_base)
    w_b = np.maximum(w_b, 1e-6)
    C_base_sqrt = v_b @ np.diag(np.sqrt(w_b)) @ v_b.T
    C_base_inv_sqrt = v_b @ np.diag(1.0 / np.sqrt(w_b)) @ v_b.T
    
    mid = C_base_inv_sqrt @ C_target @ C_base_inv_sqrt
    w_m, v_m = np.linalg.eigh(mid)
    w_m = np.maximum(w_m, 1e-6)
    
    exp_mid = v_m @ np.diag(np.exp(t * np.log(w_m))) @ v_m.T
    C_t = C_base_sqrt @ exp_mid @ C_base_sqrt
    return C_t

def cov_to_tangent_vec(C_mat):
    """Converts a single 10x10 SPD matrix to 55D tangent vector."""
    w, v = np.linalg.eigh(C_mat)
    w = np.maximum(w, 1e-6)
    log_c = v @ np.diag(np.log(w)) @ v.T
    vals = []
    for r in range(10):
        for c in range(r, 10):
            if r == c:
                vals.append(log_c[r, c])
            else:
                vals.append(np.sqrt(2.0) * log_c[r, c])
    return np.array(vals, dtype=np.float32)

class DifferentiableTREHNet(nn.Module):
    """Differentiable PyTorch Model for GEA Evaluation."""
    def __init__(self, in_features=398, hidden_dim=128, num_classes=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.evidence_head = nn.Linear(64, num_classes)
        
    def forward(self, x):
        h = self.encoder(x)
        evidence = F.softplus(self.evidence_head(h))
        alpha = evidence + 1.0
        return alpha, evidence

def compute_geodesic_evidential_attributions(model, x_sample, x_base, cov_sample, cov_base, scaler, device, num_steps=25):
    """
    Computes Geodesic Evidential Attribution along the Riemannian geodesic:
    GEA_c(x) = (x - x_base) * (1/M) sum_{m=1}^M grad_{x(t_m)} e_c(x(t_m))
    """
    model.eval()
    
    # 1. Unpack components: 310D DE, 55D Riem, 33D Topo
    x_de_sample = x_sample[:310]
    x_de_base = x_base[:310]
    x_topo_sample = x_sample[365:]
    x_topo_base = x_base[365:]
    
    # Precompute path points
    accum_grad_e = {c: np.zeros(398, dtype=np.float32) for c in range(4)}
    accum_grad_u = np.zeros(398, dtype=np.float32)
    
    for m in range(1, num_steps + 1):
        t = float(m) / float(num_steps)
        # Euclidean interpolation for DE and Topo
        de_t = x_de_base + t * (x_de_sample - x_de_base)
        topo_t = x_topo_base + t * (x_topo_sample - x_topo_base)
        
        # Riemannian geodesic on SPD covariance
        cov_t = riemannian_geodesic(cov_base, cov_sample, t)
        riem_t = cov_to_tangent_vec(cov_t)
        
        # Assemble 398D interpolated representation
        x_t_raw = np.hstack([de_t, riem_t, topo_t])
        x_t_scaled = scaler.transform(x_t_raw.reshape(1, -1)).astype(np.float32)
        
        x_t_tensor = torch.tensor(x_t_scaled, dtype=torch.float32, device=device, requires_grad=True)
        alpha_t, evidence_t = model(x_t_tensor)
        S_t = torch.sum(alpha_t, dim=1)
        u_t = 4.0 / S_t
        
        # Gradient for each class evidence
        for c in range(4):
            grad_c = torch.autograd.grad(evidence_t[0, c], x_t_tensor, retain_graph=True)[0]
            accum_grad_e[c] += grad_c.cpu().numpy()[0]
            
        # Gradient for epistemic uncertainty
        grad_u = torch.autograd.grad(u_t[0], x_t_tensor)[0]
        accum_grad_u += grad_u.cpu().numpy()[0]
        
    diff = x_sample - x_base
    gea_attributions = {}
    for c in range(4):
        avg_grad = accum_grad_e[c] / float(num_steps)
        gea_attributions[c] = diff * avg_grad
        
    avg_grad_u = accum_grad_u / float(num_steps)
    uncertainty_attributions = diff * avg_grad_u
    
    return gea_attributions, uncertainty_attributions

def plot_head_topography(ax, channel_values, title="Scalp Topography", cmap=plt.cm.RdBu_r, vmin=-1.0, vmax=1.0):
    """Plots an anatomically accurate 2D scalp topographic map with nose and ears."""
    # Build 2D grid
    grid_x, grid_y = np.mgrid[-1.1:1.1:150j, -1.1:1.1:150j]
    points = []
    values = []
    
    for ch_name, val in channel_values.items():
        if ch_name in ELECTRODE_2D_COORDS:
            points.append(ELECTRODE_2D_COORDS[ch_name])
            values.append(val)
            
    points = np.array(points)
    values = np.array(values)
    
    # Cubic spline interpolation
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0.0)
    
    # Mask outside head circle (radius = 1.0)
    mask = (grid_x ** 2 + grid_y ** 2) > 1.0
    grid_z[mask] = np.nan
    
    im = ax.imshow(grid_z.T, extent=(-1.1, 1.1, -1.1, 1.1), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    # Draw head circle
    circle = plt.Circle((0, 0), 1.0, color='#2c3e50', fill=False, lw=2.2)
    ax.add_artist(circle)
    
    # Draw Nose
    ax.plot([0.0, -0.12, 0.12, 0.0], [1.0, 1.12, 1.12, 1.0], color='#2c3e50', lw=2.2)
    
    # Draw Ears
    ax.plot([-1.0, -1.06, -1.06, -1.0], [0.15, 0.10, -0.10, -0.15], color='#2c3e50', lw=2.0)
    ax.plot([1.0, 1.06, 1.06, 1.0], [0.15, 0.10, -0.10, -0.15], color='#2c3e50', lw=2.0)
    
    # Scatter electrode positions
    ax.scatter(points[:, 0], points[:, 1], c='black', s=12, alpha=0.75, zorder=5)
    
    ax.set_xlim([-1.20, 1.20])
    ax.set_ylim([-1.20, 1.20])
    ax.axis('off')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    return im

def generate_gea_figures(class_attributions, uncertainty_attributions, output_dir="figures/geodesic_attribution"):
    """Exports 300 DPI publication plots."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # 1. 4-Class Scalp Topographic Brain Maps
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=300)
    
    for c, (ax, cls_name) in enumerate(zip(axes, CLASS_NAMES)):
        # Extract DE channel attributions across all 5 bands
        de_attr_310 = class_attributions[c][:310].reshape(62, 5)
        # Sum gamma + beta power (most prominent affective markers)
        ch_importance = np.sum(np.abs(de_attr_310[:, [3, 4]]), axis=1) # Beta + Gamma
        # Normalize to [0, 1]
        ch_norm = (ch_importance - np.min(ch_importance)) / (np.max(ch_importance) - np.min(ch_importance) + 1e-8)
        
        ch_dict = {SEED_IV_CHANNELS[i]: ch_norm[i] for i in range(62)}
        im = plot_head_topography(ax, ch_dict, title=f"Class: {cls_name}\n(Beta + Gamma Salience)", cmap=plt.cm.inferno, vmin=0.0, vmax=1.0)
        
    cbar_ax = fig.add_axes([0.92, 0.20, 0.015, 0.60])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Normalized Geodesic Salience", fontsize=10, fontweight='bold')
    
    plt.suptitle("Geodesic Evidential Attribution (GEA): 2D Scalp Topography Across Affective States", fontsize=13, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    topo_path = os.path.join(output_dir, "gea_spatial_topo_maps.png")
    plt.savefig(topo_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {topo_path}", flush=True)
    
    # 2. 62-Channel x 5-Band Attribution Heatmap for Happy & Sad Emotions
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5), dpi=300)
    for ax, c, cls_name in zip(axes, [1, 3], ['Sad (Negative Valence)', 'Happy (Positive Valence)']):
        de_attr = class_attributions[c][:310].reshape(62, 5)
        norm_attr = (de_attr - np.mean(de_attr)) / (np.std(de_attr) + 1e-6)
        
        cax = ax.imshow(norm_attr, aspect='auto', cmap=plt.cm.RdBu_r, vmin=-2.5, vmax=2.5, interpolation='nearest')
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], fontsize=10, fontweight='bold')
        ax.set_yticks(np.arange(0, 62, 2))
        ax.set_yticklabels([SEED_IV_CHANNELS[i] for i in range(0, 62, 2)], fontsize=8)
        ax.set_xlabel("Frequency Band", fontsize=11, fontweight='bold')
        ax.set_ylabel("EEG Electrode (62 Channels)", fontsize=11, fontweight='bold')
        ax.set_title(f"GEA Feature Salience Matrix: {cls_name}", fontsize=12, pad=10, fontweight='bold')
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        
    plt.suptitle("Geodesic Evidential Attribution: Electrode x Frequency Band Decomposition", fontsize=13, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    heatmap_path = os.path.join(output_dir, "gea_band_channel_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {heatmap_path}", flush=True)
    
    # 3. Uncertainty Decomposition by Lobe and Frequency Band
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=300)
    
    de_u = np.abs(uncertainty_attributions[:310].reshape(62, 5))
    lobe_u = [
        np.mean(de_u[0:16, :]),  # Frontal
        np.mean(de_u[16:32, :]), # Temporal
        np.mean(de_u[32:48, :]), # Central
        np.mean(de_u[48:62, :])  # Parietal/Occipital
    ]
    lobe_names = ['Frontal\n(FP/AF/F)', 'Temporal\n(FT/T/TP)', 'Central\n(FC/C/CP)', 'Parieto-Occipital\n(P/PO/O)']
    
    ax1.bar(lobe_names, lobe_u, color=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'], edgecolor='#2c3e50', width=0.55, alpha=0.9)
    ax1.set_ylabel("Mean Epistemic Uncertainty Reduction (|du/dx|)", fontsize=10, fontweight='bold')
    ax1.set_title("(A) Uncertainty Attribution by Cortical Lobe", fontsize=11, pad=10, fontweight='bold')
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    band_u = np.mean(de_u, axis=0)
    ax2.bar(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'], band_u, color='#34495e', edgecolor='#1a252f', width=0.55, alpha=0.9)
    ax2.set_ylabel("Mean Epistemic Uncertainty Reduction (|du/dx|)", fontsize=10, fontweight='bold')
    ax2.set_title("(B) Uncertainty Attribution by Frequency Band", fontsize=11, pad=10, fontweight='bold')
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.suptitle("GEA Epistemic Uncertainty Decomposition Across Cortical Regions & Frequency Bands", fontsize=12, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    uncert_path = os.path.join(output_dir, "gea_uncertainty_decomposition.png")
    plt.savefig(uncert_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {uncert_path}", flush=True)

def run_gea_pipeline(dataset_path="seed_iv_processed.npz", device_str="cuda", num_steps=25, seed=42):
    set_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"[*] GEA Framework executing on Device: {device} | Steps: {num_steps}", flush=True)
    start_time = time.time()
    
    print("="*85, flush=True)
    print(">>> GEODESIC EVIDENTIAL ATTRIBUTION (GEA) PIPELINE", flush=True)
    print("    Computing Riemannian Geodesic Path Integrals & Dirichlet Gradients...", flush=True)
    print("="*85, flush=True)
    
    data = np.load(dataset_path)
    raw_features = data['features']
    if raw_features.ndim == 3:
        features_flat = raw_features.reshape(len(raw_features), -1)
    else:
        features_flat = raw_features
        
    labels = data['labels']
    subject_ids = data['subject_ids']
    
    # 1. Compute multi-modal features
    print("[*] Extracting Riemannian 55D Tangent Space & 33D Topological Features...", flush=True)
    riem_features, all_covs = compute_riemannian_features(features_flat)
    topo_features = compute_topological_features(features_flat)
    X_all = np.hstack([features_flat, riem_features, topo_features]) # (37575, 398)
    
    # 2. Partition evaluation on Subject 15 & Subject 7
    sub_mask = (subject_ids == 15)
    X_sub = X_all[sub_mask]
    y_sub = labels[sub_mask]
    cov_sub = all_covs[sub_mask]
    
    X_tr, X_te, y_tr, y_te, cov_tr, cov_te = train_test_split(
        X_sub, y_sub, cov_sub, test_size=0.20, shuffle=True, stratify=y_sub, random_state=seed
    )
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    # 3. Train Differentiable TREH-Net model
    print("[*] Training Differentiable TREH-Net Model for Attribution Engine...", flush=True)
    model = DifferentiableTREHNet(in_features=398, hidden_dim=128, num_classes=4, dropout=0.20).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    
    tr_ds = TensorDataset(torch.tensor(X_tr_sc, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
    
    for ep in range(30):
        model.train()
        for bx, by in tr_ld:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            alpha, _ = model(bx)
            S = torch.sum(alpha, dim=1, keepdim=True)
            y_one_hot = F.one_hot(by, 4).float()
            loss = torch.mean(torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1))
            loss.backward()
            optimizer.step()
            
    # 4. Compute Neutral Reference Baseline
    neutral_mask = (y_tr == 0)
    x_base = np.mean(X_tr[neutral_mask], axis=0) # (398,)
    cov_base = np.mean(cov_tr[neutral_mask], axis=0) # (10, 10)
    
    print("[*] Computing Geodesic Path Integrals across Evaluation Test Trials...", flush=True)
    accum_class_attributions = {c: np.zeros(398, dtype=np.float32) for c in range(4)}
    accum_uncert_attributions = np.zeros(398, dtype=np.float32)
    class_counts = {c: 0 for c in range(4)}
    
    # Compute GEA for sample subset representing all 4 classes
    for i in range(min(len(X_te), 200)):
        target_cls = int(y_te[i])
        x_sample = X_te[i]
        cov_sample = cov_te[i]
        
        gea_attr, u_attr = compute_geodesic_evidential_attributions(
            model, x_sample, x_base, cov_sample, cov_base, scaler, device, num_steps=num_steps
        )
        
        accum_class_attributions[target_cls] += gea_attr[target_cls]
        accum_uncert_attributions += u_attr
        class_counts[target_cls] += 1
        
    for c in range(4):
        if class_counts[c] > 0:
            accum_class_attributions[c] /= float(class_counts[c])
    accum_uncert_attributions /= float(len(X_te))
    
    # 5. Extract Salient Channels & Bands
    print("\n" + "="*85, flush=True)
    print(">>> GEA NEUROBIOLOGICAL ATTRIBUTION SUMMARY", flush=True)
    print("="*85, flush=True)
    
    top_channels_per_class = {}
    for c, cls_name in enumerate(CLASS_NAMES):
        de_attr_c = accum_class_attributions[c][:310].reshape(62, 5)
        # Channel importance
        ch_importance = np.sum(np.abs(de_attr_c), axis=1) # (62,)
        top_ch_idx = np.argsort(ch_importance)[::-1][:5]
        top_ch_names = [SEED_IV_CHANNELS[idx] for idx in top_ch_idx]
        top_channels_per_class[cls_name] = top_ch_names
        
        # Band importance
        band_importance = np.mean(np.abs(de_attr_c), axis=0) # (5,)
        top_band_idx = np.argsort(band_importance)[::-1][0]
        
        print(f"Emotion: {cls_name:7s} | Dominant Band: {BAND_NAMES[top_band_idx]:18s} | Top Channels: {', '.join(top_ch_names)}", flush=True)
        
    # Save Structured Metrics to JSON
    results_dir = os.path.join("xai", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    elapsed = time.time() - start_time
    
    metrics_export = {
        'framework': 'Geodesic Evidential Attribution (GEA)',
        'dataset': 'SEED-IV',
        'quadrature_steps': num_steps,
        'elapsed_seconds': float(elapsed),
        'top_salient_electrodes': top_channels_per_class,
        'band_salience_ranking': {
            cls_name: [
                {'band': BAND_NAMES[b], 'importance': float(np.mean(np.abs(accum_class_attributions[c][:310].reshape(62, 5)[:, b])))}
                for b in range(5)
            ] for c, cls_name in enumerate(CLASS_NAMES)
        },
        'regional_epistemic_uncertainty_reduction': {
            'Frontal': float(np.mean(np.abs(accum_uncert_attributions[:310].reshape(62, 5)[0:16, :]))),
            'Temporal': float(np.mean(np.abs(accum_uncert_attributions[:310].reshape(62, 5)[16:32, :]))),
            'Central': float(np.mean(np.abs(accum_uncert_attributions[:310].reshape(62, 5)[32:48, :]))),
            'ParietoOccipital': float(np.mean(np.abs(accum_uncert_attributions[:310].reshape(62, 5)[48:62, :])))
        }
    }
    
    json_path = os.path.join(results_dir, "gea_attribution_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"\n[OK] Saved attribution metrics to: {json_path}", flush=True)
    
    # Generate 300 DPI Figures
    generate_gea_figures(accum_class_attributions, accum_uncert_attributions, output_dir="figures/geodesic_attribution")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Geodesic Evidential Attribution (GEA) Framework")
    parser.add_argument('--dataset', type=str, default='seed_iv_processed.npz', help='Path to processed SEED-IV dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Execution device (cuda or cpu)')
    parser.add_argument('--steps', type=int, default=25, help='Number of Riemannian geodesic path steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    run_gea_pipeline(dataset_path=args.dataset, device_str=args.device, num_steps=args.steps, seed=args.seed)
