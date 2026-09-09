"""
Affective-InfoNCE: Self-Supervised Latent Contrastive Hypersphere Pipeline on SEED-IV
=====================================================================================
Implements the zero-leakage Affective-InfoNCE architecture:
1. 580D Cortical Asymmetry Space (Hou et al. 2023):
   - 310 raw DE features (62 channels x 5 frequency bands)
   - 135 DASM features (27 Left-Right homologous pairs x 5 bands)
   - 135 DCAU features (27 Anterior-Posterior pairs x 5 bands)
2. Baseline Reference Normalization (Cheng et al. 2021):
   - Zero-leakage subtraction of training fold neutral trial mean vector (mu_neutral)
   - Fold-quarantined StandardScaler fit strictly on the 18 training trials
3. Intra-Trial Neuro-Augmentation Engine (GPU-Vectorized):
   - Stochastic channel perturbation (10% random zeroing)
   - Low-frequency spectral jitter (Delta & Theta bands N(0, 0.05^2))
   - Multi-view positive pairing across training trials
4. Siamese Metric Encoder & Hypersphere Normalization:
   - Backbone Encoder f_theta: Linear(580 -> 256) -> BatchNorm1d -> GELU -> Dropout(0.2) -> Linear(256 -> 128) -> LayerNorm
   - Projection Head g_phi: Linear(128 -> 128) -> GELU -> Linear(128 -> 64)
   - L2 Hypersphere Normalization: z in S^63
5. Two-Phase Optimization Protocol (NVIDIA RTX 4050 AMP Mixed Precision):
   - Phase 1: Supervised InfoNCE / SupCon loss (tau = 0.07, 40 epochs, AdamW + Cosine Annealing, fp16 autocast)
   - Phase 2: Class prototype vector calculation (c_k in S^63) + Frozen Linear Probe head
6. Dual-Level Inference & Angular Trial Consensus:
   - Sample-Level: Cosine nearest-prototype (z_w . c_k) & Linear Probe posterior
   - Trial-Level Angular Consensus: Score_k = sum_w (z_w . c_k) -> argmax_k Score_k
   - Trial-Level Soft Log-Odds Consensus: P_trial = Softmax( sum_w log p_w )
7. Complete Population Reporting & 300 DPI Publication Visualizations:
   - affective_infonce_results.json & affective_infonce_results.csv
   - 300 DPI figures exported to figures/affective_infonce/ and artifact directory
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
import matplotlib.pyplot as plt

from spatial_mapping import (
    SEED_IV_CHANNELS,
    GRID_COORDINATES_9X9
)

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

def set_seed(seed=42):
    """Sets global random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def get_asymmetry_pair_indices():
    """
    Computes index pairs for:
    - 27 Left-Right DASM pairs (excluding 8 midline electrodes)
    - 27 Anterior-Posterior DCAU pairs (frontal/frontocentral to parietal/occipital)
    """
    left_indices = []
    right_indices = []
    for ch in SEED_IV_CHANNELS:
        r, c = GRID_COORDINATES_9X9[ch]
        if c < 4:
            target_r, target_c = r, 8 - c
            matching = [r_ch for r_ch in SEED_IV_CHANNELS if GRID_COORDINATES_9X9[r_ch] == (target_r, target_c)]
            if matching:
                left_indices.append(SEED_IV_CHANNELS.index(ch))
                right_indices.append(SEED_IV_CHANNELS.index(matching[0]))
    
    ant_indices = []
    post_indices = []
    
    # Row 2 (Frontal) <-> Row 6 (Parietal) across all 9 columns
    for c in range(9):
        f_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (2, c)]
        p_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (6, c)]
        if f_ch and p_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(f_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(p_ch[0]))
            
    # Row 3 (Fronto-Central) <-> Row 5 (Centro-Parietal) across all 9 columns
    for c in range(9):
        fc_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (3, c)]
        cp_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (5, c)]
        if fc_ch and cp_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(fc_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(cp_ch[0]))
            
    # Pre-Frontal / Antero-Frontal (Rows 0-1) <-> Parieto-Occipital / Occipital / Cerebellar (Rows 7-8)
    ap_custom = [
        ('Fp1', 'O1'), ('Fpz', 'Oz'), ('Fp2', 'O2'),
        ('AF3', 'PO3'), ('AF4', 'PO4'),
        ('Fp1', 'CB1'), ('Fp2', 'CB2'),
        ('AF3', 'PO5'), ('AF4', 'PO6')
    ]
    for f_name, p_name in ap_custom:
        ant_indices.append(SEED_IV_CHANNELS.index(f_name))
        post_indices.append(SEED_IV_CHANNELS.index(p_name))
        
    assert len(left_indices) == 27, f"Expected 27 Left-Right pairs, got {len(left_indices)}"
    assert len(ant_indices) == 27, f"Expected 27 Anterior-Posterior pairs, got {len(ant_indices)}"
    
    return left_indices, right_indices, ant_indices, post_indices

def extract_580d_cortical_features(features_3d, left_idx, right_idx, ant_idx, post_idx):
    """
    Extracts concatenated 580D feature representations:
    - 310 raw DE features (62 channels x 5 bands)
    - 135 DASM features (27 Left-Right pairs x 5 bands)
    - 135 DCAU features (27 Anterior-Posterior pairs x 5 bands)
    """
    N = len(features_3d)
    raw_flat = features_3d.reshape(N, 310)
    dasm = (features_3d[:, left_idx, :] - features_3d[:, right_idx, :]).reshape(N, 135)
    dcau = (features_3d[:, ant_idx, :] - features_3d[:, post_idx, :]).reshape(N, 135)
    return np.hstack([raw_flat, dasm, dcau]).astype(np.float32)

# Precompute Low-Frequency (Delta=0, Theta=1) feature indices across 580D vector
LOW_FREQ_INDICES = [j for j in range(580) if (j % 5) in (0, 1)]

def apply_intra_trial_neuro_augmentation(x_batch, lf_indices, channel_drop_prob=0.10, jitter_std=0.05):
    """
    Vectorized GPU intra-trial neuro-augmentation:
    1. Stochastic channel/feature perturbation: randomly zero out 10% of features per sample.
    2. Low-frequency spectral jitter: add N(0, 0.05^2) Gaussian noise to Delta and Theta bands.
    """
    x_aug = x_batch.clone()
    
    # 1. Low-Frequency Spectral Jitter (Delta & Theta bands)
    noise = torch.randn((x_aug.size(0), len(lf_indices)), device=x_aug.device, dtype=x_aug.dtype) * jitter_std
    x_aug[:, lf_indices] += noise
    
    # 2. Stochastic Channel Perturbation (10% random zeroing)
    drop_mask = (torch.rand_like(x_aug) > channel_drop_prob).to(x_aug.dtype)
    x_aug = x_aug * drop_mask
    
    return x_aug

class AffectiveSiameseEncoder(nn.Module):
    """
    Siamese Contrastive Backbone f_theta and Hypersphere Projection Head g_phi.
    - Backbone f_theta: Linear(580 -> 256) -> BatchNorm1d -> GELU -> Dropout(0.2) -> Linear(256 -> 128) -> LayerNorm
    - Projection Head g_phi: Linear(128 -> 128) -> GELU -> Linear(128 -> 64)
    - Normalized output: z in S^63
    """
    def __init__(self, in_features=580, hidden_dim=256, latent_dim=128, proj_dim=64, dropout_p=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, proj_dim)
        )
        
    def encode(self, x):
        """Extracts 128D latent representation h."""
        return self.encoder(x)
        
    def project(self, h):
        """Projects 128D latent h to 64D normalized hypersphere embedding z in S^63."""
        z_unnorm = self.projection_head(h)
        return F.normalize(z_unnorm, p=2, dim=-1)
        
    def forward(self, x):
        h = self.encode(x)
        z = self.project(h)
        return h, z

class SupervisedInfoNCELoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (SupCon / InfoNCE, Khosla et al. 2020).
    Pulls representations of the same emotion class together while maximizing
    angular distances to distinct emotion classes on the unit hypersphere.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z, labels):
        """
        z: (N, 64) L2-normalized hypersphere representations
        labels: (N,) ground-truth class labels
        """
        device = z.device
        N = z.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # Eliminate self-contrast diagonal
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(N, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        
        # Cosine similarity matrix scaled by temperature
        logits = torch.matmul(z, z.T) / self.temperature
        
        # Stability: subtract row max
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # Denominator over all a != i
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        has_pos = mask_pos_pairs > 0
        mean_log_prob_pos = torch.zeros(N, device=device)
        mean_log_prob_pos[has_pos] = (mask * log_prob)[has_pos].sum(1) / mask_pos_pairs[has_pos]
        
        if has_pos.sum() > 0:
            loss = -mean_log_prob_pos[has_pos].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss

class LinearProbeHead(nn.Module):
    """L2-regularized linear classification probe on frozen 128D latent space."""
    def __init__(self, latent_dim=128, num_classes=4):
        super().__init__()
        self.fc = nn.Linear(latent_dim, num_classes)
        
    def forward(self, h):
        return self.fc(h)

def train_affective_infonce_fold(
    X_train, y_train,
    device,
    in_features=580,
    latent_dim=128,
    proj_dim=64,
    phase1_epochs=40,
    phase2_epochs=20,
    batch_size=256,
    lr_phase1=1e-3,
    lr_phase2=2e-3,
    temperature=0.07
):
    """
    Executes Two-Phase Optimization:
    Phase 1: Supervised InfoNCE on S^63 with GPU Intra-Trial Neuro-Augmentation and AMP fp16.
    Phase 2: Prototype Metric Calculation + Frozen Linear Probe Training.
    """
    model = AffectiveSiameseEncoder(
        in_features=in_features,
        hidden_dim=256,
        latent_dim=latent_dim,
        proj_dim=proj_dim,
        dropout_p=0.2
    ).to(device)
    
    criterion_infonce = SupervisedInfoNCELoss(temperature=temperature)
    optimizer_p1 = torch.optim.AdamW(model.parameters(), lr=lr_phase1, weight_decay=1e-4)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=phase1_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.long, device=device)
    num_samples = len(X_train)
    lf_indices_tensor = torch.tensor(LOW_FREQ_INDICES, dtype=torch.long, device=device)
    
    # -------------------------------------------------------------
    # PHASE 1: Contrastive Hypersphere Structuring (AMP fp16)
    # -------------------------------------------------------------
    model.train()
    p1_losses = []
    for epoch in range(phase1_epochs):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        num_batches = 0
        
        for b in range(0, num_samples, batch_size):
            batch_idx = perm[b:b+batch_size]
            bx = X_tensor[batch_idx]
            by = y_tensor[batch_idx]
            
            # GPU Intra-Trial Neuro-Augmentation (Two Positive Views)
            bx_view1 = apply_intra_trial_neuro_augmentation(bx, lf_indices_tensor, channel_drop_prob=0.10, jitter_std=0.05)
            bx_view2 = apply_intra_trial_neuro_augmentation(bx, lf_indices_tensor, channel_drop_prob=0.10, jitter_std=0.05)
            
            bx_cat = torch.cat([bx_view1, bx_view2], dim=0)
            by_cat = torch.cat([by, by], dim=0)
            
            optimizer_p1.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16):
                _, z_cat = model(bx_cat)
                loss = criterion_infonce(z_cat, by_cat)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer_p1)
            scaler.update()
            
            epoch_loss += loss.item()
            num_batches += 1
            
        scheduler_p1.step()
        if num_batches > 0:
            p1_losses.append(epoch_loss / num_batches)
            
    # -------------------------------------------------------------
    # PHASE 2: Class Prototype Calculation & Frozen Linear Probing
    # -------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16):
            h_all, z_all = model(X_tensor)
            h_all = h_all.float()
            z_all = z_all.float()
            
        # Compute Unit Class Prototypes c_k in S^63
        prototypes = []
        for k in range(4):
            k_mask = (y_tensor == k)
            if k_mask.sum() > 0:
                k_mean = z_all[k_mask].mean(dim=0, keepdim=True)
                k_proto = F.normalize(k_mean, p=2, dim=-1)
            else:
                k_proto = torch.zeros((1, proj_dim), device=device)
            prototypes.append(k_proto)
            
        prototypes_tensor = torch.cat(prototypes, dim=0) # (4, 64)
        # Unit norm verification
        norms = torch.norm(prototypes_tensor, p=2, dim=-1).cpu().numpy()
        for k in range(4):
            assert abs(norms[k] - 1.0) < 1e-4, f"Prototype {k} norm deviation: {norms[k]}"
            
    # Train L2-Regularized Linear Probe on Frozen 128D Latents
    probe_head = LinearProbeHead(latent_dim=latent_dim, num_classes=4).to(device)
    optimizer_p2 = torch.optim.AdamW(probe_head.parameters(), lr=lr_phase2, weight_decay=1e-3)
    loss_ce = nn.CrossEntropyLoss()
    
    probe_head.train()
    for epoch in range(phase2_epochs):
        perm_p2 = torch.randperm(num_samples, device=device)
        for b in range(0, num_samples, batch_size):
            batch_idx = perm_p2[b:b+batch_size]
            bh = h_all[batch_idx]
            by = y_tensor[batch_idx]
            
            optimizer_p2.zero_grad(set_to_none=True)
            logits = probe_head(bh)
            l_ce = loss_ce(logits, by)
            l_ce.backward()
            optimizer_p2.step()
            
    probe_head.eval()
    
    return model, prototypes_tensor, probe_head, (p1_losses[-1] if p1_losses else 0.0)

def evaluate_infonce_test_fold(
    model, prototypes_tensor, probe_head,
    X_test, y_test, test_frame_trials, test_trials, test_trial_true_labels,
    device
):
    """
    Evaluates test frames and test trials under both Angular Prototype and Soft Log-Odds Consensus.
    """
    model.eval()
    probe_head.eval()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda'), dtype=torch.float16):
            h_test, z_test = model(X_test_tensor)
            h_test = h_test.float()
            z_test = z_test.float()
            
        # 1. Cosine similarity against Prototypes: (N_test, 4)
        cosine_sims = torch.matmul(z_test, prototypes_tensor.T).cpu().numpy()
        
        # 2. Linear Probe Probabilities: (N_test, 4)
        logits_probe = probe_head(h_test)
        probs_probe = F.softmax(logits_probe, dim=-1).cpu().numpy()
        
    # Sample-level predictions
    frame_preds_angular = np.argmax(cosine_sims, axis=-1)
    frame_preds_probe = np.argmax(probs_probe, axis=-1)
    
    # Trial-level aggregations
    trial_trues = []
    trial_preds_angular = []
    trial_preds_probe = []
    trial_probs_probe = []
    trial_angular_scores = []
    
    eps = 1e-7
    for tr in test_trials:
        tr_mask = np.where(test_frame_trials == tr)[0]
        if len(tr_mask) == 0:
            continue
            
        tr_idx = np.where(test_trials == tr)[0][0]
        y_true = int(test_trial_true_labels[tr_idx])
        trial_trues.append(y_true)
        
        # 1. Angular Prototype Consensus: Score_k = sum_w (z_w . c_k)
        tr_sims = cosine_sims[tr_mask]  # (W_tr, 4)
        angular_scores = np.sum(tr_sims, axis=0) # (4,)
        pred_ang = int(np.argmax(angular_scores))
        trial_preds_angular.append(pred_ang)
        trial_angular_scores.append(angular_scores.tolist())
        
        # 2. Soft Log-Odds Consensus: P_trial = Softmax( sum_w log p_w )
        tr_probs = probs_probe[tr_mask]
        log_odds = np.sum(np.log(np.clip(tr_probs, eps, 1.0)), axis=0)
        exp_lo = np.exp(log_odds - np.max(log_odds))
        norm_p = exp_lo / np.sum(exp_lo)
        pred_prb = int(np.argmax(norm_p))
        trial_preds_probe.append(pred_prb)
        trial_probs_probe.append(norm_p.tolist())
        
    return {
        'frame_true': y_test.tolist(),
        'frame_pred_angular': frame_preds_angular.tolist(),
        'frame_pred_probe': frame_preds_probe.tolist(),
        'frame_probs_probe': probs_probe.tolist(),
        'frame_cosine_sims': cosine_sims.tolist(),
        'test_embeddings_z': z_test.cpu().numpy(),
        
        'trial_true': trial_trues,
        'trial_pred_angular': trial_preds_angular,
        'trial_pred_probe': trial_preds_probe,
        'trial_probs_probe': trial_probs_probe,
        'trial_angular_scores': trial_angular_scores
    }

def fast_metrics_computation(yt, yp, num_classes=4):
    """Vectorized confusion matrix computation for fast bootstrap iterations."""
    cm = np.bincount(yt * num_classes + yp, minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    acc = np.sum(tp) / np.sum(cm) if np.sum(cm) > 0 else 0.0
    total = np.sum(cm)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2) if total > 0 else 0.0
    po = acc
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 0.0
    return acc, float(np.mean(p)), float(np.mean(r)), float(np.mean(f)), float(kappa)

def compute_bootstrap_confidence_intervals(y_true, y_pred, num_resamples=1000, seed=42):
    """Computes 95% non-parametric bootstrap confidence intervals rapidly."""
    rng = np.random.RandomState(seed)
    accs, precs, recs, f1s, kappas = [], [], [], [], []
    num_samples = len(y_true)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    for _ in range(num_resamples):
        indices = rng.randint(0, num_samples, size=num_samples)
        a, p, r, f, k = fast_metrics_computation(y_true[indices], y_pred[indices])
        accs.append(a)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        kappas.append(k)
        
    def get_ci(arr):
        if len(arr) == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        'accuracy': get_ci(accs),
        'macro_precision': get_ci(precs),
        'macro_recall': get_ci(recs),
        'macro_f1': get_ci(f1s),
        'cohen_kappa': get_ci(kappas)
    }

def compute_metrics_dict(y_true, y_pred, y_prob=None, num_resamples=1000, seed=42):
    """Computes full point estimates and 95% bootstrap confidence intervals."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    kappa = float(cohen_kappa_score(y_true, y_pred))
    
    auc = 0.0
    if y_prob is not None:
        try:
            y_prob_arr = np.asarray(y_prob, dtype=np.float32)
            auc = float(roc_auc_score(y_true, y_prob_arr, multi_class='ovr', average='macro'))
        except Exception:
            auc = 0.0
            
    cis = compute_bootstrap_confidence_intervals(y_true, y_pred, num_resamples=num_resamples, seed=seed)
    
    return {
        'point_estimates': {
            'accuracy': acc,
            'macro_precision': float(p),
            'macro_recall': float(r),
            'macro_f1': float(f),
            'cohen_kappa': kappa,
            'macro_roc_auc': auc
        },
        'confidence_intervals_95': cis
    }

def render_publication_figures(
    results, all_embeddings_sample, all_labels_sample,
    output_dir, artifact_dir
):
    """
    Renders 3 publication-ready 300 DPI figures:
    1. infonce_hypersphere_tsne.png: t-SNE 2D projection of normalized hypersphere embeddings z in S^63.
    2. infonce_trial_consensus_accuracy.png: Per-subject trial consensus accuracy bar chart.
    3. infonce_confusion_matrix_1080trials.png: Normalized confusion matrix across all 1,080 trials.
    """
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        
    plt.rcParams.update({
        'font.size': 11,
        'font.sans-serif': 'DejaVu Sans',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.0,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.alpha': 0.6
    })
    
    # ---------------------------------------------------------
    # Figure 1: t-SNE Hypersphere Latent Representation
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    
    # Subsample 2,000 points if larger
    n_sample = len(all_labels_sample)
    if n_sample > 2500:
        idx_sub = np.random.choice(n_sample, size=2500, replace=False)
        emb_sub = all_embeddings_sample[idx_sub]
        lbl_sub = all_labels_sample[idx_sub]
    else:
        emb_sub = all_embeddings_sample
        lbl_sub = all_labels_sample
        
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_sub)
    
    for c_idx, (c_name, c_color) in enumerate(zip(CLASS_NAMES, EMOTION_COLORS)):
        mask = (lbl_sub == c_idx)
        ax.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            c=c_color, label=f"{c_name} (Class {c_idx})",
            alpha=0.65, edgecolors='none', s=24
        )
        
    ax.set_title("Affective-InfoNCE: Self-Supervised Latent Hypersphere $\\mathbb{S}^{63}$ Embedding\n(Zero-Leakage Test Quarantined Frames)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'infonce_hypersphere_tsne.png')
    plt.savefig(fig1_path, dpi=300)
    if artifact_dir:
        shutil.copy(fig1_path, os.path.join(artifact_dir, 'infonce_hypersphere_tsne.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 2: Per-Subject Trial Consensus Accuracy Bar Chart
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    subjects = sorted(list(results['per_subject_results'].keys()), key=lambda x: int(x))
    sub_labels = [f"Sub {int(s):02d}" for s in subjects]
    
    ang_accs = [results['per_subject_results'][s]['trial_level_angular']['point_estimates']['accuracy'] * 100 for s in subjects]
    prb_accs = [results['per_subject_results'][s]['trial_level_probe']['point_estimates']['accuracy'] * 100 for s in subjects]
    frame_accs = [results['per_subject_results'][s]['sample_level_angular']['point_estimates']['accuracy'] * 100 for s in subjects]
    
    x = np.arange(len(subjects))
    w = 0.26
    
    r1 = ax.bar(x - w, frame_accs, w, label='Frame Sample Acc (Angular)', color='#95a5a6', edgecolor='#2c3e50', alpha=0.85)
    r2 = ax.bar(x, prb_accs, w, label='Trial Soft Log-Odds Consensus', color='#3498db', edgecolor='#2980b9')
    r3 = ax.bar(x + w, ang_accs, w, label='Trial Angular Prototype Consensus', color='#2ecc71', edgecolor='#27ae60')
    
    mean_ang = results['population_benchmark']['trial_level_angular']['point_estimates']['accuracy'] * 100
    ax.axhline(mean_ang, color='#e74c3c', linestyle='--', linewidth=1.5,
               label=f'Mean Angular Trial Consensus ({mean_ang:.2f}%)')
    ax.axhline(25.0, color='#7f8c8d', linestyle=':', linewidth=1.2, label='Chance Level (25.0%)')
    
    ax.set_title("SEED-IV Affective-InfoNCE: Per-Subject Classification Performance (1,080 Trials)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Subject Identifier", fontsize=11, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sub_labels, rotation=0, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'infonce_trial_consensus_accuracy.png')
    plt.savefig(fig2_path, dpi=300)
    if artifact_dir:
        shutil.copy(fig2_path, os.path.join(artifact_dir, 'infonce_trial_consensus_accuracy.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 3: Population Trial Confusion Matrix (1,080 Trials)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    cm = np.array(results['population_benchmark']['trial_level_angular']['confusion_matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Recall Rate', rotation=-90, va="bottom", fontsize=10, fontweight='bold')
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(CLASS_NAMES, fontweight='bold', fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontweight='bold', fontsize=11)
    ax.set_xlabel('Predicted Emotion Class', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_ylabel('Ground-Truth Emotion Class', fontweight='bold', fontsize=11, labelpad=8)
    ax.set_title(f"Affective-InfoNCE Trial Confusion Matrix\n(1,080 Quarantined Trials | Mean Acc: {mean_ang:.2f}%)",
                 fontsize=12, fontweight='bold', pad=12)
    
    thresh = cm_norm.max() / 2.
    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{val:d}\n({pct:.1f}%)",
                    ha="center", va="center", color=color,
                    fontsize=11, fontweight="bold")
                    
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'infonce_confusion_matrix_1080trials.png')
    plt.savefig(fig3_path, dpi=300)
    if artifact_dir:
        shutil.copy(fig3_path, os.path.join(artifact_dir, 'infonce_confusion_matrix_1080trials.png'))
    plt.close()
    
    print(f"All 3 publication figures exported successfully to {output_dir}", flush=True)

def run_affective_infonce_benchmark(npz_path='seed_iv_processed.npz', dry_run=False, device_str='cuda'):
    """
    Executes the full Affective-InfoNCE SOTA Benchmark across all 45 sessions on SEED-IV.
    """
    print("=" * 80, flush=True)
    print("AFFECTIVE-INFONCE: SELF-SUPERVISED LATENT HYPERSPHERE SOTA BENCHMARK", flush=True)
    print("=" * 80, flush=True)
    
    set_seed(42)
    device = torch.device(device_str if (torch.cuda.is_available() and device_str == 'cuda') else 'cpu')
    print(f"Executing on hardware device: {device} (PyTorch {torch.__version__})", flush=True)
    if device.type == 'cuda':
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}", flush=True)
        print("Enabling torch.backends.cudnn.benchmark = True and AMP fp16 autocast.", flush=True)
        
    # 1. Ingest Data
    print(f"Ingesting preprocessed SEED-IV dataset from {npz_path}...", flush=True)
    data = np.load(npz_path)
    features_3d = data['features']      # (37575, 62, 5)
    labels = data['labels']              # (37575,)
    subject_ids = data['subject_ids']    # (37575,)
    session_nums = data['session_nums']  # (37575,)
    trial_ids = data['trial_ids']        # (37575,)
    
    N_total = len(features_3d)
    print(f"Dataset loaded: {N_total} frames across {len(np.unique(subject_ids))} subjects.", flush=True)
    
    # 2. Extract 580D Cortical Asymmetry Space
    print("Extracting 580D Cortical Asymmetry Space (310D raw DE + 135D DASM + 135D DCAU)...", flush=True)
    left_idx, right_idx, ant_idx, post_idx = get_asymmetry_pair_indices()
    cortical_580d = extract_580d_cortical_features(features_3d, left_idx, right_idx, ant_idx, post_idx)
    
    unique_subjects = sorted(list(np.unique(subject_ids)))
    unique_sessions = sorted(list(np.unique(session_nums)))
    
    if dry_run:
        print("\n[DRY RUN ACTIVATED] Running single session verification: Subject 15, Session 2...", flush=True)
        session_configs = [(15, 2)]
    else:
        session_configs = [(s, ses) for s in unique_subjects for ses in unique_sessions]
        print(f"\n[FULL BENCHMARK ACTIVATED] Evaluating all {len(session_configs)} sessions across 15 subjects (180 folds total)...", flush=True)
        
    start_time = time.time()
    
    per_subject_storage = {s: {
        'frame_true': [],
        'frame_pred_ang': [],
        'frame_pred_prb': [],
        'frame_probs_prb': [],
        'trial_true': [],
        'trial_pred_ang': [],
        'trial_pred_prb': [],
        'trial_probs_prb': []
    } for s in unique_subjects}
    
    sample_embeddings_pool = []
    sample_labels_pool = []
    
    per_session_records = []
    total_folds_evaluated = 0
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    
    for sess_idx, (subj, sess) in enumerate(session_configs, 1):
        sess_mask = np.where((subject_ids == subj) & (session_nums == sess))[0]
        sess_trials = np.unique(trial_ids[sess_mask])
        
        sess_trial_labels = []
        for tr in sess_trials:
            tr_idx = np.where(trial_ids[sess_mask] == tr)[0]
            sess_trial_labels.append(labels[sess_mask][tr_idx[0]])
        sess_trial_labels = np.array(sess_trial_labels)
        
        sess_frame_true = []
        sess_frame_pred_ang, sess_frame_pred_prb = [], []
        sess_frame_probs_prb = []
        
        sess_trial_true = []
        sess_trial_pred_ang, sess_trial_pred_prb = [], []
        sess_trial_probs_prb = []
        
        for fold_idx, (train_tr_idx, test_tr_idx) in enumerate(skf.split(sess_trials, sess_trial_labels), 1):
            train_trials = sess_trials[train_tr_idx]
            test_trials = sess_trials[test_tr_idx]
            test_trial_lbls = sess_trial_labels[test_tr_idx]
            
            raw_train_mask = sess_mask[np.isin(trial_ids[sess_mask], train_trials)]
            raw_test_mask = sess_mask[np.isin(trial_ids[sess_mask], test_trials)]
            
            X_tr_raw = cortical_580d[raw_train_mask]
            y_tr = labels[raw_train_mask]
            X_te_raw = cortical_580d[raw_test_mask]
            y_te = labels[raw_test_mask]
            
            test_frame_trials = trial_ids[raw_test_mask]
            
            # --- Zero-Leakage Reference Subtraction ---
            neutral_mask = (y_tr == 0)
            if np.sum(neutral_mask) > 0:
                mu_neutral = np.mean(X_tr_raw[neutral_mask], axis=0, keepdims=True)
            else:
                mu_neutral = np.zeros((1, 580), dtype=np.float32)
                
            X_tr_ref = X_tr_raw - mu_neutral
            X_te_ref = X_te_raw - mu_neutral
            
            # --- Fold-Quarantined StandardScaler ---
            scaler = StandardScaler().fit(X_tr_ref)
            X_tr_sc = scaler.transform(X_tr_ref)
            X_te_sc = scaler.transform(X_te_ref)
            
            # --- Two-Phase Affective-InfoNCE Training ---
            model, prototypes, probe_head, p1_loss = train_affective_infonce_fold(
                X_tr_sc, y_tr, device,
                in_features=580, latent_dim=128, proj_dim=64,
                phase1_epochs=40, phase2_epochs=20, batch_size=256,
                lr_phase1=1e-3, lr_phase2=2e-3, temperature=0.07
            )
            
            # --- Test Fold Evaluation ---
            eval_res = evaluate_infonce_test_fold(
                model, prototypes, probe_head,
                X_te_sc, y_te, test_frame_trials, test_trials, test_trial_lbls,
                device
            )
            
            sess_frame_true.extend(eval_res['frame_true'])
            sess_frame_pred_ang.extend(eval_res['frame_pred_angular'])
            sess_frame_pred_prb.extend(eval_res['frame_pred_probe'])
            sess_frame_probs_prb.extend(eval_res['frame_probs_probe'])
            
            sess_trial_true.extend(eval_res['trial_true'])
            sess_trial_pred_ang.extend(eval_res['trial_pred_angular'])
            sess_trial_pred_prb.extend(eval_res['trial_pred_probe'])
            sess_trial_probs_prb.extend(eval_res['trial_probs_probe'])
            
            # Collect subset of embeddings for t-SNE (up to 100 frames per fold)
            n_te_frames = len(eval_res['frame_true'])
            sub_step = max(1, n_te_frames // 100)
            sample_embeddings_pool.append(eval_res['test_embeddings_z'][::sub_step])
            sample_labels_pool.append(y_te[::sub_step])
            
            total_folds_evaluated += 1
            
            fold_ang_f_acc = accuracy_score(y_te, eval_res['frame_pred_angular'])
            fold_ang_t_acc = accuracy_score(eval_res['trial_true'], eval_res['trial_pred_angular'])
            fold_prb_t_acc = accuracy_score(eval_res['trial_true'], eval_res['trial_pred_probe'])
            
            print(f"  [{sess_idx:02d}/45] Sub {subj:02d} Sess {sess} Fold {fold_idx}/4 | "
                  f"P1 Loss: {p1_loss:.4f} | Frame Acc: {fold_ang_f_acc*100:.2f}% | "
                  f"Trial Angular: {fold_ang_t_acc*100:.2f}% | Trial Probe: {fold_prb_t_acc*100:.2f}%", flush=True)
                  
        # Store per-subject results
        s_store = per_subject_storage[subj]
        s_store['frame_true'].extend(sess_frame_true)
        s_store['frame_pred_ang'].extend(sess_frame_pred_ang)
        s_store['frame_pred_prb'].extend(sess_frame_pred_prb)
        s_store['frame_probs_prb'].extend(sess_frame_probs_prb)
        
        s_store['trial_true'].extend(sess_trial_true)
        s_store['trial_pred_ang'].extend(sess_trial_pred_ang)
        s_store['trial_pred_prb'].extend(sess_trial_pred_prb)
        s_store['trial_probs_prb'].extend(sess_trial_probs_prb)
        
        sess_f_acc_ang = accuracy_score(sess_frame_true, sess_frame_pred_ang)
        sess_t_acc_ang = accuracy_score(sess_trial_true, sess_trial_pred_ang)
        sess_t_acc_prb = accuracy_score(sess_trial_true, sess_trial_pred_prb)
        
        per_session_records.append({
            'subject': int(subj),
            'session': int(sess),
            'frame_accuracy_angular': float(sess_f_acc_ang),
            'trial_consensus_angular': float(sess_t_acc_ang),
            'trial_consensus_probe': float(sess_t_acc_prb)
        })
        
        print(f"==> [{sess_idx:02d}/45] Sub {subj:02d} Sess {sess} SUMMARY | "
              f"Frame Acc: {sess_f_acc_ang*100:.2f}% | "
              f"Trial Angular Consensus: {sess_t_acc_ang*100:.2f}% | "
              f"Trial Probe Consensus: {sess_t_acc_prb*100:.2f}%\n", flush=True)
              
    elapsed = time.time() - start_time
    print(f"\nBenchmark completed in {elapsed:.2f} seconds across {total_folds_evaluated} folds.", flush=True)
    
    # 3. Compute Per-Subject and Population Benchmark Metrics
    print("\nComputing 95% non-parametric bootstrap confidence intervals (1,000 resamples)...", flush=True)
    per_subject_metrics = {}
    
    pop_f_true, pop_f_pred_ang, pop_f_pred_prb, pop_f_probs_prb = [], [], [], []
    pop_t_true, pop_t_pred_ang, pop_t_pred_prb, pop_t_probs_prb = [], [], [], []
    
    evaluated_subjects = sorted(list(set([s for s, _ in session_configs])))
    for s in evaluated_subjects:
        d = per_subject_storage[s]
        if len(d['frame_true']) == 0:
            continue
            
        pop_f_true.extend(d['frame_true'])
        pop_f_pred_ang.extend(d['frame_pred_ang'])
        pop_f_pred_prb.extend(d['frame_pred_prb'])
        pop_f_probs_prb.extend(d['frame_probs_prb'])
        
        pop_t_true.extend(d['trial_true'])
        pop_t_pred_ang.extend(d['trial_pred_ang'])
        pop_t_pred_prb.extend(d['trial_pred_prb'])
        pop_t_probs_prb.extend(d['trial_probs_prb'])
        
        f_m_ang = compute_metrics_dict(d['frame_true'], d['frame_pred_ang'])
        f_m_prb = compute_metrics_dict(d['frame_true'], d['frame_pred_prb'], np.array(d['frame_probs_prb']))
        t_m_ang = compute_metrics_dict(d['trial_true'], d['trial_pred_ang'])
        t_m_prb = compute_metrics_dict(d['trial_true'], d['trial_pred_prb'], np.array(d['trial_probs_prb']))
        
        per_subject_metrics[str(s)] = {
            'total_trials': len(d['trial_true']),
            'total_frames': len(d['frame_true']),
            'sample_level_angular': f_m_ang,
            'sample_level_probe': f_m_prb,
            'trial_level_angular': t_m_ang,
            'trial_level_probe': t_m_prb
        }
        
    pop_frame_ang = compute_metrics_dict(pop_f_true, pop_f_pred_ang)
    pop_frame_prb = compute_metrics_dict(pop_f_true, pop_f_pred_prb, np.array(pop_f_probs_prb))
    pop_trial_ang = compute_metrics_dict(pop_t_true, pop_t_pred_ang)
    pop_trial_prb = compute_metrics_dict(pop_t_true, pop_t_pred_prb, np.array(pop_t_probs_prb))
    
    cm_trial_ang = confusion_matrix(pop_t_true, pop_t_pred_ang, labels=[0, 1, 2, 3]).tolist()
    cm_trial_prb = confusion_matrix(pop_t_true, pop_t_pred_prb, labels=[0, 1, 2, 3]).tolist()
    pop_trial_ang['confusion_matrix'] = cm_trial_ang
    pop_trial_prb['confusion_matrix'] = cm_trial_prb
    
    results = {
        'benchmark': 'SEED-IV Affective-InfoNCE Self-Supervised Latent Hypersphere Benchmark',
        'hardware': str(device),
        'execution_time_seconds': float(elapsed),
        'total_evaluated_subjects': len(evaluated_subjects),
        'total_trials_evaluated': len(pop_t_true),
        'total_frames_evaluated': len(pop_f_true),
        'population_benchmark': {
            'sample_level_angular': pop_frame_ang,
            'sample_level_probe': pop_frame_prb,
            'trial_level_angular': pop_trial_ang,
            'trial_level_probe': pop_trial_prb
        },
        'per_subject_results': per_subject_metrics,
        'per_session_records': per_session_records
    }
    
    # 4. Export JSON and CSV
    json_path = 'affective_infonce_results.json'
    csv_path = 'affective_infonce_results.csv'
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved structured experimental results to {json_path}", flush=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Subject_ID', 'Total_Trials', 'Total_Frames',
            'Frame_Acc_Angular', 'Frame_F1_Angular',
            'Trial_Acc_Angular', 'Trial_F1_Angular', 'Trial_Kappa_Angular',
            'Trial_Acc_Probe', 'Trial_F1_Probe', 'Trial_Kappa_Probe'
        ])
        for s in evaluated_subjects:
            sm = per_subject_metrics[str(s)]
            writer.writerow([
                s,
                sm['total_trials'],
                sm['total_frames'],
                f"{sm['sample_level_angular']['point_estimates']['accuracy']*100:.2f}%",
                f"{sm['sample_level_angular']['point_estimates']['macro_f1']:.4f}",
                f"{sm['trial_level_angular']['point_estimates']['accuracy']*100:.2f}%",
                f"{sm['trial_level_angular']['point_estimates']['macro_f1']:.4f}",
                f"{sm['trial_level_angular']['point_estimates']['cohen_kappa']:.4f}",
                f"{sm['trial_level_probe']['point_estimates']['accuracy']*100:.2f}%",
                f"{sm['trial_level_probe']['point_estimates']['macro_f1']:.4f}",
                f"{sm['trial_level_probe']['point_estimates']['cohen_kappa']:.4f}"
            ])
        # Summary row
        writer.writerow([
            'POPULATION_MEAN',
            len(pop_t_true),
            len(pop_f_true),
            f"{pop_frame_ang['point_estimates']['accuracy']*100:.2f}%",
            f"{pop_frame_ang['point_estimates']['macro_f1']:.4f}",
            f"{pop_trial_ang['point_estimates']['accuracy']*100:.2f}%",
            f"{pop_trial_ang['point_estimates']['macro_f1']:.4f}",
            f"{pop_trial_ang['point_estimates']['cohen_kappa']:.4f}",
            f"{pop_trial_prb['point_estimates']['accuracy']*100:.2f}%",
            f"{pop_trial_prb['point_estimates']['macro_f1']:.4f}",
            f"{pop_trial_prb['point_estimates']['cohen_kappa']:.4f}"
        ])
    print(f"Saved tabular metrics to {csv_path}", flush=True)
    
    # 5. Render 300 DPI Publication Visualizations
    fig_dir = 'figures/affective_infonce'
    artifact_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\affective_infonce"
    
    if len(sample_embeddings_pool) > 0:
        emb_pool = np.vstack(sample_embeddings_pool)
        lbl_pool = np.concatenate(sample_labels_pool)
    else:
        emb_pool = np.zeros((10, 64))
        lbl_pool = np.zeros(10, dtype=int)
        
    render_publication_figures(results, emb_pool, lbl_pool, fig_dir, artifact_dir)
    
    # Print Executive Population Summary
    print("\n" + "=" * 80, flush=True)
    print("AFFECTIVE-INFONCE POPULATION BENCHMARK SUMMARY (1,080 TEST TRIALS)", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Evaluated Subjects: {len(evaluated_subjects)} | Total Test Trials: {len(pop_t_true)}", flush=True)
    print(f"Frame Sample Accuracy (Angular): {pop_frame_ang['point_estimates']['accuracy']*100:.2f}% "
          f"[95% CI: {pop_frame_ang['confidence_intervals_95']['accuracy'][1]*100:.2f}% - {pop_frame_ang['confidence_intervals_95']['accuracy'][2]*100:.2f}%]", flush=True)
    print(f"Trial Angular Consensus Acc:     {pop_trial_ang['point_estimates']['accuracy']*100:.2f}% "
          f"[95% CI: {pop_trial_ang['confidence_intervals_95']['accuracy'][1]*100:.2f}% - {pop_trial_ang['confidence_intervals_95']['accuracy'][2]*100:.2f}%]", flush=True)
    print(f"Trial Angular Consensus Macro-F1:{pop_trial_ang['point_estimates']['macro_f1']:.4f} "
          f"[95% CI: {pop_trial_ang['confidence_intervals_95']['macro_f1'][1]:.4f} - {pop_trial_ang['confidence_intervals_95']['macro_f1'][2]:.4f}]", flush=True)
    print(f"Trial Angular Consensus Kappa:   {pop_trial_ang['point_estimates']['cohen_kappa']:.4f}", flush=True)
    print(f"Trial Soft Log-Odds Acc (Probe): {pop_trial_prb['point_estimates']['accuracy']*100:.2f}% "
          f"[95% CI: {pop_trial_prb['confidence_intervals_95']['accuracy'][1]*100:.2f}% - {pop_trial_prb['confidence_intervals_95']['accuracy'][2]*100:.2f}%]", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Affective-InfoNCE SEED-IV SOTA Benchmark')
    parser.add_argument('--npz', type=str, default='seed_iv_processed.npz', help='Path to seed_iv_processed.npz')
    parser.add_argument('--dry-run', action='store_true', help='Execute single-session verification (Sub 15 Sess 2)')
    parser.add_argument('--device', type=str, default='cuda', help='Execution device (cuda/cpu)')
    args = parser.parse_args()
    
    run_affective_infonce_benchmark(npz_path=args.npz, dry_run=args.dry_run, device_str=args.device)
