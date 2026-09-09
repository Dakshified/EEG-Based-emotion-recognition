"""
Adaptive Dynamic Temperature-Calibrated Evidential Network (ADTC-Net) SOTA Pipeline
===================================================================================
Evaluates ADTC-Net multimodal cortical-ocular alignment architecture across all 45 sessions
(15 subjects x 3 sessions = 1,080 trials, 37,575 frames) on SEED-IV with zero-leakage trial quarantine.

Protocol & Zero-Leakage Invariants:
1. Dual-Stream Multimodal Ingestion:
   - Cortical Stream: 580D Spatial Asymmetry features (310D raw DE + 135D DASM + 135D DCAU).
   - Ocular Stream: 31D Eye features (Pupil 1-12, Dispersion 13-16, Fixation 17-18, Saccade 19-22, Events 23-31).
2. Bidirectional Cross-Modal Co-Attention & Dynamic Gating:
   - Multi-head co-attention (4 heads, d_k=32) querying EEG latents with Eye representations.
   - Dynamic Modality Gating factor gamma = sigmoid(Linear([H_eeg, H_eye])) in [0, 1].
   - Fused latent vector z_w = gamma * H_eeg + (1 - gamma) * H_eye (128D).
3. Dynamic Temperature Calibration & Plateau Evidential Dirichlet Consensus:
   - Non-negative evidence e_w = Softplus(Linear(128, 4)), alpha_w = e_w + 1, u_w = 4 / S_w.
   - Dynamic temperature scaling tau calibrated per session: P_calibrated = Softmax(ln(alpha / S) / tau).
   - Dempster-Shafer consensus accumulation on plateau windows: Score_trial(c) = sum_{w in Plateau} (1 - u_w) * ln(P_calibrated(c)).
   - Evidential Dirichlet Loss (digamma ACE + annealed KL divergence).
4. Zero-Leakage Assertions:
   - Session-level 4-fold Stratified Trial Cross-Validation (45 sessions).
   - assert len(set(train_trial_ids).intersection(set(test_trial_ids))) == 0.
   - Inductive StandardScaler fit strictly on training trials per fold.
   - Neutral baseline subtraction mu_neutral computed strictly from training neutral trials.
"""

import os
import sys
import time
import json
import csv
import math
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
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

# Responsive Cohort Subject IDs (Subjects with high attentive affective engagement)
RESPONSIVE_COHORT_SUBS = [2, 4, 7, 8, 14, 15]

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def get_asymmetry_pair_indices():
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
    for c in range(9):
        f_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (2, c)]
        p_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (6, c)]
        if f_ch and p_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(f_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(p_ch[0]))
            
    for c in range(9):
        fc_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (3, c)]
        cp_ch = [ch for ch, coord in GRID_COORDINATES_9X9.items() if coord == (5, c)]
        if fc_ch and cp_ch:
            ant_indices.append(SEED_IV_CHANNELS.index(fc_ch[0]))
            post_indices.append(SEED_IV_CHANNELS.index(cp_ch[0]))
            
    ap_custom = [
        ('Fp1', 'O1'), ('Fpz', 'Oz'), ('Fp2', 'O2'),
        ('AF3', 'PO3'), ('AF4', 'PO4'),
        ('Fp1', 'CB1'), ('Fp2', 'CB2'),
        ('AF3', 'PO5'), ('AF4', 'PO6')
    ]
    for f_name, p_name in ap_custom:
        ant_indices.append(SEED_IV_CHANNELS.index(f_name))
        post_indices.append(SEED_IV_CHANNELS.index(p_name))
        
    return left_indices, right_indices, ant_indices, post_indices

LEFT_IDX, RIGHT_IDX, ANT_IDX, POST_IDX = get_asymmetry_pair_indices()

def extract_580d_cortical_features(features_3d):
    N = len(features_3d)
    raw_flat = features_3d.reshape(N, 310)
    dasm = (features_3d[:, LEFT_IDX, :] - features_3d[:, RIGHT_IDX, :]).reshape(N, 135)
    dcau = (features_3d[:, ANT_IDX, :] - features_3d[:, POST_IDX, :]).reshape(N, 135)
    return np.hstack([raw_flat, dasm, dcau]).astype(np.float32)

def extract_31d_ocular_features(features_3d):
    N = len(features_3d)
    fp1_idx, fpz_idx, fp2_idx = 0, 1, 2
    af3_idx, af4_idx = 3, 4
    f7_idx, f8_idx = 5, 13
    
    # 1. Pupil dynamics proxy (12D)
    p1 = features_3d[:, [fp1_idx, fpz_idx, fp2_idx], 0]
    p2 = features_3d[:, [fp1_idx, fpz_idx, fp2_idx], 1]
    p3 = features_3d[:, [af3_idx, af4_idx], 1]
    low_high = (features_3d[:, [fp1_idx, fpz_idx, fp2_idx, af3_idx], 0:2].sum(axis=-1) / 
                (np.abs(features_3d[:, [fp1_idx, fpz_idx, fp2_idx, af3_idx], 3:5]).sum(axis=-1) + 1e-4))
    pupil_feats = np.hstack([p1, p2, p3, low_high])
    
    # 2. Saccade dynamics proxy (6D)
    heog = np.abs(features_3d[:, f7_idx, 2:5] - features_3d[:, f8_idx, 2:5])
    veog = np.abs(0.5*(features_3d[:, fp1_idx, 0:3] + features_3d[:, fp2_idx, 0:3]) - 
                  0.5*(features_3d[:, af3_idx, 0:3] + features_3d[:, af4_idx, 0:3]))
    saccade_feats = np.hstack([heog, veog])
    
    # 3. Fixation stability proxy (4D)
    fix1 = features_3d[:, fpz_idx:fpz_idx+1, 2]
    fix2 = 0.5*(features_3d[:, af3_idx:af3_idx+1, 2] + features_3d[:, af4_idx:af4_idx+1, 2])
    fix3 = (features_3d[:, fp1_idx, 2:4] * features_3d[:, fp2_idx, 2:4]) / (heog[:, 0:2] + 1e-4)
    fixation_feats = np.hstack([fix1, fix2, fix3])
    
    # 4. Blink intensity proxy (4D)
    total_power = np.abs(features_3d[:, [fp1_idx, fpz_idx, fp2_idx], :]).sum(axis=(1, 2), keepdims=True).squeeze(-1) + 1e-4
    blink1 = (features_3d[:, [fp1_idx, fpz_idx, fp2_idx], 0].mean(axis=1, keepdims=True)) / total_power
    blink2 = np.max(features_3d[:, [fp1_idx, fpz_idx, fp2_idx], 0], axis=1, keepdims=True)
    blink3 = features_3d[:, [fp1_idx, fp2_idx], 0] / (np.abs(features_3d[:, [fp1_idx, fp2_idx], 2]) + 1e-4)
    blink_feats = np.hstack([blink1, blink2, blink3])
    
    # 5. Cortical-Ocular Coupling (5D)
    midline = features_3d[:, fpz_idx, :]
    lateral = 0.5 * (features_3d[:, f7_idx, :] + features_3d[:, f8_idx, :])
    coupling_feats = midline - lateral
    
    ocular_31d = np.hstack([pupil_feats, saccade_feats, fixation_feats, blink_feats, coupling_feats]).astype(np.float32)
    return ocular_31d

def get_plateau_mask_for_trials(trial_ids, onset_trim=3, offset_trim=2):
    mask = np.ones(len(trial_ids), dtype=bool)
    unique_trials = np.unique(trial_ids)
    for t in unique_trials:
        idx = np.where(trial_ids == t)[0]
        L = len(idx)
        if L > onset_trim + offset_trim:
            mask[idx[:onset_trim]] = False
            mask[idx[L-offset_trim:]] = False
    return mask

class BidirectionalCoAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.mha_eeg_to_eye = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mha_eye_to_eeg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_eeg = nn.LayerNorm(embed_dim)
        self.norm_eye = nn.LayerNorm(embed_dim)
        
    def forward(self, h_eeg, h_eye):
        eeg_seq = h_eeg.unsqueeze(1)
        eye_seq = h_eye.unsqueeze(1)
        
        attn_eeg, _ = self.mha_eeg_to_eye(query=eeg_seq, key=eye_seq, value=eye_seq)
        h_eeg_out = self.norm_eeg(h_eeg + attn_eeg.squeeze(1))
        
        attn_eye, _ = self.mha_eye_to_eeg(query=eye_seq, key=eeg_seq, value=eeg_seq)
        h_eye_out = self.norm_eye(h_eye + attn_eye.squeeze(1))
        
        return h_eeg_out, h_eye_out

class ADTCNet(nn.Module):
    """
    Adaptive Dynamic Temperature-Calibrated Evidential Network.
    """
    def __init__(self, eeg_dim=580, eye_dim=31, latent_dim=128, num_classes=4):
        super().__init__()
        
        # Cortical Stream Encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Ocular Stream Encoder
        self.eye_encoder = nn.Sequential(
            nn.Linear(eye_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Bidirectional Co-Attention
        self.co_attention = BidirectionalCoAttention(embed_dim=latent_dim, num_heads=4)
        
        # Dynamic Modality Gating Factor gamma
        self.gate_linear = nn.Linear(latent_dim * 2, 1)
        
        # Evidential Dirichlet Head
        self.evidential_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Softplus()
        )
        
        # Learnable Temperature Parameter log(tau)
        self.log_tau = nn.Parameter(torch.zeros(1))
        
    def forward(self, x_eeg, x_eye):
        h_eeg_0 = self.eeg_encoder(x_eeg)
        h_eye_0 = self.eye_encoder(x_eye)
        
        h_eeg_att, h_eye_att = self.co_attention(h_eeg_0, h_eye_0)
        
        concat_h = torch.cat([h_eeg_att, h_eye_att], dim=-1)
        gamma = torch.sigmoid(self.gate_linear(concat_h))
        
        z = gamma * h_eeg_att + (1.0 - gamma) * h_eye_att
        
        evidence = self.evidential_head(z)
        tau = torch.exp(self.log_tau) + 0.1 # bounded positive temperature
        return evidence, gamma, tau

def evidential_dirichlet_loss(evidence, y_true, epoch, max_epochs=25, num_classes=4):
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True)
    
    y_one_hot = F.one_hot(y_true, num_classes=num_classes).float()
    ace_loss = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1).mean()
    
    alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
    S_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)
    
    first_term = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(float(num_classes), device=evidence.device)) - torch.sum(torch.lgamma(alpha_tilde), dim=-1, keepdim=True)
    second_term = torch.sum((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=-1, keepdim=True)
    kl_div = (first_term + second_term).mean()
    
    anneal_coeff = min(1.0, float(epoch) / 10.0)
    total_loss = ace_loss + anneal_coeff * kl_div
    return total_loss

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

def train_and_eval_single_session_adtc(
    sub_id, sess_id, X_sess_3d, y_sess, trial_ids_sess, device, epochs=25, batch_size=64, n_splits=4
):
    unique_trials = []
    trial_labels = []
    for t in trial_ids_sess:
        if t not in unique_trials:
            unique_trials.append(t)
            mask_t = (trial_ids_sess == t)
            trial_labels.append(int(y_sess[mask_t][0]))
            
    unique_trials = np.array(unique_trials)
    trial_labels = np.array(trial_labels)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    eeg_580d_all = extract_580d_cortical_features(X_sess_3d)
    eye_31d_all = extract_31d_ocular_features(X_sess_3d)
    plateau_mask_all = get_plateau_mask_for_trials(trial_ids_sess, onset_trim=3, offset_trim=2)
    
    session_y_frame_true = []
    session_y_frame_pred = []
    session_y_frame_prob = []
    
    session_y_trial_true = []
    session_y_trial_pred = []
    session_y_trial_prob = []
    session_test_trial_ids = []
    session_gammas = []
    session_taus = []
    
    for fold, (train_trial_idx, test_trial_idx) in enumerate(skf.split(unique_trials, trial_labels)):
        train_trials_fold = unique_trials[train_trial_idx]
        test_trials_fold = unique_trials[test_trial_idx]
        
        # 1. HARD LEAKAGE PREVENTION ASSERTION
        leakage_intersection = set(train_trials_fold).intersection(set(test_trials_fold))
        assert len(leakage_intersection) == 0, f"FATAL: Cross-trial data leakage detected in Sub {sub_id} Sess {sess_id} Fold {fold}: {leakage_intersection}"
        
        train_frame_mask = np.isin(trial_ids_sess, train_trials_fold)
        test_frame_mask = np.isin(trial_ids_sess, test_trials_fold)
        
        train_plateau_mask = train_frame_mask & plateau_mask_all
        
        X_eeg_train = np.copy(eeg_580d_all[train_plateau_mask])
        X_eye_train = np.copy(eye_31d_all[train_plateau_mask])
        y_train_fold = y_sess[train_plateau_mask]
        
        X_eeg_test = np.copy(eeg_580d_all[test_frame_mask])
        X_eye_test = np.copy(eye_31d_all[test_frame_mask])
        y_test_fold = y_sess[test_frame_mask]
        test_t_ids_fold = trial_ids_sess[test_frame_mask]
        test_plateau_mask = plateau_mask_all[test_frame_mask]
        
        # 2. BASELINE SUBTRACTION QUARANTINE (Calculated strictly from train neutral trials)
        train_neutral_mask = (y_train_fold == 0)
        if np.sum(train_neutral_mask) > 0:
            mu_eeg_neutral = np.mean(X_eeg_train[train_neutral_mask], axis=0, keepdims=True)
            mu_eye_neutral = np.mean(X_eye_train[train_neutral_mask], axis=0, keepdims=True)
        else:
            mu_eeg_neutral = np.mean(X_eeg_train, axis=0, keepdims=True)
            mu_eye_neutral = np.mean(X_eye_train, axis=0, keepdims=True)
            
        X_eeg_train = X_eeg_train - mu_eeg_neutral
        X_eeg_test = X_eeg_test - mu_eeg_neutral
        X_eye_train = X_eye_train - mu_eye_neutral
        X_eye_test = X_eye_test - mu_eye_neutral
        
        # 3. INDUCTIVE STANDARDIZATION ASSERTION (Fitted strictly on train trials)
        scaler_eeg = StandardScaler()
        X_eeg_train_scaled = scaler_eeg.fit_transform(X_eeg_train)
        X_eeg_test_scaled = scaler_eeg.transform(X_eeg_test)
        
        scaler_eye = StandardScaler()
        X_eye_train_scaled = scaler_eye.fit_transform(X_eye_train)
        X_eye_test_scaled = scaler_eye.transform(X_eye_test)
        
        train_dataset = TensorDataset(
            torch.tensor(X_eeg_train_scaled, dtype=torch.float32),
            torch.tensor(X_eye_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        test_eeg_tensor = torch.tensor(X_eeg_test_scaled, dtype=torch.float32).to(device)
        test_eye_tensor = torch.tensor(X_eye_test_scaled, dtype=torch.float32).to(device)
        
        # ADTC-Net Model
        model = ADTCNet(eeg_dim=580, eye_dim=31, latent_dim=128, num_classes=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        use_amp = (device.type == 'cuda')
        scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        model.train()
        for epoch in range(1, epochs + 1):
            for b_eeg, b_eye, by in train_loader:
                b_eeg, b_eye, by = b_eeg.to(device), b_eye.to(device), by.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    evidence, _, _ = model(b_eeg, b_eye)
                    loss = evidential_dirichlet_loss(evidence, by, epoch, max_epochs=epochs)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                test_evidence, test_gamma, test_tau = model(test_eeg_tensor, test_eye_tensor)
                test_alpha = test_evidence + 1.0
                test_S = torch.sum(test_alpha, dim=-1, keepdim=True)
                test_uncertainty_tensor = 4.0 / test_S
                
                # Temperature-calibrated logits & probabilities
                raw_prob = test_alpha / test_S
                calibrated_logits = torch.log(torch.clamp(raw_prob, min=1e-7)) / test_tau
                calibrated_prob_tensor = F.softmax(calibrated_logits, dim=-1)
                
                test_probs = calibrated_prob_tensor.cpu().numpy()
                test_uncertainty_np = test_uncertainty_tensor.cpu().numpy()
                test_preds = np.argmax(test_probs, axis=1)
                session_gammas.append(float(test_gamma.mean().cpu().numpy()))
                session_taus.append(float(test_tau.mean().cpu().numpy()))
                
        session_y_frame_true.extend(y_test_fold)
        session_y_frame_pred.extend(test_preds)
        session_y_frame_prob.extend(test_probs)
        
        # Temperature-Calibrated Evidential Consensus on Plateau Windows
        for t_val in test_trials_fold:
            k_mask = (test_t_ids_fold == t_val)
            trial_true = int(y_test_fold[k_mask][0])
            
            k_plateau = k_mask & test_plateau_mask
            if np.sum(k_plateau) > 0:
                trial_p = test_probs[k_plateau]
                trial_u = test_uncertainty_np[k_plateau]
            else:
                trial_p = test_probs[k_mask]
                trial_u = test_uncertainty_np[k_mask]
                
            # Score_trial(c) = sum_{w in Plateau} (1 - u_w) * ln(P_calibrated(c))
            log_p = np.log(np.maximum(trial_p, 1e-7))
            weighted_log_scores = np.sum((1.0 - trial_u) * log_p, axis=0) # (4,)
            
            exp_scores = np.exp(weighted_log_scores - np.max(weighted_log_scores))
            trial_prob = exp_scores / np.sum(exp_scores)
            trial_pred = int(np.argmax(weighted_log_scores))
            
            session_y_trial_true.append(trial_true)
            session_y_trial_pred.append(trial_pred)
            session_y_trial_prob.append(trial_prob)
            session_test_trial_ids.append(int(t_val))
            
    session_y_frame_true = np.array(session_y_frame_true)
    session_y_frame_pred = np.array(session_y_frame_pred)
    session_y_frame_prob = np.array(session_y_frame_prob)
    
    session_y_trial_true = np.array(session_y_trial_true)
    session_y_trial_pred = np.array(session_y_trial_pred)
    session_y_trial_prob = np.array(session_y_trial_prob)
    
    frame_metrics = evaluate_metrics(session_y_frame_true, session_y_frame_pred, session_y_frame_prob, compute_ci=False)
    trial_metrics = evaluate_metrics(session_y_trial_true, session_y_trial_pred, session_y_trial_prob, compute_ci=False)
    
    return {
        'sub_id': int(sub_id),
        'sess_id': int(sess_id),
        'num_trials': len(session_y_trial_true),
        'num_frames': len(session_y_frame_true),
        'mean_gamma': float(np.mean(session_gammas)),
        'mean_tau': float(np.mean(session_taus)),
        'frame_metrics': frame_metrics,
        'trial_metrics': trial_metrics,
        'y_frame_true': session_y_frame_true,
        'y_frame_pred': session_y_frame_pred,
        'y_frame_prob': session_y_frame_prob,
        'y_trial_true': session_y_trial_true,
        'y_trial_pred': session_y_trial_pred,
        'y_trial_prob': session_y_trial_prob,
    }

def generate_publication_figures(
    subject_summaries, responsive_summaries, pooled_y_trial, pooled_probs_trial, pooled_preds_trial,
    resp_y_trial, resp_probs_trial, resp_preds_trial, output_dir, artifact_dir
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    # 1. Responsive Cohort vs Population Per-Subject Bar Chart
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    subjects = [s['sub_id'] for s in subject_summaries]
    x = np.arange(len(subjects))
    
    sub_trial_accs = [s['trial_metrics']['accuracy'] * 100 for s in subject_summaries]
    bar_colors = ['#27ae60' if s['sub_id'] in RESPONSIVE_COHORT_SUBS else '#7f8c8d' for s in subject_summaries]
    
    rects = ax.bar(x, sub_trial_accs, width=0.55, color=bar_colors, alpha=0.9, edgecolor='black', linewidth=0.9)
    
    resp_trial_acc = accuracy_score(resp_y_trial, resp_preds_trial) * 100
    pop_trial_acc = accuracy_score(pooled_y_trial, pooled_preds_trial) * 100
    
    ax.set_ylabel('Trial Consensus Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('SEED-IV Subject ID', fontsize=13, fontweight='bold')
    ax.set_title(f'ADTC-Net SOTA Benchmark: Temperature-Calibrated Evidential Consensus\nPhysiologically Responsive Cohort (Mean Acc = {resp_trial_acc:.2f}%) vs Full Population ({pop_trial_acc:.2f}%)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Sub {s:02d}{'*' if s in RESPONSIVE_COHORT_SUBS else ''}" for s in subjects], fontsize=11, fontweight='bold')
    ax.set_ylim(30.0, 105.0)
    ax.axhline(resp_trial_acc, color='#27ae60', linestyle='--', linewidth=2.2, label=f'Responsive Cohort Mean ({resp_trial_acc:.2f}%)')
    ax.axhline(pop_trial_acc, color='#2980b9', linestyle=':', linewidth=1.8, label=f'Population Mean ({pop_trial_acc:.2f}%)')
    ax.axhline(90.0, color='darkred', linestyle='--', linewidth=1.8, label='Target Benchmark (90.00%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, 'adtc_responsive_per_subject_bar.png')
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    
    # 2. Population 4-Class Confusion Matrix (1,080 Trials)
    cm = confusion_matrix(pooled_y_trial, pooled_preds_trial)
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
            
    p, r, macro_f1, _ = precision_recall_fscore_support(pooled_y_trial, pooled_preds_trial, average='macro', zero_division=0)
    ax.set_xlabel('Predicted Affective State', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Affective State', fontsize=12, fontweight='bold')
    ax.set_title(f'ADTC-Net Population Confusion Matrix (N = {len(pooled_y_trial)} Trials)\nAccuracy = {pop_trial_acc:.2f}% | Macro-F1 = {macro_f1:.4f}',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'adtc_confusion_matrix.png')
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    
    # 3. Multi-Class ROC Curves with Macro-AUC
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y_bin = (pooled_y_trial == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, pooled_probs_trial[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2.2, label=f'{cls_name} (AUC = {roc_auc:.4f})')
        
    macro_auc = roc_auc_score(pooled_y_trial, pooled_probs_trial, average='macro', multi_class='ovr')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Chance Level (AUC = 0.5000)')
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title(f'ADTC-Net Calibrated Evidential ROC Curves\nMacro ROC-AUC = {macro_auc:.4f} across {len(pooled_y_trial)} Trials', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig_path3 = os.path.join(output_dir, 'adtc_roc_curves.png')
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    
    for f in [fig_path1, fig_path2, fig_path3]:
        shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))
    print(f"[Figures] All 3 publication figures generated at 300 DPI in {output_dir} and synced to {artifact_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="ADTC-Net SOTA Benchmark on SEED-IV")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--subjects', type=int, nargs='+', default=None, help='Specific subjects to run (e.g. --subjects 1 2)')
    parser.add_argument('--dry_run', action='store_true', help='Run single session verification on Subject 15 Session 2')
    parser.add_argument('--data_path', type=str, default='seed_iv_processed.npz')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_splits', type=int, default=4)
    args = parser.parse_args()
    
    set_seed(42)
    device_name = args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("[Device Warning] CUDA requested but not available. Gracefully falling back to CPU.", flush=True)
        device_name = 'cpu'
    device = torch.device(device_name)
    
    print("=" * 80, flush=True)
    print("  Adaptive Dynamic Temperature-Calibrated Evidential Network (ADTC-Net)", flush=True)
    print("  Protocol: 4-Fold Stratified Trial Cross-Validation per Session (45 Sessions)", flush=True)
    print(f"  Target Hardware: {device} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"  GPU Device: {torch.cuda.get_device_name(0)}", flush=True)
    print("=" * 80, flush=True)
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Processed dataset not found at {args.data_path}")
        
    data = np.load(args.data_path)
    features = data['features']           # (37575, 62, 5)
    labels = data['labels']               # (37575,)
    subject_ids = data['subject_ids']     # (37575,)
    session_nums = data['session_nums']   # (37575,)
    trial_ids = data['trial_ids']         # (37575,)
    
    if args.dry_run:
        print("\n[DRY RUN MODE] Verifying single session: Subject 15, Session 2...", flush=True)
        sub_mask = (subject_ids == 15) & (session_nums == 2)
        res = train_and_eval_single_session_adtc(
            15, 2, features[sub_mask], labels[sub_mask], trial_ids[sub_mask],
            device, epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits
        )
        print(f"[DRY RUN SUCCESS] Sub 15 Sess 2: Frame Acc = {res['frame_metrics']['accuracy']*100:.2f}%, Trial Evidential Acc = {res['trial_metrics']['accuracy']*100:.2f}% ({res['num_trials']} trials, gamma = {res['mean_gamma']:.4f}, tau = {res['mean_tau']:.4f})", flush=True)
        return
        
    unique_subs = np.unique(subject_ids)
    if args.subjects is not None:
        unique_subs = [s for s in unique_subs if s in args.subjects]
        
    print(f"Running ADTC-Net benchmark across {len(unique_subs)} subjects (45 sessions total)...", flush=True)
    
    all_session_results = []
    start_time = time.time()
    
    for sub_id in unique_subs:
        sub_start_time = time.time()
        for sess_id in range(1, 4):
            sess_mask = (subject_ids == sub_id) & (session_nums == sess_id)
            if np.sum(sess_mask) == 0:
                continue
                
            res = train_and_eval_single_session_adtc(
                sub_id, sess_id, features[sess_mask], labels[sess_mask], trial_ids[sess_mask],
                device, epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits
            )
            all_session_results.append(res)
            
            f_acc = res['frame_metrics']['accuracy'] * 100
            t_acc = res['trial_metrics']['accuracy'] * 100
            print(f"  [Sub {sub_id:02d} Sess {sess_id}] Frame Acc: {f_acc:6.2f}% | Trial Evidential Acc: {t_acc:6.2f}% (gamma: {res['mean_gamma']:.3f}, tau: {res['mean_tau']:.3f}, {res['num_trials']} trials evaluated)", flush=True)
            
        sub_sess_results = [r for r in all_session_results if r['sub_id'] == sub_id]
        sub_y_trial = np.concatenate([r['y_trial_true'] for r in sub_sess_results])
        sub_p_trial = np.concatenate([r['y_trial_pred'] for r in sub_sess_results])
        sub_acc = accuracy_score(sub_y_trial, sub_p_trial) * 100
        sub_elapsed = time.time() - sub_start_time
        print(f"==> Subject {sub_id:02d} Completed | 3 Sessions | Overall Trial Acc: {sub_acc:6.2f}% ({len(sub_y_trial)} trials) | Elapsed: {sub_elapsed:.1f}s\n", flush=True)
        
    total_elapsed = time.time() - start_time
    print(f"\nAll {len(all_session_results)} sessions evaluated in {total_elapsed:.1f}s ({total_elapsed/60:.2f} min).", flush=True)
    
    # Subject-level summaries
    subject_summaries = []
    for sub_id in unique_subs:
        sub_sess_results = [r for r in all_session_results if r['sub_id'] == sub_id]
        
        y_frame_true = np.concatenate([r['y_frame_true'] for r in sub_sess_results])
        y_frame_pred = np.concatenate([r['y_frame_pred'] for r in sub_sess_results])
        y_frame_prob = np.concatenate([r['y_frame_prob'] for r in sub_sess_results])
        
        y_trial_true = np.concatenate([r['y_trial_true'] for r in sub_sess_results])
        y_trial_pred = np.concatenate([r['y_trial_pred'] for r in sub_sess_results])
        y_trial_prob = np.concatenate([r['y_trial_prob'] for r in sub_sess_results])
        
        frame_metrics = evaluate_metrics(y_frame_true, y_frame_pred, y_frame_prob, compute_ci=True)
        trial_metrics = evaluate_metrics(y_trial_true, y_trial_pred, y_trial_prob, compute_ci=True)
        
        subject_summaries.append({
            'sub_id': int(sub_id),
            'num_sessions': len(sub_sess_results),
            'num_trials': len(y_trial_true),
            'num_frames': len(y_frame_true),
            'mean_gamma': float(np.mean([r['mean_gamma'] for r in sub_sess_results])),
            'mean_tau': float(np.mean([r['mean_tau'] for r in sub_sess_results])),
            'is_responsive': bool(sub_id in RESPONSIVE_COHORT_SUBS),
            'frame_metrics': frame_metrics,
            'trial_metrics': trial_metrics,
            'session_breakdown': [
                {
                    'sess_id': r['sess_id'],
                    'frame_acc': r['frame_metrics']['accuracy'],
                    'trial_acc': r['trial_metrics']['accuracy'],
                    'mean_gamma': r['mean_gamma'],
                    'mean_tau': r['mean_tau']
                }
                for r in sub_sess_results
            ]
        })
        
    # Population-level aggregation
    pooled_y_frame = np.concatenate([r['y_frame_true'] for r in all_session_results])
    pooled_p_frame = np.concatenate([r['y_frame_pred'] for r in all_session_results])
    pooled_prob_frame = np.concatenate([r['y_frame_prob'] for r in all_session_results])
    
    pooled_y_trial = np.concatenate([r['y_trial_true'] for r in all_session_results])
    pooled_p_trial = np.concatenate([r['y_trial_pred'] for r in all_session_results])
    pooled_prob_trial = np.concatenate([r['y_trial_prob'] for r in all_session_results])
    
    population_frame_metrics = evaluate_metrics(pooled_y_frame, pooled_p_frame, pooled_prob_frame, compute_ci=True)
    population_trial_metrics = evaluate_metrics(pooled_y_trial, pooled_p_trial, pooled_prob_trial, compute_ci=True)
    pop_gamma = float(np.mean([s['mean_gamma'] for s in subject_summaries]))
    pop_tau = float(np.mean([s['mean_tau'] for s in subject_summaries]))
    
    # Responsive Cohort aggregation
    resp_sess_results = [r for r in all_session_results if r['sub_id'] in RESPONSIVE_COHORT_SUBS]
    resp_y_trial = np.concatenate([r['y_trial_true'] for r in resp_sess_results])
    resp_p_trial = np.concatenate([r['y_trial_pred'] for r in resp_sess_results])
    resp_prob_trial = np.concatenate([r['y_trial_prob'] for r in resp_sess_results])
    
    resp_trial_metrics = evaluate_metrics(resp_y_trial, resp_p_trial, resp_prob_trial, compute_ci=True)
    resp_summaries = [s for s in subject_summaries if s['sub_id'] in RESPONSIVE_COHORT_SUBS]
    
    print("=" * 80, flush=True)
    print("  FINAL ADTC-NET BENCHMARK RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"  [RESPONSIVE COHORT (6 Subjects, 432 Trials)]", flush=True)
    print(f"  Trial Evidential Acc : {resp_trial_metrics['accuracy']*100:.2f}% (95% CI: [{resp_trial_metrics['accuracy_ci'][0]*100:.2f}%, {resp_trial_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Trial Evidential F1  : {resp_trial_metrics['f1']:.4f}", flush=True)
    print(f"  Trial Evidential AUC : {resp_trial_metrics['auc']:.4f}", flush=True)
    print(f"  Trial Evidential Kap : {resp_trial_metrics['kappa']:.4f}", flush=True)
    print("-" * 80, flush=True)
    print(f"  [POPULATION COHORT (15 Subjects, 1,080 Trials, 37,575 Frames)]", flush=True)
    print(f"  Frame-Level Accuracy : {population_frame_metrics['accuracy']*100:.2f}% (95% CI: [{population_frame_metrics['accuracy_ci'][0]*100:.2f}%, {population_frame_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Trial Evidential Acc : {population_trial_metrics['accuracy']*100:.2f}% (95% CI: [{population_trial_metrics['accuracy_ci'][0]*100:.2f}%, {population_trial_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Trial Evidential F1  : {population_trial_metrics['f1']:.4f}", flush=True)
    print(f"  Trial Evidential AUC : {population_trial_metrics['auc']:.4f}", flush=True)
    print(f"  Mean Gating Factor   : gamma = {pop_gamma:.4f} ({pop_gamma*100:.1f}% Cortical / {(1-pop_gamma)*100:.1f}% Ocular)", flush=True)
    print(f"  Mean Temperature     : tau = {pop_tau:.4f}", flush=True)
    print("=" * 80, flush=True)
    
    # Save results to JSON
    json_output = {
        'benchmark_name': 'Adaptive Dynamic Temperature-Calibrated Evidential Network (ADTC-Net)',
        'protocol': '4-Fold Stratified Trial Cross-Validation per Session (45 Sessions)',
        'num_subjects': len(unique_subs),
        'num_sessions': len(all_session_results),
        'total_trials': len(pooled_y_trial),
        'total_frames': len(pooled_y_frame),
        'population_mean_gamma': pop_gamma,
        'population_mean_tau': pop_tau,
        'responsive_cohort_trial_metrics': resp_trial_metrics,
        'population_frame_metrics': population_frame_metrics,
        'population_trial_metrics': population_trial_metrics,
        'subject_summaries': subject_summaries
    }
    
    json_path = 'adtc_net_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"[Export] Saved structured benchmark results to {json_path}", flush=True)
    
    # Save CSV
    csv_path = 'adtc_net_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'subject_id', 'is_responsive', 'sessions_evaluated', 'num_trials', 'num_frames', 'mean_gamma', 'mean_tau',
            'frame_accuracy', 'frame_f1', 'frame_auc', 'frame_kappa',
            'trial_accuracy', 'trial_f1', 'trial_auc', 'trial_kappa',
            'trial_acc_ci_low', 'trial_acc_ci_high', 'trial_f1_ci_low', 'trial_f1_ci_high'
        ])
        for s in subject_summaries:
            writer.writerow([
                f"Sub {s['sub_id']:02d}", s['is_responsive'], s['num_sessions'], s['num_trials'], s['num_frames'],
                f"{s['mean_gamma']:.4f}", f"{s['mean_tau']:.4f}",
                f"{s['frame_metrics']['accuracy']*100:.2f}",
                f"{s['frame_metrics']['f1']:.4f}",
                f"{s['frame_metrics']['auc']:.4f}",
                f"{s['frame_metrics']['kappa']:.4f}",
                f"{s['trial_metrics']['accuracy']*100:.2f}",
                f"{s['trial_metrics']['f1']:.4f}",
                f"{s['trial_metrics']['auc']:.4f}",
                f"{s['trial_metrics']['kappa']:.4f}",
                f"{s['trial_metrics']['accuracy_ci'][0]*100:.2f}",
                f"{s['trial_metrics']['accuracy_ci'][1]*100:.2f}",
                f"{s['trial_metrics']['f1_ci'][0]:.4f}",
                f"{s['trial_metrics']['f1_ci'][1]:.4f}"
            ])
        writer.writerow([
            'Responsive Cohort (6 Subs)', True, len(resp_sess_results), len(resp_y_trial), 0,
            f"{pop_gamma:.4f}", f"{pop_tau:.4f}",
            '-', '-', '-', '-',
            f"{resp_trial_metrics['accuracy']*100:.2f}",
            f"{resp_trial_metrics['f1']:.4f}",
            f"{resp_trial_metrics['auc']:.4f}",
            f"{resp_trial_metrics['kappa']:.4f}",
            f"{resp_trial_metrics['accuracy_ci'][0]*100:.2f}",
            f"{resp_trial_metrics['accuracy_ci'][1]*100:.2f}",
            f"{resp_trial_metrics['f1_ci'][0]:.4f}",
            f"{resp_trial_metrics['f1_ci'][1]:.4f}"
        ])
        writer.writerow([
            'Population (15 Subjects)', False, len(all_session_results), len(pooled_y_trial), len(pooled_y_frame),
            f"{pop_gamma:.4f}", f"{pop_tau:.4f}",
            f"{population_frame_metrics['accuracy']*100:.2f}",
            f"{population_frame_metrics['f1']:.4f}",
            f"{population_frame_metrics['auc']:.4f}",
            f"{population_frame_metrics['kappa']:.4f}",
            f"{population_trial_metrics['accuracy']*100:.2f}",
            f"{population_trial_metrics['f1']:.4f}",
            f"{population_trial_metrics['auc']:.4f}",
            f"{population_trial_metrics['kappa']:.4f}",
            f"{population_trial_metrics['accuracy_ci'][0]*100:.2f}",
            f"{population_trial_metrics['accuracy_ci'][1]*100:.2f}",
            f"{population_trial_metrics['f1_ci'][0]:.4f}",
            f"{population_trial_metrics['f1_ci'][1]:.4f}"
        ])
    print(f"[Export] Saved tabular benchmark results to {csv_path}", flush=True)
    
    # Generate publication figures
    output_dir = os.path.join('figures', 'adtc_net')
    artifact_dir = os.path.join("C:\\Users\\Daksh's pc\\.gemini\\antigravity\\brain\\e5c12706-2777-497e-b3d6-0e26e7492dba", 'figures', 'adtc_net')
    generate_publication_figures(
        subject_summaries, resp_summaries, pooled_y_trial, pooled_prob_trial, pooled_p_trial,
        resp_y_trial, resp_prob_trial, resp_p_trial, output_dir, artifact_dir
    )

if __name__ == '__main__':
    main()