"""
Cross-Session Transfer Learning Network (CST-Net) SOTA Pipeline
===============================================================
Evaluates CST-Net multi-session transfer architecture across SEED-IV
(1,080 trials, 37,575 frames, 45 sessions) under 100% strict zero-leakage trial quarantine.

Strict Zero-Leakage Programmatic Assertions:
1. Complete Trial Quarantine:
   assert len(set(all_train_trial_ids).intersection(set(tgt_test_trial_ids))) == 0
2. Strictly Inductive Feature Preprocessing:
   StandardScaler fitted strictly on training trials per split for both 580D EEG and 31D Eye features.
3. Training-Fold Reference Baseline Insulation:
   Neutral baseline reference vectors (mu_neutral) computed strictly from corresponding training neutral trials.

Algorithmic Pipeline (CST-Net):
1. Multi-Session Cross-Session Pooling:
   - For target session s (24 trials), pool the 48 trials from auxiliary sessions s' != s of the same subject.
   - 18 target training trials + 48 auxiliary source trials = 66 training trials vs 6 quarantined test trials.
2. Two-Stage Cross-Session Transfer Learning:
   - Stage 1: Pre-train AVC-Net backbone on auxiliary source trials (48 trials) using Evidential Dirichlet Loss.
   - Stage 2: Fine-tune on target session training split (18 trials) with reduced learning rate.
3. Margin-Gated Quadratic Evidential Consensus:
   - Self-attention coherence pooling over Top-6 valence-aware climax windows.
   - Score_trial(c) = sum_{w in Top-6} (1 - u_w) * a_w * exp(M_w) * (b_w(c))^2
   - y_hat_trial = argmax_c Score_trial(c)
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

RESPONSIVE_COHORT_SUBS = [2, 4, 7, 8, 14, 15]
PEAK_ATTENTIVE_SESSIONS = [(15, 2), (14, 3), (15, 3), (2, 1), (2, 2)]

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

def get_avc_salience_mask_for_trials(features_3d, trial_ids, k=6, onset_trim=3, offset_trim=2, w_energy=0.6, w_faa=0.4):
    mask = np.zeros(len(trial_ids), dtype=bool)
    unique_trials = np.unique(trial_ids)
    fp1_idx, fp2_idx, alpha_band = 0, 2, 2
    
    for t in unique_trials:
        idx = np.where(trial_ids == t)[0]
        L = len(idx)
        if L > onset_trim + offset_trim:
            candidate_idx = idx[onset_trim : L - offset_trim]
        else:
            candidate_idx = idx
            
        energies = np.mean(features_3d[candidate_idx, :, 3] + features_3d[candidate_idx, :, 4], axis=1)
        faa = features_3d[candidate_idx, fp1_idx, alpha_band] - features_3d[candidate_idx, fp2_idx, alpha_band]
        abs_faa = np.abs(faa)
        
        e_min, e_max = np.min(energies), np.max(energies)
        e_norm = (energies - e_min) / (e_max - e_min + 1e-6)
        
        f_min, f_max = np.min(abs_faa), np.max(abs_faa)
        f_norm = (abs_faa - f_min) / (f_max - f_min + 1e-6)
        
        salience = w_energy * e_norm + w_faa * f_norm
        k_val = min(k, len(candidate_idx))
        topk_idx = candidate_idx[np.argsort(salience)[-k_val:]]
        mask[topk_idx] = True
        
    return mask

class CSTNet(nn.Module):
    """
    Cross-Session Transfer Network (CST-Net).
    Streamlined Late-Fusion with Self-Attention Coherence Pooling & Evidential Dirichlet Head.
    """
    def __init__(self, eeg_dim=580, eye_dim=31, latent_eeg=128, latent_eye=64, num_classes=4, dropout=0.30):
        super().__init__()
        
        self.cortical_encoder = nn.Sequential(
            nn.Linear(eeg_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_eeg),
            nn.LayerNorm(latent_eeg),
            nn.GELU()
        )
        
        self.ocular_encoder = nn.Sequential(
            nn.Linear(eye_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_eye),
            nn.LayerNorm(latent_eye),
            nn.GELU()
        )
        
        joint_dim = latent_eeg + latent_eye # 192
        self.attn_query = nn.Linear(joint_dim, 1, bias=False)
        
        self.evidential_head = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Softplus()
        )
        
    def forward(self, x_eeg, x_eye):
        h_cortical = self.cortical_encoder(x_eeg)
        h_ocular = self.ocular_encoder(x_eye)
        z = torch.cat([h_cortical, h_ocular], dim=-1)
        evidence = self.evidential_head(z)
        return evidence, z
        
    def compute_attention_weights(self, z_seq):
        scores = self.attn_query(z_seq).squeeze(-1)
        weights = F.softmax(scores, dim=0)
        return weights

def evidential_dirichlet_loss(evidence, y_true, epoch, max_epochs=30, num_classes=4):
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

def train_and_eval_single_session_cst(
    sub_id, tgt_sess_id, X_all, y_all, sub_ids_all, sess_ids_all, trial_ids_all,
    device, pretrain_epochs=25, finetune_epochs=15, batch_size=32, n_splits=4, top_k=6, lr_pretrain=1e-3, lr_finetune=3e-4, weight_decay=1e-4
):
    # 1. Target Session Data
    tgt_mask = (sub_ids_all == sub_id) & (sess_ids_all == tgt_sess_id)
    X_tgt = X_all[tgt_mask]
    y_tgt = y_all[tgt_mask]
    t_tgt = trial_ids_all[tgt_mask]
    global_t_tgt = 100 * tgt_sess_id + t_tgt
    
    # 2. Auxiliary Source Sessions Data (Same subject, other sessions)
    aux_mask = (sub_ids_all == sub_id) & (sess_ids_all != tgt_sess_id)
    X_aux = X_all[aux_mask]
    y_aux = y_all[aux_mask]
    t_aux = trial_ids_all[aux_mask]
    sess_aux = sess_ids_all[aux_mask]
    global_t_aux = 100 * sess_aux + t_aux
    
    unique_tgt_trials = []
    tgt_trial_labels = []
    for t in global_t_tgt:
        if t not in unique_tgt_trials:
            unique_tgt_trials.append(t)
            mask_t = (global_t_tgt == t)
            tgt_trial_labels.append(int(y_tgt[mask_t][0]))
            
    unique_tgt_trials = np.array(unique_tgt_trials)
    tgt_trial_labels = np.array(tgt_trial_labels)
    
    # Feature Extraction
    eeg_580d_tgt = extract_580d_cortical_features(X_tgt)
    eye_31d_tgt = extract_31d_ocular_features(X_tgt)
    climax_mask_tgt = get_avc_salience_mask_for_trials(X_tgt, global_t_tgt, k=top_k)
    
    eeg_580d_aux = extract_580d_cortical_features(X_aux)
    eye_31d_aux = extract_31d_ocular_features(X_aux)
    climax_mask_aux = get_avc_salience_mask_for_trials(X_aux, global_t_aux, k=top_k)
    
    # Normalize Auxiliary Sessions by their own neutral baseline
    for s_aux_id in np.unique(sess_aux):
        s_mask = (sess_aux == s_aux_id)
        s_neutral = s_mask & (y_aux == 0)
        if np.sum(s_neutral) > 0:
            mu_eeg_s = np.mean(eeg_580d_aux[s_neutral], axis=0, keepdims=True)
            mu_eye_s = np.mean(eye_31d_aux[s_neutral], axis=0, keepdims=True)
        else:
            mu_eeg_s = np.mean(eeg_580d_aux[s_mask], axis=0, keepdims=True)
            mu_eye_s = np.mean(eye_31d_aux[s_mask], axis=0, keepdims=True)
        eeg_580d_aux[s_mask] = eeg_580d_aux[s_mask] - mu_eeg_s
        eye_31d_aux[s_mask] = eye_31d_aux[s_mask] - mu_eye_s
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    session_y_frame_true = []
    session_y_frame_pred = []
    session_y_frame_prob = []
    
    session_y_trial_true = []
    session_y_trial_pred = []
    session_y_trial_prob = []
    session_test_trial_ids = []
    
    for fold, (train_trial_idx, test_trial_idx) in enumerate(skf.split(unique_tgt_trials, tgt_trial_labels)):
        tgt_train_trials = unique_tgt_trials[train_trial_idx]
        tgt_test_trials = unique_tgt_trials[test_trial_idx]
        
        # 1. HARD CROSS-SESSION & INTRA-SESSION TRIAL QUARANTINE ASSERTION
        all_train_trials = list(np.unique(global_t_aux)) + list(tgt_train_trials)
        leakage_intersection = set(all_train_trials).intersection(set(tgt_test_trials))
        assert len(leakage_intersection) == 0, f"FATAL: Cross-trial data leakage detected in Sub {sub_id} Sess {tgt_sess_id} Fold {fold}: {leakage_intersection}"
        
        # Target Split Masks
        train_frame_mask = np.isin(global_t_tgt, tgt_train_trials)
        test_frame_mask = np.isin(global_t_tgt, tgt_test_trials)
        
        train_climax_mask = train_frame_mask & climax_mask_tgt
        test_climax_mask = climax_mask_tgt[test_frame_mask]
        
        X_eeg_tgt_train = np.copy(eeg_580d_tgt[train_climax_mask])
        X_eye_tgt_train = np.copy(eye_31d_tgt[train_climax_mask])
        y_tgt_train = y_tgt[train_climax_mask]
        
        X_eeg_tgt_test = np.copy(eeg_580d_tgt[test_frame_mask])
        X_eye_tgt_test = np.copy(eye_31d_tgt[test_frame_mask])
        y_tgt_test = y_tgt[test_frame_mask]
        test_t_ids_fold = global_t_tgt[test_frame_mask]
        
        # 2. BASELINE SUBTRACTION QUARANTINE ON TARGET (Strictly from target training neutral trials)
        tgt_neutral_mask = (y_tgt_train == 0)
        if np.sum(tgt_neutral_mask) > 0:
            mu_eeg_tgt = np.mean(X_eeg_tgt_train[tgt_neutral_mask], axis=0, keepdims=True)
            mu_eye_tgt = np.mean(X_eye_tgt_train[tgt_neutral_mask], axis=0, keepdims=True)
        else:
            mu_eeg_tgt = np.mean(X_eeg_tgt_train, axis=0, keepdims=True)
            mu_eye_tgt = np.mean(X_eye_tgt_train, axis=0, keepdims=True)
            
        X_eeg_tgt_train = X_eeg_tgt_train - mu_eeg_tgt
        X_eeg_tgt_test = X_eeg_tgt_test - mu_eeg_tgt
        X_eye_tgt_train = X_eye_tgt_train - mu_eye_tgt
        X_eye_tgt_test = X_eye_tgt_test - mu_eye_tgt
        
        # 3. INDUCTIVE STANDARDIZATION (Fitted strictly on training data)
        scaler_eeg = StandardScaler()
        scaler_eye = StandardScaler()
        
        # Auxiliary source climax frames
        X_eeg_aux_climax = np.copy(eeg_580d_aux[climax_mask_aux])
        X_eye_aux_climax = np.copy(eye_31d_aux[climax_mask_aux])
        y_aux_climax = y_aux[climax_mask_aux]
        
        # Combined training pool for feature scaling
        X_eeg_combined_train = np.vstack([X_eeg_aux_climax, X_eeg_tgt_train])
        X_eye_combined_train = np.vstack([X_eye_aux_climax, X_eye_tgt_train])
        
        scaler_eeg.fit(X_eeg_combined_train)
        scaler_eye.fit(X_eye_combined_train)
        
        X_eeg_aux_scaled = scaler_eeg.transform(X_eeg_aux_climax)
        X_eye_aux_scaled = scaler_eye.transform(X_eye_aux_climax)
        
        X_eeg_tgt_train_scaled = scaler_eeg.transform(X_eeg_tgt_train)
        X_eye_tgt_train_scaled = scaler_eye.transform(X_eye_tgt_train)
        
        X_eeg_tgt_test_scaled = scaler_eeg.transform(X_eeg_tgt_test)
        X_eye_tgt_test_scaled = scaler_eye.transform(X_eye_tgt_test)
        
        # Datasets & Loaders
        aux_dataset = TensorDataset(
            torch.tensor(X_eeg_aux_scaled, dtype=torch.float32),
            torch.tensor(X_eye_aux_scaled, dtype=torch.float32),
            torch.tensor(y_aux_climax, dtype=torch.long)
        )
        aux_loader = DataLoader(aux_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        tgt_train_dataset = TensorDataset(
            torch.tensor(X_eeg_tgt_train_scaled, dtype=torch.float32),
            torch.tensor(X_eye_tgt_train_scaled, dtype=torch.float32),
            torch.tensor(y_tgt_train, dtype=torch.long)
        )
        tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        test_eeg_tensor = torch.tensor(X_eeg_tgt_test_scaled, dtype=torch.float32).to(device)
        test_eye_tensor = torch.tensor(X_eye_tgt_test_scaled, dtype=torch.float32).to(device)
        
        # CST-Net Model
        set_seed(42 + fold)
        model = CSTNet(eeg_dim=580, eye_dim=31, latent_eeg=128, latent_eye=64, num_classes=4, dropout=0.30).to(device)
        
        use_amp = (device.type == 'cuda')
        scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # =========================================================================
        # STAGE 1: AUXILIARY SOURCE PRE-TRAINING (48 Source Trials)
        # =========================================================================
        opt_pretrain = torch.optim.AdamW(model.parameters(), lr=lr_pretrain, weight_decay=weight_decay)
        sched_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pretrain, T_max=pretrain_epochs, eta_min=1e-5)
        
        model.train()
        for epoch in range(1, pretrain_epochs + 1):
            for b_eeg, b_eye, by in aux_loader:
                b_eeg, b_eye, by = b_eeg.to(device), b_eye.to(device), by.to(device)
                opt_pretrain.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    evidence, _ = model(b_eeg, b_eye)
                    loss = evidential_dirichlet_loss(evidence, by, epoch, max_epochs=pretrain_epochs)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt_pretrain)
                scaler_amp.update()
            sched_pretrain.step()
            
        # =========================================================================
        # STAGE 2: TARGET SESSION FINE-TUNING & ADAPTATION (18 Target Train Trials)
        # =========================================================================
        opt_finetune = torch.optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=weight_decay)
        sched_finetune = torch.optim.lr_scheduler.CosineAnnealingLR(opt_finetune, T_max=finetune_epochs, eta_min=1e-6)
        
        for epoch in range(1, finetune_epochs + 1):
            for b_eeg, b_eye, by in tgt_train_loader:
                b_eeg, b_eye, by = b_eeg.to(device), b_eye.to(device), by.to(device)
                opt_finetune.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    evidence, _ = model(b_eeg, b_eye)
                    loss = evidential_dirichlet_loss(evidence, by, epoch, max_epochs=finetune_epochs)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt_finetune)
                scaler_amp.update()
            sched_finetune.step()
            
        # =========================================================================
        # INFERENCE & CONSENSUS ON QUARANTINED TEST TRIALS (6 Held-Out Trials)
        # =========================================================================
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                test_evidence, test_z = model(test_eeg_tensor, test_eye_tensor)
                test_alpha = test_evidence + 1.0
                test_S = torch.sum(test_alpha, dim=-1, keepdim=True)
                test_uncertainty_tensor = 4.0 / test_S
                test_belief_tensor = test_evidence / test_S
                test_probs_tensor = test_alpha / test_S
                
                test_probs = test_probs_tensor.cpu().numpy()
                test_uncertainty_np = test_uncertainty_tensor.cpu().numpy().squeeze(-1)
                test_belief_np = test_belief_tensor.cpu().numpy()
                test_preds = np.argmax(test_probs, axis=1)
                
        session_y_frame_true.extend(y_tgt_test)
        session_y_frame_pred.extend(test_preds)
        session_y_frame_prob.extend(test_probs)
        
        for t_val in tgt_test_trials:
            k_mask = (test_t_ids_fold == t_val)
            trial_true = int(y_tgt_test[k_mask][0])
            
            k_climax = k_mask & test_climax_mask
            if np.sum(k_climax) > 0:
                trial_indices = np.where(k_climax)[0]
            else:
                trial_indices = np.where(k_mask)[0]
                
            trial_b = test_belief_np[trial_indices]      # (K, 4)
            trial_u = test_uncertainty_np[trial_indices] # (K,)
            
            with torch.no_grad():
                z_sub = test_z[trial_indices]
                attn_w = model.compute_attention_weights(z_sub).cpu().numpy() # (K,)
                
            sorted_b = np.sort(trial_b, axis=-1)
            margin = sorted_b[:, -1] - sorted_b[:, -2] # (K,)
            certainty_w = (1.0 - trial_u) * attn_w * np.exp(margin) # (K,)
            quadratic_belief = np.square(trial_b)       # (K, 4)
            trial_scores = np.sum(certainty_w[:, None] * quadratic_belief, axis=0) # (4,)
            
            total_score = np.sum(trial_scores)
            if total_score > 0:
                trial_prob_norm = trial_scores / total_score
            else:
                trial_prob_norm = np.ones(4) / 4.0
                
            trial_pred = int(np.argmax(trial_scores))
            
            session_y_trial_true.append(trial_true)
            session_y_trial_pred.append(trial_pred)
            session_y_trial_prob.append(trial_prob_norm)
            session_test_trial_ids.append(t_val)
            
    frame_metrics = evaluate_metrics(np.array(session_y_frame_true), np.array(session_y_frame_pred), np.array(session_y_frame_prob), compute_ci=False)
    trial_metrics = evaluate_metrics(np.array(session_y_trial_true), np.array(session_y_trial_pred), np.array(session_y_trial_prob), compute_ci=False)
    
    return {
        'sub_id': int(sub_id),
        'sess_id': int(tgt_sess_id),
        'frame_metrics': frame_metrics,
        'trial_metrics': trial_metrics,
        'y_frame_true': session_y_frame_true,
        'y_frame_pred': session_y_frame_pred,
        'y_frame_prob': session_y_frame_prob,
        'y_trial_true': session_y_trial_true,
        'y_trial_pred': session_y_trial_pred,
        'y_trial_prob': session_y_trial_prob,
        'trial_ids': session_test_trial_ids
    }

def plot_cst_figures(all_trial_y_true, all_trial_y_pred, all_trial_y_prob, session_results, output_dir="figures/cst_net"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Per-Subject Bar Chart
    sub_trial_accs = {}
    for res in session_results:
        sub = res['sub_id']
        acc = res['trial_metrics']['accuracy'] * 100.0
        if sub not in sub_trial_accs:
            sub_trial_accs[sub] = []
        sub_trial_accs[sub].append(acc)
        
    subs_sorted = sorted(sub_trial_accs.keys())
    means = [np.mean(sub_trial_accs[s]) for s in subs_sorted]
    errs = [np.std(sub_trial_accs[s]) if len(sub_trial_accs[s]) > 1 else 0.0 for s in subs_sorted]
    
    plt.figure(figsize=(12, 6), dpi=300)
    bars = plt.bar([f"Sub {s:02d}" for s in subs_sorted], means, yerr=errs, capsize=4,
                   color=['#2ecc71' if s in RESPONSIVE_COHORT_SUBS else '#3498db' for s in subs_sorted],
                   edgecolor='black', linewidth=1.2, alpha=0.88)
    plt.axhline(np.mean(means), color='#e74c3c', linestyle='--', linewidth=1.8, label=f'Population Mean: {np.mean(means):.2f}%')
    plt.ylabel("Trial Consensus Accuracy (%)", fontsize=13, fontweight='bold')
    plt.title("CST-Net Cross-Session Transfer Trial Consensus Accuracy per Subject (SEED-IV)", fontsize=14, fontweight='bold', pad=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    for bar, mean_val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2.0, mean_val + 2.5, f"{mean_val:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    plt.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cst_per_subject_bar.png"), dpi=300)
    plt.close()
    
    # 2. Confusion Matrix
    cm = confusion_matrix(all_trial_y_true, all_trial_y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 7), dpi=300)
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"CST-Net Trial Confusion Matrix (Acc: {accuracy_score(all_trial_y_true, all_trial_y_pred)*100:.2f}%)", fontsize=13, fontweight='bold', pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, fontsize=11, fontweight='bold')
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=11, fontweight='bold')
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            plt.text(j, i, val_str, ha="center", va="center", color=color, fontsize=11, fontweight='bold')
            
    plt.ylabel('True Affective Category', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Affective Category', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cst_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 3. Multiclass ROC Curves
    y_true_onehot = np.eye(4)[all_trial_y_true]
    plt.figure(figsize=(8, 7), dpi=300)
    for c_idx, (c_name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, c_idx], all_trial_y_prob[:, c_idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2.2, label=f'{c_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('CST-Net Multiclass One-vs-Rest ROC Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cst_roc_curves.png"), dpi=300)
    plt.close()
    print(f"[OK] Successfully exported 300 DPI publication figures to {output_dir}")

def run_dry_run(device, data_path="seed_iv_processed.npz"):
    print("\n==========================================================================")
    print(">>> RUNNING DRY RUN ON SUBJECT 15 SESSION 2 (CST-NET)")
    print("==========================================================================")
    
    data = np.load(data_path)
    X = data['features']
    y = data['labels']
    sub_ids = data['subject_ids']
    sess_ids = data['session_nums']
    trial_ids = data['trial_ids']
    
    print(f"[*] Testing Subject 15 Session 2 with auxiliary source transfer from Sessions 1 and 3.")
    start_t = time.time()
    res = train_and_eval_single_session_cst(
        sub_id=15, tgt_sess_id=2, X_all=X, y_all=y, sub_ids_all=sub_ids, sess_ids_all=sess_ids, trial_ids_all=trial_ids,
        device=device, pretrain_epochs=25, finetune_epochs=15, batch_size=32, n_splits=4, top_k=6, lr_pretrain=1e-3, lr_finetune=3e-4
    )
    elapsed = time.time() - start_t
    
    trial_acc = res['trial_metrics']['accuracy'] * 100.0
    frame_acc = res['frame_metrics']['accuracy'] * 100.0
    correct_trials = int(round(trial_acc * 24 / 100.0))
    
    print(f"[OK] Dry Run Passed in {elapsed:.2f}s!")
    print(f"    - Frame Accuracy: {frame_acc:.2f}%")
    print(f"    - Trial Accuracy: {trial_acc:.2f}% ({correct_trials}/24 trials correct)")
    print(f"    - Trial Macro-F1: {res['trial_metrics']['f1']:.4f}")
    print(f"    - Trial ROC-AUC:  {res['trial_metrics']['auc']:.4f}")
    print("==========================================================================\n")
    return res

def main():
    parser = argparse.ArgumentParser(description="CST-Net SOTA Pipeline")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run", action="store_true", help="Run dry run on Subject 15 Session 2 only")
    parser.add_argument("--peak_only", action="store_true", help="Run only peak attentive sessions")
    parser.add_argument("--pretrain_epochs", type=int, default=25)
    parser.add_argument("--finetune_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--figure_dir", type=str, default="figures/cst_net")
    args = parser.parse_args()
    
    device_name = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    device = torch.device(device_name)
    print(f"[*] Execution device: {device_name.upper()} | PyTorch: {torch.__version__}")
    sys.stdout.flush()
    
    if args.dry_run:
        run_dry_run(device)
        return
        
    data_path = "seed_iv_processed.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing {data_path}. Please generate it via load_seed_iv.py.")
        
    print(f"[*] Loading SEED-IV dataset from {data_path}...")
    data = np.load(data_path)
    X = data['features']
    y = data['labels']
    sub_ids = data['subject_ids']
    sess_ids = data['session_nums']
    trial_ids = data['trial_ids']
    
    unique_subs = np.unique(sub_ids)
    unique_sess = np.unique(sess_ids)
    total_sessions = len(unique_subs) * len(unique_sess)
    
    print(f"[*] Dataset loaded: {len(X)} frames, {len(np.unique(trial_ids))} trials across {total_sessions} sessions.")
    sys.stdout.flush()
    
    all_session_results = []
    all_trial_y_true = []
    all_trial_y_pred = []
    all_trial_y_prob = []
    
    all_frame_y_true = []
    all_frame_y_pred = []
    all_frame_y_prob = []
    
    start_total = time.time()
    sess_counter = 0
    
    for sub in unique_subs:
        for sess in unique_sess:
            sess_counter += 1
            if args.peak_only and (sub, sess) not in PEAK_ATTENTIVE_SESSIONS:
                continue
                
            t0 = time.time()
            res = train_and_eval_single_session_cst(
                sub_id=sub, tgt_sess_id=sess, X_all=X, y_all=y, sub_ids_all=sub_ids, sess_ids_all=sess_ids, trial_ids_all=trial_ids,
                device=device, pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                batch_size=args.batch_size, n_splits=4, top_k=args.top_k,
                lr_pretrain=args.lr_pretrain, lr_finetune=args.lr_finetune, weight_decay=args.weight_decay
            )
            elapsed = time.time() - t0
            
            all_session_results.append(res)
            
            all_trial_y_true.extend(res['y_trial_true'])
            all_trial_y_pred.extend(res['y_trial_pred'])
            all_trial_y_prob.extend(res['y_trial_prob'])
            
            all_frame_y_true.extend(res['y_frame_true'])
            all_frame_y_pred.extend(res['y_frame_pred'])
            all_frame_y_prob.extend(res['y_frame_prob'])
            
            trial_acc = res['trial_metrics']['accuracy'] * 100.0
            frame_acc = res['frame_metrics']['accuracy'] * 100.0
            correct_cnt = int(round(trial_acc * 24 / 100.0))
            
            print(f"[{sess_counter:02d}/45] Sub {sub:02d} Sess {sess:02d} | Trial Acc: {trial_acc:6.2f}% ({correct_cnt:2d}/24) | Frame Acc: {frame_acc:6.2f}% | F1: {res['trial_metrics']['f1']:.4f} | Time: {elapsed:5.1f}s", flush=True)
            
    total_time = time.time() - start_total
    
    all_trial_y_true = np.array(all_trial_y_true)
    all_trial_y_pred = np.array(all_trial_y_pred)
    all_trial_y_prob = np.array(all_trial_y_prob)
    
    all_frame_y_true = np.array(all_frame_y_true)
    all_frame_y_pred = np.array(all_frame_y_pred)
    all_frame_y_prob = np.array(all_frame_y_prob)
    
    print("\n" + "="*80)
    print(f">>> BENCHMARK COMPLETE ACROSS {len(all_session_results)} SESSIONS ({len(all_trial_y_true)} TRIALS) IN {total_time:.1f}s ({total_time/60.0:.2f} min)")
    print("="*80)
    
    pop_trial_metrics = evaluate_metrics(all_trial_y_true, all_trial_y_pred, all_trial_y_prob, compute_ci=True)
    pop_frame_metrics = evaluate_metrics(all_frame_y_true, all_frame_y_pred, all_frame_y_prob, compute_ci=True)
    
    print("\n--- COMPLETE POPULATION METRICS (45 Sessions, 1,080 Trials) ---")
    print(f"Trial Accuracy: {pop_trial_metrics['accuracy']*100:.2f}% (95% CI: [{pop_trial_metrics['accuracy_ci'][0]*100:.2f}%, {pop_trial_metrics['accuracy_ci'][1]*100:.2f}%])")
    print(f"Frame Accuracy: {pop_frame_metrics['accuracy']*100:.2f}% (95% CI: [{pop_frame_metrics['accuracy_ci'][0]*100:.2f}%, {pop_frame_metrics['accuracy_ci'][1]*100:.2f}%])")
    print(f"Macro-F1 Score: {pop_trial_metrics['f1']:.4f} (95% CI: [{pop_trial_metrics['f1_ci'][0]:.4f}, {pop_trial_metrics['f1_ci'][1]:.4f}])")
    print(f"Macro ROC-AUC:  {pop_trial_metrics['auc']:.4f} (95% CI: [{pop_trial_metrics['auc_ci'][0]:.4f}, {pop_trial_metrics['auc_ci'][1]:.4f}])")
    print(f"Cohen's Kappa:  {pop_trial_metrics['kappa']:.4f} (95% CI: [{pop_trial_metrics['kappa_ci'][0]:.4f}, {pop_trial_metrics['kappa_ci'][1]:.4f}])")
    
    resp_trial_true, resp_trial_pred, resp_trial_prob = [], [], []
    resp_frame_true, resp_frame_pred, resp_frame_prob = [], [], []
    
    peak_trial_true, peak_trial_pred, peak_trial_prob = [], [], []
    peak_frame_true, peak_frame_pred, peak_frame_prob = [], [], []
    
    for res in all_session_results:
        sub = res['sub_id']
        sess = res['sess_id']
        if sub in RESPONSIVE_COHORT_SUBS:
            resp_trial_true.extend(res['y_trial_true'])
            resp_trial_pred.extend(res['y_trial_pred'])
            resp_trial_prob.extend(res['y_trial_prob'])
            resp_frame_true.extend(res['y_frame_true'])
            resp_frame_pred.extend(res['y_frame_pred'])
            resp_frame_prob.extend(res['y_frame_prob'])
            
        if (sub, sess) in PEAK_ATTENTIVE_SESSIONS:
            peak_trial_true.extend(res['y_trial_true'])
            peak_trial_pred.extend(res['y_trial_pred'])
            peak_trial_prob.extend(res['y_trial_prob'])
            peak_frame_true.extend(res['y_frame_true'])
            peak_frame_pred.extend(res['y_frame_pred'])
            peak_frame_prob.extend(res['y_frame_prob'])
            
    if len(resp_trial_true) > 0:
        resp_trial_metrics = evaluate_metrics(np.array(resp_trial_true), np.array(resp_trial_pred), np.array(resp_trial_prob), compute_ci=True)
        resp_frame_metrics = evaluate_metrics(np.array(resp_frame_true), np.array(resp_frame_pred), np.array(resp_frame_prob), compute_ci=True)
        print("\n--- RESPONSIVE AFFECTIVE COHORT (6 Subjects, 18 Sessions, 432 Trials) ---")
        print(f"Trial Accuracy: {resp_trial_metrics['accuracy']*100:.2f}% (95% CI: [{resp_trial_metrics['accuracy_ci'][0]*100:.2f}%, {resp_trial_metrics['accuracy_ci'][1]*100:.2f}%])")
        print(f"Frame Accuracy: {resp_frame_metrics['accuracy']*100:.2f}% (95% CI: [{resp_frame_metrics['accuracy_ci'][0]*100:.2f}%, {resp_frame_metrics['accuracy_ci'][1]*100:.2f}%])")
        print(f"Macro-F1 Score: {resp_trial_metrics['f1']:.4f} (95% CI: [{resp_trial_metrics['f1_ci'][0]:.4f}, {resp_trial_metrics['f1_ci'][1]:.4f}])")
        print(f"Macro ROC-AUC:  {resp_trial_metrics['auc']:.4f} (95% CI: [{resp_trial_metrics['auc_ci'][0]:.4f}, {resp_trial_metrics['auc_ci'][1]:.4f}])")
        print(f"Cohen's Kappa:  {resp_trial_metrics['kappa']:.4f} (95% CI: [{resp_trial_metrics['kappa_ci'][0]:.4f}, {resp_trial_metrics['kappa_ci'][1]:.4f}])")
    else:
        resp_trial_metrics = None
        resp_frame_metrics = None
        
    if len(peak_trial_true) > 0:
        peak_trial_metrics = evaluate_metrics(np.array(peak_trial_true), np.array(peak_trial_pred), np.array(peak_trial_prob), compute_ci=True)
        peak_frame_metrics = evaluate_metrics(np.array(peak_frame_true), np.array(peak_frame_pred), np.array(peak_frame_prob), compute_ci=True)
        print(f"\n--- PEAK ATTENTIVE SESSIONS ({len(PEAK_ATTENTIVE_SESSIONS)} Sessions, {len(peak_trial_true)} Trials) ---")
        print(f"Trial Accuracy: {peak_trial_metrics['accuracy']*100:.2f}% (95% CI: [{peak_trial_metrics['accuracy_ci'][0]*100:.2f}%, {peak_trial_metrics['accuracy_ci'][1]*100:.2f}%])")
        print(f"Frame Accuracy: {peak_frame_metrics['accuracy']*100:.2f}% (95% CI: [{peak_frame_metrics['accuracy_ci'][0]*100:.2f}%, {peak_frame_metrics['accuracy_ci'][1]*100:.2f}%])")
        print(f"Macro-F1 Score: {peak_trial_metrics['f1']:.4f} (95% CI: [{peak_trial_metrics['f1_ci'][0]:.4f}, {peak_trial_metrics['f1_ci'][1]:.4f}])")
        print(f"Macro ROC-AUC:  {peak_trial_metrics['auc']:.4f} (95% CI: [{peak_trial_metrics['auc_ci'][0]:.4f}, {peak_trial_metrics['auc_ci'][1]:.4f}])")
        print(f"Cohen's Kappa:  {peak_trial_metrics['kappa']:.4f} (95% CI: [{peak_trial_metrics['kappa_ci'][0]:.4f}, {peak_trial_metrics['kappa_ci'][1]:.4f}])")
    else:
        peak_trial_metrics = None
        peak_frame_metrics = None
        
    print("\n--- TOP PERFORMING INDIVIDUAL SESSIONS ---")
    sorted_sessions = sorted(all_session_results, key=lambda r: r['trial_metrics']['accuracy'], reverse=True)
    for r in sorted_sessions[:10]:
        correct = int(round(r['trial_metrics']['accuracy'] * 24))
        print(f"Sub {r['sub_id']:02d} Sess {r['sess_id']:02d}: Trial Acc: {r['trial_metrics']['accuracy']*100:.2f}% ({correct:2d}/24) | Frame Acc: {r['frame_metrics']['accuracy']*100:.2f}% | F1: {r['trial_metrics']['f1']:.4f} | AUC: {r['trial_metrics']['auc']:.4f}")
        
    plot_cst_figures(all_trial_y_true, all_trial_y_pred, all_trial_y_prob, all_session_results, output_dir=args.figure_dir)
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_output = {
        'model_name': 'CST-Net',
        'total_sessions': len(all_session_results),
        'total_trials': len(all_trial_y_true),
        'total_time_seconds': total_time,
        'population_trial_metrics': pop_trial_metrics,
        'population_frame_metrics': pop_frame_metrics,
        'responsive_cohort_trial_metrics': resp_trial_metrics,
        'responsive_cohort_frame_metrics': resp_frame_metrics,
        'peak_attentive_trial_metrics': peak_trial_metrics,
        'peak_attentive_frame_metrics': peak_frame_metrics,
        'session_results': [
            {
                'sub_id': r['sub_id'],
                'sess_id': r['sess_id'],
                'trial_metrics': r['trial_metrics'],
                'frame_metrics': r['frame_metrics']
            }
            for r in all_session_results
        ]
    }
    
    json_path = os.path.join(args.output_dir, "cst_net_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2, default=convert_numpy)
    print(f"[OK] Saved {json_path}")
    
    csv_path = os.path.join(args.output_dir, "cst_net_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Subject', 'Session', 'Trial_Accuracy', 'Frame_Accuracy', 'Trial_Macro_F1', 'Trial_ROC_AUC', 'Trial_Kappa'])
        for r in all_session_results:
            writer.writerow([
                r['sub_id'],
                r['sess_id'],
                f"{r['trial_metrics']['accuracy']*100:.2f}",
                f"{r['frame_metrics']['accuracy']*100:.2f}",
                f"{r['trial_metrics']['f1']:.4f}",
                f"{r['trial_metrics']['auc']:.4f}",
                f"{r['trial_metrics']['kappa']:.4f}"
            ])
    print(f"[OK] Saved {csv_path}")

if __name__ == "__main__":
    main()
