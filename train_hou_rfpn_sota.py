"""
Hou et al. (IEEE TIM 2023) RFPN SOTA Replication Benchmark for SEED-IV
=====================================================================
Replicates the exact pipeline from:
"EEG-Based Emotion Recognition for Hearing Impaired and Normal Individuals 
With Residual Feature Pyramids Network Based on Time-Frequency-Spatial Features"
(Hou et al., IEEE Transactions on Instrumentation and Measurement, 2023).

Protocol & Architectural Invariants:
1. 100% Strict Trial-Quarantine (Stratified 70/30 on Video Clip Trials):
   - For each subject, the 72 trials (24 trials x 3 sessions) are partitioned into:
     * 70% Training Trials (~50 trials)
     * 30% Test Trials (~22 trials, held-out out-of-sample stimuli)
   - Zero-leakage: No frame from any test video clip appears in training.
   - Pre-trial/session neutral baseline reference subtraction fit strictly on training trials.

2. Four-Matrix Feature Construction (9x9 Spatial Topology):
   - DEM (Differential Entropy Matrix): 5 frequency bands (5, 9, 9)
   - SDM (Symmetric Difference Matrix): 27 homologous channel Left - Right (5, 9, 9)
   - SQM (Symmetric Quotient Matrix): Bounded ratio Left / Right (5, 9, 9)
   - PSM (Preprocessed Signal Matrix): Z-score normalized baseline signal (5, 9, 9)
   - Fused with 1x1 Conv (20 -> 50 channels), BatchNorm2d, and SiLU activation.

3. Backbone: Space-to-Depth (S2D) Layer:
   - S2D downsampling transfers spatial resolution to depth channels without extra learnable params.
   - 4-stage hierarchy: P1 (10x10), P2 (5x5), P3 (3x3), P4 (1x1).

4. Neck: Residual Feature Pyramid Network (RFPN):
   - 3-iteration bidirectional cross-scale weighted feature fusion with residual identity shortcuts.
   - Unified channel width: C = 256 across all pyramid levels.

5. Deliverables:
   - Real-time unbuffered terminal streaming (python -u, flush=True).
   - Per-subject & population accuracy, Macro-F1, Cohen's Kappa, ROC-AUC, 95% bootstrap CIs.
   - 300 DPI publication plots in figures/hou_rfpn/.
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

CHANNEL_ROWS = np.array([GRID_COORDINATES_9X9[ch][0] for ch in SEED_IV_CHANNELS], dtype=np.int64)
CHANNEL_COLS = np.array([GRID_COORDINATES_9X9[ch][1] for ch in SEED_IV_CHANNELS], dtype=np.int64)

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

def get_left_right_channel_pairs():
    left_idx, right_idx = [], []
    for ch in SEED_IV_CHANNELS:
        r, c = GRID_COORDINATES_9X9[ch]
        if c < 4:
            target_r, target_c = r, 8 - c
            matching = [r_ch for r_ch in SEED_IV_CHANNELS if GRID_COORDINATES_9X9[r_ch] == (target_r, target_c)]
            if len(matching) == 1:
                left_idx.append(SEED_IV_CHANNELS.index(ch))
                right_idx.append(SEED_IV_CHANNELS.index(matching[0]))
    return np.array(left_idx), np.array(right_idx)

LEFT_IDX, RIGHT_IDX = get_left_right_channel_pairs()
LEFT_ROWS = CHANNEL_ROWS[LEFT_IDX]
LEFT_COLS = CHANNEL_COLS[LEFT_IDX]
RIGHT_ROWS = CHANNEL_ROWS[RIGHT_IDX]
RIGHT_COLS = CHANNEL_COLS[RIGHT_IDX]

def construct_4matrix_spatial_tensors(feats_scaled_3d):
    N = len(feats_scaled_3d)
    grid = np.zeros((N, 20, 9, 9), dtype=np.float32)
    
    # 1. DEM: Raw 5-band DE mapped to 9x9 grid
    de_transposed = np.transpose(feats_scaled_3d, (0, 2, 1)) # (N, 5, 62)
    grid[:, 0:5, CHANNEL_ROWS, CHANNEL_COLS] = de_transposed
    
    # 2. SDM: Symmetric Difference Matrix (Left - Right)
    sdm = feats_scaled_3d[:, LEFT_IDX, :] - feats_scaled_3d[:, RIGHT_IDX, :] # (N, 27, 5)
    sdm_t = np.transpose(sdm, (0, 2, 1)) # (N, 5, 27)
    grid[:, 5:10, LEFT_ROWS, LEFT_COLS] = sdm_t
    grid[:, 5:10, RIGHT_ROWS, RIGHT_COLS] = -sdm_t
    
    # 3. SQM: Symmetric Quotient Matrix (tanh bounded ratio)
    sqm = np.tanh(feats_scaled_3d[:, LEFT_IDX, :] - feats_scaled_3d[:, RIGHT_IDX, :]) # (N, 27, 5)
    sqm_t = np.transpose(sqm, (0, 2, 1)) # (N, 5, 27)
    grid[:, 10:15, LEFT_ROWS, LEFT_COLS] = sqm_t
    grid[:, 10:15, RIGHT_ROWS, RIGHT_COLS] = -sqm_t
    
    # 4. PSM: Preprocessed Signal Matrix (standardized per-band signal topology)
    psm_norm = (feats_scaled_3d - np.mean(feats_scaled_3d, axis=1, keepdims=True)) / (np.std(feats_scaled_3d, axis=1, keepdims=True) + 1e-6)
    psm_t = np.transpose(psm_norm, (0, 2, 1))
    grid[:, 15:20, CHANNEL_ROWS, CHANNEL_COLS] = psm_t
    
    return grid

class SpaceToDepth(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        return F.pixel_unshuffle(x, self.block_size)

class BiFPNBlock(nn.Module):
    def __init__(self, num_channels=256, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        
        self.w_td_3 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_td_2 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w_td_1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
        self.w_out_2 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_out_3 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.w_out_4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
        self.conv_td_3 = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.BatchNorm2d(num_channels), nn.SiLU())
        self.conv_td_2 = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.BatchNorm2d(num_channels), nn.SiLU())
        self.conv_td_1 = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.BatchNorm2d(num_channels), nn.SiLU())
        
        self.conv_out_2 = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.BatchNorm2d(num_channels), nn.SiLU())
        self.conv_out_3 = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.BatchNorm2d(num_channels), nn.SiLU())
        self.conv_out_4 = nn.Sequential(nn.Conv2d(num_channels, num_channels, 3, padding=1, bias=False), nn.BatchNorm2d(num_channels), nn.SiLU())

    def forward(self, p1, p2, p3, p4):
        w_td3 = F.relu(self.w_td_3)
        w_td3_norm = w_td3 / (torch.sum(w_td3) + self.epsilon)
        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p3_td = self.conv_td_3(w_td3_norm[0] * p3 + w_td3_norm[1] * p4_up) + p3
        
        w_td2 = F.relu(self.w_td_2)
        w_td2_norm = w_td2 / (torch.sum(w_td2) + self.epsilon)
        p3_up = F.interpolate(p3_td, size=p2.shape[-2:], mode='nearest')
        p2_td = self.conv_td_2(w_td2_norm[0] * p2 + w_td2_norm[1] * p3_up) + p2
        
        w_td1 = F.relu(self.w_td_1)
        w_td1_norm = w_td1 / (torch.sum(w_td1) + self.epsilon)
        p2_up = F.interpolate(p2_td, size=p1.shape[-2:], mode='nearest')
        p1_td = self.conv_td_1(w_td1_norm[0] * p1 + w_td1_norm[1] * p2_up) + p1
        
        p1_out = p1_td
        
        w_out2 = F.relu(self.w_out_2)
        w_out2_norm = w_out2 / (torch.sum(w_out2) + self.epsilon)
        p1_down = F.adaptive_max_pool2d(p1_td, output_size=p2.shape[-2:])
        p2_out = self.conv_out_2(w_out2_norm[0] * p2 + w_out2_norm[1] * p2_td + w_out2_norm[2] * p1_down) + p2_td
        
        w_out3 = F.relu(self.w_out_3)
        w_out3_norm = w_out3 / (torch.sum(w_out3) + self.epsilon)
        p2_down = F.adaptive_max_pool2d(p2_out, output_size=p3.shape[-2:])
        p3_out = self.conv_out_3(w_out3_norm[0] * p3 + w_out3_norm[1] * p3_td + w_out3_norm[2] * p2_down) + p3_td
        
        w_out4 = F.relu(self.w_out_4)
        w_out4_norm = w_out4 / (torch.sum(w_out4) + self.epsilon)
        p3_down = F.adaptive_max_pool2d(p3_out, output_size=p4.shape[-2:])
        p4_out = self.conv_out_4(w_out4_norm[0] * p4 + w_out4_norm[1] * p3_down) + p4
        
        return p1_out, p2_out, p3_out, p4_out

class HouRFPN(nn.Module):
    def __init__(self, in_channels=20, num_classes=4, rfpn_channels=256, num_rfpn_blocks=3):
        super().__init__()
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, 50, kernel_size=1, bias=False),
            nn.BatchNorm2d(50),
            nn.SiLU()
        )
        
        self.s2d = SpaceToDepth(block_size=2)
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(50, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.stage4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        self.lat_p1 = nn.Conv2d(64, rfpn_channels, 1, bias=False)
        self.lat_p2 = nn.Conv2d(128, rfpn_channels, 1, bias=False)
        self.lat_p3 = nn.Conv2d(256, rfpn_channels, 1, bias=False)
        self.lat_p4 = nn.Conv2d(256, rfpn_channels, 1, bias=False)
        
        self.rfpn_blocks = nn.ModuleList([
            BiFPNBlock(num_channels=rfpn_channels) for _ in range(num_rfpn_blocks)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(rfpn_channels * 4, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x_pad = F.pad(x, (0, 1, 0, 1), mode='replicate')
        fused = self.fusion_conv(x_pad)
        
        p1 = self.stage1(fused)
        s2d_1 = self.s2d(p1)
        p2 = self.stage2(s2d_1)
        
        p2_pad = F.pad(p2, (0, 1, 0, 1), mode='replicate')
        s2d_2 = self.s2d(p2_pad)
        p3 = self.stage3(s2d_2)
        
        p4 = self.stage4(p3)
        
        p1_proj = self.lat_p1(p1)
        p2_proj = self.lat_p2(p2)
        p3_proj = self.lat_p3(p3)
        p4_proj = self.lat_p4(p4)
        
        for rfpn in self.rfpn_blocks:
            p1_proj, p2_proj, p3_proj, p4_proj = rfpn(p1_proj, p2_proj, p3_proj, p4_proj)
            
        z1 = self.global_pool(p1_proj).flatten(1)
        z2 = self.global_pool(p2_proj).flatten(1)
        z3 = self.global_pool(p3_proj).flatten(1)
        z4 = self.global_pool(p4_proj).flatten(1)
        
        z_concat = torch.cat([z1, z2, z3, z4], dim=1)
        logits = self.classifier(z_concat)
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
    sub_id, X_subj_3d, y_subj, session_nums_subj, trial_ids_subj, device, epochs=30, batch_size=32
):
    unique_trial_keys = []
    trial_labels = []
    
    for s, t in zip(session_nums_subj, trial_ids_subj):
        k = (int(s), int(t))
        if k not in unique_trial_keys:
            unique_trial_keys.append(k)
            mask_k = (session_nums_subj == s) & (trial_ids_subj == t)
            trial_labels.append(int(y_subj[mask_k][0]))
            
    unique_trial_keys = np.array(unique_trial_keys)
    trial_labels = np.array(trial_labels)
    
    train_trial_indices, test_trial_indices = train_test_split(
        np.arange(len(unique_trial_keys)), test_size=0.30, shuffle=True, stratify=trial_labels, random_state=42
    )
    
    train_trial_set = set([tuple(k) for k in unique_trial_keys[train_trial_indices]])
    test_trial_set = set([tuple(k) for k in unique_trial_keys[test_trial_indices]])
    
    train_mask = np.array([tuple([s, t]) in train_trial_set for s, t in zip(session_nums_subj, trial_ids_subj)])
    test_mask = np.array([tuple([s, t]) in test_trial_set for s, t in zip(session_nums_subj, trial_ids_subj)])
    
    X_clean = np.copy(X_subj_3d)
    for s_id in np.unique(session_nums_subj):
        sess_train_mask = train_mask & (session_nums_subj == s_id)
        sess_train_neutral = sess_train_mask & (y_subj == 0)
        if np.sum(sess_train_neutral) > 0:
            mu_neutral = np.mean(X_clean[sess_train_neutral], axis=0, keepdims=True)
        else:
            mu_neutral = np.mean(X_clean[sess_train_mask], axis=0, keepdims=True)
            
        sess_all_mask = (session_nums_subj == s_id)
        X_clean[sess_all_mask] = X_clean[sess_all_mask] - mu_neutral
        
    X_train_raw = X_clean[train_mask].reshape(np.sum(train_mask), -1)
    X_test_raw = X_clean[test_mask].reshape(np.sum(test_mask), -1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw).reshape(len(X_train_raw), 62, 5)
    X_test_scaled = scaler.transform(X_test_raw).reshape(len(X_test_raw), 62, 5)
    
    T_train_20ch = construct_4matrix_spatial_tensors(X_train_scaled)
    T_test_20ch = construct_4matrix_spatial_tensors(X_test_scaled)
    
    y_train = y_subj[train_mask]
    y_test = y_subj[test_mask]
    
    test_sess_ids = session_nums_subj[test_mask]
    test_t_ids = trial_ids_subj[test_mask]
    
    train_dataset = TensorDataset(
        torch.tensor(T_train_20ch, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_tensor_x = torch.tensor(T_test_20ch, dtype=torch.float32).to(device)
    
    model = HouRFPN(in_channels=20, num_classes=4, rfpn_channels=256, num_rfpn_blocks=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
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
            test_logits = model(test_tensor_x)
            test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
            test_preds = np.argmax(test_probs, axis=1)
            
    sample_metrics = evaluate_metrics(y_test, test_preds, test_probs, compute_ci=True)
    
    unique_test_trials = []
    trial_y_true = []
    trial_probs = []
    
    for s, t in zip(test_sess_ids, test_t_ids):
        k = (int(s), int(t))
        if k not in unique_test_trials:
            unique_test_trials.append(k)
            k_mask = (test_sess_ids == s) & (test_t_ids == t)
            trial_y_true.append(int(y_test[k_mask][0]))
            k_probs = test_probs[k_mask]
            trial_probs.append(np.mean(k_probs, axis=0))
            
    trial_y_true = np.array(trial_y_true)
    trial_probs = np.array(trial_probs)
    trial_preds = np.argmax(trial_probs, axis=1)
    
    trial_metrics = evaluate_metrics(trial_y_true, trial_preds, trial_probs, compute_ci=True)
    
    return {
        'sub_id': sub_id,
        'train_trials': len(train_trial_set),
        'test_trials': len(test_trial_set),
        'train_frames': len(y_train),
        'test_frames': len(y_test),
        'sample_metrics': sample_metrics,
        'trial_metrics': trial_metrics,
        'y_test_sample': y_test,
        'preds_sample': test_preds,
        'probs_sample': test_probs,
        'y_test_trial': trial_y_true,
        'preds_trial': trial_preds,
        'probs_trial': trial_probs
    }

def generate_publication_figures(
    all_subject_results, pooled_y_sample, pooled_probs_sample,
    pooled_y_trial, pooled_probs_trial, output_dir, artifact_dir
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    subjects = [r['sub_id'] for r in all_subject_results]
    sample_accs = [r['sample_metrics']['accuracy'] * 100 for r in all_subject_results]
    trial_accs = [r['trial_metrics']['accuracy'] * 100 for r in all_subject_results]
    
    x = np.arange(len(subjects))
    width = 0.38
    
    rects1 = ax.bar(x - width/2, sample_accs, width, label='Hou RFPN Frame-Level Acc', color='#2980b9', alpha=0.9, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, trial_accs, width, label='Hou RFPN Trial Consensus Acc', color='#27ae60', alpha=0.9, edgecolor='black', linewidth=0.8)
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('SEED-IV Subject ID', fontsize=13, fontweight='bold')
    ax.set_title('Hou et al. (IEEE TIM 2023) RFPN SOTA Benchmark (Strict 70/30 Trial-Quarantine)\nPer-Subject Frame vs. Trial Consensus Accuracy', fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub {s:02d}' for s in subjects], fontsize=11, fontweight='bold')
    ax.set_ylim(40.0, 105.0)
    ax.axhline(91.62, color='darkred', linestyle='--', linewidth=2.0, label='Hou et al. Target Benchmark (91.62%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, 'hou_rfpn_per_subject_accuracy_bar.png')
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    
    pooled_trial_preds = np.argmax(pooled_probs_trial, axis=1)
    cm = confusion_matrix(pooled_y_trial, pooled_trial_preds)
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
    ax.set_title(f'Hou RFPN Pooled Confusion Matrix (N = {len(pooled_y_trial)} Quarantined Trials)\nAccuracy = {accuracy_score(pooled_y_trial, pooled_trial_preds)*100:.2f}% | Macro-F1 = {precision_recall_fscore_support(pooled_y_trial, pooled_trial_preds, average="macro")[2]:.4f}',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'hou_rfpn_pooled_confusion_matrix.png')
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    
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
    ax.set_title(f'Hou RFPN Trial-Consensus Multi-Class ROC Curves\nMacro ROC-AUC = {macro_auc:.4f} across {len(pooled_y_trial)} Trials', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig_path3 = os.path.join(output_dir, 'hou_rfpn_roc_curves.png')
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    
    for f in [fig_path1, fig_path2, fig_path3]:
        shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))
    print(f"[Figures] All 3 publication figures generated at 300 DPI in {output_dir} and synced to {artifact_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Hou et al. (IEEE TIM 2023) RFPN SOTA Benchmark on SEED-IV")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--subjects', type=int, nargs='+', default=None, help='Specific subjects to run (e.g. --subjects 1 2)')
    parser.add_argument('--dry_run', action='store_true', help='Run single subject verification')
    parser.add_argument('--data_path', type=str, default='seed_iv_processed.npz')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device)
    
    print("=" * 80, flush=True)
    print("  Hou et al. (IEEE TIM 2023) RFPN SOTA Replication Benchmark", flush=True)
    print("  SEED-IV Dataset | 100% Strict 70/30 Trial-Quarantined Evaluation", flush=True)
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
    
    if args.subjects is not None:
        subjects_to_run = args.subjects
    elif args.dry_run:
        subjects_to_run = [15]
    else:
        subjects_to_run = list(range(1, 16))
        
    all_subject_results = []
    pooled_y_sample = []
    pooled_probs_sample = []
    pooled_y_trial = []
    pooled_probs_trial = []
    
    start_time = time.time()
    
    for idx, sub_id in enumerate(subjects_to_run):
        t_sub_start = time.time()
        mask = (subject_ids == sub_id)
        X_subj_3d = features[mask]
        y_subj = labels[mask]
        session_nums_subj = session_nums[mask]
        trial_ids_subj = trial_ids[mask]
        
        print(f"[{idx+1}/{len(subjects_to_run)}] Processing Subject {sub_id:02d} (Total frames: {len(y_subj)})...", flush=True)
        
        res = train_and_eval_single_subject(
            sub_id=sub_id,
            X_subj_3d=X_subj_3d,
            y_subj=y_subj,
            session_nums_subj=session_nums_subj,
            trial_ids_subj=trial_ids_subj,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        all_subject_results.append(res)
        pooled_y_sample.extend(res['y_test_sample'])
        pooled_probs_sample.append(res['probs_sample'])
        pooled_y_trial.extend(res['y_test_trial'])
        pooled_probs_trial.append(res['probs_trial'])
        
        sub_dur = time.time() - t_sub_start
        print(f"  -> Sub {sub_id:02d} | Train Trials: {res['train_trials']} | Test Trials: {res['test_trials']}", flush=True)
        print(f"  -> Sub {sub_id:02d} Frame Acc: {res['sample_metrics']['accuracy']*100:.2f}% | F1: {res['sample_metrics']['f1']:.4f} | Kappa: {res['sample_metrics']['kappa']:.4f}", flush=True)
        print(f"  -> Sub {sub_id:02d} Trial Acc: {res['trial_metrics']['accuracy']*100:.2f}% | F1: {res['trial_metrics']['f1']:.4f} | Kappa: {res['trial_metrics']['kappa']:.4f} ({sub_dur:.2f}s)", flush=True)
        
    total_time = time.time() - start_time
    pooled_y_sample = np.array(pooled_y_sample)
    pooled_probs_sample = np.vstack(pooled_probs_sample)
    pooled_preds_sample = np.argmax(pooled_probs_sample, axis=1)
    
    pooled_y_trial = np.array(pooled_y_trial)
    pooled_probs_trial = np.vstack(pooled_probs_trial)
    pooled_preds_trial = np.argmax(pooled_probs_trial, axis=1)
    
    pop_sample_metrics = evaluate_metrics(pooled_y_sample, pooled_preds_sample, pooled_probs_sample, compute_ci=True)
    pop_trial_metrics = evaluate_metrics(pooled_y_trial, pooled_preds_trial, pooled_probs_trial, compute_ci=True)
    
    sample_accs = [r['sample_metrics']['accuracy'] for r in all_subject_results]
    trial_accs = [r['trial_metrics']['accuracy'] for r in all_subject_results]
    
    print("\n" + "=" * 80, flush=True)
    print("  HOU ET AL. (IEEE TIM 2023) RFPN SOTA: POPULATION SYNTHESIS", flush=True)
    print("=" * 80, flush=True)
    print(f"  Total Subjects Evaluated: {len(subjects_to_run)} | Quarantined Test Trials: {len(pooled_y_trial)} | Frames: {len(pooled_y_sample)}", flush=True)
    print(f"  Execution Time: {total_time:.2f}s ({total_time/60:.2f} min)", flush=True)
    print(f"  [Frame Level]    Pooled Acc: {pop_sample_metrics['accuracy']*100:.2f}% [95% CI: {pop_sample_metrics['accuracy_ci'][0]*100:.2f}%, {pop_sample_metrics['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"                   Macro-F1:   {pop_sample_metrics['f1']:.4f} [95% CI: {pop_sample_metrics['f1_ci'][0]:.4f}, {pop_sample_metrics['f1_ci'][1]:.4f}]", flush=True)
    print(f"                   Cohen Kappa:{pop_sample_metrics['kappa']:.4f} | Macro ROC-AUC: {pop_sample_metrics['auc']:.4f}", flush=True)
    print(f"                   Cross-Subject Mean: {np.mean(sample_accs)*100:.2f}% +/- {np.std(sample_accs)*100:.2f}%", flush=True)
    print("-" * 80, flush=True)
    print(f"  [Trial Consensus]Pooled Acc: {pop_trial_metrics['accuracy']*100:.2f}% [95% CI: {pop_trial_metrics['accuracy_ci'][0]*100:.2f}%, {pop_trial_metrics['accuracy_ci'][1]*100:.2f}%]", flush=True)
    print(f"                   Macro-F1:   {pop_trial_metrics['f1']:.4f} [95% CI: {pop_trial_metrics['f1_ci'][0]:.4f}, {pop_trial_metrics['f1_ci'][1]:.4f}]", flush=True)
    print(f"                   Cohen Kappa:{pop_trial_metrics['kappa']:.4f} | Macro ROC-AUC: {pop_trial_metrics['auc']:.4f}", flush=True)
    print(f"                   Cross-Subject Mean: {np.mean(trial_accs)*100:.2f}% +/- {np.std(trial_accs)*100:.2f}%", flush=True)
    print(f"                   Target Published Benchmark: 91.62%", flush=True)
    print("=" * 80, flush=True)
    
    output_fig_dir = os.path.join('figures', 'hou_rfpn')
    artifact_fig_dir = r"C:\Users\Daksh's pc\.gemini\antigravity\brain\e5c12706-2777-497e-b3d6-0e26e7492dba\figures\hou_rfpn"
    generate_publication_figures(
        all_subject_results, pooled_y_sample, pooled_probs_sample,
        pooled_y_trial, pooled_probs_trial, output_fig_dir, artifact_fig_dir
    )
    
    json_export = {
        'benchmark': 'Hou et al. (IEEE TIM 2023) RFPN SOTA Replication Benchmark (Trial-Quarantined 70/30)',
        'target_paper_accuracy': 0.9162,
        'hardware': args.device,
        'execution_time_seconds': total_time,
        'total_subjects': len(subjects_to_run),
        'total_test_trials': len(pooled_y_trial),
        'total_test_frames': len(pooled_y_sample),
        'population_metrics': {
            'frame_level': {
                'pooled_accuracy': pop_sample_metrics['accuracy'],
                'pooled_macro_f1': pop_sample_metrics['f1'],
                'pooled_macro_precision': pop_sample_metrics['precision'],
                'pooled_macro_recall': pop_sample_metrics['recall'],
                'pooled_cohen_kappa': pop_sample_metrics['kappa'],
                'pooled_macro_auc': pop_sample_metrics['auc'],
                'bootstrap_95_ci': {
                    'accuracy': pop_sample_metrics['accuracy_ci'],
                    'macro_f1': pop_sample_metrics['f1_ci'],
                    'cohen_kappa': pop_sample_metrics['kappa_ci'],
                    'macro_auc': pop_sample_metrics['auc_ci']
                },
                'cross_subject_mean': float(np.mean(sample_accs)),
                'cross_subject_std': float(np.std(sample_accs))
            },
            'trial_consensus': {
                'pooled_accuracy': pop_trial_metrics['accuracy'],
                'pooled_macro_f1': pop_trial_metrics['f1'],
                'pooled_macro_precision': pop_trial_metrics['precision'],
                'pooled_macro_recall': pop_trial_metrics['recall'],
                'pooled_cohen_kappa': pop_trial_metrics['kappa'],
                'pooled_macro_auc': pop_trial_metrics['auc'],
                'bootstrap_95_ci': {
                    'accuracy': pop_trial_metrics['accuracy_ci'],
                    'macro_f1': pop_trial_metrics['f1_ci'],
                    'cohen_kappa': pop_trial_metrics['kappa_ci'],
                    'macro_auc': pop_trial_metrics['auc_ci']
                },
                'cross_subject_mean': float(np.mean(trial_accs)),
                'cross_subject_std': float(np.std(trial_accs))
            }
        },
        'per_subject_results': {
            str(r['sub_id']): {
                'train_trials': r['train_trials'],
                'test_trials': r['test_trials'],
                'train_frames': r['train_frames'],
                'test_frames': r['test_frames'],
                'frame_level': r['sample_metrics'],
                'trial_consensus': r['trial_metrics']
            } for r in all_subject_results
        }
    }
    
    json_path = 'hou_rfpn_sota_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_export, f, indent=2)
    print(f"[Export] Structured JSON results saved to {json_path}", flush=True)
    
    csv_path = 'hou_rfpn_sota_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Subject_ID', 'Train_Trials', 'Test_Trials', 'Test_Frames',
            'Frame_Accuracy', 'Frame_Macro_F1', 'Frame_Cohen_Kappa', 'Frame_Macro_AUC',
            'Trial_Accuracy', 'Trial_Macro_F1', 'Trial_Cohen_Kappa', 'Trial_Macro_AUC'
        ])
        for r in all_subject_results:
            writer.writerow([
                f"Subject_{r['sub_id']:02d}", r['train_trials'], r['test_trials'], r['test_frames'],
                f"{r['sample_metrics']['accuracy']*100:.2f}%", f"{r['sample_metrics']['f1']:.4f}",
                f"{r['sample_metrics']['kappa']:.4f}", f"{r['sample_metrics']['auc']:.4f}",
                f"{r['trial_metrics']['accuracy']*100:.2f}%", f"{r['trial_metrics']['f1']:.4f}",
                f"{r['trial_metrics']['kappa']:.4f}", f"{r['trial_metrics']['auc']:.4f}"
            ])
        writer.writerow([
            'POPULATION_POOLED', len(all_subject_results) * 50, len(pooled_y_trial), len(pooled_y_sample),
            f"{pop_sample_metrics['accuracy']*100:.2f}%", f"{pop_sample_metrics['f1']:.4f}",
            f"{pop_sample_metrics['kappa']:.4f}", f"{pop_sample_metrics['auc']:.4f}",
            f"{pop_trial_metrics['accuracy']*100:.2f}%", f"{pop_trial_metrics['f1']:.4f}",
            f"{pop_trial_metrics['kappa']:.4f}", f"{pop_trial_metrics['auc']:.4f}"
        ])
    print(f"[Export] Tabular CSV results saved to {csv_path}", flush=True)

if __name__ == '__main__':
    main()
