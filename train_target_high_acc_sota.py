"""
Session-Level Trial-Quarantined SOTA RFPN Benchmark for SEED-IV
==============================================================
Evaluates multi-scale 4-Matrix Spatio-Temporal RFPN architecture across all 45 sessions
(15 subjects x 3 sessions = 1,080 trials, 37,575 frames) on SEED-IV with mathematically
guaranteed zero-leakage trial quarantine and Soft Log-Odds consensus aggregation.

Protocol & Zero-Leakage Invariants:
1. Session-Level 4-Fold Stratified Trial Cross-Validation:
   - Each session contains 24 trials (6 trials x 4 classes).
   - StratifiedKFold (n_splits=4) partitions the 24 trials per session into 18 train trials (75%)
     and 6 test trials (25%) per fold.
   - Total evaluation across 4 folds evaluates all 24 trials per session (1,080 trials population-wide).
   - HARD ASSERTION: assert len(set(train_trial_ids).intersection(set(test_trial_ids))) == 0.

2. Inductive Zero-Leakage Preprocessing:
   - StandardScaler fitted strictly on training trials: scaler.fit(X_train).
   - Baseline neutral reference subtraction computed strictly from training neutral trials:
     mu_neutral = X_train[y_train == 0].mean(axis=0)

3. Four-Matrix Spatial Feature Representation (9x9 Topology):
   - DEM: Differential Entropy across 5 frequency bands (Delta, Theta, Alpha, Beta, Gamma)
   - SDM: Symmetric Difference Matrix across 27 homologous channel pairs (Left - Right)
   - SQM: Symmetric Quotient Matrix across 27 homologous channel pairs (tanh bounded ratio)
   - PSM: Standardized preprocessed signal matrix across 10-20 layout
   - Fused via 1x1 Conv2d (20 -> 64 channels), BatchNorm2d, and SiLU.

4. S2D Backbone + 3-Iteration Bidirectional RFPN Neck:
   - Space-to-Depth downsampling preserving fine-grained spatial representations.
   - 3 stacked iterations of bidirectional weighted cross-scale fusion (C=256) with identity shortcuts.
   - Multi-scale pooling classifier head.

5. Soft Log-Odds Trial-Consensus Aggregation:
   - s_trial(c) = sum_{w in Trial} ln(max(P(y_w = c | x_w), 1e-7))
   - P_trial(c) = Softmax(s_trial(c))
   - y_pred_trial = argmax_c P_trial(c)
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
    
    # 1. DEM: 5-band DE mapped to 9x9 grid
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

class TargetRFPN(nn.Module):
    def __init__(self, in_channels=20, num_classes=4, rfpn_channels=256, num_rfpn_blocks=3):
        super().__init__()
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        self.s2d = SpaceToDepth(block_size=2)
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
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
            nn.Dropout(0.2),
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

def train_and_eval_single_session(
    sub_id, sess_id, X_sess_3d, y_sess, trial_ids_sess, device, epochs=30, batch_size=32, n_splits=4
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
    
    session_y_frame_true = []
    session_y_frame_pred = []
    session_y_frame_prob = []
    
    session_y_trial_true = []
    session_y_trial_pred = []
    session_y_trial_prob = []
    session_test_trial_ids = []
    
    for fold, (train_trial_idx, test_trial_idx) in enumerate(skf.split(unique_trials, trial_labels)):
        train_trials_fold = unique_trials[train_trial_idx]
        test_trials_fold = unique_trials[test_trial_idx]
        
        # 1. HARD LEAKAGE PREVENTION ASSERTION
        leakage_intersection = set(train_trials_fold).intersection(set(test_trials_fold))
        assert len(leakage_intersection) == 0, f"FATAL: Cross-trial data leakage detected in Sub {sub_id} Sess {sess_id} Fold {fold}: {leakage_intersection}"
        
        train_frame_mask = np.isin(trial_ids_sess, train_trials_fold)
        test_frame_mask = np.isin(trial_ids_sess, test_trials_fold)
        
        X_train_raw = np.copy(X_sess_3d[train_frame_mask])
        y_train_fold = y_sess[train_frame_mask]
        
        X_test_raw = np.copy(X_sess_3d[test_frame_mask])
        y_test_fold = y_sess[test_frame_mask]
        test_t_ids_fold = trial_ids_sess[test_frame_mask]
        
        # 2. BASELINE SUBTRACTION QUARANTINE (Calculated strictly from train neutral trials)
        train_neutral_mask = (y_train_fold == 0)
        if np.sum(train_neutral_mask) > 0:
            mu_neutral = np.mean(X_train_raw[train_neutral_mask], axis=0, keepdims=True)
        else:
            mu_neutral = np.mean(X_train_raw, axis=0, keepdims=True)
            
        X_train_raw = X_train_raw - mu_neutral
        X_test_raw = X_test_raw - mu_neutral
        
        # 3. INDUCTIVE STANDARDIZATION ASSERTION (Fitted strictly on train trials)
        X_train_flat = X_train_raw.reshape(len(X_train_raw), -1)
        X_test_flat = X_test_raw.reshape(len(X_test_raw), -1)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat).reshape(len(X_train_raw), 62, 5)
        X_test_scaled = scaler.transform(X_test_flat).reshape(len(X_test_raw), 62, 5)
        
        # 4. FOUR-MATRIX SPATIAL TENSOR CONSTRUCTION
        T_train = construct_4matrix_spatial_tensors(X_train_scaled)
        T_test = construct_4matrix_spatial_tensors(X_test_scaled)
        
        train_dataset = TensorDataset(
            torch.tensor(T_train, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_tensor_x = torch.tensor(T_test, dtype=torch.float32).to(device)
        
        model = TargetRFPN(in_channels=20, num_classes=4, rfpn_channels=256, num_rfpn_blocks=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
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
                
        session_y_frame_true.extend(y_test_fold)
        session_y_frame_pred.extend(test_preds)
        session_y_frame_prob.extend(test_probs)
        
        for t_val in test_trials_fold:
            k_mask = (test_t_ids_fold == t_val)
            trial_true = int(y_test_fold[k_mask][0])
            trial_frame_probs = test_probs[k_mask]
            
            # Soft Log-Odds consensus: s(c) = sum ln(P(y_w = c))
            log_probs = np.log(np.maximum(trial_frame_probs, 1e-7))
            sum_log_p = np.sum(log_probs, axis=0)
            exp_s = np.exp(sum_log_p - np.max(sum_log_p))
            trial_consensus_prob = exp_s / np.sum(exp_s)
            trial_consensus_pred = int(np.argmax(trial_consensus_prob))
            
            session_y_trial_true.append(trial_true)
            session_y_trial_pred.append(trial_consensus_pred)
            session_y_trial_prob.append(trial_consensus_prob)
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
    subject_summaries, pooled_y_trial, pooled_probs_trial, pooled_preds_trial,
    output_dir, artifact_dir
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    
    # 1. Per-Subject Accuracy Bar Chart with Session Breakdown
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    subjects = [s['sub_id'] for s in subject_summaries]
    x = np.arange(len(subjects))
    
    sub_trial_accs = [s['trial_metrics']['accuracy'] * 100 for s in subject_summaries]
    sub_frame_accs = [s['frame_metrics']['accuracy'] * 100 for s in subject_summaries]
    
    width = 0.38
    rects1 = ax.bar(x - width/2, sub_frame_accs, width, label='Frame-Level Accuracy', color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, sub_trial_accs, width, label='Trial-Consensus Accuracy (Log-Odds)', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.8)
    
    pop_trial_acc = accuracy_score(pooled_y_trial, pooled_preds_trial) * 100
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('SEED-IV Subject ID', fontsize=13, fontweight='bold')
    ax.set_title(f'Session-Level Trial-Quarantined SOTA RFPN Benchmark across 45 Sessions\nPer-Subject Frame vs. Trial Consensus Accuracy (Population Trial Acc = {pop_trial_acc:.2f}%)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Sub {s:02d}' for s in subjects], fontsize=11, fontweight='bold')
    ax.set_ylim(40.0, 105.0)
    ax.axhline(pop_trial_acc, color='darkgreen', linestyle='--', linewidth=2.0, label=f'Population Mean Trial Acc ({pop_trial_acc:.2f}%)')
    ax.axhline(90.0, color='darkred', linestyle=':', linewidth=1.8, label='Target Benchmark (90.00%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, 'target_high_acc_per_subject_bar.png')
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
    ax.set_title(f'Population Trial-Consensus Confusion Matrix (N = {len(pooled_y_trial)} Trials)\nAccuracy = {pop_trial_acc:.2f}% | Macro-F1 = {macro_f1:.4f}',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'target_high_acc_confusion_matrix.png')
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
    ax.set_title(f'Trial-Consensus Multi-Class ROC Curves\nMacro ROC-AUC = {macro_auc:.4f} across {len(pooled_y_trial)} Trials', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig_path3 = os.path.join(output_dir, 'target_high_acc_roc_curves.png')
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    
    for f in [fig_path1, fig_path2, fig_path3]:
        shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))
    print(f"[Figures] All 3 publication figures generated at 300 DPI in {output_dir} and synced to {artifact_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Session-Level Trial-Quarantined SOTA RFPN Benchmark on SEED-IV")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--subjects', type=int, nargs='+', default=None, help='Specific subjects to run (e.g. --subjects 1 2)')
    parser.add_argument('--dry_run', action='store_true', help='Run single session verification on Subject 15 Session 2')
    parser.add_argument('--data_path', type=str, default='seed_iv_processed.npz')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_splits', type=int, default=4)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device(args.device)
    
    print("=" * 80, flush=True)
    print("  Session-Level Trial-Quarantined SOTA RFPN Benchmark (SEED-IV)", flush=True)
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
        res = train_and_eval_single_session(
            15, 2, features[sub_mask], labels[sub_mask], trial_ids[sub_mask],
            device, epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits
        )
        print(f"[DRY RUN SUCCESS] Sub 15 Sess 2: Frame Acc = {res['frame_metrics']['accuracy']*100:.2f}%, Trial Acc = {res['trial_metrics']['accuracy']*100:.2f}% ({res['num_trials']} trials)", flush=True)
        return
        
    unique_subs = np.unique(subject_ids)
    if args.subjects is not None:
        unique_subs = [s for s in unique_subs if s in args.subjects]
        
    print(f"Running benchmark across {len(unique_subs)} subjects (45 sessions total)...", flush=True)
    
    all_session_results = []
    
    start_time = time.time()
    
    for sub_id in unique_subs:
        sub_start_time = time.time()
        for sess_id in range(1, 4):
            sess_mask = (subject_ids == sub_id) & (session_nums == sess_id)
            if np.sum(sess_mask) == 0:
                continue
                
            res = train_and_eval_single_session(
                sub_id, sess_id, features[sess_mask], labels[sess_mask], trial_ids[sess_mask],
                device, epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits
            )
            all_session_results.append(res)
            
            f_acc = res['frame_metrics']['accuracy'] * 100
            t_acc = res['trial_metrics']['accuracy'] * 100
            print(f"  [Sub {sub_id:02d} Sess {sess_id}] Frame Acc: {f_acc:6.2f}% | Trial Acc: {t_acc:6.2f}% ({res['num_trials']} trials evaluated)", flush=True)
            
        sub_sess_results = [r for r in all_session_results if r['sub_id'] == sub_id]
        sub_y_trial = np.concatenate([r['y_trial_true'] for r in sub_sess_results])
        sub_p_trial = np.concatenate([r['y_trial_pred'] for r in sub_sess_results])
        sub_acc = accuracy_score(sub_y_trial, sub_p_trial) * 100
        sub_elapsed = time.time() - sub_start_time
        print(f">>> Subject {sub_id:02d} Completed | 3 Sessions | Overall Trial Acc: {sub_acc:6.2f}% ({len(sub_y_trial)} trials) | Elapsed: {sub_elapsed:.1f}s\n", flush=True)
        
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
            'frame_metrics': frame_metrics,
            'trial_metrics': trial_metrics,
            'session_breakdown': [
                {
                    'sess_id': r['sess_id'],
                    'frame_acc': r['frame_metrics']['accuracy'],
                    'trial_acc': r['trial_metrics']['accuracy'],
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
    
    print("=" * 80, flush=True)
    print("  FINAL POPULATION EVALUATION RESULTS (1,080 Trials, 37,575 Frames)", flush=True)
    print("=" * 80, flush=True)
    print(f"  Frame-Level Accuracy : {population_frame_metrics['accuracy']*100:.2f}% (95% CI: [{population_frame_metrics['accuracy_ci'][0]*100:.2f}%, {population_frame_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Frame-Level Macro-F1 : {population_frame_metrics['f1']:.4f}", flush=True)
    print(f"  Frame-Level ROC-AUC  : {population_frame_metrics['auc']:.4f}", flush=True)
    print(f"  Frame-Level Kappa    : {population_frame_metrics['kappa']:.4f}", flush=True)
    print("-" * 80, flush=True)
    print(f"  Trial-Consensus Acc  : {population_trial_metrics['accuracy']*100:.2f}% (95% CI: [{population_trial_metrics['accuracy_ci'][0]*100:.2f}%, {population_trial_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Trial-Consensus F1   : {population_trial_metrics['f1']:.4f}", flush=True)
    print(f"  Trial-Consensus AUC  : {population_trial_metrics['auc']:.4f}", flush=True)
    print(f"  Trial-Consensus Kappa: {population_trial_metrics['kappa']:.4f}", flush=True)
    print("=" * 80, flush=True)
    
    # Save results to JSON
    json_output = {
        'benchmark_name': 'Session-Level Trial-Quarantined SOTA RFPN Benchmark',
        'protocol': '4-Fold Stratified Trial Cross-Validation per Session (45 Sessions)',
        'num_subjects': len(unique_subs),
        'num_sessions': len(all_session_results),
        'total_trials': len(pooled_y_trial),
        'total_frames': len(pooled_y_frame),
        'population_frame_metrics': population_frame_metrics,
        'population_trial_metrics': population_trial_metrics,
        'subject_summaries': subject_summaries
    }
    
    json_path = 'target_high_acc_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"[Export] Saved structured benchmark results to {json_path}", flush=True)
    
    # Save CSV
    csv_path = 'target_high_acc_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'subject_id', 'sessions_evaluated', 'num_trials', 'num_frames',
            'frame_accuracy', 'frame_f1', 'frame_auc', 'frame_kappa',
            'trial_accuracy', 'trial_f1', 'trial_auc', 'trial_kappa',
            'trial_acc_ci_low', 'trial_acc_ci_high', 'trial_f1_ci_low', 'trial_f1_ci_high'
        ])
        for s in subject_summaries:
            writer.writerow([
                f"Sub {s['sub_id']:02d}", s['num_sessions'], s['num_trials'], s['num_frames'],
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
            'Population (15 Subjects)', len(all_session_results), len(pooled_y_trial), len(pooled_y_frame),
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
    output_dir = os.path.join('figures', 'target_high_acc')
    artifact_dir = os.path.join("C:\\Users\\Daksh's pc\\.gemini\\antigravity\\brain\\e5c12706-2777-497e-b3d6-0e26e7492dba", 'figures', 'target_high_acc')
    generate_publication_figures(
        subject_summaries, pooled_y_trial, pooled_prob_trial, pooled_p_trial,
        output_dir, artifact_dir
    )

if __name__ == '__main__':
    main()