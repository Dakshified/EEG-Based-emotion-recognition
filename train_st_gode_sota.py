"""
Spatio-Temporal Graph Neural ODE with Evidential Dirichlet Consensus (ST-GODE)
=============================================================================
Implementation and benchmark for SEED-IV 4-class emotion recognition across 45 sessions.

Mathematical Specifications:
1. Spatial Functional Graph Construction:
   - Reshape 310D raw DE (62 channels x 5 frequency bands) into node feature tensors X in R^{62 x 5}.
   - Construct hybrid adjacency A = 0.5 * A_phys + 0.5 * Softmax(ReLU(Q K^T / sqrt(d) + B_g)).
   - Chebyshev Spectral Graph Convolution (Order K=2, 5 -> 64 dimensions).
   - Global spatial pooling & projection -> 128D latent state h_0.

2. Continuous Neural-ODE Temporal Evolution (Explicit RK4):
   - Continuous vector field dh(t)/dt = f_phi(h(t), t) with ELU non-linearities.
   - 4th-order explicit Runge-Kutta integration across dynamic trajectory -> 128D manifold state z.

3. Evidential Dirichlet Head (EAD) & Dempster-Shafer Consensus:
   - Non-negative evidence e = Softplus(Dense(64 -> 4)), Dirichlet alpha = e + 1, uncertainty u = 4 / S.
   - Dempster-Shafer trial-level evidence accumulation: E_trial = sum_{w} (1 - u_w) * e_w.
   - Evidential Dirichlet Loss (digamma ACE + annealed KL divergence).

4. Zero-Leakage Protocol (Hard Assertions):
   - Session-level 4-fold Stratified Trial Cross-Validation (45 sessions).
   - assert len(set(train_trial_ids).intersection(set(test_trial_ids))) == 0.
   - StandardScaler fitted strictly on training trials.
   - Neutral baseline reference computed strictly on training neutral trials.
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

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def compute_physical_adjacency(sigma=2.5):
    coords = np.array([GRID_COORDINATES_9X9[ch] for ch in SEED_IV_CHANNELS], dtype=np.float32)
    N = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    A_phys = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
    A_phys[A_phys < 0.1] = 0.0
    np.fill_diagonal(A_phys, 1.0)
    # Row normalize
    d = np.sum(A_phys, axis=1, keepdims=True)
    A_phys = A_phys / np.maximum(d, 1e-6)
    return torch.tensor(A_phys, dtype=torch.float32)

PHYSICAL_ADJACENCY = compute_physical_adjacency()

class ChebyshevGraphConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolution (K=2) on dynamic hybrid graph.
    """
    def __init__(self, in_features=5, out_features=64, K=2):
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        
        # Chebyshev filter weights for k = 0, 1, 2
        self.weights = nn.Parameter(torch.Tensor(K + 1, in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Dynamic adjacency projection
        self.proj_q = nn.Linear(in_features, 32, bias=False)
        self.proj_k = nn.Linear(in_features, 32, bias=False)
        self.bg = nn.Parameter(torch.zeros(62, 62))
        
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.GELU()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for k in range(self.K + 1):
            nn.init.kaiming_uniform_(self.weights[k], a=math.sqrt(5))
        nn.init.zeros_(self.bias)
        
    def forward(self, x, A_phys):
        # x: (B, 62, 5)
        B, N, F_in = x.shape
        
        # 1. Compute Dynamic Graph Adjacency
        Q = self.proj_q(x) # (B, 62, 32)
        K_mat = self.proj_k(x) # (B, 62, 32)
        scores = torch.matmul(Q, K_mat.transpose(-1, -2)) / math.sqrt(32) + self.bg # (B, 62, 62)
        A_dyn = F.softmax(F.relu(scores), dim=-1)
        
        # 2. Hybrid Adjacency Matrix
        A_phys_batch = A_phys.unsqueeze(0).to(x.device) # (1, 62, 62)
        A_hybrid = 0.5 * A_phys_batch + 0.5 * A_dyn # (B, 62, 62)
        
        # 3. Normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}
        D = torch.sum(A_hybrid, dim=-1, keepdim=True) # (B, 62, 1)
        D_inv_sqrt = torch.pow(torch.clamp(D, min=1e-6), -0.5)
        A_norm = D_inv_sqrt * A_hybrid * D_inv_sqrt.transpose(-1, -2) # (B, 62, 62)
        
        # Scaled Laplacian L_tilde = -A_norm
        L_tilde = -A_norm
        
        # 4. Chebyshev Polynomials T_0, T_1, T_2
        T0 = x # (B, 62, 5)
        T1 = torch.matmul(L_tilde, x) # (B, 62, 5)
        T2 = 2.0 * torch.matmul(L_tilde, T1) - T0 # (B, 62, 5)
        
        T_stack = [T0, T1, T2]
        
        out = torch.zeros(B, N, self.out_features, device=x.device)
        for k in range(self.K + 1):
            out = out + torch.matmul(T_stack[k], self.weights[k])
            
        out = out + self.bias
        # (B, 64, 62) -> BN -> (B, 62, 64)
        out = out.transpose(1, 2)
        out = self.bn(out)
        out = self.act(out.transpose(1, 2))
        return out

class ODEVectorField(nn.Module):
    """
    Parameter-efficient continuous vector field f_phi(h, t) with ELU non-linearities.
    """
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, t, h):
        return self.net(h)

class ExplicitRK4ODESolver(nn.Module):
    """
    Explicit 4th-Order Runge-Kutta (RK4) ODE Integrator.
    """
    def __init__(self, vector_field, num_steps=4, t_span=(0.0, 1.0)):
        super().__init__()
        self.f = vector_field
        self.num_steps = num_steps
        self.t_span = t_span
        
    def forward(self, h0):
        t0, t1 = self.t_span
        dt = (t1 - t0) / self.num_steps
        h = h0
        t = t0
        for _ in range(self.num_steps):
            k1 = self.f(t, h)
            k2 = self.f(t + 0.5 * dt, h + 0.5 * dt * k1)
            k3 = self.f(t + 0.5 * dt, h + 0.5 * dt * k2)
            k4 = self.f(t + dt, h + dt * k3)
            h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t = t + dt
        return h

class EvidentialDirichletHead(nn.Module):
    """
    Evidential Uncertainty & Belief Parameterization Head (EAD).
    """
    def __init__(self, in_dim=128, hidden_dim=64, num_classes=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus()
        )
        
    def forward(self, z):
        # Returns non-negative evidence e in R_+^4
        evidence = self.head(z)
        return evidence

class STGODE(nn.Module):
    """
    Spatio-Temporal Graph Neural ODE with Evidential Dirichlet Consensus Architecture.
    """
    def __init__(self, in_features=5, graph_dim=64, latent_dim=128, num_classes=4, ode_steps=4):
        super().__init__()
        self.gconv1 = ChebyshevGraphConv(in_features=in_features, out_features=graph_dim, K=2)
        self.gconv2 = ChebyshevGraphConv(in_features=graph_dim, out_features=graph_dim, K=2)
        
        # Spatial readout & channel projection
        self.spatial_attn = nn.Sequential(
            nn.Linear(graph_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.proj_h0 = nn.Sequential(
            nn.Linear(graph_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        
        # Neural-ODE Trajectory Evolution
        self.ode_field = ODEVectorField(hidden_dim=latent_dim)
        self.ode_solver = ExplicitRK4ODESolver(self.ode_field, num_steps=ode_steps)
        
        # Evidential Dirichlet Head
        self.evidential_head = EvidentialDirichletHead(in_dim=latent_dim, hidden_dim=64, num_classes=num_classes)
        
    def forward(self, x, A_phys):
        # x: (B, 62, 5)
        h_graph = self.gconv1(x, A_phys) # (B, 62, 64)
        h_graph = self.gconv2(h_graph, A_phys) # (B, 62, 64)
        
        # Spatial Attention Pooling over 62 channels
        attn_weights = F.softmax(self.spatial_attn(h_graph), dim=1) # (B, 62, 1)
        h_pooled = torch.sum(attn_weights * h_graph, dim=1) # (B, 64)
        
        # Initial ODE state
        h0 = self.proj_h0(h_pooled) # (B, 128)
        
        # Continuous ODE Evolution
        z_ode = self.ode_solver(h0) # (B, 128)
        
        # Evidential evidence output
        evidence = self.evidential_head(z_ode) # (B, 4)
        return evidence

def evidential_dirichlet_loss(evidence, y_true, epoch, max_epochs=35, num_classes=4):
    """
    Computes Evidential Dirichlet Loss = Digamma ACE + Annealed KL Divergence.
    """
    alpha = evidence + 1.0
    S = torch.sum(alpha, dim=-1, keepdim=True) # (B, 1)
    
    # 1. Adjusted Cross Entropy Loss
    y_one_hot = F.one_hot(y_true, num_classes=num_classes).float()
    ace_loss = torch.sum(y_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=-1).mean()
    
    # 2. Annealed KL Divergence against Flat Dirichlet Prior
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

def train_and_eval_single_session_stgode(
    sub_id, sess_id, X_sess_3d, y_sess, trial_ids_sess, device, epochs=35, batch_size=32, n_splits=4
):
    """
    Evaluates 1 session using 4-fold Stratified Trial Cross-Validation with ST-GODE.
    """
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
    
    A_phys = PHYSICAL_ADJACENCY.to(device)
    
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
        
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.long)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        test_tensor_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        # ST-GODE Model
        model = STGODE(in_features=5, graph_dim=64, latent_dim=128, num_classes=4, ode_steps=4).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
        use_amp = (device.type == 'cuda')
        scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        model.train()
        for epoch in range(1, epochs + 1):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    evidence = model(bx, A_phys)
                    loss = evidential_dirichlet_loss(evidence, by, epoch, max_epochs=epochs)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=use_amp):
                test_evidence = model(test_tensor_x, A_phys) # (N_test, 4)
                test_alpha = test_evidence + 1.0
                test_S = torch.sum(test_alpha, dim=-1, keepdim=True)
                test_probs_tensor = test_alpha / test_S
                test_uncertainty_tensor = 4.0 / test_S
                
                test_evidence_np = test_evidence.cpu().numpy()
                test_probs = test_probs_tensor.cpu().numpy()
                test_uncertainty_np = test_uncertainty_tensor.cpu().numpy()
                test_preds = np.argmax(test_probs, axis=1)
                
        session_y_frame_true.extend(y_test_fold)
        session_y_frame_pred.extend(test_preds)
        session_y_frame_prob.extend(test_probs)
        
        # Dempster-Shafer Trial-Level Evidential Consensus
        for t_val in test_trials_fold:
            k_mask = (test_t_ids_fold == t_val)
            trial_true = int(y_test_fold[k_mask][0])
            trial_evidence = test_evidence_np[k_mask] # (W, 4)
            trial_u = test_uncertainty_np[k_mask]     # (W, 1)
            
            # E_trial = sum (1 - u_w) * e_w
            weighted_evidence = (1.0 - trial_u) * trial_evidence
            E_trial = np.sum(weighted_evidence, axis=0) # (4,)
            trial_alpha = E_trial + 1.0
            trial_prob = trial_alpha / np.sum(trial_alpha)
            trial_pred = int(np.argmax(E_trial))
            
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
    rects2 = ax.bar(x + width/2, sub_trial_accs, width, label='ST-GODE Evidential Consensus Acc', color='#2ecc71', alpha=0.9, edgecolor='black', linewidth=0.8)
    
    pop_trial_acc = accuracy_score(pooled_y_trial, pooled_preds_trial) * 100
    
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('SEED-IV Subject ID', fontsize=13, fontweight='bold')
    ax.set_title(f'Spatio-Temporal Graph Neural ODE (ST-GODE) Benchmark across 45 Sessions\nPer-Subject Frame vs. Evidential Trial Consensus Accuracy (Population Trial Acc = {pop_trial_acc:.2f}%)',
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
    fig_path1 = os.path.join(output_dir, 'st_gode_per_subject_bar.png')
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
    ax.set_title(f'ST-GODE Evidential Trial Consensus Confusion Matrix (N = {len(pooled_y_trial)} Trials)\nAccuracy = {pop_trial_acc:.2f}% | Macro-F1 = {macro_f1:.4f}',
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, 'st_gode_confusion_matrix.png')
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
    ax.set_title(f'ST-GODE Trial Evidential Consensus ROC Curves\nMacro ROC-AUC = {macro_auc:.4f} across {len(pooled_y_trial)} Trials', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    plt.tight_layout()
    fig_path3 = os.path.join(output_dir, 'st_gode_roc_curves.png')
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    
    for f in [fig_path1, fig_path2, fig_path3]:
        shutil.copy2(f, os.path.join(artifact_dir, os.path.basename(f)))
    print(f"[Figures] All 3 publication figures generated at 300 DPI in {output_dir} and synced to {artifact_dir}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="ST-GODE SOTA Benchmark on SEED-IV")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--subjects', type=int, nargs='+', default=None, help='Specific subjects to run (e.g. --subjects 1 2)')
    parser.add_argument('--dry_run', action='store_true', help='Run single session verification on Subject 15 Session 2')
    parser.add_argument('--data_path', type=str, default='seed_iv_processed.npz')
    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_splits', type=int, default=4)
    args = parser.parse_args()
    
    set_seed(42)
    device_name = args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("[Device Warning] CUDA requested but not available. Gracefully falling back to CPU.", flush=True)
        device_name = 'cpu'
    device = torch.device(device_name)
    
    print("=" * 80, flush=True)
    print("  Spatio-Temporal Graph Neural ODE with Evidential Dirichlet Consensus (ST-GODE)", flush=True)
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
        res = train_and_eval_single_session_stgode(
            15, 2, features[sub_mask], labels[sub_mask], trial_ids[sub_mask],
            device, epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits
        )
        print(f"[DRY RUN SUCCESS] Sub 15 Sess 2: Frame Acc = {res['frame_metrics']['accuracy']*100:.2f}%, Trial Evidential Acc = {res['trial_metrics']['accuracy']*100:.2f}% ({res['num_trials']} trials)", flush=True)
        return
        
    unique_subs = np.unique(subject_ids)
    if args.subjects is not None:
        unique_subs = [s for s in unique_subs if s in args.subjects]
        
    print(f"Running ST-GODE benchmark across {len(unique_subs)} subjects (45 sessions total)...", flush=True)
    
    all_session_results = []
    start_time = time.time()
    
    for sub_id in unique_subs:
        sub_start_time = time.time()
        for sess_id in range(1, 4):
            sess_mask = (subject_ids == sub_id) & (session_nums == sess_id)
            if np.sum(sess_mask) == 0:
                continue
                
            res = train_and_eval_single_session_stgode(
                sub_id, sess_id, features[sess_mask], labels[sess_mask], trial_ids[sess_mask],
                device, epochs=args.epochs, batch_size=args.batch_size, n_splits=args.n_splits
            )
            all_session_results.append(res)
            
            f_acc = res['frame_metrics']['accuracy'] * 100
            t_acc = res['trial_metrics']['accuracy'] * 100
            print(f"  [Sub {sub_id:02d} Sess {sess_id}] Frame Acc: {f_acc:6.2f}% | Trial Evidential Acc: {t_acc:6.2f}% ({res['num_trials']} trials evaluated)", flush=True)
            
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
    print("  FINAL ST-GODE POPULATION RESULTS (1,080 Trials, 37,575 Frames)", flush=True)
    print("=" * 80, flush=True)
    print(f"  Frame-Level Accuracy : {population_frame_metrics['accuracy']*100:.2f}% (95% CI: [{population_frame_metrics['accuracy_ci'][0]*100:.2f}%, {population_frame_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Frame-Level Macro-F1 : {population_frame_metrics['f1']:.4f}", flush=True)
    print(f"  Frame-Level ROC-AUC  : {population_frame_metrics['auc']:.4f}", flush=True)
    print(f"  Frame-Level Kappa    : {population_frame_metrics['kappa']:.4f}", flush=True)
    print("-" * 80, flush=True)
    print(f"  Trial Evidential Acc : {population_trial_metrics['accuracy']*100:.2f}% (95% CI: [{population_trial_metrics['accuracy_ci'][0]*100:.2f}%, {population_trial_metrics['accuracy_ci'][1]*100:.2f}%])", flush=True)
    print(f"  Trial Evidential F1  : {population_trial_metrics['f1']:.4f}", flush=True)
    print(f"  Trial Evidential AUC : {population_trial_metrics['auc']:.4f}", flush=True)
    print(f"  Trial Evidential Kap : {population_trial_metrics['kappa']:.4f}", flush=True)
    print("=" * 80, flush=True)
    
    # Save results to JSON
    json_output = {
        'benchmark_name': 'Spatio-Temporal Graph Neural ODE with Evidential Dirichlet Consensus (ST-GODE)',
        'protocol': '4-Fold Stratified Trial Cross-Validation per Session (45 Sessions)',
        'num_subjects': len(unique_subs),
        'num_sessions': len(all_session_results),
        'total_trials': len(pooled_y_trial),
        'total_frames': len(pooled_y_frame),
        'population_frame_metrics': population_frame_metrics,
        'population_trial_metrics': population_trial_metrics,
        'subject_summaries': subject_summaries
    }
    
    json_path = 'st_gode_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"[Export] Saved structured benchmark results to {json_path}", flush=True)
    
    # Save CSV
    csv_path = 'st_gode_results.csv'
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
    output_dir = os.path.join('figures', 'st_gode')
    artifact_dir = os.path.join("C:\\Users\\Daksh's pc\\.gemini\\antigravity\\brain\\e5c12706-2777-497e-b3d6-0e26e7492dba", 'figures', 'st_gode')
    generate_publication_figures(
        subject_summaries, pooled_y_trial, pooled_prob_trial, pooled_p_trial,
        output_dir, artifact_dir
    )

if __name__ == '__main__':
    main()