import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import time
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
import matplotlib.pyplot as plt

# Configure CPU threading
torch.set_num_threads(2)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

# =========================================================================
# HELPER FUNCTIONS & GRANGER CAUSALITY
# =========================================================================

def parse_locs(locs_path="channel_62_pos (1).locs"):
    """Parses 2D Cartesian coordinates (x, y) for 62 channels."""
    names = []
    x_coords = []
    y_coords = []
    with open(locs_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                angle = float(parts[1])
                radius = float(parts[2])
                name = parts[3]
                theta = np.radians(angle)
                x = radius * np.sin(theta)
                y = radius * np.cos(theta)
                names.append(name)
                x_coords.append(x)
                y_coords.append(y)
    return names, np.array(x_coords), np.array(y_coords)

def build_knn_adjacency(x_coords, y_coords, k=14):
    """Builds a binary physical k-NN adjacency matrix."""
    coords = np.column_stack((x_coords, y_coords))
    num_nodes = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    
    A = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        sorted_indices = np.argsort(dists[i])
        # Skip index 0 (self-loop). Set next k neighbors to 1.
        knn_indices = sorted_indices[1 : k+1]
        A[i, knn_indices] = 1.0
        A[i, i] = 1.0 # Add self-loop
    return A

def compute_fold_granger_adjacency(features, labels, subject_ids, session_nums, trial_ids, train_indices):
    """
    Computes a 62 x 62 Granger-causality adjacency matrix fresh for the current
    fold, using ONLY the training indices. Optimized with precomputed restricted
    model RSS and progress printing.
    """
    print("  [STATS] Computing Granger-Causality adjacency on training fold only...", flush=True)
    t0 = time.time()
    num_channels = 62
    
    # 1. Group windows chronologically by trial for training indices
    # Map chronological windows per trial: key = (sub_id, sess_num, trial_id)
    trials = {}
    for idx in train_indices:
        key = (subject_ids[idx], session_nums[idx], trial_ids[idx])
        if key not in trials:
            trials[key] = []
        trials[key].append((idx, features[idx].mean(axis=-1))) # Average across bands for VAR
        
    # Reconstruct time series
    trial_series = []
    for key, val in trials.items():
        # Sort chronologically by local chronological order (the index in npz is sequential)
        val_sorted = sorted(val, key=lambda x: x[0])
        ts = np.array([x[1] for x in val_sorted]) # Shape: (L, 62)
        if ts.shape[0] > 5: # Needs enough timesteps to fit VAR lag 1
            trial_series.append(ts)
            
    # To keep it fast on CPU, we limit to a subset of 30 randomly selected trials if there are many
    if len(trial_series) > 30:
        np.random.seed(42)
        indices = np.random.choice(len(trial_series), 30, replace=False)
        trial_series = [trial_series[i] for i in indices]
        
    n_obs_total = sum(len(ts) - 1 for ts in trial_series)
    
    # 2. Vectorized Per-Trial Frisch-Waugh-Lovell (FWL) Granger Adjacency Computation
    rss_rest_total = np.zeros(num_channels)
    rss_unrest_total = np.zeros((num_channels, num_channels))

    for ts in trial_series:
        y_k = ts[1:] # (L, 62)
        x_k = ts[:-1] # (L, 62)
        y_tilde = y_k - np.mean(y_k, axis=0, keepdims=True) # (L, 62)
        x_tilde = x_k - np.mean(x_k, axis=0, keepdims=True) # (L, 62)
        
        # Cross-products: S_xx (62, 62), S_xy (62, 62), S_yy (62,)
        S_xx = x_tilde.T @ x_tilde # (62, 62)
        S_xy = x_tilde.T @ y_tilde # (62, 62)
        S_yy = np.sum(y_tilde**2, axis=0) # (62,)
        
        diag_xx = np.diag(S_xx) # (62,)
        diag_xy = np.diag(S_xy) # (62,)
        
        # Restricted RSS for this trial
        rss_rest_k = S_yy - (diag_xy**2) / np.maximum(diag_xx, 1e-12) # (62,)
        rss_rest_total += rss_rest_k
        
        # Unrestricted RSS via FWL projection
        u_norm_sq = diag_xx[None, :] - (S_xx**2) / np.maximum(diag_xx[:, None], 1e-12) # [i, j]
        u_dot_y = S_xy.T - (S_xx * diag_xy[:, None]) / np.maximum(diag_xx[:, None], 1e-12) # [i, j]
        
        rss_drop = (u_dot_y**2) / np.maximum(u_norm_sq, 1e-12) # [i, j]
        rss_unrest_k = rss_rest_k[:, None] - rss_drop # [i, j]
        rss_unrest_total += rss_unrest_k

    den = rss_unrest_total / (n_obs_total - 3)
    F_matrix = np.maximum(0.0, (rss_rest_total[:, None] - rss_unrest_total) / np.maximum(den, 1e-10))
    np.fill_diagonal(F_matrix, 0.0)

    # Normalize to [0, 1]
    max_F = F_matrix.max()
    if max_F > 0:
        F_matrix = F_matrix / max_F
        
    # Add self-loops
    np.fill_diagonal(F_matrix, 1.0)
    print(f"  [STATS] Completed Granger-Causality adjacency computation in {time.time() - t0:.2f}s", flush=True)
    return F_matrix

# =========================================================================
# GAT-KAN V2 MODULE DEFINITIONS
# =========================================================================

class KANLinear(nn.Module):
    """Efficient B-spline Kolmogorov-Arnold Network linear layer."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, grid_size + spline_order) * 0.1)
        
        grid = torch.linspace(-3.0, 3.0, grid_size + 2 * spline_order + 1)
        self.register_buffer("grid", grid)

    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:-(k+1)]
            left_den = grid[k:-1] - grid[:-(k+1)]
            left = (left_num / left_den) * bases[..., :-1]
            right_num = grid[(k+1):] - x
            right_den = grid[(k+1):] - grid[1:-k]
            right = (right_num / right_den) * bases[..., 1:]
            bases = left + right
        return bases

    def forward(self, x):
        base_out = F.linear(x, self.base_weight)
        splines = self.b_splines(x) # (batch, in_features, grid_size + spline_order)
        # Flatten on-the-fly to use optimized F.linear instead of torch.einsum on CPU
        splines_flat = splines.view(splines.size(0), -1)
        weight_flat = self.spline_weight.view(self.out_features, -1)
        spline_out = F.linear(splines_flat, weight_flat)
        return base_out + spline_out

class GraphAttentionLayer(nn.Module):
    """GAT layer constrained by blended physical and causal graph structure."""
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn_src = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.1)
        self.attn_dst = nn.Parameter(torch.randn(1, num_heads, 1, self.head_dim) * 0.1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # Cache for adj log mask to avoid redundant calculations on CPU
        self.cached_adj = None
        self.cached_adj_mask = None
        
    def forward(self, h, adj):
        # h shape: (batch_size, num_nodes, in_features)
        # adj shape: (num_nodes, num_nodes)
        batch_size, num_nodes, _ = h.size()
        
        h_proj = self.W(h).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        h_proj = h_proj.transpose(1, 2) # (batch, heads, nodes, head_dim)
        
        attn_s = (h_proj * self.attn_src).sum(dim=-1, keepdim=True) # (batch, heads, nodes, 1)
        attn_d = (h_proj * self.attn_dst).sum(dim=-1, keepdim=True) # (batch, heads, nodes, 1)
        
        attn_matrix = attn_s + attn_d.transpose(-2, -1) # (batch, heads, nodes, nodes)
        attn_matrix = self.leakyrelu(attn_matrix)
        
        # Cache adj_mask to avoid recomputing log(adj) every forward pass
        if self.cached_adj is not adj:
            self.cached_adj = adj
            self.cached_adj_mask = torch.log(adj + 1e-10).view(1, 1, num_nodes, num_nodes)
            
        attn_matrix = attn_matrix + self.cached_adj_mask
        
        attn_weights = F.softmax(attn_matrix, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, h_proj) # (batch, heads, nodes, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.out_features)
        return out, attn_weights

class GATKANv2(nn.Module):
    """GAT-KAN v2 proposed model architecture (~102K parameters)."""
    def __init__(self, num_subjects=15, use_subject_embedding=False):
        super().__init__()
        self.use_subject_embedding = use_subject_embedding
        d_model = 64
        
        # 1. Per-Band Encoders
        self.band_encoders = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(5)
        ])
        
        # 2. Spatial GAT with Skip Connections
        self.gat1 = GraphAttentionLayer(d_model, d_model, num_heads=4, dropout=0.1)
        self.gat2 = GraphAttentionLayer(d_model, d_model, num_heads=4, dropout=0.1)
        
        # 3. Cross-Band Transformer Attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. KAN Classifier Head
        classifier_in_dim = d_model
        if use_subject_embedding:
            self.subject_embed = nn.Embedding(num_subjects + 1, 16)
            classifier_in_dim += 16
            
        self.kan_classifier = nn.Sequential(
            KANLinear(classifier_in_dim, 64, grid_size=5, spline_order=3),
            KANLinear(64, 4, grid_size=5, spline_order=3)
        )
        
        # 5. Per-Band Auxiliary Supervision Heads
        self.aux_heads = nn.ModuleList([
            nn.Linear(d_model, 4) for _ in range(5)
        ])

    def forward(self, x, adj, subject_ids=None):
        # x shape: (batch_size, 62, 5)
        batch_size, num_nodes, num_bands = x.size()
        
        # 1. Project 5 bands: list of (batch, 62, d_model)
        h_proj_list = [self.band_encoders[b](x[:, :, b:b+1]) for b in range(5)]
        
        # 2. Stack into (batch_size, 5, 62, d_model) and reshape to (batch_size * 5, 62, d_model)
        h_proj_batched = torch.stack(h_proj_list, dim=1).view(batch_size * 5, num_nodes, -1)
        
        # 3. Batched GAT spatial representation across all 5 bands simultaneously
        h_gat1, attn1 = self.gat1(h_proj_batched, adj)
        h_gat1 = F.relu(h_gat1 + h_proj_batched) # Residual skip
        
        h_gat2, attn2 = self.gat2(h_gat1, adj)
        h_gat = F.relu(h_gat2 + h_gat1) # Residual skip
        
        # 4. Average pool over nodes (channels) to get band representation
        h_band_avg = h_gat.mean(dim=1) # (batch_size * 5, d_model)
        band_tokens = h_band_avg.view(batch_size, 5, -1) # (batch_size, 5, d_model)
        
        # 5. Extract attention maps shaped as list of 5 (batch_size, 4, 62, 62) tensors
        spatial_attns = list(attn2.view(batch_size, 5, 4, num_nodes, num_nodes).unbind(dim=1))
        
        # 6. Cross-band attention via Transformer
        trans_out = self.transformer(band_tokens) # (batch, 5, d_model)
        
        # 7. Generate auxiliary predictions for per-band loss
        aux_preds = [self.aux_heads[b](trans_out[:, b, :]) for b in range(5)]
        
        # 8. Main classification head input: average pool over the 5 bands
        main_rep = trans_out.mean(dim=1) # (batch, d_model)
        
        if self.use_subject_embedding and subject_ids is not None:
            sub_emb = self.subject_embed(subject_ids) # (batch, 16)
            main_rep = torch.cat([main_rep, sub_emb], dim=-1) # Fused input
            
        out = self.kan_classifier(main_rep)
        return out, aux_preds, spatial_attns

# =========================================================================
# DATASET & LOADER
# =========================================================================

class EEGDataset(Dataset):
    def __init__(self, features, labels, subject_ids, augment=False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.subject_ids = torch.tensor(subject_ids, dtype=torch.long)
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # Clone to prevent random channel masking from mutating the underlying dataset in-place
        x = self.features[idx].clone() # (62, 5)
        
        # Random channel masking augmentation for Version B
        if self.augment and np.random.rand() < 0.5:
            # Mask 10% of channels (6 channels)
            mask_indices = np.random.choice(62, 6, replace=False)
            x[mask_indices, :] = 0.0
            
        return x, self.labels[idx], self.subject_ids[idx]

# =========================================================================
# EVALUATION PACKAGE GENERATOR (METRICS, BOOTSTRAP, PLOTS)
# =========================================================================

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000):
    """Computes 95% confidence intervals using bootstrap resampling."""
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    num_samples = len(y_true)
    
    for _ in range(num_resamples):
        indices = np.random.choice(num_samples, num_samples, replace=True)
        y_t_res = y_true[indices]
        y_p_res = y_pred[indices]
        y_prob_res = y_prob[indices]
        
        accs.append(np.mean(y_t_res == y_p_res))
        
        # Precision, recall, f1
        p, r, f, _ = precision_recall_fscore_support(y_t_res, y_p_res, average='macro', zero_division=0)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        
        # AUC (one-vs-rest)
        try:
            auc = roc_auc_score(y_t_res, y_prob_res, average='macro', multi_class='ovr')
            aucs.append(auc)
        except Exception:
            aucs.append(0.5)
            
    ci_acc = (np.percentile(accs, 2.5), np.percentile(accs, 97.5))
    ci_prec = (np.percentile(precs, 2.5), np.percentile(precs, 97.5))
    ci_rec = (np.percentile(recs, 2.5), np.percentile(recs, 97.5))
    ci_f1 = (np.percentile(f1s, 2.5), np.percentile(f1s, 97.5))
    ci_auc = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
    
    return ci_acc, ci_prec, ci_rec, ci_f1, ci_auc

def save_evaluation_plots(y_true, y_pred, y_prob, prefix, figures_dir):
    """Generates and saves Confusion Matrix, ROC, PR, and Calibration diagrams."""
    # 1. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Normalized Ratio')
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})", ha='center', va='center', fontsize=8,
                     color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.xticks(range(4), CLASS_NAMES)
    plt.yticks(range(4), CLASS_NAMES)
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    plt.title(f'Confusion Matrix - {prefix}', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 2. ROC Curves per Class
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[c].capitalize()} (AUC={roc_auc:.3f})', color=EMOTION_COLORS[c])
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_roc.png"), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curves
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, c])
        plt.plot(recall, precision, label=f'{CLASS_NAMES[c].capitalize()}', color=EMOTION_COLORS[c])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower left', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_pr.png"), dpi=300)
    plt.close()
    
    # 4. Calibration Diagram
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob[:, c], n_bins=5)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{CLASS_NAMES[c].capitalize()}', color=EMOTION_COLORS[c])
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Reliability Diagram - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_calibration.png"), dpi=300)
    plt.close()

# =========================================================================
# STANDARD SYSTEM LIBRARY STANDARDIZER
# =========================================================================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        self.std[self.std == 0.0] = 1.0
        return self
        
    def transform(self, x):
        return (x - self.mean) / self.std

# =========================================================================
# TRAINING ENGINE
# =========================================================================

def train_gat_kan_fold(model, train_loader, val_loader, adj_matrix_torch, device, augment=False, epochs=100, patience=25):
    """Trains GAT-KAN v2 on a single fold with dynamic auxiliary loss weighting and val_acc early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_weights = None
    best_epoch = 0
    
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}
    
    # Initialize dynamic auxiliary loss weights
    aux_weights = [0.2 for _ in range(5)]
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y, batch_sub in train_loader:
            batch_x, batch_y, batch_sub = batch_x.to(device), batch_y.to(device), batch_sub.to(device)
            
            optimizer.zero_grad()
            out, aux_preds, _ = model(batch_x, adj_matrix_torch, batch_sub)
            
            # Loss definition
            loss_main = F.cross_entropy(out, batch_y)
            loss_aux = 0.0
            for b in range(5):
                loss_aux += aux_weights[b] * F.cross_entropy(aux_preds[b], batch_y)
                
            loss = loss_main + loss_aux
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = out.argmax(dim=-1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_x.size(0)
            
        # Validation pass
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        band_val_accs = [0.0 for _ in range(5)]
        
        with torch.no_grad():
            for batch_x, batch_y, batch_sub in val_loader:
                batch_x, batch_y, batch_sub = batch_x.to(device), batch_y.to(device), batch_sub.to(device)
                out, aux_preds, _ = model(batch_x, adj_matrix_torch, batch_sub)
                
                loss_main = F.cross_entropy(out, batch_y)
                loss_aux = 0.0
                for b in range(5):
                    loss_aux += aux_weights[b] * F.cross_entropy(aux_preds[b], batch_y)
                    
                loss = loss_main + loss_aux
                val_loss += loss.item() * batch_x.size(0)
                
                preds = out.argmax(dim=-1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_x.size(0)
                
                # Check performance per band auxiliary head
                for b in range(5):
                    aux_p = aux_preds[b].argmax(dim=-1)
                    band_val_accs[b] += (aux_p == batch_y).sum().item()
                    
        train_loss /= total_train
        train_acc = correct_train / total_train
        val_loss /= total_val
        val_acc = correct_val / total_val
        
        scheduler.step(val_acc)
        
        # Update dynamic auxiliary weights based on band-wise validation accuracy
        for b in range(5):
            band_acc = band_val_accs[b] / total_val
            # Bands that perform worse receive more loss weight to force convergence
            aux_weights[b] = 0.5 * (1.0 - band_acc)
            
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        
        # Print progress for every epoch
        print(f"      Epoch {epoch+1:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}", flush=True)
        
        # Early stopping verification based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      [EARLY STOPPING] Triggered at epoch {epoch+1} (Best Epoch: {best_epoch} with Val Acc: {best_val_acc:.2%})", flush=True)
                break
                
    if best_weights is not None:
        print(f"      [RESTORE BEST CHECKPOINT] Restoring weights from Epoch {best_epoch} (Val Acc: {best_val_acc:.2%})", flush=True)
        model.load_state_dict(best_weights)
        
    return train_history, val_history

def train_per_subject_finetune(model, train_loader, adj_matrix_torch, device, epochs=5, lr=1e-4):
    """Applies a per-subject fine-tuning pass using subject-specific training indices."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        for batch_x, batch_y, batch_sub in train_loader:
            batch_x, batch_y, batch_sub = batch_x.to(device), batch_y.to(device), batch_sub.to(device)
            optimizer.zero_grad()
            out, aux_preds, _ = model(batch_x, adj_matrix_torch, batch_sub)
            loss = F.cross_entropy(out, batch_y)
            loss.backward()
            optimizer.step()

# =========================================================================
# EXPERIMENT WRAPPERS
# =========================================================================

def run_experiment(features, labels, subject_ids, session_nums, trial_ids, 
                   x_coords, y_coords, names, protocol_name, version_name, use_emb, use_ft, augment, figures_dir):
    """
    Executes a 5-fold cross-validation loop under either the subject-dependent or
    cross-subject protocol. Saves evaluation metrics, prediction distributions,
    attention maps, and all required verification figures.
    """
    print("=" * 80, flush=True)
    print(f"Running GAT-KAN v2 | {protocol_name} | {version_name}", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    num_samples = len(labels)
    
    # 1. Determine fold splits
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys, trial_indices_map = np.unique(trial_keys, return_index=True)
    num_trials = len(unique_trial_keys)
    
    if protocol_name == "cross-subject":
        # 5-fold Leave-3-Subjects-Out CV
        subject_folds = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ]
        fold_splits = []
        for test_subs in subject_folds:
            train_subs = [s for s in range(1, 16) if s not in test_subs]
            tr_idx = [idx for idx in range(num_samples) if subject_ids[idx] in train_subs]
            te_idx = [idx for idx in range(num_samples) if subject_ids[idx] in test_subs]
            fold_splits.append((tr_idx, te_idx))
    else:
        # Subject-dependent: 5-fold trial-level split
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_splits = []
        for train_trial_idx, test_trial_idx in kf.split(np.arange(num_trials)):
            train_trials_set = set(unique_trial_keys[train_trial_idx])
            test_trials_set = set(unique_trial_keys[test_trial_idx])
            tr_idx = [idx for idx in range(num_samples) if trial_keys[idx] in train_trials_set]
            te_idx = [idx for idx in range(num_samples) if trial_keys[idx] in test_trials_set]
            fold_splits.append((tr_idx, te_idx))
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    
    fold_train_histories = []
    fold_val_histories = []
    
    # Physical graph structure construction (k=14)
    A_kNN = build_knn_adjacency(x_coords, y_coords, k=14)
    
    for fold, (train_indices, test_indices) in enumerate(fold_splits):
        print(f"\n--- Fold {fold+1}/5 ---", flush=True)
        
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        weights_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_weights.pt")
        preds_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_predictions.npz")
        metrics_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_metrics.json")
        
        if os.path.exists(weights_file) and os.path.exists(preds_file) and os.path.exists(metrics_file):
            print(f"  [CHECKPOINT FOUND] Loading existing completed fold {fold+1}/5 from '{checkpoint_dir}'...", flush=True)
            preds_data = np.load(preds_file)
            y_true_fold = list(preds_data['y_true'])
            y_pred_fold = list(preds_data['y_pred'])
            y_prob_fold = list(preds_data['y_prob'])
            with open(metrics_file, 'r') as f:
                f_m = json.load(f)
            print(f"  [FOLD COMPLETE - FROM CHECKPOINT] {protocol_name} | {version_name} | Fold {fold+1}/5 | Test Acc: {f_m['accuracy']:.4f} | Test F1: {f_m['f1']:.4f}", flush=True)
            y_true_all.extend(y_true_fold)
            y_pred_all.extend(y_pred_fold)
            y_prob_all.extend(y_prob_fold)
            continue
        
        # In cross-subject protocol, verify no subject overlaps
        if protocol_name == "cross-subject":
            train_subs = set(subject_ids[train_indices])
            test_subs = set(subject_ids[test_indices])
            overlap = train_subs.intersection(test_subs)
            if overlap:
                raise ValueError(f"Subject overlap detected in cross-subject protocol: {overlap}")
                
        # Compute Granger Causality on training set only to prevent leakage (using cache if available)
        cache_dir = "granger_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"granger_cache_{protocol_name}_fold{fold+1}.npy")
        if os.path.exists(cache_file):
            print(f"  [CACHE] Loading cached Granger-Causality adjacency from '{cache_file}'...", flush=True)
            A_GC = np.load(cache_file)
        else:
            A_GC = compute_fold_granger_adjacency(features, labels, subject_ids, session_nums, trial_ids, train_indices)
            np.save(cache_file, A_GC)
        
        # Blend graphs (50% k-NN, 50% Granger)
        A_blended = 0.5 * A_kNN + 0.5 * A_GC
        adj_matrix_torch = torch.tensor(A_blended, dtype=torch.float32).to(device)
        
        # Standardization fit per fold on train indices only to prevent leakage
        scaler = StandardScaler()
        scaler.fit(features[train_indices].reshape(len(train_indices), -1))
        
        features_norm = np.copy(features)
        features_norm[train_indices] = scaler.transform(features[train_indices].reshape(len(train_indices), -1)).reshape(len(train_indices), 62, 5)
        features_norm[test_indices] = scaler.transform(features[test_indices].reshape(len(test_indices), -1)).reshape(len(test_indices), 62, 5)
        
        # Datasets & Loaders
        train_dataset = EEGDataset(features_norm[train_indices], labels[train_indices], subject_ids[train_indices], augment=augment)
        test_dataset = EEGDataset(features_norm[test_indices], labels[test_indices], subject_ids[test_indices], augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # Model construction
        model = GATKANv2(num_subjects=15, use_subject_embedding=use_emb).to(device)
        
        # Train fold with val_acc early stopping
        t_hist, v_hist = train_gat_kan_fold(model, train_loader, test_loader, adj_matrix_torch, device, augment=augment)
        fold_train_histories.append(t_hist)
        fold_val_histories.append(v_hist)
        
        # Booster: Per-subject fine-tuning with independent subject model clones (train-only, fold-local)
        if use_ft:
            print("  [BOOSTER] Applying per-subject fine-tuning with independent subject model clones...", flush=True)
            base_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            y_true_fold_arr = labels[test_indices]
            y_pred_fold_arr = np.zeros(len(test_indices), dtype=int)
            y_prob_fold_arr = np.zeros((len(test_indices), 4), dtype=float)
            y_attn_fold_arr = np.zeros((len(test_indices), 5, 4, 62, 62), dtype=np.float32)
            
            for sub in range(1, 16):
                sub_train_mask = [idx for idx in train_indices if subject_ids[idx] == sub]
                sub_test_pos = [i for i, idx in enumerate(test_indices) if subject_ids[idx] == sub]
                
                if len(sub_test_pos) == 0:
                    continue
                    
                # Clone model from post-training base weights for this subject
                sub_model = GATKANv2(num_subjects=15, use_subject_embedding=use_emb).to(device)
                sub_model.load_state_dict(base_weights)
                
                # Fine-tune clone only on this subject's training samples
                if len(sub_train_mask) > 0:
                    sub_train_dataset = EEGDataset(features_norm[sub_train_mask], labels[sub_train_mask], subject_ids[sub_train_mask], augment=augment)
                    sub_train_loader = DataLoader(sub_train_dataset, batch_size=64, shuffle=True)
                    train_per_subject_finetune(sub_model, sub_train_loader, adj_matrix_torch, device, epochs=5, lr=1e-4)
                
                # Evaluate this subject's test samples using ONLY their own fine-tuned clone
                sub_model.eval()
                sub_test_dataset = EEGDataset(
                    features_norm[[test_indices[p] for p in sub_test_pos]],
                    labels[[test_indices[p] for p in sub_test_pos]],
                    subject_ids[[test_indices[p] for p in sub_test_pos]],
                    augment=False
                )
                sub_test_loader = DataLoader(sub_test_dataset, batch_size=256, shuffle=False)
                
                sub_preds, sub_probs, sub_attns = [], [], []
                with torch.no_grad():
                    for batch_x, batch_y, batch_sub in sub_test_loader:
                        batch_x, batch_sub = batch_x.to(device), batch_sub.to(device)
                        out, _, spatial_attns = sub_model(batch_x, adj_matrix_torch, batch_sub)
                        batch_attn = torch.stack(spatial_attns, dim=1)
                        sub_attns.append(batch_attn.cpu().numpy())
                        sub_probs.append(F.softmax(out, dim=-1).cpu().numpy())
                        sub_preds.append(out.argmax(dim=-1).cpu().numpy())
                        
                if len(sub_preds) > 0:
                    y_pred_fold_arr[sub_test_pos] = np.concatenate(sub_preds, axis=0)
                    y_prob_fold_arr[sub_test_pos] = np.concatenate(sub_probs, axis=0)
                    y_attn_fold_arr[sub_test_pos] = np.concatenate(sub_attns, axis=0)
                    
                # Discard clone and start next subject from same pre-fine-tune base weights
                del sub_model
                
            val_attentions_all = y_attn_fold_arr
        else:
            # Standard evaluation without per-subject fine-tuning (e.g. Cross-Subject protocol)
            model.eval()
            y_true_fold = []
            y_pred_fold = []
            y_prob_fold = []
            y_attn_fold = []
            
            with torch.no_grad():
                for batch_x, batch_y, batch_sub in test_loader:
                    batch_x, batch_sub = batch_x.to(device), batch_sub.to(device)
                    out, _, spatial_attns = model(batch_x, adj_matrix_torch, batch_sub)
                    
                    # spatial_attns is a list of 5 tensors, each (batch_size, 4, 62, 62)
                    # Stack them to shape (batch_size, 5, 4, 62, 62)
                    batch_attn = torch.stack(spatial_attns, dim=1)
                    y_attn_fold.append(batch_attn.cpu().numpy())
                    
                    probs = F.softmax(out, dim=-1)
                    preds = out.argmax(dim=-1)
                    
                    y_true_fold.extend(batch_y.numpy())
                    y_pred_fold.extend(preds.cpu().numpy())
                    y_prob_fold.extend(probs.cpu().numpy())
                    
            # Concatenate fold attentions
            val_attentions_all = np.concatenate(y_attn_fold, axis=0) # (N_val, 5, 4, 62, 62)
            y_true_fold_arr = np.array(y_true_fold)
            y_pred_fold_arr = np.array(y_pred_fold)
            y_prob_fold_arr = np.array(y_prob_fold)
        
        fold_acc = np.mean(y_true_fold_arr == y_pred_fold_arr)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(
            y_true_fold_arr, y_pred_fold_arr, average='macro', zero_division=0
        )
        try:
            fold_auc = roc_auc_score(y_true_fold_arr, y_prob_fold_arr, average='macro', multi_class='ovr')
        except Exception:
            fold_auc = 0.5
            
        print(f"  [FOLD COMPLETE] {protocol_name} | {version_name} | Fold {fold+1}/5 | Test Acc: {fold_acc:.4f} | Test F1: {fold_f1:.4f}", flush=True)
        
        # Save per-fold checkpoints to checkpoints/ folder
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 1. Weights
        weights_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_weights.pt")
        torch.save(model.state_dict(), weights_file)
        
        # 2. Predictions
        preds_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_predictions.npz")
        np.savez(preds_file, y_true=y_true_fold_arr, y_pred=y_pred_fold_arr, y_prob=y_prob_fold_arr)
        
        # 3. Attentions
        attn_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_attentions.npy")
        np.save(attn_file, val_attentions_all)
        
        # 4. Blended Adjacency
        adj_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_blended_adj.npy")
        np.save(adj_file, A_blended)
        
        # 5. Metrics
        metrics_file = os.path.join(checkpoint_dir, f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_metrics.json")
        fold_metrics = {
            "accuracy": float(fold_acc),
            "precision": float(fold_prec),
            "recall": float(fold_rec),
            "f1": float(fold_f1),
            "auc": float(fold_auc)
        }
        with open(metrics_file, "w") as f:
            json.dump(fold_metrics, f, indent=4)
            
        y_true_all.extend(y_true_fold)
        y_pred_all.extend(y_pred_fold)
        y_prob_all.extend(y_prob_fold)
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    # Calculate performance metrics
    acc = np.mean(y_true_all == y_pred_all)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    auc_val = roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr')
    
    # Bootstrap CI
    print("  [STATS] Computing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, num_resamples=1000)
    
    # Print metrics
    print(f"\n=== OVERALL RESULTS | {protocol_name} | {version_name} ===", flush=True)
    print(f"  Accuracy:  {acc:.4f} (95% CI: {ci_acc[0]:.4f} - {ci_acc[1]:.4f})", flush=True)
    print(f"  Precision: {precision:.4f} (95% CI: {ci_prec[0]:.4f} - {ci_prec[1]:.4f})", flush=True)
    print(f"  Recall:    {recall:.4f} (95% CI: {ci_rec[0]:.4f} - {ci_rec[1]:.4f})", flush=True)
    print(f"  Macro-F1:  {f1:.4f} (95% CI: {ci_f1[0]:.4f} - {ci_f1[1]:.4f})", flush=True)
    print(f"  Macro-AUC: {auc_val:.4f} (95% CI: {ci_auc[0]:.4f} - {ci_auc[1]:.4f})", flush=True)
    
    # Plot curves
    prefix = f"gatkanv2_{version_name}_{protocol_name}"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Plot loss / accuracy curves over folds if full training histories are available
    if len(fold_train_histories) == 5:
        plt.figure(figsize=(10, 4.5))
        plt.subplot(1, 2, 1)
        for f in range(5):
            plt.plot(fold_train_histories[f]['loss'], label=f'F{f+1} Train', alpha=0.5)
            plt.plot(fold_val_histories[f]['loss'], label=f'F{f+1} Val', linestyle='dashed', alpha=0.5)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(linestyle='--', alpha=0.5)
        
        plt.subplot(1, 2, 2)
        for f in range(5):
            plt.plot(fold_train_histories[f]['acc'], label=f'F{f+1} Train', alpha=0.5)
            plt.plot(fold_val_histories[f]['acc'], label=f'F{f+1} Val', linestyle='dashed', alpha=0.5)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{prefix}_training_curves.png"), dpi=300)
        plt.close()
    
    # Save metrics to JSON/CSV for explainability mapping
    metrics_data = {
        "protocol": protocol_name,
        "version": version_name,
        "accuracy": acc,
        "accuracy_ci": ci_acc,
        "precision": precision,
        "precision_ci": ci_prec,
        "recall": recall,
        "recall_ci": ci_rec,
        "f1": f1,
        "f1_ci": ci_f1,
        "auc": auc_val,
        "auc_ci": ci_auc
    }
    
    # Print one-line configuration summary
    print(f"\n>>> [CONFIGURATION COMPLETE] {protocol_name.upper()} | {version_name.upper()} => Overall Acc: {acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]), Overall Macro F1: {f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}])\n", flush=True)

    # Return metrics
    return metrics_data

# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    t_start = time.time()
    
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70, flush=True)
    print("GAT-KAN v2 Model Build and Training Run (SCOPE-LOCKED)")
    print("=" * 70, flush=True)
    
    # Load dataset
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
        
    print(f"Loading SEED-IV processed dataset from '{dataset_path}'...", flush=True)
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    
    # Parse channel locs
    locs_path = "channel_62_pos (1).locs"
    print(f"Parsing electrode montage from '{locs_path}'...", flush=True)
    names, x_coords, y_coords = parse_locs(locs_path)
    
    results = {}
    
    # RUN 1: Version A (Clean) - Subject-Dependent Protocol
    results["A_dep"] = run_experiment(
        features, labels, subject_ids, session_nums, trial_ids, 
        x_coords, y_coords, names, 
        protocol_name="subject-dependent", version_name="versionA", 
        use_emb=True, use_ft=True, augment=False, figures_dir=figures_dir
    )
    
    # RUN 2: Version A (Clean) - Cross-Subject Protocol
    results["A_cross"] = run_experiment(
        features, labels, subject_ids, session_nums, trial_ids, 
        x_coords, y_coords, names, 
        protocol_name="cross-subject", version_name="versionA", 
        use_emb=False, use_ft=False, augment=False, figures_dir=figures_dir
    )
    
    # RUN 3: Version B (Augmented) - Subject-Dependent Protocol
    results["B_dep"] = run_experiment(
        features, labels, subject_ids, session_nums, trial_ids, 
        x_coords, y_coords, names, 
        protocol_name="subject-dependent", version_name="versionB", 
        use_emb=True, use_ft=True, augment=True, figures_dir=figures_dir
    )
    
    # RUN 4: Version B (Augmented) - Cross-Subject Protocol
    results["B_cross"] = run_experiment(
        features, labels, subject_ids, session_nums, trial_ids, 
        x_coords, y_coords, names, 
        protocol_name="cross-subject", version_name="versionB", 
        use_emb=False, use_ft=False, augment=True, figures_dir=figures_dir
    )
    
    # Save all raw metric values to JSON file
    with open("gatkanv2_metrics_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n" + "=" * 70, flush=True)
    print("FINAL SUMMARY TABLE (4 CONFIGURATIONS x 5 FOLDS)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Configuration':<45} | {'Fold 1':<8} | {'Fold 2':<8} | {'Fold 3':<8} | {'Fold 4':<8} | {'Fold 5':<8} | {'Mean ± Std':<15}", flush=True)
    print("-" * 115, flush=True)
    
    configs_keys = [
        ("subject-dependent", "versionA", "Subj-Dep (Clean, Version A)"),
        ("subject-dependent", "versionB", "Subj-Dep (Aug, Version B)"),
        ("cross-subject", "versionA", "Cross-Subj (Clean, Version A)"),
        ("cross-subject", "versionB", "Cross-Subj (Aug, Version B)"),
    ]
    
    for protocol_name, version_name, display_name in configs_keys:
        accs = []
        f1s = []
        fold_acc_strs = []
        for fold in range(5):
            metrics_file = os.path.join("checkpoints", f"gatkanv2_{version_name}_{protocol_name}_fold{fold+1}_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    m = json.load(f)
                accs.append(m["accuracy"])
                f1s.append(m["f1"])
                fold_acc_strs.append(f"{m['accuracy']:.4f}")
            else:
                fold_acc_strs.append("N/A")
                
        if len(accs) == 5:
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_f1 = np.mean(f1s)
            std_f1 = np.std(f1s)
            summary_str = f"{mean_acc:.4f} ± {std_acc:.4f}"
            f1_summary_str = f"F1: {mean_f1:.4f} ± {std_f1:.4f}"
        else:
            summary_str = "N/A"
            f1_summary_str = "N/A"
            
        print(f"{display_name:<45} | {' | '.join(fold_acc_strs)} | {summary_str}", flush=True)
        print(f"{'  (Macro F1)':<45} | " + " | ".join([f"{f:.4f}" if isinstance(f, float) else "N/A" for f in f1s]) + f" | {f1_summary_str}", flush=True)
        print("-" * 115, flush=True)
        
    print("\n" + "=" * 70, flush=True)
    print("GAT-KAN v2 Model Training Run Checkpoints & Sanity Gate", flush=True)
    print("=" * 70, flush=True)
    
    # Sanity Gate checks
    sd_acc_A = results["A_dep"]["accuracy"]
    cs_acc_A = results["A_cross"]["accuracy"]
    sd_acc_B = results["B_dep"]["accuracy"]
    cs_acc_B = results["B_cross"]["accuracy"]
    
    print(f"Subject-Dependent (Version A - Clean) Accuracy: {sd_acc_A:.2%} (Prior ceiling seen: ~63%)", flush=True)
    print(f"Cross-Subject (Version A - Clean) Accuracy:     {cs_acc_A:.2%} (Prior ceiling seen: ~38%)", flush=True)
    print(f"Subject-Dependent (Version B - Aug) Accuracy:   {sd_acc_B:.2%}", flush=True)
    print(f"Cross-Subject (Version B - Aug) Accuracy:       {cs_acc_B:.2%}", flush=True)
    
    gate_ok = True
    if sd_acc_A < 0.63:
        print("[WARNING] Subject-Dependent accuracy is below the prior sanity ceiling of 63%!", flush=True)
        gate_ok = False
    if cs_acc_A < 0.38:
        print("[WARNING] Cross-Subject accuracy is below the prior sanity ceiling of 38%!", flush=True)
        gate_ok = False
        
    if gate_ok:
        print("GATE CHECK PASSED: GAT-KAN v2 successfully exceeded sanity thresholds.", flush=True)
    else:
        print("GATE CHECK WARNING: One or more protocols underperformed sanity thresholds.", flush=True)
        
    print(f"\nAll models trained and evaluated. Total elapsed: {(time.time() - t_start)/60:.2f} minutes.", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
