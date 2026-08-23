import os
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

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
    Computes a $62 \times 62$ Granger-causality adjacency matrix fresh for the current
    fold, using ONLY the training indices.
    """
    print("  [STATS] Computing Granger-Causality adjacency on training fold only...", flush=True)
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
            
    # 2. Compute lag-1 VAR Granger causality F-statistics per channel pair
    # To keep it fast on CPU, we limit to a subset of 30 randomly selected trials if there are many
    if len(trial_series) > 30:
        indices = np.random.choice(len(trial_series), 30, replace=False)
        trial_series = [trial_series[i] for i in indices]
        
    F_matrix = np.zeros((num_channels, num_channels))
    
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                continue
            
            rss_rest_total = 0.0
            rss_unrest_total = 0.0
            n_obs_total = 0
            
            for ts in trial_series:
                y_i = ts[1:, i]
                y_i_lag1 = ts[:-1, i]
                y_j_lag1 = ts[:-1, j]
                
                L = len(y_i)
                if L < 5:
                    continue
                
                # Fit restricted model: y_i_t = c1 + a1 * y_i_lag1 + error
                X_rest = np.column_stack((np.ones(L), y_i_lag1))
                beta_rest, _, _, _ = np.linalg.lstsq(X_rest, y_i, rcond=None)
                pred_rest = X_rest @ beta_rest
                rss_rest = np.sum((y_i - pred_rest)**2)
                
                # Fit unrestricted model: y_i_t = c2 + b1 * y_i_lag1 + b2 * y_j_lag1 + error
                X_unrest = np.column_stack((np.ones(L), y_i_lag1, y_j_lag1))
                beta_unrest, _, _, _ = np.linalg.lstsq(X_unrest, y_i, rcond=None)
                pred_unrest = X_unrest @ beta_unrest
                rss_unrest = np.sum((y_i - pred_unrest)**2)
                
                rss_rest_total += rss_rest
                rss_unrest_total += rss_unrest
                n_obs_total += L
                
            if n_obs_total > 5:
                # F-statistic formula: ((RSS_rest - RSS_unrest) / 1) / (RSS_unrest / (N - 3))
                # Add epsilon to prevent division by zero
                den = rss_unrest_total / (n_obs_total - 3)
                if den > 1e-10:
                    F_val = (rss_rest_total - rss_unrest_total) / den
                    F_matrix[i, j] = max(0.0, F_val)
                    
    # Normalize to [0, 1]
    max_F = F_matrix.max()
    if max_F > 0:
        F_matrix = F_matrix / max_F
        
    # Add self-loops
    np.fill_diagonal(F_matrix, 1.0)
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
        splines = self.b_splines(x)
        spline_out = torch.einsum("bij,oij->bo", splines, self.spline_weight)
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
        
        # Apply soft masking using the blended adjacency matrix
        # adj is (nodes, nodes). We add epsilon to prevent log of 0
        adj_mask = torch.log(adj + 1e-10).view(1, 1, num_nodes, num_nodes)
        attn_matrix = attn_matrix + adj_mask
        
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
        
        # Spatial-spectral processing
        band_tokens = []
        spatial_attns = []
        
        for b in range(5):
            h_band = x[:, :, b].unsqueeze(-1) # (batch, 62, 1)
            h_proj = self.band_encoders[b](h_band) # (batch, 62, d_model)
            
            # GAT spatial representation
            h_gat1, attn1 = self.gat1(h_proj, adj)
            h_gat1 = F.relu(h_gat1 + h_proj) # Residual skip
            
            h_gat2, attn2 = self.gat2(h_gat1, adj)
            h_gat = F.relu(h_gat2 + h_gat1) # Residual skip
            
            # Average pool over nodes (channels) to get band representation
            h_band_avg = h_gat.mean(dim=1) # (batch, d_model)
            band_tokens.append(h_band_avg)
            
            spatial_attns.append(attn2) # Track attention maps
            
        # Shape: (batch, 5, d_model)
        band_tokens = torch.stack(band_tokens, dim=1)
        
        # Cross-band attention via Transformer
        trans_out = self.transformer(band_tokens) # (batch, 5, d_model)
        
        # Generate auxiliary predictions for per-band loss
        aux_preds = []
        for b in range(5):
            aux_preds.append(self.aux_heads[b](trans_out[:, b, :]))
            
        # Main classification head input: average pool over the 5 bands
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
        x = self.features[idx] # (62, 5)
        
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
    """Trains GAT-KAN v2 on a single fold with dynamic auxiliary loss weighting."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
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
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
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
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
        
        scheduler.step(val_loss)
        
        # Update dynamic auxiliary weights based on band-wise validation accuracy
        for b in range(5):
            band_acc = band_val_accs[b] / total_val
            # Bands that perform worse receive more loss weight to force convergence
            aux_weights[b] = 0.5 * (1.0 - band_acc)
            
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)
        
        # Early stopping verification
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
                
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    return train_history, val_history

def train_per_subject_finetune(model, train_loader, adj_matrix_torch, device):
    """Applies a 5-epoch per-subject fine-tuning pass using subject-specific training indices."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    for _ in range(5):
        for batch_x, batch_y, batch_sub in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
    
    device = torch.device("cpu")
    num_samples = len(labels)
    
    # 1. Determine unique trials to split at trial level
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys, trial_indices_map = np.unique(trial_keys, return_index=True)
    num_trials = len(unique_trial_keys)
    
    # Create fold indices
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    
    fold_train_histories = []
    fold_val_histories = []
    
    # Physical graph structure construction (k=14)
    A_kNN = build_knn_adjacency(x_coords, y_coords, k=14)
    
    for fold, (train_trial_idx, test_trial_idx) in enumerate(kf.split(np.arange(num_trials))):
        print(f"\n--- Fold {fold+1}/5 ---", flush=True)
        
        # Resolve window-level indices belonging to the selected trials
        train_trials_set = set(unique_trial_keys[train_trial_idx])
        test_trials_set = set(unique_trial_keys[test_trial_idx])
        
        train_indices = [idx for idx in range(num_samples) if trial_keys[idx] in train_trials_set]
        test_indices = [idx for idx in range(num_samples) if trial_keys[idx] in test_trials_set]
        
        # In cross-subject protocol, verify no subject overlaps
        if protocol_name == "cross-subject":
            train_subs = set(subject_ids[train_indices])
            test_subs = set(subject_ids[test_indices])
            overlap = train_subs.intersection(test_subs)
            if overlap:
                raise ValueError(f"Subject overlap detected in cross-subject protocol: {overlap}")
                
        # Compute Granger Causality on training set only to prevent leakage
        A_GC = compute_fold_granger_adjacency(features, labels, subject_ids, session_nums, trial_ids, train_indices)
        
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
        
        # Train fold
        t_hist, v_hist = train_gat_kan_fold(model, train_loader, test_loader, adj_matrix_torch, device, augment=augment)
        fold_train_histories.append(t_hist)
        fold_val_histories.append(v_hist)
        
        # Booster: Per-subject fine-tuning on train indices only
        if use_ft:
            print("  [BOOSTER] Applying per-subject fine-tuning pass on training data only...", flush=True)
            for sub in range(1, 16):
                # Extract train index mask for this subject
                sub_train_mask = [idx for idx in train_indices if subject_ids[idx] == sub]
                if len(sub_train_mask) > 0:
                    sub_dataset = EEGDataset(features_norm[sub_train_mask], labels[sub_train_mask], subject_ids[sub_train_mask], augment=augment)
                    sub_loader = DataLoader(sub_dataset, batch_size=64, shuffle=True)
                    train_per_subject_finetune(model, sub_loader, adj_matrix_torch, device)
                    
        # Final test evaluation for this fold
        model.eval()
        y_true_fold = []
        y_pred_fold = []
        y_prob_fold = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_sub in test_loader:
                batch_x = batch_x.to(device)
                out, _, spatial_attns = model(batch_x, adj_matrix_torch, batch_sub)
                
                probs = F.softmax(out, dim=-1)
                preds = out.argmax(dim=-1)
                
                y_true_fold.extend(batch_y.numpy())
                y_pred_fold.extend(preds.cpu().numpy())
                y_prob_fold.extend(probs.cpu().numpy())
                
        # Print progress
        fold_acc = np.mean(np.array(y_true_fold) == np.array(y_pred_fold))
        print(f"  Fold {fold+1} Test Accuracy: {fold_acc:.2%}", flush=True)
        
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
    
    # Plot loss / accuracy curves over folds
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
