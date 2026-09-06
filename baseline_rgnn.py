import os
import time
import copy
import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    cohen_kappa_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

# =========================================================================
# HELPER FUNCTIONS & PHYSICAL DISTANCE GRAPH PRIOR
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
    """Builds physical k-NN adjacency matrix with Gaussian distance weights."""
    coords = np.column_stack((x_coords, y_coords))
    num_nodes = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    
    A = np.zeros((num_nodes, num_nodes))
    sigma = np.mean(dists)
    for i in range(num_nodes):
        sorted_indices = np.argsort(dists[i])
        knn_indices = sorted_indices[1 : k+1]
        A[i, knn_indices] = np.exp(-dists[i, knn_indices]**2 / (2 * sigma**2))
        A[i, i] = 1.0 # Self-loop
    return A

def build_emotion_dl_matrix(sigma=0.5):
    """Builds EmotionDL soft target distribution matrix from continuous Valence-Arousal coordinates."""
    coords = np.array([
        [0.0, 0.0],   # 0: Neutral
        [-1.0, -1.0], # 1: Sad
        [-1.0, 1.0],  # 2: Fear
        [1.0, 1.0]    # 3: Happy
    ])
    dist_sq = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1)
    prob_matrix = np.exp(-dist_sq / (2.0 * sigma**2))
    prob_matrix = prob_matrix / np.sum(prob_matrix, axis=1, keepdims=True)
    return torch.tensor(prob_matrix, dtype=torch.float32)

# =========================================================================
# GRADIENT REVERSAL LAYER & RGNN ARCHITECTURE
# =========================================================================

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
        
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(self, x, adj):
        # x: (B, N, F_in), adj: (N, N)
        # 1. Feature transformation: (B, N, F_in) @ (F_in, F_out) -> (B, N, F_out)
        support = torch.matmul(x, self.weight)
        # 2. Graph neighborhood aggregation: (N, N) @ (B, N, F_out) -> (B, N, F_out)
        out = torch.matmul(adj, support)
        if self.bias is not None:
            out = out + self.bias
        return out

class RGNN(nn.Module):
    def __init__(self, num_nodes=62, in_channels=5, hidden_dim1=64, hidden_dim2=64, n_classes=4, n_domains=10, dropout=0.2, init_adj=None):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Learnable Adjacency Matrix (initialized from physical k-NN distance prior)
        if init_adj is not None:
            self.raw_adj = nn.Parameter(torch.tensor(init_adj, dtype=torch.float32))
        else:
            self.raw_adj = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
            nn.init.xavier_uniform_(self.raw_adj)
            
        # 2. Graph Convolution Layers
        self.gc1 = GraphConvolution(in_channels, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout = nn.Dropout(dropout)
        
        # 3. Emotion Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
        
        # 4. NodeDAT Domain Discriminator Head
        self.grl = GradientReversal(alpha=1.0)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_dim2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_domains)
        )
        
    def get_normalized_adj(self):
        # Symmetrize and apply ReLU for non-negative weights
        adj = F.relu(self.raw_adj)
        adj = (adj + adj.t()) / 2.0
        # Add self-loops
        adj = adj + torch.eye(self.num_nodes, device=adj.device)
        # Degree normalization: D^{-1/2} A D^{-1/2}
        d = torch.sum(adj, dim=1)
        d_inv_sqrt = torch.pow(torch.clamp(d, min=1e-6), -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return norm_adj
        
    def forward(self, x, alpha=1.0):
        # x: (B, 62, 5)
        norm_adj = self.get_normalized_adj()
        
        # Layer 1: Graph neighborhood aggregation + Feature transformation
        h1 = self.gc1(x, norm_adj)
        h1 = self.bn1(h1.transpose(1, 2)).transpose(1, 2)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        
        # Layer 2: Graph neighborhood aggregation + Feature transformation
        h2 = self.gc2(h1, norm_adj)
        h2 = self.bn2(h2.transpose(1, 2)).transpose(1, 2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        
        # Global graph pooling for emotion prediction
        graph_emb = torch.mean(h2, dim=1)
        class_out = self.classifier(graph_emb)
        
        # NodeDAT: Node-wise domain classification with GRL
        self.grl.alpha = alpha
        node_emb_rev = self.grl(h2)
        domain_feat = torch.mean(node_emb_rev, dim=1)
        domain_out = self.domain_discriminator(domain_feat)
        
        return class_out, domain_out, norm_adj

# =========================================================================
# EVALUATION & METRICS HELPERS
# =========================================================================

def compute_bootstrap_confidence_intervals(y_true, y_pred, y_prob, num_resamples=1000):
    """Computes 95% confidence intervals using bootstrap resampling."""
    accs, precs, recs, f1s, aucs, kappas = [], [], [], [], [], []
    num_samples = len(y_true)
    
    for _ in range(num_resamples):
        indices = np.random.choice(num_samples, num_samples, replace=True)
        y_t_res = y_true[indices]
        y_p_res = y_pred[indices]
        y_prob_res = y_prob[indices]
        
        accs.append(np.mean(y_t_res == y_p_res))
        
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
            
    ci_acc = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    ci_prec = (float(np.percentile(precs, 2.5)), float(np.percentile(precs, 97.5)))
    ci_rec = (float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5)))
    ci_f1 = (float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5)))
    ci_auc = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    ci_kappa = (float(np.percentile(kappas, 2.5)), float(np.percentile(kappas, 97.5)))
    
    return ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa

def save_evaluation_plots(y_true, y_pred, y_prob, prefix, figures_dir):
    """Generates and saves Confusion Matrix, ROC, PR, and Calibration diagrams (300 DPI)."""
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Confusion Matrix (Raw + Row-Normalized)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(label='Normalized Ratio')
    for i in range(4):
        for j in range(4):
            txt = f"{cm[i,j]}\n({cm_norm[i,j]:.1%})"
            plt.text(j, i, txt, ha='center', va='center', fontsize=9,
                     color='white' if cm_norm[i,j] > 0.5 else 'black')
    plt.xticks(range(4), CLASS_NAMES)
    plt.yticks(range(4), CLASS_NAMES)
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    plt.title(f'Confusion Matrix - {prefix}', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 2. ROC Curves per Class
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{CLASS_NAMES[c]} (AUC={roc_auc:.3f})', color=EMOTION_COLORS[c], lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'ROC Curves - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_roc.png"), dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curves
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, c])
        plt.plot(recall, precision, label=f'{CLASS_NAMES[c]}', color=EMOTION_COLORS[c], lw=2)
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title(f'Precision-Recall Curves - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='lower left', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_pr.png"), dpi=300)
    plt.close()
    
    # 4. Calibration Diagram
    plt.figure(figsize=(6, 5))
    for c in range(4):
        y_true_binary = (y_true == c).astype(int)
        prob_true, prob_pred = calibration_curve(y_true_binary, y_prob[:, c], n_bins=5)
        plt.plot(prob_pred, prob_true, marker='o', label=f'{CLASS_NAMES[c]}', color=EMOTION_COLORS[c], lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Mean Predicted Probability', fontweight='bold')
    plt.ylabel('Fraction of Positives', fontweight='bold')
    plt.title(f'Calibration Diagram - {prefix}', fontsize=10, fontweight='bold')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"{prefix}_calibration.png"), dpi=300)
    plt.close()

# =========================================================================
# SINGLE-FOLD TRAINING HELPER (CALIBRATED OPTION B)
# =========================================================================

def train_and_evaluate_fold(X_train_raw, y_train, d_train, X_val_raw, y_val, X_test_raw, y_test, 
                            n_domains, A_init, emotion_dl_matrix, device, 
                            w_dom=0.2, sparsity_beta=1e-4, max_epochs=100, patience=15, batch_size=128):
    """
    Trains RGNN (Option B: sigma=0.5, w_dom=0.2, beta=1e-4) with early stopping on validation emotion accuracy.
    Validation and test domain targets are never used during training (strict inductive quarantine).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw.reshape(len(X_train_raw), -1)).reshape(-1, 62, 5)
    X_val_scaled = scaler.transform(X_val_raw.reshape(len(X_val_raw), -1)).reshape(-1, 62, 5)
    X_test_scaled = scaler.transform(X_test_raw.reshape(len(X_test_raw), -1)).reshape(-1, 62, 5)
    
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                      torch.tensor(y_train, dtype=torch.long),
                      torch.tensor(d_train, dtype=torch.long)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), 
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    
    model = RGNN(num_nodes=62, in_channels=5, hidden_dim1=64, hidden_dim2=64, n_classes=4, n_domains=n_domains, dropout=0.2, init_adj=A_init).to(device)
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    
    best_val_acc = -1.0
    best_epoch = -1
    best_weights = None
    epochs_no_improve = 0
    epoch_times = []
    
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        
        # Ganin dynamic lambda
        p = float(epoch - 1) / max_epochs
        lambda_p = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        
        # Training
        model.train()
        train_emo_loss, train_dom_loss, train_sparse_loss = 0.0, 0.0, 0.0
        train_emo_correct, train_dom_correct = 0, 0
        train_total = 0
        
        for bx, by, bd in train_loader:
            bx, by, bd = bx.to(device), by.to(device), bd.to(device)
            optimizer.zero_grad()
            
            c_out, d_out, norm_adj = model(bx, alpha=lambda_p)
            
            # EmotionDL soft loss
            targets_soft = emotion_dl_matrix[by]
            loss_emo = torch.sum(-targets_soft * F.log_softmax(c_out, dim=1), dim=1).mean()
            
            # NodeDAT loss
            loss_dom = criterion_domain(d_out, bd)
            
            # L1 sparsity penalty on off-diagonal entries
            eye = torch.eye(62, device=device)
            loss_sparse = sparsity_beta * torch.sum(torch.abs(model.raw_adj * (1.0 - eye))) / (62 * 61)
            
            total_loss = loss_emo + w_dom * loss_dom + loss_sparse
            total_loss.backward()
            optimizer.step()
            
            n_b = len(by)
            train_total += n_b
            train_emo_loss += loss_emo.item() * n_b
            train_dom_loss += loss_dom.item() * n_b
            train_sparse_loss += loss_sparse.item() * n_b
            train_emo_correct += (c_out.argmax(dim=1) == by).sum().item()
            train_dom_correct += (d_out.argmax(dim=1) == bd).sum().item()
            
        scheduler.step()
        
        tr_e_l = train_emo_loss / train_total
        tr_e_acc = train_emo_correct / train_total
        tr_d_l = train_dom_loss / train_total
        tr_d_acc = train_dom_correct / train_total
        tr_s_l = train_sparse_loss / train_total
        
        # Validation
        model.eval()
        val_emo_loss, val_emo_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                c_out, _, _ = model(bx, alpha=0.0)
                targets_soft = emotion_dl_matrix[by]
                loss_emo = torch.sum(-targets_soft * F.log_softmax(c_out, dim=1), dim=1).mean()
                val_emo_loss += loss_emo.item() * len(by)
                val_emo_correct += (c_out.argmax(dim=1) == by).sum().item()
                val_total += len(by)
                
        val_e_l = val_emo_loss / val_total
        val_e_acc = val_emo_correct / val_total
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)
        
        if val_e_acc > best_val_acc:
            best_val_acc = val_e_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            mark = '*'
        else:
            epochs_no_improve += 1
            mark = ''
            
        if epoch % 5 == 0 or epoch == 1 or mark == '*':
            tr_e_str = f"{tr_e_l:.3f}/{tr_e_acc*100:.1f}%"
            tr_d_str = f"{tr_d_l:.3f}/{tr_d_acc*100:.1f}%"
            val_e_str = f"{val_e_l:.3f}/{val_e_acc*100:.1f}% {mark}"
            print(f"    Epoch [{epoch:03d}/{max_epochs:03d}] ({t_epoch:.2f}s) | Lam: {lambda_p:.3f} | Emo: {tr_e_str} | Dom: {tr_d_str} | Spars: {tr_s_l:.5f} | Val: {val_e_str}", flush=True)
            
        if epochs_no_improve >= patience:
            print(f"    --> Early stopped at epoch {epoch} (Best Val Acc: {best_val_acc*100:.2f}% at epoch {best_epoch})", flush=True)
            break
            
    # Evaluate best model on test set
    model.load_state_dict(best_weights)
    model.eval()
    
    test_preds, test_probs, test_targets = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            c_out, _, _ = model(bx, alpha=0.0)
            probs = torch.softmax(c_out, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_targets.extend(by.numpy())
            
    y_test_arr = np.array(test_targets)
    y_pred_arr = np.array(test_preds)
    y_prob_arr = np.array(test_probs)
    
    acc = np.mean(y_test_arr == y_pred_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test_arr, y_pred_arr, average='macro', zero_division=0)
    auc_val = roc_auc_score(y_test_arr, y_prob_arr, average='macro', multi_class='ovr')
    kappa_val = cohen_kappa_score(y_test_arr, y_pred_arr)
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc_val),
        'kappa': float(kappa_val),
        'best_epoch': int(best_epoch),
        'best_val_acc': float(best_val_acc),
        'avg_epoch_time': float(np.mean(epoch_times)),
        'y_true': y_test_arr,
        'y_pred': y_pred_arr,
        'y_prob': y_prob_arr
    }

# =========================================================================
# PROTOCOL RUNNERS
# =========================================================================

def run_cross_subject(features, labels, subject_ids, A_init, emotion_dl_matrix, device, figures_dir):
    """Cross-subject protocol: 5-fold Leave-3-Subjects-Out nested CV."""
    print("=" * 80, flush=True)
    print("RUNNING RGNN (CALIBRATED OPTION B): Cross-Subject Protocol (Leave-3-Subjects-Out)", flush=True)
    print("=" * 80, flush=True)
    
    subject_folds = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        [13, 14, 15]
    ]
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    fold_metrics_list = []
    
    for fold, test_subs in enumerate(subject_folds):
        t_fold_start = time.time()
        train_pool_subs = [s for s in range(1, 16) if s not in test_subs]
        
        val_subs = train_pool_subs[-2:]
        train_subs = train_pool_subs[:-2] # 10 training subjects = 10 source domains
        
        print(f"\n--- Outer Fold {fold+1}/5 (Test Subjects: {test_subs}, Val Subjects: {val_subs}, Train Subjects: {train_subs}) ---", flush=True)
        
        assert len(set(train_subs).intersection(set(val_subs))) == 0
        assert len(set(train_subs).intersection(set(test_subs))) == 0
        assert len(set(val_subs).intersection(set(test_subs))) == 0
        
        train_mask = np.isin(subject_ids, train_subs)
        val_mask = np.isin(subject_ids, val_subs)
        test_mask = np.isin(subject_ids, test_subs)
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        s_train = subject_ids[train_mask]
        
        sub_to_domain = {sub: idx for idx, sub in enumerate(train_subs)}
        d_train = np.array([sub_to_domain[s] for s in s_train])
        
        X_val = features[val_mask]
        y_val = labels[val_mask]
        
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        print(f"  Fold {fold+1}: Train={len(y_train)} samples ({train_subs}), Val={len(y_val)} samples ({val_subs}), Test={len(y_test)} samples ({test_subs}).", flush=True)
        
        res = train_and_evaluate_fold(X_train, y_train, d_train, X_val, y_val, X_test, y_test, 
                                      len(train_subs), A_init, emotion_dl_matrix, device, w_dom=0.2)
        fold_time = time.time() - t_fold_start
        
        print(f"  >>> Fold {fold+1} Complete ({fold_time:.1f}s) | Test Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | AUC: {res['auc']:.4f} | Kappa: {res['kappa']:.4f}", flush=True)
        
        y_true_all.extend(res['y_true'])
        y_pred_all.extend(res['y_pred'])
        y_prob_all.extend(res['y_prob'])
        
        fold_metrics_list.append({
            'fold': fold + 1,
            'test_subjects': test_subs,
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
            'kappa': res['kappa'],
            'best_epoch': res['best_epoch'],
            'best_val_acc': res['best_val_acc'],
            'fold_time_sec': fold_time
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    prefix = "rgnn_cross_subject"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Save out-of-fold sample predictions for verified downstream statistical testing
    np.savez_compressed(
        "oof_preds_rgnn_cross_subject.npz",
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all
    )
    
    results = {
        "protocol": "cross-subject",
        "accuracy": overall_acc,
        "accuracy_ci": ci_acc,
        "precision": float(overall_prec),
        "precision_ci": ci_prec,
        "recall": float(overall_rec),
        "recall_ci": ci_rec,
        "f1": float(overall_f1),
        "f1_ci": ci_f1,
        "auc": overall_auc,
        "auc_ci": ci_auc,
        "kappa": overall_kappa,
        "kappa_ci": ci_kappa,
        "fold_metrics": fold_metrics_list
    }
    
    print(f"\n>>> [CROSS-SUBJECT COMPLETE] Acc: {overall_acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]) | Macro-F1: {overall_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]) | Kappa: {overall_kappa:.4f}\n", flush=True)
    return results

def run_subject_dependent(features, labels, subject_ids, session_nums, trial_ids, A_init, emotion_dl_matrix, device, figures_dir):
    """Subject-dependent protocol: pooled data, 5-fold nested CV grouped by trial."""
    print("=" * 80, flush=True)
    print("RUNNING RGNN (CALIBRATED OPTION B): Subject-Dependent Protocol (Pooled, 5-Fold Trial-Grouped CV)", flush=True)
    print("=" * 80, flush=True)
    
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys = np.unique(trial_keys)
    num_trials = len(unique_trial_keys)
    
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_true_all = []
    y_pred_all = []
    y_prob_all = []
    fold_metrics_list = []
    
    for fold, (train_trial_idx, test_trial_idx) in enumerate(kf_outer.split(np.arange(num_trials))):
        t_fold_start = time.time()
        print(f"\n--- Outer Fold {fold+1}/5 ---", flush=True)
        
        outer_train_trials = unique_trial_keys[train_trial_idx]
        outer_test_trials = unique_trial_keys[test_trial_idx]
        
        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(kf_inner.split(np.arange(len(outer_train_trials))))
        
        train_trials_set = set(outer_train_trials[inner_train_idx])
        val_trials_set = set(outer_train_trials[inner_val_idx])
        test_trials_set = set(outer_test_trials)
        
        assert len(train_trials_set.intersection(val_trials_set)) == 0
        assert len(train_trials_set.intersection(test_trials_set)) == 0
        assert len(val_trials_set.intersection(test_trials_set)) == 0
        
        train_mask = np.isin(trial_keys, list(train_trials_set))
        val_mask = np.isin(trial_keys, list(val_trials_set))
        test_mask = np.isin(trial_keys, list(test_trials_set))
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        s_train = subject_ids[train_mask]
        d_train = s_train - 1 # 15 domains 0..14
        
        X_val = features[val_mask]
        y_val = labels[val_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        print(f"  Fold {fold+1}: Train={len(y_train)} samples, Val={len(y_val)} samples, Test={len(y_test)} samples.", flush=True)
        
        res = train_and_evaluate_fold(X_train, y_train, d_train, X_val, y_val, X_test, y_test, 
                                      15, A_init, emotion_dl_matrix, device, w_dom=0.2)
        fold_time = time.time() - t_fold_start
        
        print(f"  >>> Fold {fold+1} Complete ({fold_time:.1f}s) | Test Acc: {res['accuracy']:.4f} | F1: {res['f1']:.4f} | AUC: {res['auc']:.4f} | Kappa: {res['kappa']:.4f}", flush=True)
        
        y_true_all.extend(res['y_true'])
        y_pred_all.extend(res['y_pred'])
        y_prob_all.extend(res['y_prob'])
        
        fold_metrics_list.append({
            'fold': fold + 1,
            'accuracy': res['accuracy'],
            'precision': res['precision'],
            'recall': res['recall'],
            'f1': res['f1'],
            'auc': res['auc'],
            'kappa': res['kappa'],
            'best_epoch': res['best_epoch'],
            'best_val_acc': res['best_val_acc'],
            'fold_time_sec': fold_time
        })
        
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)
    
    overall_acc = float(np.mean(y_true_all == y_pred_all))
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro', zero_division=0)
    overall_auc = float(roc_auc_score(y_true_all, y_prob_all, average='macro', multi_class='ovr'))
    overall_kappa = float(cohen_kappa_score(y_true_all, y_pred_all))
    
    print("\nComputing 95% bootstrap confidence intervals (1,000 resamples)...", flush=True)
    ci_acc, ci_prec, ci_rec, ci_f1, ci_auc, ci_kappa = compute_bootstrap_confidence_intervals(y_true_all, y_pred_all, y_prob_all, 1000)
    
    prefix = "rgnn_subject_dependent"
    save_evaluation_plots(y_true_all, y_pred_all, y_prob_all, prefix, figures_dir)
    
    # Save out-of-fold sample predictions for verified downstream statistical testing
    np.savez_compressed(
        "oof_preds_rgnn_subject_dependent.npz",
        y_true=y_true_all,
        y_pred=y_pred_all,
        y_prob=y_prob_all
    )
    
    results = {
        "protocol": "subject-dependent",
        "accuracy": overall_acc,
        "accuracy_ci": ci_acc,
        "precision": float(overall_prec),
        "precision_ci": ci_prec,
        "recall": float(overall_rec),
        "recall_ci": ci_rec,
        "f1": float(overall_f1),
        "f1_ci": ci_f1,
        "auc": overall_auc,
        "auc_ci": ci_auc,
        "kappa": overall_kappa,
        "kappa_ci": ci_kappa,
        "fold_metrics": fold_metrics_list
    }
    
    print(f"\n>>> [SUBJECT-DEPENDENT COMPLETE] Acc: {overall_acc:.4f} (95% CI: [{ci_acc[0]:.4f}, {ci_acc[1]:.4f}]) | Macro-F1: {overall_f1:.4f} (95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]) | Kappa: {overall_kappa:.4f}\n", flush=True)
    return results

# =========================================================================
# MAIN PIPELINE ENTRYPOINT
# =========================================================================

def main():
    t_start = time.time()
    figures_dir = os.path.join("figures", "baselines")
    os.makedirs(figures_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("REGULARIZED GRAPH NEURAL NETWORK (RGNN) BASELINE (GPU)", flush=True)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("Config: EmotionDL sigma=0.5, NodeDAT w_dom=0.2, L1 Sparsity beta=1e-4", flush=True)
    print("=" * 80, flush=True)
    
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found at '{dataset_path}'")
        
    print(f"Loading SEED-IV dataset from '{dataset_path}'...", flush=True)
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    num_samples = len(labels)
    
    print(f"Dataset loaded: {num_samples} samples across 15 subjects.", flush=True)
    
    # Load physical electrode coordinates and construct physical distance prior
    _, x_c, y_c = parse_locs("channel_62_pos (1).locs")
    A_init = build_knn_adjacency(x_c, y_c, k=14)
    print(f"Physical distance prior adjacency constructed (62x62, density: {(A_init > 0).mean()*100:.1f}%)", flush=True)
    
    emotion_dl_matrix = build_emotion_dl_matrix(sigma=0.5).to(device)
    
    # 1. Run Cross-Subject Protocol
    res_cross = run_cross_subject(features, labels, subject_ids, A_init, emotion_dl_matrix, device, figures_dir)
    
    # 2. Run Subject-Dependent Protocol
    res_dep = run_subject_dependent(features, labels, subject_ids, session_nums, trial_ids, A_init, emotion_dl_matrix, device, figures_dir)
    
    # 3. Save structured results (JSON & CSV)
    all_results = {
        "cross_subject": res_cross,
        "subject_dependent": res_dep
    }
    
    json_path = "rgnn_baseline_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Saved structured JSON results to '{json_path}'", flush=True)
    
    csv_path = "rgnn_baseline_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Protocol", "Accuracy", "Accuracy_95CI_Lower", "Accuracy_95CI_Upper", 
            "Macro_Precision", "Macro_Precision_95CI_Lower", "Macro_Precision_95CI_Upper",
            "Macro_Recall", "Macro_Recall_95CI_Lower", "Macro_Recall_95CI_Upper",
            "Macro_F1", "Macro_F1_95CI_Lower", "Macro_F1_95CI_Upper",
            "Macro_AUC", "Macro_AUC_95CI_Lower", "Macro_AUC_95CI_Upper",
            "Cohen_Kappa", "Cohen_Kappa_95CI_Lower", "Cohen_Kappa_95CI_Upper"
        ])
        for p_key, r in [("cross_subject", res_cross), ("subject_dependent", res_dep)]:
            writer.writerow([
                r["protocol"],
                f"{r['accuracy']:.4f}", f"{r['accuracy_ci'][0]:.4f}", f"{r['accuracy_ci'][1]:.4f}",
                f"{r['precision']:.4f}", f"{r['precision_ci'][0]:.4f}", f"{r['precision_ci'][1]:.4f}",
                f"{r['recall']:.4f}", f"{r['recall_ci'][0]:.4f}", f"{r['recall_ci'][1]:.4f}",
                f"{r['f1']:.4f}", f"{r['f1_ci'][0]:.4f}", f"{r['f1_ci'][1]:.4f}",
                f"{r['auc']:.4f}", f"{r['auc_ci'][0]:.4f}", f"{r['auc_ci'][1]:.4f}",
                f"{r['kappa']:.4f}", f"{r['kappa_ci'][0]:.4f}", f"{r['kappa_ci'][1]:.4f}"
            ])
    print(f"Saved structured CSV results to '{csv_path}'", flush=True)
    
    # 4. Print Final Summary Table
    print("\n" + "=" * 80, flush=True)
    print("FINAL SUMMARY: REGULARIZED GRAPH NEURAL NETWORK (RGNN) RESULTS", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Protocol':<20} | {'Accuracy (95% CI)':<25} | {'Macro-F1 (95% CI)':<25} | {'Cohen Kappa (95% CI)':<25}", flush=True)
    print("-" * 105, flush=True)
    for p_key, r in [("cross_subject", res_cross), ("subject_dependent", res_dep)]:
        acc_str = f"{r['accuracy']:.4f} [{r['accuracy_ci'][0]:.4f}, {r['accuracy_ci'][1]:.4f}]"
        f1_str = f"{r['f1']:.4f} [{r['f1_ci'][0]:.4f}, {r['f1_ci'][1]:.4f}]"
        kappa_str = f"{r['kappa']:.4f} [{r['kappa_ci'][0]:.4f}, {r['kappa_ci'][1]:.4f}]"
        print(f"{r['protocol']:<20} | {acc_str:<25} | {f1_str:<25} | {kappa_str:<25}", flush=True)
    print("=" * 80, flush=True)
    print(f"Total RGNN baseline pipeline runtime: {(time.time() - t_start)/60:.2f} minutes.", flush=True)

if __name__ == '__main__':
    main()
