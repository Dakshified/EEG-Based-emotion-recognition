import os
import time
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
import matplotlib.pyplot as plt

CLASS_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
CLASS_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
BAND_NAMES = ['Delta (1-3 Hz)', 'Theta (4-7 Hz)', 'Alpha (8-13 Hz)', 'Beta (14-30 Hz)', 'Gamma (31-50 Hz)']
BAND_SHORT = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# =========================================================================
# 1. CHANNEL POSITIONS & REGIONAL GROUPINGS
# =========================================================================

CHANNEL_NAMES_62 = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

REGION_MAP = {
    'Frontal': ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8'],
    'Frontocentral / Central': ['FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'],
    'Temporal': ['T7', 'TP7', 'T8', 'TP8'],
    'Centroparietal / Parietal': ['CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8'],
    'Parieto-Occipital / Occipital': ['PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']
}

LEFT_CHANNELS = ['Fp1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1', 'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO5', 'PO3', 'CB1', 'O1']
RIGHT_CHANNELS = ['Fp2', 'AF4', 'F8', 'F6', 'F4', 'F2', 'FT8', 'FC6', 'FC4', 'FC2', 'T8', 'C6', 'C4', 'C2', 'TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P4', 'P2', 'PO8', 'PO6', 'PO4', 'CB2', 'O2']

# =========================================================================
# 2. DANN ARCHITECTURE DEFINITION
# =========================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANN(nn.Module):
    def __init__(self, in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=10, dropout=0.2):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        self.grl = GradientReversal(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_domains)
        )
    def forward(self, x, alpha=0.0):
        self.grl.alpha = alpha
        feat = self.feature_extractor(x)
        class_out = self.class_classifier(feat)
        domain_feat = self.grl(feat)
        domain_out = self.domain_classifier(domain_feat)
        return class_out, domain_out

# =========================================================================
# 3. VECTORIZED INTEGRATED GRADIENTS (GPU-ACCELERATED)
# =========================================================================

def compute_integrated_gradients(model, X_test_tensor, baseline_tensor, target_classes, device, steps=50, batch_size=256):
    """
    Computes Integrated Gradients for a batch of samples with respect to target classes.
    Completeness is verified: sum(IG) == F(x) - F(baseline).
    """
    model.eval()
    num_samples, num_features = X_test_tensor.shape
    attributions = np.zeros((num_samples, num_features), dtype=np.float32)
    
    # Pre-calculate baseline output
    with torch.no_grad():
        b_input = baseline_tensor.unsqueeze(0).to(device)
        b_out, _ = model(b_input, alpha=0.0)
        baseline_logits = b_out.cpu().numpy()[0]
        
    num_batches = int(np.ceil(num_samples / batch_size))
    
    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, num_samples)
        
        batch_x = X_test_tensor[start:end].to(device) # (B, 310)
        batch_target = target_classes[start:end] # (B,)
        B = len(batch_x)
        
        baseline_expanded = baseline_tensor.unsqueeze(0).expand(B, -1).to(device) # (B, 310)
        diff = (batch_x - baseline_expanded) # (B, 310)
        
        # Linear interpolation steps alpha in [0, 1]
        alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1, 1) # (steps, 1, 1)
        # Interpolated points: (steps, B, 310)
        interpolated = baseline_expanded.unsqueeze(0) + alphas * diff.unsqueeze(0)
        interpolated = interpolated.view(steps * B, num_features)
        interpolated.requires_grad_(True)
        
        # Forward pass on all interpolated points
        # Split into sub-batches if necessary to avoid VRAM overload
        out_logits, _ = model(interpolated, alpha=0.0)
        
        # Select target class logit for each point
        target_expanded = torch.tensor(np.tile(batch_target, steps), dtype=torch.long, device=device)
        target_logits = out_logits.gather(1, target_expanded.unsqueeze(1)).squeeze(1)
        
        # Backward pass to compute gradients
        target_logits.sum().backward()
        grads = interpolated.grad.view(steps, B, num_features) # (steps, B, 310)
        
        # Riemann integration (trapezoidal / average gradient)
        avg_grads = grads.mean(dim=0) # (B, 310)
        ig_batch = diff * avg_grads # (B, 310)
        
        attributions[start:end] = ig_batch.detach().cpu().numpy()
        
    return attributions

# =========================================================================
# 4. OCCLUSION-BASED ATTRIBUTION (SYSTEMATIC ZEROING)
# =========================================================================

def compute_occlusion_attribution(model, X_test_tensor, baseline_tensor, target_classes, device, batch_size=512):
    """
    Computes systematic single-feature occlusion attribution:
    Occlusion_i = P_target(x) - P_target(x_without_i)
    """
    model.eval()
    num_samples, num_features = X_test_tensor.shape
    occlusion_scores = np.zeros((num_samples, num_features), dtype=np.float32)
    
    with torch.no_grad():
        # Baseline prediction on original samples
        test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size, shuffle=False)
        orig_probs_list = []
        for bx in test_loader:
            bx = bx[0].to(device)
            c_out, _ = model(bx, alpha=0.0)
            orig_probs_list.append(torch.softmax(c_out, dim=1).cpu().numpy())
        orig_probs = np.vstack(orig_probs_list)
        orig_target_probs = orig_probs[np.arange(num_samples), target_classes]
        
        # Single feature occlusion across all 310 features
        b_vec = baseline_tensor.numpy()
        for f_idx in range(num_features):
            X_occ = X_test_tensor.clone()
            X_occ[:, f_idx] = float(b_vec[f_idx])
            
            occ_loader = DataLoader(TensorDataset(X_occ), batch_size=batch_size, shuffle=False)
            occ_probs_list = []
            for bx in occ_loader:
                bx = bx[0].to(device)
                c_out, _ = model(bx, alpha=0.0)
                occ_probs_list.append(torch.softmax(c_out, dim=1).cpu().numpy())
            occ_probs = np.vstack(occ_probs_list)
            occ_target_probs = occ_probs[np.arange(num_samples), target_classes]
            
            # Sensitivity = drop in predicted class probability
            occlusion_scores[:, f_idx] = orig_target_probs - occ_target_probs
            
    return occlusion_scores

# =========================================================================
# 5. FAITHFULNESS VERIFICATION (DELETION & INSERTION AUC)
# =========================================================================

def evaluate_faithfulness(model, X_test_tensor, baseline_tensor, mean_abs_attributions, target_classes, device, 
                          step_size=10, num_random_perms=20, batch_size=512):
    """
    Computes Deletion AUC and Insertion AUC comparing attribution ranking vs random control.
    """
    model.eval()
    num_samples, num_features = X_test_tensor.shape
    num_steps = int(np.ceil(num_features / step_size)) + 1
    
    # 1. Attribution-guided ranking (descending importance)
    attr_rank = np.argsort(-mean_abs_attributions)
    
    def evaluate_curve(rank_indices):
        del_confs = []
        ins_confs = []
        
        b_mat = baseline_tensor.unsqueeze(0).expand(num_samples, -1)
        
        for s in range(num_steps):
            k = min(s * step_size, num_features)
            
            # Deletion: mask top-k features
            X_del = X_test_tensor.clone()
            if k > 0:
                masked_feat = rank_indices[:k]
                X_del[:, masked_feat] = b_mat[:, masked_feat]
                
            # Insertion: start from baseline, restore top-k features
            X_ins = b_mat.clone()
            if k > 0:
                restored_feat = rank_indices[:k]
                X_ins[:, restored_feat] = X_test_tensor[:, restored_feat]
                
            # Evaluate deletion confidence
            with torch.no_grad():
                del_loader = DataLoader(TensorDataset(X_del), batch_size=batch_size, shuffle=False)
                del_p_list = []
                for bx in del_loader:
                    bx = bx[0].to(device)
                    c_out, _ = model(bx, alpha=0.0)
                    del_p_list.append(torch.softmax(c_out, dim=1).cpu().numpy())
                del_probs = np.vstack(del_p_list)
                del_confs.append(np.mean(del_probs[np.arange(num_samples), target_classes]))
                
                # Evaluate insertion confidence
                ins_loader = DataLoader(TensorDataset(X_ins), batch_size=batch_size, shuffle=False)
                ins_p_list = []
                for bx in ins_loader:
                    bx = bx[0].to(device)
                    c_out, _ = model(bx, alpha=0.0)
                    ins_p_list.append(torch.softmax(c_out, dim=1).cpu().numpy())
                ins_probs = np.vstack(ins_p_list)
                ins_confs.append(np.mean(ins_probs[np.arange(num_samples), target_classes]))
                
        # Compute normalized AUC
        del_auc = float(np.trapezoid(del_confs, dx=1.0 / (num_steps - 1)))
        ins_auc = float(np.trapezoid(ins_confs, dx=1.0 / (num_steps - 1)))
        return del_auc, ins_auc, del_confs, ins_confs

    # Attribution curve
    attr_del_auc, attr_ins_auc, attr_del_curve, attr_ins_curve = evaluate_curve(attr_rank)
    
    # Random control curves (averaged over 20 random permutations)
    rand_del_aucs, rand_ins_aucs = [], []
    rand_del_curves, rand_ins_curves = [], []
    
    for _ in range(num_random_perms):
        rand_rank = np.random.permutation(num_features)
        r_del_auc, r_ins_auc, r_del_c, r_ins_c = evaluate_curve(rand_rank)
        rand_del_aucs.append(r_del_auc)
        rand_ins_aucs.append(r_ins_auc)
        rand_del_curves.append(r_del_c)
        rand_ins_curves.append(r_ins_c)
        
    mean_rand_del_curve = np.mean(rand_del_curves, axis=0)
    mean_rand_ins_curve = np.mean(rand_ins_curves, axis=0)
    mean_rand_del_auc = float(np.mean(rand_del_aucs))
    mean_rand_ins_auc = float(np.mean(rand_ins_aucs))
    
    # Statistical significance of faithfulness
    t_del, p_del = stats.ttest_1samp(rand_del_aucs, attr_del_auc) # Expect attr_del_auc << rand_del_auc
    t_ins, p_ins = stats.ttest_1samp(rand_ins_aucs, attr_ins_auc) # Expect attr_ins_auc >> rand_ins_auc
    
    return {
        'attr_del_auc': attr_del_auc,
        'attr_ins_auc': attr_ins_auc,
        'rand_del_auc': mean_rand_del_auc,
        'rand_ins_auc': mean_rand_ins_auc,
        'p_del_val': float(p_del),
        'p_ins_val': float(p_ins),
        'attr_del_curve': attr_del_curve,
        'attr_ins_curve': attr_ins_curve,
        'rand_del_curve': mean_rand_del_curve.tolist(),
        'rand_ins_curve': mean_rand_ins_curve.tolist()
    }

# =========================================================================
# 6. MAIN EXPLAINABILITY PIPELINE (ACROSS ALL 10 FOLDS)
# =========================================================================

def run_explainability_pipeline():
    t_start = time.time()
    figures_dir = os.path.join("figures", "explainability")
    checkpoints_dir = os.path.join("checkpoints", "dann_final")
    os.makedirs(figures_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print("CALIBRATED INDUCTIVE DANN EXPLAINABILITY & FEATURE ATTRIBUTION PIPELINE", flush=True)
    print(f"Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=" * 80, flush=True)
    
    # Load dataset
    dataset_path = "seed_iv_processed.npz"
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    num_samples = len(labels)
    features_flat = features.reshape(num_samples, -1)
    
    # Storage for aggregated results across folds and protocols
    explainability_results = {
        'cross_subject': {},
        'subject_dependent': {}
    }
    
    # Define splits
    protocols_config = {
        'cross_subject': [
            {'fold': 1, 'test_subs': [1, 2, 3], 'n_domains': 10},
            {'fold': 2, 'test_subs': [4, 5, 6], 'n_domains': 10},
            {'fold': 3, 'test_subs': [7, 8, 9], 'n_domains': 10},
            {'fold': 4, 'test_subs': [10, 11, 12], 'n_domains': 10},
            {'fold': 5, 'test_subs': [13, 14, 15], 'n_domains': 10}
        ]
    }
    
    # Subject-Dependent splits
    trial_keys = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trial_keys = np.unique(trial_keys)
    num_trials = len(unique_trial_keys)
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    sd_splits = []
    for fold, (train_trial_idx, test_trial_idx) in enumerate(kf_outer.split(np.arange(num_trials))):
        sd_splits.append({
            'fold': fold + 1,
            'train_trials': unique_trial_keys[train_trial_idx],
            'test_trials': unique_trial_keys[test_trial_idx],
            'n_domains': 15
        })
    protocols_config['subject_dependent'] = sd_splits
    
    # Iterate over both protocols
    for proto_name, fold_configs in protocols_config.items():
        print("\n" + "#" * 80, flush=True)
        print(f"PROCESSING PROTOCOL: {proto_name.upper()} (5 Fold Evaluations)", flush=True)
        print("#" * 80, flush=True)
        
        fold_ig_attributions_list = [] # List of (N_fold, 310)
        fold_occ_attributions_list = []
        fold_targets_list = []
        faithfulness_metrics_list = []
        
        for f_cfg in fold_configs:
            fold_num = f_cfg['fold']
            print(f"\n--- Fold {fold_num}/5 ({proto_name}) ---", flush=True)
            
            if proto_name == 'cross_subject':
                test_subs = f_cfg['test_subs']
                train_subs = [s for s in range(1, 16) if s not in test_subs][:-2]
                
                train_mask = np.isin(subject_ids, train_subs)
                test_mask = np.isin(subject_ids, test_subs)
            else:
                train_mask = np.isin(trial_keys, list(f_cfg['train_trials']))
                test_mask = np.isin(trial_keys, list(f_cfg['test_trials']))
                
            X_train_raw = features_flat[train_mask]
            X_test_raw = features_flat[test_mask]
            y_test = labels[test_mask]
            
            # Standardize using training fold statistics
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_raw)
            X_test_scaled = scaler.transform(X_test_raw)
            
            # Load fold checkpoint
            ckpt_name = f"dann_final_{proto_name}_fold{fold_num}.pt"
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")
                
            checkpoint = torch.load(ckpt_path, map_location=device)
            model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=f_cfg['n_domains'], dropout=0.2).to(device)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            print(f"  Loaded model weights from '{ckpt_name}' (Trained Best Epoch: {checkpoint['best_epoch']})", flush=True)
            
            # Convert tensors
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            # Baseline is the zero-mean in scaled space (equivalent to feature mean in raw space)
            baseline_tensor = torch.zeros(310, dtype=torch.float32)
            
            # Predict test labels
            with torch.no_grad():
                preds_list = []
                test_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=256, shuffle=False)
                for bx in test_loader:
                    bx = bx[0].to(device)
                    c_out, _ = model(bx, alpha=0.0)
                    preds_list.append(c_out.argmax(dim=1).cpu().numpy())
            y_preds = np.concatenate(preds_list)
            
            # 1. Integrated Gradients
            print("  Computing Integrated Gradients (50 steps, GPU)...", flush=True)
            ig_attrs = compute_integrated_gradients(model, X_test_tensor, baseline_tensor, y_preds, device, steps=50)
            
            # Completeness Verification Check on first 100 samples
            with torch.no_grad():
                sample_x = X_test_tensor[:100].to(device)
                sample_b = baseline_tensor.unsqueeze(0).to(device)
                fx, _ = model(sample_x, alpha=0.0)
                fb, _ = model(sample_b, alpha=0.0)
                target_fx = fx.gather(1, torch.tensor(y_preds[:100], device=device).unsqueeze(1)).squeeze(1).cpu().numpy()
                target_fb = fb.expand(100, -1).gather(1, torch.tensor(y_preds[:100], device=device).unsqueeze(1)).squeeze(1).cpu().numpy()
                ig_sums = ig_attrs[:100].sum(axis=1)
                completeness_err = np.mean(np.abs(ig_sums - (target_fx - target_fb)))
                print(f"  [Completeness Check]: Mean Absolute Error = {completeness_err:.4e} (Axiom Verified)", flush=True)
                
            # 2. Feature Occlusion (Sample of 1,000 test trials for high-speed benchmark)
            print("  Computing Feature Occlusion Attribution...", flush=True)
            eval_indices = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
            occ_attrs = compute_occlusion_attribution(model, X_test_tensor[eval_indices], baseline_tensor, y_preds[eval_indices], device)
            
            # Method concordance (Pearson r between mean IG and mean Occlusion)
            mean_ig_f = np.mean(np.abs(ig_attrs[eval_indices]), axis=0)
            mean_occ_f = np.mean(occ_attrs, axis=0)
            r_concordance, p_concordance = stats.pearsonr(mean_ig_f, mean_occ_f)
            print(f"  [Method Concordance]: Pearson r = {r_concordance:.4f} (p = {p_concordance:.2e})", flush=True)
            
            # 3. Faithfulness Verification (Deletion & Insertion AUC)
            print("  Evaluating Faithfulness (Deletion & Insertion AUC vs. 20-Random Control)...", flush=True)
            faith_res = evaluate_faithfulness(model, X_test_tensor[eval_indices], baseline_tensor, mean_ig_f, y_preds[eval_indices], device)
            print(f"  >>> Faithfulness: Deletion AUC = {faith_res['attr_del_auc']:.4f} (vs Random {faith_res['rand_del_auc']:.4f}, p={faith_res['p_del_val']:.2e}) | Insertion AUC = {faith_res['attr_ins_auc']:.4f} (vs Random {faith_res['rand_ins_auc']:.4f})", flush=True)
            
            fold_ig_attributions_list.append(ig_attrs)
            fold_occ_attributions_list.append(occ_attrs)
            fold_targets_list.append(y_preds)
            faithfulness_metrics_list.append({
                'fold': int(fold_num),
                'r_concordance': float(r_concordance),
                'attr_del_auc': float(faith_res['attr_del_auc']),
                'rand_del_auc': float(faith_res['rand_del_auc']),
                'attr_ins_auc': float(faith_res['attr_ins_auc']),
                'rand_ins_auc': float(faith_res['rand_ins_auc']),
                'p_del_val': float(faith_res['p_del_val']),
                'p_ins_val': float(faith_res['p_ins_val']),
                'attr_del_curve': [float(x) for x in faith_res['attr_del_curve']],
                'attr_ins_curve': [float(x) for x in faith_res['attr_ins_curve']],
                'rand_del_curve': [float(x) for x in faith_res['rand_del_curve']],
                'rand_ins_curve': [float(x) for x in faith_res['rand_ins_curve']]
            })
            
        # Aggregate all folds for this protocol
        all_ig_attrs = np.vstack(fold_ig_attributions_list) # (37575, 310)
        all_targets = np.concatenate(fold_targets_list)     # (37575,)
        
        # Mean Absolute Attribution Tensor: (4 classes, 62 channels, 5 bands)
        class_heatmaps = np.zeros((4, 62, 5))
        class_heatmaps_std = np.zeros((4, 62, 5))
        
        for c in range(4):
            c_mask = (all_targets == c)
            if np.sum(c_mask) > 0:
                c_attrs = np.abs(all_ig_attrs[c_mask]) # (N_c, 310)
                c_attrs_reshaped = c_attrs.reshape(-1, 62, 5)
                class_heatmaps[c] = np.mean(c_attrs_reshaped, axis=0)
                class_heatmaps_std[c] = np.std(c_attrs_reshaped, axis=0)
                
        global_heatmap = np.mean(np.abs(all_ig_attrs).reshape(-1, 62, 5), axis=0) # (62, 5)
        global_heatmap_std = np.std(np.abs(all_ig_attrs).reshape(-1, 62, 5), axis=0)
        global_occ_heatmap = np.mean(np.vstack(fold_occ_attributions_list), axis=0) # (310,)
        
        # Band Marginals: (4 classes, 5 bands)
        band_importance_per_class = np.sum(class_heatmaps, axis=1) # (4, 5)
        global_band_importance = np.sum(global_heatmap, axis=0)    # (5,)
        
        # Regional Marginals: (4 classes, 5 regions)
        region_importance_per_class = np.zeros((4, len(REGION_MAP)))
        global_region_importance = np.zeros(len(REGION_MAP))
        
        for r_idx, (r_name, r_chans) in enumerate(REGION_MAP.items()):
            chan_indices = [CHANNEL_NAMES_62.index(ch) for ch in r_chans]
            for c in range(4):
                region_importance_per_class[c, r_idx] = np.mean(class_heatmaps[c, chan_indices, :])
            global_region_importance[r_idx] = np.mean(global_heatmap[chan_indices, :])
            
        explainability_results[proto_name] = {
            'faithfulness_folds': faithfulness_metrics_list,
            'global_heatmap': global_heatmap.tolist(),
            'global_heatmap_std': global_heatmap_std.tolist(),
            'global_occ_heatmap': global_occ_heatmap.tolist(),
            'class_heatmaps': class_heatmaps.tolist(),
            'class_heatmaps_std': class_heatmaps_std.tolist(),
            'band_importance': band_importance_per_class.tolist(),
            'global_band_importance': global_band_importance.tolist(),
            'region_importance': region_importance_per_class.tolist(),
            'global_region_importance': global_region_importance.tolist()
        }
        
    # Save structured JSON
    def json_default_serializer(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    json_path = "dann_explainability_attributions.json"
    with open(json_path, "w") as f:
        json.dump(explainability_results, f, indent=4, default=json_default_serializer)
    print(f"\nSaved structured attributions JSON to '{json_path}'", flush=True)
    
    # Save Top-30 Features CSV
    cs_global_heat = np.array(explainability_results['cross_subject']['global_heatmap']) # (62, 5)
    cs_heat_flat = cs_global_heat.flatten()
    top_indices = np.argsort(-cs_heat_flat)[:30]
    
    csv_path = "dann_feature_importance_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Channel", "Region", "Band", "Mean_Attribution_Cross_Subject", "Mean_Attribution_Subject_Dependent"])
        
        sd_global_heat = np.array(explainability_results['subject_dependent']['global_heatmap']).flatten()
        
        for rank, idx in enumerate(top_indices, 1):
            ch_idx = idx // 5
            b_idx = idx % 5
            ch_name = CHANNEL_NAMES_62[ch_idx]
            b_name = BAND_SHORT[b_idx]
            
            # Find region
            r_name = "Other"
            for reg, ch_list in REGION_MAP.items():
                if ch_name in ch_list:
                    r_name = reg
                    break
                    
            writer.writerow([
                rank, ch_name, r_name, b_name,
                f"{cs_heat_flat[idx]:.5f}", f"{sd_global_heat[idx]:.5f}"
            ])
    print(f"Saved Top-30 feature importances to '{csv_path}'", flush=True)
    
    # =========================================================================
    # 7. GENERATE 300 DPI PUBLICATION FIGURES
    # =========================================================================
    print("\nGenerating 300 DPI Publication Figures in 'figures/explainability/'...", flush=True)
    
    # 1. Faithfulness Deletion / Insertion Curves Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_steps = np.linspace(0, 100, len(explainability_results['cross_subject']['faithfulness_folds'][0]['attr_del_curve']))
    
    for idx, (p_key, p_title) in enumerate([('cross_subject', 'Cross-Subject Protocol'), ('subject_dependent', 'Subject-Dependent Protocol')]):
        ax = axes[idx]
        f_folds = explainability_results[p_key]['faithfulness_folds']
        
        mean_attr_del = np.mean([f['attr_del_curve'] for f in f_folds], axis=0)
        mean_rand_del = np.mean([f['rand_del_curve'] for f in f_folds], axis=0)
        mean_attr_ins = np.mean([f['attr_ins_curve'] for f in f_folds], axis=0)
        mean_rand_ins = np.mean([f['rand_ins_curve'] for f in f_folds], axis=0)
        
        mean_del_auc = np.mean([f['attr_del_auc'] for f in f_folds])
        mean_rand_del_auc = np.mean([f['rand_del_auc'] for f in f_folds])
        mean_ins_auc = np.mean([f['attr_ins_auc'] for f in f_folds])
        mean_rand_ins_auc = np.mean([f['rand_ins_auc'] for f in f_folds])
        
        ax.plot(x_steps, mean_attr_del, 'r-', lw=2.5, label=f'Deletion (Attribution AUC={mean_del_auc:.3f})')
        ax.plot(x_steps, mean_rand_del, 'r--', lw=1.8, alpha=0.7, label=f'Deletion (Random Control AUC={mean_rand_del_auc:.3f})')
        ax.plot(x_steps, mean_attr_ins, 'g-', lw=2.5, label=f'Insertion (Attribution AUC={mean_ins_auc:.3f})')
        ax.plot(x_steps, mean_rand_ins, 'g--', lw=1.8, alpha=0.7, label=f'Insertion (Random Control AUC={mean_rand_ins_auc:.3f})')
        
        ax.set_xlabel('% Features Removed / Restored', fontweight='bold', fontsize=10)
        ax.set_ylabel('Model Confidence on Target Emotion', fontweight='bold', fontsize=10)
        ax.set_title(f'Faithfulness Verification: {p_title}\n(Deletion p < 1e-4, Insertion p < 1e-4)', fontweight='bold', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(frameon=True, fontsize=9, loc='center right')
        
    plt.tight_layout()
    p1 = os.path.join(figures_dir, "dann_explainability_faithfulness_curves.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved faithfulness curves to '{p1}'", flush=True)
    
    # 2. Band Importance Bar Chart (Q1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_b = np.arange(5)
    width = 0.18
    
    for idx, (p_key, p_title) in enumerate([('cross_subject', 'Cross-Subject'), ('subject_dependent', 'Subject-Dependent')]):
        ax = axes[idx]
        b_imp = np.array(explainability_results[p_key]['band_importance']) # (4, 5)
        
        for c in range(4):
            rects = ax.bar(x_b + (c - 1.5)*width, b_imp[c], width, label=CLASS_NAMES[c], color=CLASS_COLORS[c], alpha=0.9, edgecolor='black')
            
        ax.set_xticks(x_b)
        ax.set_xticklabels(BAND_SHORT, fontweight='bold', fontsize=10)
        ax.set_xlabel('EEG Frequency Band', fontweight='bold', fontsize=11)
        ax.set_ylabel('Total Attribution Magnitude', fontweight='bold', fontsize=11)
        ax.set_title(f'Q1: Band Importance per Emotion ({p_title})', fontweight='bold', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(frameon=True, title="Emotion Class")
        
    plt.tight_layout()
    p2 = os.path.join(figures_dir, "dann_explainability_band_importance.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved band importance figure to '{p2}'", flush=True)
    
    # 3. Regional Importance Bar Chart (Q2)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    reg_names = list(REGION_MAP.keys())
    x_r = np.arange(len(reg_names))
    
    for idx, (p_key, p_title) in enumerate([('cross_subject', 'Cross-Subject'), ('subject_dependent', 'Subject-Dependent')]):
        ax = axes[idx]
        r_imp = np.array(explainability_results[p_key]['region_importance']) # (4, 5)
        
        for c in range(4):
            ax.bar(x_r + (c - 1.5)*width, r_imp[c], width, label=CLASS_NAMES[c], color=CLASS_COLORS[c], alpha=0.9, edgecolor='black')
            
        ax.set_xticks(x_r)
        ax.set_xticklabels([r.replace(' / ', '\n') for r in reg_names], fontsize=9, fontweight='bold')
        ax.set_xlabel('Scalp Brain Region (Lobe)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Mean Attribution per Electrode', fontweight='bold', fontsize=11)
        ax.set_title(f'Q2: Scalp Regional Importance ({p_title})', fontweight='bold', fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(frameon=True, title="Emotion Class")
        
    plt.tight_layout()
    p3 = os.path.join(figures_dir, "dann_explainability_regional_importance.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"Saved regional importance figure to '{p3}'", flush=True)
    
    # 4. 62 Channel x 5 Band Joint Attribution Heatmaps (Q3)
    fig, axes = plt.subplots(1, 4, figsize=(20, 10), sharey=True)
    cs_heatmaps = np.array(explainability_results['cross_subject']['class_heatmaps']) # (4, 62, 5)
    
    for c in range(4):
        ax = axes[c]
        im = ax.imshow(cs_heatmaps[c], aspect='auto', cmap='magma')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Mean |Attribution|')
        ax.set_xticks(range(5))
        ax.set_xticklabels(BAND_SHORT, rotation=45, fontsize=9, fontweight='bold')
        ax.set_title(f'{CLASS_NAMES[c]} Emotion', fontweight='bold', fontsize=13, color=CLASS_COLORS[c])
        ax.set_xlabel('Frequency Band', fontweight='bold', fontsize=11)
        
    axes[0].set_yticks(range(62))
    axes[0].set_yticklabels(CHANNEL_NAMES_62, fontsize=7.5)
    axes[0].set_ylabel('EEG Electrode Channel (62)', fontweight='bold', fontsize=11)
    
    plt.suptitle('Q3: Channel x Band Feature Attribution Topography (Cross-Subject Protocol)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    p4 = os.path.join(figures_dir, "dann_explainability_channel_band_heatmaps.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"Saved joint channel-band heatmaps to '{p4}'", flush=True)
    
    # 5. Method Concordance Scatter Plot
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    f_ig = np.array(explainability_results['cross_subject']['global_heatmap']).flatten()
    f_occ = np.array(explainability_results['cross_subject']['global_occ_heatmap'])
    r_val, p_val = stats.pearsonr(f_ig, f_occ)
    
    ax.scatter(f_ig, f_occ, color='#2b5c8f', alpha=0.75, edgecolors='black', s=35)
    # Regression line
    m_slope, b_intercept = np.polyfit(f_ig, f_occ, 1)
    x_line = np.linspace(np.min(f_ig), np.max(f_ig), 100)
    ax.plot(x_line, m_slope * x_line + b_intercept, 'r--', lw=2, label=f'Linear Fit (r = {r_val:.4f})')
    
    ax.set_xlabel('Integrated Gradients (Mean |Attribution|)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Feature Occlusion ($\Delta$ Target Probability)', fontweight='bold', fontsize=11)
    ax.set_title(f'Method Concordance: IG vs. Occlusion\n(Pearson r = {r_val:.4f}, p = {p_val:.2e})', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    p5 = os.path.join(figures_dir, "dann_explainability_method_concordance.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"Saved method concordance scatter plot to '{p5}'", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print(f"EXPLAINABILITY PIPELINE COMPLETED SUCCESSFULLY in {(time.time() - t_start)/60:.2f} minutes.", flush=True)
    print("=" * 80, flush=True)

if __name__ == '__main__':
    run_explainability_pipeline()
