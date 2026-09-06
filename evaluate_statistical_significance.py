"""
Statistical Significance Testing Framework Across 6 Baselines:
- LightGBM
- Compact EEGNet
- Plain Transformer
- Inductive DANN
- Regularized Graph Neural Network (RGNN)
- Riemannian Tangent Space + Logistic Regression

Performs:
1. Friedman Omnibus Test (df=5) on per-fold Accuracy and Macro-F1.
2. Post-Hoc Pairwise Wilcoxon Signed-Rank Tests (m=15) with Holm-Bonferroni correction.
3. Non-Parametric Effect Sizes (Cliff's Delta, Rank-Biserial Correlation, Mean Difference).
4. Strictly Sample-Aligned Paired Bootstrap (B=10,000 resamples) on N=37,575 pooled predictions.
5. Structured JSON/CSV Export & 300 DPI Publication-Quality Figures in figures/baselines/.
"""

import os
import time
import json
import csv
import copy
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# Fix random seed for exact reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

MODEL_KEYS = [
    'dann',
    'lightgbm',
    'eegnet',
    'transformer',
    'rgnn',
    'riemannian'
]

MODEL_DISPLAY_NAMES = {
    'dann': 'Inductive DANN',
    'lightgbm': 'LightGBM',
    'eegnet': 'Compact EEGNet',
    'transformer': 'Plain Transformer',
    'rgnn': 'RGNN',
    'riemannian': 'Riemannian TS + LR'
}

RESULT_FILES = {
    'dann': 'dann_baseline_results.json',
    'lightgbm': 'lightgbm_baseline_results.json',
    'eegnet': 'eegnet_baseline_results.json',
    'transformer': 'transformer_baseline_results.json',
    'rgnn': 'rgnn_baseline_results.json',
    'riemannian': 'riemannian_baseline_results.json'
}

PROTOCOLS = ['subject_dependent', 'cross_subject']
FIGURES_DIR = os.path.join("figures", "baselines")
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================================================================
# 1. EFFECT SIZE HELPERS
# =========================================================================

def compute_cliffs_delta(x, y):
    """Computes Cliff's delta non-parametric effect size between two vectors."""
    n_x = len(x)
    n_y = len(y)
    diff = x[:, None] - y[None, :]
    greater = np.sum(diff > 0)
    less = np.sum(diff < 0)
    return float((greater - less) / (n_x * n_y))

def compute_rank_biserial(diffs):
    """Computes Rank-Biserial Correlation for Wilcoxon signed-rank test."""
    diffs = np.asarray(diffs)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diffs))
    pos_sum = np.sum(ranks[diffs > 0])
    neg_sum = np.sum(ranks[diffs < 0])
    total = pos_sum + neg_sum
    if total == 0:
        return 0.0
    return float((pos_sum - neg_sum) / total)

def holm_bonferroni_correction(p_values):
    """
    Applies Holm-Bonferroni step-down correction to an array/list of p-values.
    Returns adjusted p-values maintaining input order.
    """
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_p = [0.0] * m
    running_max = 0.0
    for rank_idx, (orig_idx, p_val) in enumerate(indexed_p):
        multiplier = m - rank_idx
        adj = min(1.0, p_val * multiplier)
        running_max = max(running_max, adj)
        adjusted_p[orig_idx] = min(1.0, running_max)
    return adjusted_p

# =========================================================================
# 2. FOLD-LEVEL FRIEDMAN & WILCOXON TESTS
# =========================================================================

def run_fold_level_statistical_tests(loaded_results):
    """
    Executes Friedman test and pairwise Wilcoxon signed-rank tests with Holm correction
    across per-fold metrics (Accuracy and Macro-F1).
    """
    statistical_summary = {}

    for proto in PROTOCOLS:
        statistical_summary[proto] = {}
        for metric in ['accuracy', 'f1']:
            model_scores = {}
            score_matrix = []
            for m_key in MODEL_KEYS:
                fold_metrics = loaded_results[m_key][proto]['fold_metrics']
                scores = [fm[metric] for fm in fold_metrics]
                model_scores[m_key] = scores
                score_matrix.append(scores)

            score_matrix = np.array(score_matrix) # shape: (6, 5)

            # 1. Friedman Test
            stat_f, p_f = stats.friedmanchisquare(*score_matrix)
            is_friedman_sig = bool(p_f < 0.05)

            # Mean model ranks (1 = best / highest score)
            ranks_per_fold = []
            for col in range(score_matrix.shape[1]):
                col_scores = score_matrix[:, col]
                col_ranks = stats.rankdata(-col_scores)
                ranks_per_fold.append(col_ranks)
            mean_ranks = np.mean(ranks_per_fold, axis=0)
            mean_rank_dict = {m: float(r) for m, r in zip(MODEL_KEYS, mean_ranks)}

            # 2. Pairwise Comparisons (15 pairs)
            pairwise_results = []
            raw_p_values = []
            pair_keys = []

            for i in range(len(MODEL_KEYS)):
                for j in range(i + 1, len(MODEL_KEYS)):
                    m1 = MODEL_KEYS[i]
                    m2 = MODEL_KEYS[j]
                    s1 = np.array(model_scores[m1])
                    s2 = np.array(model_scores[m2])
                    diffs = s1 - s2

                    try:
                        w_res = stats.wilcoxon(s1, s2, zero_method='pratt', alternative='two-sided')
                        w_stat = float(w_res.statistic)
                        p_val = float(w_res.pvalue)
                    except Exception:
                        w_stat = 0.0
                        p_val = 1.0

                    c_delta = compute_cliffs_delta(s1, s2)
                    r_biserial = compute_rank_biserial(diffs)
                    mean_diff = float(np.mean(diffs))

                    raw_p_values.append(p_val)
                    pair_keys.append((m1, m2, w_stat, mean_diff, c_delta, r_biserial, s1, s2))

            holm_p_values = holm_bonferroni_correction(raw_p_values)

            for idx, (m1, m2, w_stat, mean_diff, c_delta, r_biserial, s1, s2) in enumerate(pair_keys):
                p_raw = raw_p_values[idx]
                p_holm = holm_p_values[idx]
                pairwise_results.append({
                    'model_1': m1,
                    'model_2': m2,
                    'model_1_name': MODEL_DISPLAY_NAMES[m1],
                    'model_2_name': MODEL_DISPLAY_NAMES[m2],
                    'metric': metric,
                    'mean_1': float(np.mean(s1)),
                    'mean_2': float(np.mean(s2)),
                    'mean_diff': mean_diff,
                    'wilcoxon_stat': w_stat,
                    'p_value_raw': p_raw,
                    'p_value_holm': p_holm,
                    'is_significant_holm': bool(p_holm < 0.05),
                    'status': 'Confirmatory' if is_friedman_sig else 'Exploratory (Friedman non-significant)',
                    'cliffs_delta': c_delta,
                    'rank_biserial': r_biserial
                })

            statistical_summary[proto][metric] = {
                'friedman_statistic': float(stat_f),
                'friedman_df': len(MODEL_KEYS) - 1,
                'friedman_p_value': float(p_f),
                'is_friedman_significant': is_friedman_sig,
                'mean_ranks': mean_rank_dict,
                'pairwise_comparisons': pairwise_results
            }

    return statistical_summary

# =========================================================================
# 3. CANONICALLY-ALIGNED SAMPLE PREDICTION GENERATOR
# =========================================================================

def generate_canonical_oof_predictions(data_path="seed_iv_processed.npz", device="cuda"):
    cache_path = "baseline_oof_predictions.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached aligned predictions from '{cache_path}'...", flush=True)
        return dict(np.load(cache_path))

    print("Generating strictly aligned out-of-fold sample-level predictions across all 6 models...", flush=True)
    data = np.load(data_path)
    features = data['features']
    labels = data['labels']
    subject_ids = data['subject_ids']
    session_ids = data['session_nums']
    trial_ids = data['trial_ids']
    N_TOTAL = len(labels)

    trial_keys = np.array([f"{s}_{sess}_{t}" for s, sess, t in zip(subject_ids, session_ids, trial_ids)])
    unique_trials = np.unique(trial_keys)

    oof_predictions = {
        'labels': labels,
        'subject_ids': subject_ids,
        'trial_keys': trial_keys
    }

    from baseline_eegnet import CompactEEGNet
    from baseline_transformer import PlainTransformer
    from baseline_dann import DANN
    from baseline_rgnn import RGNN, parse_locs, build_knn_adjacency, build_emotion_dl_matrix
    from baseline_riemannian import build_spd_matrices_gpu, project_tangent_space_gpu

    names, x_c, y_c = parse_locs("channel_62_pos (1).locs")
    A_init = build_knn_adjacency(x_c, y_c, k=14)
    emo_dl_matrix = build_emotion_dl_matrix(sigma=0.5).to(device)

    # Pre-compute covariance matrices on GPU for Riemannian baseline
    P_all_gpu = build_spd_matrices_gpu(features, device=device)

    # A. SUBJECT-DEPENDENT PROTOCOL
    print("\n--- Generating Aligned Predictions: Subject-Dependent Protocol ---", flush=True)
    sd_preds = {m: np.zeros(N_TOTAL, dtype=np.int64) for m in MODEL_KEYS}
    sd_test_coverage = np.zeros(N_TOTAL, dtype=bool)

    gkf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_pool_idx, test_idx) in enumerate(gkf_outer.split(unique_trials)):
        outer_train_trials = unique_trials[train_pool_idx]
        outer_test_trials = unique_trials[test_idx]

        kf_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_train_idx, inner_val_idx = next(kf_inner.split(np.arange(len(outer_train_trials))))
        train_trials_set = set(outer_train_trials[inner_train_idx])
        val_trials_set = set(outer_train_trials[inner_val_idx])
        test_trials_set = set(outer_test_trials)

        train_mask = np.isin(trial_keys, list(train_trials_set))
        val_mask = np.isin(trial_keys, list(val_trials_set))
        test_mask = np.isin(trial_keys, list(test_trials_set))

        assert not np.any(train_mask & test_mask)
        assert not np.any(val_mask & test_mask)
        sd_test_coverage[test_mask] = True

        X_tr, y_tr = features[train_mask], labels[train_mask]
        d_tr = subject_ids[train_mask] - 1
        X_va, y_va = features[val_mask], labels[val_mask]
        X_te, y_te = features[test_mask], labels[test_mask]

        # 1. LightGBM
        scaler = StandardScaler()
        X_tr_2d = scaler.fit_transform(X_tr.reshape(len(X_tr), -1))
        X_va_2d = scaler.transform(X_va.reshape(len(X_va), -1))
        X_te_2d = scaler.transform(X_te.reshape(len(X_te), -1))
        params = {'objective': 'multiclass', 'num_class': 4, 'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8, 'verbose': -1, 'random_state': 42, 'n_jobs': -1}
        train_data = lgb.Dataset(X_tr_2d, label=y_tr)
        val_data = lgb.Dataset(X_va_2d, label=y_va, reference=train_data)
        lgb_model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
        sd_preds['lightgbm'][test_mask] = np.argmax(lgb_model.predict(X_te_2d), axis=1)

        # 2. Riemannian TS + LR (GPU Log-Euclidean Projection)
        train_indices_arr = np.where(train_mask)[0]
        test_indices_arr = np.where(test_mask)[0]
        X_train_ts, X_test_ts = project_tangent_space_gpu(P_all_gpu[train_indices_arr], P_all_gpu[test_indices_arr], n_channels=62)
        r_scaler = StandardScaler()
        X_train_ts_s = r_scaler.fit_transform(X_train_ts)
        X_test_ts_s = r_scaler.transform(X_test_ts)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
        lr.fit(X_train_ts_s, y_tr)
        sd_preds['riemannian'][test_mask] = lr.predict(X_test_ts_s)

        # 3. Compact EEGNet
        X_tr_s = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(-1, 1, 62, 5)
        X_va_s = scaler.transform(X_va.reshape(len(X_va), -1)).reshape(-1, 1, 62, 5)
        X_te_s = scaler.transform(X_te.reshape(len(X_te), -1)).reshape(-1, 1, 62, 5)
        eeg_model = CompactEEGNet().to(device)
        opt = torch.optim.AdamW(eeg_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_s, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            eeg_model.train()
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                F.cross_entropy(eeg_model(bx), by).backward()
                opt.step()
            sch.step()
            eeg_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    corr += (eeg_model(bx).argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(eeg_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        eeg_model.load_state_dict(best_w)
        eeg_model.eval()
        with torch.no_grad():
            preds = eeg_model(torch.tensor(X_te_s, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
        sd_preds['eegnet'][test_mask] = preds

        # 4. Plain Transformer
        X_tr_st = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(-1, 62, 5)
        X_va_st = scaler.transform(X_va.reshape(len(X_va), -1)).reshape(-1, 62, 5)
        X_te_st = scaler.transform(X_te.reshape(len(X_te), -1)).reshape(-1, 62, 5)
        trans_model = PlainTransformer().to(device)
        opt = torch.optim.AdamW(trans_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_st, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_st, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            trans_model.train()
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                F.cross_entropy(trans_model(bx), by).backward()
                opt.step()
            sch.step()
            trans_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    corr += (trans_model(bx).argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(trans_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        trans_model.load_state_dict(best_w)
        trans_model.eval()
        with torch.no_grad():
            preds = trans_model(torch.tensor(X_te_st, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
        sd_preds['transformer'][test_mask] = preds

        # 5. Inductive DANN
        X_tr_sd = scaler.fit_transform(X_tr.reshape(len(X_tr), -1))
        X_va_sd = scaler.transform(X_va.reshape(len(X_va), -1))
        X_te_sd = scaler.transform(X_te.reshape(len(X_te), -1))
        dann_model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=15, dropout=0.2).to(device)
        opt = torch.optim.AdamW(dann_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_sd, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long), torch.tensor(d_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_sd, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            p = float(ep - 1) / 100
            lam = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            dann_model.train()
            for bx, by, bd in tr_ld:
                bx, by, bd = bx.to(device), by.to(device), bd.to(device)
                opt.zero_grad()
                c_out, d_out = dann_model(bx, alpha=lam)
                (F.cross_entropy(c_out, by) + F.cross_entropy(d_out, bd)).backward()
                opt.step()
            sch.step()
            dann_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    c_out, _ = dann_model(bx, alpha=0.0)
                    corr += (c_out.argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(dann_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        dann_model.load_state_dict(best_w)
        dann_model.eval()
        with torch.no_grad():
            c_out, _ = dann_model(torch.tensor(X_te_sd, dtype=torch.float32).to(device), alpha=0.0)
            preds = c_out.argmax(dim=1).cpu().numpy()
        sd_preds['dann'][test_mask] = preds

        # 6. RGNN
        rgnn_model = RGNN(num_nodes=62, in_channels=5, hidden_dim1=64, hidden_dim2=64, n_classes=4, n_domains=15, dropout=0.2, init_adj=A_init).to(device)
        opt = torch.optim.AdamW(rgnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_st, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long), torch.tensor(d_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_st, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            p = float(ep - 1) / 100
            lam = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            rgnn_model.train()
            for bx, by, bd in tr_ld:
                bx, by, bd = bx.to(device), by.to(device), bd.to(device)
                opt.zero_grad()
                c_out, d_out, _ = rgnn_model(bx, alpha=lam)
                t_soft = emo_dl_matrix[by]
                l_emo = torch.sum(-t_soft * F.log_softmax(c_out, dim=1), dim=1).mean()
                l_dom = F.cross_entropy(d_out, bd)
                eye = torch.eye(62, device=device)
                l_sp = 1e-4 * torch.sum(torch.abs(rgnn_model.raw_adj * (1.0 - eye))) / (62 * 61)
                (l_emo + 0.2 * l_dom + l_sp).backward()
                opt.step()
            sch.step()
            rgnn_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    c_out, _, _ = rgnn_model(bx, alpha=0.0)
                    corr += (c_out.argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(rgnn_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        rgnn_model.load_state_dict(best_w)
        rgnn_model.eval()
        with torch.no_grad():
            c_out, _, _ = rgnn_model(torch.tensor(X_te_st, dtype=torch.float32).to(device), alpha=0.0)
            preds = c_out.argmax(dim=1).cpu().numpy()
        sd_preds['rgnn'][test_mask] = preds

        print(f"  Fold {fold+1}/5 complete across all 6 models.", flush=True)

    assert np.all(sd_test_coverage), "Error: Not all samples covered in Subject-Dependent splits!"
    for m in MODEL_KEYS:
        oof_predictions[f'subject_dependent_{m}'] = sd_preds[m]

    # B. CROSS-SUBJECT PROTOCOL
    print("\n--- Generating Aligned Predictions: Cross-Subject Protocol ---", flush=True)
    cs_preds = {m: np.zeros(N_TOTAL, dtype=np.int64) for m in MODEL_KEYS}
    cs_test_coverage = np.zeros(N_TOTAL, dtype=bool)

    subject_folds = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    for fold, test_subs in enumerate(subject_folds):
        train_pool_subs = [s for s in range(1, 16) if s not in test_subs]
        val_subs = train_pool_subs[-2:]
        train_subs = train_pool_subs[:-2]

        train_mask = np.isin(subject_ids, train_subs)
        val_mask = np.isin(subject_ids, val_subs)
        test_mask = np.isin(subject_ids, test_subs)

        assert not np.any(train_mask & test_mask)
        assert not np.any(val_mask & test_mask)
        cs_test_coverage[test_mask] = True

        X_tr, y_tr = features[train_mask], labels[train_mask]
        sub_to_dom = {s: idx for idx, s in enumerate(train_subs)}
        d_tr = np.array([sub_to_dom[s] for s in subject_ids[train_mask]])

        X_va, y_va = features[val_mask], labels[val_mask]
        X_te, y_te = features[test_mask], labels[test_mask]

        # 1. LightGBM
        scaler = StandardScaler()
        X_tr_2d = scaler.fit_transform(X_tr.reshape(len(X_tr), -1))
        X_va_2d = scaler.transform(X_va.reshape(len(X_va), -1))
        X_te_2d = scaler.transform(X_te.reshape(len(X_te), -1))
        params = {'objective': 'multiclass', 'num_class': 4, 'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8, 'verbose': -1, 'random_state': 42, 'n_jobs': -1}
        train_data = lgb.Dataset(X_tr_2d, label=y_tr)
        val_data = lgb.Dataset(X_va_2d, label=y_va, reference=train_data)
        lgb_model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[val_data], callbacks=[lgb.early_stopping(50, verbose=False)])
        cs_preds['lightgbm'][test_mask] = np.argmax(lgb_model.predict(X_te_2d), axis=1)

        # 2. Riemannian TS + LR (GPU Log-Euclidean Projection)
        train_indices_arr = np.where(train_mask)[0]
        test_indices_arr = np.where(test_mask)[0]
        X_train_ts, X_test_ts = project_tangent_space_gpu(P_all_gpu[train_indices_arr], P_all_gpu[test_indices_arr], n_channels=62)
        r_scaler = StandardScaler()
        X_train_ts_s = r_scaler.fit_transform(X_train_ts)
        X_test_ts_s = r_scaler.transform(X_test_ts)
        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs')
        lr.fit(X_train_ts_s, y_tr)
        cs_preds['riemannian'][test_mask] = lr.predict(X_test_ts_s)

        # 3. Compact EEGNet
        X_tr_s = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(-1, 1, 62, 5)
        X_va_s = scaler.transform(X_va.reshape(len(X_va), -1)).reshape(-1, 1, 62, 5)
        X_te_s = scaler.transform(X_te.reshape(len(X_te), -1)).reshape(-1, 1, 62, 5)
        eeg_model = CompactEEGNet().to(device)
        opt = torch.optim.AdamW(eeg_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_s, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            eeg_model.train()
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                F.cross_entropy(eeg_model(bx), by).backward()
                opt.step()
            sch.step()
            eeg_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    corr += (eeg_model(bx).argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(eeg_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        eeg_model.load_state_dict(best_w)
        eeg_model.eval()
        with torch.no_grad():
            preds = eeg_model(torch.tensor(X_te_s, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
        cs_preds['eegnet'][test_mask] = preds

        # 4. Plain Transformer
        X_tr_st = scaler.fit_transform(X_tr.reshape(len(X_tr), -1)).reshape(-1, 62, 5)
        X_va_st = scaler.transform(X_va.reshape(len(X_va), -1)).reshape(-1, 62, 5)
        X_te_st = scaler.transform(X_te.reshape(len(X_te), -1)).reshape(-1, 62, 5)
        trans_model = PlainTransformer().to(device)
        opt = torch.optim.AdamW(trans_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_st, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_st, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            trans_model.train()
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                F.cross_entropy(trans_model(bx), by).backward()
                opt.step()
            sch.step()
            trans_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    corr += (trans_model(bx).argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(trans_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        trans_model.load_state_dict(best_w)
        trans_model.eval()
        with torch.no_grad():
            preds = trans_model(torch.tensor(X_te_st, dtype=torch.float32).to(device)).argmax(dim=1).cpu().numpy()
        cs_preds['transformer'][test_mask] = preds

        # 5. Inductive DANN
        X_tr_sd = scaler.fit_transform(X_tr.reshape(len(X_tr), -1))
        X_va_sd = scaler.transform(X_va.reshape(len(X_va), -1))
        X_te_sd = scaler.transform(X_te.reshape(len(X_te), -1))
        dann_model = DANN(in_features=310, hidden_dim1=256, hidden_dim2=128, n_classes=4, n_domains=10, dropout=0.2).to(device)
        opt = torch.optim.AdamW(dann_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_sd, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long), torch.tensor(d_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_sd, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            p = float(ep - 1) / 100
            lam = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            dann_model.train()
            for bx, by, bd in tr_ld:
                bx, by, bd = bx.to(device), by.to(device), bd.to(device)
                opt.zero_grad()
                c_out, d_out = dann_model(bx, alpha=lam)
                (F.cross_entropy(c_out, by) + F.cross_entropy(d_out, bd)).backward()
                opt.step()
            sch.step()
            dann_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    c_out, _ = dann_model(bx, alpha=0.0)
                    corr += (c_out.argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(dann_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        dann_model.load_state_dict(best_w)
        dann_model.eval()
        with torch.no_grad():
            c_out, _ = dann_model(torch.tensor(X_te_sd, dtype=torch.float32).to(device), alpha=0.0)
            preds = c_out.argmax(dim=1).cpu().numpy()
        cs_preds['dann'][test_mask] = preds

        # 6. RGNN
        rgnn_model = RGNN(num_nodes=62, in_channels=5, hidden_dim1=64, hidden_dim2=64, n_classes=4, n_domains=10, dropout=0.2, init_adj=A_init).to(device)
        opt = torch.optim.AdamW(rgnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
        tr_ld = DataLoader(TensorDataset(torch.tensor(X_tr_st, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long), torch.tensor(d_tr, dtype=torch.long)), batch_size=128, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(X_va_st, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)), batch_size=128, shuffle=False)
        best_va, best_w, no_imp = -1.0, None, 0
        for ep in range(1, 101):
            p = float(ep - 1) / 100
            lam = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            rgnn_model.train()
            for bx, by, bd in tr_ld:
                bx, by, bd = bx.to(device), by.to(device), bd.to(device)
                opt.zero_grad()
                c_out, d_out, _ = rgnn_model(bx, alpha=lam)
                t_soft = emo_dl_matrix[by]
                l_emo = torch.sum(-t_soft * F.log_softmax(c_out, dim=1), dim=1).mean()
                l_dom = F.cross_entropy(d_out, bd)
                eye = torch.eye(62, device=device)
                l_sp = 1e-4 * torch.sum(torch.abs(rgnn_model.raw_adj * (1.0 - eye))) / (62 * 61)
                (l_emo + 0.2 * l_dom + l_sp).backward()
                opt.step()
            sch.step()
            rgnn_model.eval()
            corr, tot = 0, 0
            with torch.no_grad():
                for bx, by in va_ld:
                    bx, by = bx.to(device), by.to(device)
                    c_out, _, _ = rgnn_model(bx, alpha=0.0)
                    corr += (c_out.argmax(dim=1) == by).sum().item()
                    tot += len(by)
            v_acc = corr / tot
            if v_acc > best_va:
                best_va = v_acc
                best_w = copy.deepcopy(rgnn_model.state_dict())
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 15:
                break
        rgnn_model.load_state_dict(best_w)
        rgnn_model.eval()
        with torch.no_grad():
            c_out, _, _ = rgnn_model(torch.tensor(X_te_st, dtype=torch.float32).to(device), alpha=0.0)
            preds = c_out.argmax(dim=1).cpu().numpy()
        cs_preds['rgnn'][test_mask] = preds

        print(f"  Fold {fold+1}/5 complete across all 6 models.", flush=True)

    assert np.all(cs_test_coverage), "Error: Not all samples covered in Cross-Subject splits!"
    for m in MODEL_KEYS:
        oof_predictions[f'cross_subject_{m}'] = cs_preds[m]

    np.savez_compressed(cache_path, **oof_predictions)
    print(f"Saved canonical aligned predictions to '{cache_path}'.", flush=True)
    return oof_predictions

# =========================================================================
# 4. SAMPLE-LEVEL PAIRED BOOTSTRAP (10,000 RESAMPLES)
# =========================================================================

def run_paired_bootstrap_sample_level(oof_data, n_resamples=10000):
    print(f"\nRunning Sample-Level Paired Bootstrap ({n_resamples:,} resamples, N=37,575 samples)...", flush=True)
    y_true = oof_data['labels']
    n_samples = len(y_true)
    bootstrap_results = {}

    np.random.seed(42)
    batch_size = 1000
    n_batches = n_resamples // batch_size

    for proto in PROTOCOLS:
        bootstrap_results[proto] = {}
        for metric in ['accuracy', 'f1']:
            print(f"  --> Protocol: {proto.upper()} | Metric: {metric.upper()}", flush=True)
            pair_records = []

            for i in range(len(MODEL_KEYS)):
                for j in range(i + 1, len(MODEL_KEYS)):
                    m1 = MODEL_KEYS[i]
                    m2 = MODEL_KEYS[j]
                    pred1 = oof_data[f"{proto}_{m1}"]
                    pred2 = oof_data[f"{proto}_{m2}"]

                    if metric == 'accuracy':
                        obs1 = np.mean(pred1 == y_true)
                        obs2 = np.mean(pred2 == y_true)
                    else:
                        obs1 = precision_recall_fscore_support(y_true, pred1, average='macro', zero_division=0)[2]
                        obs2 = precision_recall_fscore_support(y_true, pred2, average='macro', zero_division=0)[2]
                    obs_diff = float(obs1 - obs2)

                    diff_distributions = []
                    for b in range(n_batches):
                        rand_idx = np.random.randint(0, n_samples, size=(batch_size, n_samples))
                        for idx_row in rand_idx:
                            y_b = y_true[idx_row]
                            p1_b = pred1[idx_row]
                            p2_b = pred2[idx_row]

                            if metric == 'accuracy':
                                val1 = np.mean(p1_b == y_b)
                                val2 = np.mean(p2_b == y_b)
                            else:
                                val1 = precision_recall_fscore_support(y_b, p1_b, average='macro', zero_division=0)[2]
                                val2 = precision_recall_fscore_support(y_b, p2_b, average='macro', zero_division=0)[2]
                            diff_distributions.append(val1 - val2)

                    diff_distributions = np.array(diff_distributions)
                    ci_lower = float(np.percentile(diff_distributions, 2.5))
                    ci_upper = float(np.percentile(diff_distributions, 97.5))
                    se_boot = float(np.std(diff_distributions))

                    count_le_zero = np.sum(diff_distributions <= 0)
                    count_ge_zero = np.sum(diff_distributions >= 0)
                    p_boot = float(2.0 * min(count_le_zero, count_ge_zero) / n_resamples)
                    p_boot = min(1.0, max(1.0 / n_resamples, p_boot)) if p_boot == 0.0 else min(1.0, p_boot)

                    pair_records.append({
                        'model_1': m1,
                        'model_2': m2,
                        'model_1_name': MODEL_DISPLAY_NAMES[m1],
                        'model_2_name': MODEL_DISPLAY_NAMES[m2],
                        'metric': metric,
                        'obs_score_1': float(obs1),
                        'obs_score_2': float(obs2),
                        'obs_diff': obs_diff,
                        'ci_95_lower': ci_lower,
                        'ci_95_upper': ci_upper,
                        'se_boot': se_boot,
                        'p_value_boot': p_boot,
                        'is_significant_boot': bool(p_boot < 0.05)
                    })

            bootstrap_results[proto][metric] = pair_records
    return bootstrap_results

# =========================================================================
# 5. PUBLICATION PLOTS & MATRICES (300 DPI)
# =========================================================================

def generate_significance_heatmaps(fold_stats, boot_stats, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    n_models = len(MODEL_KEYS)
    labels = [MODEL_DISPLAY_NAMES[m] for m in MODEL_KEYS]

    for proto in PROTOCOLS:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

        acc_p_matrix = np.ones((n_models, n_models))
        acc_delta_matrix = np.zeros((n_models, n_models))
        acc_boot_matrix = np.ones((n_models, n_models))

        fold_pairs = fold_stats[proto]['accuracy']['pairwise_comparisons']
        boot_pairs = boot_stats[proto]['accuracy']

        for fp, bp in zip(fold_pairs, boot_pairs):
            i = MODEL_KEYS.index(fp['model_1'])
            j = MODEL_KEYS.index(fp['model_2'])
            p_h = fp['p_value_holm']
            p_b = bp['p_value_boot']
            delta = fp['cliffs_delta']

            acc_p_matrix[i, j] = p_h
            acc_p_matrix[j, i] = p_h
            acc_delta_matrix[i, j] = delta
            acc_delta_matrix[j, i] = -delta
            acc_boot_matrix[i, j] = p_b
            acc_boot_matrix[j, i] = p_b

        # Left: Holm p-values with text annotations
        im1 = axes[0].imshow(acc_p_matrix, cmap="Blues_r", vmin=0.0, vmax=0.10)
        cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar1.set_label("Holm-Corrected p-value", fontweight="bold")
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    axes[0].text(j, i, "-", ha="center", va="center", color="black", fontsize=9)
                else:
                    p_val = acc_p_matrix[i, j]
                    p_b = acc_boot_matrix[i, j]
                    if p_val < 0.001:
                        star = "***"
                    elif p_val < 0.01:
                        star = "**"
                    elif p_val < 0.05:
                        star = "*"
                    else:
                        star = "ns"
                    txt = f"p={p_val:.3f}\n({star})\n[boot:{p_b:.2e}]"
                    col = "white" if p_val < 0.04 else "black"
                    axes[0].text(j, i, txt, ha="center", va="center", color=col, fontsize=7.5)

        axes[0].set_xticks(range(n_models))
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        axes[0].set_yticks(range(n_models))
        axes[0].set_yticklabels(labels, fontsize=9)
        friedman_p = fold_stats[proto]['accuracy']['friedman_p_value']
        sig_str = "SIG" if friedman_p < 0.05 else "NON-SIG"
        f_stat = fold_stats[proto]["accuracy"]["friedman_statistic"]
        proto_title = proto.replace("_", " ").title()
        axes[0].set_title(f"Pairwise Significance: {proto_title}\n(Friedman {sig_str}: $\\chi^2_F={f_stat:.2f}$, p={friedman_p:.4f})", fontweight="bold", fontsize=10)

        # Right: Cliff's Delta Effect Sizes
        im2 = axes[1].imshow(acc_delta_matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
        cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.set_label("Cliff's Delta (Row vs Column)", fontweight="bold")
        for i in range(n_models):
            for j in range(n_models):
                d_val = acc_delta_matrix[i, j]
                col = "white" if abs(d_val) > 0.6 else "black"
                axes[1].text(j, i, f"{d_val:+.2f}", ha="center", va="center", color=col, fontsize=8.5)

        axes[1].set_xticks(range(n_models))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        axes[1].set_yticks(range(n_models))
        axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].set_title(f"Effect Sizes (Cliff\'s $\\delta$): {proto_title}\\n[>0.474: Large, 0.33-0.474: Medium, 0.147-0.33: Small]", fontweight="bold", fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(figures_dir, f"significance_matrix_{proto}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved significance heatmap: '{out_path}'", flush=True)

def generate_model_rankings_figure(fold_stats, loaded_results, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    for idx, proto in enumerate(PROTOCOLS):
        ax = axes[idx]
        acc_data = []
        model_names_plot = []
        mean_accs = []
        for m in MODEL_KEYS:
            fms = loaded_results[m][proto]['fold_metrics']
            accs = [fm['accuracy'] * 100 for fm in fms]
            acc_data.append(accs)
            model_names_plot.append(MODEL_DISPLAY_NAMES[m])
            mean_accs.append(np.mean(accs))

        sort_order = np.argsort(mean_accs)[::-1]
        sorted_acc_data = [acc_data[i] for i in sort_order]
        sorted_names = [model_names_plot[i] for i in sort_order]

        bplot = ax.boxplot(sorted_acc_data, patch_artist=True, tick_labels=sorted_names, vert=False, widths=0.55)
        for patch, col in zip(bplot['boxes'], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        for r_idx, accs in enumerate(sorted_acc_data):
            y_pts = np.random.normal(r_idx + 1, 0.04, size=len(accs))
            ax.scatter(accs, y_pts, color='black', s=25, alpha=0.8, zorder=5)

        friedman_info = fold_stats[proto]['accuracy']
        f_stat = friedman_info['friedman_statistic']
        f_p = friedman_info['friedman_p_value']
        ax.set_xlabel('Per-Fold Accuracy (%)', fontweight='bold')
        proto_title = proto.replace("_", " ").title()
        ax.set_title(f"{proto_title} Protocol\\n(Friedman Omnibus $\\chi^2_F={f_stat:.2f}$, $p={f_p:.2e}$)", fontweight="bold", fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "model_rankings_and_differences.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved ranking figure: '{out_path}'", flush=True)

# =========================================================================
# 6. EXPORT HELPERS (JSON & CSV)
# =========================================================================

def export_results_to_json_and_csv(fold_stats, boot_stats):
    full_output = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            'n_models': len(MODEL_KEYS),
            'models': MODEL_DISPLAY_NAMES,
            'protocols': PROTOCOLS,
            'n_folds': 5,
            'n_samples_total': 37575,
            'n_bootstrap_resamples': 10000
        },
        'fold_level_statistics': fold_stats,
        'sample_level_paired_bootstrap': boot_stats
    }

    json_path = "statistical_significance_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=4)
    print(f"Saved structured JSON to '{json_path}'.", flush=True)

    friedman_csv_path = "statistical_significance_friedman.csv"
    with open(friedman_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Protocol', 'Metric', 'Friedman_Chi2', 'df', 'p_value', 'Is_Significant', 'Model_Mean_Ranks'])
        for proto in PROTOCOLS:
            for metric in ['accuracy', 'f1']:
                f_data = fold_stats[proto][metric]
                ranks_str = "; ".join([f"{MODEL_DISPLAY_NAMES[m]}: {r:.2f}" for m, r in f_data["mean_ranks"].items()])
                writer.writerow([
                    proto, metric, f"{f_data['friedman_statistic']:.4f}", f_data['friedman_df'],
                    f"{f_data['friedman_p_value']:.6e}", f_data['is_friedman_significant'], ranks_str
                ])
    print(f"Saved Friedman CSV to '{friedman_csv_path}'.", flush=True)

    pairwise_csv_path = "statistical_significance_pairwise.csv"
    with open(pairwise_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Protocol', 'Metric', 'Model_1', 'Model_2', 'Mean_Score_1', 'Mean_Score_2', 'Mean_Diff',
            'Wilcoxon_W', 'p_val_raw', 'p_val_holm', 'Is_Sig_Holm', 'Test_Status',
            'Cliffs_Delta', 'Rank_Biserial', 'Boot_Obs_Diff', 'Boot_CI_95_Lower', 'Boot_CI_95_Upper',
            'Boot_SE', 'p_val_boot', 'Is_Sig_Boot'
        ])
        for proto in PROTOCOLS:
            for metric in ['accuracy', 'f1']:
                f_pairs = fold_stats[proto][metric]['pairwise_comparisons']
                b_pairs = boot_stats[proto][metric]
                for fp, bp in zip(f_pairs, b_pairs):
                    writer.writerow([
                        proto, metric, fp['model_1_name'], fp['model_2_name'],
                        f"{fp['mean_1']:.4f}", f"{fp['mean_2']:.4f}", f"{fp['mean_diff']:.4f}",
                        f"{fp['wilcoxon_stat']:.1f}", f"{fp['p_value_raw']:.6e}", f"{fp['p_value_holm']:.6e}",
                        fp['is_significant_holm'], fp['status'], f"{fp['cliffs_delta']:.4f}", f"{fp['rank_biserial']:.4f}",
                        f"{bp['obs_diff']:.4f}", f"{bp['ci_95_lower']:.4f}", f"{bp['ci_95_upper']:.4f}",
                        f"{bp['se_boot']:.6e}", f"{bp['p_value_boot']:.6e}", bp['is_significant_boot']
                    ])
    print(f"Saved Pairwise CSV to '{pairwise_csv_path}'.", flush=True)

# =========================================================================
# MAIN EXECUTION ROUTINE
# =========================================================================

def main():
    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING: 6 COMPLETED BASELINES")
    print("=" * 80)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device}", flush=True)

    loaded_results = {}
    for m_key, f_name in RESULT_FILES.items():
        with open(f_name, 'r') as f:
            loaded_results[m_key] = json.load(f)

    print("\n--- 1. Running Fold-Level Statistical Tests ---", flush=True)
    fold_stats = run_fold_level_statistical_tests(loaded_results)

    print("\n--- 2. Preparing Canonical Aligned Sample Predictions ---", flush=True)
    oof_data = generate_canonical_oof_predictions(data_path="seed_iv_processed.npz", device=device)

    print("\n--- 3. Running Sample-Level Paired Bootstrap (10,000 resamples) ---", flush=True)
    boot_stats = run_paired_bootstrap_sample_level(oof_data, n_resamples=10000)

    print("\n--- 4. Exporting Structured Results ---", flush=True)
    export_results_to_json_and_csv(fold_stats, boot_stats)

    print("\n--- 5. Generating 300 DPI Publication Visualizations ---", flush=True)
    generate_significance_heatmaps(fold_stats, boot_stats, FIGURES_DIR)
    generate_model_rankings_figure(fold_stats, loaded_results, FIGURES_DIR)

    print("\n" + "=" * 80)
    print("SUMMARY: FRIEDMAN OMNIBUS TESTS (k=6 Models, N=5 Folds)")
    print("=" * 80)
    for proto in PROTOCOLS:
        print(f"Protocol: {proto.upper()}")
        for metric in ['accuracy', 'f1']:
            f_res = fold_stats[proto][metric]
            sig_flag = "SIGNIFICANT (p < 0.05)" if f_res["is_friedman_significant"] else "NOT SIGNIFICANT (p >= 0.05)"
            f_stat = f_res["friedman_statistic"]
            f_p = f_res["friedman_p_value"]
            print(f"  Metric: {metric.upper():8s} | Chi2_F: {f_stat:7.4f} (df=5) | p-value: {f_p:.4e} | Result: {sig_flag}")
            print(f"    Model Mean Ranks (1=Best): {f_res['mean_ranks']}")

    print("\n" + "=" * 80)
    print("SUMMARY: KEY PAIRWISE COMPARISONS (Wilcoxon Holm vs Paired Bootstrap 10k)")
    print("=" * 80)
    for proto in PROTOCOLS:
        print(f"\nProtocol: {proto.upper()} (Metric: ACCURACY)")
        print(f"{'Model Pair':<40s} | {'Mean Diff':<10s} | {'Wilcoxon Holm p':<18s} | {'Cliff delta':<12s} | {'Bootstrap 10k 95% CI':<24s} | {'Boot p':<10s}")
        print("-" * 125)
        f_pairs = fold_stats[proto]['accuracy']['pairwise_comparisons']
        b_pairs = boot_stats[proto]['accuracy']
        for fp, bp in zip(f_pairs, b_pairs):
            pair_label = f"{fp['model_1_name']} vs {fp['model_2_name']}"
            m_diff_str = f"{fp['mean_diff']:+.4f}"
            sig_label = "SIG" if fp['is_significant_holm'] else "ns"
            p_h_str = f"{fp['p_value_holm']:.4f} ({sig_label})"
            delta_str = f"{fp['cliffs_delta']:+.2f}"
            ci_str = f"[{bp['ci_95_lower']:+.4f}, {bp['ci_95_upper']:+.4f}]"
            p_b_str = f"{bp['p_value_boot']:.4e}"
            print(f"{pair_label:<40s} | {m_diff_str:<10s} | {p_h_str:<18s} | {delta_str:<12s} | {ci_str:<24s} | {p_b_str:<10s}")

    print("\nAll statistical significance evaluations complete.")

if __name__ == '__main__':
    main()