"""
Fold-Level Statistical Significance Testing Suite Across All 7 Models:
1. Calibrated Inductive DANN (w_dom=0.1)
2. LightGBM
3. Compact EEGNet
4. Plain Transformer
5. RGNN
6. Riemannian TS + LR
7. GAT-KAN v2 (Version A Clean)

Performs:
1. Friedman Omnibus Test (df=6, k=7) on per-fold Accuracy and Macro-F1 (N=5 folds).
2. Post-Hoc Pairwise Wilcoxon Signed-Rank Tests (m=21 pairs) with Holm-Bonferroni correction.
3. Non-Parametric Effect Sizes: Cliff's Delta, Rank-Biserial Correlation, and Mean Differences.
4. Model Mean Ranking calculations.
5. Structured JSON/CSV Export & 300 DPI Publication-Quality Heatmaps in figures/baselines/.
"""

import os
import time
import json
import csv
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

MODEL_KEYS = [
    'dann',
    'lightgbm',
    'eegnet',
    'transformer',
    'rgnn',
    'riemannian',
    'gatkanv2'
]

MODEL_DISPLAY_NAMES = {
    'dann': 'Calibrated DANN (w=0.1)',
    'lightgbm': 'LightGBM',
    'eegnet': 'Compact EEGNet',
    'transformer': 'Plain Transformer',
    'rgnn': 'RGNN',
    'riemannian': 'Riemannian TS + LR',
    'gatkanv2': 'GAT-KAN v2 (Ver A)'
}

PROTOCOLS = ['subject_dependent', 'cross_subject']
FIGURES_DIR = os.path.join("figures", "baselines")
os.makedirs(FIGURES_DIR, exist_ok=True)

# =========================================================================
# 1. LOAD MODEL RESULTS DIRECTLY FROM SAVED ARTIFACTS
# =========================================================================

def load_all_7_model_results():
    results = {}
    
    # 1. Calibrated Inductive DANN (w_dom=0.1)
    with open("dann_final_results.json", "r") as f:
        results['dann'] = json.load(f)
        
    # 2. LightGBM
    with open("lightgbm_baseline_results.json", "r") as f:
        results['lightgbm'] = json.load(f)
        
    # 3. Compact EEGNet
    with open("eegnet_baseline_results.json", "r") as f:
        results['eegnet'] = json.load(f)
        
    # 4. Plain Transformer
    with open("transformer_baseline_results.json", "r") as f:
        results['transformer'] = json.load(f)
        
    # 5. RGNN
    with open("rgnn_baseline_results.json", "r") as f:
        results['rgnn'] = json.load(f)
        
    # 6. Riemannian TS + LR
    with open("riemannian_baseline_results.json", "r") as f:
        results['riemannian'] = json.load(f)
        
    # 7. GAT-KAN v2 (Version A Clean) from checkpoints
    gatkan_cross_folds = []
    for fold in range(1, 6):
        path = os.path.join("checkpoints", f"gatkanv2_versionA_cross-subject_fold{fold}_metrics.json")
        with open(path, "r") as f:
            gatkan_cross_folds.append(json.load(f))
            
    gatkan_dep_folds = []
    for fold in range(1, 6):
        path = os.path.join("checkpoints", f"gatkanv2_versionA_subject-dependent_fold{fold}_metrics.json")
        with open(path, "r") as f:
            gatkan_dep_folds.append(json.load(f))
            
    results['gatkanv2'] = {
        'cross_subject': {'fold_metrics': gatkan_cross_folds},
        'subject_dependent': {'fold_metrics': gatkan_dep_folds}
    }
    
    return results

# =========================================================================
# 2. EFFECT SIZE & CORRECTION HELPERS
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
    Applies Holm-Bonferroni step-down correction to an array of p-values.
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
# 3. STATISTICAL SIGNIFICANCE TESTING PIPELINE
# =========================================================================

def run_statistical_significance_suite(loaded_results):
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

            score_matrix = np.array(score_matrix) # shape: (7, 5)

            # 1. Friedman Omnibus Test (df = 6)
            stat_f, p_f = stats.friedmanchisquare(*score_matrix)
            is_friedman_sig = bool(p_f < 0.05)

            # Mean model ranks across folds (1 = best / highest score)
            ranks_per_fold = []
            for col in range(score_matrix.shape[1]):
                col_scores = score_matrix[:, col]
                col_ranks = stats.rankdata(-col_scores) # negative for descending rank
                ranks_per_fold.append(col_ranks)
            mean_ranks = np.mean(ranks_per_fold, axis=0)
            mean_rank_dict = {m: float(r) for m, r in zip(MODEL_KEYS, mean_ranks)}

            # 2. Pairwise Comparisons (21 pairs)
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
                    'status': 'Confirmatory' if is_friedman_sig else f'Exploratory (Friedman non-sig, p={p_f:.4f})',
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
# 4. PUBLICATION FIGURES (300 DPI)
# =========================================================================

def generate_significance_heatmaps(fold_stats, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    n_models = len(MODEL_KEYS)
    labels = [MODEL_DISPLAY_NAMES[m] for m in MODEL_KEYS]

    for proto in PROTOCOLS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.8))

        acc_p_matrix = np.ones((n_models, n_models))
        acc_delta_matrix = np.zeros((n_models, n_models))

        fold_pairs = fold_stats[proto]['accuracy']['pairwise_comparisons']

        for fp in fold_pairs:
            i = MODEL_KEYS.index(fp['model_1'])
            j = MODEL_KEYS.index(fp['model_2'])
            p_h = fp['p_value_holm']
            delta = fp['cliffs_delta']

            acc_p_matrix[i, j] = p_h
            acc_p_matrix[j, i] = p_h
            acc_delta_matrix[i, j] = delta
            acc_delta_matrix[j, i] = -delta

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
                    txt = f"p={p_val:.3f}"
                    col = "white" if p_val < 0.04 else "black"
                    axes[0].text(j, i, txt, ha="center", va="center", color=col, fontsize=8)

        axes[0].set_xticks(range(n_models))
        axes[0].set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        axes[0].set_yticks(range(n_models))
        axes[0].set_yticklabels(labels, fontsize=9)
        friedman_p = fold_stats[proto]['accuracy']['friedman_p_value']
        sig_str = "SIG" if friedman_p < 0.05 else "NON-SIG"
        f_stat = fold_stats[proto]["accuracy"]["friedman_statistic"]
        proto_title = proto.replace("_", " ").title()
        axes[0].set_title(f"Pairwise Wilcoxon + Holm (Accuracy): {proto_title}\n(Friedman {sig_str}: $\\chi^2_F={f_stat:.2f}$, p={friedman_p:.2e})", fontweight="bold", fontsize=10)

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
        axes[1].set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        axes[1].set_yticks(range(n_models))
        axes[1].set_yticklabels(labels, fontsize=9)
        axes[1].set_title(f"Effect Sizes (Cliff's $\\delta$): {proto_title}\n[|$\\delta$| > 0.474: Large, 0.33-0.474: Medium, 0.147-0.33: Small]", fontweight="bold", fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(figures_dir, f"significance_matrix_{proto}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved significance heatmap: '{out_path}'", flush=True)

def generate_model_rankings_figure(fold_stats, loaded_results, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

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
        colors_palette = plt.cm.viridis(np.linspace(0.2, 0.85, len(MODEL_KEYS)))
        for patch, color in zip(bplot['boxes'], colors_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        for i, accs in enumerate(sorted_acc_data):
            y = np.random.normal(i + 1, 0.04, size=len(accs))
            ax.plot(accs, y, 'o', color='darkblue', alpha=0.6, markersize=5)

        proto_title = proto.replace("_", " ").title()
        ax.set_xlabel("Accuracy (%) across 5 Folds", fontweight="bold", fontsize=10)
        f_stat = fold_stats[proto]["accuracy"]["friedman_statistic"]
        f_p = fold_stats[proto]["accuracy"]["friedman_p_value"]
        ax.set_title(f"{proto_title}\n(Friedman $\\chi^2_F={f_stat:.2f}$, p={f_p:.2e})", fontweight="bold", fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "model_rankings_boxplot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved rankings boxplot figure: '{out_path}'", flush=True)

# =========================================================================
# 5. MAIN ENTRYPOINT & CSV EXPORT
# =========================================================================

def main():
    print("=" * 80, flush=True)
    print("ALL 7 MODELS STATISTICAL SIGNIFICANCE TESTING SUITE", flush=True)
    print("=" * 80, flush=True)

    loaded_results = load_all_7_model_results()
    print(f"Successfully loaded results for all 7 models: {list(loaded_results.keys())}", flush=True)

    fold_stats = run_statistical_significance_suite(loaded_results)

    # 1. Save JSON
    json_path = "statistical_significance_results.json"
    with open(json_path, "w") as f:
        json.dump(fold_stats, f, indent=4)
    print(f"Saved full statistical JSON results to '{json_path}'", flush=True)

    # 2. Save Friedman Omnibus CSV
    f_csv_path = "statistical_significance_friedman.csv"
    with open(f_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Protocol", "Metric", "Friedman_Chi2", "DF", "p_value", "Is_Significant", "Model_Mean_Ranks"])
        for proto in PROTOCOLS:
            for metric in ['accuracy', 'f1']:
                st = fold_stats[proto][metric]
                ranks_str = "; ".join([f"{MODEL_DISPLAY_NAMES[m]}: {st['mean_ranks'][m]:.2f}" for m in MODEL_KEYS])
                writer.writerow([
                    proto, metric, f"{st['friedman_statistic']:.4f}", st['friedman_df'],
                    f"{st['friedman_p_value']:.4e}", st['is_friedman_significant'], ranks_str
                ])
    print(f"Saved Friedman summary CSV to '{f_csv_path}'", flush=True)

    # 3. Save Pairwise Wilcoxon + Effect Sizes CSV
    p_csv_path = "statistical_significance_pairwise.csv"
    with open(p_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Protocol", "Metric", "Model_1", "Model_2", "Mean_1", "Mean_2", "Mean_Difference",
            "Wilcoxon_Stat", "p_value_raw", "p_value_holm", "Is_Significant_Holm",
            "Cliffs_Delta", "Rank_Biserial", "Status"
        ])
        for proto in PROTOCOLS:
            for metric in ['accuracy', 'f1']:
                for pw in fold_stats[proto][metric]['pairwise_comparisons']:
                    writer.writerow([
                        proto, metric, pw['model_1_name'], pw['model_2_name'],
                        f"{pw['mean_1']:.4f}", f"{pw['mean_2']:.4f}", f"{pw['mean_diff']:.4f}",
                        f"{pw['wilcoxon_stat']:.1f}", f"{pw['p_value_raw']:.4f}", f"{pw['p_value_holm']:.4f}",
                        pw['is_significant_holm'], f"{pw['cliffs_delta']:.4f}", f"{pw['rank_biserial']:.4f}",
                        pw['status']
                    ])
    print(f"Saved Pairwise summary CSV to '{p_csv_path}'", flush=True)

    # 4. Generate Figures
    generate_significance_heatmaps(fold_stats, FIGURES_DIR)
    generate_model_rankings_figure(fold_stats, loaded_results, FIGURES_DIR)

    # 5. Print Console Summary
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY OF FRIEDMAN OMNIBUS TESTS (k=7 Models, df=6)", flush=True)
    print("=" * 80, flush=True)
    for proto in PROTOCOLS:
        for metric in ['accuracy', 'f1']:
            st = fold_stats[proto][metric]
            print(f"\n[{proto.upper()} - {metric.upper()}] Friedman Chi2 = {st['friedman_statistic']:.4f} (p = {st['friedman_p_value']:.4e}) | Sig: {st['is_friedman_significant']}")
            sorted_ranks = sorted(st['mean_ranks'].items(), key=lambda x: x[1])
            rank_str = " > ".join([f"{MODEL_DISPLAY_NAMES[m]} ({r:.2f})" for m, r in sorted_ranks])
            print(f"  Ranking (1=Best): {rank_str}")

if __name__ == '__main__':
    main()
