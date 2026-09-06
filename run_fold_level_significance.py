"""
Fold-Level Statistical Significance Testing Suite Across 6 Baselines:
- Inductive DANN
- LightGBM
- Compact EEGNet
- Plain Transformer
- Regularized Graph Neural Network (RGNN)
- Riemannian Tangent Space + Logistic Regression

Performs:
1. Friedman Omnibus Test (df=5) on per-fold Accuracy and Macro-F1 (N=5 folds).
2. Post-Hoc Pairwise Wilcoxon Signed-Rank Tests (m=15 pairs) with Holm-Bonferroni correction.
3. Non-Parametric Effect Sizes: Cliff's Delta, Rank-Biserial Correlation, and Mean Differences.
4. Structured JSON/CSV Export & 300 DPI Publication-Quality Heatmaps and Forest Plots in figures/baselines/.
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

            # 1. Friedman Omnibus Test
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
                    'status': 'Confirmatory' if is_friedman_sig else f'Exploratory (Friedman non-significant, p={p_f:.4f})',
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
# 3. PUBLICATION FIGURES (300 DPI)
# =========================================================================

def generate_significance_heatmaps(fold_stats, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    n_models = len(MODEL_KEYS)
    labels = [MODEL_DISPLAY_NAMES[m] for m in MODEL_KEYS]

    for proto in PROTOCOLS:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

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
                    if p_val < 0.001:
                        star = "***"
                    elif p_val < 0.01:
                        star = "**"
                    elif p_val < 0.05:
                        star = "*"
                    else:
                        star = "ns"
                    txt = f"p={p_val:.3f}\n({star})"
                    col = "white" if p_val < 0.04 else "black"
                    axes[0].text(j, i, txt, ha="center", va="center", color=col, fontsize=8.5)

        axes[0].set_xticks(range(n_models))
        axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        axes[0].set_yticks(range(n_models))
        axes[0].set_yticklabels(labels, fontsize=9)
        friedman_p = fold_stats[proto]['accuracy']['friedman_p_value']
        sig_str = "SIG" if friedman_p < 0.05 else "NON-SIG"
        f_stat = fold_stats[proto]["accuracy"]["friedman_statistic"]
        proto_title = proto.replace("_", " ").title()
        axes[0].set_title(f"Pairwise Wilcoxon + Holm: {proto_title}\n(Friedman {sig_str}: $\\chi^2_F={f_stat:.2f}$, p={friedman_p:.4f})", fontweight="bold", fontsize=10)

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
        axes[1].set_title(f"Effect Sizes (Cliff's $\\delta$): {proto_title}\n[>0.474: Large, 0.33-0.474: Medium, 0.147-0.33: Small]", fontweight="bold", fontsize=10)

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
        ax.set_title(f"{proto_title} Protocol\n(Friedman Omnibus $\\chi^2_F={f_stat:.2f}$, $p={f_p:.4f}$)", fontweight="bold", fontsize=11)
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "model_rankings_and_differences.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved ranking figure: '{out_path}'", flush=True)

# =========================================================================
# 4. EXPORT HELPERS (JSON & CSV)
# =========================================================================

def export_results_to_json_and_csv(fold_stats):
    full_output = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            'n_models': len(MODEL_KEYS),
            'models': MODEL_DISPLAY_NAMES,
            'protocols': PROTOCOLS,
            'n_folds': 5,
            'n_samples_total': 37575
        },
        'fold_level_statistics': fold_stats
    }

    json_path = "statistical_significance_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=4)
    print(f"Saved structured JSON to '{json_path}'.", flush=True)

    # 1. Friedman CSV
    friedman_csv_path = "statistical_significance_friedman.csv"
    with open(friedman_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Protocol", "Metric", "Friedman_Chi2", "DF", "p_value", "Is_Significant", "Mean_Ranks"])
        for proto in PROTOCOLS:
            for metric in ['accuracy', 'f1']:
                info = fold_stats[proto][metric]
                ranks_str = "; ".join([f"{MODEL_DISPLAY_NAMES[m]}={r:.2f}" for m, r in info['mean_ranks'].items()])
                writer.writerow([
                    proto,
                    metric,
                    f"{info['friedman_statistic']:.4f}",
                    info['friedman_df'],
                    f"{info['friedman_p_value']:.4e}",
                    info['is_friedman_significant'],
                    ranks_str
                ])
    print(f"Saved Friedman results CSV to '{friedman_csv_path}'.", flush=True)

    # 2. Pairwise CSV
    pairwise_csv_path = "statistical_significance_pairwise.csv"
    with open(pairwise_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Protocol", "Metric", "Model_1", "Model_2", "Mean_1", "Mean_2", "Mean_Diff",
            "Wilcoxon_W", "p_raw", "p_holm", "Is_Sig_Holm", "Status", "Cliffs_Delta", "Rank_Biserial"
        ])
        for proto in PROTOCOLS:
            for metric in ['accuracy', 'f1']:
                for row in fold_stats[proto][metric]['pairwise_comparisons']:
                    writer.writerow([
                        proto,
                        metric,
                        row['model_1_name'],
                        row['model_2_name'],
                        f"{row['mean_1']:.4f}",
                        f"{row['mean_2']:.4f}",
                        f"{row['mean_diff']:.4f}",
                        f"{row['wilcoxon_stat']:.1f}",
                        f"{row['p_value_raw']:.4f}",
                        f"{row['p_value_holm']:.4f}",
                        row['is_significant_holm'],
                        row['status'],
                        f"{row['cliffs_delta']:.4f}",
                        f"{row['rank_biserial']:.4f}"
                    ])
    print(f"Saved Pairwise results CSV to '{pairwise_csv_path}'.", flush=True)

# =========================================================================
# 5. PRINT SUMMARY TABLES TO CONSOLE
# =========================================================================

def print_summary_tables(fold_stats):
    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE SUMMARY TABLE: FRIEDMAN OMNIBUS TESTS (df=5)")
    print("=" * 100)
    print(f"{'Protocol':<20} | {'Metric':<10} | {'Chi-Square (df=5)':<18} | {'p-value':<12} | {'Significance (alpha=0.05)':<25}")
    print("-" * 100)
    for proto in PROTOCOLS:
        for metric in ['accuracy', 'f1']:
            info = fold_stats[proto][metric]
            sig_tag = "SIGNIFICANT (p < 0.05)" if info['is_friedman_significant'] else "NON-SIGNIFICANT (p >= 0.05)"
            print(f"{proto:<20} | {metric:<10} | {info['friedman_statistic']:<18.4f} | {info['friedman_p_value']:<12.4e} | {sig_tag:<25}")
    print("=" * 100)

    for proto in PROTOCOLS:
        print(f"\n" + "=" * 100)
        print(f"PAIRWISE POST-HOC COMPARISONS: {proto.upper()} PROTOCOL (Wilcoxon Signed-Rank + Holm Correction)")
        print("=" * 100)
        for metric in ['accuracy', 'f1']:
            info = fold_stats[proto][metric]
            print(f"\n--- Metric: {metric.upper()} (Friedman p = {info['friedman_p_value']:.4f}, Status: {'Confirmatory' if info['is_friedman_significant'] else 'Exploratory'}) ---")
            print(f"{'Model 1':<20} vs {'Model 2':<20} | {'Diff (%)':<9} | {'W-Stat':<8} | {'p (raw)':<9} | {'p (Holm)':<9} | {'Cliff d':<8} | {'Rank-Bi':<8} | {'Significant?'}")
            print("-" * 100)
            for row in info['pairwise_comparisons']:
                sig_str = "YES (*)" if row['is_significant_holm'] else "NO (ns)"
                diff_pct = row['mean_diff'] * 100
                print(f"{row['model_1_name']:<20} vs {row['model_2_name']:<20} | {diff_pct:>+8.2f}% | {row['wilcoxon_stat']:<8.1f} | {row['p_value_raw']:<9.4f} | {row['p_value_holm']:<9.4f} | {row['cliffs_delta']:>+7.2f} | {row['rank_biserial']:>+7.2f} | {sig_str}")
        print("=" * 100)

# =========================================================================
# MAIN ENTRYPOINT
# =========================================================================

def main():
    print("=" * 80)
    print("FOLD-LEVEL STATISTICAL SIGNIFICANCE EVALUATION (6 BASELINES)")
    print("=" * 80)

    loaded_results = {}
    for m_key, fname in RESULT_FILES.items():
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing results file: {fname}")
        with open(fname, 'r') as f:
            loaded_results[m_key] = json.load(f)
        print(f"Loaded '{fname}' ({MODEL_DISPLAY_NAMES[m_key]}) successfully.")

    # 1. Run Fold-Level Tests
    print("\nRunning Friedman Omnibus and Pairwise Wilcoxon tests (with Holm correction)...")
    fold_stats = run_fold_level_statistical_tests(loaded_results)

    # 2. Generate 300 DPI Publication Figures
    print("\nGenerating publication figures (300 DPI)...")
    generate_significance_heatmaps(fold_stats, FIGURES_DIR)
    generate_model_rankings_figure(fold_stats, loaded_results, FIGURES_DIR)

    # 3. Export JSON & CSVs
    print("\nExporting structured results to disk...")
    export_results_to_json_and_csv(fold_stats)

    # 4. Print Summary Tables
    print_summary_tables(fold_stats)

    print("\nStatistical significance testing complete!")

if __name__ == "__main__":
    main()
