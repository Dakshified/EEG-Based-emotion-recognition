import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import kruskal, f_oneway
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Set style parameters for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Color palette definition
EMOTION_COLORS = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71'] # Neutral, Sad, Fear, Happy
CLASS_NAMES = ["neutral", "sad", "fear", "happy"]
BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_RANGES = ["1-4 Hz", "4-8 Hz", "8-14 Hz", "14-31 Hz", "31-50 Hz"]

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def parse_locs(locs_path="channel_62_pos (1).locs"):
    """
    Parses polar coordinates (angle, radius) from channel location file
    and maps them to 2D Cartesian coordinates (x, y) for scalp plotting.
    0 degrees corresponds to the top (front/nose), positive is right.
    """
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
                
                # Polar to Cartesian conversion
                theta = np.radians(angle)
                x = radius * np.sin(theta)
                y = radius * np.cos(theta)
                
                names.append(name)
                x_coords.append(x)
                y_coords.append(y)
                
    return names, np.array(x_coords), np.array(y_coords)

def draw_scalp_elements(ax, radius=0.5):
    """Draws scalp circle, nose, and ears for custom topomap plots."""
    circle = plt.Circle((0, 0), radius, color='#333333', fill=False, linewidth=1.2)
    ax.add_patch(circle)
    ax.plot([-0.04, 0, 0.04], [radius, radius + 0.04, radius], color='#333333', linewidth=1.2)
    ax.plot([-radius, -radius - 0.02, -radius], [0.08, 0, -0.08], color='#333333', linewidth=1.2)
    ax.plot([radius, radius + 0.02, radius], [0.08, 0, -0.08], color='#333333', linewidth=1.2)

def bh_fdr_correction(p_values, alpha=0.05):
    """
    Applies the Benjamini-Hochberg false discovery rate (FDR) multiple
    comparisons correction procedure. Returns a boolean array of rejections.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = p_values[sorted_indices]
    
    q_vals = (np.arange(1, n + 1) / n) * alpha
    significant = sorted_pvals <= q_vals
    
    if not np.any(significant):
        return np.zeros(n, dtype=bool)
        
    max_sig_idx = np.max(np.where(significant)[0])
    cutoff = sorted_pvals[max_sig_idx]
    
    return p_values <= cutoff

def plot_topomap(ax, values, x_coords, y_coords, significant_mask=None, title=None, cmap='RdYlBu_r', vmin=None, vmax=None):
    """
    Interpolates sparse electrode channel values onto a 2D grid and plots a
    scalp topomap masked inside the head circle.
    Marks FDR-significant channels with a red star marker.
    """
    grid_x, grid_y = np.mgrid[-0.55:0.55:200j, -0.55:0.55:200j]
    grid_z = griddata((x_coords, y_coords), values, (grid_x, grid_y), method='cubic')
    
    mask = (grid_x**2 + grid_y**2) > 0.5**2
    grid_z[mask] = np.nan
    
    im = ax.contourf(grid_x, grid_y, grid_z, levels=60, cmap=cmap, vmin=vmin, vmax=vmax, extend='both')
    draw_scalp_elements(ax, radius=0.5)
    
    # Scatter electrodes
    if significant_mask is not None:
        ax.scatter(x_coords[~significant_mask], y_coords[~significant_mask], color='#222222', s=5, alpha=0.6, edgecolors='none')
        ax.scatter(x_coords[significant_mask], y_coords[significant_mask], color='#ff2a00', marker='*', s=25, zorder=5)
    else:
        ax.scatter(x_coords, y_coords, color='#222222', s=5, alpha=0.6, edgecolors='none')
        
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
    return im

# =========================================================================
# FIGURE GENERATORS
# =========================================================================

def figure_1_average_band_power(features, labels, figures_dir):
    """
    Figure 1: Grouped bar chart showing average DE power per class across bands.
    Includes Kruskal-Wallis significance markers (* p<0.05, ** p<0.01, ns).
    """
    features_band_mean = features.mean(axis=1) # (N, 5)
    
    class_averages = []
    for c in range(4):
        class_mask = (labels == c)
        avg = features_band_mean[class_mask].mean(axis=0)
        class_averages.append(avg)
        
    # Statistical significance testing via Kruskal-Wallis
    p_values = []
    print("\n[STATISTICS] Kruskal-Wallis Test Results per Band:")
    for b in range(5):
        groups = [features_band_mean[labels == c, b] for c in range(4)]
        stat, p = kruskal(*groups)
        p_values.append(p)
        print(f"  Band {BAND_NAMES[b].capitalize()} ({BAND_RANGES[b]}): H-statistic = {stat:.2f}, p-value = {p:.2e}")
        
    x = np.arange(5)
    width = 0.18
    
    plt.figure(figsize=(8, 5))
    for c in range(4):
        plt.bar(x + (c - 1.5) * width, class_averages[c], width, 
                label=CLASS_NAMES[c].capitalize(), color=EMOTION_COLORS[c], 
                edgecolor='#555555', alpha=0.85)
        
    # Annotate significance
    for b in range(5):
        p = p_values[b]
        sig = "** (p<0.01)" if p < 0.01 else "* (p<0.05)" if p < 0.05 else "ns (p>=0.05)"
        y_max = max(class_averages[c][b] for c in range(4))
        plt.text(b, y_max + 0.1, sig, ha='center', va='bottom', fontsize=8.5, color='#c0392b', fontweight='semibold')
        
    plt.xticks(x, [f"{name.capitalize()}\n({rng})" for name, rng in zip(BAND_NAMES, BAND_RANGES)], fontsize=9)
    plt.ylabel('Average Differential Entropy (DE)', fontsize=10, fontweight='bold')
    plt.xlabel('Frequency Band', fontsize=10, fontweight='bold')
    plt.title('Spectral Band Activity per Emotion Class with Kruskal-Wallis Significance', fontsize=10, fontweight='bold', pad=15)
    plt.legend(title='Emotion Class', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.ylim(0, max(max(avg) for avg in class_averages) * 1.25)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "eda_01_average_band_power.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_2_scalp_topomaps_all_bands(features, labels, x_coords, y_coords, figures_dir):
    """
    Figure 2: 5 separate 4-panel scalp topomaps (one per band) showing average DE
    activity per class, highlighting FDR-corrected ANOVA significant channels.
    """
    for band_idx, band_name in enumerate(BAND_NAMES):
        band_features = features[:, :, band_idx] # Shape: (37575, 62)
        
        # ANOVA per channel, Benjamini-Hochberg FDR corrected
        p_vals_channels = []
        for ch in range(62):
            groups = [band_features[labels == c, ch] for c in range(4)]
            stat, p = f_oneway(*groups)
            p_vals_channels.append(p)
            
        significant_mask = bh_fdr_correction(p_vals_channels, alpha=0.05)
        sig_count = np.sum(significant_mask)
        print(f"  ANOVA-FDR ({band_name.capitalize()} band): {sig_count}/62 channels show significant differences (p<0.05).")
        
        fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
        
        class_averages = []
        for c in range(4):
            class_mask = (labels == c)
            avg = band_features[class_mask].mean(axis=0)
            class_averages.append(avg)
            
        vmin = min(np.min(avg) for avg in class_averages)
        vmax = max(np.max(avg) for avg in class_averages)
        
        for c in range(4):
            im = plot_topomap(axes[c], class_averages[c], x_coords, y_coords, 
                              significant_mask=significant_mask,
                              title=f"{CLASS_NAMES[c].upper()}", 
                              cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
            
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
        fig.colorbar(im, cax=cbar_ax, label=f'Average {band_name.capitalize()} Band DE')
        
        plt.suptitle(f'EEG {band_name.capitalize()}-Band Scalp Topomaps per Emotion Class\n(Red stars mark channels with significant differences across emotions, FDR-corrected p<0.05)', 
                     fontsize=11, fontweight='bold', y=0.98)
        
        fig_path = os.path.join(figures_dir, f"eda_02_scalp_topomaps_{band_name}.png")
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()

def figure_3_correlation_heatmap(features, figures_dir):
    """
    Figure 3: 62x62 Pearson correlation matrix of channel activations,
    motivating spatial message-passing GAT layer designs.
    """
    channel_features = features.mean(axis=2) # Average across bands
    corr_matrix = np.corrcoef(channel_features.T)
    
    plt.figure(figsize=(9, 8))
    plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
    plt.colorbar(label='Pearson Correlation (r)')
    
    plt.title('Electrode Channel Co-activation Correlation Heatmap (62x62)', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('EEG Channel Index', fontsize=10, fontweight='bold')
    plt.ylabel('EEG Channel Index', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "eda_03_correlation_heatmap.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_4_class_separability(features, labels, figures_dir):
    """
    Figure 4: Side-by-side 2D PCA and 2D t-SNE projections of raw features,
    colored by class, using stratified sampling for the t-SNE projection.
    """
    features_flat = features.reshape(features.shape[0], -1) # Shape: (37575, 310)
    
    print("\n[DIMENSIONALITY REDUCTION] Running PCA and t-SNE projections...")
    # 1. PCA Projection
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(features_flat)
    
    # 2. Stratified Subsample for t-SNE (2000 points)
    _, x_tsne_raw, _, y_tsne = train_test_split(
        features_flat, labels, test_size=2000, stratify=labels, random_state=42
    )
    
    tsne = TSNE(n_components=2, random_state=42)
    x_tsne = tsne.fit_transform(x_tsne_raw)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA Plot
    for c in range(4):
        mask = (labels == c)
        axes[0].scatter(x_pca[mask, 0], x_pca[mask, 1], color=EMOTION_COLORS[c], 
                        label=CLASS_NAMES[c].capitalize(), alpha=0.4, s=6, edgecolors='none')
    axes[0].set_title('A. 2D PCA Projection (All Samples)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1', fontsize=9)
    axes[0].set_ylabel('Principal Component 2', fontsize=9)
    axes[0].grid(linestyle='--', alpha=0.5)
    axes[0].legend(title='Emotion Class', frameon=True, markerscale=3)
    
    # t-SNE Plot
    for c in range(4):
        mask = (y_tsne == c)
        axes[1].scatter(x_tsne[mask, 0], x_tsne[mask, 1], color=EMOTION_COLORS[c], 
                        label=CLASS_NAMES[c].capitalize(), alpha=0.6, s=12, edgecolors='none')
    axes[1].set_title('B. 2D t-SNE Projection (2,000 Stratified Samples)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=9)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=9)
    axes[1].grid(linestyle='--', alpha=0.5)
    axes[1].legend(title='Emotion Class', frameon=True, markerscale=2)
    
    plt.suptitle('Class Separability in Raw EEG Differential Entropy Feature Space', fontsize=12, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "eda_04_class_separability.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_5_subject_variability(features, subject_ids, figures_dir):
    """
    Figure 5: Box plots of subject-wise Alpha band activity, illustrating
    cross-subject variability.
    """
    # Extract Alpha band (index 2) and mean across all 62 channels: (N, 62, 5) -> (N,)
    alpha_features = features[:, :, 2].mean(axis=1)
    
    subject_data = []
    for sub in range(1, 16):
        mask = (subject_ids == sub)
        subject_data.append(alpha_features[mask])
        
    plt.figure(figsize=(10, 5))
    box = plt.boxplot(subject_data, 
                      patch_artist=True,
                      medianprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.2, markeredgecolor='none'))
    plt.xticks(range(1, 16), [f"Subj {s}" for s in range(1, 16)])
    
    # Custom coloring for subject boxes
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(i/14) for i in range(15)]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    plt.title('Cross-Subject EEG Spectral Variability (Mean Alpha Band DE)', fontsize=11, fontweight='bold')
    plt.ylabel('DE Feature Value (Log-Power)', fontsize=10, fontweight='bold')
    plt.xlabel('Subject ID (1 to 15)', fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "eda_05_subject_variability.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_6_outlier_check(features, figures_dir):
    """
    Figure 6: Histogram of sample-wise L2 feature norms to identify extreme
    outliers beyond 3 standard deviations.
    """
    features_flat = features.reshape(features.shape[0], -1) # Shape: (37575, 310)
    
    # L2 norms
    norms = np.linalg.norm(features_flat, axis=1)
    
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    threshold_up = mean_norm + 3 * std_norm
    threshold_low = mean_norm - 3 * std_norm
    
    outliers_mask = (norms > threshold_up) | (norms < threshold_low)
    outlier_count = np.sum(outliers_mask)
    outlier_pct = (outlier_count / len(norms)) * 100
    
    print(f"\n[OUTLIER DETECTION] L2 Norm Statistics:")
    print(f"  Mean L2 Norm: {mean_norm:.2f} | Std: {std_norm:.2f}")
    print(f"  Thresholds: Lower = {threshold_low:.2f} | Upper = {threshold_up:.2f}")
    print(f"  Outliers Flagged: {outlier_count} samples ({outlier_pct:.3f}%)")
    
    plt.figure(figsize=(7, 4.5))
    plt.hist(norms, bins=50, color='#34495e', edgecolor='#555555', alpha=0.85)
    
    plt.axvline(mean_norm, color='#2ecc71', linestyle='solid', linewidth=1.5, label=f"Mean = {mean_norm:.1f}")
    plt.axvline(threshold_up, color='#e74c3c', linestyle='dashed', linewidth=1.5, label=f"+3 SD Threshold = {threshold_up:.1f}")
    plt.axvline(threshold_low, color='#e74c3c', linestyle='dashed', linewidth=1.5, label=f"-3 SD Threshold = {threshold_low:.1f}")
    
    plt.title(f'Sample Feature Tensor L2 Norm Distribution\n(Outliers Flagged: {outlier_count} samples, {outlier_pct:.3f}%)', fontsize=10, fontweight='bold', pad=12)
    plt.xlabel('Sample L2 Norm value', fontsize=10, fontweight='bold')
    plt.ylabel('Frequency (Samples)', fontsize=10, fontweight='bold')
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "eda_06_outlier_check.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    t_start = time.time()
    
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("EEG Exploratory Data Analysis (EDA) Figures Generator")
    print("=" * 70)
    
    # Parse locs
    locs_path = "channel_62_pos (1).locs"
    print(f"Parsing electrode montage from '{locs_path}'...")
    names, x_coords, y_coords = parse_locs(locs_path)
    
    # Load dataset
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
        
    print(f"Loading SEED-IV processed dataset from '{dataset_path}'...")
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    
    # Generate EDA figures
    figure_1_average_band_power(features, labels, figures_dir)
    figure_2_scalp_topomaps_all_bands(features, labels, x_coords, y_coords, figures_dir)
    figure_3_correlation_heatmap(features, figures_dir)
    figure_4_class_separability(features, labels, figures_dir)
    figure_5_subject_variability(features, subject_ids, figures_dir)
    figure_6_outlier_check(features, figures_dir)
    
    print("\nVerifying saved figure files...")
    figure_files = [
        "eda_01_average_band_power.png",
        "eda_02_scalp_topomaps_delta.png",
        "eda_02_scalp_topomaps_theta.png",
        "eda_02_scalp_topomaps_alpha.png",
        "eda_02_scalp_topomaps_beta.png",
        "eda_02_scalp_topomaps_gamma.png",
        "eda_03_correlation_heatmap.png",
        "eda_04_class_separability.png",
        "eda_05_subject_variability.png",
        "eda_06_outlier_check.png"
    ]
    
    all_ok = True
    for f_file in figure_files:
        full_path = os.path.join(figures_dir, f_file)
        if os.path.exists(full_path):
            size = os.path.getsize(full_path)
            if size > 0:
                print(f"[VERIFIED] {f_file:<35} | Path: {full_path:<45} | Size: {size/1024:.1f} KB")
            else:
                print(f"[ERROR]    {f_file:<35} | Path: {full_path:<45} | Size is 0 bytes!")
                all_ok = False
        else:
            print(f"[ERROR]    {f_file:<35} | Path: {full_path:<45} | File does not exist!")
            all_ok = False
            
    print("=" * 70)
    if all_ok:
        print(f"All EDA figures generated successfully. Total elapsed: {time.time() - t_start:.2f}s")
    else:
        print("Some EDA figure files are missing or empty!")
    print("=" * 70)

if __name__ == "__main__":
    main()
