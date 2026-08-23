import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif

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

# =========================================================================
# FIGURE GENERATORS
# =========================================================================

def figure_1_feature_importance(features, labels, names, figures_dir):
    """
    Figure 1: Horizontal bar chart showing the top 20 individual band-channel
    features ranked by ANOVA F-score.
    """
    features_flat = features.reshape(features.shape[0], -1) # Shape: (37575, 310)
    
    # Compute ANOVA F-scores
    print("\n[STATISTICS] Computing ANOVA F-scores for 310 features...")
    f_scores, _ = f_classif(features_flat, labels)
    
    # Map back to channel_band names
    feature_names = []
    for c in range(62):
        for b in range(5):
            feature_names.append(f"{names[c]}_{BAND_NAMES[b]}")
            
    # Sort in descending order
    sorted_indices = np.argsort(f_scores)[::-1]
    top_indices = sorted_indices[:20]
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.barh(np.arange(20), f_scores[top_indices][::-1], color='#34495e', edgecolor='#2c3e50', alpha=0.85, height=0.6)
    plt.yticks(np.arange(20), [feature_names[idx] for idx in top_indices][::-1], fontsize=9)
    plt.xlabel('ANOVA F-Score', fontsize=10, fontweight='bold')
    plt.ylabel('Band-Channel Feature Name', fontsize=10, fontweight='bold')
    plt.title('Top 20 Most Discriminative EEG Features (SEED-IV)', fontsize=11, fontweight='bold', pad=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "fe_01_feature_importance.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    # Print the top 5 features to console for p-value table mapping
    print("  Top 5 Features:")
    for idx in range(5):
        feat_idx = sorted_indices[idx]
        print(f"    Rank {idx+1}: {feature_names[feat_idx]} | F-score = {f_scores[feat_idx]:.2f}")
        
    return f_scores

def figure_2_importance_by_band(f_scores, figures_dir):
    """
    Figure 2: Bar chart showing average F-score aggregated per frequency band,
    illustrating relative discriminative power.
    """
    # f_scores shape: (310,) which is in order of (c0_b0, c0_b1, ... c0_b4, c1_b0, ...)
    # Let's reshape to (62, 5) to average along axis 0 (channels)
    f_scores_reshaped = f_scores.reshape(62, 5)
    band_averages = f_scores_reshaped.mean(axis=0)
    
    plt.figure(figsize=(7, 4.5))
    colors = ['#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    plt.bar(BAND_NAMES, band_averages, color=colors, edgecolor='#555555', alpha=0.85, width=0.45)
    
    plt.xticks(range(5), [f"{n.capitalize()}\n({rng})" for n, rng in zip(BAND_NAMES, BAND_RANGES)], fontsize=9)
    plt.ylabel('Mean ANOVA F-Score', fontsize=10, fontweight='bold')
    plt.xlabel('Frequency Band', fontsize=10, fontweight='bold')
    plt.title('Mean Feature Discriminative Power per Frequency Band', fontsize=11, fontweight='bold', pad=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "fe_02_importance_by_band.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    print("\n[STATISTICS] Mean Band F-scores:")
    for b in range(5):
        print(f"    Band {BAND_NAMES[b].capitalize()}: Mean F-score = {band_averages[b]:.2f}")

def figure_3_representation_design(figures_dir):
    """
    Figure 3: Schematic diagram comparing flat vs. band-separated representation layouts.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Strategy A: Flattened
    rect_a = plt.Rectangle((0.05, 0.45), 0.42, 0.4, facecolor='#f5f6fa', edgecolor='#7f8c8d', linewidth=1.5)
    ax.add_patch(rect_a)
    ax.text(0.26, 0.8, "A. Concatenated / Flattened Layout", fontsize=10, fontweight='bold', color='#2c3e50', ha='center')
    
    # Flat vector blocks
    for i in range(5):
        rect_block = plt.Rectangle((0.08 + i * 0.075, 0.58), 0.07, 0.12, facecolor='#ffebee', edgecolor='#c62828', linewidth=1.0)
        ax.add_patch(rect_block)
        ax.text(0.115 + i * 0.075, 0.64, BAND_NAMES[i].capitalize()[:3], fontsize=8, ha='center', va='center', fontweight='bold')
        
    ax.text(0.26, 0.5, "Used by: SVM, CNN-LSTM, Transformer baselines.\nRationale: Treats all 310 features as a single unstructured vector,\nignoring physical spectral-spatial partitions.", fontsize=7.5, ha='center', va='center')
    
    # Strategy B: Band-Separated
    rect_b = plt.Rectangle((0.53, 0.45), 0.42, 0.4, facecolor='#e1f5fe', edgecolor='#0288d1', linewidth=1.5)
    ax.add_patch(rect_b)
    ax.text(0.74, 0.8, "B. Band-Separated Token Layout", fontsize=10, fontweight='bold', color='#01579b', ha='center')
    
    # Separate tokens
    for i in range(5):
        rect_token = plt.Rectangle((0.56 + i * 0.075, 0.58), 0.07, 0.12, facecolor='#e8f5e9', edgecolor='#2e7d32', linewidth=1.0)
        ax.add_patch(rect_token)
        ax.text(0.595 + i * 0.075, 0.64, f"T_{i+1}\n({BAND_NAMES[i].capitalize()[:3]})", fontsize=7.5, ha='center', va='center', fontweight='bold')
        
    ax.text(0.74, 0.5, "Used by: Proposed GAT-KAN model.\nRationale: Structured as 5 band tokens of 62 channels each,\nallowing self-attention to map explicit band-to-band relations.", fontsize=7.5, ha='center', va='center')
    
    plt.title('Feature Representation Layout Strategies', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.35, 0.95)
    
    fig_path = os.path.join(figures_dir, "fe_03_representation_design.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_4_graph_construction(names, x_coords, y_coords, figures_dir, k=8):
    """
    Figure 4: Visualizing the sparse physical k-NN graph constructed from channel positions.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_scalp_elements(ax, radius=0.5)
    
    # Plot connections (k-NN)
    coords = np.column_stack((x_coords, y_coords))
    num_nodes = coords.shape[0]
    
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    
    for i in range(num_nodes):
        sorted_indices = np.argsort(dists[i])
        # Closest is itself (index 0). Draw edges to the next k closest.
        knn_indices = sorted_indices[1 : k+1]
        for neighbor in knn_indices:
            ax.plot([x_coords[i], x_coords[neighbor]], [y_coords[i], y_coords[neighbor]], 
                    color='#7f8c8d', linewidth=0.6, alpha=0.7, zorder=2)
            
    # Plot electrodes
    ax.scatter(x_coords, y_coords, color='#dff9fb', edgecolors='#130cb7', s=130, zorder=3, linewidth=1.2)
    for name, x, y in zip(names, x_coords, y_coords):
        ax.text(x, y - 0.005, name, fontsize=7.0, ha='center', va='center', zorder=4, fontweight='semibold', color='#2c3e50')
        
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title(f'SEED-IV Sparse Physical k-NN Adjacency Layout (k={k})', fontsize=11, fontweight='bold', pad=15)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "fe_04_graph_construction.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_5_sequence_windowing(figures_dir):
    """
    Figure 5: Timeline schematic of the sliding sequence windowing layout.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Trial timeline
    ax.plot([0.05, 0.95], [0.75, 0.75], color='#2c3e50', linewidth=2.0)
    ax.text(0.5, 0.85, "SEED-IV Session Trial Timeline (Variable Duration)", fontsize=11, fontweight='bold', ha='center', color='#2c3e50')
    
    # Draw individual windows
    for i in range(12):
        x = 0.06 + i * 0.075
        rect = plt.Rectangle((x, 0.7), 0.055, 0.1, facecolor='#eef2f3', edgecolor='#7f8c8d')
        ax.add_patch(rect)
        ax.text(x + 0.0275, 0.75, f"W{i+1}", fontsize=7.0, ha='center', va='center')
        
    # Draw sliding sequence bracket (CNN-LSTM)
    rect_seq = plt.Rectangle((0.055, 0.42), 0.74, 0.15, facecolor='none', edgecolor='#e74c3c', linestyle='dashed', linewidth=1.5)
    ax.add_patch(rect_seq)
    ax.text(0.42, 0.5, "Overlapping Sliding Sequence (Length = 10, Stride = 1) [Used by CNN-LSTM]", fontsize=8.5, color='#e74c3c', fontweight='bold', ha='center')
    
    # Draw static window bracket (SVM, DGCNN, Transformer, GAT-KAN)
    rect_static = plt.Rectangle((0.81, 0.42), 0.06, 0.15, facecolor='none', edgecolor='#2ecc71', linestyle='dotted', linewidth=1.5)
    ax.add_patch(rect_static)
    ax.text(0.84, 0.35, "Static Window\n(Length=1)\n[Non-Sequential]", fontsize=8.0, color='#2ecc71', fontweight='bold', ha='center')
    
    plt.title('Sequence Windowing Strategies per Architecture', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.2, 0.95)
    
    fig_path = os.path.join(figures_dir, "fe_05_sequence_windowing.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_6_summary_table(figures_dir):
    """
    Figure 6: Summary table of the feature space configurations rendered as an image.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis('off')
    
    data = [
        ["Feature Configuration", "Parameter Detail"],
        ["Total Raw Features", "310 (62 channels x 5 bands)"],
        ["Feature Representation Type", "Differential Entropy (DE), LDS-smoothed"],
        ["Representation Variants", "Flattened (SVM/Transformer) vs. Band-Separated (GAT-KAN) vs. Sequence (CNN-LSTM)"],
        ["Standardization Approach", "z-score Scaling (Fit on Training trials strictly to prevent leakage)"],
        ["Graph structure", "k-NN graph, k=8 (GAT-KAN only); DGCNN uses a fully-learnable dense adjacency"]
    ]
    
    # Table plot
    table = ax.table(cellText=data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    
    # Custom styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_text_props(color='#333333')
            cell.set_facecolor('#fdfdfd' if row % 2 == 0 else '#f5f6fa')
            
    plt.title('Final EEG Feature Space Configurations Summary', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "fe_06_summary_table.png")
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
    print("EEG Feature Engineering Exploration Figures Generator")
    print("=" * 70)
    
    # 1. Parse locs
    locs_path = "channel_62_pos (1).locs"
    print(f"Parsing electrode montage from '{locs_path}'...")
    names, x_coords, y_coords = parse_locs(locs_path)
    
    # 2. Load dataset
    dataset_path = "seed_iv_processed.npz"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")
        
    print(f"Loading SEED-IV processed dataset from '{dataset_path}'...")
    data = np.load(dataset_path)
    features = data["features"]         # (37575, 62, 5)
    labels = data["labels"]             # (37575,)
    
    # 3. Generate figures
    f_scores = figure_1_feature_importance(features, labels, names, figures_dir)
    figure_2_importance_by_band(f_scores, figures_dir)
    figure_3_representation_design(figures_dir)
    figure_4_graph_construction(names, x_coords, y_coords, figures_dir, k=8)
    figure_5_sequence_windowing(figures_dir)
    figure_6_summary_table(figures_dir)
    
    print("\nVerifying saved figure files...")
    figure_files = [
        "fe_01_feature_importance.png",
        "fe_02_importance_by_band.png",
        "fe_03_representation_design.png",
        "fe_04_graph_construction.png",
        "fe_05_sequence_windowing.png",
        "fe_06_summary_table.png"
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
        print(f"All Feature Engineering figures generated successfully. Total elapsed: {time.time() - t_start:.2f}s")
    else:
        print("Some Feature Engineering figure files are missing or empty!")
    print("=" * 70)

if __name__ == "__main__":
    main()
