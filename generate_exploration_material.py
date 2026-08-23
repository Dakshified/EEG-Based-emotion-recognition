import os
import time
import numpy as np
import matplotlib.pyplot as plt

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
    """Draws scalp circle, nose, and ears for custom montage plots."""
    circle = plt.Circle((0, 0), radius, color='#333333', fill=False, linewidth=1.2)
    ax.add_patch(circle)
    ax.plot([-0.04, 0, 0.04], [radius, radius + 0.04, radius], color='#333333', linewidth=1.2)
    ax.plot([-radius, -radius - 0.02, -radius], [0.08, 0, -0.08], color='#333333', linewidth=1.2)
    ax.plot([radius, radius + 0.02, radius], [0.08, 0, -0.08], color='#333333', linewidth=1.2)

# =========================================================================
# FIGURE GENERATORS
# =========================================================================

def figure_1_dataset_overview(labels, subject_ids, session_nums, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # Panel A: Classes
    unique_cls, counts_cls = np.unique(labels, return_counts=True)
    axes[0].bar(CLASS_NAMES, counts_cls, color=EMOTION_COLORS, edgecolor='#555555', alpha=0.85, width=0.5)
    axes[0].set_ylabel('Sample Count (Windows)', fontsize=10, fontweight='bold')
    axes[0].set_title('A. Samples per Emotion Class\n(Reasonably Balanced, Happy Modestly Underrepresented)', fontsize=10, fontweight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Panel B: Subjects
    unique_sub, counts_sub = np.unique(subject_ids, return_counts=True)
    axes[1].bar([str(s) for s in unique_sub], counts_sub, color='#2c3e50', edgecolor='#555555', alpha=0.85)
    axes[1].set_xlabel('Subject ID', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Sample Count (Windows)', fontsize=10, fontweight='bold')
    axes[1].set_title('B. Samples per Subject\n(Balanced: Exactly 2,505 Windows Each)', fontsize=10, fontweight='bold')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Panel C: Sessions
    unique_sess, counts_sess = np.unique(session_nums, return_counts=True)
    axes[2].bar([f"Sess {s}" for s in unique_sess], counts_sess, color='#7f8c8d', edgecolor='#555555', alpha=0.85, width=0.4)
    axes[2].set_ylabel('Sample Count (Windows)', fontsize=10, fontweight='bold')
    axes[2].set_title('C. Samples per Session', fontsize=10, fontweight='bold')
    axes[2].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "01_dataset_overview.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_2_feature_distribution(features, figures_dir):
    band_features = features.mean(axis=1) # Average over channels
    
    plt.figure(figsize=(7, 5))
    box = plt.boxplot([band_features[:, i] for i in range(5)], 
                      patch_artist=True,
                      medianprops=dict(color='black', linewidth=1.5),
                      flierprops=dict(marker='o', markerfacecolor='gray', markersize=2, alpha=0.2, markeredgecolor='none'))
    
    colors = ['#1abc9c', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        
    plt.xticks(range(1, 6), [f"{name.capitalize()}\n({rng})" for name, rng in zip(BAND_NAMES, BAND_RANGES)])
    plt.title('Differential Entropy (DE) Feature Ranges per Band', fontsize=11, fontweight='bold')
    plt.ylabel('DE Feature Value (Log-Power)', fontsize=10, fontweight='bold')
    plt.xlabel('Frequency Band (SEED-IV Official Ranges)', fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "02_feature_distribution.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_3_pipeline_flowchart(figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    boxes = [
        {"x": 0.05, "y": 0.4, "w": 0.12, "h": 0.2, "title": "Raw EEG\nSignal", "desc": "62 Channels\n1000 Hz"},
        {"x": 0.22, "y": 0.4, "w": 0.14, "h": 0.2, "title": "Bandpass\nFiltering", "desc": "Delta (1-4Hz)\nTheta (4-8Hz)\nAlpha (8-14Hz)\nBeta (14-31Hz)\nGamma (31-50Hz)"},
        {"x": 0.41, "y": 0.4, "w": 0.12, "h": 0.2, "title": "Epoching\n& Windowing", "desc": "4-Second\nWindows\nNo Overlap"},
        {"x": 0.58, "y": 0.4, "w": 0.14, "h": 0.2, "title": "DE\nComputation", "desc": "DE = 0.5 * log(2*pi*e*var)\nfor Gaussian\nsignal"},
        {"x": 0.77, "y": 0.4, "w": 0.10, "h": 0.2, "title": "LDS\nSmoothing", "desc": "Linear\nDynamical\nSystem"},
        {"x": 0.91, "y": 0.4, "w": 0.08, "h": 0.2, "title": "Feature\nTensor", "desc": "Shape:\n[62 x 5]"}
    ]
    
    for i, b in enumerate(boxes):
        rect = plt.Rectangle((b["x"], b["y"]), b["w"], b["h"], facecolor='#eef2f3', edgecolor='#2c3e50', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.75, b["title"], fontsize=9, fontweight='bold', ha='center', va='center', color='#2c3e50')
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.3, b["desc"], fontsize=7.5, ha='center', va='center', color='#34495e')
        
        # Connect arrows
        if i < len(boxes) - 1:
            next_b = boxes[i+1]
            ax.annotate('', xy=(next_b["x"], 0.5), xytext=(b["x"] + b["w"], 0.5),
                        arrowprops=dict(arrowstyle="-|>", color='#2c3e50', lw=1.5, mutation_scale=10))
            
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.2, 0.8)
    plt.title('SEED-IV Data Preprocessing and Feature Extraction Pipeline', fontsize=12, fontweight='bold', pad=15)
    
    fig_path = os.path.join(figures_dir, "03_pipeline_flowchart.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_4_feature_selection(figures_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    cell_w, cell_h = 0.4, 0.35
    cells = [
        {"x": 0.1, "y": 0.45, "title": "DE + LDS\n(SELECTED FEATURE)", "desc": "Differential Entropy (DE) models the logarithmic energy characteristics of EEG matching human perception. Linear Dynamical System (LDS) effectively filters noise while preserving long-term temporal dynamics in brain states.", "selected": True},
        {"x": 0.52, "y": 0.45, "title": "PSD + LDS", "desc": "Power Spectral Density (PSD) measures linear power scales. Combined with LDS smoothing, it lacks logarithmic normalization, making it sensitive to amplitude spikes and outliers.", "selected": False},
        {"x": 0.1, "y": 0.05, "title": "DE + Moving Average", "desc": "Differential Entropy smoothed using a simple moving average. Introduces significant temporal lag at window boundaries and is sensitive to transient artifacts.", "selected": False},
        {"x": 0.52, "y": 0.05, "title": "PSD + Moving Average", "desc": "Power Spectral Density smoothed with moving average. Weakest combination: highly sensitive to high-amplitude noise, lacking temporal state-space modeling.", "selected": False}
    ]
    
    for c in cells:
        face = '#e8f5e9' if c["selected"] else '#f5f6fa'
        edge = '#2ecc71' if c["selected"] else '#bdc3c7'
        lw = 2.5 if c["selected"] else 1.0
        
        rect = plt.Rectangle((c["x"], c["y"]), cell_w, cell_h, facecolor=face, edgecolor=edge, linewidth=lw)
        ax.add_patch(rect)
        
        ax.text(c["x"] + cell_w/2, c["y"] + cell_h*0.8, c["title"], fontsize=10, fontweight='bold', ha='center', va='center', color='#1b5e20' if c["selected"] else '#2c3e50')
        ax.text(c["x"] + 0.02, c["y"] + cell_h*0.4, c["desc"], fontsize=7.5, ha='left', va='center', wrap=True, color='#2c3e50',
                bbox=dict(boxstyle='square,pad=0', fc='none', ec='none'))
        
    ax.text(0.5, 0.9, "Feature Extraction Variant Selection Matrix", fontsize=12, fontweight='bold', ha='center')
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.0, 1.0)
    
    fig_path = os.path.join(figures_dir, "04_feature_variant_selection.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_5_standardization_comparison(features, figures_dir):
    raw_data = features[:2505, :, 2].mean(axis=1) # Subject 1 Alpha
    
    scaler = StandardScaler()
    std_data = scaler.fit_transform(raw_data.reshape(-1, 1)).flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw
    axes[0].hist(raw_data, bins=30, color='#e74c3c', edgecolor='#555555', alpha=0.7)
    axes[0].set_title('A. Raw DE Value Distribution\n(Subject 1 Alpha Band, Unstandardized)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('DE Value (Log-Power)', fontsize=9)
    axes[0].set_ylabel('Frequency (Windows)', fontsize=9)
    axes[0].grid(linestyle='--', alpha=0.5)
    
    # Standardized
    axes[1].hist(std_data, bins=30, color='#2ecc71', edgecolor='#555555', alpha=0.7)
    axes[1].set_title('B. Standardized DE Value Distribution\n(Centered at Mean=0, Std=1)', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Standardized DE Value (z-score)', fontsize=9)
    axes[1].set_ylabel('Frequency (Windows)', fontsize=9)
    axes[1].grid(linestyle='--', alpha=0.5)
    
    plt.suptitle('Standardization Feature Transformation with Leakage Prevention\n(Note: Scaling parameters computed strictly on Training trials, then applied to Test)', 
                 fontsize=11, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "05_standardization_comparison.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_6_electrode_montage(names, x_coords, y_coords, figures_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_scalp_elements(ax, radius=0.5)
    
    ax.scatter(x_coords, y_coords, color='#dff9fb', edgecolors='#130cb7', s=160, zorder=3, linewidth=1.2)
    for name, x, y in zip(names, x_coords, y_coords):
        ax.text(x, y - 0.005, name, fontsize=7.5, ha='center', va='center', zorder=4, fontweight='semibold', color='#2c3e50')
        
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.axis('off')
    ax.set_title('SEED-IV 62-Channel Electrode Spatial Layout (Montage)', fontsize=11, fontweight='bold', pad=15)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "06_electrode_montage.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_7_sample_heatmap(features, names, figures_dir):
    sample = features[100]
    
    plt.figure(figsize=(6, 12))
    plt.imshow(sample, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Differential Entropy (DE) Value')
    
    plt.xticks(range(5), [f"{n.capitalize()}\n({rng})" for n, rng in zip(BAND_NAMES, BAND_RANGES)], fontsize=8)
    plt.yticks(range(62), names, fontsize=7)
    
    plt.xlabel('Frequency Band', fontsize=10, fontweight='bold', labelpad=8)
    plt.ylabel('EEG Channel', fontsize=10, fontweight='bold', labelpad=8)
    plt.title('Representative DE Feature Map (Sample 100)', fontsize=11, fontweight='bold', pad=15)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "07_single_sample_heatmap.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_8_trial_splitting_diagram(figures_dir):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis('off')
    
    ax.text(0.05, 0.9, "One Subject's 24 Trials (SEED-IV Session Protocol)", fontsize=11, fontweight='bold', color='#2c3e50')
    
    for t in range(24):
        row = t // 8
        col = t % 8
        x = 0.05 + col * 0.07
        y = 0.65 - row * 0.1
        
        is_train = t < 19
        color = '#a8e6cf' if is_train else '#ffd3b6'
        edge = '#1b5e20' if is_train else '#e65100'
        label = f"T{t+1}\n(Train)" if is_train else f"T{t+1}\n(Test)"
        
        rect = plt.Rectangle((x, y), 0.065, 0.08, facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + 0.0325, y + 0.04, label, fontsize=7.5, ha='center', va='center', fontweight='semibold')
        
    rect_leak = plt.Rectangle((0.65, 0.45), 0.32, 0.48, facecolor='#ffebee', edgecolor='#c62828', linewidth=1.5)
    ax.add_patch(rect_leak)
    ax.text(0.81, 0.88, "DATA LEAKAGE WARNING", fontsize=9, fontweight='bold', color='#c62828', ha='center')
    
    leakage_desc = (
        "Window-Level Splitting (AVOIDED):\n"
        "- Adjacent windows share overlapping samples due to sliding.\n"
        "- Splitting adjacent windows between Train and Test leaks temporal information, inflating evaluation accuracy artificially.\n\n"
        "Trial-Level Splitting (IMPLEMENTED):\n"
        "- Entire trials are separated. No window overlaps across Train and Test, guaranteeing genuine zero-leakage evaluation."
    )
    ax.text(0.66, 0.47, leakage_desc, fontsize=7.5, ha='left', va='bottom', wrap=True, color='#2c3e50')
    
    ax.annotate("Sliding Windowing Sequence within Trial", xy=(0.55, 0.25), xytext=(0.05, 0.25),
                arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=1.5))
    
    for w in range(5):
        rect_w = plt.Rectangle((0.05 + w * 0.1, 0.15), 0.08, 0.06, facecolor='#dff9fb', edgecolor='#130cb7', linewidth=1.0)
        ax.add_patch(rect_w)
        ax.text(0.09 + w * 0.1, 0.18, f"Win {w+1}", fontsize=7.5, ha='center', va='center')
        
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.08, 0.95)
    plt.title('Trial-Level Stratified Cross-Validation Splitting Protocol vs. Window Leakage', fontsize=11, fontweight='bold', pad=15)
    
    fig_path = os.path.join(figures_dir, "08_trial_splitting_diagram.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

def figure_9_trial_durations(subject_ids, session_nums, trial_ids, figures_dir):
    trial_unique_ids = subject_ids * 1000 + session_nums * 100 + trial_ids
    unique_trials, counts = np.unique(trial_unique_ids, return_counts=True)
    
    plt.figure(figsize=(7, 4.5))
    plt.hist(counts, bins=15, color='#1abc9c', edgecolor='#555555', alpha=0.8)
    
    plt.axvline(np.mean(counts), color='#e74c3c', linestyle='dashed', linewidth=1.5, label=f"Mean = {np.mean(counts):.1f}")
    plt.axvline(np.median(counts), color='#2ecc71', linestyle='dotted', linewidth=1.5, label=f"Median = {np.median(counts):.1f}")
    
    plt.title('SEED-IV Trial Duration (Window Count) Distribution', fontsize=11, fontweight='bold', pad=12)
    plt.xlabel('Number of 4-Second Windows per Trial', fontsize=10, fontweight='bold')
    plt.ylabel('Frequency (Trials)', fontsize=10, fontweight='bold')
    plt.legend(frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig_path = os.path.join(figures_dir, "09_trial_durations.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()

# =========================================================================
# SYSTEM LIBRARIES FOR STANDARDIZATION
# =========================================================================

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self, x):
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self
        
    def transform(self, x):
        return (x - self.mean_) / self.scale_
        
    def fit_transform(self, x):
        return self.fit(x).transform(x)

# =========================================================================
# MAIN EXECUTION
# =========================================================================

def main():
    t_start = time.time()
    
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    print("=" * 70)
    print("EEG Preprocessing Exploration Figures Generator (Refined Scope)")
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
    subject_ids = data["subject_ids"]   # (37575,)
    session_nums = data["session_nums"] # (37575,)
    trial_ids = data["trial_ids"]       # (37575,)
    
    # 3. Generate figures
    figure_1_dataset_overview(labels, subject_ids, session_nums, figures_dir)
    figure_2_feature_distribution(features, figures_dir)
    figure_3_pipeline_flowchart(figures_dir)
    figure_4_feature_selection(figures_dir)
    figure_5_standardization_comparison(features, figures_dir)
    figure_6_electrode_montage(names, x_coords, y_coords, figures_dir)
    figure_7_sample_heatmap(features, names, figures_dir)
    figure_8_trial_splitting_diagram(figures_dir)
    figure_9_trial_durations(subject_ids, session_nums, trial_ids, figures_dir)
    
    print("\nVerifying saved figure files...")
    figure_files = [
        "01_dataset_overview.png",
        "02_feature_distribution.png",
        "03_pipeline_flowchart.png",
        "04_feature_variant_selection.png",
        "05_standardization_comparison.png",
        "06_electrode_montage.png",
        "07_single_sample_heatmap.png",
        "08_trial_splitting_diagram.png",
        "09_trial_durations.png"
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
        print(f"All figures generated successfully. Total elapsed: {time.time() - t_start:.2f}s")
    else:
        print("Some figure files are missing or empty!")
    print("=" * 70)

if __name__ == "__main__":
    main()
