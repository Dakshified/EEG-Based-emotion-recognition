# Preprocessing & Data Exploration Report: SEED-IV EEG Dataset (Refined Scope)

This report provides a comprehensive overview of the SEED-IV dataset characteristics, electrode coordinates montage, pipeline features, and data leakage prevention mechanisms. The figures and analyses contained herein motivate the design of the GAT-KAN model and serve as direct references for the **Methodology** and **Feature Engineering** sections of our paper.

---

## SECTION 1 - Dataset Overview & Sample Balance
This section details the sample configurations, subject balance, and class properties of the preprocessed SEED-IV dataset.

### 1. Dataset Overview & Balance
The preprocessed SEED-IV dataset contains 37,575 sliding-window samples (4-second duration each) extracted across 15 subjects and 3 experimental sessions. 

![Dataset Overview](figures/01_dataset_overview.png)
* **Figure 1**: SEED-IV sample distribution across classes, subjects, and sessions. Panel A shows that the dataset is reasonably balanced with modest class variation (Neutral: 10,170, Sad: 10,245, Fear: 9,225, Happy: 7,935), with Happy samples being somewhat underrepresented in accordance with the official stimulation protocols. Panel B confirms that subject contribution is perfectly balanced, with exactly 2,505 windows extracted per subject. Panel C shows session-wise sample count stability across the three sessions.

---

## SECTION 2 - Feature Value Distribution & Standardization
We examine the raw Differential Entropy (DE) feature values across the five frequency bands (Delta, Theta, Alpha, Beta, Gamma) averaged across all 62 channels.

### 2. Feature Value Distribution
The raw DE features vary widely in scale and variance across different bands.

![Feature Distribution](figures/02_feature_distribution.png)
* **Figure 2**: Box plots of raw DE feature values across bands. The plots reveal a distinct difference in raw feature scales, with high-frequency bands (Beta, Gamma) exhibiting significantly higher values and wider variances compared to lower bands (Delta, Theta). This variance motivates the absolute necessity of **feature standardization** to prevent models from being dominated by high-frequency power scales.

### 5. Standardization Comparison
Standardization transforms features to have a mean of 0 and standard deviation of 1.

![Standardization Comparison](figures/05_standardization_comparison.png)
* **Figure 3**: DE feature distribution before (Raw) and after (Standardized) scaling.
> [!IMPORTANT]
> To prevent data leakage, the standardization scaler is fit strictly on the training trials only, and then applied as a transform to validation and test trials. This ensures that validation/testing properties do not leak into the training distribution.

---

## SECTION 3 - Feature Extraction Pipeline & Selection
This section details the preprocessing steps, feature variant selection, and leakage-free standardization applied to the raw EEG signals.

### 3. Pipeline Flowchart
Raw 62-channel EEG signals are transformed into spatial-spectral feature tensors through a structured, multi-step pipeline.

![Pipeline Flowchart](figures/03_pipeline_flowchart.png)
* **Figure 4**: Programmatic flowchart of the SEED-IV preprocessing and feature extraction pipeline. The steps consist of: 
  1. Bandpass filtering into five bands using official SEED-IV ranges: Delta (1-4 Hz), Theta (4-8 Hz), Alpha (8-14 Hz), Beta (14-31 Hz), and Gamma (31-50 Hz).
  2. Segmenting into 4-second non-overlapping windows.
  3. Computing Differential Entropy (DE) per channel per band, defined for a Gaussian distribution as:
     $$DE = \frac{1}{2} \log(2\pi e \sigma^2)$$
  4. Smoothing transient artifacts using a Linear Dynamical System (LDS) state-space model to produce a final $62 \times 5$ feature tensor per window.

### 4. Feature Variant Selection
We evaluate feature representation combinations across power calculation (DE vs. PSD) and smoothing algorithms (LDS vs. Moving Average).

![Feature Selection Matrix](figures/04_feature_variant_selection.png)
* **Figure 5**: Feature extraction variant selection matrix. DE + LDS is selected as the optimal variant. DE's logarithmic power representation matches human sensory perception and limits the impact of amplitude outliers. LDS provides optimal smoothing by modeling underlying brain-state transitions, avoiding the temporal lag introduced by moving averages.

---

## SECTION 4 - Electrode Montage & Input Representation
This section maps the physical topography and spatial-spectral co-activations of the 62 electrodes.

### 6. 62-Channel Electrode Montage
The 62 electrode positions are mapped using physical coordinates from `channel_62_pos (1).locs`.

![Electrode Montage](figures/06_electrode_montage.png)
* **Figure 6**: 2D top-down scalp layout (montage) showing the spatial coordinates of the 62 EEG channels used throughout the study.

### 7. Single-Sample Spectral-Spatial Heatmap
A representative $62 \text{ channels} \times 5 \text{ bands}$ DE feature map for a single 4-second window.

![Sample Heatmap](figures/07_single_sample_heatmap.png)
* **Figure 7**: Heatmap of DE values for a single window (Sample 100). The y-axis shows channels ordered by their original index, and the x-axis shows the 5 spectral bands.

---

## SECTION 5 - Trial-Level Evaluation & Durations
This section describes the stratified, trial-level splitting protocol designed to prevent data leakage and ensure rigorous evaluation.

### 8. Trial-Level Splitting & Leakage Prevention
In sliding-window EEG processing, adjacent windows share up to 90% overlapping samples. 

![Trial Splitting](figures/08_trial_splitting_diagram.png)
* **Figure 8**: Trial-level stratified splitting protocol vs. window-level leakage. By separating data at the trial level (assigning all windows from a trial to either train or test together), we prevent temporal information leakage. Window-level splitting leaks raw future temporal states into the training set, artificially inflating evaluation accuracy.

### 9. Trial Duration Distribution
SEED-IV trials have variable durations depending on the length of the stimulation film clips.

![Trial Durations](figures/09_trial_durations.png)
* **Figure 9**: Histogram of the number of windows per trial across all trials. The mean duration is 34.8 windows (~139 seconds) and the median is 34.0 windows. This duration variance justifies window-level modeling and sliding window sequences over padding or truncating whole trials, which would introduce excessive zero-padding.
