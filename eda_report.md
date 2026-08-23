# Exploratory Data Analysis (EDA) Report: SEED-IV EEG Dataset

This report provides a comprehensive Exploratory Data Analysis (EDA) of the SEED-IV dataset, exploring spectral activity across bands, spatial electrode topology per class, spatial co-activation structures, class separability, subject variability, and outlier distributions. These analyses guide model design decisions (e.g., spatial graph networks) and serve as references for the **Exploratory Data Analysis** section of our paper.

---

## SECTION 1 - Spectral Band Power & Statistical Signatures
We explore the average Differential Entropy (DE) spectral activity across frequency bands per emotion class.

### 1. Average Band Power by Emotion Class
The DE features averaged across channels and samples show differences in spectral activity between classes.

![Average Band Power](figures/eda_01_average_band_power.png)
* **Figure 1**: Grouped bar chart of average DE spectral activity. A Kruskal-Wallis test is performed across the 4 classes for each band to statistically verify differences. All bands demonstrate highly significant differences (indicated by `**`, p < 0.01) across emotional states, verifying that raw spectral distributions contain distinct emotion-discriminative signatures prior to model training.

#### Kruskal-Wallis Test Results Companion Table
| Frequency Band | H-Statistic | p-value | Significance |
| :--- | :---: | :---: | :---: |
| **Delta (1-4 Hz)** | 192.08 | $2.17 \times 10^{-41}$ | Highly Significant ($p < 0.01$) |
| **Theta (4-8 Hz)** | 489.67 | $8.28 \times 10^{-106}$ | Highly Significant ($p < 0.01$) |
| **Alpha (8-14 Hz)** | 458.68 | $4.30 \times 10^{-99}$ | Highly Significant ($p < 0.01$) |
| **Beta (14-31 Hz)** | 440.70 | $3.38 \times 10^{-95}$ | Highly Significant ($p < 0.01$) |
| **Gamma (31-50 Hz)** | 794.66 | $6.22 \times 10^{-172}$ | Highly Significant ($p < 0.01$) |

---

## SECTION 2 - Spatial EEG Topology per Emotion Class
We visualize the spatial topological patterns per class for all five bands to identify regional differences. To control for multiple comparisons across the 62 channels (310 total tests), we apply a **Benjamini-Hochberg FDR correction (alpha = 0.05)** on the one-way ANOVA tests. Channels showing statistically significant differences across emotions after correction are marked with red stars.

```carousel
![Delta Topomap](figures/eda_02_scalp_topomaps_delta.png)
<!-- slide -->
![Theta Topomap](figures/eda_02_scalp_topomaps_theta.png)
<!-- slide -->
![Alpha Topomap](figures/eda_02_scalp_topomaps_alpha.png)
<!-- slide -->
![Beta Topomap](figures/eda_02_scalp_topomaps_beta.png)
<!-- slide -->
![Gamma Topomap](figures/eda_02_scalp_topomaps_gamma.png)
```
* **Figure 2**: Scalp topomaps showing average DE activity per emotion class across the five bands. Red star markers highlight channels showing significant difference in activity across classes under a Benjamini-Hochberg FDR correction (p < 0.05). Alpha band (slide 3) shows prominent asymmetry and suppression in frontal channels during fear compared to happy and sad states.
* **FDR Results Summary**:
  * Delta Band: 62/62 channels show significant differences (p < 0.05).
  * Theta Band: 62/62 channels show significant differences (p < 0.05).
  * Alpha Band: 61/62 channels show significant differences (p < 0.05).
  * Beta Band: 60/62 channels show significant differences (p < 0.05).
  * Gamma Band: 62/62 channels show significant differences (p < 0.05).

---

## SECTION 3 - Spatial Channel Correlation Heatmap
A $62 \times 62$ Pearson correlation matrix computes co-activations between electrodes.

![Correlation Heatmap](figures/eda_03_correlation_heatmap.png)
* **Figure 3**: Electrode channel Pearson correlation heatmap. The matrix shows strong localized correlation blocks along the diagonal (e.g., adjacent frontal and parietal channels), motivating our graph-based GAT layers: local spatial message-passing over physically adjacent channels is critical to capture regional co-activation patterns.

---

## SECTION 4 - Class Separability in Raw Feature Space
To evaluate class separability prior to model training, we project the raw features (flattened 310 dimensions) into 2D space.

![Class Separability](figures/eda_04_class_separability.png)
* **Figure 4**: 2D PCA projection of all samples (Panel A) and 2D t-SNE projection of 2,000 stratified samples (Panel B), colored by class. Stratified sampling ensures all four classes are proportionally represented in the t-SNE subset. Both projections show significant overlap in the raw input space, illustrating that emotions are not linearly separable in raw feature space and verifying the necessity of deep learning architectures (e.g., GAT-KAN) to separate emotional brain states.

---

## SECTION 5 - Subject-Wise Variability & Outlier Detection
We analyze cross-subject variance and sample anomalies in the dataset.

### 5. Per-Subject Variability
Subject-wise box plots illustrate distributions of mean Alpha DE values across subjects.

![Subject Variability](figures/eda_05_subject_variability.png)
* **Figure 5**: Box plots of mean Alpha-band DE values grouped by subject. The plot shows distinct distribution shifts across the 15 subjects, indicating that baseline brain-state characteristics vary meaningfully across individuals, which motivates subject-fairness and subject-dependent model evaluations.

### 6. Outlier / Anomaly Check
We compute the L2 norm of the 310-dimensional feature vector for each sample to identify extreme anomalies beyond 3 standard deviations.

![Outlier Check](figures/eda_06_outlier_check.png)
* **Figure 6**: Histogram of sample-wise L2 norm values. Thresholds at mean $\pm$ 3 standard deviations are marked in red. Outlier analysis reveals:
  * Mean L2 Norm: 373.34
  * Standard Deviation: 14.89
  * Lower Threshold: 328.65 | Upper Threshold: 418.02
  * Outliers Flagged: 1,066 samples (2.837% of total)
  
Only 2.837% of samples fall outside the 3-sigma range, confirming that the SEED-IV features contain high-quality, smoothed activations with no extreme anomalous windows.
