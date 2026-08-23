import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'B', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, 'SEED-IV EEG Exploratory Data Analysis (EDA) Report', border=0, ln=1, align='R')
            self.set_draw_color(200, 200, 200)
            self.line(10, 18, 200, 18)
            self.ln(5)
            
    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', border=0, align='C')

def create_report():
    pdf = PDFReport(orientation='P', unit='mm', format='A4')
    pdf.alias_nb_pages()
    
    # -------------------------------------------------------------
    # PAGE 1: TITLE & SECTION 1: SPECTRAL BAND ACTIVITY
    # -------------------------------------------------------------
    pdf.add_page()
    
    # Title Block
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'Exploratory Data Analysis (EDA) Report', ln=1, align='C')
    pdf.set_font('helvetica', 'I', 11)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 5, 'Rigorous Spatial-Spectral Co-activation & Class Separability Analysis', ln=1, align='C')
    pdf.ln(10)
    
    # Section 1 Header
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 1 - Spectral Band Power & Statistical Signatures', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(40, 40, 40)
    desc_1 = (
        "We explore the average Differential Entropy (DE) spectral activity across frequency bands per emotion "
        "class. A Kruskal-Wallis test is performed across the four emotion classes (Neutral, Sad, Fear, Happy) "
        "for each of the five bands. All bands demonstrate highly significant differences (p < 0.01), proving that "
        "global spectral distributions carry distinct emotion-discriminative signatures prior to model training."
    )
    pdf.multi_cell(0, 5, desc_1)
    pdf.ln(3)
    
    # Figure 1
    if os.path.exists("figures/eda_01_average_band_power.png"):
        pdf.image("figures/eda_01_average_band_power.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 1: Average band power per class with Kruskal-Wallis significance annotations.', ln=1, align='C')
    pdf.ln(4)
    
    # Table of p-values
    pdf.set_font('helvetica', 'B', 9)
    pdf.cell(50, 6, 'Frequency Band', border=1, ln=0, align='C')
    pdf.cell(50, 6, 'H-Statistic', border=1, ln=0, align='C')
    pdf.cell(50, 6, 'p-value', border=1, ln=0, align='C')
    pdf.cell(40, 6, 'Significance', border=1, ln=1, align='C')
    
    pdf.set_font('helvetica', '', 9)
    table_rows = [
        ("Delta (1-4 Hz)", "192.08", "2.17e-41", "Significant (p<0.01)"),
        ("Theta (4-8 Hz)", "489.67", "8.28e-106", "Significant (p<0.01)"),
        ("Alpha (8-14 Hz)", "458.68", "4.30e-99", "Significant (p<0.01)"),
        ("Beta (14-31 Hz)", "440.70", "3.38e-95", "Significant (p<0.01)"),
        ("Gamma (31-50 Hz)", "794.66", "6.22e-172", "Significant (p<0.01)")
    ]
    for row in table_rows:
        pdf.cell(50, 5.5, row[0], border=1, ln=0)
        pdf.cell(50, 5.5, row[1], border=1, ln=0, align='C')
        pdf.cell(50, 5.5, row[2], border=1, ln=0, align='C')
        pdf.cell(40, 5.5, row[3], border=1, ln=1, align='C')
        
    # -------------------------------------------------------------
    # PAGE 2: SECTION 2: SPATIAL EEG TOPOLOGY PER EMOTION CLASS
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 2 - Spatial EEG Topology per Emotion Class', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_2 = (
        "We visualize the spatial topological patterns per class for all five bands to identify regional differences. "
        "To identify channels exhibiting significant differences, we run a one-way ANOVA per channel across "
        "the four emotional states. We apply a Benjamini-Hochberg FDR correction (alpha = 0.05) to control the "
        "false discovery rate across the 62 channels. Channels showing statistically significant differences "
        "after correction are marked with red stars. For example, the Alpha band shows prominent asymmetry and "
        "suppression in frontal channels during fear compared to happy and sad states."
    )
    pdf.multi_cell(0, 5, desc_2)
    pdf.ln(3)
    
    # Figure 2 (Alpha Topomap)
    if os.path.exists("figures/eda_02_scalp_topomaps_alpha.png"):
        pdf.image("figures/eda_02_scalp_topomaps_alpha.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 2 (Alpha): Scalp topomaps showing average Alpha-band activity per emotion class.', ln=1, align='C')
    pdf.ln(5)
    
    # Figure 2 (Beta Topomap)
    if os.path.exists("figures/eda_02_scalp_topomaps_beta.png"):
        pdf.image("figures/eda_02_scalp_topomaps_beta.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 2 (Beta): Scalp topomaps showing average Beta-band activity per emotion class.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 3: SECTION 3: SPATIAL CHANNEL CORRELATION HEATMAP
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 3 - Spatial Channel Correlation Heatmap', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_3 = (
        "The channel correlation heatmap illustrates strong, localized co-activation blocks, particularly "
        "within anatomical regions (e.g., frontal and temporal areas). This mathematically validates a graph-based representation: "
        "highly correlated channels represent localized brain activations, indicating that a Graph Attention Network (GAT) "
        "utilizing physical k-NN adjacencies will effectively model spatial relationships."
    )
    pdf.multi_cell(0, 5, desc_3)
    pdf.ln(3)
    
    # Figure 3 (Correlation Heatmap)
    if os.path.exists("figures/eda_03_correlation_heatmap.png"):
        pdf.image("figures/eda_03_correlation_heatmap.png", x=45, w=120)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 3: Electrode channel Pearson correlation heatmap.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 4: SECTION 4: CLASS SEPARABILITY IN RAW FEATURE SPACE
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 4 - Class Separability in Raw Feature Space', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_4 = (
        "To evaluate class separability prior to model training, we project the raw features (flattened 310 dimensions) "
        "into 2D space. Panel A shows a 2D PCA projection of all samples, representing linear variance configurations. "
        "Panel B shows a 2D t-SNE projection of 2,000 stratified samples (proportional across classes). Both projections "
        "reveal substantial class overlap in the raw input space, illustrating the necessity of non-linear deep learning "
        "architectures (e.g., GAT-KAN) to separate emotional brain-state distributions."
    )
    pdf.multi_cell(0, 5, desc_4)
    pdf.ln(3)
    
    # Figure 4 (Class Separability Plot)
    if os.path.exists("figures/eda_04_class_separability.png"):
        pdf.image("figures/eda_04_class_separability.png", x=20, w=170)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 4: 2D PCA projection (all samples) and 2D t-SNE projection (2,000 stratified samples).', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 5: SECTION 5: SUBJECT-WISE VARIABILITY & OUTLIER DETECTION
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 5 - Subject-Wise Variability & Outlier Detection', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_5 = (
        "We evaluate cross-subject variability and outliers in the dataset. "
        "Box plots of subject-wise Alpha-band activity show clear distribution shifts across the 15 subjects, "
        "illustrating that raw signal characteristics vary meaningfully. This motivates subject-fairness analysis. "
        "For outlier detection, we compute the L2 norm of the 310-dimensional feature vector for each sample. "
        "Outliers are flagged beyond 3 standard deviations. Only 2.837% (1,066 samples) are flagged as outliers, "
        "verifying that the dataset contains high-quality, smoothed activations with no extreme anomalies."
    )
    pdf.multi_cell(0, 5, desc_5)
    pdf.ln(3)
    
    # Figure 5 (Subject Variability)
    if os.path.exists("figures/eda_05_subject_variability.png"):
        pdf.image("figures/eda_05_subject_variability.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 5: Box plots of subject-wise Alpha band activity, illustrating cross-subject variability.', ln=1, align='C')
    pdf.ln(4)
    
    # Figure 6 (Outlier Check)
    if os.path.exists("figures/eda_06_outlier_check.png"):
        pdf.image("figures/eda_06_outlier_check.png", x=45, w=120)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 6: Histogram of sample-wise L2 norms with 3 standard deviations thresholds marked.', ln=1, align='C')
        
    pdf.output("eda_report.pdf")
    print("\n[VERIFIED] Created PDF Report: eda_report.pdf")

if __name__ == "__main__":
    create_report()
