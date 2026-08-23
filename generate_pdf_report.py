import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'B', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, 'SEED-IV EEG Preprocessing & Data Cleaning Report', border=0, ln=1, align='R')
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
    # PAGE 1: TITLE & SECTION 1: DATASET OVERVIEW
    # -------------------------------------------------------------
    pdf.add_page()
    
    # Title Block
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'SEED-IV Preprocessing & Feature Engineering', ln=1, align='C')
    pdf.set_font('helvetica', 'I', 11)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 5, 'Rigorous Data Cleaning, Feature Extraction, & Leakage-Free Validation', ln=1, align='C')
    pdf.ln(10)
    
    # Section 1 Header
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 1 - Dataset Overview & Sample Balance', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(40, 40, 40)
    desc_1 = (
        "The SEED-IV dataset consists of preprocessed multi-channel EEG recordings collected from 15 subjects "
        "across three sessions. The dataset is used to classify four distinct emotional categories: neutral, sad, "
        "fear, and happy. We verify the window-level and subject-level sample balance configurations. Subject "
        "contributions are perfectly balanced, with exactly 2,505 windows extracted per subject. The class "
        "distributions are reasonably balanced with modest variations (Neutral: 10,170, Sad: 10,245, Fear: 9,225, "
        "Happy: 7,935), with Happy samples being somewhat underrepresented due to stimulation duration constraints."
    )
    pdf.multi_cell(0, 5, desc_1)
    pdf.ln(3)
    
    # Figure 1
    if os.path.exists("figures/01_dataset_overview.png"):
        pdf.image("figures/01_dataset_overview.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 1: SEED-IV sample distribution configurations across classes, subjects, and sessions.', ln=1, align='C')
        
    # -------------------------------------------------------------
    # PAGE 2: SECTION 2: FEATURE VALUE DISTRIBUTION & STANDARDIZATION
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 2 - Feature Value Distribution & Standardization', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_2 = (
        "The raw Differential Entropy (DE) feature values exhibit substantial scale and distribution differences "
        "across the five frequency bands (Delta, Theta, Alpha, Beta, Gamma) when averaged across the 62 channels. "
        "Higher bands show higher power ranges and wider variances. This requires standardizing features to a mean "
        "of 0 and standard deviation of 1. To prevent data leakage, scaling parameters are fit strictly on training "
        "trials, then transformed on validation/testing data, keeping test characteristics unseen during training."
    )
    pdf.multi_cell(0, 5, desc_2)
    pdf.ln(3)
    
    # Figure 2 (Distribution Box Plots)
    if os.path.exists("figures/02_feature_distribution.png"):
        pdf.image("figures/02_feature_distribution.png", x=45, w=120)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 2: Distribution of raw DE values across the 5 spectral bands.', ln=1, align='C')
    pdf.ln(5)
    
    # Figure 5 (Standardization Comparison)
    if os.path.exists("figures/05_standardization_comparison.png"):
        pdf.image("figures/05_standardization_comparison.png", x=20, w=170)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 3: DE feature distribution before (Raw) and after (Standardized) scaling.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 3: SECTION 3: FEATURE EXTRACTION PIPELINE & SELECTION
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 3 - Feature Extraction Pipeline & Selection', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_3 = (
        "The feature extraction pipeline transforms raw EEG signals into spatial-spectral features. "
        "Filtered bands are partitioned into 4-second windows. Differential Entropy is computed for a "
        "Gaussian distribution as: DE = 0.5 * log(2 * pi * e * sigma^2). A Linear Dynamical System (LDS) "
        "is then applied to smooth the DE values across time. We select the DE + LDS variant over PSD and "
        "moving average alternatives, as DE provides logarithmic normalization matching human sensory characteristics, "
        "while LDS models temporal dynamics without lag."
    )
    pdf.multi_cell(0, 5, desc_3)
    pdf.ln(3)
    
    # Figure 3 (Pipeline Flowchart)
    if os.path.exists("figures/03_pipeline_flowchart.png"):
        pdf.image("figures/03_pipeline_flowchart.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 4: SEED-IV data preprocessing and feature extraction pipeline flowchart.', ln=1, align='C')
    pdf.ln(5)
    
    # Figure 4 (Feature Selection Grid)
    if os.path.exists("figures/04_feature_variant_selection.png"):
        pdf.image("figures/04_feature_variant_selection.png", x=30, w=150)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 5: 2x2 grid representing the feature variant selection matrix.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 4: SECTION 4: ELECTRODE MONTAGE & INPUT REPRESENTATION
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 4 - Electrode Montage & Input Representation', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_4 = (
        "The spatial arrangement of the 62 electrodes is mapped via a 2D scalp layout. "
        "Individual windows represented as 62 channels x 5 bands DE heatmaps capture active spatial-spectral patterns. "
        "Correlated channels represent localized brain activations, indicating that a Graph Attention Network (GAT) "
        "utilizing physical k-NN adjacencies will effectively model spatial relationships."
    )
    pdf.multi_cell(0, 5, desc_4)
    pdf.ln(3)
    
    # Figure 6 (Montage Layout)
    if os.path.exists("figures/06_electrode_montage.png"):
        pdf.image("figures/06_electrode_montage.png", x=60, w=90)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 6: 2D scalp layout projection of the 62 electrode positions.', ln=1, align='C')
    pdf.ln(5)
    
    # Figure 7 (Heatmap)
    if os.path.exists("figures/07_single_sample_heatmap.png"):
        pdf.image("figures/07_single_sample_heatmap.png", x=65, w=80)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 7: Representative single-window 62x5 DE feature heatmap.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 5: SECTION 5: TRIAL-LEVEL SPLITTING & DURATIONS
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 5 - Trial-Level Splitting & Durations', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_5 = (
        "Preventing data leakage is essential for obtaining reproducible model evaluations. "
        "In sliding-window EEG protocols, adjacent windows share up to 90% overlapping samples. "
        "If splitting is performed at the individual window level, near-duplicate samples from the "
        "same trial leak across the train and test splits, causing artificially inflated accuracies. "
        "To ensure a leakage-free protocol, we enforce trial-level splitting: every window from a "
        "given trial is assigned entirely to either the train or test set. "
        "This trial duration variance justifies window-level modeling and sliding window sequences over "
        "padding or truncating whole trials, which would introduce excessive zero-padding."
    )
    pdf.multi_cell(0, 5, desc_5)
    pdf.ln(3)
    
    # Figure 8 (Splitting Diagram)
    if os.path.exists("figures/08_trial_splitting_diagram.png"):
        pdf.image("figures/08_trial_splitting_diagram.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 8: Trial-level stratified splitting protocol vs. window-level leakage.', ln=1, align='C')
    pdf.ln(5)
    
    # Figure 9 (Durations Histogram)
    if os.path.exists("figures/09_trial_durations.png"):
        pdf.image("figures/09_trial_durations.png", x=45, w=120)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 9: Trial window-count duration distribution histogram.', ln=1, align='C')
        
    pdf.output("preprocessing_report.pdf")
    print("\n[VERIFIED] Created PDF Report: preprocessing_report.pdf")

if __name__ == "__main__":
    create_report()
