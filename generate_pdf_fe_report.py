import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'B', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, 'SEED-IV EEG Feature Engineering & Selection Report', border=0, ln=1, align='R')
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
    # PAGE 1: TITLE & SECTION 1: QUANTITATIVE FEATURE IMPORTANCE
    # -------------------------------------------------------------
    pdf.add_page()
    
    # Title Block
    pdf.set_font('helvetica', 'B', 22)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'Feature Engineering & Selection Report', ln=1, align='C')
    pdf.set_font('helvetica', 'I', 11)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 5, 'Decision Layouts, Quantitative Rankers, & Spatial Graphs', ln=1, align='C')
    pdf.ln(10)
    
    # Section 1 Header
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 1 - Quantitative Feature Importance Ranking', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(40, 40, 40)
    desc_1 = (
        "We compute ANOVA F-scores for all 310 individual features (62 channels x 5 bands) to quantitatively "
        "verify which spectral-spatial regions carry the most discriminative emotional signatures. "
        "The ranking results reveal that high-frequency channels (Gamma and Beta bands) located in the temporal "
        "and frontal regions (such as FT7, T7, and FC5) exhibit the highest F-scores, providing solid "
        "statistical justification for the discriminability of high-frequency spatial networks."
    )
    pdf.multi_cell(0, 5, desc_1)
    pdf.ln(3)
    
    # Figure 1
    if os.path.exists("figures/fe_01_feature_importance.png"):
        pdf.image("figures/fe_01_feature_importance.png", x=25, w=150)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 1: Top 20 most discriminative band-channel features sorted by ANOVA F-score.', ln=1, align='C')
    pdf.ln(4)
    
    # Table of top features
    pdf.set_font('helvetica', 'B', 9)
    pdf.cell(30, 6, 'Rank', border=1, ln=0, align='C')
    pdf.cell(60, 6, 'Feature Name (Channel_Band)', border=1, ln=0, align='C')
    pdf.cell(50, 6, 'ANOVA F-Score', border=1, ln=0, align='C')
    pdf.cell(50, 6, 'Anatomical Region', border=1, ln=1, align='C')
    
    pdf.set_font('helvetica', '', 9)
    table_rows = [
        ("1", "FT7_gamma", "790.08", "Frontotemporal (Left)"),
        ("2", "T7_gamma", "642.67", "Temporal (Left)"),
        ("3", "FC5_gamma", "617.05", "Frontocentral (Left)"),
        ("4", "C5_gamma", "580.29", "Central (Left)"),
        ("5", "FT7_beta", "512.45", "Frontotemporal (Left)")
    ]
    for row in table_rows:
        pdf.cell(30, 5.5, row[0], border=1, ln=0, align='C')
        pdf.cell(60, 5.5, row[1], border=1, ln=0, align='C')
        pdf.cell(50, 5.5, row[2], border=1, ln=0, align='C')
        pdf.cell(50, 5.5, row[3], border=1, ln=1, align='C')
        
    # -------------------------------------------------------------
    # PAGE 2: SECTION 2: BAND-WISE FEATURE IMPORTANCE
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 2 - Band-Wise Aggregate Feature Importance', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_2 = (
        "Aggregating feature F-scores within each of the 5 frequency bands reveals that the Gamma band "
        "possesses the highest aggregate discriminative power (mean F-score = 147.34), followed closely by the "
        "Beta band (mean F-score = 93.40). The lower bands (Delta, Theta, Alpha) have lower mean F-scores but "
        "remain statistically significant. This spectral disparity motivates our cross-band attention design, "
        "which allows the GAT-KAN model to adaptively weight the interactions across frequency bands."
    )
    pdf.multi_cell(0, 5, desc_2)
    pdf.ln(3)
    
    # Figure 2 (Band Importance Bar Chart)
    if os.path.exists("figures/fe_02_importance_by_band.png"):
        pdf.image("figures/fe_02_importance_by_band.png", x=40, w=130)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 2: Mean feature discriminative power per frequency band (ANOVA F-score).', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 3: SECTION 3: FEATURE REPRESENTATION DESIGN
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 3 - Feature Representation Design layouts', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_3 = (
        "We evaluate representation strategies for feeding the $62 \times 5$ features into different deep learning architectures. "
        "Standard models (SVM, CNN-LSTM, Transformer baselines) flatten the features into a single 310-dimensional vector. "
        "This concatenative approach loses structural and geographical boundaries across spectral bands. "
        "To preserve these boundaries, our proposed GAT-KAN model structures the features as 5 separate band tokens "
        "of 62 channels each. This enables spatial graph attention layers to process channels independently per band, "
        "followed by cross-band self-attention to map interactions between the bands."
    )
    pdf.multi_cell(0, 5, desc_3)
    pdf.ln(3)
    
    # Figure 3 (Representation Layout Diagram)
    if os.path.exists("figures/fe_03_representation_design.png"):
        pdf.image("figures/fe_03_representation_design.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 3: Feature representation strategies: Concatenated (flat) vs. Band-Separated tokens.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 4: SECTION 4: k-NN GRAPH SPATIAL ADJACENCIES
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 4 - k-NN Graph Spatial Adjacency Construction', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_4 = (
        "Graph neural networks require an adjacency structure to define message-passing paths. "
        "Using 2D Cartesian scalp coordinates from channel_62_pos (1).locs, we construct a sparse physical "
        "k-nearest-neighbors (k-NN) graph with $k=8$ neighbors per electrode. "
        "This physical constraint allows the GAT-KAN model to perform local message-passing, simulating actual "
        "biological cortical propagation and avoiding the over-smoothing and noise vulnerability seen in fully-connected "
        "dense graphs or learnable dense layouts."
    )
    pdf.multi_cell(0, 5, desc_4)
    pdf.ln(3)
    
    # Figure 4 (k-NN Graph Montage)
    if os.path.exists("figures/fe_04_graph_construction.png"):
        pdf.image("figures/fe_04_graph_construction.png", x=45, w=120)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 4: 2D scalp projection showing sparse k-NN physical edges (k=8) connecting electrodes.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 5: SECTION 5: SEQUENCE WINDOWING & SUMMARY TABLE
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 5 - Sequence Windowing & Final Feature Configurations', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_5 = (
        "For models that capture temporal dynamics (such as CNN-LSTM), we implement a sequence windowing protocol. "
        "A sliding window of length 10 and stride 1 is used to slice overlapping window sequences from a trial's timeline. "
        "For non-sequential models (SVM, DGCNN, Transformer, GAT-KAN), samples are processed as individual static windows. "
        "The table below details the final feature configuration parameters used across our experiments, including the "
        "essential distinction that GAT-KAN utilizes a sparse physical k-NN graph while DGCNN employs a fully-learnable dense adjacency."
    )
    pdf.multi_cell(0, 5, desc_5)
    pdf.ln(3)
    
    # Figure 5 (Sequence Windowing Timeline)
    if os.path.exists("figures/fe_05_sequence_windowing.png"):
        pdf.image("figures/fe_05_sequence_windowing.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 5: Timeline diagram comparing sequence windowing (CNN-LSTM) vs. static window representations.', ln=1, align='C')
    pdf.ln(4)
    
    # Figure 6 (Summary Table Image)
    if os.path.exists("figures/fe_06_summary_table.png"):
        pdf.image("figures/fe_06_summary_table.png", x=20, w=170)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 6: Final EEG Feature Space Configurations Summary Table.', ln=1, align='C')
        
    pdf.output("feature_engineering_report.pdf")
    print("\n[VERIFIED] Created PDF Report: feature_engineering_report.pdf")

if __name__ == "__main__":
    create_report()
