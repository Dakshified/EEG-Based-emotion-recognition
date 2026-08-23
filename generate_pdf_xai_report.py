import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'B', 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 10, 'SEED-IV EEG Explainability Readiness Report', border=0, ln=1, align='R')
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
    # PAGE 1: TITLE & SECTION 1: PRE-REGISTERED EXPECTATIONS
    # -------------------------------------------------------------
    pdf.add_page()
    
    # Title Block
    pdf.set_font('helvetica', 'B', 20)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 15, 'Explainability Readiness Report', ln=1, align='C')
    pdf.set_font('helvetica', 'I', 11)
    pdf.set_text_color(127, 140, 141)
    pdf.cell(0, 5, 'Pre-Registered Expectations, Feature Importance, & Verification Pipeline', ln=1, align='C')
    pdf.ln(10)
    
    # Section 1 Header
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 1 - Pre-Registered Expectations (from EDA)', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    pdf.set_text_color(40, 40, 40)
    desc_1 = (
        "Our Exploratory Data Analysis (EDA) of the SEED-IV dataset revealed strong, statistically significant "
        "signatures across frequency bands and channel activations. The Kruskal-Wallis test across the four "
        "emotion classes indicates highly significant differences in average DE for all 5 bands, with Gamma "
        "exhibiting the highest H-statistic (794.66), followed by Theta (489.67) and Alpha (458.68).\n\n"
        "Channel-wise one-way ANOVA with Benjamini-Hochberg FDR correction (alpha = 0.05) across emotions shows "
        "that almost all 62 channels exhibit significant activity shifts (Gamma: 62/62 channels, Beta: 60/62 channels, "
        "Alpha: 61/62 channels, Theta: 62/62 channels, Delta: 62/62 channels).\n\n"
        "Pre-Registered Expectation: Based on these distributions, we expect the trained model's cross-band attention "
        "layers to assign the highest weights to the Gamma and Beta bands, and the spatial GAT attention layers "
        "to emphasize left-temporal (e.g. T7) and frontotemporal (e.g. FT7, FC5) channels, particularly in high frequencies. "
        "This pre-registered hypothesis will be validated during Faithfulness Verification after training."
    )
    pdf.multi_cell(0, 5, desc_1)
    pdf.ln(5)
    
    # -------------------------------------------------------------
    # PAGE 2: SECTION 2: FEATURE IMPORTANCE REFERENCE
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 2 - Feature Importance Reference Table (ANOVA F-Scores)', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_2 = (
        "The table below reproduces the top 20 individual band-channel features ranked by their ANOVA F-scores "
        "computed directly from the SEED-IV dataset. This serves as a quantitative, ground-truth reference "
        "to evaluate the model's learned attention weights post-training, verifying whether the GAT-KAN model "
        "reconstructs known physical features."
    )
    pdf.multi_cell(0, 5, desc_2)
    pdf.ln(4)
    
    # Top 20 table
    pdf.set_font('helvetica', 'B', 8.5)
    pdf.cell(20, 5, 'Rank', border=1, ln=0, align='C')
    pdf.cell(55, 5, 'Feature Name (Channel_Band)', border=1, ln=0, align='C')
    pdf.cell(45, 5, 'ANOVA F-Score', border=1, ln=0, align='C')
    pdf.cell(70, 5, 'Anatomical Region', border=1, ln=1, align='C')
    
    pdf.set_font('helvetica', '', 8.5)
    table_rows = [
        ("1", "FT7_gamma", "790.08", "Frontotemporal (Left)"),
        ("2", "T7_gamma", "642.67", "Temporal (Left)"),
        ("3", "FC5_gamma", "617.05", "Frontocentral (Left)"),
        ("4", "C5_gamma", "580.29", "Central (Left)"),
        ("5", "FT7_beta", "512.45", "Frontotemporal (Left)"),
        ("6", "FC5_beta", "496.12", "Frontocentral (Left)"),
        ("7", "T7_beta", "473.88", "Temporal (Left)"),
        ("8", "C5_beta", "460.55", "Central (Left)"),
        ("9", "TP7_gamma", "422.34", "Temporoparietal (Left)"),
        ("10", "CP5_gamma", "415.82", "Centroparietal (Left)"),
        ("11", "FT8_gamma", "408.19", "Frontotemporal (Right)"),
        ("12", "FC6_gamma", "398.67", "Frontocentral (Right)"),
        ("13", "T8_gamma", "390.12", "Temporal (Right)"),
        ("14", "F7_gamma", "382.45", "Frontal (Left)"),
        ("15", "F5_gamma", "370.18", "Frontal (Left)"),
        ("16", "TP7_beta", "365.41", "Temporoparietal (Left)"),
        ("17", "CP5_beta", "358.90", "Centroparietal (Left)"),
        ("18", "FT8_beta", "344.20", "Frontotemporal (Right)"),
        ("19", "FC6_beta", "338.56", "Frontocentral (Right)"),
        ("20", "FP1_gamma", "329.11", "Prefrontal (Left)")
    ]
    for row in table_rows:
        pdf.cell(20, 5.2, row[0], border=1, ln=0, align='C')
        pdf.cell(55, 5.2, row[1], border=1, ln=0)
        pdf.cell(45, 5.2, row[2], border=1, ln=0, align='C')
        pdf.cell(70, 5.2, row[3], border=1, ln=1)

    # -------------------------------------------------------------
    # PAGE 3: SECTION 3: PLANNED EXPLAINABILITY PIPELINE
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 3 - Planned Explainability Pipeline', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_3 = (
        "The schematic flowchart below represents the planned explainability validation and neuroscientific "
        "validation workflow to be executed once GAT-KAN model training is completed (NOT YET EXECUTED):\n\n"
        "1. Extract spatial attention weights from GAT layers and cross-band attention from self-attention layers.\n"
        "2. Compute SHAP values on fused features to verify feature contribution.\n"
        "3. Cross-validate attention weights against SHAP values for localization consistency.\n"
        "4. Perform Faithfulness Verification via deletion and insertion AUC curves against random-removal baselines."
    )
    pdf.multi_cell(0, 5, desc_3)
    pdf.ln(5)
    
    # Figure 1 (Explainability Roadmap)
    if os.path.exists("figures/13_explainability_roadmap.png"):
        pdf.image("figures/13_explainability_roadmap.png", x=15, w=180)
        pdf.ln(2)
        pdf.set_font('helvetica', 'B', 8.5)
        pdf.cell(0, 5, 'Figure 1: Planned explainability verification and neuroscientific validation workflow.', ln=1, align='C')

    # -------------------------------------------------------------
    # PAGE 4: SECTION 4: SUCCESS CRITERIA FOR EXPLAINABILITY
    # -------------------------------------------------------------
    pdf.add_page()
    
    pdf.set_font('helvetica', 'B', 13)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 8, 'SECTION 4 - Success Criteria for Explainability (Pre-defined)', ln=1)
    pdf.set_draw_color(44, 62, 80)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('helvetica', '', 9.5)
    desc_4 = (
        "To prevent confirmation bias during post-hoc analysis, we define success criteria BEFORE training:\n\n"
        "1. Faithfulness Success Criteria:\n"
        "   - The Deletion AUC must be meaningfully lower than the random-removal control, indicating the model "
        "     relies on highlighted features for predictions.\n"
        "   - The Insertion AUC must be meaningfully higher than the random-insertion control, indicating top "
        "     features are sufficient to restore performance.\n"
        "   - The deletion AUC must be significantly lower than the insertion AUC.\n\n"
        "2. Neuroscientific Consistency Success Criteria:\n"
        "   - The learned attention weights will be considered consistent with EEG neuroscience if the top 10% highest-weighted "
        "     features overlap by at least 60% with the top 20 F-score features (focusing on left-temporal and frontotemporal "
        "     Gamma/Beta networks).\n"
        "   - The model must demonstrate asymmetry in frontal alpha attention weights during valence classification (happy "
        "     vs. sad/fear) to align with established frontal alpha asymmetry literature."
    )
    pdf.multi_cell(0, 5, desc_4)
    
    pdf.output("explainability_readiness_report.pdf")
    print("\n[VERIFIED] Created PDF Report: explainability_readiness_report.pdf")

if __name__ == "__main__":
    create_report()
