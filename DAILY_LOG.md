#  Development Log

This file records the complete engineering journey of the project.
It documents planning, implementation, debugging, design decisions, experiments, failures, fixes, and milestones.
The objective is to maintain complete transparency throughout development.

---

## 31 July 2026
### Objective
Begin data acquisition for the SEED-IV EEG dataset.
### Work Completed
- Submitted a request for access to the SEED-IV dataset through the official distribution portal.
### Outcome
Access request submitted; approval pending.
### Notes
>  Next Steps: Await dataset access approval before beginning data preparation.

---

## 1 August 2026
### Objective
Prepare the project workflow while waiting on dataset access.
### Work Completed
- Continued waiting for the dataset access link.
- Planned the overall workflow to be followed for the remainder of the project.
### Outcome
No dataset access yet; project roadmap outlined in the meantime.
### Notes
>  Next Steps: Continue monitoring for dataset access approval.

---

## 2 August 2026
### Objective
Monitor dataset access status.
### Work Completed
- Checked for the dataset access link; access not yet granted.
### Outcome
No change in dataset access status.
### Notes
>  Next Steps: Continue waiting for dataset link.
---

## 3 August 2026

### Objective
Monitor dataset access status.
### Work Completed
- Checked for the dataset access link; access not yet granted.
### Outcome
No change in dataset access status.
### Notes
>  Next Steps: Continue waiting for dataset link.

---

## 4 August 2026

### Dataset Acquisition
Received the SEED-IV dataset.
### Work Completed
- Obtained access to the SEED-IV dataset.
### Outcome
Milestone Achieved — Dataset access granted.
### Notes
>  Next Steps: Locate the dataset files and finalize the research direction.

---

## 5 August 2026
### Planning
Reviewed the guide's project brief (multi-band EEG transformer) and selected the paper direction.
- Defined 4–5 novel contributions: faithfulness-verified XAI, calibrated uncertainty, subject fairness, and neuroscience-grounded validation.
> **Engineering Decision:** Direction finalized as EEG-only (no multimodal input), using a GAT + cross-band Transformer + KAN head architecture, benchmarked against a CNN-LSTM baseline per the guide's instruction.
### Documentation
Built a formal Project Proposal & Research Plan document covering title, objectives, methodology, novelty table, baseline tiers, metrics, and an ethics section.
### Work Completed
- `EEG_Emotion_Project_Report.docx` created.
- Document later expanded with an Ethical / Responsible-AI Implementation Roadmap and a baseline-comparison clause.
### Outcome
Milestone Achieved — Research direction and formal project proposal finalized.
### Data Access
Located the dataset on the SJTU cloud File Station (SEED_IV); noted a 14-day link expiry warning.
- Decided to download `eeg_feature_smooth/` (Differential Entropy features) plus small metadata files.
- Deferred `eeg_raw_data/` and eye-tracking data as unnecessary for the EEG-only scope.
Read `ReadMe.txt` and recorded the session-wise emotion label order (24 trials × 3 sessions, labels differ per session).
### Outcome
Session label mapping documented and saved for use in the data loader: `0 = neutral, 1 = sad, 2 = fear, 3 = happy`.
### Notes
>  Next Steps: Begin downloading the dataset files.

---

## 6 August 2026
### Data Download
Downloaded `eeg_feature_smooth/1`, `/2`, `/3` (15 subject files each), spanning Aug 5–6.
### Issue
Identified a missing Subject 7 (session 1) file and a duplicate-download filename issue.
### Resolution
Re-downloaded the missing file and corrected the duplicate filename.
### Outcome
 Milestone Achieved — All 3 sessions confirmed complete: 15/15 subject files each.
---
### Environment Setup
Installed Python 3.11.9 alongside the existing 3.14.2 installation using the Python Install Manager (`py install 3.11`).
### Work Completed
- Created a project-local virtual environment (`venv`) inside `eri/`.
- Installed `scipy`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ipykernel`.
### Outcome
Working, isolated Python 3.11.9 environment confirmed via `sys.executable` / `sys.version` inside `first.ipynb`.
---
### Data Inspection
Loaded a sample `.mat` file via `scipy.io.loadmat` and inspected the variable structure.
### Outcome
Confirmed structure: `de_LDS1`–`de_LDS24` per file, shape `(62 channels, variable time-windows, 5 bands)`.
>  **Engineering Decision:** Selected `de_LDS` (Differential Entropy, LDS-smoothed) as the feature to use, over the `de_movingAve` / `psd` variants.
---
### Pipeline Development
### Issue
Debugging file-path / working-directory issues in the VS Code notebook surfaced a multi-minute, unexplained load hang.
### Root Cause
OneDrive was suspected as the cause but was ruled out on inspection; the root cause was not conclusively identified.
### Resolution
Decided to hand off robust loader implementation to an AI coding agent (Antigravity) rather than continue manual debugging.
### Work Completed
- Directed Antigravity to build `load_seed_iv.py`, which:
  - Loads all 45 `.mat` files.
  - Extracts `de_LDS` features.
  - Attaches session-correct labels.
  - Splits each trial into individual time-window samples.
  - Tags each sample with subject / session / trial IDs.
  - Saves the result to `seed_iv_processed.npz`.
### Outcome
Milestone Achieved — `seed_iv_processed.npz` created successfully.
---
### Validation
Ran an independent verification script, `verify_seed_iv.py`, checking shapes, class balance, subject balance, NaN/Inf integrity, and sample values.
### Outcome
 Milestone Achieved — **DATA VALIDATED OK**
| Check | Result |
|---|---|
| Total samples | 37,575 |
| Classes | 4 (balanced-ish) |
| Subjects | 15/15 present, 2,505 samples each |
| NaN / Inf / all-zero issues | None |
| Load time | 1.46s |
---
### Baseline Experiments
Directed Antigravity to build an SVM baseline (RBF kernel) under two protocols: subject-dependent (per-subject 80/20 split) and cross-subject (train on subjects 1–12, test on 13–15).
### Issue
First run produced a subject-dependent accuracy of **99.69%**, flagged as implausible (SEED-IV literature typically reports 70–80%).
### Root Cause
The 80/20 split was performed at the individual time-window level, allowing near-duplicate adjacent windows from the same trial to leak across the train/test split — classic data leakage.
### Resolution
Issued a corrected instruction: the split must occur at the trial level, so all windows of a trial are confined to either train or test.
### Work Completed
- Re-ran the SVM baseline with a trial-level stratified 80/20 split (56/16 trials per subject).
### Outcome
Milestone Achieved — Corrected, literature-consistent results obtained (see Section 4); accepted as the first valid baseline result.
### Notes
>  Next Steps: Proceed to cross-subject baseline evaluation and begin architecture implementation (GAT + cross-band Transformer + KAN head).
---
## Lessons Learned
-  **Engineering Decision:** Time spent waiting on external dataset approval was used productively to plan the project workflow and research direction in advance, rather than left idle.
-  **Engineering Decision:** An implausibly high baseline accuracy (99.69%) was treated as a red flag rather than a result, prompting an investigation that uncovered trial-level data leakage — a reminder to sanity-check results against literature before accepting them.
- 💡 **Engineering Decision:** Handed off routine but error-prone implementation work (data loading, verification, baseline modeling) to an AI coding agent, while retaining ownership of experimental design, validation, and error diagnosis.
