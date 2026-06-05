# NeuroQA Copilot

**Human-in-the-Loop Radiotherapy QA Decision Support for Brain Tumor Auto-Segmentation**

> Research Prototype — Not for clinical use

---

## Live Demo

**App:** https://radiotherapy-insel-test.streamlit.app

**Demo Video:** `screenshots/neuroqa_demo.mp4` (38s walkthrough)

**Sample Reports:** `reports/` folder (PDF with charts + MRI previews)

---

## Overview

NeuroQA Copilot is a Streamlit-based clinical decision-support dashboard that helps medical physicists review AI-generated brain tumor segmentations before radiation therapy treatment planning.

- **Rule-based risk engine** — triages 50 patients by tumor volume, proximity to critical structures (OAR), and AI uncertainty scores
- **RAG-simulated knowledge base** — retrieves clinical guidelines (AAPM TG-132, ESTRO, QUANTEC, ASTRO, NRG) matched to each patient's risk triggers
- **Interactive MRI viewer** — renders BraTS 2020 brain scans with segmentation overlays (Core / Edema / Enhancing) using Plotly with scroll-to-zoom
- **PDF reports** — generates per-patient and group reports with embedded charts, MRI previews, and statistical tables
- **10 statistical methods** — descriptive stats, distribution fitting, Spearman correlations, Kruskal-Wallis, bootstrap CIs, risk factor analysis, outlier detection, power analysis

---

## Features

| Tab | What it shows |
|-----|---------------|
| **Patient Queue** (default) | Risk distribution pie chart, patient selector with risk badge, sortable dataframe — click any row to review |
| **Patient Review** | 3-plane MRI viewer (axial/coronal/sagittal) with opacity slider, risk metrics, percentile comparison, PDF download |
| **Group Statistics** | Plotly bar/scatter charts (click to navigate), box plots, correlation heatmap, forest plot, distribution fitting, bootstrap CIs |
| **Clinical Reference** | 9 indexed guideline entries from AAPM, ESTRO, QUANTEC, ASTRO, NRG Oncology |

---

## Project Structure

```
radiotherapy-qa-copilot/
├── app.py                     # Streamlit dashboard (main entry point)
├── mri_viewer.py              # MRI rendering: PNG + mask overlay, Plotly charts
├── pdf_report.py              # PDF generation with charts, MRI previews, tables
├── stats_analysis.py          # 10 statistical analysis functions
├── knowledge_base.py          # Clinical guideline RAG knowledge base
├── real_clinical_queue.csv    # Patient data (50 patients from BraTS 2020)
├── mri_previews/              # Pre-rendered MRI PNGs + segmentation masks (5.1 MB)
│   └── BraTS20_Training_XXX/
│       ├── axial.png          # Grayscale MRI slice
│       ├── axial_mask.png     # Segmentation mask (0/1/2/4 pixel values)
│       └── axial.txt          # YOLO-Seg contour polygons
├── reports/                   # Sample PDF reports
│   ├── NeuroQA_Group_Report.pdf
│   └── NeuroQA_Patient_*.pdf
├── screenshots/               # App screenshots and demo video
│   └── neuroqa_demo.mp4
├── render_mri_slices.py       # Regenerate MRI previews from NIfTI data
├── extract_real_cases.py      # Extract patient data from BraTS dataset
├── generate_report.py         # Legacy HTML report generator
├── knowledge_base.py          # Clinical guidelines knowledge base
├── .streamlit/config.toml     # Dark theme configuration
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- No MRI data needed — pre-rendered PNGs are included in `mri_previews/`

### Installation

```bash
git clone https://github.com/priyankanagabhushana/radiotherapy-qa-copilot.git
cd radiotherapy-qa-copilot
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The Patient Queue tab opens first with a pie chart and patient selector.

### Generate PDF Reports

PDF reports are generated on-the-fly via the download buttons in the app. To generate sample reports programmatically:

```python
import pandas as pd
from pdf_report import generate_patient_pdf, generate_group_pdf

df = pd.read_csv('real_clinical_queue.csv')
df['Risk_Level'] = df.apply(compute_risk_level, axis=1)

# Individual patient report
pdf_bytes = generate_patient_pdf(df.iloc[0], df)

# Group report
pdf_bytes = generate_group_pdf(df)
```

---

## Regenerating MRI Previews

Pre-rendered MRI PNGs are included in the repo. To regenerate from NIfTI data:

1. Download BraTS 2020 from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
2. Extract into `data/` or `brats_real/`
3. Run:
```bash
python render_mri_slices.py
```

This generates 9 files per patient (3 PNGs + 3 mask PNGs + 3 YOLO-Seg contour TXTs) in `mri_previews/`.

---

## Risk Engine

Each patient is classified by three parameters:

| Parameter | HIGH Threshold | MODERATE Threshold |
|-----------|---------------|-------------------|
| AI Uncertainty Score | > 0.75 | > 0.50 |
| Distance to OAR | < 3.0 mm | — |
| Tumor Volume | — | > 25.0 cc |

A patient is **HIGH** if any HIGH threshold is breached, **MODERATE** if any MODERATE threshold is breached, otherwise **LOW**.

---

## Statistical Methods

| Method | Purpose |
|--------|---------|
| Descriptive statistics | Mean, median, std, skewness, kurtosis, CV% |
| Shapiro-Wilk normality | Test if data follows normal distribution |
| Spearman correlation | Rank-based correlations with Bonferroni correction |
| Kruskal-Wallis | Non-parametric group comparisons with eta-squared |
| Cliff's delta | Robust effect size (HIGH vs non-HIGH) |
| Mann-Whitney U | Univariate risk factor association + AUC |
| Bootstrap CIs | 10,000-resample confidence intervals |
| Distribution fitting | Normal/LogNormal/Gamma via MLE + KS test + AIC |
| Outlier detection | IQR and Z-score methods |
| Power analysis | Minimum detectable effect size |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Dashboard | Streamlit 1.58+ with dark theme |
| Interactive charts | Plotly (scroll-to-zoom, click-to-navigate) |
| Statistical charts | Matplotlib |
| PDF reports | fpdf2 + Matplotlib (pure Python, no system deps) |
| MRI processing | PIL/Pillow, NumPy, SciPy |
| Data | BraTS 2020 Challenge (Kaggle) |
| Clinical guidelines | AAPM TG-132, ESTRO, QUANTEC, ASTRO, NRG Oncology |

---

## Deployment

### Streamlit Community Cloud (free)

1. Push to GitHub
2. Go to https://share.streamlit.io
3. Connect repository: `priyankanagabhushana/radiotherapy-qa-copilot`
4. Branch: `main`, Main file: `app.py`
5. Click Deploy

All dependencies are pure Python — no system packages needed. MRI data is not required (pre-rendered PNGs are in the repo).

---

## Clinical Guidelines Referenced

- **AAPM TG-132** — Use of Image Registration and Fusion Algorithms in Radiation Oncology (2017)
- **ESTRO** — Guideline on Automated Contouring in Radiotherapy (2020)
- **QUANTEC** — Spinal Cord / Brainstem Dose Constraints (2010)
- **ASTRO** — AI in Radiation Oncology Best Practices (2023)
- **NRG Oncology CC001** — Hippocampal Avoidance Guidelines (2018)

---

## License

Research prototype. Not intended for clinical use. All flagged contours require human verification before treatment delivery.

## Acknowledgments

- [BraTS 2020 Challenge](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) for the brain tumor MRI dataset
- AAPM, ESTRO, QUANTEC, ASTRO, and NRG Oncology for clinical guideline references
