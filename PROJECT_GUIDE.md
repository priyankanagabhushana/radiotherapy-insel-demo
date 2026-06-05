# Project Guide — NeuroQA Copilot

**In plain language: what this project does, why it exists, and how every piece fits together.**

---

## The Problem This Solves

When a patient has a brain tumor, doctors can use radiation to shrink it. Before treatment, someone needs to carefully outline the tumor on MRI scans so the radiation beam targets *only* the tumor and avoids healthy brain tissue.

Today, hospitals use **AI programs** to automatically draw these outlines. The AI is fast, but it's not perfect. If the AI makes a mistake near a sensitive area — like the optic nerve (which controls vision) or the brainstem (which controls breathing) — the consequences can be serious.

**This project is a safety net.** It reviews the AI's work *before* a human doctor signs off, flagging cases that need extra attention.

---

## How It Works (Step by Step)

### Step 1: Patient Data
We have MRI scan data from **50 real brain tumor patients** (from the BraTS 2020 medical dataset). For each patient, we know:

- **How big the tumor is** (volume in cubic centimeters)
- **How close the tumor is to critical structures** (distance in millimeters)
- **How confident the AI was** when it drew the tumor outline (a score from 0 to 1 — lower means the AI was unsure)

### Step 2: The Rule Engine
A simple set of rules automatically assigns a risk level to each patient:

| Risk Level | What triggers it |
|------------|-----------------|
| 🔴 **HIGH** | The AI was very unsure (score > 0.75) OR the tumor is dangerously close (< 3mm) to a critical structure |
| 🟡 **MODERATE** | The AI was somewhat unsure (score > 0.50) OR the tumor is large (> 25 cc) |
| 🟢 **LOW** | Everything looks safe |

### Step 3: The Copilot (RAG System)
For each flagged patient, the system searches a **knowledge base** of real clinical guidelines from:

- **AAPM TG-132** — Rules for using AI in radiation planning
- **ESTRO** — European guidelines for auto-contouring
- **QUANTEC** — Dose limits for sensitive organs
- **ASTRO** — Best practices for AI in cancer treatment
- **NRG Oncology** — Rules for protecting memory centers (hippocampus)

It pulls the most relevant guideline excerpts and generates a **recommended action list** — telling the physicist exactly what to double-check.

### Step 4: MRI Visualisation
For each patient, pre-rendered MRI scans are shown with a **color overlay** controlled by an opacity slider:

- 🔴 **Red** = Tumor Core (the dangerous part)
- 🔵 **Blue** = Edema (swelling around the tumor)
- 🟡 **Yellow** = Enhancing Tumor (actively growing)

Three views are shown side-by-side: **axial**, **coronal**, and **sagittal**. Each supports **scroll-to-zoom** and **drag-to-pan** via Plotly. The MRI data is stored as lightweight PNGs (5.1 MB total) rather than the original NIfTI files (4.2 GB).

### Step 5: Statistical Analysis
The Group Statistics tab provides 10 statistical methods:

- Descriptive statistics (mean, median, std, skewness, kurtosis)
- Distribution fitting (Normal / LogNormal / Gamma)
- Correlation matrix (Spearman with significance stars)
- Group comparisons (Kruskal-Wallis)
- Risk factor analysis (Cliff's delta forest plot)
- Bootstrap confidence intervals (10K resamples)
- Outlier detection (IQR method)
- Power analysis (minimum detectable effect)

### Step 6: PDF Report Generation
The system produces two types of PDF reports:

1. **Patient Report** — percentile bar charts, MRI previews, risk triggers, cohort comparison, recommended actions
2. **Group Report** — pie chart, descriptive stats, box plots, correlation heatmap, forest plot, full patient list

All reports are generated with `fpdf2` (pure Python) — no system dependencies needed.

---

## The Files Explained

| File | What it does |
|------|-------------|
| `app.py` | Main Streamlit dashboard. Launch with `streamlit run app.py`. |
| `mri_viewer.py` | Loads MRI PNGs + segmentation masks, creates Plotly charts with scroll-to-zoom. |
| `pdf_report.py` | Generates PDF reports with embedded charts, MRI previews, and tables. |
| `stats_analysis.py` | 10 statistical analysis functions (descriptive stats, correlations, bootstrap, etc.). |
| `knowledge_base.py` | Clinical guideline knowledge base (9 entries from AAPM, ESTRO, QUANTEC, ASTRO, NRG). |
| `render_mri_slices.py` | Regenerates MRI PNGs + mask PNGs + YOLO-Seg contours from NIfTI data. |
| `extract_real_cases.py` | Extracts patient data from BraTS 2020 dataset into CSV. |
| `generate_report.py` | Legacy HTML report generator (replaced by `pdf_report.py`). |
| `real_clinical_queue.csv` | Patient data for 50 patients. Core data file. |
| `mri_previews/` | Pre-rendered MRI PNGs, mask PNGs, and YOLO-Seg contour files (5.1 MB). |
| `reports/` | Sample PDF reports for portfolio/demo. |
| `screenshots/` | App screenshots and demo video. |
| `.streamlit/config.toml` | Dark theme configuration. |
| `requirements.txt` | Python dependencies. |

---

## The Data Flow

```
BraTS 2020 MRI Scans (NIfTI, 4.2 GB)
         │
         ▼
   render_mri_slices.py
         │
         ▼
   mri_previews/ (PNGs, 5.1 MB)  ──────┐
         │                              │
         ▼                              ▼
   real_clinical_queue.csv          app.py
         │                         (Streamlit)
         │                              │
         ├──► pdf_report.py             ├──► Patient Queue (pie chart + table)
         │         │                    ├──► Patient Review (MRI + metrics + PDF)
         │         ▼                    ├──► Group Statistics (charts + stats)
         │    PDF reports               └──► Clinical Reference (guidelines)
         │
         └──► stats_analysis.py
                    │
                    ▼
              Statistical tables + charts
```

---

## Key Thresholds Explained

| Threshold | Value | Why this number? |
|-----------|-------|-----------------|
| Distance to OAR < 3mm | HIGH risk | 3mm is the typical planning margin. If the tumor is closer than this, even a tiny contour error could irradiate the critical structure. |
| AI Uncertainty > 0.75 | HIGH risk | At this level, the AI model has significant doubt about its own contour. |
| AI Uncertainty > 0.50 | MODERATE risk | The AI is somewhat unsure. A spot-check review is recommended. |
| Tumor Volume > 25cc | MODERATE trigger | Large tumors have irregular shapes that AI struggles with. |

---

## Running the Project

### First time setup
```bash
git clone https://github.com/priyankanagabhushana/radiotherapy-qa-copilot.git
cd radiotherapy-qa-copilot
pip install -r requirements.txt
```

### Launch the dashboard
```bash
streamlit run app.py
```
Then open `http://localhost:8501`. The Patient Queue tab opens first.

### Regenerate MRI previews (optional)
Download BraTS 2020 from Kaggle, place in `data/`, then:
```bash
python render_mri_slices.py
```

---

## Who Is This For?

- **Medical Physicists** — Primary users who review AI auto-segmentations before treatment
- **Radiation Oncologists** — Physicians who make the final treatment decision
- **Researchers** — Anyone studying AI-assisted radiotherapy workflows
- **Students** — Learning about clinical AI, RAG systems, and radiation therapy QA

---

## Limitations

- This is a **research prototype**, not a medical device
- The LLM copilot uses template-based text, not a real language model
- All 50 patients have MRI previews (pre-rendered from BraTS 2020)
- All risk thresholds are configurable and should be validated by each institution
- **Never use auto-segmented contours without human verification**
