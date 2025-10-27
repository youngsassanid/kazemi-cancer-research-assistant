# Kazemi Cancer Research Assistant

> **AI-Powered Oncology Research Platform**  
> *Prognosis prediction • Treatment recommendations • Clinical trial matching • Drug interaction checker*

**[Live Demo](https://youngsassanid.github.io/kazemi-cancer-research-assistant/)** | [Report Issues](https://github.com/youngsassanid/kazemi-cancer-research-assistant/issues)

---

## Features

| Feature | Description |
|-------|-----------|
| **Prognosis Prediction** | Ensemble ML models (RF, XGB, NN) predict 5-year survival and risk score |
| **Treatment Intelligence** | NCCN-aligned protocols by cancer type, stage, and biomarkers |
| **Drug Interaction Checker** | Real -time severity scoring and safer alternatives |
| **Clinical Trial Matcher** | Filters trials by age, stage, and biomarker eligibility |
| **Survival Analysis** | Interactive Kaplan-Meier curves with Plotly.js |
| **EHR Integration** | Secure SQLite-backed patient data management |

---

## Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6), Plotly.js
- **Styling**: Custom CSS with Google Fonts (Inter)
- **Backend (Demo)**: Python 3.11+ with `cancer-research-assistant.py`
- **ML Models**: Scikit-learn, XGBoost, TensorFlow/Keras
- **Data**: Synthetic cohort + DrugBank + ClinicalTrials.gov API
- **Hosting**: GitHub Pages (static), Python (local demo)

---

## Project Structure

```bash
kazemi-cancer-research-assistant/
├── index.html              # Home / Hero
├── features.html           # Key features
├── demo.html               # Live system demo
├── assistant.html          # AI Research Assistant (interactive)
├── about.html              # About & disclaimer
├── styles.css              # Global styles
├── data.js                 # Mock data & config
├── cancer-research-assistant.py  # Python backend (local)
├── data/                   # Synthetic datasets
├── js/                     # Optional JS modules
├── .gitignore
├── .gitattributes
└── README.md
