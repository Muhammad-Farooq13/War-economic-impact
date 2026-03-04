# 🌍 War Economic Impact Predictor

> **Predicting the economic damage of armed conflicts using pre-war economic
> indicators and machine learning — built on 100,000+ conflict-economic data
> points spanning WWII to 2026.**

[![CI](https://github.com/Muhammad-Farooq13/War-economic-impact/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Farooq13/War-economic-impact/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-blue.svg)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B.svg)](https://streamlit.io/)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-purple.svg)](https://shap.readthedocs.io/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-lightblue.svg)](https://optuna.org/)

---

## Problem Statement

Armed conflicts cause catastrophic and long-lasting damage to national economies.
Policy makers, NGOs, and international organisations often lack tools to
**quantitatively forecast** the likely economic impact of a conflict before or
during its progression.

This project builds an end-to-end machine learning system that:

1. **Regresses** the expected GDP change (%) from pre-war economic indicators  
2. **Classifies** economic severity into four tiers:
   `Mild → Moderate → Severe → Catastrophic`

---

## 📊 Dataset

| Attribute | Value |
|-----------|-------|
| Source | Synthetic conflict-economic dataset |
| Records | 100,001 rows |
| Features | 28 columns |
| Coverage | WWII (1939) → Israel-Hamas War (2026) |
| Target | `GDP_Change_%` (regression) + derived `Severity_Label` (classification) |

**Key feature groups:**  
Economic indicators · Unemployment metrics · Poverty & food security ·
Informal economy · Black market activity · Conflict metadata

---

## 🏗 Project Structure

```
war-economic-impact/
├── data/
│   ├── raw/                    ← Original CSV (committed)
│   ├── processed/              ← Parquet outputs (gitignored)
│   └── external/               ← Any supplementary data
│
├── notebooks/
│   ├── 01_eda.ipynb            ← Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb       ← Training, CV, Optuna tuning
│   └── 04_evaluation.ipynb     ← SHAP, residuals, confusion matrix
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py     ← Pipeline entry-point
│   │   └── preprocess.py       ← DataPreprocessor class
│   ├── features/
│   │   └── build_features.py   ← FeatureEngineer class
│   ├── models/
│   │   ├── train_model.py      ← ModelTrainer (MLflow + Optuna)
│   │   ├── predict_model.py    ← Batch inference
│   │   └── evaluate_model.py   ← SHAP + metrics reports
│   ├── visualization/
│   │   └── visualize.py        ← Reusable plot functions
│   └── utils.py                ← Config, logging helpers
│
├── models/                     ← Saved joblib artefacts (gitignored)
├── reports/figures/            ← Auto-generated plots (gitignored)
├── app/
│   └── app.py                  ← Streamlit web application
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── config/
│   └── config.yaml             ← Single source of truth for all settings
├── .github/workflows/ci.yml    ← GitHub Actions CI pipeline
├── Makefile                    ← Dev workflow shortcuts
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Muhammad-Farooq13/War-economic-impact.git
cd war-economic-impact

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Run the full ML pipeline

```bash
make pipeline
# Equivalent to:
#   make data        → clean & preprocess raw CSV → parquet
#   make features    → engineer advanced features
#   make train       → train + CV + Optuna tune + MLflow log
#   make evaluate    → SHAP, plots, text reports
```

### 3. Launch the Streamlit app

```bash
make app
# Opens: http://localhost:8501
```

### 4. View MLflow experiment tracking

```bash
make mlflow-ui
# Opens: http://localhost:5001
```

---

## 🔬 Methodology

### Feature Engineering

| Technique | Example |
|-----------|---------|
| Log transform | `log_Cost_of_War_USD` (right-skewed finance columns) |
| Ratio interaction | `During_War_Unemployment_% / Pre_War_Unemployment_%` |
| Multiplicative interaction | `Unemployment_Spike × Conflict_Duration_Years` |
| Composite score | `Economic_Stress_Index` (normalised [0,1]) |
| Black market composite | `BM_Level × log1p(Currency_Black_Market_Rate_Gap_%)` |
| Ordinal target derivation | `Severity_Label ∈ {0,1,2,3}` from GDP thresholds |

### Models

| Model | Task | Notes |
|-------|------|-------|
| XGBoost Regressor | Regression | Tuned via Optuna (50 trials) |
| LightGBM Regressor | Regression | Gradient boosted trees, fast train |
| Random Forest Regressor | Regression | Ensemble baseline |
| Ridge Regression | Regression | Linear baseline |
| XGBoost Classifier | Classification | Stratified 5-fold CV |
| LightGBM Classifier | Classification | Gradient boosted trees |
| Random Forest Classifier | Classification | Ensemble baseline |
| Logistic Regression | Classification | Multinomial baseline |

### Validation Strategy

- **Stratified 5-fold cross-validation** (classification)  
- **Holdout test set** (20%) — never touched during tuning  
- **Metrics:**
  - Regression: RMSE, MAE, MAPE, R²
  - Classification: F1-weighted, F1-macro, per-class precision/recall

### Explainability

- **SHAP TreeExplainer** — global feature importance (bar + beeswarm)  
- **Local SHAP force plots** — explain individual conflict predictions  
- **Residual analysis** — identify systematic errors by severity class  

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Test RMSE (GDP Change %) | **7.84** |
| Test MAE (GDP Change %) | **5.61** |
| Test R² | **0.89** |
| Test F1-weighted (severity) | **0.91** |
| Test F1-macro (severity) | **0.88** |
| Top-3 SHAP features | `Unemployment_Spike_Percentage_Points`, `Economic_Stress_Index`, `Inflation_Rate_%` |
| Best model (regression) | XGBoost (Optuna-tuned, 50 trials) |
| Best model (classification) | XGBoost (Optuna-tuned, 50 trials) |

---

## 🖥 Streamlit Application

The interactive dashboard provides:

- **Prediction panel** — adjust 18 conflict indicators via sliders and get instant GDP / severity predictions  
- **Gauge chart** — visualise predicted GDP change on a colour-coded scale  
- **Data explorer** — browse all 100k records with key EDA charts  
- **Model insights panel** — view SHAP plots, confusion matrices, and feature importance charts  

---

## 🧪 Running Tests

```bash
# All tests
make test

# With coverage report
make coverage
# → htmlcov/index.html
```

The test suite covers:

- Schema validation  
- Duplicate removal and missing-value imputation  
- Severity label derivation and distribution  
- Feature engineering correctness (log transforms, interactions, composites)  
- Model output shape, dtype, and serialization  

---

## ⚙️ Configuration

All project settings live in [`config/config.yaml`](config/config.yaml). No
hardcoded values exist in the source code. Key sections:

```yaml
data:
  target_regression: "GDP_Change_%"
  severity_thresholds:
    mild: -10     # GDP > -10% → Mild
    moderate: -25 # -25% < GDP ≤ -10% → Moderate
    severe: -50   # -50% < GDP ≤ -25% → Severe
                  # GDP ≤ -50% → Catastrophic

models:
  hyperparameter_tuning:
    enabled: true
    method: "optuna"
    n_trials: 50
```

---

## 🎯 Job Market Alignment

This project is intentionally designed to demonstrate the full DS skill stack:

| Skill | Where demonstrated |
|-------|-------------------|
| **Data engineering** | `DataPreprocessor` with schema validation, type coercion, imputation |
| **Feature engineering** | 20+ engineered features with documented rationale |
| **ML modelling** | Multi-model comparison, stratified CV, Optuna HPO |
| **Experiment tracking** | MLflow runs with params, metrics, and model artefacts |
| **Model explainability** | SHAP (global + local) |
| **Software engineering** | Modular src/, tests, type hints, docstrings |
| **CI/CD** | GitHub Actions pipeline (lint → test → smoke test) |
| **Deployment** | Streamlit app with polished UI |
| **Documentation** | Config-driven, reproducible, README-first |

---

## 📚 References

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*  
- Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*  
- Akiba, T. et al. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework*  
- World Bank. (2023). *Conflict and Development Resource Guide*

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

Pull requests are welcome. Please open an issue first to discuss major changes.  
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

*Built with Python 3.11 · scikit-learn · XGBoost · Optuna · MLflow · SHAP · Streamlit*
