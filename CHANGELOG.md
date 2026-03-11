# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-03-11

### Added
- End-to-end ML pipeline: data preprocessing → feature engineering → model training → evaluation
- `DataPreprocessor` class with schema validation, type coercion, and median/mode imputation
- `FeatureEngineer` class producing 20+ engineered features (log transforms, interaction terms, Economic Stress Index)
- Multi-model training: XGBoost, LightGBM, Random Forest, Ridge/Logistic Regression
- Optuna hyperparameter optimisation (50 trials, XGBoost) with MLflow experiment tracking
- Stratified 5-fold cross-validation for classification; holdout test set (20%) for both tasks
- SHAP TreeExplainer for global (bar + beeswarm) and local (force plot) model explanations
- Streamlit dashboard with prediction panel, gauge chart, dataset explorer, and model insights tab
- GitHub Actions CI pipeline: lint (ruff/black) → unit tests → smoke test
- `config/config.yaml` as single source of truth; no hardcoded values in source
- Dockerfile for containerised deployment
- Full test suite: schema validation, feature engineering correctness, model output shape/dtype

### Changed
- Replaced GBT with XGBoost/LightGBM as primary gradient-boosted models
- Migrated build config fully to `pyproject.toml` (removed legacy `setup.py`)

### Fixed
- Lambda bug in Optuna objective selection
- Import organisation across `src/` modules
