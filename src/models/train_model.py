"""
train_model.py
──────────────
Full model training pipeline with:
  - Stratified k-fold cross-validation
  - Multiple model types (XGBoost, LightGBM, Ridge)
  - Optuna hyperparameter optimisation
  - MLflow experiment tracking
  - Artefact persistence (joblib)

Usage:
    python -m src.models.train_model
    python -m src.models.train_model --task regression --model xgb
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb_lib
from loguru import logger
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

_CFG = load_config()
_DATA_CFG = _CFG["data"]
_MODEL_CFG = _CFG["models"]
_MLFLOW_CFG = _CFG["mlflow"]


# ─── Model Registry ───────────────────────────────────────────────────────────

def _build_regression_models() -> dict[str, Any]:
    reg_cfg = _MODEL_CFG["regression"]
    xgb_p = reg_cfg["xgb"]
    lgb_p = reg_cfg["lightgbm"]
    rf_p = reg_cfg["random_forest"]
    return {
        "xgb": xgb_lib.XGBRegressor(
            n_estimators=xgb_p["n_estimators"],
            learning_rate=xgb_p["learning_rate"],
            max_depth=xgb_p["max_depth"],
            subsample=xgb_p.get("subsample", 0.8),
            colsample_bytree=xgb_p.get("colsample_bytree", 0.8),
            random_state=xgb_p["random_state"],
            verbosity=0,
            n_jobs=-1,
        ),
        "lgbm": lgb.LGBMRegressor(
            n_estimators=lgb_p["n_estimators"],
            learning_rate=lgb_p["learning_rate"],
            max_depth=lgb_p["max_depth"],
            num_leaves=lgb_p.get("num_leaves", 63),
            random_state=lgb_p["random_state"],
            n_jobs=-1,
            verbose=-1,
        ),
        "rf": RandomForestRegressor(
            n_estimators=rf_p["n_estimators"],
            max_depth=rf_p["max_depth"],
            random_state=rf_p["random_state"],
            n_jobs=-1,
        ),
        "ridge": Ridge(alpha=reg_cfg["ridge"]["alpha"]),
    }


def _build_classification_models() -> dict[str, Any]:
    cls_cfg = _MODEL_CFG["classification"]
    xgb_p = cls_cfg["xgb"]
    lgb_p = cls_cfg["lightgbm"]
    rf_p = cls_cfg["random_forest"]
    return {
        "xgb": xgb_lib.XGBClassifier(
            n_estimators=xgb_p["n_estimators"],
            learning_rate=xgb_p["learning_rate"],
            max_depth=xgb_p["max_depth"],
            subsample=xgb_p.get("subsample", 0.8),
            colsample_bytree=xgb_p.get("colsample_bytree", 0.8),
            random_state=xgb_p["random_state"],
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        ),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=lgb_p["n_estimators"],
            learning_rate=lgb_p["learning_rate"],
            max_depth=lgb_p["max_depth"],
            num_leaves=lgb_p.get("num_leaves", 63),
            random_state=lgb_p["random_state"],
            n_jobs=-1,
            verbose=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=rf_p["n_estimators"],
            max_depth=rf_p["max_depth"],
            random_state=rf_p["random_state"],
            n_jobs=-1,
        ),
        "logreg": LogisticRegression(**cls_cfg["logistic_regression"]),
    }


# ─── Optuna objectives ────────────────────────────────────────────────────────

def _regression_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": _DATA_CFG["random_state"],
        "verbosity": 0,
        "n_jobs": -1,
    }
    model = xgb_lib.XGBRegressor(**params)
    cv_scores = cross_val_score(
        model, X, y,
        cv=_MODEL_CFG["cv_folds"],
        scoring=_MODEL_CFG["scoring_regression"],
        n_jobs=-1,
    )
    return float(np.mean(cv_scores))


def _classification_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": _DATA_CFG["random_state"],
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,
    }
    model = xgb_lib.XGBClassifier(**params)
    cv_scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits=_MODEL_CFG["cv_folds"], shuffle=True,
                           random_state=_DATA_CFG["random_state"]),
        scoring=_MODEL_CFG["scoring_classification"],
        n_jobs=-1,
    )
    return float(np.mean(cv_scores))


# ─── Main Trainer ─────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Orchestrates train / CV / hyperparameter-tune / MLflow logging / save.

    Parameters
    ----------
    task : "regression" | "classification"
    model_key : "xgb" | "rf" | "ridge" | "logreg"  (or None → train all)
    """

    def __init__(self, task: str = "regression", model_key: str | None = None) -> None:
        self.task = task
        self.model_key = model_key
        self.cfg = _CFG
        self.model_dir = ROOT / self.cfg["paths"]["model_dir"]
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        df = self._load_data()
        X, y = self._split_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=_DATA_CFG["test_size"],
            random_state=_DATA_CFG["random_state"],
            stratify=y if self.task == "classification" else None,
        )

        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        mlflow.set_tracking_uri(_MLFLOW_CFG["tracking_uri"])
        mlflow.set_experiment(_MLFLOW_CFG["experiment_name"])

        models = (
            _build_regression_models()
            if self.task == "regression"
            else _build_classification_models()
        )
        if self.model_key:
            models = {self.model_key: models[self.model_key]}

        results: dict[str, dict] = {}

        for name, model in models.items():
            logger.info(f"\n{'='*60}\nTraining [{self.task}] → {name}\n{'='*60}")
            t0 = time.time()

            # ── Optional Optuna tuning ─────────────────────────────────────────
            if (
                _MODEL_CFG["hyperparameter_tuning"]["enabled"]
                and name.startswith("xgb")
            ):
                model = self._tune(name, X_train_sc, y_train)

            with mlflow.start_run(run_name=f"{self.task}_{name}"):
                # ── Cross-validation ──────────────────────────────────────────
                cv_scorer = (
                    _MODEL_CFG["scoring_regression"]
                    if self.task == "regression"
                    else _MODEL_CFG["scoring_classification"]
                )
                cv_kfold = (
                    StratifiedKFold(
                        n_splits=_MODEL_CFG["cv_folds"],
                        shuffle=True,
                        random_state=_DATA_CFG["random_state"],
                    )
                    if self.task == "classification"
                    else _MODEL_CFG["cv_folds"]
                )
                cv_scores = cross_val_score(
                    model, X_train_sc, y_train,
                    cv=cv_kfold,
                    scoring=cv_scorer,
                    n_jobs=-1,
                )
                logger.info(
                    f"CV {cv_scorer}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
                )

                # ── Final fit ─────────────────────────────────────────────────
                model.fit(X_train_sc, y_train)
                y_pred = model.predict(X_test_sc)
                elapsed = time.time() - t0

                metrics = self._compute_metrics(y_test, y_pred)
                metrics["cv_mean"] = float(np.mean(cv_scores))
                metrics["cv_std"] = float(np.std(cv_scores))
                metrics["train_time_s"] = round(elapsed, 2)

                self._log_mlflow(model, metrics, X_train_sc, name)
                self._save_artefacts(model, scaler, name)

                results[name] = metrics
                logger.success(f"  Done in {elapsed:.1f}s | Metrics: {metrics}")

        self._print_leaderboard(results)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        feat_path = (
            ROOT / "data" / "processed" / "war_economic_features.parquet"
        )
        if not feat_path.exists():
            logger.warning(
                "Feature file not found. Falling back to processed parquet."
            )
            feat_path = ROOT / _CFG["paths"]["processed_data"]
        if not feat_path.exists():
            raise FileNotFoundError(
                f"No processed data at {feat_path}. Run make data features first."
            )
        df = pd.read_parquet(feat_path)
        logger.info(f"Loaded data: {df.shape}")
        return df

    def _split_xy(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        target_col = (
            _DATA_CFG["target_regression"]
            if self.task == "regression"
            else _DATA_CFG["target_classification"]
        )
        exclude = {
            _DATA_CFG["target_regression"],
            _DATA_CFG["target_classification"],
        }
        feat_cols = [c for c in df.columns if c not in exclude]
        X = df[feat_cols].fillna(0)
        y = df[target_col]
        logger.info(f"Target: {target_col} | Features: {len(feat_cols)}")
        return X, y

    def _tune(self, name: str, X: np.ndarray, y: np.ndarray) -> Any:
        logger.info("Running Optuna hyperparameter search …")
        tune_cfg = _MODEL_CFG["hyperparameter_tuning"]
        # NOTE: must assign objective via if/else to avoid Python lambda-parse bug
        # where `lambda t: a if cond else lambda t: b` embeds cond inside the
        # first lambda body instead of selecting between two lambdas.
        if self.task == "regression":
            objective = lambda t: _regression_objective(t, X, y)  # noqa: E731
        else:
            objective = lambda t: _classification_objective(t, X, y)  # noqa: E731
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{self.task}_{name}",
        )
        study.optimize(
            objective,
            n_trials=tune_cfg["n_trials"],
            timeout=tune_cfg["timeout_seconds"],
            show_progress_bar=False,
        )
        best = study.best_params
        best["random_state"] = _DATA_CFG["random_state"]
        best["verbosity"] = 0
        best["n_jobs"] = -1
        logger.info(f"Best params: {best}")
        if self.task == "regression":
            best.pop("use_label_encoder", None)
            best.pop("eval_metric", None)
            return xgb_lib.XGBRegressor(**best)
        else:
            best["eval_metric"] = "mlogloss"
            return xgb_lib.XGBClassifier(**best)

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        if self.task == "regression":
            return {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }
        return {
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }

    def _log_mlflow(
        self, model: Any, metrics: dict, X_train: np.ndarray, name: str
    ) -> None:
        mlflow.log_params({"model_type": name, "task": self.task})
        mlflow.log_metrics(metrics)
        if _MLFLOW_CFG["log_artifacts"]:
            mlflow.sklearn.log_model(model, artifact_path=f"model_{name}")

    def _save_artefacts(self, model: Any, scaler: Any, name: str) -> None:
        model_path = self.model_dir / f"{self.task}_{name}.joblib"
        scaler_path = self.model_dir / f"scaler_{name}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved model → {model_path}")

    def _print_leaderboard(self, results: dict) -> None:
        logger.info("\n" + "=" * 60)
        logger.info("MODEL LEADERBOARD")
        logger.info("=" * 60)
        for model_name, metrics in results.items():
            logger.info(f"  {model_name:12s} | {metrics}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train war economic impact models.")
    p.add_argument(
        "--task",
        choices=["regression", "classification", "both"],
        default="both",
    )
    p.add_argument(
        "--model",
        choices=["xgb", "lgbm", "rf", "ridge", "logreg"],
        default=None,
        help="Train a specific model only (default: all models).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = (
        ["regression", "classification"] if args.task == "both" else [args.task]
    )
    for task in tasks:
        trainer = ModelTrainer(task=task, model_key=args.model)
        trainer.run()


if __name__ == "__main__":
    main()
