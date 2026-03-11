"""
evaluate_model.py
─────────────────
Comprehensive model evaluation including:
  - Hold-out test-set metrics
  - SHAP feature importance (global + local)
  - Residual analysis for regression
  - Confusion matrix and classification report
  - Saved results to reports/

Usage:
    python -m src.models.evaluate_model --task regression --model xgb
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

_CFG = load_config()
_DATA_CFG = _CFG["data"]
_FIG_DIR = ROOT / _CFG["paths"]["figures_dir"]
_FIG_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:
    """
    Evaluates trained models using test-set metrics and SHAP explainability.

    Parameters
    ----------
    task : "regression" | "classification"
    model_key : "xgb" | "rf" | etc.
    """

    def __init__(self, task: str = "regression", model_key: str = "xgb") -> None:
        self.task = task
        self.model_key = model_key
        model_dir = ROOT / _CFG["paths"]["model_dir"]

        self.model = joblib.load(model_dir / f"{task}_{model_key}.joblib")
        scaler_path = model_dir / f"scaler_{model_key}.joblib"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        logger.info(f"Loaded {task}/{model_key} for evaluation.")

    def run(self) -> dict:
        df = self._load_data()
        X, y = self._split_xy(df)
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=_DATA_CFG["test_size"],
            random_state=_DATA_CFG["random_state"],
            stratify=y if self.task == "classification" else None,
        )
        X_test_sc = self.scaler.transform(X_test) if self.scaler else X_test

        y_pred = self.model.predict(X_test_sc)
        metrics = self._compute_metrics(y_test, y_pred)

        logger.info(f"\nTest-set metrics:\n{pd.Series(metrics).to_string()}")

        self._plot_results(y_test, y_pred, X_test, X.columns.tolist())

        report_path = ROOT / "reports" / f"evaluation_{self.task}_{self.model_key}.txt"
        self._save_text_report(metrics, y_test, y_pred, report_path)

        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        feat_path = ROOT / "data" / "processed" / "war_economic_features.parquet"
        if not feat_path.exists():
            feat_path = ROOT / _CFG["paths"]["processed_data"]
        return pd.read_parquet(feat_path)

    def _split_xy(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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
        return df[feat_cols].fillna(0), df[target_col]

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> dict:
        if self.task == "regression":
            return {
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
                "R²": float(r2_score(y_true, y_pred)),
            }
        from sklearn.metrics import accuracy_score, f1_score

        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),  # type: ignore
            "F1_Weighted": float(f1_score(y_true, y_pred, average="weighted")),
            "F1_Macro": float(f1_score(y_true, y_pred, average="macro")),
        }

    def _plot_results(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X_test: pd.DataFrame,
        feature_names: list[str],
    ) -> None:
        if self.task == "regression":
            self._plot_regression(y_true, y_pred)
        else:
            self._plot_confusion(y_true, y_pred)

        self._plot_shap(X_test, feature_names)

    def _plot_regression(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Actual vs Predicted
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.3, s=15, color="steelblue")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual GDP Change (%)")
        ax.set_ylabel("Predicted GDP Change (%)")
        ax.set_title("Actual vs Predicted")
        ax.legend()

        # Residuals
        residuals = y_true - y_pred
        ax2 = axes[1]
        ax2.scatter(y_pred, residuals, alpha=0.3, s=15, color="coral")
        ax2.axhline(0, color="black", linewidth=1)
        ax2.set_xlabel("Predicted GDP Change (%)")
        ax2.set_ylabel("Residuals")
        ax2.set_title("Residual Plot")

        fig.suptitle(f"Regression Evaluation – {self.model_key.upper()}", fontsize=14)
        fig.tight_layout()
        path = _FIG_DIR / f"regression_{self.model_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved regression plot → {path}")

    def _plot_confusion(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        labels = ["Mild", "Moderate", "Severe", "Catastrophic"]
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix – {self.model_key.upper()}")
        fig.tight_layout()
        path = _FIG_DIR / f"confusion_{self.model_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved confusion matrix → {path}")

    def _plot_shap(self, X_test: pd.DataFrame, feature_names: list[str]) -> None:
        try:
            X_sc = self.scaler.transform(X_test) if self.scaler else X_test.values
            X_df = pd.DataFrame(X_sc, columns=feature_names)
            sample = X_df.iloc[:500]  # limit sample size for speed

            if hasattr(self.model, "feature_importances_"):
                # XGBoost, LightGBM, RandomForest — native TreeExplainer
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(sample)
            else:
                # Ridge, LogReg
                explainer = shap.LinearExplainer(
                    self.model, X_df, feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(sample)

            # For multi-class classifiers LightGBM/RF returns a list of arrays;
            # use the mean-absolute across classes for the bar summary.
            import numpy as _np

            if isinstance(shap_values, list):
                shap_plot_values = _np.abs(_np.stack(shap_values, axis=0)).mean(axis=0)
            elif shap_values.ndim == 3:  # XGBoost multi-class (n, feat, classes)
                shap_plot_values = _np.abs(shap_values).mean(axis=2)
            else:
                shap_plot_values = shap_values

            shap.summary_plot(
                shap_plot_values,
                sample,
                plot_type="bar",
                show=False,
                max_display=15,
            )
            path = _FIG_DIR / f"shap_{self.task}_{self.model_key}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved SHAP plot → {path}")

            # Also save beeswarm for regression (single-output)
            if self.task == "regression" and not isinstance(shap_values, list):
                shap.summary_plot(shap_plot_values, sample, show=False, max_display=15)
                beeswarm_path = _FIG_DIR / f"shap_beeswarm_{self.model_key}.png"
                plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved SHAP beeswarm → {beeswarm_path}")
        except Exception as exc:
            logger.warning(f"SHAP plot skipped: {exc}")

    def _save_text_report(
        self,
        metrics: dict,
        y_true: pd.Series,
        y_pred: np.ndarray,
        path: Path,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"=== Evaluation Report: {self.task} / {self.model_key} ===\n\n")
            for k, v in metrics.items():
                f.write(f"  {k}: {v:.4f}\n")
            if self.task == "classification":
                labels = ["Mild", "Moderate", "Severe", "Catastrophic"]
                report = classification_report(y_true, y_pred, target_names=labels)
                f.write(f"\nClassification Report:\n{report}\n")
        logger.info(f"Saved evaluation report → {path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

import argparse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--model", default="xgb")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = ModelEvaluator(task=args.task, model_key=args.model)
    evaluator.run()


if __name__ == "__main__":
    main()
