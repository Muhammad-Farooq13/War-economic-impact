"""
predict_model.py
────────────────
Load a saved model artefact and generate predictions.

Usage:
    python -m src.models.predict_model --task regression --model xgb \
        --input data/processed/war_economic_features.parquet \
        --output reports/predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

_CFG = load_config()
_DATA_CFG = _CFG["data"]


class Predictor:
    """
    Loads a trained model + scaler and generates predictions.

    Parameters
    ----------
    task : "regression" | "classification"
    model_key : "xgb" | "rf" | "ridge" | "logreg"
    """

    def __init__(self, task: str = "regression", model_key: str = "xgb") -> None:
        self.task = task
        self.model_key = model_key
        model_dir = ROOT / _CFG["paths"]["model_dir"]

        model_path = model_dir / f"{task}_{model_key}.joblib"
        scaler_path = model_dir / f"scaler_{model_key}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artefact not found: {model_path}\n"
                "Run `make train` or `python -m src.models.train_model` first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        logger.info(f"Loaded model from {model_path}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate predictions for the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Feature-engineered DataFrame (targets excluded).

        Returns
        -------
        pd.Series : predictions indexed like df.
        """
        exclude = {
            _DATA_CFG["target_regression"],
            _DATA_CFG["target_classification"],
        }
        feat_cols = [c for c in df.columns if c not in exclude]
        X = df[feat_cols].fillna(0).values

        if self.scaler is not None:
            X = self.scaler.transform(X)

        preds = self.model.predict(X)
        target = (
            _DATA_CFG["target_regression"]
            if self.task == "regression"
            else _DATA_CFG["target_classification"]
        )
        label = f"predicted_{target}"
        return pd.Series(preds, index=df.index, name=label)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate model predictions.")
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--model", default="xgb")
    p.add_argument(
        "--input",
        default=str(ROOT / "data" / "processed" / "war_economic_features.parquet"),
    )
    p.add_argument(
        "--output",
        default=str(ROOT / "reports" / "predictions.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    predictor = Predictor(task=args.task, model_key=args.model)
    preds = predictor.predict(df)

    result = df.copy()
    result[preds.name] = preds
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    logger.success(f"Predictions saved → {out}  ({len(preds):,} rows)")


if __name__ == "__main__":
    main()
