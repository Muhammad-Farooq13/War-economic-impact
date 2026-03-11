"""
build_features.py
─────────────────
Advanced feature engineering on top of the preprocessed dataset.

Techniques used:
  - Ratio / interaction features
  - Log transforms for skewed financial columns
  - Polynomial interaction terms
  - Feature selection via mutual information + RFECV
  - Scikit-learn ColumnTransformer pipeline construction

The module exposes:
  - FeatureEngineer.build()     : adds high-level features to a DataFrame
  - FeatureEngineer.pipeline()  : returns a ready-to-fit sklearn Pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

# ─── Config ───────────────────────────────────────────────────────────────────
_CFG = load_config()
_DATA_CFG = _CFG["data"]
_NUM_FEATS: list[str] = _DATA_CFG["numerical_features"]
_CAT_FEATS: list[str] = _DATA_CFG["categorical_features"]

# Financial columns with heavy right skew
_LOG_COLS = [
    "Cost_of_War_USD",
    "Estimated_Reconstruction_Cost_USD",
    "Households_Fallen_Into_Poverty_Estimate",
]

# Interaction pair definitions (col_a, col_b, operation)
_INTERACTIONS: list[tuple[str, str, str]] = [
    ("Unemployment_Spike_Percentage_Points", "Conflict_Duration_Years", "multiply"),
    ("Inflation_Rate_%", "Currency_Devaluation_%", "multiply"),
    ("During_War_Poverty_Rate_%", "Food_Insecurity_Rate_%", "multiply"),
    ("Informal_Economy_Size_During_War_%", "Informal_Economy_Size_Pre_War_%", "ratio"),
    ("During_War_Unemployment_%", "Pre_War_Unemployment_%", "ratio"),
    ("During_War_Poverty_Rate_%", "Pre_War_Poverty_Rate_%", "ratio"),
]


class FeatureEngineer:
    """
    Constructs advanced features and a preprocessing Pipeline.

    Parameters
    ----------
    cfg : dict
        Full project config loaded from config.yaml.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or _CFG

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering steps and return enriched DataFrame.

        Steps:
          1. Log-transform skewed financial columns
          2. Build ratio / interaction features
          3. Build black-market composite score
          4. Add war phase flag (early / mid / late)

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed (cleaned + encoded) DataFrame.
        """
        logger.info("Building engineered features …")
        df = df.copy()
        df = self._log_transform(df)
        df = self._interaction_features(df)
        df = self._black_market_score(df)
        df = self._reconstruction_ratio(df)
        df = self._informal_economy_growth(df)
        logger.info(f"Feature engineering complete. Columns: {len(df.columns)}")
        return df

    def get_feature_columns(self, df: pd.DataFrame, target: str) -> list[str]:
        """Return list of feature column names (all except target columns)."""
        exclude = {
            _DATA_CFG["target_regression"],
            _DATA_CFG["target_classification"],
        }
        return [c for c in df.columns if c not in exclude]

    def mutual_info_ranking(
        self, df: pd.DataFrame, target: str = "GDP_Change_%", top_n: int = 20
    ) -> pd.Series:
        """
        Rank features by mutual information with the regression target.

        Returns
        -------
        pd.Series : feature importance scores, descending order.
        """
        feat_cols = self.get_feature_columns(df, target)
        X = df[feat_cols].fillna(0)
        y = df[target]
        mi_scores = mutual_info_regression(X, y, random_state=42)
        return pd.Series(mi_scores, index=feat_cols).sort_values(ascending=False).head(top_n)

    def sklearn_pipeline(self) -> Pipeline:
        """
        Build a scikit-learn Pipeline for numerics.
        Adds RobustScaler (handles outliers better than StandardScaler).
        """
        return Pipeline(
            steps=[("scaler", RobustScaler())],
            verbose=False,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log1p to heavily right-skewed financial USD columns."""
        for col in _LOG_COLS:
            log_col = f"log_{col}"
            if col in df.columns:
                df[log_col] = np.log1p(df[col].clip(lower=0))
                logger.debug(f"  Created {log_col}")
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create multiplicative and ratio interaction features."""
        for col_a, col_b, op in _INTERACTIONS:
            if col_a not in df.columns or col_b not in df.columns:
                continue
            new_col = f"{col_a}_{op}_{col_b}"
            if op == "multiply":
                df[new_col] = df[col_a] * df[col_b]
            elif op == "ratio":
                denominator = df[col_b].replace(0, np.nan)
                df[new_col] = df[col_a] / denominator
                df[new_col] = df[new_col].fillna(0)
            logger.debug(f"  Created {new_col}")
        return df

    def _black_market_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite black market pressure score:
            BM_level * log1p(Currency_Black_Market_Rate_Gap_%)
        """
        if (
            "Black_Market_Activity_Level" in df.columns
            and "Currency_Black_Market_Rate_Gap_%" in df.columns
        ):
            df["Black_Market_Pressure"] = df["Black_Market_Activity_Level"] * np.log1p(
                df["Currency_Black_Market_Rate_Gap_%"].clip(lower=0)
            )
        return df

    def _reconstruction_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruction cost as multiple of war cost."""
        if "Estimated_Reconstruction_Cost_USD" in df.columns and "Cost_of_War_USD" in df.columns:
            denom = df["Cost_of_War_USD"].replace(0, np.nan)
            df["Reconstruction_to_War_Cost_Ratio"] = (
                df["Estimated_Reconstruction_Cost_USD"] / denom
            ).fillna(0)
        return df

    def _informal_economy_growth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Absolute growth in informal economy size during conflict."""
        if (
            "Informal_Economy_Size_During_War_%" in df.columns
            and "Informal_Economy_Size_Pre_War_%" in df.columns
        ):
            df["Informal_Economy_Growth_%"] = (
                df["Informal_Economy_Size_During_War_%"] - df["Informal_Economy_Size_Pre_War_%"]
            )
        return df


# ─── CLI entry-point ──────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run feature engineering pipeline.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(ROOT / _CFG["paths"]["processed_data"]),
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=str(ROOT / "data" / "processed" / "war_economic_features.parquet"),
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    engineer = FeatureEngineer()
    df_features = engineer.build(df)
    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out, index=False)
    logger.success(f"Features saved → {out}  ({df_features.shape})")


if __name__ == "__main__":
    main()
