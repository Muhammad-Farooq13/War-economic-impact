"""
preprocess.py
─────────────
All data cleaning and base transformation logic.

Responsibilities:
  - Load raw CSV into a DataFrame
  - Validate schema and column types
  - Handle duplicates and missing values
  - Encode / clean categorical fields
  - Derive base engineered columns (duration, severity label)
  - Save cleaned data as parquet for downstream use
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    """Full preprocessing pipeline for the war economic impact dataset."""

    # Required columns – will raise if missing
    REQUIRED_COLS: list[str] = [
        "Conflict_Name",
        "Conflict_Type",
        "Region",
        "Start_Year",
        "End_Year",
        "Status",
        "Primary_Country",
        "GDP_Change_%",
        "Inflation_Rate_%",
        "Pre_War_Unemployment_%",
    ]

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.data_cfg = cfg["data"]
        self._label_encoders: dict[str, LabelEncoder] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> pd.DataFrame:
        """Load raw CSV and perform schema validation."""
        path = Path(path)
        logger.info(f"Loading raw data from {path}")
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Shape: {df.shape}")

        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, fix types, handle missing values."""
        logger.info("Cleaning data …")
        df = df.copy()

        # ── Duplicates ────────────────────────────────────────────────────────
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Dropped {before - len(df):,} duplicate rows.")

        # ── Type coercion ─────────────────────────────────────────────────────
        df["Start_Year"] = pd.to_numeric(df["Start_Year"], errors="coerce")
        df["End_Year"] = pd.to_numeric(df["End_Year"], errors="coerce")
        df["War_Profiteering_Documented"] = (
            df["War_Profiteering_Documented"]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "yes": 1, "No": 0, "no": 0, "1": 1, "0": 0})
            .fillna(0)
            .astype(int)
        )

        # ── Boolean Black market ──────────────────────────────────────────────
        bm_map = {"Dominant": 3, "High": 2, "Moderate": 1, "Low": 0}
        df["Black_Market_Activity_Level"] = (
            df["Black_Market_Activity_Level"]
            .str.strip()
            .map(bm_map)
            .fillna(0)
            .astype(int)
        )

        # ── Missing value imputation ───────────────────────────────────────────
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in num_cols:
            n_missing = df[col].isna().sum()
            if n_missing:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.debug(f"  {col}: filled {n_missing} NaN with median={median_val:.4f}")

        for col in cat_cols:
            n_missing = df[col].isna().sum()
            if n_missing:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.debug(f"  {col}: filled {n_missing} NaN with mode='{mode_val}'")

        logger.info(f"Post-clean shape: {df.shape}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer base features and encode categoricals."""
        logger.info("Transforming data …")
        df = df.copy()

        # ── Engineered columns ────────────────────────────────────────────────
        df = self._add_duration(df)
        df = self._add_severity_label(df)
        df = self._add_economic_stress_index(df)
        df = self._encode_categoricals(df)

        # ── Drop low-value identity columns ───────────────────────────────────
        drop_cols = [
            c for c in self.data_cfg.get("drop_features", []) if c in df.columns
        ]
        df = df.drop(columns=drop_cols)
        logger.info(f"Dropped columns: {drop_cols}")

        logger.info(f"Final shape: {df.shape}")
        return df

    def save(self, df: pd.DataFrame, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Saved processed data → {path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _add_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute conflict duration in years (End - Start, capped at 0)."""
        df["Conflict_Duration_Years"] = (
            df["End_Year"] - df["Start_Year"]
        ).clip(lower=0)
        return df

    def _add_severity_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive ordinal severity label from GDP_Change_%.

        Label mapping (configurable via config.yaml):
            0 – Mild        (GDP > -10 %)
            1 – Moderate    (-25 % < GDP ≤ -10 %)
            2 – Severe      (-50 % < GDP ≤ -25 %)
            3 – Catastrophic (GDP ≤ -50 %)
        """
        thresholds = self.data_cfg["severity_thresholds"]
        mild_thresh = thresholds["mild"]
        mod_thresh = thresholds["moderate"]
        sev_thresh = thresholds["severe"]

        conditions = [
            df["GDP_Change_%"] > mild_thresh,
            (df["GDP_Change_%"] <= mild_thresh) & (df["GDP_Change_%"] > mod_thresh),
            (df["GDP_Change_%"] <= mod_thresh) & (df["GDP_Change_%"] > sev_thresh),
            df["GDP_Change_%"] <= sev_thresh,
        ]
        choices = [0, 1, 2, 3]
        df["Severity_Label"] = np.select(conditions, choices, default=0).astype(int)

        dist = df["Severity_Label"].value_counts().sort_index()
        logger.info(f"Severity label distribution:\n{dist.to_string()}")
        return df

    def _add_economic_stress_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Composite Economic Stress Index (ESI) normalised to [0, 1].
        Aggregates unemployment, poverty, inflation, and food insecurity.
        """
        stress = (
            df["Unemployment_Spike_Percentage_Points"].fillna(0)
            + df["During_War_Poverty_Rate_%"].fillna(0)
            + df["Inflation_Rate_%"].fillna(0) / 100
            + df["Food_Insecurity_Rate_%"].fillna(0)
        )
        esi_min, esi_max = stress.min(), stress.max()
        df["Economic_Stress_Index"] = (stress - esi_min) / (esi_max - esi_min + 1e-9)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ordinal-encode categorical columns that remain after drop_features.
        Label encoders are stored for inverse-transform / app use.
        """
        cat_feats = [
            c for c in self.data_cfg["categorical_features"] if c in df.columns
        ]
        # Black_Market_Activity_Level already mapped to int in clean()
        cat_feats = [
            c
            for c in cat_feats
            if c != "Black_Market_Activity_Level"
            and df[c].dtype == object
        ]

        for col in cat_feats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self._label_encoders[col] = le
            logger.debug(f"  Label-encoded '{col}' → {len(le.classes_)} classes")

        return df

    @property
    def label_encoders(self) -> dict[str, LabelEncoder]:
        return self._label_encoders
