"""Tests for data preprocessing pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import DataPreprocessor


class TestDataPreprocessor:

    def test_load_validates_schema(self, cfg, tmp_path, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        csv_path = tmp_path / "raw.csv"
        sample_raw_df.to_csv(csv_path, index=False)
        df = preprocessor.load(csv_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_raw_df)

    def test_load_raises_on_missing_columns(self, cfg, tmp_path):
        preprocessor = DataPreprocessor(cfg)
        df_bad = pd.DataFrame({"col1": [1, 2, 3]})
        csv_path = tmp_path / "bad.csv"
        df_bad.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocessor.load(csv_path)

    def test_clean_removes_duplicates(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_with_dups = pd.concat([sample_raw_df, sample_raw_df.iloc[:10]], ignore_index=True)
        df_clean = preprocessor.clean(df_with_dups)
        assert len(df_clean) == len(sample_raw_df)

    def test_clean_fills_missing_values(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_with_nan = sample_raw_df.copy()
        df_with_nan.loc[0, "GDP_Change_%"] = np.nan
        df_with_nan.loc[1, "Inflation_Rate_%"] = np.nan
        df_clean = preprocessor.clean(df_with_nan)
        assert df_clean["GDP_Change_%"].isna().sum() == 0
        assert df_clean["Inflation_Rate_%"].isna().sum() == 0

    def test_clean_encodes_war_profiteering(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        assert df_clean["War_Profiteering_Documented"].isin([0, 1]).all()

    def test_clean_encodes_black_market(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        assert df_clean["Black_Market_Activity_Level"].isin([0, 1, 2, 3]).all()

    def test_transform_adds_severity_label(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        df_proc = preprocessor.transform(df_clean)
        assert "Severity_Label" in df_proc.columns
        assert df_proc["Severity_Label"].isin([0, 1, 2, 3]).all()

    def test_transform_adds_duration(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        df_proc = preprocessor.transform(df_clean)
        assert "Conflict_Duration_Years" in df_proc.columns
        assert (df_proc["Conflict_Duration_Years"] >= 0).all()

    def test_transform_adds_esi(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        df_proc = preprocessor.transform(df_clean)
        assert "Economic_Stress_Index" in df_proc.columns
        assert df_proc["Economic_Stress_Index"].between(0, 1).all()

    def test_transform_drops_configured_columns(self, cfg, sample_raw_df):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        df_proc = preprocessor.transform(df_clean)
        for col in cfg["data"]["drop_features"]:
            assert col not in df_proc.columns, f"Column {col} should have been dropped"

    def test_save_and_reload(self, cfg, sample_raw_df, tmp_path):
        preprocessor = DataPreprocessor(cfg)
        df_clean = preprocessor.clean(sample_raw_df)
        df_proc = preprocessor.transform(df_clean)
        out_path = tmp_path / "test_processed.parquet"
        preprocessor.save(df_proc, out_path)
        df_reloaded = pd.read_parquet(out_path)
        assert df_reloaded.shape == df_proc.shape
        pd.testing.assert_frame_equal(df_proc.reset_index(drop=True),
                                      df_reloaded.reset_index(drop=True))
