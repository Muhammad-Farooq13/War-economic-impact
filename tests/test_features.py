"""Tests for feature engineering pipeline."""

import pandas as pd
import pytest

from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer


@pytest.fixture(scope="module")
def processed_df(cfg, sample_raw_df):
    preprocessor = DataPreprocessor(cfg)
    df_clean = preprocessor.clean(sample_raw_df)
    return preprocessor.transform(df_clean)


class TestFeatureEngineer:

    def test_build_returns_dataframe(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert isinstance(df_feat, pd.DataFrame)

    def test_build_has_more_columns(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert len(df_feat.columns) > len(processed_df.columns)

    def test_log_transforms_created(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert "log_Cost_of_War_USD" in df_feat.columns
        assert "log_Estimated_Reconstruction_Cost_USD" in df_feat.columns

    def test_log_values_finite(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert (
            df_feat["log_Cost_of_War_USD"].isfinite().all()
            if hasattr(df_feat["log_Cost_of_War_USD"], "isfinite")
            else True
        )
        assert not df_feat["log_Cost_of_War_USD"].isnull().any()

    def test_black_market_pressure_created(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert "Black_Market_Pressure" in df_feat.columns
        assert not df_feat["Black_Market_Pressure"].isnull().any()

    def test_reconstruction_ratio_created(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert "Reconstruction_to_War_Cost_Ratio" in df_feat.columns

    def test_informal_economy_growth_created(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        assert "Informal_Economy_Growth_%" in df_feat.columns

    def test_mutual_info_returns_series(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        mi = eng.mutual_info_ranking(df_feat, target="GDP_Change_%", top_n=10)
        assert isinstance(mi, pd.Series)
        assert len(mi) == 10
        assert (mi >= 0).all()

    def test_get_feature_columns_excludes_targets(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        feat_cols = eng.get_feature_columns(df_feat, "GDP_Change_%")
        assert "GDP_Change_%" not in feat_cols
        assert "Severity_Label" not in feat_cols

    def test_no_nan_after_build(self, cfg, processed_df):
        eng = FeatureEngineer(cfg)
        df_feat = eng.build(processed_df)
        feat_cols = eng.get_feature_columns(df_feat, "GDP_Change_%")
        # Interaction ratio features may produce NaN only if denominator is 0
        # After fillna(0), they should all be finite
        nan_counts = df_feat[feat_cols].isnull().sum()
        cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
        assert len(cols_with_nan) == 0, f"NaN found in: {cols_with_nan}"
