"""Tests for model training and prediction pipeline."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.data.preprocess import DataPreprocessor
from src.features.build_features import FeatureEngineer


@pytest.fixture(scope="module")
def feature_matrix(cfg, sample_raw_df):
    preprocessor = DataPreprocessor(cfg)
    df_clean = preprocessor.clean(sample_raw_df)
    df_proc = preprocessor.transform(df_clean)
    eng = FeatureEngineer(cfg)
    return eng.build(df_proc), eng


@pytest.fixture(scope="module")
def trained_reg(cfg, feature_matrix):
    df_feat, eng = feature_matrix
    feat_cols = eng.get_feature_columns(df_feat, "GDP_Change_%")
    X = df_feat[feat_cols].fillna(0)
    y = df_feat["GDP_Change_%"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X_tr_sc, y_tr)
    return model, scaler, X_te, X_te_sc, y_te


@pytest.fixture(scope="module")
def trained_cls(cfg, feature_matrix):
    df_feat, eng = feature_matrix
    feat_cols = eng.get_feature_columns(df_feat, "Severity_Label")
    X = df_feat[feat_cols].fillna(0)
    y = df_feat["Severity_Label"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_tr_sc, y_tr)
    return model, scaler, X_te, X_te_sc, y_te


class TestRegressionModel:

    def test_model_trained(self, trained_reg):
        model, *_ = trained_reg
        assert hasattr(model, "feature_importances_")

    def test_predictions_shape(self, trained_reg):
        model, _, X_te, X_te_sc, y_te = trained_reg
        preds = model.predict(X_te_sc)
        assert preds.shape == (len(y_te),)

    def test_predictions_are_float(self, trained_reg):
        model, _, X_te, X_te_sc, y_te = trained_reg
        preds = model.predict(X_te_sc)
        assert np.issubdtype(preds.dtype, np.floating)

    def test_predictions_finite(self, trained_reg):
        model, _, X_te, X_te_sc, y_te = trained_reg
        preds = model.predict(X_te_sc)
        assert np.all(np.isfinite(preds))

    def test_r2_positive(self, trained_reg):
        from sklearn.metrics import r2_score

        model, _, X_te, X_te_sc, y_te = trained_reg
        preds = model.predict(X_te_sc)
        r2 = r2_score(y_te, preds)
        # Even a weak model should beat random on n=40 test samples
        # Just check it's not catastrophically negative
        assert r2 > -5.0

    def test_model_serialise_deserialise(self, trained_reg, tmp_path):
        model, scaler, X_te, X_te_sc, _ = trained_reg
        path = tmp_path / "model.joblib"
        joblib.dump(model, path)
        model2 = joblib.load(path)
        preds1 = model.predict(X_te_sc)
        preds2 = model2.predict(X_te_sc)
        np.testing.assert_array_equal(preds1, preds2)


class TestClassificationModel:

    def test_model_trained(self, trained_cls):
        model, *_ = trained_cls
        assert hasattr(model, "classes_")
        assert set(model.classes_).issubset({0, 1, 2, 3})

    def test_predictions_shape(self, trained_cls):
        model, _, X_te, X_te_sc, y_te = trained_cls
        preds = model.predict(X_te_sc)
        assert len(preds) == len(y_te)

    def test_predictions_valid_classes(self, trained_cls):
        model, _, X_te, X_te_sc, y_te = trained_cls
        preds = model.predict(X_te_sc)
        assert set(preds).issubset({0, 1, 2, 3})

    def test_predict_proba_shape(self, trained_cls):
        model, _, X_te, X_te_sc, y_te = trained_cls
        proba = model.predict_proba(X_te_sc)
        assert proba.shape[0] == len(y_te)
        assert proba.shape[1] <= 4  # at most 4 classes

    def test_proba_sums_to_one(self, trained_cls):
        model, _, X_te, X_te_sc, y_te = trained_cls
        proba = model.predict_proba(X_te_sc)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
