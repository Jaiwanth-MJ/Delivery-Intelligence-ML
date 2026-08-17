"""
End-to-end smoke tests for train.py. These run on the real dataset but with
reduced estimators for speed, verifying the pipeline trains, evaluates, and
persists artifacts without error.
"""

import json

from src import config
from src.train import train_classification, train_regression


def test_train_classification_produces_metrics(tmp_path, monkeypatch):
    # Redirect model/report outputs to a temp dir so we don't overwrite real artifacts
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(config, "CLASSIFIER_MODEL_PATH", tmp_path / "models" / "clf.joblib")
    monkeypatch.setattr(config, "CLASSIFIER_METRICS_PATH", tmp_path / "reports" / "clf_metrics.json")
    monkeypatch.setitem(config.RF_CLASSIFIER_PARAMS, "n_estimators", 20)

    metrics = train_classification()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    # class_weight="balanced" trades raw accuracy for minority-class recall on an
    # imbalanced target, so F1/ROC-AUC (not accuracy) is the fair comparison metric.
    assert metrics["f1_weighted"] > 0.0
    assert metrics["roc_auc_ovr_weighted"] is None or metrics["roc_auc_ovr_weighted"] >= 0.0
    assert (tmp_path / "models" / "clf.joblib").exists()


def test_train_regression_produces_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(config, "REGRESSOR_MODEL_PATH", tmp_path / "models" / "reg.joblib")
    monkeypatch.setattr(config, "REGRESSOR_METRICS_PATH", tmp_path / "reports" / "reg_metrics.json")
    monkeypatch.setitem(config.RF_REGRESSOR_PARAMS, "n_estimators", 20)

    metrics = train_regression()
    assert metrics["mae"] >= 0
    assert (tmp_path / "models" / "reg.joblib").exists()
