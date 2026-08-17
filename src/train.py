"""
train.py
Trains and persists two supervised models on the delivery-intelligence
dataset:

1. Classification -> predict customer `segment` (RandomForest vs Logistic
   Regression baseline vs majority-class baseline).
2. Regression      -> predict `delivery_lead_time` in days (RandomForest vs
   Linear Regression baseline).

Both models are wrapped end-to-end (preprocessing + estimator) in a single
sklearn Pipeline object and saved with joblib, so `predict.py` never needs to
re-implement feature encoding.

Usage
-----
    python -m src.train --task classification
    python -m src.train --task regression
    python -m src.train --task all
"""

import argparse
import json
import logging

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src import config
from src.data_loader import load_merged_dataset
from src.evaluate import evaluate_classifier, evaluate_regressor
from src.preprocessing import build_preprocessing_pipeline, clean_dataset, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_classification():
    logger.info("=== Training: Customer Segment Classification ===")
    df = clean_dataset(load_merged_dataset())
    X, y = split_features_target(df, config.CLASSIFICATION_FEATURES, config.CLASSIFICATION_TARGET)

    categorical = [c for c in config.CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in config.NUMERIC_COLUMNS if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessing_pipeline(categorical, numeric)
    model = RandomForestClassifier(**config.RF_CLASSIFIER_PARAMS)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    # Baselines for justified comparison
    baseline_majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    logreg_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(categorical, numeric)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    ).fit(X_train, y_train)

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")

    metrics = evaluate_classifier(
        pipeline, X_test, y_test, class_labels=sorted(y.unique().tolist())
    )
    metrics["cross_validation_f1_weighted"] = {
        "mean": float(np.mean(cv_scores)),
        "std": float(np.std(cv_scores)),
        "folds": cv_scores.tolist(),
    }
    metrics["baseline_majority_class_accuracy"] = float(
        baseline_majority.score(X_test, y_test)
    )
    metrics["baseline_logistic_regression_accuracy"] = float(
        logreg_pipeline.score(X_test, y_test)
    )
    metrics["model"] = "RandomForestClassifier"
    metrics["hyperparameters"] = config.RF_CLASSIFIER_PARAMS

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, config.CLASSIFIER_MODEL_PATH)
    with open(config.CLASSIFIER_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Classifier saved to %s", config.CLASSIFIER_MODEL_PATH)
    logger.info("Test accuracy: %.4f | Test F1 (weighted): %.4f",
                metrics["accuracy"], metrics["f1_weighted"])
    logger.info("Majority-class baseline accuracy: %.4f", metrics["baseline_majority_class_accuracy"])
    return metrics


def train_regression():
    logger.info("=== Training: Delivery Lead Time Regression ===")
    df = clean_dataset(load_merged_dataset())
    X, y = split_features_target(df, config.REGRESSION_FEATURES, config.REGRESSION_TARGET)

    categorical = [c for c in config.CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in config.NUMERIC_COLUMNS if c in X.columns and c != config.REGRESSION_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    preprocessor = build_preprocessing_pipeline(categorical, numeric)
    model = RandomForestRegressor(**config.RF_REGRESSOR_PARAMS)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    baseline_mean = DummyRegressor(strategy="mean").fit(X_train, y_train)
    linreg_pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessing_pipeline(categorical, numeric)),
            ("model", LinearRegression()),
        ]
    ).fit(X_train, y_train)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=config.CV_FOLDS, scoring="r2"
    )

    metrics = evaluate_regressor(pipeline, X_test, y_test)
    metrics["cross_validation_r2"] = {
        "mean": float(np.mean(cv_scores)),
        "std": float(np.std(cv_scores)),
        "folds": cv_scores.tolist(),
    }
    metrics["baseline_mean_predictor_r2"] = float(baseline_mean.score(X_test, y_test))
    metrics["baseline_linear_regression_r2"] = float(linreg_pipeline.score(X_test, y_test))
    metrics["model"] = "RandomForestRegressor"
    metrics["hyperparameters"] = config.RF_REGRESSOR_PARAMS

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, config.REGRESSOR_MODEL_PATH)
    with open(config.REGRESSOR_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Regressor saved to %s", config.REGRESSOR_MODEL_PATH)
    logger.info("Test R2: %.4f | Test MAE: %.4f | Test RMSE: %.4f",
                metrics["r2"], metrics["mae"], metrics["rmse"])
    logger.info("Linear regression baseline R2: %.4f", metrics["baseline_linear_regression_r2"])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train delivery-intelligence ML models.")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "all"],
        default="all",
        help="Which model to train.",
    )
    args = parser.parse_args()

    if args.task in ("classification", "all"):
        train_classification()
    if args.task in ("regression", "all"):
        train_regression()


if __name__ == "__main__":
    main()
