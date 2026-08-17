"""
evaluate.py
Evaluation metrics for the classification and regression tasks, plus
diagnostic plots (confusion matrix, feature importance, residuals) saved to
reports/figures/ for inclusion in the project report.
"""

from typing import List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI / Docker environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src import config


def evaluate_classifier(pipeline, X_test, y_test, class_labels: List[str]) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=class_labels).tolist(),
        "class_labels": class_labels,
    }

    try:
        y_test_bin = label_binarize(y_test, classes=class_labels)
        metrics["roc_auc_ovr_weighted"] = float(
            roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
        )
    except ValueError:
        metrics["roc_auc_ovr_weighted"] = None

    _plot_confusion_matrix(y_test, y_pred, class_labels)
    _plot_feature_importance(pipeline, "classification_feature_importance.png")

    return metrics


def evaluate_regressor(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    _plot_residuals(y_test, y_pred)
    _plot_feature_importance(pipeline, "regression_feature_importance.png")

    return metrics


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def _plot_confusion_matrix(y_test, y_pred, class_labels):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=class_labels, ax=ax, xticks_rotation=45, cmap="Blues"
    )
    ax.set_title("Confusion Matrix — Customer Segment Classifier")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def _plot_residuals(y_test, y_pred):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    residuals = np.array(y_test) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted Delivery Lead Time (days)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs Predicted")

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Residual")
    axes[1].set_title("Residual Distribution")

    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "regression_residuals.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(pipeline, filename: str):
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_
    except AttributeError:
        return  # model without feature_importances_ (e.g. linear baseline)

    order = np.argsort(importances)[::-1][:15]
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="seagreen",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Feature Importances")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=150)
    plt.close(fig)
