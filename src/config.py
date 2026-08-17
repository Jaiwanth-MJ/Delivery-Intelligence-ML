"""
config.py
Central configuration for the Delivery Intelligence ML project.
Keeping all paths, constants, and hyperparameters in one place makes the
pipeline reproducible and avoids hardcoded values scattered across scripts.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Path configuration (all relative to project root -> portable across machines)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MERGED_DATASET_PATH = DATA_PROCESSED_DIR / "merged_dataset.csv"
FORECAST_RAW_PATH = DATA_RAW_DIR / "FORECAST.csv"

# --------------------------------------------------------------------------
# Raw source files (as provided in the original project)
# --------------------------------------------------------------------------
RAW_FILES = {
    "ob1": DATA_RAW_DIR / "OB1.csv",
    "ob2": DATA_RAW_DIR / "OB2.csv",
    "ob3": DATA_RAW_DIR / "OB3.csv",
    "ob4": DATA_RAW_DIR / "OB4.csv",
    "ob5": DATA_RAW_DIR / "OB5.csv",
    "forecast": DATA_RAW_DIR / "FORECAST.csv",
}
RAW_ENCODING = "latin1"  # source files contain non-UTF8 characters

# --------------------------------------------------------------------------
# Feature / target configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Classification task: predict customer segment
CLASSIFICATION_TARGET = "segment"
CLASSIFICATION_FEATURES = [
    "quantity",
    "delivery_lead_time",
    "price",
    "category",
    "carrier",
    "city_tier",
]

# Regression task: predict delivery lead time (days)
REGRESSION_TARGET = "delivery_lead_time"
REGRESSION_FEATURES = [
    "quantity",
    "price",
    "category",
    "carrier",
    "city_tier",
    "segment",
]

CATEGORICAL_COLUMNS = ["category", "carrier", "city_tier", "segment"]
NUMERIC_COLUMNS = ["quantity", "price", "delivery_lead_time"]

# --------------------------------------------------------------------------
# Model artifact names
# --------------------------------------------------------------------------
CLASSIFIER_MODEL_PATH = MODELS_DIR / "segment_classifier.joblib"
REGRESSOR_MODEL_PATH = MODELS_DIR / "lead_time_regressor.joblib"
CLASSIFIER_METRICS_PATH = REPORTS_DIR / "classification_metrics.json"
REGRESSOR_METRICS_PATH = REPORTS_DIR / "regression_metrics.json"

# --------------------------------------------------------------------------
# Hyperparameters (kept explicit and simple -> justified, not over-engineered)
# --------------------------------------------------------------------------
RF_CLASSIFIER_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 3,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_REGRESSOR_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CV_FOLDS = 5
