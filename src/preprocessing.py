"""
preprocessing.py
Feature preparation shared by both the classification and regression tasks.

Design notes
------------
* Encoders/scalers are fit ONLY on training data and reused on test data to
  prevent data leakage.
* A scikit-learn ColumnTransformer + Pipeline is used so the exact same
  transformation is guaranteed at inference time (predict.py loads the same
  fitted pipeline object that was trained, never re-fits on new data).
"""

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning shared by all tasks: dedupe, trim strings, drop nulls."""
    df = df.copy()
    df = df.drop_duplicates()

    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # Normalise inconsistent carrier casing/spelling observed in EDA
    if "carrier" in df.columns:
        df["carrier"] = df["carrier"].str.lower().str.strip()

    return df


def build_preprocessing_pipeline(
    categorical_columns: List[str], numeric_columns: List[str]
) -> ColumnTransformer:
    """Build a reusable ColumnTransformer for categorical + numeric features."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_columns),
            ("num", numeric_pipeline, numeric_columns),
        ]
    )
    return preprocessor


def split_features_target(
    df: pd.DataFrame, feature_columns: List[str], target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    missing = set(feature_columns + [target_column]) - set(df.columns)
    if missing:
        raise KeyError(f"Columns missing from dataframe: {missing}")
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    return X, y
