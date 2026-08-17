import pandas as pd
import pytest

from src.data_loader import build_merged_dataset, load_forecast_series


def test_build_merged_dataset_shape():
    df = build_merged_dataset(save=False)
    assert df.shape[0] > 0
    expected_cols = {
        "delivery_lead_time",
        "city_tier",
        "carrier",
        "category",
        "quantity",
        "price",
        "customer",
        "segment",
        "order_date",
    }
    assert expected_cols.issubset(set(df.columns))


def test_no_negative_values():
    df = build_merged_dataset(save=False)
    assert (df["delivery_lead_time"] >= 0).all()
    assert (df["quantity"] >= 0).all()
    assert (df["price"] >= 0).all()


def test_segment_labels_present():
    df = build_merged_dataset(save=False)
    assert df["segment"].isna().sum() == 0
    assert df["segment"].nunique() >= 2


def test_forecast_series_sorted_by_date():
    df = load_forecast_series()
    assert df["date"].is_monotonic_increasing
    assert df["sales"].notna().all()
