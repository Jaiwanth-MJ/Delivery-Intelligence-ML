"""
data_loader.py
Loads the original five order-analytics extracts (OB1-OB5) and merges them
into a single canonical dataset used by the rest of the pipeline.

The five files are row-aligned (verified 1:1 match on Category, Quantity and
Delivery Lead Time across files during EDA) so a positional merge is valid
and avoids introducing a synthetic join key.
"""

import logging

import pandas as pd

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_raw(name: str) -> pd.DataFrame:
    path = config.RAW_FILES[name]
    if not path.exists():
        raise FileNotFoundError(f"Required raw file not found: {path}")
    return pd.read_csv(path, encoding=config.RAW_ENCODING)


def build_merged_dataset(save: bool = True) -> pd.DataFrame:
    """Merge OB1, OB2, OB3, OB5 into one tidy dataframe with snake_case columns.

    Returns
    -------
    pd.DataFrame
        Columns: delivery_lead_time, city_tier, carrier, category, quantity,
        order_date_serial, price, customer, segment, order_date
    """
    logger.info("Loading raw extracts (OB1, OB2, OB3, OB5)...")
    ob1 = _read_raw("ob1")
    ob2 = _read_raw("ob2")
    ob3 = _read_raw("ob3")
    ob5 = _read_raw("ob5")

    if not (len(ob1) == len(ob2) == len(ob3) == len(ob5)):
        raise ValueError("Row counts differ across source files; positional merge unsafe.")

    merged = ob1.rename(
        columns={
            "Delivey Lead Time": "delivery_lead_time",
            "City Tier": "city_tier",
            "Order Date": "order_date_serial",
        }
    )
    merged.columns = [c.lower().replace(" ", "_") for c in merged.columns]
    merged["customer"] = ob2["Customer"].values
    merged["segment"] = ob2["Segment"].values
    merged["order_date"] = pd.to_datetime(ob3["Order Date"], dayfirst=True, errors="coerce")

    # Sanity checks — fail loudly rather than silently propagate bad data
    assert merged["delivery_lead_time"].ge(0).all(), "Negative lead times detected"
    assert merged["quantity"].ge(0).all(), "Negative quantity detected"
    assert merged["segment"].notna().all(), "Missing segment labels detected"

    logger.info("Merged dataset shape: %s", merged.shape)

    if save:
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        merged.to_csv(config.MERGED_DATASET_PATH, index=False)
        logger.info("Saved merged dataset to %s", config.MERGED_DATASET_PATH)

    return merged


def load_merged_dataset() -> pd.DataFrame:
    """Load the merged dataset, building it first if it does not yet exist."""
    if config.MERGED_DATASET_PATH.exists():
        return pd.read_csv(config.MERGED_DATASET_PATH)
    return build_merged_dataset(save=True)


def load_forecast_series() -> pd.DataFrame:
    """Load the standalone time-series sales extract used for forecasting."""
    df = pd.read_csv(config.FORECAST_RAW_PATH, encoding=config.RAW_ENCODING)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    build_merged_dataset(save=True)
