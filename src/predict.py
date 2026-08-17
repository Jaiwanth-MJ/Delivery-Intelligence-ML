"""
predict.py
Loads trained model pipelines and exposes a simple inference API used by
both the CLI and (optionally) a REST wrapper. No re-fitting happens here —
the exact preprocessing fitted during training is reused, which prevents
train/serve skew.
"""

import argparse
import json
from typing import Dict, List, Union

import joblib
import pandas as pd

from src import config


def _load_pipeline(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m src.train` first."
        )
    return joblib.load(path)


def predict_segment(records: Union[Dict, List[Dict]]) -> List[str]:
    """Predict customer segment for one or more order records.

    Each record must contain: quantity, delivery_lead_time, price,
    category, carrier, city_tier.
    """
    pipeline = _load_pipeline(config.CLASSIFIER_MODEL_PATH)
    df = pd.DataFrame(records if isinstance(records, list) else [records])
    df = df[config.CLASSIFICATION_FEATURES]
    return pipeline.predict(df).tolist()


def predict_lead_time(records: Union[Dict, List[Dict]]) -> List[float]:
    """Predict delivery lead time (days) for one or more order records.

    Each record must contain: quantity, price, category, carrier,
    city_tier, segment.
    """
    pipeline = _load_pipeline(config.REGRESSOR_MODEL_PATH)
    df = pd.DataFrame(records if isinstance(records, list) else [records])
    df = df[config.REGRESSION_FEATURES]
    return pipeline.predict(df).tolist()


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained models.")
    parser.add_argument(
        "--task", choices=["segment", "lead_time"], required=True, help="Which model to query."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSON string or path to a .json file describing one record or a list of records.",
    )
    args = parser.parse_args()

    if args.input.strip().startswith("{") or args.input.strip().startswith("["):
        payload = json.loads(args.input)
    else:
        with open(args.input) as f:
            payload = json.load(f)

    if args.task == "segment":
        result = predict_segment(payload)
    else:
        result = predict_lead_time(payload)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
