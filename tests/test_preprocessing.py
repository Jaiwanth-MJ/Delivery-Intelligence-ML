import pandas as pd

from src.preprocessing import build_preprocessing_pipeline, clean_dataset, split_features_target


def test_clean_dataset_drops_duplicates():
    df = pd.DataFrame(
        {"a": [1, 1, 2], "b": [" x", " x", "y"]}
    )
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2


def test_split_features_target_raises_on_missing_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    try:
        split_features_target(df, ["a", "c"], "b")
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_preprocessing_pipeline_transforms_shapes():
    df = pd.DataFrame(
        {
            "category": ["Noodles", "Snacks", "Noodles"],
            "carrier": ["akr express", "blue dart", "akr express"],
            "quantity": [10, 20, 30],
            "price": [5.5, 6.5, 7.5],
        }
    )
    pipeline = build_preprocessing_pipeline(
        categorical_columns=["category", "carrier"], numeric_columns=["quantity", "price"]
    )
    transformed = pipeline.fit_transform(df)
    assert transformed.shape[0] == 3
