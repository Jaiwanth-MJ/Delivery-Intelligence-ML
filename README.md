# Delivery Intelligence ML

Predicting **customer segment** and **delivery lead time** from order-logistics data, using a reproducible, testable, containerized scikit-learn pipeline.

This project converts an original MBA-style descriptive analytics exercise (correlation tables, ANOVA, groupby summaries) into a proper supervised-learning project: trained models, held-out evaluation, baseline comparisons, and a deployable REST API.

---

## 1. Problem Statement

An e-commerce/logistics operation logs orders across five extracts (`OB1`–`OB5`) capturing category, carrier, city tier, price, quantity, delivery lead time, customer, and an assigned RFM-style segment. Two supervised problems are addressed:

| Task | Type | Target | Why it matters |
|---|---|---|---|
| Customer Segment Prediction | Multi-class classification | `segment` (Champion / Loyal / Potential / At Risk / Lost) | Segment new/unlabelled customers for targeted marketing without waiting for a full RFM recompute |
| Delivery Lead Time Prediction | Regression | `delivery_lead_time` (days) | Estimate delivery time at order time from category, carrier, and location, before the order actually ships |

---

## 2. Dataset

- **Source**: `OB1.csv`–`OB5.csv` (11,436 rows each, row-aligned — verified 1:1 identical values on shared columns across files before merging positionally) + `FORECAST.csv` (3,728-row daily sales series, not used by the two supervised tasks above but retained for future time-series work).
- **Merged schema** (`data/processed/merged_dataset.csv`, built by `src/data_loader.py`):

  | Column | Type | Description |
  |---|---|---|
  | `delivery_lead_time` | int | Days between order and delivery |
  | `city_tier` | category | Tier-1 / Tier-2 / Tier-3 |
  | `carrier` | category | Shipping carrier (lower-cased, whitespace-normalized) |
  | `category` | category | Product category |
  | `quantity` | int | Units ordered |
  | `price` | float | Unit price |
  | `customer` | string | Customer identifier |
  | `segment` | category | RFM-style customer segment label (pre-assigned in source data) |
  | `order_date` | date | Parsed order date |

- **Class balance** (`segment`): Champion 40.5%, Loyal 22.7%, Potential 18.2%, At Risk 13.3%, Lost 5.3% — imbalanced, handled via `class_weight="balanced"` and weighted F1/ROC-AUC rather than raw accuracy.
- **Data quality**: no missing values or duplicate rows found in the merged set; assertions in `data_loader.py` enforce non-negative quantity/lead-time and complete segment labels at build time.

**Known limitation (documented, not hidden):** the `segment` label's original derivation rule (e.g. exact RFM thresholds) is not available in the source files, so it is used as given. This is disclosed rather than assumed to be leakage-free.

---

## 3. Project Structure

```
delivery-intelligence-ml/
├── data/
│   ├── raw/                  # Original OB1-OB5.csv, FORECAST.csv (untouched)
│   └── processed/            # merged_dataset.csv (generated, gitignored)
├── src/
│   ├── config.py             # All paths, hyperparameters, feature lists — single source of truth
│   ├── data_loader.py         # Merge + validate OB1-OB5 into one dataset
│   ├── preprocessing.py       # Cleaning + ColumnTransformer (encode/scale)
│   ├── train.py               # Trains classifier + regressor, runs CV, saves metrics
│   ├── evaluate.py            # Metrics + confusion matrix / residual / importance plots
│   ├── predict.py             # Inference API (loads saved pipeline, no re-fitting)
│   └── app.py                 # FastAPI serving layer
├── models/                    # segment_classifier.joblib, lead_time_regressor.joblib
├── reports/
│   ├── figures/               # confusion_matrix.png, residuals.png, feature_importance.png
│   ├── classification_metrics.json
│   └── regression_metrics.json
├── notebooks/
│   └── 01_eda_and_pipeline_overview.ipynb
├── tests/                     # pytest suite for data, preprocessing, training
├── .github/workflows/ci.yml   # lint -> test -> train -> docker build
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 4. Methodology

### Preprocessing (`src/preprocessing.py`)
- Deduplication, string trimming, carrier name normalization.
- `ColumnTransformer`: `OneHotEncoder` (categoricals) + `StandardScaler` (numerics), wrapped with `SimpleImputer` for robustness to missing values.
- Encoder/scaler are **fit only on training data** inside an sklearn `Pipeline`, then persisted as one artifact — eliminates train/serve skew by construction.

### Models & baselines (`src/train.py`)
| Task | Model | Baseline(s) |
|---|---|---|
| Classification | `RandomForestClassifier` (n=300, max_depth=12, class_weight="balanced") | Majority-class `DummyClassifier`, `LogisticRegression` |
| Regression | `RandomForestRegressor` (n=300, max_depth=12) | Mean-predictor `DummyRegressor`, `LinearRegression` |

- 5-fold cross-validation (`StratifiedKFold` for classification) on the training split.
- All hyperparameters centralized in `config.py`.

### Evaluation (`src/evaluate.py`)
- Classification: accuracy, precision/recall/F1 (weighted), ROC-AUC (OvR weighted), confusion matrix.
- Regression: MAE, MSE, RMSE, R².
- Feature importance (top 15) and diagnostic plots saved to `reports/figures/`.

---

## 5. Results

*(Generated by running this exact pipeline on the full dataset — reproduce with `python -m src.train --task all`.)*

### Classification — Customer Segment

| Metric | RandomForest | Logistic Regression | Majority-class baseline |
|---|---|---|---|
| Accuracy | 0.454 | 0.284 | 0.464 |
| F1 (weighted) | 0.468 | — | — |
| ROC-AUC (OvR, weighted) | 0.738 | — | — |
| 5-fold CV F1 (mean ± std) | 0.453 ± 0.015 | — | — |

**Honest interpretation:** raw accuracy is close to (and slightly below) the majority-class baseline. This is expected: `class_weight="balanced"` deliberately trades accuracy for better recall on minority classes (At Risk, Lost, Potential — see `reports/figures/confusion_matrix.png`), which a majority-only classifier would score 0% on. ROC-AUC of 0.738 shows the model does carry real discriminative signal beyond the class-imbalance baseline; F1-weighted is the fairer comparison metric here, not accuracy alone.

### Regression — Delivery Lead Time

| Metric | RandomForest | Linear Regression | Mean-predictor baseline |
|---|---|---|---|
| MAE | 9.92 days | — | — |
| RMSE | 18.47 days | — | — |
| R² (test) | 0.222 | 0.068 | -0.003 |
| 5-fold CV R² (mean ± std) | 0.090 ± 0.038 | — | — |

**Honest interpretation:** RandomForest clearly beats both baselines, confirming the available features (category, carrier, city tier, price, quantity, segment) carry predictive signal for lead time. The gap between test R² (0.222) and CV R² (0.090) suggests some variance across folds — a documented limitation, not a hidden one. Lead time is likely also driven by factors not present in this dataset (e.g. warehouse-to-destination distance, carrier real-time capacity), which caps achievable R² with these features alone.

Full metrics: `reports/classification_metrics.json`, `reports/regression_metrics.json`.
Figures: `reports/figures/`.

---

## 6. Setup & Usage

### Local (Python)

```bash
git clone <your-repo-url>
cd delivery-intelligence-ml
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Build the merged dataset
python -m src.data_loader

# Train both models (saves to models/, metrics to reports/)
python -m src.train --task all

# Run the test suite
pytest tests/ -v

# CLI inference example
python -m src.predict --task segment --input '{"quantity":250,"delivery_lead_time":12,"price":62.3,"category":"Noodles","carrier":"akr express","city_tier":"Tier-2"}'
```

### Docker

```bash
docker build -t delivery-intelligence-ml .
docker run -p 8000:8000 delivery-intelligence-ml
```

or with Compose:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### API Endpoints

| Method | Endpoint | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{"status": "ok"}` |
| POST | `/predict/segment` | `{"quantity", "delivery_lead_time", "price", "category", "carrier", "city_tier"}` | `{"predicted_segment": "..."}` |
| POST | `/predict/lead-time` | `{"quantity", "price", "category", "carrier", "city_tier", "segment"}` | `{"predicted_delivery_lead_time_days": ...}` |

Example:
```bash
curl -X POST http://localhost:8000/predict/segment \
  -H "Content-Type: application/json" \
  -d '{"quantity":250,"delivery_lead_time":12,"price":62.3,"category":"Noodles","carrier":"akr express","city_tier":"Tier-2"}'
```

---

## 7. CI/CD

`.github/workflows/ci.yml` runs on every push/PR to `main`:
1. Install dependencies
2. Lint (flake8, non-blocking)
3. Build the processed dataset
4. Run pytest
5. Retrain both models (integration smoke test) and upload artifacts
6. Build the Docker image

---

## 8. Limitations & Future Work

- Segment label provenance (original RFM rule) is not documented in source data — flagged, not fabricated.
- Delivery lead time R² is modest; likely missing predictive features (distance, real-time carrier load) not present in the source extracts.
- `FORECAST.csv` (daily sales time series) is loaded (`data_loader.load_forecast_series`) but not yet modeled — a natural next step is a lag-feature regression or ARIMA/Prophet forecast.
- Hyperparameter tuning was kept deliberately simple (fixed, justified values in `config.py`) rather than an exhaustive grid search, to keep the pipeline's behavior transparent and reproducible; `GridSearchCV`/`RandomizedSearchCV` over the existing `Pipeline` objects is a straightforward extension.

---

## 9. For AI Assistants / Future Contributors

- **Single source of truth for paths/hyperparameters**: `src/config.py`. Change values there, not inline in scripts.
- **Never re-fit encoders at inference time** — `predict.py` and `app.py` only call `.predict()` on the joblib-loaded `Pipeline`; if you need to retrain, use `train.py`.
- **Data flow**: `data/raw/*.csv` → `data_loader.build_merged_dataset()` → `data/processed/merged_dataset.csv` → `preprocessing.py` → `train.py` → `models/*.joblib` + `reports/*.json`.
- Run `pytest tests/ -v` before committing any change to `src/`.
- The row alignment between OB1–OB5 (positional merge) was empirically verified (see `notebooks/01_eda_and_pipeline_overview.ipynb`) — do not change the merge to a key-based join without re-validating row correspondence first.

---

## License

MIT — see `LICENSE`.
