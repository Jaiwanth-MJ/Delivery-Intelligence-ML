# Delivery Intelligence ML

An end-to-end, production-grade machine learning system designed to predict **customer segment** and **delivery lead times** from order and logistics transaction logs using a reproducible, containerized scikit-learn pipeline and FastAPI REST API.

> [!NOTE]
> **Confidentiality & Anonymization Notice**
> The raw datasets (`data/raw/`) contain completely synthetic, anonymized dummy data generated to protect proprietary commercial metrics. All client names, prices, quantities, and carrier associations have been fully randomized. Performance metrics reflect training on these synthetic distributions for demonstration, testing, and deployment verification.

---

## 1. Project Overview

This project implements a multi-task machine learning system to address two primary business objectives:

| Target | Task Type | Description |
|---|---|---|
| `segment` | Multi-class Classification | Predicts the RFM customer segment (Champion, Loyal, Potential, At Risk, Lost) to enable real-time targeted marketing. |
| `delivery_lead_time` | Regression | Estimates the delivery duration in days at order time, enabling dynamic SLA predictions before fulfillment. |

---

## 2. Tech Stack

*   **Language**: Python 3.11+
*   **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`
*   **REST API**: `FastAPI`, `uvicorn`, `pydantic`
*   **Testing**: `pytest`
*   **Visualizations**: `matplotlib`
*   **Containerization & Deployment**: `Docker`, `Docker Compose`
*   **CI/CD**: GitHub Actions

---

## 3. Project Structure

```text
delivery-intelligence-ml/
├── .github/workflows/ci.yml   # CI pipeline (lint -> test -> build)
├── data/
│   ├── raw/                  # Original OB1-OB5.csv, FORECAST.csv (anonymized)
│   └── processed/            # merged_dataset.csv (generated at runtime)
├── src/
│   ├── config.py             # Paths, feature settings, and model hyperparameters
│   ├── data_loader.py         # Merges raw datasets positionally & runs assertions
│   ├── preprocessing.py       # Data cleaning and scikit-learn ColumnTransformer
│   ├── train.py               # Pipeline fitting, baseline comparison, model export
│   ├── evaluate.py            # Generates classification/regression metrics & plots
│   ├── predict.py             # CLI inference loading serialized pipelines
│   └── app.py                 # FastAPI serving endpoints
├── models/                    # Serialized joblib pipelines (preprocessor + estimator)
├── reports/
│   ├── figures/               # Confusion matrix, residual plots, feature importances
│   ├── classification_metrics.json
│   └── regression_metrics.json
├── notebooks/                 # Exploratory data analysis (EDA)
├── tests/                     # Unit and integration test suite
├── Dockerfile                 # Multi-stage container definition
├── docker-compose.yml         # Local orchestration file
├── requirements.txt           # Package dependencies
└── README.md
```

---

## 4. Pipeline Architecture & Methodology

*   **Robust Preprocessing**: Uses a scikit-learn `ColumnTransformer` wrapping a `OneHotEncoder` (handling unseen categories gracefully at inference) and a `StandardScaler`. All steps are nested within a parent `Pipeline` to prevent data leakage.
*   **Zero Train-Serve Skew**: Encoders and scalers are fitted on the training split *only* and serialized along with the model weights into a single `joblib` object. The API directly calls `.predict()` on the loaded pipeline, ensuring identical data preparation steps.
*   **Baseline Comparisons**: Evaluates models against standard baseline estimators (`DummyClassifier` majority-class and `DummyRegressor` mean-predictor) and linear benchmarks (`LogisticRegression`, `LinearRegression`) to justify model choice.

---

## 5. Setup & Usage

### Local Python Environment

1.  **Clone and Install Dependencies**:
    ```bash
    git clone https://github.com/Jaiwanth-MJ/Delivery-Intelligence-ML.git
    cd delivery-intelligence-ml
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Ingest & Merge Data**:
    ```bash
    python -m src.data_loader
    ```

3.  **Train & Evaluate Models**:
    ```bash
    python -m src.train --task all
    ```

4.  **Run Tests**:
    ```bash
    pytest tests/ -v
    ```

5.  **Run CLI Inference Example**:
    ```bash
    python -m src.predict --task segment --input '{"quantity":250,"delivery_lead_time":12,"price":62.3,"category":"Noodles","carrier":"akr express","city_tier":"Tier-2"}'
    ```

---

## 6. Docker Deployment

### Multi-stage Docker Build
Build and run the containerized FastAPI server locally:
```bash
docker build -t delivery-intelligence-ml .
docker run -p 8000:8000 delivery-intelligence-ml
```

### Docker Compose
Run the stack using Docker Compose:
```bash
docker compose up --build
```
The REST API will be available at `http://localhost:8000`. Access interactive docs at `http://localhost:8000/docs`.

---

## 7. API Endpoints

| Method | Endpoint | Description | Payload Schema |
|---|---|---|---|
| **GET** | `/health` | Liveness health check | None |
| **POST** | `/predict/segment` | Predicts RFM Customer Segment | `{"quantity", "delivery_lead_time", "price", "category", "carrier", "city_tier"}` |
| **POST** | `/predict/lead-time` | Predicts Delivery Lead Time (Days) | `{"quantity", "price", "category", "carrier", "city_tier", "segment"}` |

---

## 8. Continuous Integration

The repository uses GitHub Actions (`.github/workflows/ci.yml`) to automatically validate modifications on every push or pull request:
1. Installs project dependencies.
2. Runs style checks (`flake8`).
3. Compiles the dataset (`src.data_loader`).
4. Executes unit tests (`pytest`).
5. Runs smoke test model training and uploads reports/model artifacts.
6. Builds the Docker production image.
