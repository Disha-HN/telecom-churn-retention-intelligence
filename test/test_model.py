from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "model"
    / "final_lightgbm_churn_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telecom_churn_prepared.csv"
)


def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_model_feature_schema_matches_data():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn_probability", "id"])

    assert list(model.feature_name_) == list(X.columns)


def test_model_produces_valid_probabilities():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn_probability", "id"])

    # Use a small sample to keep automated testing fast.
    predictions = model.predict_proba(X.head(10))[:, 1]

    assert len(predictions) == 10
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_model_produces_binary_predictions():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["churn_probability", "id"])

    predictions = model.predict(X.head(10))

    assert len(predictions) == 10
    assert set(predictions).issubset({0, 1})