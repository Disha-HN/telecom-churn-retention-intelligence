from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telecom_retention_intelligence.csv"
)


def test_dataset_exists():
    assert DATA_PATH.exists(), f"Dataset not found: {DATA_PATH}"


def test_dataset_structure():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "id",
        "predicted_churn_risk",
        "risk_tier",
        "behavioral_state",
        "retention_priority",
        "candidate_action",
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing required column: {column}"


def test_customer_ids_are_unique():
    df = pd.read_csv(DATA_PATH)

    assert df["id"].is_unique


def test_churn_risk_is_valid():
    df = pd.read_csv(DATA_PATH)

    assert df["predicted_churn_risk"].between(0, 1).all()


def test_retention_priority_values_are_valid():
    df = pd.read_csv(DATA_PATH)

    valid_priorities = {
        "Priority 1",
        "Priority 2",
        "Priority 3",
    }

    assert set(df["retention_priority"].dropna()).issubset(
        valid_priorities
    )