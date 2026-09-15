import sys
from pathlib import Path

import pandas as pd


# Add the project root so application packages can be imported during testing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.tool import get_customer_profile


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telecom_retention_intelligence.csv"
)


def test_application_dataset_loads():
    df = pd.read_csv(DATA_PATH)

    assert not df.empty
    assert "id" in df.columns
    assert "predicted_churn_risk" in df.columns
    assert "risk_tier" in df.columns
    assert "retention_priority" in df.columns


def test_agent_reads_application_dataset():
    df = pd.read_csv(DATA_PATH)

    customer_id = str(df.iloc[0]["id"])

    result = get_customer_profile.invoke({
        "customer_id": customer_id
    })

    assert result["status"] == "success"
    assert result["customer_id"] == customer_id


def test_agent_and_dataset_use_same_customer():
    df = pd.read_csv(DATA_PATH)

    customer_id = str(df.iloc[0]["id"])

    result = get_customer_profile.invoke({
        "customer_id": customer_id
    })

    assert result["customer_id"] == customer_id
    assert result["risk_tier"] in {
        "Low",
        "Moderate",
        "High",
        "Very High",
    }