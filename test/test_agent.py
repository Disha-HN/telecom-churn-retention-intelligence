import sys
from pathlib import Path


# Add the project root so the agent package can be imported during testing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent.tool import (
    get_customer_profile,
    get_high_risk_customers,
    get_priority_customers,
    get_retention_summary,
    analyze_customer_behavior,
)


def test_customer_profile():
    result = get_customer_profile.invoke({
        "customer_id": "2"
    })

    assert result is not None
    assert result["status"] == "success"
    assert result["customer_id"] == "2"

    assert "ml_churn_probability" in result
    assert "predicted_churn_risk_raw" in result
    assert "risk_tier" in result
    assert "retention_priority" in result
    assert "behavioral_state" in result


def test_invalid_customer_profile():
    result = get_customer_profile.invoke({
        "customer_id": "999999999"
    })

    assert result is not None
    assert result["status"] == "not_found"


def test_high_risk_customers():
    result = get_high_risk_customers.invoke({
        "limit": 10
    })

    assert result["status"] == "success"
    assert result["requested_limit"] == 10
    assert result["customers_returned"] == 10

    customers = result["customers"]

    assert len(customers) == 10

    risks = [
        customer["predicted_churn_risk_raw"]
        for customer in customers
    ]

    assert risks == sorted(risks, reverse=True)


def test_priority_customers():
    result = get_priority_customers.invoke({
        "limit": 10
    })

    assert result["status"] == "success"
    assert result["requested_limit"] == 10
    assert result["customers_returned"] == 10

    customers = result["customers"]

    assert len(customers) == 10

    valid_priorities = {
        "Priority 1",
        "Priority 2",
        "Priority 3",
    }

    for customer in customers:
        assert customer["retention_priority"] in valid_priorities


def test_retention_summary():
    result = get_retention_summary.invoke({})

    assert result is not None
    assert result["status"] == "success"


def test_customer_behavior():
    result = analyze_customer_behavior.invoke({
        "customer_id": "2"
    })

    assert result is not None
    assert result["status"] == "success"