"""
Telecom Customer Churn Retention Agent - Tools

This module contains all tools used by the LangChain agent.

The tools provide two types of information:

1. Customer-level retention intelligence
2. Dataset-level retention statistics

The LLM does not directly read the CSV file.

Instead, the architecture is:

    CSV Dataset
         |
         v
    Retention Intelligence Tools
         |
         v
    LangChain Agent
         |
         v
      Groq LLM
         |
         v
    Business Explanation


IMPORTANT DATA DEFINITIONS
--------------------------

predicted_churn_risk
    This is the ML model's churn probability.

    Example:
        0.9860 -> 98.60%
        0.9377 -> 93.77%
        0.7515 -> 75.15%

risk_tier
    This is the risk classification produced by the
    retention-intelligence layer.

retention_priority
    This is the business retention priority produced by
    the retention-intelligence layer.

behavioral_state
    This describes the engineered behavioral state.

behavioral_evidence
    This describes the engineered behavioral evidence.

candidate_action
    This is the retention action recommended by the
    retention-intelligence layer.

churn_probability
    This field exists in the dataset but contains binary
    0/1 values in the current dataset.

    Therefore, it is NOT used as the ML churn probability.

The ML churn probability is always taken from:

    predicted_churn_risk
"""


# ============================================================
# IMPORTS
# ============================================================

import os
from typing import Optional

import pandas as pd
from langchain_core.tools import tool


# ============================================================
# PROJECT PATH CONFIGURATION
# ============================================================

# Get the absolute path of the current Python file.
CURRENT_FILE = os.path.abspath(__file__)

# Move one directory above the "agent" folder.
# This gives us the main project directory.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(CURRENT_FILE)
)

# Build the complete path to the processed dataset.
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "telecom_retention_intelligence.csv"
)


# ============================================================
# REQUIRED DATASET COLUMNS
# ============================================================

# These columns are required by the retention-intelligence tools.
REQUIRED_COLUMNS = {
    "id",
    "predicted_churn_risk",
    "risk_tier",
    "behavioral_state",
    "behavioral_evidence",
    "recent_deterioration_count",
    "persistent_deterioration_count",
    "coordinated_deterioration_count",
    "value_tier",
    "arpu_8",
    "retention_priority",
    "candidate_action",
    "decision_rationale",
    "churn_probability",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_retention_data() -> pd.DataFrame:
    """
    Load and validate the retention-intelligence dataset.

    Returns:
        pd.DataFrame:
            Validated retention-intelligence dataset.

    Raises:
        FileNotFoundError:
            If the processed CSV file does not exist.

        ValueError:
            If required columns are missing.
    """

    # Check whether the processed dataset exists.
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Retention intelligence dataset was not found at:\n"
            f"{DATA_PATH}"
        )

    # Load the CSV dataset.
    data = pd.read_csv(DATA_PATH)

    # Find required columns that are missing.
    missing_columns = (
        REQUIRED_COLUMNS - set(data.columns)
    )

    # Stop execution if the dataset structure is incorrect.
    if missing_columns:
        raise ValueError(
            "The retention intelligence dataset is missing "
            f"the following columns: {sorted(missing_columns)}"
        )

    # Convert customer IDs to strings.
    #
    # This prevents matching problems such as:
    # 13985 vs "13985"
    data["id"] = (
        data["id"]
        .astype(str)
        .str.strip()
    )

    # Return the validated dataset.
    return data


# Load the dataset once when this module is imported.
df = load_retention_data()


# ============================================================
# HELPER FUNCTION: NUMERIC CONVERSION
# ============================================================

def _to_numeric(value) -> Optional[float]:
    """
    Safely convert a value into a floating-point number.

    Args:
        value:
            Value to convert.

    Returns:
        float or None:
            Numeric value when conversion is successful.
            None when conversion is not possible.
    """

    # Convert the value to numeric.
    numeric_value = pd.to_numeric(
        value,
        errors="coerce"
    )

    # Return None when the value is missing or invalid.
    if pd.isna(numeric_value):
        return None

    # Return the numeric value.
    return float(numeric_value)


# ============================================================
# HELPER FUNCTION: ML CHURN PROBABILITY
# ============================================================

def _get_ml_churn_probability(
    value
) -> Optional[float]:
    """
    Convert the ML churn probability into percentage form.

    The dataset stores predicted_churn_risk as a decimal
    probability between 0 and 1.

    Examples:

        0.9860 -> 98.60
        0.9377 -> 93.77
        0.7515 -> 75.15
        0.1147 -> 11.47

    The returned value represents a percentage.
    """

    # Convert the value to a number.
    numeric_value = _to_numeric(value)

    # Return None if the value is invalid.
    if numeric_value is None:
        return None

    # Convert decimal probability into percentage.
    if 0 <= numeric_value <= 1:
        return numeric_value * 100

    # If the value is already expressed as a percentage,
    # return it unchanged.
    return numeric_value


# ============================================================
# HELPER FUNCTION: ML PROBABILITY SERIES
# ============================================================

def _get_ml_probability_series(
    data: pd.DataFrame
) -> pd.Series:
    """
    Convert predicted_churn_risk into percentage values.

    This function is used for:

        - High-risk ranking
        - Priority ranking
        - Average ML probability
        - Dataset statistics

    IMPORTANT:
        predicted_churn_risk is used.

        churn_probability is NOT used.
    """

    # Convert the ML probability column into numeric values.
    probabilities = pd.to_numeric(
        data["predicted_churn_risk"],
        errors="coerce"
    )

    # Identify decimal probability values.
    decimal_mask = probabilities.between(
        0,
        1
    )

    # Convert decimal values into percentages.
    probabilities.loc[decimal_mask] = (
        probabilities.loc[decimal_mask] * 100
    )

    # Return the converted probability series.
    return probabilities


# ============================================================
# HELPER FUNCTION: SAFE RESULT LIMIT
# ============================================================

def _safe_limit(
    value,
    default: int = 10,
    minimum: int = 1,
    maximum: int = 50
) -> int:
    """
    Safely convert a requested result limit into an integer.

    The limit is restricted to prevent excessively large
    responses from being sent to the LLM.
    """

    # Try to convert the value into an integer.
    try:
        limit = int(value)

    except (TypeError, ValueError):

        # Use the default value if conversion fails.
        limit = default

    # Keep the limit inside the allowed range.
    limit = max(
        minimum,
        min(limit, maximum)
    )

    # Return the validated limit.
    return limit


# ============================================================
# TOOL 1: GET CUSTOMER PROFILE
# ============================================================

@tool
def get_customer_profile(
    customer_id: str
) -> dict:
    """
    Retrieve the complete retention profile of one customer.

    Use this tool when the user asks about a specific customer.
    """

    # Convert the customer ID to a clean string.
    customer_id = str(customer_id).strip()

    # Search for the customer in the dataset.
    customer = df[
        df["id"] == customer_id
    ]

    # Return a clear response if the customer is not found.
    if customer.empty:
        return {
            "status": "not_found",
            "message": (
                f"Customer {customer_id} was not found in the "
                "available retention-intelligence dataset."
            )
        }

    # Get the first matching customer record.
    row = customer.iloc[0]

    # Convert the ML probability to percentage form.
    ml_probability = _get_ml_churn_probability(
        row["predicted_churn_risk"]
    )

    # Return the complete customer profile.
    return {
        "status": "success",

        # ----------------------------------------------------
        # Customer identification
        # ----------------------------------------------------

        "customer_id": customer_id,

        # ----------------------------------------------------
        # ML prediction
        # ----------------------------------------------------

        "ml_churn_probability": ml_probability,

        # Keep the original ML value available.
        "predicted_churn_risk_raw": row[
            "predicted_churn_risk"
        ],

        # ----------------------------------------------------
        # Retention intelligence
        # ----------------------------------------------------

        "risk_tier": row["risk_tier"],

        "retention_priority": row[
            "retention_priority"
        ],

        # ----------------------------------------------------
        # Behavioral information
        # ----------------------------------------------------

        "behavioral_state": row[
            "behavioral_state"
        ],

        "behavioral_evidence": row[
            "behavioral_evidence"
        ],

        "recent_deterioration_count": row[
            "recent_deterioration_count"
        ],

        "persistent_deterioration_count": row[
            "persistent_deterioration_count"
        ],

        "coordinated_deterioration_count": row[
            "coordinated_deterioration_count"
        ],

        # ----------------------------------------------------
        # Customer value
        # ----------------------------------------------------

        "value_tier": row[
            "value_tier"
        ],

        "arpu_8": row[
            "arpu_8"
        ],

        # ----------------------------------------------------
        # Retention recommendation
        # ----------------------------------------------------

        "candidate_action": row[
            "candidate_action"
        ],

        "decision_rationale": row[
            "decision_rationale"
        ],

        # ----------------------------------------------------
        # Raw dataset field
        # ----------------------------------------------------
        #
        # This field is preserved for transparency.
        # It is NOT treated as the ML probability.
        #

        "churn_probability_raw": row[
            "churn_probability"
        ],
    }


# ============================================================
# TOOL 2: GET HIGH-RISK CUSTOMERS
# ============================================================

@tool
def get_high_risk_customers(
    limit: int = 10
) -> dict:
    """
    Return customers with the highest ML churn probability.

    Ranking is performed using predicted_churn_risk.

    The existing risk_tier is preserved from the
    retention-intelligence layer.
    """

    # Validate the requested number of customers.
    limit = _safe_limit(limit)

    # Create a copy so the original dataset is not modified.
    data = df.copy()

    # Convert ML probability into percentage values.
    data["_ml_probability_percent"] = (
        _get_ml_probability_series(data)
    )

    # Remove records with invalid ML probability.
    data = data.dropna(
        subset=[
            "_ml_probability_percent"
        ]
    )

    # Sort from highest ML probability to lowest.
    data = data.sort_values(
        by="_ml_probability_percent",
        ascending=False
    )

    # Select only the requested number of customers.
    data = data.head(limit)

    # Create the result list.
    customers = []

    # Process each selected customer.
    for _, row in data.iterrows():

        customers.append(
            {
                # Customer identifier.
                "customer_id": str(
                    row["id"]
                ),

                # ML churn probability.
                "ml_churn_probability": round(
                    float(
                        row[
                            "_ml_probability_percent"
                        ]
                    ),
                    2
                ),

                # Original ML value.
                "predicted_churn_risk_raw": row[
                    "predicted_churn_risk"
                ],

                # Risk classification.
                "risk_tier": row[
                    "risk_tier"
                ],

                # Retention priority.
                "retention_priority": row[
                    "retention_priority"
                ],

                # Customer value tier.
                "value_tier": row[
                    "value_tier"
                ],

                # Behavioral state.
                "behavioral_state": row[
                    "behavioral_state"
                ],

                # Recommended action.
                "candidate_action": row[
                    "candidate_action"
                ],
            }
        )

    # Return the ranked customer list.
    return {
        "status": "success",

        "requested_limit": limit,

        "customers_returned": len(
            customers
        ),

        # Explicitly tell the LLM what was used
        # for the ranking.
        "ranking_basis": (
            "predicted_churn_risk, representing the "
            "ML model churn probability, in descending order."
        ),

        "customers": customers,
    }


# ============================================================
# TOOL 3: GET PRIORITY CUSTOMERS
# ============================================================

@tool
def get_priority_customers(
    limit: int = 10
) -> dict:
    """
    Return customers according to retention priority.

    Priority order:

        Priority 1
        Priority 2
        Priority 3

    Within the same priority group, customers with higher
    ML churn probability are shown first.
    """

    # Validate the requested limit.
    limit = _safe_limit(limit)

    # Create a copy of the dataset.
    data = df.copy()

    # Convert ML probability into percentages.
    data["_ml_probability_percent"] = (
        _get_ml_probability_series(data)
    )

    # Define the business priority order.
    priority_order = {
        "Priority 1": 1,
        "Priority 2": 2,
        "Priority 3": 3,
    }

    # Convert textual priority into a numerical ranking.
    data["_priority_rank"] = data[
        "retention_priority"
    ].map(priority_order)

    # Remove records with invalid priority or probability.
    data = data.dropna(
        subset=[
            "_priority_rank",
            "_ml_probability_percent"
        ]
    )

    # Sort using two criteria:
    #
    # 1. Retention priority
    # 2. ML churn probability
    #
    # Priority 1 comes first.
    # Within a priority group, higher churn probability
    # comes first.
    data = data.sort_values(
        by=[
            "_priority_rank",
            "_ml_probability_percent"
        ],
        ascending=[
            True,
            False
        ]
    )

    # Select the requested number of customers.
    data = data.head(limit)

    # Create the result list.
    customers = []

    # Process every selected customer.
    for _, row in data.iterrows():

        customers.append(
            {
                # Customer identifier.
                "customer_id": str(
                    row["id"]
                ),

                # Retention priority.
                "retention_priority": row[
                    "retention_priority"
                ],

                # ML churn probability.
                "ml_churn_probability": round(
                    float(
                        row[
                            "_ml_probability_percent"
                        ]
                    ),
                    2
                ),

                # Original ML value.
                "predicted_churn_risk_raw": row[
                    "predicted_churn_risk"
                ],

                # Risk tier.
                "risk_tier": row[
                    "risk_tier"
                ],

                # Customer value.
                "value_tier": row[
                    "value_tier"
                ],

                # Behavioral state.
                "behavioral_state": row[
                    "behavioral_state"
                ],

                # Recommended action.
                "candidate_action": row[
                    "candidate_action"
                ],
            }
        )

    # Return the prioritized customer list.
    return {
        "status": "success",

        "requested_limit": limit,

        "customers_returned": len(
            customers
        ),

        "ranking_basis": (
            "Retention priority first, followed by "
            "predicted_churn_risk in descending order."
        ),

        "customers": customers,
    }


# ============================================================
# TOOL 4: ANALYZE CUSTOMER BEHAVIOR
# ============================================================

@tool
def analyze_customer_behavior(
    customer_id: str
) -> dict:
    """
    Analyze the engineered behavioral evidence of a customer.

    This tool is used when the user asks why a customer
    is considered at risk.
    """

    # Clean the customer ID.
    customer_id = str(customer_id).strip()

    # Search for the customer.
    customer = df[
        df["id"] == customer_id
    ]

    # Return a not-found response when necessary.
    if customer.empty:
        return {
            "status": "not_found",
            "message": (
                f"Customer {customer_id} was not found in the "
                "available retention-intelligence dataset."
            )
        }

    # Get the customer's record.
    row = customer.iloc[0]

    # Return the behavioral analysis.
    return {
        "status": "success",

        # Customer identifier.
        "customer_id": customer_id,

        # ML probability.
        "ml_churn_probability": (
            _get_ml_churn_probability(
                row["predicted_churn_risk"]
            )
        ),

        # Original ML probability.
        "predicted_churn_risk_raw": row[
            "predicted_churn_risk"
        ],

        # Risk classification.
        "risk_tier": row[
            "risk_tier"
        ],

        # Retention priority.
        "retention_priority": row[
            "retention_priority"
        ],

        # Behavioral state.
        "behavioral_state": row[
            "behavioral_state"
        ],

        # Engineered behavioral evidence.
        "behavioral_evidence": row[
            "behavioral_evidence"
        ],

        # Deterioration counts.
        "recent_deterioration_count": row[
            "recent_deterioration_count"
        ],

        "persistent_deterioration_count": row[
            "persistent_deterioration_count"
        ],

        "coordinated_deterioration_count": row[
            "coordinated_deterioration_count"
        ],

        # Customer value.
        "value_tier": row[
            "value_tier"
        ],

        "arpu_8": row[
            "arpu_8"
        ],

        # Retention recommendation.
        "candidate_action": row[
            "candidate_action"
        ],

        # Reason for the recommendation.
        "decision_rationale": row[
            "decision_rationale"
        ],
    }


# ============================================================
# TOOL 5: GET RETENTION SUMMARY
# ============================================================

@tool
def get_retention_summary() -> dict:
    """
    Return overall retention statistics.

    All numerical calculations are performed by the tool.

    The LLM only explains the returned results.

    ML probability statistics are calculated from:

        predicted_churn_risk
    """

    # Create a copy of the dataset.
    data = df.copy()

    # ========================================================
    # TOTAL CUSTOMERS
    # ========================================================

    # Count the total number of customer records.
    total_customers = len(data)

    # ========================================================
    # ML CHURN PROBABILITY
    # ========================================================

    # Convert predicted_churn_risk into percentages.
    probability_series = (
        _get_ml_probability_series(data)
    )

    # Calculate the average ML churn probability.
    average_probability = (
        probability_series.mean()
    )

    # Count valid ML probability values.
    valid_probability_count = int(
        probability_series.notna().sum()
    )

    # ========================================================
    # RISK DISTRIBUTION
    # ========================================================

    # Count customers in each risk tier.
    risk_distribution = (
        data["risk_tier"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    # ========================================================
    # HIGH-RISK CUSTOMERS
    # ========================================================

    # These are the high-risk categories already defined
    # by the retention-intelligence layer.
    high_risk_tiers = {
        "High",
        "Very High",
    }

    # Count High and Very High customers.
    high_risk_count = int(
        data["risk_tier"]
        .isin(high_risk_tiers)
        .sum()
    )

    # Calculate their percentage of the dataset.
    if total_customers > 0:

        high_risk_percentage = (
            high_risk_count
            / total_customers
            * 100
        )

    else:

        high_risk_percentage = 0.0

    # ========================================================
    # RETENTION PRIORITY DISTRIBUTION
    # ========================================================

    # Count customers in each retention priority.
    priority_distribution = (
        data["retention_priority"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    # Count Priority 1 customers.
    priority_1_count = int(
        (
            data["retention_priority"]
            == "Priority 1"
        ).sum()
    )

    # Calculate Priority 1 percentage.
    if total_customers > 0:

        priority_1_percentage = (
            priority_1_count
            / total_customers
            * 100
        )

    else:

        priority_1_percentage = 0.0

    # ========================================================
    # BEHAVIORAL STATE DISTRIBUTION
    # ========================================================

    # Count customers in each behavioral state.
    behavioral_state_distribution = (
        data["behavioral_state"]
        .fillna("Unknown")
        .value_counts()
        .to_dict()
    )

    # ========================================================
    # RETURN SUMMARY
    # ========================================================

    return {
        "status": "success",

        # Dataset size.
        "total_customers": int(
            total_customers
        ),

        # Average ML churn probability.
        "average_ml_churn_probability": (
            round(
                float(average_probability),
                2
            )
            if pd.notna(
                average_probability
            )
            else None
        ),

        # Number of valid ML probability values.
        "customers_with_valid_ml_probability": (
            valid_probability_count
        ),

        # High-risk statistics.
        "high_or_very_high_risk_customers": (
            high_risk_count
        ),

        "high_or_very_high_risk_percentage": (
            round(
                float(high_risk_percentage),
                2
            )
        ),

        # Complete risk distribution.
        "risk_distribution": (
            risk_distribution
        ),

        # Priority 1 statistics.
        "priority_1_customers": (
            priority_1_count
        ),

        "priority_1_percentage": (
            round(
                float(priority_1_percentage),
                2
            )
        ),

        # Complete priority distribution.
        "priority_distribution": (
            priority_distribution
        ),

        # Behavioral state distribution.
        "behavioral_state_distribution": (
            behavioral_state_distribution
        ),

        # Explicitly identify the ML probability source.
        "ml_probability_source": (
            "predicted_churn_risk"
        ),

        # Explicitly identify the unused binary field.
        "churn_probability_note": (
            "The churn_probability dataset field is not "
            "used as the ML probability because it contains "
            "binary 0/1 values in the current dataset."
        ),
    }