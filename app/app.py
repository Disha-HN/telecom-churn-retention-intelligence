import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Customer Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "telecom_retention_intelligence.csv",
)


# ============================================================
# OPTIONAL AI AGENT IMPORT
# ============================================================

try:
    from agent.agent import ask_agent

    AGENT_AVAILABLE = True
    AGENT_ERROR = None

except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

    def ask_agent(question):
        return (
            "The AI Retention Agent is currently unavailable.\n\n"
            f"Reason: {e}"
        )


# ============================================================
# CUSTOM CSS
#
# IMPORTANT:
# We use CSS only for styling.
# We DO NOT use HTML div cards for the UI.
# This avoids the HTML rendering problem you were seeing.
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       GLOBAL
    -------------------------------------------------------- */

    .stApp {
        background-color: #faf9ff;
    }

    .main {
        background-color: #faf9ff;
    }

    h1, h2, h3, h4 {
        color: #30204f !important;
    }

    p, label, span {
        color: #333333;
    }


    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    section[data-testid="stSidebar"] {
        background-color: #f0eaff;
        border-right: 1px solid #ddd2f5;
    }

    section[data-testid="stSidebar"] * {
        color: #30204f;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #30204f !important;
        font-weight: 500;
    }


    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #8e6bc7;
        background-color: #ffffff;
        color: #5d3c91;
        font-weight: 600;
    }

    .stButton > button:hover {
        border-color: #6e4aa5;
        color: #ffffff;
        background-color: #6e4aa5;
    }


    /* --------------------------------------------------------
       METRIC BOX
    -------------------------------------------------------- */

    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e1d9f2;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(60, 40, 90, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #6d6280 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #4f3281 !important;
        font-weight: 700;
    }


    /* --------------------------------------------------------
       DATAFRAME
    -------------------------------------------------------- */

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }


    /* --------------------------------------------------------
       EXPANDER
    -------------------------------------------------------- */

    div[data-testid="stExpander"] {
        border: 1px solid #ded5ef;
        border-radius: 10px;
        background-color: #ffffff;
    }


    /* --------------------------------------------------------
       INFO BOXES
    -------------------------------------------------------- */

    .info-text {
        background-color: #f1ebff;
        border-left: 5px solid #7952b3;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert NumPy/Pandas values into normal Python values
    so Streamlit never displays things like np.int64(3).
    """

    if pd.isna(value):
        return "-"

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_probability(value):
    """
    Convert probability into percentage.
    """

    value = safe_float(value)

    # Handle values already represented as percentages
    if value > 1:
        return f"{value:.2f}%"

    return f"{value * 100:.2f}%"


def risk_color(risk):
    risk = str(risk).lower()

    if "very high" in risk:
        return "🔴"

    if risk == "high":
        return "🟠"

    if "moderate" in risk:
        return "🟡"

    if "low" in risk:
        return "🟢"

    return "⚪"


def display_page_title(title, description=None):
    """
    Native Streamlit title.
    No custom HTML.
    """

    st.title(title)

    if description:
        st.caption(description)


def display_info_box(title, message):
    """
    Use Streamlit's native info component.
    """

    st.info(f"**{title}**\n\n{message}")


def display_profile_card(customer):
    """
    Display customer information using native Streamlit
    columns instead of HTML.
    """

    st.subheader("👤 Customer Profile")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Customer ID",
            str(clean_value(customer.get("id", "-")))
        )

    with c2:
        st.metric(
            "Churn Risk",
            format_probability(
                customer.get("predicted_churn_risk", 0)
            )
        )

    with c3:
        risk = str(
            clean_value(
                customer.get("risk_tier", "-")
            )
        )

        st.metric(
            "Risk Tier",
            f"{risk_color(risk)} {risk}"
        )

    with c4:
        value_tier = str(
            clean_value(
                customer.get("value_tier", "-")
            )
        )

        st.metric(
            "Value Tier",
            value_tier
        )


def display_metric_card(label, value, help_text=None):
    """
    Native Streamlit metric.
    """

    st.metric(
        label=label,
        value=value,
        help=help_text
    )


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    return df


df = load_data(DATA_PATH)


# ============================================================
# DATA VALIDATION
# ============================================================

if df is None:

    st.error(
        "❌ Processed retention intelligence dataset was not found."
    )

    st.code(DATA_PATH)

    st.stop()


if df.empty:

    st.error(
        "❌ The retention intelligence dataset is empty."
    )

    st.stop()


# ============================================================
# BASIC COLUMN CHECK
# ============================================================

required_columns = [
    "id",
    "predicted_churn_risk",
    "risk_tier",
]


missing_columns = [
    col for col in required_columns
    if col not in df.columns
]


if missing_columns:

    st.error(
        "The processed dataset is missing required columns:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

df["predicted_churn_risk"] = pd.to_numeric(
    df["predicted_churn_risk"],
    errors="coerce"
).fillna(0)


# Make sure risk tier is readable
df["risk_tier"] = (
    df["risk_tier"]
    .astype(str)
    .replace("nan", "Unknown")
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📡 Telecom AI")

    st.caption("Churn Intelligence System")

    st.divider()

    st.subheader("Navigation")

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Executive Dashboard",
            "👤 Customer Analysis",
            "🚨 High-Risk Customers",
            "🧠 Retention Intelligence",
            "🤖 AI Retention Agent",
            "📈 Retention Operations",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Dataset Information")

    st.write(
        f"**Customers:** {len(df):,}"
    )

    st.success(
        "ML churn predictions available"
    )

    behavioral_columns = [
        "behavioral_state",
        "behavioral_evidence",
        "recent_deterioration_count",
        "persistent_deterioration_count",
        "coordinated_deterioration_count",
    ]

    behavioral_available = any(
        col in df.columns
        for col in behavioral_columns
    )

    if behavioral_available:
        st.success(
            "Behavioral intelligence available"
        )
    else:
        st.warning(
            "Behavioral intelligence unavailable"
        )

    st.divider()

    st.caption(
        "Telecom Customer Churn Intelligence"
    )


# ============================================================
# HEADER
# ============================================================

st.title("📡 Telecom Customer Churn Intelligence")

st.write(
    "AI-powered customer churn analysis, "
    "behavioral intelligence and retention "
    "decision support."
)

st.divider()


# ============================================================
# PAGE 1
# EXECUTIVE DASHBOARD
# ============================================================

if page == "📊 Executive Dashboard":

    display_page_title(
        "📊 Executive Dashboard",
        "Overview of customer churn risk, value and retention priorities."
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    total_customers = len(df)

    high_risk_count = int(
        df["risk_tier"]
        .astype(str)
        .str.lower()
        .isin(["high", "very high"])
        .sum()
    )
    priority_1_count = int(
        (df["retention_priority"] == "Priority 1").sum())
    priority_2_count = int(
        (df["retention_priority"] == "Priority 2").sum())
    retention_priority_count = priority_1_count + priority_2_count

    average_churn_risk = (
        df["predicted_churn_risk"].mean()
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        display_metric_card(
            "Total Customers",
            f"{total_customers:,}"
        )

    with m2:
        display_metric_card(
            "High-Risk Customers",
            f"{high_risk_count:,}"
        )

    with m3:
        display_metric_card(
            "Priority 1 + 2 Customers",
            f"{retention_priority_count:,}",
            "Customers currently assigned higher retention attention."
        )

    with m4:
        display_metric_card(
            "Average Churn Risk",
            format_probability(average_churn_risk)
        )

    st.divider()

    # --------------------------------------------------------
    # RISK OVERVIEW
    # --------------------------------------------------------

    st.subheader("Risk Overview")

    risk_order = [
        "Low",
        "Moderate",
        "High",
        "Very High",
    ]

    risk_counts = (
        df["risk_tier"]
        .value_counts()
        .reindex(risk_order)
        .fillna(0)
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Tier",
        "Customers"
    ]

    c1, c2 = st.columns(2)

    with c1:

        fig_risk = px.bar(
            risk_counts,
            x="Risk Tier",
            y="Customers",
            title="Customers by Risk Tier",
        )

        fig_risk.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig_risk,
            width="stretch"
        )

    with c2:

        fig_pie = px.pie(
            risk_counts,
            names="Risk Tier",
            values="Customers",
            title="Risk Distribution",
        )

        fig_pie.update_layout(
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_pie,
            width="stretch"
        )

    # --------------------------------------------------------
    # CUSTOMER VALUE
    # --------------------------------------------------------

    st.subheader("Customer Value")

    if "value_tier" in df.columns:

        value_counts = (
            df["value_tier"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        value_counts.columns = [
            "Value Tier",
            "Customers"
        ]

        fig_value = px.bar(
            value_counts,
            x="Value Tier",
            y="Customers",
            title="Customers by Value Tier",
        )

        fig_value.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig_value,
            width="stretch"
        )

    else:

        st.info(
            "Value-tier information is not available in the processed dataset."
        )


# ============================================================
# PAGE 2
# CUSTOMER ANALYSIS
# ============================================================

elif page == "👤 Customer Analysis":

    display_page_title(
        "👤 Customer Analysis",
        "Inspect an individual customer's churn risk and behavioral signals."
    )

    customer_ids = df["id"].tolist()

    selected_id = st.selectbox(
        "Select Customer",
        customer_ids,
        format_func=lambda x: str(clean_value(x)),
    )

    selected_rows = df[
        df["id"] == selected_id
    ]

    if selected_rows.empty:

        st.warning(
            "Customer not found."
        )

        st.stop()

    customer = selected_rows.iloc[0].to_dict()

    display_profile_card(customer)

    st.divider()

    # --------------------------------------------------------
    # BEHAVIORAL STATE
    # --------------------------------------------------------

    st.subheader("🧠 Behavioral Intelligence")

    behavioral_state = customer.get(
        "behavioral_state",
        "Not available"
    )

    behavioral_evidence = customer.get(
        "behavioral_evidence",
        "Not available"
    )

    st.write(
        f"**Behavioral State:** "
        f"{clean_value(behavioral_state)}"
    )

    st.write(
        f"**Evidence:** "
        f"{clean_value(behavioral_evidence)}"
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        display_metric_card(
            "Recent Deterioration",
            safe_int(
                customer.get(
                    "recent_deterioration_count",
                    0
                )
            )
        )

    with b2:
        display_metric_card(
            "Persistent Deterioration",
            safe_int(
                customer.get(
                    "persistent_deterioration_count",
                    0
                )
            )
        )

    with b3:
        display_metric_card(
            "Coordinated Deterioration",
            safe_int(
                customer.get(
                    "coordinated_deterioration_count",
                    0
                )
            )
        )

    st.divider()

    # --------------------------------------------------------
    # FINANCIAL / USAGE INFORMATION
    # --------------------------------------------------------

    st.subheader("📈 Customer Indicators")

    indicator_columns = [
        "arpu_8",
        "total_rech_amt_8",
        "total_rech_num_8",
        "total_og_mou_8",
        "total_ic_mou_8",
    ]

    available_indicators = [
        col for col in indicator_columns
        if col in df.columns
    ]

    if available_indicators:

        cols = st.columns(
            min(len(available_indicators), 5)
        )

        for i, col in enumerate(available_indicators):

            value = customer.get(col)

            with cols[i % len(cols)]:

                display_metric_card(
                    col.replace("_8", "")
                    .replace("_", " ")
                    .title(),
                    f"{safe_float(value):,.2f}"
                )

    else:

        st.info(
            "Customer indicator fields are not available."
        )

    # --------------------------------------------------------
    # DECISION SUPPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader("🎯 Retention Decision")

    retention_priority = customer.get(
        "retention_priority",
        "Not available"
    )

    candidate_action = customer.get(
        "candidate_action",
        "Not available"
    )

    decision_rationale = customer.get(
        "decision_rationale",
        "Not available"
    )

    st.write(
        f"**Retention Priority:** "
        f"{clean_value(retention_priority)}"
    )

    st.write(
        f"**Candidate Action:** "
        f"{clean_value(candidate_action)}"
    )

    st.write(
        f"**Decision Rationale:** "
        f"{clean_value(decision_rationale)}"
    )


# ============================================================
# PAGE 3
# HIGH-RISK CUSTOMERS
# ============================================================

elif page == "🚨 High-Risk Customers":

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------

    display_page_title(
        "🚨 High-Risk Customers",
        "Customers ranked by predicted churn probability."
    )

    # --------------------------------------------------------
    # HIGH-RISK CUSTOMER FILTER
    #
    # This view focuses on customers classified as High or
    # Very High according to the ML-derived risk tier.
    #
    # Important distinction:
    # Risk ranking answers:
    # "Who is most likely to churn?"
    #
    # Retention priority answers:
    # "Who should the business prioritize?"
    #
    # These are intentionally kept as separate concepts.
    # --------------------------------------------------------

    high_risk_df = df[
        df["risk_tier"]
        .astype(str)
        .str.lower()
        .isin(["high", "very high"])
    ].copy()

    # --------------------------------------------------------
    # EMPTY-STATE HANDLING
    #
    # Prevent downstream operations such as iloc[0] when
    # no customers satisfy the selected risk criteria.
    # --------------------------------------------------------

    if high_risk_df.empty:

        st.success(
            "No high-risk customers found."
        )

    else:

        st.write(
            f"**{len(high_risk_df):,} customers in the "
            f"High / Very High risk tiers.**"
        )

        # ----------------------------------------------------
        # RISK INTERPRETATION
        #
        # The customer ranking on this page is based entirely
        # on the ML predicted churn probability.
        #
        # Retention priority is calculated separately using
        # risk, customer value and behavioural evidence.
        # ----------------------------------------------------

        st.info(
            "This view ranks customers by ML-predicted churn "
            "probability. Retention Priority is a separate "
            "business decision that also considers customer "
            "value and behavioural evidence."
        )

        # ----------------------------------------------------
        # SORT BY ML CHURN RISK
        #
        # Highest predicted churn probability appears first.
        # ----------------------------------------------------

        high_risk_df = high_risk_df.sort_values(
            "predicted_churn_risk",
            ascending=False
        )

        # ----------------------------------------------------
        # TOP CUSTOMER METRICS
        #
        # These metrics provide a quick summary of the highest
        # churn-risk customer and the size of the risk population.
        # ----------------------------------------------------

        top_risk = high_risk_df.iloc[0]

        m1, m2, m3 = st.columns(3)

        with m1:

            display_metric_card(
                "Highest Churn-Risk Customer",
                str(
                    clean_value(
                        top_risk["id"]
                    )
                )
            )

        with m2:

            display_metric_card(
                "Highest Churn Risk",
                format_probability(
                    top_risk["predicted_churn_risk"]
                )
            )

        with m3:

            display_metric_card(
                "High-Risk Population",
                f"{len(high_risk_df):,}"
            )

        st.divider()

        # ----------------------------------------------------
        # CUSTOMER RISK TABLE
        #
        # ML risk is shown together with behavioural state,
        # customer value and retention priority.
        #
        # This allows the user to understand why a high-risk
        # customer may still receive a different retention
        # priority.
        # ----------------------------------------------------

        st.subheader(
            "Customers Ranked by Churn Probability"
        )

        display_columns = [
            "id",
            "predicted_churn_risk",
            "risk_tier",
            "behavioral_state",
            "value_tier",
            "retention_priority",
            "candidate_action",
        ]

        # Keep only columns that are available in the processed
        # retention-intelligence dataset.
        display_columns = [
            col
            for col in display_columns
            if col in high_risk_df.columns
        ]

        table_df = high_risk_df[
            display_columns
        ].copy()

        # ----------------------------------------------------
        # DISPLAY-ONLY RISK FORMATTING
        #
        # The original dataframe retains numeric probabilities.
        # Formatting is applied only to the table copy so that
        # the underlying data remains suitable for calculations.
        # ----------------------------------------------------

        if "predicted_churn_risk" in table_df.columns:

            table_df["predicted_churn_risk"] = (
                table_df["predicted_churn_risk"]
                .apply(format_probability)
            )

        st.dataframe(
            table_df,
            width="stretch",
            hide_index=True,
        )

        # ----------------------------------------------------
        # DOWNLOAD
        #
        # The downloadable file retains the original numeric
        # churn-risk values rather than the display-formatted
        # percentages.
        # ----------------------------------------------------

        csv_data = high_risk_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download High-Risk Customers",
            data=csv_data,
            file_name="high_risk_customers.csv",
            mime="text/csv",
        )
# ============================================================
# PAGE 4
# RETENTION INTELLIGENCE
# ============================================================

elif page == "🧠 Retention Intelligence":

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------
    #
    # This page represents the business decision-support layer.
    #
    # The ML model answers:
    #   "How likely is the customer to churn?"
    #
    # This layer adds:
    #   Behaviour + Value + Business Priority + Candidate Action
    #
    # The objective is therefore to move from prediction toward
    # actionable retention prioritization.
    # --------------------------------------------------------

    display_page_title(
        "🧠 Retention Intelligence",
        "Combine churn risk, behavioural deterioration and customer "
        "value to support retention prioritization."
    )

    # --------------------------------------------------------
    # RISK-TIER INTERPRETATION
    # --------------------------------------------------------
    #
    # Risk tiers are relative quartile-based segments created
    # from predicted churn probabilities.
    #
    # Therefore:
    #   Very High ≠ churn probability above 50%
    #
    # It means the customer belongs to the highest-risk quartile
    # within the scored customer population.
    # --------------------------------------------------------

    st.info(
        "Risk tiers are relative segments of the scored customer "
        "population. 'Very High' represents the top churn-risk "
        "quartile; it does not mean that every customer has a "
        "churn probability above 50%."
    )

    # --------------------------------------------------------
    # PORTFOLIO OVERVIEW
    # --------------------------------------------------------
    #
    # Provides a quick view of the retention population.
    # --------------------------------------------------------

    st.subheader(
        "Retention Portfolio Overview"
    )

    total_customers = len(df)

    very_high_count = int(
        (
            df["risk_tier"].astype(str)
            == "Very High"
        ).sum()
    )

    priority_1_count = int(
        (
            df["retention_priority"].astype(str)
            == "Priority 1"
        ).sum()
    )

    priority_2_count = int(
        (
            df["retention_priority"].astype(str)
            == "Priority 2"
        ).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        display_metric_card(
            "Customers",
            f"{total_customers:,}"
        )

    with c2:

        display_metric_card(
            "Very High Risk",
            f"{very_high_count:,}"
        )

    with c3:

        display_metric_card(
            "Priority 1",
            f"{priority_1_count:,}"
        )

    with c4:

        display_metric_card(
            "Priority 2",
            f"{priority_2_count:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # BEHAVIOURAL STATES
    # --------------------------------------------------------
    #
    # Behavioural states summarize observable deterioration
    # patterns across the available monthly observations.
    #
    # They are descriptive analytical categories and should not
    # be interpreted as causal explanations of churn.
    # --------------------------------------------------------

    if "behavioral_state" in df.columns:

        st.subheader(
            "Behavioural State Distribution"
        )

        behavioral_counts = (
            df["behavioral_state"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        behavioral_counts.columns = [
            "Behavioural State",
            "Customers"
        ]

        fig_behavior = px.bar(
            behavioral_counts,
            x="Customers",
            y="Behavioural State",
            orientation="h",
            title="Customer Distribution by Behavioural State"
        )

        fig_behavior.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_behavior,
            width="stretch"
        )

        st.caption(
            "Behavioural states summarize observed customer "
            "deterioration patterns across the available monthly period."
        )

    st.divider()

    # --------------------------------------------------------
    # DETERIORATION SIGNALS
    # --------------------------------------------------------
    #
    # These metrics show the number of customers exhibiting
    # deterioration across multiple behavioural dimensions.
    #
    # A threshold of >= 2 is used here to focus the dashboard
    # on customers showing deterioration across more than one
    # behavioural dimension rather than a single isolated signal.
    # --------------------------------------------------------

    st.subheader(
        "Customer Deterioration Signals"
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        if "recent_deterioration_count" in df.columns:

            recent_values = pd.to_numeric(
                df["recent_deterioration_count"],
                errors="coerce"
            )

            count = int(
                (recent_values >= 2).sum()
            )

        else:

            count = 0

        display_metric_card(
            "Recent Deterioration",
            f"{count:,}"
        )

    with d2:

        if "persistent_deterioration_count" in df.columns:

            persistent_values = pd.to_numeric(
                df["persistent_deterioration_count"],
                errors="coerce"
            )

            count = int(
                (persistent_values >= 2).sum()
            )

        else:

            count = 0

        display_metric_card(
            "Persistent Deterioration",
            f"{count:,}"
        )

    with d3:

        if "coordinated_deterioration_count" in df.columns:

            coordinated_values = pd.to_numeric(
                df["coordinated_deterioration_count"],
                errors="coerce"
            )

            count = int(
                (coordinated_values >= 4).sum()
            )

        else:

            count = 0

        display_metric_card(
            "Broad Deterioration",
            f"{count:,}"
        )

    st.caption(
        "Deterioration counts are based on the project's predefined "
        "behavioural definitions."
    )

    st.divider()

    # --------------------------------------------------------
    # RETENTION PRIORITY
    # --------------------------------------------------------
    #
    # Retention priority is a business-rule layer.
    #
    # It is NOT simply the ranking of predicted churn probability.
    #
    # The project's priority logic considers:
    #   - churn risk
    #   - customer value
    #   - behavioural evidence
    # --------------------------------------------------------

    if "retention_priority" in df.columns:

        st.subheader(
            "Retention Priority Distribution"
        )

        priority_counts = (
            df["retention_priority"]
            .astype(str)
            .value_counts()
            .reindex(
                [
                    "Priority 1",
                    "Priority 2",
                    "Priority 3"
                ],
                fill_value=0
            )
            .reset_index()
        )

        priority_counts.columns = [
            "Retention Priority",
            "Customers"
        ]

        fig_priority = px.bar(
            priority_counts,
            x="Retention Priority",
            y="Customers",
            title="Customer Distribution by Retention Priority"
        )

        fig_priority.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig_priority,
            width="stretch"
        )

        st.caption(
            "Retention priority combines churn risk, customer value "
            "and behavioural evidence. It is separate from the ML "
            "churn-risk ranking."
        )

    st.divider()

    # --------------------------------------------------------
    # RISK × VALUE
    # --------------------------------------------------------
    #
    # This view connects predicted churn risk with relative
    # customer value.
    #
    # It helps explain why the customer with the highest churn
    # probability is not necessarily the customer with the
    # highest retention priority.
    #
    # Value tier is a relative value indicator. It should not be
    # interpreted as CLV, profit or customer margin.
    # --------------------------------------------------------

    if (
        "risk_tier" in df.columns
        and "value_tier" in df.columns
    ):

        st.subheader(
            "Risk × Value Retention View"
        )

        risk_value_df = (
            df.groupby(
                [
                    "risk_tier",
                    "value_tier"
                ],
                observed=True
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        risk_value_pivot = risk_value_df.pivot(
            index="risk_tier",
            columns="value_tier",
            values="Customers"
        ).fillna(0)

        risk_order = [
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]

        value_order = [
            "Lower",
            "Moderate",
            "Higher",
            "Highest"
        ]

        risk_value_pivot = risk_value_pivot.reindex(
            index=risk_order,
            columns=value_order,
            fill_value=0
        )

        st.dataframe(
            risk_value_pivot,
            width="stretch"
        )

        st.caption(
            "Risk and value are separate dimensions. Retention "
            "priority uses these dimensions together with behavioural "
            "evidence."
        )

    st.divider()

    # --------------------------------------------------------
    # CANDIDATE RETENTION ACTIONS
    # --------------------------------------------------------
    #
    # Candidate actions identify the type of retention review that
    # may be appropriate based on the available evidence.
    #
    # These are NOT proven causal treatments.
    # --------------------------------------------------------

    if "candidate_action" in df.columns:

        st.subheader(
            "Candidate Retention Actions"
        )

        action_counts = (
            df["candidate_action"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        action_counts.columns = [
            "Candidate Action",
            "Customers"
        ]

        st.dataframe(
            action_counts,
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Candidate actions represent retention-review categories "
            "generated by the intelligence layer. They do not establish "
            "that a particular action will prevent churn."
        )

    st.divider()

    # --------------------------------------------------------
    # DECISION INTERPRETATION
    # --------------------------------------------------------
    #
    # This short explanation makes the complete business flow
    # explicit without introducing another model.
    # --------------------------------------------------------

    st.subheader(
        "How Retention Intelligence Works"
    )

    st.markdown(
        """
        **1. Predict** → Estimate the customer's churn probability.

        **2. Understand** → Identify observable behavioural deterioration.

        **3. Assess Value** → Determine the customer's relative value tier.

        **4. Prioritize** → Combine risk, behaviour and value through
        the retention rules.

        **5. Route** → Assign a candidate retention-review category.
        """
    )

    st.info(
        "The system separates prediction from decision-making: "
        "the ML model estimates churn risk, while the retention-intelligence "
        "layer converts risk, behaviour and value into business-oriented "
        "retention prioritization."
    )

# ============================================================
# PAGE 5
# AI RETENTION AGENT
# ============================================================

elif page == "🤖 AI Retention Agent":

    # --------------------------------------------------------
    # PAGE HEADER
    # --------------------------------------------------------
    #
    # The AI agent acts as the explanation and decision-support
    # layer over the outputs generated by the analytical pipeline.
    #
    # ML Model
    #     ↓
    # Retention Intelligence
    #     ↓
    # AI Agent
    #     ↓
    # Business Explanation
    # --------------------------------------------------------

    display_page_title(
        "🤖 AI Retention Agent",
        "Ask questions about churn risk, customer behaviour and "
        "retention priorities."
    )

    # --------------------------------------------------------
    # AGENT AVAILABILITY
    # --------------------------------------------------------
    #
    # The dashboard remains usable even if the optional AI agent
    # dependencies or configuration are unavailable.
    # --------------------------------------------------------

    if not AGENT_AVAILABLE:

        st.warning(
            "The AI agent could not be initialized."
        )

        with st.expander(
            "Technical information"
        ):

            st.code(
                AGENT_ERROR or "Unknown error"
            )

    else:

        # ----------------------------------------------------
        # AGENT STATUS
        # ----------------------------------------------------

        st.success(
            "AI Retention Agent is ready."
        )

        st.write(
            "The agent explains ML churn-risk predictions using "
            "behavioural evidence, customer value and retention "
            "intelligence."
        )

        # ----------------------------------------------------
        # AGENT ROLE
        # ----------------------------------------------------
        #
        # The LLM does not replace the analytical pipeline.
        #
        # It consumes the prepared retention-intelligence outputs
        # and converts them into business-oriented explanations.
        # ----------------------------------------------------

        st.info(
            "The AI agent explains retention-intelligence outputs "
            "using the customer data and business rules. It does not "
            "retrain the ML model or independently change churn risk "
            "or retention priority."
        )

        # ----------------------------------------------------
        # SAMPLE QUESTIONS
        # ----------------------------------------------------
        #
        # The first two questions intentionally represent different
        # business decisions:
        #
        # 1. Highest churn risk
        #    → ML probability ranking
        #
        # 2. Highest retention priority
        #    → Risk + value + behavioural business rules
        #
        # This distinction is central to the project's architecture.
        # ----------------------------------------------------

        st.subheader(
            "Example Questions"
        )

        example_questions = [
            "Which customers have the highest predicted churn risk?",
            "Which customers should we prioritize for retention?",
            "What is the retention summary?",
            "Analyze customer 2.",
            "Why is customer 2 Priority 2?",
            "What behavioural signals indicate customer deterioration?",
        ]

        selected_question = st.selectbox(
            "Choose an example question",
            ["Custom question"] + example_questions,
        )

        # ----------------------------------------------------
        # QUESTION INPUT
        # ----------------------------------------------------

        if selected_question == "Custom question":

            question = st.text_area(
                "Ask the AI Retention Agent",
                placeholder=(
                    "Example: Which high-risk customers "
                    "should the telecom company contact first?"
                ),
                height=120,
            )

        else:

            question = selected_question

            st.info(
                f"Selected question: {question}"
            )

        # ----------------------------------------------------
        # SOURCE-OF-TRUTH NOTE
        # ----------------------------------------------------
        #
        # This reinforces that the agent is an interpretation
        # layer rather than an independent prediction engine.
        # ----------------------------------------------------

        st.caption(
            "Source of truth: the retention-intelligence dataset "
            "generated by the ML and behavioural pipeline. The agent "
            "explains these outputs but does not replace the underlying "
            "analytical logic."
        )

        # ----------------------------------------------------
        # ASK AGENT
        # ----------------------------------------------------

        if st.button(
            "🤖 Ask Retention Agent",
            width="stretch"
        ):

            # ------------------------------------------------
            # INPUT VALIDATION
            # ------------------------------------------------

            if not question or not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                # ------------------------------------------------
                # AGENT EXECUTION
                # ------------------------------------------------
                #
                # The question is passed to the existing agent
                # implementation. The dashboard does not perform
                # separate ML calculations here.
                # ------------------------------------------------

                with st.spinner(
                    "Analyzing customer intelligence..."
                ):

                    try:

                        response = ask_agent(
                            question.strip()
                        )

                        # ----------------------------------------
                        # AGENT RESPONSE
                        # ----------------------------------------

                        st.subheader(
                            "🧠 Agent Response"
                        )

                        st.write(
                            response
                        )

                    except Exception as e:

                        # ----------------------------------------
                        # ERROR HANDLING
                        # ----------------------------------------

                        st.error(
                            "The AI agent encountered an error."
                        )

                        st.code(
                            str(e)
                        )
# ============================================================
# PAGE 5
# AI RETENTION AGENT
# ============================================================

elif page == "🤖 AI Retention Agent":

    display_page_title(
        "🤖 AI Retention Agent",
        "Ask questions about customer churn, risk and retention actions."
    )

    if not AGENT_AVAILABLE:

        st.warning(
            "The AI agent could not be initialized."
        )

        with st.expander("Technical information"):

            st.code(
                AGENT_ERROR or "Unknown error"
            )

    else:

        st.success(
            "AI Retention Agent is ready."
        )

        st.write(
            "The agent combines ML churn predictions, "
            "behavioral intelligence and retention evidence."
        )

        # ----------------------------------------------------
        # SAMPLE QUESTIONS
        # ----------------------------------------------------

        st.subheader("Example Questions")

        example_questions = [
            "Which customers are at very high churn risk?",
            "Show me the highest priority customers.",
            "What is the retention summary?",
            "Analyze customer 2.",
            "What behavioral signals indicate churn?",
        ]

        selected_question = st.selectbox(
            "Choose an example question",
            ["Custom question"] + example_questions,
        )

        if selected_question == "Custom question":

            question = st.text_area(
                "Ask the AI Retention Agent",
                placeholder=(
                    "Example: Which high-risk customers "
                    "should the telecom company contact first?"
                ),
                height=120,
            )

        else:

            question = selected_question

            st.info(
                f"Selected question: {question}"
            )

        # ----------------------------------------------------
        # ASK AGENT
        # ----------------------------------------------------

        if st.button(
            "🤖 Ask Retention Agent",
            width="stretch"
        ):

            if not question or not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Analyzing customer intelligence..."
                ):

                    try:

                        response = ask_agent(
                            question.strip()
                        )

                        st.subheader(
                            "🧠 Agent Response"
                        )

                        st.write(
                            response
                        )

                    except Exception as e:

                        st.error(
                            "The AI agent encountered an error."
                        )

                        st.code(
                            str(e)
                        )
                        
# ============================================================
# PAGE 6
# RETENTION OPERATIONS
# ============================================================

elif page == "📈 Retention Operations":
    display_page_title(
        "📈 Retention Operations",
        "Turn retention intelligence into a prioritized action queue for the retention team."
    )
    # --------------------------------------------------------
    # PURPOSE
    # --------------------------------------------------------
    # This page represents the operational layer of the system.
    #
    # Retention Intelligence determines:
    #   Risk + Behaviour + Value → Priority + Candidate Action
    #
    # This page focuses on what the retention team should review
    # and how the identified workload is distributed.
    #
    # It intentionally does not repeat the analytical views from
    # the Retention Intelligence page.
    # --------------------------------------------------------

    st.info(
        "This page converts the retention-intelligence output into an "
        "operational review queue. It helps the retention team understand "
        "which customers require attention and what issue should be reviewed."
    )

    # ========================================================
    # 1. CUSTOMERS REQUIRING RETENTION FOCUS
    # ========================================================

    priority_1_df = df[
        df["retention_priority"] == "Priority 1"
    ].copy()

    priority_2_df = df[
        df["retention_priority"] == "Priority 2"
    ].copy()

    priority_focus_df = df[
        df["retention_priority"].isin(
            ["Priority 1", "Priority 2"]
        )
    ].copy()

    priority_1_count = len(priority_1_df)
    priority_2_count = len(priority_2_df)
    priority_focus_count = len(priority_focus_df)

    st.subheader("🎯 Customers Requiring Retention Focus")

    m1, m2, m3 = st.columns(3)

    with m1:
        display_metric_card(
            "Customers Requiring Attention",
            f"{priority_focus_count:,}"
        )

    with m2:
        display_metric_card(
            "Priority 1",
            f"{priority_1_count:,}"
        )

    with m3:
        display_metric_card(
            "Priority 2",
            f"{priority_2_count:,}"
        )

    st.caption(
        "Priority 1 and Priority 2 represent the current retention-focus "
        "population. Priority is determined by the project's retention rules, "
        "not by churn probability alone."
    )

    st.divider()

    # ========================================================
    # 2. RETENTION ACTION QUEUE
    # ========================================================

    st.subheader("🚨 Retention Action Queue")

    st.write(
        "Use the filters below to identify customers requiring a specific "
        "retention review."
    )

    f1, f2, f3 = st.columns(3)

    with f1:
        selected_priority = st.selectbox(
            "Retention Priority",
            ["All", "Priority 1", "Priority 2"]
        )

    with f2:
        action_options = sorted(
            priority_focus_df["candidate_action"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_action = st.selectbox(
            "Candidate Action",
            ["All"] + action_options
        )

    with f3:
        selected_value = st.selectbox(
            "Customer Value",
            ["All", "Lower", "Moderate", "Higher", "Highest"]
        )

    action_queue_df = priority_focus_df.copy()

    if selected_priority != "All":
        action_queue_df = action_queue_df[
            action_queue_df["retention_priority"]
            == selected_priority
        ]

    if selected_action != "All":
        action_queue_df = action_queue_df[
            action_queue_df["candidate_action"]
            == selected_action
        ]

    if selected_value != "All":
        action_queue_df = action_queue_df[
            action_queue_df["value_tier"]
            == selected_value
        ]

    # Highest-risk customers appear first within the selected queue.
    action_queue_df = action_queue_df.sort_values(
        "predicted_churn_risk",
        ascending=False
    )

    st.write(
        f"**{len(action_queue_df):,} customers** match the current "
        "retention-review filters."
    )

    queue_columns = [
        "id",
        "predicted_churn_risk",
        "behavioral_state",
        "value_tier",
        "retention_priority",
        "candidate_action",
    ]

    queue_columns = [
        col
        for col in queue_columns
        if col in action_queue_df.columns
    ]

    queue_display_df = action_queue_df[
        queue_columns
    ].copy()

    if "predicted_churn_risk" in queue_display_df.columns:
        queue_display_df["predicted_churn_risk"] = (
            queue_display_df["predicted_churn_risk"]
            .apply(format_probability)
        )

    st.dataframe(
        queue_display_df,
        width="stretch",
        hide_index=True,
    )

    # Allow the operational queue to be exported.
    if not action_queue_df.empty:

        action_queue_csv = action_queue_df.to_csv(
            index=False
        )

        st.download_button(
            "⬇️ Download Retention Action Queue",
            data=action_queue_csv,
            file_name="retention_action_queue.csv",
            mime="text/csv",
        )

    st.divider()

    # ========================================================
    # 3. RETENTION WORKLOAD
    # ========================================================

    st.subheader("📊 Retention Workload")

    st.write(
        "Shows the types of retention reviews currently represented "
        "within the Priority 1 + Priority 2 population."
    )

    workload_df = (
        priority_focus_df["candidate_action"]
        .value_counts()
        .rename_axis("candidate_action")
        .reset_index(name="customers")
    )

    if not workload_df.empty:

        st.bar_chart(
            workload_df.set_index("candidate_action")["customers"],
            horizontal=True,
        )

    st.caption(
        "Candidate actions are review categories generated by the "
        "retention-intelligence layer. They are not proven causal treatments."
    )

    st.divider()

    # ========================================================
    # 4. PRIORITY CUSTOMER ARPU EXPOSURE
    # ========================================================

    st.subheader("💰 Priority Customer ARPU Exposure")

    p1_arpu = priority_1_df["arpu_8"].sum()
    p2_arpu = priority_2_df["arpu_8"].sum()

    a1, a2 = st.columns(2)

    with a1:
        display_metric_card(
            "Priority 1 Month-8 ARPU Exposure",
            f"₹{p1_arpu:,.0f}"
        )

    with a2:
        display_metric_card(
            "Priority 2 Month-8 ARPU Exposure",
            f"₹{p2_arpu:,.0f}"
        )

    st.caption(
        "ARPU exposure represents observed Month-8 ARPU within the "
        "priority population. It is not an estimate of revenue loss, "
        "profit, customer lifetime value, or revenue saved."
    )

    st.divider()

    # ========================================================
    # 5. RETENTION OPERATIONS PLAYBOOK
    # ========================================================

    st.subheader("🧭 Retention Operations Playbook")

    st.markdown(
        """
### From prediction to action

**1. Identify risk**
- Use the ML-predicted churn probability to identify customers with elevated risk.

**2. Check retention priority**
- Priority 1 and Priority 2 customers form the current operational focus.

**3. Understand the behavioural evidence**
- Review recent, persistent and coordinated deterioration patterns.

**4. Review the candidate action**
- Use the candidate action as the recommended *review category*.

**5. Human retention review**
- The retention team investigates the customer context before deciding on an intervention.

### Priority interpretation

| Priority | Operational meaning |
|---|---|
| **Priority 1** | Very-high-risk customers with higher or highest customer value |
| **Priority 2** | Very-high-risk customers with lower/moderate value but strong behavioural deterioration |
| **Priority 3** | Outside the current focused retention queue |
"""
    )

    st.divider()

    # ========================================================
    # 6. OPERATIONAL READINESS
    # ========================================================

    st.subheader("⚙️ Operational Readiness")

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(
            """
### Current Prototype

**Data mode:** Batch-based

The current application operates on the prepared
retention-intelligence dataset.

**Decision flow:**

`Prepared Data → Churn Risk → Retention Intelligence → Action Queue`
"""
        )

    with r2:

        st.markdown(
            """
### Production Direction

The same architecture can be connected to:

- scheduled customer-data refreshes,
- CRM/customer-service systems,
- event-driven feature updates,
- periodic model scoring,
- refreshed retention queues.

`Customer Events → Feature Refresh → Model → Retention Queue`
"""
        )

    st.warning(
        "The current prototype is not a real-time intervention system. "
        "Production deployment would require scheduled or event-driven "
        "data and model refresh mechanisms."
    )

    # ========================================================
    # FINAL BUSINESS MESSAGE
    # ========================================================

    st.success(
        "Business flow: Predict churn → understand deterioration → "
        "prioritize customers → route the retention review → support human action."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Telecom Customer Churn Intelligence • "
    "ML Prediction + Behavioral Intelligence + AI Retention Support"
)