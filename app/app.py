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

    page = st.radio(
        "Go to",
        [
            "📊 Executive Dashboard",
            "👤 Customer Analysis",
            "🚨 High-Risk Customers",
            "🧠 Retention Intelligence",
            "🤖 AI Retention Agent",
            "➕ Add New Customer",
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

    retention_priority_count = total_customers

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
            "Retention Priorities",
            f"{retention_priority_count:,}"
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

    display_page_title(
        "🚨 High-Risk Customers",
        "Customers requiring immediate retention attention."
    )

    high_risk_df = df[
        df["risk_tier"]
        .astype(str)
        .str.lower()
        .isin(["high", "very high"])
    ].copy()

    if high_risk_df.empty:

        st.success(
            "No high-risk customers found."
        )

    else:

        st.write(
            f"**{len(high_risk_df):,} high-risk customers identified.**"
        )

        # ----------------------------------------------------
        # SORT BY CHURN RISK
        # ----------------------------------------------------

        high_risk_df = high_risk_df.sort_values(
            "predicted_churn_risk",
            ascending=False
        )

        # ----------------------------------------------------
        # TOP CUSTOMER METRICS
        # ----------------------------------------------------

        top_risk = high_risk_df.iloc[0]

        m1, m2, m3 = st.columns(3)

        with m1:
            display_metric_card(
                "Highest Risk Customer",
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
        # TABLE
        # ----------------------------------------------------

        display_columns = [
            "id",
            "predicted_churn_risk",
            "risk_tier",
            "behavioral_state",
            "value_tier",
            "retention_priority",
            "candidate_action",
        ]

        display_columns = [
            col
            for col in display_columns
            if col in high_risk_df.columns
        ]

        table_df = high_risk_df[
            display_columns
        ].copy()

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

    display_page_title(
        "🧠 Retention Intelligence",
        "Behavioral evidence used to support retention decisions."
    )

    # --------------------------------------------------------
    # BEHAVIORAL STATES
    # --------------------------------------------------------

    if "behavioral_state" in df.columns:

        st.subheader("Behavioral State Distribution")

        behavioral_counts = (
            df["behavioral_state"]
            .astype(str)
            .value_counts()
            .reset_index()
        )

        behavioral_counts.columns = [
            "Behavioral State",
            "Customers"
        ]

        fig_behavior = px.bar(
            behavioral_counts,
            x="Behavioral State",
            y="Customers",
            title="Customer Behavioral States",
        )

        fig_behavior.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig_behavior,
            width="stretch"
        )

    # --------------------------------------------------------
    # DETERIORATION
    # --------------------------------------------------------

    st.subheader("Customer Deterioration Signals")

    d1, d2, d3 = st.columns(3)

    with d1:

        if "recent_deterioration_count" in df.columns:

            count = int(
                (
                    pd.to_numeric(
                        df["recent_deterioration_count"],
                        errors="coerce"
                    )
                    > 0
                ).sum()
            )

        else:
            count = 0

        display_metric_card(
            "Recent Deterioration",
            f"{count:,}"
        )

    with d2:

        if "persistent_deterioration_count" in df.columns:

            count = int(
                (
                    pd.to_numeric(
                        df["persistent_deterioration_count"],
                        errors="coerce"
                    )
                    > 0
                ).sum()
            )

        else:
            count = 0

        display_metric_card(
            "Persistent Deterioration",
            f"{count:,}"
        )

    with d3:

        if "coordinated_deterioration_count" in df.columns:

            count = int(
                (
                    pd.to_numeric(
                        df["coordinated_deterioration_count"],
                        errors="coerce"
                    )
                    > 0
                ).sum()
            )

        else:
            count = 0

        display_metric_card(
            "Coordinated Deterioration",
            f"{count:,}"
        )

    st.divider()

    # --------------------------------------------------------
    # RETENTION PRIORITY
    # --------------------------------------------------------

    if "retention_priority" in df.columns:

        st.subheader("Retention Priority Distribution")

        priority_counts = (
            df["retention_priority"]
            .astype(str)
            .value_counts()
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
            title="Retention Priority",
        )

        fig_priority.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(
            fig_priority,
            width="stretch"
        )

    # --------------------------------------------------------
    # ACTION RECOMMENDATIONS
    # --------------------------------------------------------

    if "candidate_action" in df.columns:

        st.subheader("Candidate Retention Actions")

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
            hide_index=True,
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
# ADD NEW CUSTOMER
# ============================================================

elif page == "➕ Add New Customer":

    display_page_title(
        "➕ Add New Customer",
        "Enter customer information for future churn assessment."
    )

    st.info(
        "This page collects customer information. "
        "A genuine ML prediction requires the complete "
        "189-feature prepared model vector used during training."
    )

    st.subheader("Customer Information")

    c1, c2 = st.columns(2)

    with c1:

        new_customer_id = st.text_input(
            "Customer ID",
            placeholder="Example: NEW001"
        )

        arpu = st.number_input(
            "Average Revenue Per User (ARPU)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

        recharge_amount = st.number_input(
            "Recharge Amount",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

    with c2:

        recharge_count = st.number_input(
            "Recharge Count",
            min_value=0,
            value=0,
            step=1,
        )

        outgoing_usage = st.number_input(
            "Outgoing Usage",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

        incoming_usage = st.number_input(
            "Incoming Usage",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

    st.divider()

    st.subheader("Customer Summary")

    summary_df = pd.DataFrame(
        {
            "Field": [
                "Customer ID",
                "ARPU",
                "Recharge Amount",
                "Recharge Count",
                "Outgoing Usage",
                "Incoming Usage",
            ],
            "Value": [
                new_customer_id or "-",
                f"{arpu:,.2f}",
                f"{recharge_amount:,.2f}",
                str(recharge_count),
                f"{outgoing_usage:,.2f}",
                f"{incoming_usage:,.2f}",
            ],
        }
    )

    st.dataframe(
        summary_df,
        width="stretch",
        hide_index=True,
    )

    if st.button(
        "➕ Add Customer",
        width="stretch"
    ):

        if not new_customer_id.strip():

            st.warning(
                "Please enter a Customer ID."
            )

        else:

            st.success(
                f"Customer {new_customer_id} information captured."
            )

            st.warning(
                "⚠️ ML churn prediction was not generated because "
                "the model requires the exact 189-feature prepared "
                "input vector used during training."
            )

            st.info(
                "To enable genuine prediction for new customers, "
                "the original model preprocessing pipeline must be "
                "connected to this form."
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Telecom Customer Churn Intelligence • "
    "ML Prediction + Behavioral Intelligence + AI Retention Support"
)