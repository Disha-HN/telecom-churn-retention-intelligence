# 📡 Telecom Churn Retention Intelligence

> **Predict Churn • Understand Behavior • Prioritize Customers • Recommend Action**

An AI-assisted telecom customer retention system that combines **machine learning, behavioral intelligence, customer value analysis, retention prioritization, and a natural-language AI agent** to help identify customers who are most likely to churn and determine who should be prioritized for retention.

---

## 🚀 Overview

Customer churn prediction is useful, but a churn probability alone does not answer the most important business questions:

* **Which customers are most likely to churn?**
* **What behavioral changes indicate that they are becoming disengaged?**
* **Which high-risk customers are also valuable to the business?**
* **Who should the retention team prioritize first?**
* **What retention action should be considered?**

This project addresses these questions by combining a machine-learning-based churn prediction model with **behavioral intelligence, customer value analysis, retention prioritization, and an AI-powered Retention Agent**.

The system transforms raw telecom customer data into actionable retention intelligence:

```text
Customer Data
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
Churn Prediction
      ↓
Behavior + Value Analysis
      ↓
Retention Priority
      ↓
Candidate Action + Rationale
      ↓
Dashboard + AI Retention Agent
```

---

# 🎯 Problem Statement

Traditional churn prediction systems generally focus on estimating:

> **"Will this customer churn?"**

However, a business also needs to understand:

> **"Why is this customer at risk?"**
> **"Is the customer's behavior deteriorating?"**
> **"How valuable is this customer?"**
> **"Which customers should be contacted first?"**

A high churn probability does not necessarily mean that every customer should receive the same retention treatment.

Therefore, this project combines:

```text
Churn Risk
     +
Customer Behaviour
     +
Customer Value
     ↓
Retention Intelligence
```

The goal is to move from **prediction-only** to **decision-oriented customer retention**.

---

# 🏗️ System Architecture

The complete system follows a layered pipeline that converts raw telecom data into churn predictions, behavioral insights, customer value analysis, retention priorities, and actionable recommendations.

```text
                         TELECOM CUSTOMER DATA
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   DATA UNDERSTANDING           │
                    │                                │
                    │ • Data profiling               │
                    │ • Missingness analysis         │
                    │ • Churn distribution           │
                    │ • Temporal analysis            │
                    │ • Data-quality checks          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   DATA PREPARATION             │
                    │                                │
                    │ • Metadata handling            │
                    │ • Missing-value logic          │
                    │ • Activity indicators          │
                    │ • Validation                   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  FEATURE ENGINEERING           │
                    │                                │
                    │ • Monthly behaviour            │
                    │ • Temporal changes             │
                    │ • Persistent decline           │
                    │ • Recent deterioration         │
                    │ • Coordinated decline          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    CHURN MODELING              │
                    │                                │
                    │ RF / XGBoost / LightGBM        │
                    │          │                     │
                    │          ▼                     │
                    │    Model evaluation            │
                    │          │                     │
                    │          ▼                     │
                    │     Final LightGBM             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     CHURN RISK SCORE           │
                    │                                │
                    │ predicted_churn_risk           │
                    │          │                     │
                    │          ▼                     │
                    │ Low / Moderate / High          │
                    │ Very High                      │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
 ┌─────────────────────────┐          ┌──────────────────────┐
 │ BEHAVIOURAL INTELLIGENC        │          │   CUSTOMER VALUE        │
 │                                │          │                         │
 │ • Recent deterioration         │          │ • ARPU                  │
 │ • Persistent decline           │          │ • Value tier            │
 │ • Coordinated decline          │          │ • Revenue exposure      │
 │ • Behavioural state            │          │                         │
 └────────────┬────────────┘          └────────────┬─────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ RETENTION INTELLIGENCE         │
                    │                                │
                    │ Risk + Behaviour +             │
                    │ Value                          │
                    │          │                     │
                    │          ▼                     │
                    │ Retention Priority             │
                    │          │                     │
                    │          ▼                     │
                    │ Candidate Action               │
                    │          │                     │
                    │          ▼                     │
                    │ Decision Rationale              │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐      ┌────────────────────┐
          │ STREAMLIT             │      │ AI RETENTION AGENT     │
          │ DASHBOARD             │      │                        │
          │                       │      │ Customer Context       │
          │ • Executive           │      │        ↓               │
          │ • Customer            │      │ Retention Tools        │
          │ • High Risk           │      │        ↓               │
          │ • Intelligence        │      │ LangChain Agent        │
          └──────────────────┘      │        ↓               │
                                          │ Groq LLM               │
                                          │        ↓               │
                                          │ Natural Language       │
                                          │ Retention Insights     │
                                         └──────────────────────┘
```

---

# 🔥 Key Features

## 🎯 ML-Based Churn Prediction

The system uses machine learning to estimate the probability that a customer will churn.

Multiple models were evaluated:

* Random Forest
* XGBoost
* LightGBM

The final system uses **LightGBM** for churn prediction.

---

## 📊 Behavioral Intelligence

The system does not rely only on the churn score.

It analyzes customer behavior across multiple months to identify:

* Recent deterioration
* Persistent decline
* Coordinated decline
* Changes in ARPU
* Changes in recharge behavior
* Changes in outgoing usage
* Changes in incoming usage
* Overall behavioral state
* Evidence supporting the behavioral assessment

This helps answer:

> **"What is happening to the customer?"**

rather than only:

> **"Will the customer churn?"**

---

## 💰 Customer Value Analysis

Customer value is incorporated into the retention decision using indicators such as:

* ARPU
* Value tier
* Revenue exposure

This allows the system to distinguish between different types of high-risk customers.

For example:

```text
High Churn Risk
      +
High Customer Value
      +
Behavioral Decline
      ↓
High Retention Priority
```

---

## 🎯 Retention Prioritization

The system combines:

```text
Churn Risk
     +
Behaviour
     +
Customer Value
     ↓
Retention Priority
     ↓
Candidate Action
     ↓
Decision Rationale
```

This converts a raw ML prediction into a more actionable business recommendation.

---

# 🧠 Machine Learning

## Feature Engineering

The final model uses **189 engineered features** derived from telecom customer behavior.

Feature groups include:

* Monthly customer activity
* ARPU
* Recharge amount
* Recharge count
* Outgoing usage
* Incoming usage
* Month-to-month changes
* Persistent decline indicators
* Recent deterioration indicators
* Coordinated deterioration indicators
* Activity indicators

### Temporal Analysis

Customer behavior is compared across multiple months:

```text
June → July → August
```

The system calculates:

```text
June → July change
July → August change
June → August change
```

This allows the model and intelligence layer to identify both short-term and sustained behavioral changes.

---

## 🤖 Models Evaluated

| Model         |    ROC-AUC |     PR-AUC |
| ------------- | ---------: | ---------: |
| Random Forest |     0.9349 |     0.7272 |
| XGBoost       |     0.9406 |     0.7388 |
| **LightGBM**  | **0.9426** | **0.7453** |

### Final LightGBM

The final LightGBM model achieved:

| Metric    |      Score |
| --------- | ---------: |
| ROC-AUC   | **0.9426** |
| PR-AUC    | **0.7453** |
| Precision | **0.7504** |
| Recall    | **0.6157** |
| F1 Score  | **0.6764** |

The final trained model is stored as:

```text
model/final_lightgbm_churn_model.joblib
```

---

# 📈 Churn Risk Segmentation

The predicted churn probabilities are used to divide customers into four risk tiers:

```text
┌──────────────┐
│     Low          │
└──────────────┘

┌──────────────┐
│   Moderate       │
└──────────────┘

┌──────────────┐
│     High         │
└──────────────┘

┌──────────────┐
│   Very High      │
└──────────────┘
```

This allows the business team to quickly identify customers requiring greater attention.

---

# 🔎 Behavioral Intelligence

Behavioral intelligence is generated from customer-level temporal patterns.

### Recent Deterioration

Identifies whether important customer activity has declined recently.

Examples:

```text
ARPU ↓
Recharge Amount ↓
Recharge Count ↓
Outgoing Usage ↓
Incoming Usage ↓
```

### Persistent Decline

Identifies behavioral signals that show decline across multiple periods rather than only one recent change.

### Coordinated Decline

Measures how many important behavioral metrics are declining together.

For example:

```text
ARPU              ↓
Recharge Count    ↓
Outgoing Usage    ↓
Incoming Usage    ↓
                  ↓
      Coordinated Deterioration
```

This provides stronger behavioral evidence than looking at a single metric.

---

# 🎯 Retention Intelligence

The retention intelligence layer combines three major dimensions:

```text
        ┌───────────────┐
        │   Churn Risk      │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │   Behaviour       │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │ Customer Value     │
        └───────┬───────┘
                │
                ▼
      ┌───────────────────┐
      │ Retention Priority      │
      └─────────┬─────────┘
                │
                ▼
       Candidate Action
                │
                ▼
       Decision Rationale
```

The resulting retention intelligence contains information such as:

* Risk tier
* Behavioral state
* Behavioral evidence
* Value tier
* Retention priority
* Candidate action
* Decision rationale

---

# 🤖 AI Retention Agent

The AI Retention Agent provides a natural-language interface to the retention intelligence layer.

### Architecture

```text
User Question
      │
      ▼
Customer / Business Context
      │
      ▼
Retention Intelligence Tools
      │
      ▼
LangChain Agent
      │
      ▼
Groq LLM
      │
      ▼
Natural Language Response
```

The agent uses controlled tools to retrieve information from the processed retention-intelligence dataset.

### Available Intelligence Tools

The agent can access:

* Customer profiles
* High-risk customers
* Priority customers
* Customer behavioral analysis
* Retention summaries

### Example Questions

```text
Which customers are at very high churn risk?
```

```text
Show me the highest-priority customers.
```

```text
What is the retention status of customer 10001?
```

```text
Why is this customer considered high risk?
```

```text
What are the major behavioral patterns?
```

The agent is designed to ground responses in the available customer intelligence rather than inventing customer-level information.

---

# ⚡ Intelligent Query Routing

Simple questions do not always require a full LLM reasoning cycle.

The system therefore uses deterministic routing for straightforward queries such as:

```text
Which customers are at very high churn risk?
```

These requests can directly retrieve the required information from the local intelligence tools.

More complex questions can be passed to the:

```text
LangChain → Groq LLM
```

pipeline.

This improves responsiveness while keeping the AI layer useful for natural-language analysis.

---

# 🖥️ Streamlit Application

The project provides an interactive Streamlit dashboard with multiple views.

## 📊 Executive Dashboard

Provides an overall view of the customer base, including:

* Customer population
* Churn risk distribution
* Risk tiers
* Retention priorities
* Business-level trends

---

## 👤 Customer Analysis

Provides detailed analysis for an individual customer.

The view includes:

* Customer profile
* Churn risk
* Risk tier
* Behavioral state
* Behavioral evidence
* Deterioration indicators
* Customer value
* Retention decision

---

## 🚨 High-Risk Customers

Provides a focused view of customers belonging to the highest risk groups.

Users can identify:

* High-risk customers
* Very-high-risk customers
* Predicted churn risk
* Customer value
* Retention-related information

The table can also be downloaded for further analysis.

---

## 🎯 Retention Intelligence

Provides a business-oriented view of customer retention signals.

It includes:

* Behavioral state distribution
* Deterioration patterns
* Retention priorities
* Candidate actions
* Decision rationale

---

## 🤖 AI Retention Agent

Provides a conversational interface for asking questions about the customer base.

Users can interact with the system without manually navigating through multiple tables.

---

# 🛠️ Technology Stack

| Category               | Technology                 |
| ---------------------- | -------------------------- |
| Programming Language   | Python                     |
| Machine Learning       | LightGBM                   |
| Model Comparison       | Random Forest, XGBoost     |
| Data Processing        | Pandas, NumPy              |
| Visualization          | Plotly                     |
| Web Framework          | Streamlit                  |
| AI Agent Framework     | LangChain                  |
| Large Language Model   | Groq                       |
| Model Serialization    | Joblib                     |
| Environment Management | Python Virtual Environment |
| Version Control        | Git                        |
| Repository             | GitHub                     |

---

# 📁 Project Structure

```text
telecom-churn-retention-intelligence/
│
├── app/
│   └── app.py
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── prompts.py
│   └── tool.py
│
├── data/
│   └── processed/
│       └── telecom_retention_intelligence.csv
│
├── model/
│   └── final_lightgbm_churn_model.joblib
│
├── notebook/
│   └── ...
│
├── reports/
│   └── ...
│
├── package.json
├── package-lock.json
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Disha-HN/telecom-churn-retention-intelligence.git
```

Move into the project directory:

```bash
cd telecom-churn-retention-intelligence
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

If a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

Otherwise, install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn lightgbm xgboost plotly joblib python-dotenv langchain langchain-groq
```

---

# 🔐 Environment Variables

The AI Retention Agent requires a Groq API key.

Create a `.env` file in the project environment used by the application and add:

```env
GROQ_API_KEY=your_groq_api_key
```

The application reads the key through environment variables.

### Security

Never commit API keys to GitHub.

Your `.gitignore` should include:

```gitignore
.env
__pycache__/
*.pyc
node_modules/
```

---

# ▶️ Running the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit interface will then be available through the local URL displayed by Streamlit.

---

# 📊 Data Flow

The complete data flow can be summarized as:

```text
Raw Telecom Dataset
        │
        ▼
Data Understanding
        │
        ▼
Data Preparation
        │
        ▼
Feature Engineering
        │
        ▼
ML Model
        │
        ▼
Predicted Churn Risk
        │
        ├────────────┐
        │               │
        ▼               ▼
Behaviour Analysis   Customer Value
        │               │
        └───────┬───────┘
                  ▼
       Retention Intelligence
                │
      ┌───────┴────────┐
        ▼                    ▼
   Streamlit              AI Agent
   Dashboard             LangChain
                            │
                            ▼
                         Groq LLM
```

---

# 📌 Source-of-Truth Design

The system separates machine-learning predictions from business intelligence.

### ML Model

The ML model is the source of truth for:

```text
predicted_churn_risk
```

### Retention Intelligence Dataset

The processed retention-intelligence dataset is the source of truth for:

```text
risk_tier
behavioral_state
behavioral_evidence
retention_priority
value_tier
candidate_action
decision_rationale
```

This separation helps prevent the AI agent from confusing raw churn labels with predicted churn probability.

---

# 📈 Key Results

The final LightGBM model achieved:

```text
ROC-AUC   : 0.9426
PR-AUC    : 0.7453
Precision : 0.7504
Recall    : 0.6157
F1 Score  : 0.6764
```

The system then extends the prediction layer with:

```text
Churn Risk
    +
Behavioural Intelligence
    +
Customer Value
    +
Retention Priority
    +
Candidate Action
```

This provides a more complete view of customer retention than a standalone churn classifier.

---

# 🔒 Responsible Use

The predictions generated by this project should be treated as **decision-support signals**, not guaranteed predictions of future customer behavior.

Retention decisions should consider additional business context and appropriate human review.

The AI agent is also constrained to the available customer intelligence and should not be treated as an independent source of customer facts.

---

# 🔮 Future Scope

Potential future improvements include:

* Real-time telecom data integration
* Automated CRM integration
* Explainable AI using SHAP
* Automated retention campaign execution
* Customer Lifetime Value modeling
* Cost-aware retention optimization
* Real-time churn monitoring
* Cloud deployment
* Scalable model serving
* Automated model retraining
* Genuine ML prediction for newly entered customers using the complete training preprocessing pipeline

---

# 👩‍💻 Project

## Telecom Churn Retention Intelligence

A machine-learning and generative-AI-based decision-support system designed to help telecom businesses move from:

```text
"Who might churn?"
```

to:

```text
"Who should we retain first,
why are they at risk,
and what action should be considered?"
```

---

## ⭐ Core Idea

```text
              PREDICT
                 ↓
             CHURN RISK
                 ↓
             UNDERSTAND
                 ↓
        CUSTOMER BEHAVIOUR
                 ↓
             EVALUATE
                 ↓
         CUSTOMER VALUE
                 ↓
             PRIORITIZE
                 ↓
       RETENTION INTELLIGENCE
                 ↓
             RECOMMEND
                 ↓
          CANDIDATE ACTION
                 ↓
              EXPLAIN
                 ↓
          DECISION RATIONALE
                 ↓
             INTERACT
                 ↓
          AI RETENTION AGENT
```

**Built with Python, LightGBM, Streamlit, LangChain, and Groq.**
