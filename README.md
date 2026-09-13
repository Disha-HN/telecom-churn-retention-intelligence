# 📡 Telecom Churn Retention Intelligence

> **Predict Churn • Understand Behavior • Prioritize Customers • Guide Retention Action**

An AI-assisted telecom customer retention decision-support system that combines **machine learning, behavioral intelligence, customer value analysis, retention prioritization, candidate action routing, and a natural-language AI agent**.

The system moves beyond traditional churn prediction by helping answer not only **who is likely to churn**, but also **what behavioral signals support the risk, which customers deserve greater retention attention, and what retention issue should be reviewed**.

🌐 **Live Application:** `https://telecom-churn-retention-intelligence-mlihtcsdxhlddnmxuvxjeh.streamlit.app/`

📦 **GitHub Repository:** `https://github.com/Disha-HN/telecom-churn-retention-intelligence`

---

# 🚀 Overview

Traditional churn prediction systems mainly answer:

> **"Which customers are likely to churn?"**

However, a telecom retention team also needs to understand:

* What is happening to the customer?
* Is the customer's behavior deteriorating?
* How valuable is the customer?
* Which customers should be prioritized?
* What retention issue should be reviewed?

This project combines a **regularized LightGBM churn model** with behavioral intelligence, customer value analysis, retention prioritization, candidate action routing, and an AI-powered Retention Agent.

```text
Customer Data
      ↓
Data Understanding
      ↓
Data Preparation
      ↓
Feature Engineering
      ↓
Churn Prediction
      ↓
Risk + Behaviour + Value
      ↓
Retention Priority
      ↓
Candidate Action + Rationale
      ↓
Streamlit Dashboard + AI Agent
      ↓
Business Decision Support
```

### Core idea

> **Predict the risk. Understand the behaviour. Prioritize the customer. Assist the decision.**

---

# 🎯 Problem Statement

A high churn probability alone does not provide enough information for a retention team to decide where to focus its efforts.

For example, two customers may have similar churn risk but very different:

* behavioral deterioration,
* customer value,
* retention priority,
* and retention-review requirements.

Therefore, the project combines:

```text
                 Churn Risk
                     +
                Behaviour
                     +
               Customer Value
                     ↓
          Retention Intelligence
                     ↓
             Retention Priority
                     ↓
             Candidate Action
                     ↓
             Decision Support
```

The goal is to move from **prediction-only** to **decision-oriented customer retention intelligence**.

---

# 🏗️ Complete System Architecture

```text
                         TELECOM CUSTOMER DATA
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   DATA UNDERSTANDING    │
                    │                         │
                    │ • Data profiling        │
                    │ • Missingness analysis  │
                    │ • Churn distribution   │
                    │ • Temporal analysis    │
                    │ • Data-quality checks  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   DATA PREPARATION      │
                    │                         │
                    │ • Metadata handling     │
                    │ • Missing-value logic   │
                    │ • Activity indicators  │
                    │ • Validation            │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  FEATURE ENGINEERING    │
                    │                         │
                    │ • Monthly behaviour     │
                    │ • Temporal changes      │
                    │ • Persistent decline    │
                    │ • Recent deterioration  │
                    │ • Coordinated decline   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    CHURN MODELING       │
                    │                         │
                    │ • Random Forest         │
                    │ • XGBoost               │
                    │ • LightGBM              │
                    │ • Model evaluation      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     CHURN RISK SCORE    │
                    │                         │
                    │ predicted_churn_risk   │
                    │                         │
                    │ Low / Moderate / High  │
                    │ Very High               │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
 ┌─────────────────────────┐          ┌─────────────────────────┐
 │ BEHAVIOURAL INTELLIGENCE│          │   CUSTOMER VALUE        │
 │                         │          │                         │
 │ • Recent deterioration  │          │ • ARPU                  │
 │ • Persistent decline    │          │ • Value tier            │
 │ • Coordinated decline   │          │ • ARPU exposure         │
 │ • Behavioural state     │          │                         │
 └────────────┬────────────┘          └────────────┬────────────┘
              │                                     │
              └──────────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ RETENTION INTELLIGENCE  │
                    │                         │
                    │ Risk + Behaviour +      │
                    │ Value                   │
                    │          ↓              │
                    │ Retention Priority      │
                    │          ↓              │
                    │ Candidate Action        │
                    │          ↓              │
                    │ Decision Rationale      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐      ┌────────────────────────┐
          │ STREAMLIT        │      │ AI RETENTION AGENT     │
          │ DASHBOARD        │      │                        │
          │                  │      │ Customer Context       │
          │ • Executive      │      │        ↓               │
          │ • Customer       │      │ Retention Tools        │
          │ • High Risk      │      │        ↓               │
          │ • Intelligence   │      │ LangChain Agent        │
          │ • Operations     │      │        ↓               │
          └──────────────────┘      │ Groq LLM               │
                                    │        ↓               │
                                    │ Natural Language       │
                                    │ Retention Insights     │
                                    └────────────────────────┘
```

---

# 🔥 Key Features

## 🎯 ML-Based Churn Prediction

The system uses machine learning to estimate the probability that a customer will churn.

Three models were evaluated:

* Random Forest
* XGBoost
* LightGBM

The final system uses a **regularized LightGBM classifier**.

```text
Customer
    ↓
189 Predictive Features
    ↓
LightGBM
    ↓
Predicted Churn Risk
```

The model-generated probability is represented by:

```text
predicted_churn_risk
```

---

## 📊 Behavioral Intelligence

The system analyzes customer behavior across multiple observed months instead of relying only on a single churn score.

Behavioral signals include:

* Recent deterioration
* Persistent decline
* Coordinated deterioration
* ARPU changes
* Recharge behavior changes
* Outgoing usage changes
* Incoming usage changes
* Overall behavioral state
* Evidence supporting the behavioral assessment

This helps answer:

> **"What is happening to the customer?"**

---

## 💰 Customer Value Analysis

Customer value is represented using observable indicators such as:

* ARPU
* Value tier
* ARPU exposure

This allows the system to distinguish between different types of high-risk customers.

For example:

```text
High Churn Risk
       +
Higher Customer Value
       +
Behavioural Deterioration
       ↓
Higher Retention Priority
```

> **Note:** ARPU is used as a value indicator. The system does not claim ARPU to be Customer Lifetime Value, profit, or margin.

---

# 🧠 Machine Learning

## Feature Engineering

The final model uses **189 predictive features** after excluding the customer identifier.

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

---

## 📅 Temporal Behaviour

Customer behavior is analyzed across observed monthly periods:

```text
June → July → August
```

The system calculates temporal changes such as:

```text
June → July
July → August
June → August
```

This allows the system to represent:

* Short-term behavioral changes
* Overall deterioration
* Persistent decline
* Coordinated deterioration

---

# 🤖 Model Comparison

| Model         |    ROC-AUC |     PR-AUC |
| ------------- | ---------: | ---------: |
| Random Forest |     0.9349 |     0.7272 |
| XGBoost       |     0.9406 |     0.7388 |
| **LightGBM**  | **0.9426** | **0.7453** |

LightGBM achieved the strongest validation performance among the evaluated models.

---

# 🌿 Final LightGBM Model

The final model uses regularization and constrained tree complexity to reduce overfitting.

Key configuration:

```text
n_estimators       = 500
learning_rate      = 0.05
max_depth          = 4
num_leaves         = 15
min_child_samples  = 60
subsample           = 0.8
colsample_bytree    = 0.8
reg_lambda          = 1
```

### Validation Performance

```text
ROC-AUC : 0.9426
PR-AUC  : 0.7453
```

The trained model is stored at:

```text
model/final_lightgbm_churn_model.joblib
```

> **Evaluation note:** These metrics are validation-set results from the model-development stage. The same validation population was used during model and threshold exploration, so they should not be interpreted as an unbiased production performance estimate.

---

# 📈 Churn Risk Segmentation

Predicted churn probabilities are converted into four relative risk tiers:

```text
Low
Moderate
High
Very High
```

The tiers are based on **relative quartiles of the scored population**.

Therefore:

> **Very High means the customer belongs to the highest relative risk segment. It does not necessarily mean the customer's absolute churn probability is greater than 50%.**

---

# 🔎 Behavioral Intelligence

## Recent Deterioration

Identifies whether important customer activity has declined recently.

Examples:

```text
ARPU              ↓
Recharge Amount   ↓
Recharge Count    ↓
Outgoing Usage    ↓
Incoming Usage    ↓
```

---

## Persistent Decline

Identifies behavioral signals that show decline across multiple observed periods.

```text
June → July → August
       ↓       ↓
   Sustained decline
```

Persistent deterioration provides stronger evidence of sustained decline than a single month-to-month change.

---

## Coordinated Decline

Measures how many important behavioral dimensions are declining together.

```text
ARPU              ↓
Recharge Count    ↓
Outgoing Usage    ↓
Incoming Usage    ↓
                  ↓
      Coordinated Deterioration
```

---

## Behavioral States

The system converts deterioration signals into interpretable descriptive states:

```text
Stable / Limited Deterioration
Recent Deterioration
Recent Broad Deterioration
Persistent Deterioration
Severe Persistent Deterioration
```

These are analytical categories describing observed data patterns. They are not psychological states or causal explanations.

---

# 🎯 Retention Intelligence

The retention layer combines:

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

Customer-level retention intelligence contains:

* Predicted churn risk
* Risk tier
* Behavioral state
* Behavioral evidence
* Recent deterioration count
* Persistent deterioration count
* Coordinated deterioration count
* Value tier
* ARPU
* Retention priority
* Candidate action
* Decision rationale

The processed dataset is stored at:

```text
data/processed/telecom_retention_intelligence.csv
```

---

# 🧭 Candidate Retention Actions

Customers are routed to candidate retention-review categories based on observable behavioral patterns.

Examples include:

```text
Persistent Deterioration Review
Broad Recent Deterioration Review
Recharge / Affordability Review
Usage / Engagement Review
General Retention Review
```

These are **candidate review categories**, not guaranteed treatments.

The system does not claim that a particular action will causally prevent churn.

---

# 🤖 AI Retention Agent

The AI Retention Agent provides a natural-language interface over the retention-intelligence layer.

### Architecture

```text
User Question
      ↓
Customer / Business Context
      ↓
Retention Intelligence Tools
      ↓
LangChain Agent
      ↓
Groq LLM
      ↓
Natural Language Retention Insight
```

The agent does not independently calculate or modify the ML churn probability.

Instead:

```text
ML Model
   ↓
Predicted Churn Risk
   ↓
Retention Intelligence Dataset
   ↓
Agent Tools
   ↓
LLM Reasoning
   ↓
Business Explanation
```

---

# 🛠️ Agent Tools

The AI agent can access:

* Customer profiles
* Highest churn-risk customers
* Priority customers
* Customer behavioral analysis
* Retention summaries

---

# ⚡ Intelligent Query Routing

Simple and predictable queries can be handled directly through deterministic tools without requiring a full LLM reasoning cycle.

```text
                    User Question
                         ↓
                    Query Router
                    /          \
                   /            \
          Simple Query       Complex Query
               ↓                   ↓
        Deterministic Tool    LangChain Agent
                                   ↓
                                Groq LLM
```

This design provides:

* Faster responses for simple queries
* Lower unnecessary LLM usage
* Deterministic answers for known operations
* LLM reasoning for complex natural-language questions

---

# 💬 Example AI Agent Queries

```text
Which customers have the highest predicted churn risk?
```

```text
Which customers should we prioritize for retention?
```

```text
What is the retention summary?
```

```text
Analyze customer 2.
```

```text
Why is this customer Priority 2?
```

```text
What behavioural signals indicate customer deterioration?
```

The system maintains separation between:

```text
Observed Evidence
        ≠
ML Prediction
        ≠
Business Priority
        ≠
Agent Explanation
        ≠
Causal Treatment Effect
```

---

# 🖥️ Streamlit Application

The project provides an interactive Streamlit dashboard with six major views.

## 📊 Executive Dashboard

Provides an overall view of the customer portfolio:

* Customer population
* Churn distribution
* Risk distribution
* Retention priorities
* Business-level indicators

---

## 👤 Customer Analysis

Provides detailed analysis for an individual customer:

* Customer profile
* Predicted churn risk
* Risk tier
* Behavioral state
* Behavioral evidence
* Deterioration indicators
* Customer value
* Retention priority
* Candidate action
* Decision rationale

---

## 🚨 High-Risk Customers

Provides a focused view of customers with high predicted churn risk.

Users can examine:

* Predicted churn probability
* Risk tier
* Customer value
* Retention priority
* Behavioral information

The results can also be downloaded for further analysis.

> **Important:** High churn risk and retention priority are different concepts.

---

## 🧠 Retention Intelligence

Provides a business-oriented view of:

* Risk segmentation
* Behavioral state distribution
* Deterioration patterns
* Retention priorities
* Risk × value analysis
* Candidate actions
* Decision rationale

---

## 🤖 AI Retention Agent

Provides a conversational interface for asking questions about:

* Customers
* Churn risk
* Behavioral deterioration
* Retention priorities
* Retention summaries

---

## 📈 Retention Operations

Converts retention intelligence into an operational customer-review queue.

The view provides:

* Priority 1 customers
* Priority 2 customers
* Retention action queue
* Candidate action workload
* ARPU exposure
* Operational playbook
* Retention-review workflow

```text
Predicted Risk
      ↓
Retention Priority
      ↓
Behavioural Context
      ↓
Candidate Action
      ↓
Human Review
      ↓
Retention Decision
```

The current prototype is **batch-based decision support** using prepared historical customer data. It is not a real-time production scoring system.

---

# 🔄 End-to-End Data Flow

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
        ├────────────────┐
        │                │
        ▼                ▼
Behaviour Analysis   Customer Value
        │                │
        └────────┬───────┘
                 │
                 ▼
       Retention Intelligence
                 │
          ┌──────┴──────┐
          │             │
          ▼             ▼
     Streamlit       AI Agent
     Dashboard       Tools
                         │
                         ▼
                    LangChain
                         │
                         ▼
                      Groq LLM
```

---

# 📌 Source-of-Truth Design

The system deliberately separates ML predictions from business intelligence.

### ML Model

The ML model is the source of truth for:

```text
predicted_churn_risk
```

This represents the model-generated churn probability.

### Retention Intelligence Dataset

The processed dataset is the source of truth for:

```text
risk_tier
behavioral_state
behavioral_evidence
retention_priority
value_tier
candidate_action
decision_rationale
```

This prevents the AI agent from confusing:

```text
Observed Churn Label
        ≠
Predicted Churn Probability
        ≠
Business Priority
```

---

# 🔄 CI/CD & Deployment

The project uses **GitHub Actions** for continuous integration and **Streamlit Community Cloud** for deployment.

## Continuous Integration

Every push to the `master` branch and pull request targeting `master` can trigger the CI workflow.

The workflow performs:

* Python environment setup
* Dependency installation
* Python syntax validation
* Core dependency validation
* Project import validation

```text
Developer Push
      ↓
GitHub Repository
      ↓
GitHub Actions
      ↓
Install Dependencies
      ↓
Validate Python Files
      ↓
Validate Project Imports
      ↓
🟢 CI Pass
```

Workflow:

```text
.github/workflows/ci.yml
```

---

# 🌐 Deployment

The Streamlit application is designed for deployment through **Streamlit Community Cloud**.

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
app/app.py
       ↓
Live Streamlit Application
       ↓
Groq AI Retention Agent
```

### Streamlit Configuration

```text
Repository:
Disha-HN/telecom-churn-retention-intelligence

Branch:
master

Main file:
app/app.py
```

---

# 🔐 Secret Management

The Groq API key should never be committed to GitHub.

### Local Development

Create a local `.env` file:

```text
GROQ_API_KEY=your_groq_api_key
```

### Streamlit Cloud

Configure the key through Streamlit Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

The application reads the key through environment configuration.

> **Never upload `.env` or expose API keys in source code.**

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
| Continuous Integration | GitHub Actions             |
| Deployment             | Streamlit Community Cloud  |

---

# 📁 Project Structure

```text
telecom-churn-retention-intelligence/
│
├── .github/
│   └── workflows/
│       └── ci.yml
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
│   ├── 01Data_understanding.ipynb
│   ├── 02Data_preparation.ipynb
│   ├── 03Model_development.ipynb
│   └── 04Retention_intelligence.ipynb
│
├── requirements.txt
├── README.md
├── package.json
└── package-lock.json
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Disha-HN/telecom-churn-retention-intelligence.git
cd telecom-churn-retention-intelligence
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

From the project root:

```bash
streamlit run app/app.py
```

The local Streamlit URL will be displayed in the terminal.

---

# 📊 Key Results

### Model Comparison

```text
Random Forest
ROC-AUC : 0.9349
PR-AUC  : 0.7272

XGBoost
ROC-AUC : 0.9406
PR-AUC  : 0.7388

LightGBM
ROC-AUC : 0.9426
PR-AUC  : 0.7453
```

### Final Model

```text
Model    : LightGBM
Features : 189
ROC-AUC  : 0.9426
PR-AUC   : 0.7453
```

The project then extends the predictive layer with:

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
    +
Decision Rationale
```

---

# 🔒 Responsible Use

The predictions generated by this project should be treated as **decision-support signals**, not guaranteed predictions of future customer behavior.

The system does not claim:

* Causal treatment effectiveness
* Guaranteed churn prevention
* Customer Lifetime Value
* Profit or margin impact
* Guaranteed ROI from retention actions

Behavioral intelligence describes observed patterns in the available data.

Candidate actions represent **retention-review categories**, not proven interventions.

The AI agent is grounded in the available customer intelligence and should not be treated as an independent source of customer facts.

Human review remains important when making actual retention decisions.

---

# ⚠️ Current Limitations

The current prototype has several limitations:

* Uses prepared historical telecom data
* Does not perform real-time customer scoring
* Does not directly integrate with a telecom CRM
* Candidate actions are not causal treatment recommendations
* ARPU is used as a value indicator rather than true CLV
* Validation metrics are not a production performance guarantee
* New-customer prediction requires the complete original preprocessing pipeline
* Production monitoring and model-drift detection are not yet implemented

---

# 🔮 Future Scope

Potential future improvements include:

* Real-time telecom data integration
* Automated CRM integration
* Explainable AI using SHAP
* Customer Lifetime Value modeling
* Cost-aware retention optimization
* Automated retention campaign execution
* Real-time churn monitoring
* Scalable model serving
* Automated model retraining
* Genuine ML prediction for newly entered customers using the complete preprocessing pipeline
* Production-grade monitoring
* Model drift detection
* Automated model performance monitoring

---

# ⭐ Project Impact

The project demonstrates how a traditional churn prediction system can be extended into a broader **retention intelligence platform**.

Instead of stopping at:

```text
"Who might churn?"
```

the system moves toward:

```text
"Who might churn?"
        ↓
"Why does the data indicate risk?"
        ↓
"What behavioural changes are visible?"
        ↓
"How valuable is the customer?"
        ↓
"Who should receive greater retention attention?"
        ↓
"What retention issue should be reviewed?"
```

---

# 🚀 Engineering Workflow

```text
             GitHub
                ↓
       GitHub Actions CI
                ↓
      Automated Validation
                ↓
   Streamlit Community Cloud
                ↓
       Live Application
                ↓
       AI Retention Agent
```

The project demonstrates an end-to-end workflow covering:

**Data Preparation → Feature Engineering → Machine Learning → Behavioral Intelligence → Retention Prioritization → AI-Assisted Reasoning → Streamlit Dashboard → CI/CD → Cloud Deployment**

---

# 👩‍💻 Project

## Telecom Churn Retention Intelligence

A machine-learning and generative-AI-assisted decision-support system designed to help telecom businesses move from:

> **Prediction-only churn analysis**

to:

> **Risk-aware, behavior-aware, value-aware retention decision support.**

---

# 💡 Core Idea

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
              ROUTE
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
                 ↓
        BUSINESS DECISION SUPPORT
```

---

**Built with Python, LightGBM, Pandas, NumPy, Plotly, Streamlit, LangChain, Groq, GitHub Actions, and Streamlit Community Cloud.**
