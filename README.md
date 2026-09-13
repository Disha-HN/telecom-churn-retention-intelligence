# 🚀 Telecom Churn Retention Intelligence

AI-powered customer churn prediction and retention decision-support
system for telecom businesses.

[Python] [LightGBM] [LangChain] [Groq] [Streamlit]

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Predict churn → Understand behavior → Prioritize → Act   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

## 🎯 What is this?

Telecom companies don't just need to know which customers may churn.
They need to know:

• Who is most likely to churn?
• What behavioral changes are happening?
• Which customers need attention first?
• What retention action should be considered?

This project combines a machine-learning churn model with
retention intelligence and an AI agent to answer those questions
through an interactive dashboard.

## ✨ Key Features

🔮 Churn Prediction
   LightGBM-based churn probability prediction

📊 Customer Risk Intelligence
   Low / Moderate / High / Very High risk classification

🧠 Behavioral Intelligence
   Detects recent, persistent and coordinated deterioration

🎯 Retention Prioritization
   Identifies customers who should receive attention first

🤖 AI Retention Agent
   Ask questions about customers and retention data in natural language

📈 Interactive Dashboard
   Streamlit dashboard with analytics and customer-level insights

## 🏗️ System Architecture

                 TELECOM CUSTOMER DATA
                          │
                          ▼
                  DATA PREPROCESSING
                          │
                          ▼
                  FEATURE ENGINEERING
                          │
                          ▼
                 ┌─────────────────┐
                 │    LightGBM     │
                 │  Churn Model    │
                 └────────┬────────┘
                          │
                          ▼
                  CHURN PROBABILITY
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       RISK CLASSIFICATION      BEHAVIORAL ANALYSIS
             │                         │
             └────────────┬────────────┘
                          ▼
                 RETENTION INTELLIGENCE
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       STREAMLIT DASHBOARD       AI RETENTION AGENT
                                       │
                                  LangChain
                                       │
                                     Groq
                                       │
                                       ▼
                              Natural Language
                                  Insights

## 🤖 AI Retention Agent

Instead of manually filtering thousands of customers, users can ask:

> Which customers are at very high churn risk?

> Show me the highest priority customers.

> What is the retention summary?

> Analyze customer 2.

> What behavioral signals indicate churn?

The agent uses the project's retention-intelligence data and
specialized tools to produce grounded answers.

## 🧠 Machine Learning

Model: LightGBM

Features: 189 engineered features

Validation Performance:

| Metric | Score |
|--------|------:|
| ROC-AUC | 0.9426 |
| PR-AUC | 0.7453 |
| Precision | 0.7504 |
| Recall | 0.6157 |
| F1 Score | 0.6764 |

## 📊 Retention Intelligence

The system goes beyond a simple churn score.

It analyzes:

• ARPU changes
• Recharge amount changes
• Recharge frequency
• Outgoing usage
• Incoming usage

and derives:

• Recent deterioration
• Persistent deterioration
• Coordinated deterioration
• Behavioral state
• Value tier
• Retention priority
• Candidate retention action
• Decision rationale

## 🖥️ Application

### Executive Dashboard
[actual screenshot]

### Customer Analysis
[actual screenshot]

### High-Risk Customers
[actual screenshot]

### Retention Intelligence
[actual screenshot]

### AI Retention Agent
[actual screenshot]

## 🛠️ Tech Stack

Python
├── Pandas
├── NumPy
├── Scikit-learn
├── LightGBM
└── Joblib

AI
├── LangChain
└── Groq

Frontend
├── Streamlit
└── Plotly

## 📁 Project Structure

telecom-churn-retention-intelligence/
│
├── app/
│   └── app.py
│
├── agent/
│   ├── agent.py
│   ├── config.py
│   ├── prompts.py
│   ├── tool.py
│   └── __init__.py
│
├── data/
│   └── processed/
│       └── telecom_retention_intelligence.csv
│
├── model/
│   └── final_lightgbm_churn_model.joblib
│
├── notebook/
├── reports/
├── package.json
├── package-lock.json
└── README.md

## ⚡ Run Locally

git clone https://github.com/Disha-HN/telecom-churn-retention-intelligence.git

cd telecom-churn-retention-intelligence

pip install -r requirements.txt

streamlit run app/app.py

## 🔐 Environment Variables

Create a local `.env` file:

GROQ_API_KEY=your_api_key

Never commit `.env` to GitHub.

## 🎯 Why this project?

Traditional churn prediction:
    "This customer has a 78% probability of churning."

This system:
    "This customer is high risk because multiple behavioral
     indicators have deteriorated, the customer should be
     prioritized, and the system provides a candidate retention
     action."

That is the difference between **prediction** and
**retention intelligence**.

## 🔮 Future Scope

• Connect new-customer input to the complete ML preprocessing pipeline
• SHAP-based model explainability
• Automated retention campaign generation
• CRM integration
• Real-time churn monitoring
• Automated high-risk customer alerts

---

Built with Python • Machine Learning • Generative AI • Streamlit