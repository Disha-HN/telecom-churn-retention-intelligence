📡 Telecom Churn Retention Intelligence — AI-Powered Customer Retention



An AI-powered telecom customer retention decision-support system combining churn prediction, behavioural intelligence, customer value, retention prioritization, and an AI retention agent.

















📖 About



Telecom Churn Retention Intelligence is a customer retention system built to go beyond traditional churn prediction.



Instead of stopping at:



"Which customers are likely to churn?"



the system extends the analysis to:



"Which customers should receive retention attention, why do they need attention, and what should the retention team review?"



It combines machine learning with behavioural analysis, customer-value segmentation, business prioritization, and an AI-powered retention agent to turn churn predictions into a more actionable customer-retention workflow.



🖼️ Screenshots



Executive Dashboard



<!-- Add screenshot here -->



Customer Analysis



<!-- Add screenshot here -->



High-Risk Customers



<!-- Add screenshot here -->



AI Retention Agent



<!-- Add screenshot here -->



🚀 Key Features



Churn Risk Prediction: Predicts customer churn risk using machine-learning models trained on telecom customer behaviour.



Behavioural Deterioration Detection: Detects recent, persistent, and coordinated deterioration across ARPU, recharge activity, incoming usage, and outgoing usage.



Customer Value Segmentation: Uses the latest ARPU as a relative indicator of customer value.



Retention Prioritization: Combines churn risk, customer value, and behavioural condition to assign retention priorities.



High-Risk Customer Queue: Ranks customers by predicted churn risk for focused retention review.



Candidate Action Routing: Routes customers into review categories such as persistent deterioration, broad recent deterioration, recharge/affordability, usage/engagement, and general retention review.



AI Retention Agent: Lets retention managers ask natural-language questions about customers and receive explanations based on the actual retention-intelligence data.



Interactive Dashboard: Provides customer-level and executive-level views through Streamlit.



🧠 How It Works



&#x20;                   Customer Data

&#x20;                        │

&#x20;                        ▼

&#x20;             Data Preparation

&#x20;                        │

&#x20;                        ▼

&#x20;         Behavioural Feature Engineering

&#x20;                        │

&#x20;                        ▼

&#x20;             Churn Risk Prediction

&#x20;                   LightGBM

&#x20;                        │

&#x20;             ┌──────────┼──────────┐

&#x20;             ▼          ▼          ▼

&#x20;            Risk     Behaviour    Value

&#x20;             │          │          │

&#x20;             └──────────┼──────────┘

&#x20;                        ▼

&#x20;             Retention Prioritization

&#x20;                        │

&#x20;                        ▼

&#x20;             Candidate Action Routing

&#x20;                        │

&#x20;             ┌──────────┴──────────┐

&#x20;             ▼                     ▼

&#x20;      Streamlit Dashboard     AI Retention Agent

&#x20;             │                     │

&#x20;             └──────────┬──────────┘

&#x20;                        ▼

&#x20;             Business Decision Support



📊 Machine Learning



Multiple models were evaluated during development:



Random Forest



XGBoost



LightGBM



The final churn-risk model uses LightGBM.



Validation Performance



Metric



LightGBM



ROC-AUC



0.943



PR-AUC



0.745



The model produces a continuous predicted\_churn\_risk for each customer.



This score is then consumed by the retention-intelligence layer rather than being treated as a simple churn/no-churn label.



📈 Behavioural Intelligence



Traditional churn systems often focus heavily on the customer's latest values.



This project also examines how customer behaviour changes over time.



The system tracks:



ARPU



Recharge amount



Recharge frequency



Outgoing usage



Incoming usage



and derives:



Recent deterioration



Persistent deterioration



Coordinated deterioration



Example:



ARPU



June        July        August

&#x20;│           │            │

&#x20;▼           ▼            ▼

308         219          114

&#x20;            ↓            ↓

&#x20;      Continuous deterioration



These signals provide behavioural context around the model's churn-risk prediction.



🎯 Retention Intelligence



The main idea of the project is to move from:



"Who might churn?"



to:



"Who should we pay attention to, and why?"



The retention-intelligence layer combines:



Risk + Behaviour + Value

&#x20;         │

&#x20;         ▼

Retention Priority

&#x20;         │

&#x20;         ▼

Candidate Retention Review



Example



Customer

&#x20;  │

&#x20;  ├── Very High Churn Risk

&#x20;  │

&#x20;  ├── Persistent Deterioration

&#x20;  │

&#x20;  ├── Higher Customer Value

&#x20;  │

&#x20;  └── Priority 1

&#x20;         │

&#x20;         ▼

&#x20;   Retention Review



The priority is generated using predefined business rules rather than allowing the AI agent to arbitrarily change customer priority.



🤖 AI Retention Agent



The AI agent operates after the machine-learning and retention-intelligence layers.



Customer Dataset

&#x20;      ↓

Churn Prediction

&#x20;      ↓

Retention Intelligence

&#x20;      ↓

Retention Tools

&#x20;      ↓

AI Retention Agent

&#x20;      ↓

Business Explanation



The agent can answer questions such as:



Why should this customer receive attention?



What behavioural signals are concerning?



Why is this customer Priority 1 or Priority 2?



What retention issue should be reviewed?



Summarize this customer for a retention manager.



Which customer should be reviewed first?



The agent works from the generated retention-intelligence dataset and does not independently invent churn scores, customer value, or retention priorities.



🖥️ Dashboard



The Streamlit application provides dedicated views for:



📊 Executive Dashboard



Overview of customer risk, behavioural segments, value and retention priorities.



👤 Customer Analysis



Detailed customer-level retention intelligence.



🚨 High-Risk Customers



Prioritized customer list based on predicted churn risk.



🧠 Retention Intelligence



Behavioural deterioration, value segmentation and retention-priority analysis.



🤖 AI Retention Agent



Natural-language interaction with customer retention intelligence.



🛠️ Tech Stack



Language: Python



Data Processing: Pandas, NumPy



Machine Learning: Scikit-learn, LightGBM, XGBoost



AI Agent: LangChain, Groq



Dashboard: Streamlit



Visualization: Plotly



Development: Jupyter Notebook, VS Code



Version Control: Git, GitHub



📂 Project Structure



telecom-churn-retention-intelligence/

│

├── app/

│   └── app.py

│

├── agent/

│   ├── agent.py

│   └── tools.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── notebook/

│   ├── 01Data\_understanding.ipynb

│   ├── 02Data\_preparation.ipynb

│   ├── 03Model\_development.ipynb

│   └── 04Retention\_intelligence.ipynb

│

├── reports/

├── src/

├── requirements.txt

├── README.md

└── .gitignore



⚙️ Getting Started



Prerequisites



Python 3.11+



Git



Groq API key for the AI retention agent



Installation



git clone https://github.com/Disha-HN/telecom-churn-retention-intelligence.git

cd telecom-churn-retention-intelligence

python -m venv .venv



Activate the environment:



Windows



.venv\\Scripts\\activate



Install dependencies:



pip install -r requirements.txt



Configure Environment Variables



Create a .env file and add:



GROQ\_API\_KEY=your\_api\_key\_here



Run the Application



streamlit run app/app.py



⚠️ Project Scope



The current churn model is designed for existing telecom customers with historical behavioural information.



Because the model relies on multi-month behavioural features, it is not intended to make the same prediction for a completely new customer with no behavioural history.



A separate onboarding/acquisition model would be required for new customers.



🔮 Future Scope



Customer Lifetime Value (CLV)



Intervention-cost-aware prioritization



Uplift / treatment-effect modelling



Real-time customer behaviour monitoring



Customer support and complaint signals



Longer behavioural histories



Retention campaign feedback loops



Automated retention optimization



👩‍💻 Project



Telecom Churn Retention Intelligence



Predict the risk. Understand the behaviour. Prioritize the customer. Assist the decision.



