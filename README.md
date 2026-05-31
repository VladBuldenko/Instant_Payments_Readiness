# 💶 Instant Payments Readiness & Impact Simulator (DE Banking)

A data & simulation project exploring how Verification of Payee (VoP) and Fraud Filter thresholds affect:

Conversion Rate (%)

Latency p95 (s)

Manual Review Rate (%)

Risk Exposure (EUR)

Real Bundesbank trends (2022–2024) show the rise of Instant Payments and growing system load.
Synthetic simulation tests “what-if” operational scenarios.

The project also includes a machine learning fraud detection extension using a real transaction-level fraud dataset.

---

## 🔗 Data

Deutsche Bundesbank — Statistics on payments and securities trading, clearing and settlement in Germany.
Section I. Payments statistics (24.07.2025).
Source hub: https://www.bundesbank.de/en/statistics/banks-and-other-financial-corporations/payments-statistics/statistics-on-payments-and-securites-trading-clearing-and-settlement-in-germany-810330

Credit Card Fraud Detection — Anonymized credit card transactions made by European cardholders. Used for binary fraud classification with `Class` as the target variable. Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Used for:

SCT Inst growth

Domestic transaction volume/value

System load trends

ML fraud classification

Fraud model evaluation

Fraud probability estimation

Not included in Bundesbank source data:

VoP scores

Transaction-level fraud labels

Fraud probabilities

➡ VoP scores are simulated.
➡ Fraud probabilities are simulated in the original simulator and estimated with ML models in the fraud detection extension.

---

## 🎯 Purpose

Help a bank find the optimal balance between:

⚡ Speed (conversion, latency)

🛡 Security (fraud risk)

👤 Operational load (manual reviews)

The ML extension supports the fraud-risk part by training models to classify transactions as normal or fraudulent.

---

## 🔍 Hypotheses (Short Version)

### H1 — SCT Inst Growth (Real Data)

Instant payments grow while paper-based transfers decline → market is shifting to instant rails.

### H2 — Domestic Load Growth (Real Data)

Domestic electronic payments grow steadily → higher infrastructure pressure.

Conclusion (H1 + H2):
Instant payments adoption + rising system load → banks need operational optimization.

### H3 — VoP Strictness (Simulation)

Stricter VoP →

Conversion ↓

Latency ↑

### H4 — Fraud Filter Strictness (Simulation)

Higher sensitivity →

Manual reviews ↑

Risk exposure ↓

Conclusion (H3 + H4):
There is no perfect setting → optimal region is a balance.

---

## 🧮 Methodology

Bundesbank data → tidy format

Exploratory analysis (trends, descriptive stats)

Synthetic generator of instant payments (amount, VoP score, fraud probability)

Simulation engine (`sim_core.py`) calculating KPIs

Final heatmap showing global optimum (VoP × Fraud)

Interactive Streamlit app for exploration

Machine learning extension:

Credit Card Fraud Detection data → ETL process

EDA for class imbalance, amount, time, and fraud-related features

Model training with DummyClassifier, Logistic Regression, and Random Forest

Model evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC

---

## 🧭 Key Result: Recommended Settings

Based on simulation trade-offs:

VoP ≈ 0.75–0.85
Fraud ≈ 0.40–0.55

Balances Conversion, Latency, Risk, and Manual Workload.

### 🤖 ML Result: Fraud Detection

DummyClassifier confirmed that accuracy is misleading for imbalanced fraud data.

Logistic Regression detected many fraud cases but created many false positives.

Random Forest provided the best practical balance between precision, recall, F1-score, and PR-AUC.

Random Forest is the preferred model for future fraud probability and threshold analysis.

---

## ⚖️ Ethics and Limitations

Fraud detection models can affect real customers.

False positives can create unnecessary manual reviews and customer friction.

False negatives can allow fraudulent transactions to pass and increase risk exposure.

The Credit Card Fraud Detection dataset is anonymized and does not include demographic attributes, so full fairness testing is limited.

The ML dataset is credit card data, not real instant payment data.

VoP scores are simulated because public transaction-level VoP datasets are not available.

The model should support decision-making, not fully replace human review.

---

## 📁 Project Structure

app/                   # Streamlit app
src/                   # Simulation engine
data/raw/              # Bundesbank source
data/processed/        # Tidy datasets
notebooks/             # Analysis & experiments
tests/                 # Unit tests
reports/figures/       # Saved plots

---

## 🚀 Run the Simulator

```bash
git clone <repo>.git
cd Instant-Payments-Readiness-Impact-Simulator-DE-Banking-

python -m venv venv
source venv/bin/activate     # macOS/Linux
# .\venv\Scripts\Activate.ps1 # Windows

pip install -r requirements.txt

streamlit run app/instant_simulator_app.py
```

Tests:

```bash
pytest -q
```

Version Control (example)

```bash
git status
git add .
git commit -m "Stage 1: Data extraction & cleaning (Bundesbank payments)"
git push
```

---

### AI Assistance

Parts of this project were assisted by ChatGPT (OpenAI):

* Brainstorming the overall project idea and hypotheses (H1–H5)
* Drafting some markdown explanations in the Jupyter notebooks
* Getting help with plotting code (matplotlib / seaborn) and refactoring functions in `src/sim_core.py`
* Explaining ML concepts such as classification, threshold, precision, recall, F1-score, ROC-AUC, and PR-AUC
* Helping structure the ETL, EDA, and model evaluation workflow

All final decisions, code execution, validation, interpretation, and conclusions were reviewed and completed by the author.

---

## 👤 Author

Vladyslav Buldenko — Data Analyst
📧 [vlad.buldenko.96@gmail.com](mailto:vlad.buldenko.96@gmail.com)
🔗 LinkedIn: https://www.linkedin.com/in/vladyslav-buldenko-4b2069182/
