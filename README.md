💶 Instant Payments Readiness & Impact Simulator (DE Banking)

A data & simulation project exploring how Verification of Payee (VoP) and Fraud Filter thresholds affect:

Conversion Rate (%)

Latency p95 (s)

Manual Review Rate (%)

Risk Exposure (EUR)

Real Bundesbank trends (2022–2024) show the rise of Instant Payments and growing system load.
Synthetic simulation tests “what-if” operational scenarios.

🔗 Data

Deutsche Bundesbank — Statistics on payments and securities trading, clearing and settlement in Germany.
Section I. Payments statistics (24.07.2025).
Source hub: https://www.bundesbank.de/en/statistics/banks-and-other-financial-corporations/payments-statistics/statistics-on-payments-and-securites-trading-clearing-and-settlement-in-germany-810330

Credit Card Fraud Detection — Anonymized credit card transactions made by European cardholders. Used for binary fraud classification with Class as the target variable. Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Used for:

SCT Inst growth

Domestic transaction volume/value

System load trends

Not included in source data:

VoP scores

Fraud probabilities
➡ These are simulated.

🎯 Purpose

Help a bank find the optimal balance between:

⚡ Speed (conversion, latency)

🛡 Security (fraud risk)

👤 Operational load (manual reviews)

🔍 Hypotheses (Short Version)
H1 — SCT Inst Growth (Real Data)

Instant payments grow while paper-based transfers decline → market is shifting to instant rails.

H2 — Domestic Load Growth (Real Data)

Domestic electronic payments grow steadily → higher infrastructure pressure.

Conclusion (H1 + H2):
Instant payments adoption + rising system load → banks need operational optimization.

H3 — VoP Strictness (Simulation)

Stricter VoP →

Conversion ↓

Latency ↑

H4 — Fraud Filter Strictness (Simulation)

Higher sensitivity →

Manual reviews ↑

Risk exposure ↓

Conclusion (H3 + H4):
There is no perfect setting → optimal region is a balance.

🧮 Methodology

Bundesbank data → tidy format

Exploratory analysis (trends, descriptive stats)

Synthetic generator of instant payments (amount, VoP score, fraud probability)

Simulation engine (sim_core.py) calculating KPIs

Final heatmap showing global optimum (VoP × Fraud)

Interactive Streamlit app for exploration

🧭 Key Result: Recommended Settings

Based on simulation trade-offs:

VoP ≈ 0.75–0.85
Fraud ≈ 0.40–0.55

Balances Conversion, Latency, Risk, and Manual Workload.

📁 Project Structure
app/                   # Streamlit app
src/                   # Simulation engine
data/raw/              # Bundesbank source
data/processed/        # Tidy datasets
notebooks/             # Analysis & experiments
tests/                 # Unit tests
reports/figures/       # Saved plots

🚀 Run the Simulator
git clone <repo>.git
cd Instant-Payments-Readiness-Impact-Simulator-DE-Banking-

python -m venv venv
source venv/bin/activate     # macOS/Linux
# .\venv\Scripts\Activate.ps1 # Windows

pip install -r requirements.txt

streamlit run app/instant_simulator_app.py


Tests:

pytest -q

Version Control (example)

git status
git add .
git commit -m "Stage 1: Data extraction & cleaning (Bundesbank payments)"
git push

### AI Assistance

Parts of this project were assisted by ChatGPT (OpenAI):
- Brainstorming the overall project idea and hypotheses (H1–H5)
- Drafting some markdown explanations in the Jupyter notebooks
- Getting help with plotting code (matplotlib / seaborn) and refactoring functions in `src/sim_core.py`

👤 Author

Vladyslav Buldenko — Data Analyst
📧 vlad.buldenko.96@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/vladyslav-buldenko-4b2069182/
