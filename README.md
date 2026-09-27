# Churn Analysis & Retention — Telco Customer Churn

End-to-end churn analysis on the IBM Telco Customer Churn dataset (7,043 customers,
21 columns): cleaning → feature engineering → EDA → modeling → business-impact
threshold optimization → risk-tier prioritized outreach list.

## How to run

```bash
pip install -r requirements.txt
python 01_clean_data.py
python 02_feature_engineering.py
python 03_eda.py
python 04_modeling.py
python 05_business_impact.py
```

Each script reads the previous script's output from `data/` or `outputs/` —
run them in order the first time.

## Dashboard

Once scripts 01, 02, and 04 have been run at least once (so `data/telco_featured.csv`
and `models/best_model.pkl` exist):

```bash
streamlit run app.py
```

Opens an interactive dashboard with 4 tabs:
- **Overview** — headline KPIs and the model comparison table
- **EDA** — interactive contract/tenure/payment-method charts
- **Predict a Customer** — fill in a hypothetical customer's details and get a
  live churn probability + risk tier from the trained model
- **Risk List** — filterable/sortable prioritized outreach list, downloadable as CSV

## Files

| Script | What it does |
|---|---|
| `01_clean_data.py` | Fixes `TotalCharges` type, handles the 11 blank-string rows (new customers), creates `ChurnFlag` |
| `02_feature_engineering.py` | Tenure buckets, revenue-per-tenure-month, add-on service count, contract×price interaction flag |
| `03_eda.py` | 5 plots + printed interpretation each — contract type, tenure "danger zone", an add-on/contract interaction, price×contract interaction, payment method |
| `04_modeling.py` | Logistic Regression, Random Forest, XGBoost — compares `class_weight`/`scale_pos_weight` vs SMOTE, full precision/recall/F1/ROC-AUC table, saves best model |
| `05_business_impact.py` | Cost-based threshold optimization (net $ value, not accuracy) + High/Medium/Low risk tiers + a prioritized outreach CSV |

## Key results (this run)

- Overall churn rate: **26.5%**
- Month-to-month churn (**42.7%**) vs longest-contract churn (**2.8%**) — a ~15x gap
- First 6 months are the highest-risk window (**52.9%** churn vs **9.5%** after 4+ years)
- Best model: **Logistic Regression** (ROC-AUC 0.841) — beat both tree-based models on this dataset, which is itself worth mentioning in an interview: more complexity isn't automatically better
- Switching from a naive 0.5 probability cutoff to the cost-optimal threshold (given assumed $15 campaign cost / $500 avg CLV) improved modeled net campaign value by **~$68k** on the test set alone

## Honest caveats (say these out loud in an interview — they show judgment, not weakness)

- The counter-intuitive add-on finding (customers WITH add-ons churn *more* within
  month-to-month contracts) is worth investigating further, not just reporting —
  a real analyst would check billing complaints or run this by a stakeholder
  before acting on it.
- `CAMPAIGN_COST` and `AVG_CLV` in `05_business_impact.py` are assumed placeholder
  numbers. In a real setting these come from finance/marketing — swap them in and
  the optimal threshold and $ impact will change.
- The optimal threshold found (0.06) is aggressive — it flags most customers for
  outreach because the assumed CLV ($500) is much larger than the campaign cost
  ($15). That's a legitimate result of these assumptions, not a bug, but it's
  worth sanity-checking against what a real retention team could operationally
  handle (call center capacity, etc.).
