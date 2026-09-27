"""
app.py
------
Interactive Streamlit dashboard for the churn analysis project.
Wraps the outputs of 01-05 scripts into a clickable UI:
- Overview: headline KPIs
- EDA: interactive charts (contract, tenure, payment method)
- Predict: score a new customer using the saved best model
- Risk List: browse/filter the prioritized outreach list

Self-bootstrapping: if the cleaned/featured data or trained model aren't
present (e.g. on a fresh Streamlit Cloud deploy where only the raw CSV is
committed to git), this file regenerates them on first load from the raw
CSV, so `streamlit run app.py` works on its own without running 01/02/04
by hand first. Results are cached so this only happens once per session.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Churn Analysis Dashboard", layout="wide")

RAW_PATH = "data/Telco-Customer-Churn.csv"
DATA_PATH = "data/telco_featured.csv"
MODEL_PATH = "models/best_model.pkl"
OUTREACH_PATH = "outputs/prioritized_outreach_list.csv"
COMPARISON_PATH = "outputs/model_comparison.csv"

NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "RevenuePerTenureMonth", "NumAddonServices",
]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthToMonthHighCharge",
]
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def clean_and_engineer(raw_path: str) -> pd.DataFrame:
    """Same logic as 01_clean_data.py + 02_feature_engineering.py, inlined
    so the dashboard can bootstrap itself from just the raw CSV."""
    df = pd.read_csv(raw_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    df["TenureBucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )
    df["RevenuePerTenureMonth"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["NumAddonServices"] = (df[ADDON_COLS] == "Yes").sum(axis=1)
    df["HasNoAddons"] = (df["NumAddonServices"] == 0).astype(int)
    median_charge = df["MonthlyCharges"].median()
    df["MonthToMonthHighCharge"] = (
        (df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > median_charge)
    ).astype(int)
    return df


def train_quick_model(df: pd.DataFrame):
    """Trains a Logistic Regression pipeline (the model that won in
    04_modeling.py's comparison) as a fallback when no saved model exists."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["ChurnFlag"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test


def build_outreach_list(pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    out = X_test.copy()
    out["ChurnFlag"] = y_test.values
    out["ChurnProba"] = proba
    out["RiskTier"] = out["ChurnProba"].apply(
        lambda p: "High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low"
    )
    return out


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    if os.path.exists(RAW_PATH):
        df = clean_and_engineer(RAW_PATH)
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df
    return None


@st.cache_resource
def load_model_and_outreach(df: pd.DataFrame):
    """Returns (model_obj, outreach_df). Loads saved artifacts if present,
    otherwise trains/generates them on the fly from df."""
    if os.path.exists(MODEL_PATH):
        model_obj = joblib.load(MODEL_PATH)
    else:
        model_obj = None  # trained below alongside the outreach list

    if os.path.exists(OUTREACH_PATH) and model_obj is not None:
        outreach = pd.read_csv(OUTREACH_PATH)
        return model_obj, outreach

    # Bootstrap: train a fresh model and build the outreach list from it
    pipe, X_test, y_test = train_quick_model(df)
    outreach = build_outreach_list(pipe, X_test, y_test)
    if model_obj is None:
        model_obj = pipe
    return model_obj, outreach


def predict_one(model_obj, row_df):
    """Handles both a plain sklearn Pipeline and the
    {'preprocessor':..., 'model':...} dict saved for the SMOTE/XGBoost case."""
    if isinstance(model_obj, dict):
        X_enc = model_obj["preprocessor"].transform(row_df)
        proba = model_obj["model"].predict_proba(X_enc)[:, 1]
    else:
        proba = model_obj.predict_proba(row_df)[:, 1]
    return proba[0]


df = load_data()

st.title("📉 Customer Churn Analysis — Telco Dataset")

if df is None:
    st.error(
        f"Could not find `{RAW_PATH}`. Make sure the raw dataset CSV is "
        "committed to the repo under `data/`."
    )
    st.stop()

with st.spinner("Preparing model (first load only, cached after this)..."):
    model_obj, outreach_df = load_model_and_outreach(df)

tab_overview, tab_eda, tab_predict, tab_risk = st.tabs(
    ["📊 Overview", "🔍 EDA", "🧮 Predict a Customer", "🎯 Risk List"]
)

# ---------------- Overview ----------------
with tab_overview:
    churn_rate = df["ChurnFlag"].mean()
    mtm_rate = df[df["Contract"] == "Month-to-month"]["ChurnFlag"].mean()
    longterm_rate = df[df["Contract"] != "Month-to-month"]["ChurnFlag"].mean()
    early_churn = df[df["tenure"] <= 6]["ChurnFlag"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall churn rate", f"{churn_rate:.1%}")
    c2.metric("Month-to-month churn", f"{mtm_rate:.1%}")
    c3.metric("Longer-contract churn", f"{longterm_rate:.1%}")
    c4.metric("First-6-month churn", f"{early_churn:.1%}")

    st.markdown("---")
    if os.path.exists(COMPARISON_PATH):
        st.subheader("Model comparison")
        comp = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comp.style.format({
            "precision": "{:.3f}", "recall": "{:.3f}",
            "f1": "{:.3f}", "roc_auc": "{:.3f}"
        }), use_container_width=True)

    st.subheader("Key findings")
    st.markdown(
        f"""
        - Month-to-month contracts churn at **{mtm_rate:.1%}** vs **{longterm_rate:.1%}**
          for longer contracts — the single biggest retention lever.
        - The first 6 months are the highest-risk window (**{early_churn:.1%}** churn).
        - See the **EDA** tab for the full interactive breakdown, including a
          counter-intuitive add-on-services finding worth digging into further.
        """
    )

# ---------------- EDA ----------------
with tab_eda:
    st.subheader("Churn rate by contract type")
    rate_by_contract = df.groupby("Contract")["ChurnFlag"].mean().reset_index()
    fig = px.bar(rate_by_contract, x="Contract", y="ChurnFlag",
                 labels={"ChurnFlag": "Churn rate"}, color="Contract")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tenure vs churn")
        fig2 = px.histogram(df, x="tenure", color="Churn", barmode="overlay",
                             nbins=30, opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.subheader("Churn rate by payment method")
        rate_by_pay = df.groupby("PaymentMethod")["ChurnFlag"].mean().reset_index()
        fig3 = px.bar(rate_by_pay.sort_values("ChurnFlag", ascending=False),
                       x="PaymentMethod", y="ChurnFlag",
                       labels={"ChurnFlag": "Churn rate"})
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Monthly charges by contract type and churn status")
    fig4 = px.box(df, x="Contract", y="MonthlyCharges", color="Churn")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------- Predict ----------------
with tab_predict:
    st.subheader("Score a hypothetical customer")
    st.caption("Fill in customer details to get a live churn probability from the trained model.")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 65.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with c2:
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Has partner", ["Yes", "No"])
        with c3:
            dependents = st.selectbox("Has dependents", ["Yes", "No"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            num_addons = st.slider("Number of add-on services", 0, 6, 1)

        submitted = st.form_submit_button("Predict churn probability")

    if submitted:
        total_charges = tenure * monthly_charges
        revenue_per_tenure = total_charges / max(tenure, 1)
        median_charge = df["MonthlyCharges"].median()
        mtm_high = int(contract == "Month-to-month" and monthly_charges > median_charge)

        row = pd.DataFrame([{
            "tenure": tenure, "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges, "RevenuePerTenureMonth": revenue_per_tenure,
            "NumAddonServices": num_addons,
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": internet, "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthToMonthHighCharge": mtm_high,
        }])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        proba = predict_one(model_obj, row)
        tier = "High" if proba >= 0.6 else "Medium" if proba >= 0.3 else "Low"
        color = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[tier]

        st.metric("Predicted churn probability", f"{proba:.1%}")
        st.markdown(f"**Risk tier:** {color} {tier}")

# ---------------- Risk list ----------------
with tab_risk:
    st.subheader("Prioritized outreach list")
    tier_filter = st.multiselect(
        "Filter by risk tier", options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    filtered = outreach_df[outreach_df["RiskTier"].isin(tier_filter)]
    st.write(f"Showing {len(filtered)} of {len(outreach_df)} customers")
    st.dataframe(
        filtered[["ChurnProba", "RiskTier", "MonthlyCharges", "tenure", "Contract"]]
        .sort_values("ChurnProba", ascending=False),
        use_container_width=True,
    )
    st.download_button(
        "Download filtered list as CSV",
        filtered.to_csv(index=False),
        "filtered_outreach_list.csv",
    )
