# 📉 Customer Churn Analysis & Retention Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end churn analytics project on the IBM Telco Customer Churn dataset
(7,043 customers, 21 features): data cleaning → feature engineering → EDA →
predictive modeling → **cost-based business impact optimization** → an
interactive Streamlit dashboard for live predictions and prioritized retention
outreach.

> 🎯 **Why this project is different from a typical churn tutorial:** most
> churn projects stop at model accuracy. This one goes one step further and
> asks *"what decision threshold actually maximizes retention campaign ROI,
> given real campaign costs and customer lifetime value?"* — that's the
> question a business actually cares about.

---

## 🖥️ Live Demo

*(Add your Streamlit Cloud link here once deployed)*

## 📸 Screenshots

*(Add 2-3 screenshots of the dashboard here — Overview tab, Predict tab, and Risk List tab work best)*

---

## 🧠 The Business Problem

In subscription businesses (OTT, SaaS, telecom), acquiring a new customer
costs far more than retaining an existing one. This project simulates a
Data Analyst / ML task: **identify which customers are at high risk of
churning, understand why, and quantify the financial impact of acting on
that risk before it happens.**

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| Visualization | matplotlib, seaborn, Plotly |
| Dashboard | Streamlit |
| Dataset | [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) (7,043 rows) |

## 📁 Project Structure
