"""
01_clean_data.py
-----------------
Loads the raw Telco Customer Churn CSV, fixes data types, handles missing
values, and saves a cleaned version for the rest of the pipeline to use.

Run: python 01_clean_data.py
"""

import pandas as pd
import numpy as np

RAW_PATH = "data/Telco-Customer-Churn.csv"
CLEAN_PATH = "data/telco_clean.csv"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # TotalCharges is read as object because a few rows have blank strings
    # instead of NaN (new customers with 0 tenure). Coerce to numeric.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_missing = df["TotalCharges"].isna().sum()
    print(f"Rows with missing TotalCharges (new customers, tenure=0): {n_missing}")

    # These are customers who just joined -> TotalCharges should be 0, not NaN
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0

    # Standardize target to 0/1 int, keep a readable copy too
    df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

    # SeniorCitizen is already 0/1 but stored as int64 - fine as is.
    # Drop customerID from modeling later (kept here for traceability).

    # Quick sanity checks
    assert df["ChurnFlag"].isin([0, 1]).all()
    assert df["MonthlyCharges"].min() > 0

    print(f"Final shape: {df.shape}")
    print(f"Churn rate: {df['ChurnFlag'].mean():.1%}")

    return df


if __name__ == "__main__":
    df = load_and_clean(RAW_PATH)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data to {CLEAN_PATH}")
