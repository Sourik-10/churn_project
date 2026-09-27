"""
02_feature_engineering.py
--------------------------
Adds engineered features on top of the cleaned data:
- tenure buckets
- avg revenue per month of tenure (a proxy for pricing sensitivity)
- number of add-on services subscribed (engagement proxy)
- a simple interaction flag: month-to-month + high monthly charges

Run: python 02_feature_engineering.py
"""

import pandas as pd

CLEAN_PATH = "data/telco_clean.csv"
FEATURED_PATH = "data/telco_featured.csv"

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Tenure buckets - easier to read in EDA / business summaries than raw months
    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "7-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )

    # Revenue per month of tenure - flags customers paying a lot relative
    # to how new they are (higher price shock risk)
    df["RevenuePerTenureMonth"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

    # Count of add-on services actually subscribed (Yes vs No/No internet service)
    df["NumAddonServices"] = (df[ADDON_COLS] == "Yes").sum(axis=1)

    # Customers with zero add-ons are typically the least "sticky"
    df["HasNoAddons"] = (df["NumAddonServices"] == 0).astype(int)

    # Interaction: month-to-month contract AND above-median monthly charges
    median_charge = df["MonthlyCharges"].median()
    df["MonthToMonthHighCharge"] = (
        (df["Contract"] == "Month-to-month") & (df["MonthlyCharges"] > median_charge)
    ).astype(int)

    return df


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    df = add_features(df)
    df.to_csv(FEATURED_PATH, index=False)
    print(f"Added {5} new features. Saved to {FEATURED_PATH}")
    print(df[["TenureBucket", "NumAddonServices", "MonthToMonthHighCharge"]].describe(include="all"))
