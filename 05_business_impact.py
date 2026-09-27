"""
05_business_impact.py
----------------------
This is the "differentiator" piece most tutorial churn projects skip:
instead of stopping at a 0.5 probability cutoff, this finds the cutoff
that actually maximizes money saved, given real campaign economics.

Assumptions (edit these for your own numbers):
- CAMPAIGN_COST: cost of a retention outreach (email/call/discount) per customer
- AVG_CLV: average lifetime value lost when a customer we *don't* target churns

Logic per threshold t:
- Customers with predicted churn probability >= t get a retention offer
- True positive (correctly caught churner): we save them -> gain AVG_CLV, pay CAMPAIGN_COST
- False positive (offered to a non-churner): wasted cost -> pay CAMPAIGN_COST, no gain
- False negative (missed churner): full loss -> lose AVG_CLV
- True negative: no cost, no loss

Net value(t) = TP(t)*(AVG_CLV - CAMPAIGN_COST) - FP(t)*CAMPAIGN_COST - FN(t)*AVG_CLV

Also segments customers into Low/Medium/High risk tiers for a
business-facing prioritized outreach list.

Run: python 05_business_impact.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
TEST_PRED_PATH = f"{OUT_DIR}/test_predictions.csv"

# ---- Business assumptions - change these to match a real scenario ----
CAMPAIGN_COST = 15.0     # cost per customer of a retention offer (e.g. 1 month discount)
AVG_CLV = 500.0          # avg remaining lifetime value of a customer if retained


def net_value_at_threshold(y_true, y_proba, threshold, campaign_cost, avg_clv):
    pred = (y_proba >= threshold).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    tn = ((pred == 0) & (y_true == 0)).sum()
    value = tp * (avg_clv - campaign_cost) - fp * campaign_cost - fn * avg_clv
    return value, tp, fp, fn, tn


def optimize_threshold(y_true, y_proba, campaign_cost, avg_clv):
    thresholds = np.arange(0.05, 0.95, 0.01)
    records = []
    for t in thresholds:
        value, tp, fp, fn, tn = net_value_at_threshold(y_true, y_proba, t, campaign_cost, avg_clv)
        records.append({"threshold": t, "net_value": value, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(records)


def assign_risk_tier(proba):
    if proba >= 0.6:
        return "High"
    elif proba >= 0.3:
        return "Medium"
    else:
        return "Low"


def main():
    df = pd.read_csv(TEST_PRED_PATH)
    y_true = df["ChurnFlag"].values
    y_proba = df["ChurnProba"].values

    # --- Threshold optimization ---
    curve = optimize_threshold(y_true, y_proba, CAMPAIGN_COST, AVG_CLV)
    best_row = curve.loc[curve["net_value"].idxmax()]

    # Compare against naive 0.5 threshold
    naive_value, *_ = net_value_at_threshold(y_true, y_proba, 0.5, CAMPAIGN_COST, AVG_CLV)

    print("=== Cost-based threshold optimization ===")
    print(f"Assumptions: campaign cost = {CAMPAIGN_COST}, avg CLV saved = {AVG_CLV}")
    print(f"Naive 0.5 threshold  -> net value: {naive_value:,.2f}")
    print(f"Optimal threshold {best_row['threshold']:.2f} -> net value: {best_row['net_value']:,.2f} "
          f"(TP={int(best_row['tp'])}, FP={int(best_row['fp'])}, FN={int(best_row['fn'])})")
    improvement = best_row["net_value"] - naive_value
    print(f"Switching from 0.5 to the optimal threshold improves net campaign value by "
          f"{improvement:,.2f} on this test set alone.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["threshold"], curve["net_value"])
    ax.axvline(best_row["threshold"], color="red", linestyle="--", label=f"Optimal = {best_row['threshold']:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="Naive 0.5")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Net campaign value ($)")
    ax.set_title("Net value vs. decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/threshold_optimization.png")
    plt.close(fig)

    # --- Risk tier segmentation for a business-facing outreach list ---
    df["RiskTier"] = df["ChurnProba"].apply(assign_risk_tier)
    tier_summary = df.groupby("RiskTier").agg(
        customers=("ChurnFlag", "count"),
        actual_churn_rate=("ChurnFlag", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
    ).reindex(["High", "Medium", "Low"])

    print("\n=== Risk tier summary (for a prioritized outreach list) ===")
    print(tier_summary.to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    tier_summary["customers"].plot(kind="bar", ax=ax, color=["#C44E52", "#DD8452", "#55A868"])
    ax.set_ylabel("Number of customers")
    ax.set_title("Customers by risk tier")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/risk_tier_counts.png")
    plt.close(fig)

    # Save the prioritized list itself - this is the actual deliverable a growth team would use
    priority_list = df.sort_values("ChurnProba", ascending=False)
    priority_list.to_csv(f"{OUT_DIR}/prioritized_outreach_list.csv", index=False)
    print(f"\nSaved prioritized outreach list to {OUT_DIR}/prioritized_outreach_list.csv")
    print(f"Saved threshold curve and risk tier chart to {OUT_DIR}/")


if __name__ == "__main__":
    main()
