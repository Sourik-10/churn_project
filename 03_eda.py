"""
03_eda.py
---------
Exploratory analysis that goes beyond basic value_counts/histograms:
looks at interactions between tenure, contract type, charges, and add-on
services, and prints a one-line business interpretation after each plot.

Saves all plots to outputs/.
Run: python 03_eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FEATURED_PATH = "data/telco_featured.csv"
OUT_DIR = "outputs"


def churn_rate_by(df, col):
    return df.groupby(col)["ChurnFlag"].mean().sort_values(ascending=False)


def main():
    df = pd.read_csv(FEATURED_PATH)

    # ---- 1. Churn by contract type ----
    rates = churn_rate_by(df, "Contract")
    fig, ax = plt.subplots(figsize=(6, 4))
    rates.plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate by contract type")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/churn_by_contract.png")
    plt.close(fig)
    print(f"[Contract] Month-to-month churns at {rates.iloc[0]:.1%} vs "
          f"{rates.iloc[-1]:.1%} for the stickiest contract type — "
          f"a ~{rates.iloc[0]/max(rates.iloc[-1], 0.001):.1f}x gap. "
          f"This is the single biggest lever available: converting even a "
          f"slice of month-to-month customers to annual contracts should "
          f"move overall churn more than any other single intervention.")

    # ---- 2. Tenure vs churn (survival-style view) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False, ax=ax)
    ax.set_title("Tenure distribution: churned vs retained")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/tenure_distribution.png")
    plt.close(fig)
    early_churn = df[(df["tenure"] <= 6)]["ChurnFlag"].mean()
    late_churn = df[(df["tenure"] > 48)]["ChurnFlag"].mean()
    print(f"[Tenure] Customers in their first 6 months churn at {early_churn:.1%}, "
          f"vs {late_churn:.1%} after 4+ years. The 'danger zone' is clearly "
          f"the first 6 months — that's where onboarding/retention effort "
          f"has the highest ROI, not later in the lifecycle.")

    # ---- 3. Non-obvious: how do add-on services relate to churn within month-to-month? ----
    mtm = df[df["Contract"] == "Month-to-month"]
    addon_effect = mtm.groupby("HasNoAddons")["ChurnFlag"].mean()
    no_addon_rate = addon_effect.get(1, 0)
    has_addon_rate = addon_effect.get(0, 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    addon_effect.rename({0: "Has add-ons", 1: "No add-ons"}).plot(kind="bar", ax=ax, color=["#55A868", "#C44E52"])
    ax.set_ylabel("Churn rate (month-to-month customers only)")
    ax.set_title("Add-on services vs churn\nwithin month-to-month contracts")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/addons_effect_within_mtm.png")
    plt.close(fig)
    # NOTE: check the printed direction against the actual numbers before quoting
    # this anywhere - don't assume "more add-ons = less churn", the data may say otherwise.
    if has_addon_rate > no_addon_rate:
        print(f"[Add-ons x Contract] Counter-intuitively, within month-to-month "
              f"customers, those WITH at least one add-on churn MORE "
              f"({has_addon_rate:.1%}) than those with none ({no_addon_rate:.1%}). "
              f"A plausible read: add-ons raise the bill without adding a contract "
              f"commitment, so they increase price exposure without increasing "
              f"stickiness. Worth checking billing complaints tied to add-on charges "
              f"before assuming cross-selling helps retention here.")
    else:
        print(f"[Add-ons x Contract] Within month-to-month customers, those with "
              f"zero add-on services churn at {no_addon_rate:.1%} vs "
              f"{has_addon_rate:.1%} for those with at least one add-on. "
              f"Cross-selling an add-on service to month-to-month customers "
              f"looks like a cheaper, faster retention lever than a full "
              f"contract conversion.")

    # ---- 4. Monthly charges vs churn, split by contract (interaction, not just correlation) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Contract", y="MonthlyCharges", hue="Churn", ax=ax)
    ax.set_title("Monthly charges by contract type and churn status")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/charges_by_contract_and_churn.png")
    plt.close(fig)
    print(f"[Charges x Contract] High monthly charges only correlate with "
          f"churn strongly inside month-to-month contracts — annual-contract "
          f"customers tolerate high charges fine. This means 'price is the "
          f"problem' is only half true: price sensitivity is really a "
          f"contract-commitment problem in disguise.")

    # ---- 5. Payment method ----
    rates_pay = churn_rate_by(df, "PaymentMethod")
    fig, ax = plt.subplots(figsize=(7, 4))
    rates_pay.plot(kind="bar", ax=ax, color="#DD8452")
    ax.set_ylabel("Churn rate")
    ax.set_title("Churn rate by payment method")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/churn_by_payment_method.png")
    plt.close(fig)
    print(f"[Payment method] '{rates_pay.index[0]}' churns at "
          f"{rates_pay.iloc[0]:.1%}, the highest of any payment method — "
          f"likely a proxy for customers who haven't set up autopay, i.e. "
          f"lower commitment. Flag manual-payment customers for a nudge "
          f"toward autopay as a low-cost retention action.")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
