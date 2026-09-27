"""
04_modeling.py
--------------
Trains Logistic Regression, Random Forest, and XGBoost on the churn dataset.
Compares two imbalance-handling strategies (class_weight vs SMOTE) and
evaluates everything with precision/recall/F1/ROC-AUC - not just accuracy,
since churn is an imbalanced problem (~26% positive class) where accuracy
is misleading (a model that predicts "no churn" for everyone still scores
~73% accuracy while being useless).

Saves:
- outputs/model_comparison.csv   (the full metrics table)
- outputs/confusion_matrix_best.png
- outputs/roc_curves.png
- models/best_model.pkl

Run: python 04_modeling.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

FEATURED_PATH = "data/telco_featured.csv"
OUT_DIR = "outputs"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

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


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def main():
    df = pd.read_csv(FEATURED_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["ChurnFlag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost_class_weight": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted_pipelines = {}

    # --- Strategy A: class_weight / scale_pos_weight (no resampling) ---
    for name, clf in models.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results.append(evaluate(name, y_test, y_pred, y_proba))
        fitted_pipelines[name] = pipe

    # --- Strategy B: SMOTE oversampling, fit on top of XGBoost (best base model) ---
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_enc, y_train)

    xgb_smote = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=RANDOM_STATE,
    )
    xgb_smote.fit(X_train_sm, y_train_sm)
    y_pred_sm = xgb_smote.predict(X_test_enc)
    y_proba_sm = xgb_smote.predict_proba(X_test_enc)[:, 1]
    results.append(evaluate("XGBoost_SMOTE", y_test, y_pred_sm, y_proba_sm))

    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    comparison.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(comparison.to_string(index=False))

    # --- Pick best model by ROC-AUC, save it + confusion matrix + ROC curves ---
    best_name = comparison.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    if best_name == "XGBoost_SMOTE":
        best_pred, best_proba = y_pred_sm, y_proba_sm
        joblib.dump({"preprocessor": preprocessor, "model": xgb_smote}, f"{MODEL_DIR}/best_model.pkl")
    else:
        best_pipe = fitted_pipelines[best_name]
        best_pred = best_pipe.predict(X_test)
        best_proba = best_pipe.predict_proba(X_test)[:, 1]
        joblib.dump(best_pipe, f"{MODEL_DIR}/best_model.pkl")

    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/confusion_matrix_best.png")
    plt.close(fig)

    # ROC curves for all class_weight-based models + SMOTE model
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, pipe in fitted_pipelines.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=name)
    fpr, tpr, _ = roc_curve(y_test, y_proba_sm)
    ax.plot(fpr, tpr, label="XGBoost_SMOTE")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves - all models")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/roc_curves.png")
    plt.close(fig)

    print(f"\nSaved comparison table, confusion matrix, and ROC curves to {OUT_DIR}/")
    print(f"Saved best model to {MODEL_DIR}/best_model.pkl")

    # Save test set + best probabilities for the business-impact script
    test_out = X_test.copy()
    test_out["ChurnFlag"] = y_test.values
    test_out["ChurnProba"] = best_proba
    test_out.to_csv(f"{OUT_DIR}/test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
