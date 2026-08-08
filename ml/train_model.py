"""
UPI Fraud Detection - Model Training Script
--------------------------------------------
Generates a realistic synthetic UPI transaction dataset (since real UPI data
is not public) and trains a gradient-boosted fraud classifier.

Swap `generate_synthetic_data()` with a real dataset loader (e.g. Kaggle's
IEEE-CIS Fraud Detection, or your own UPI-simulated CSV) when you have one --
just make sure the output columns match FEATURE_COLUMNS below.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import shap
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "..", "backend", "model_artifacts")

FEATURE_COLUMNS = [
    "amount",
    "avg_amount_7d",
    "txn_count_1h",
    "txn_count_24h",
    "hour_of_day",
    "is_new_payee",
    "distance_from_usual_location_km",
    "amount_to_avg_ratio",
    "account_age_days",
    "is_night_txn",
]


def generate_synthetic_data(n_samples: int = 20000, fraud_rate: float = 0.03) -> pd.DataFrame:
    """Creates a synthetic but behaviourally realistic UPI transaction dataset."""
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud

    # ---- Legitimate transactions: consistent behaviour ----
    legit = pd.DataFrame({
        "amount": np.random.gamma(2, 400, n_legit).clip(10, 20000),
        "avg_amount_7d": np.random.gamma(2, 380, n_legit).clip(10, 15000),
        "txn_count_1h": np.random.poisson(0.3, n_legit),
        "txn_count_24h": np.random.poisson(3, n_legit),
        "hour_of_day": np.random.choice(range(24), n_legit,
                                         p=_hour_weights(day_heavy=True)),
        "is_new_payee": np.random.binomial(1, 0.08, n_legit),
        "distance_from_usual_location_km": np.random.exponential(3, n_legit).clip(0, 50),
        "account_age_days": np.random.randint(30, 2000, n_legit),
        "label": 0,
    })

    # ---- Fraudulent transactions: spikes, odd hours, new payees, distance ----
    fraud = pd.DataFrame({
        "amount": np.random.gamma(4, 3000, n_fraud).clip(500, 200000),
        "avg_amount_7d": np.random.gamma(2, 400, n_fraud).clip(10, 15000),
        "txn_count_1h": np.random.poisson(2.5, n_fraud),
        "txn_count_24h": np.random.poisson(9, n_fraud),
        "hour_of_day": np.random.choice(range(24), n_fraud,
                                         p=_hour_weights(day_heavy=False)),
        "is_new_payee": np.random.binomial(1, 0.75, n_fraud),
        "distance_from_usual_location_km": np.random.exponential(80, n_fraud).clip(0, 3000),
        "account_age_days": np.random.randint(1, 2000, n_fraud),
        "label": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE)
    df["amount_to_avg_ratio"] = (df["amount"] / df["avg_amount_7d"].replace(0, 1)).clip(0, 200)
    df["is_night_txn"] = df["hour_of_day"].apply(lambda h: 1 if (h >= 23 or h <= 4) else 0)
    return df.reset_index(drop=True)


def _hour_weights(day_heavy: bool) -> np.ndarray:
    weights = np.ones(24)
    if day_heavy:
        weights[9:21] *= 4  # legit txns cluster in daytime
    else:
        weights[0:5] *= 3  # fraud skews toward odd night hours
        weights[9:21] *= 1.2
    return weights / weights.sum()


def train():
    print("Generating synthetic dataset...")
    df = generate_synthetic_data()

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Training XGBoost classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))

    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_ARTIFACTS_DIR, "fraud_model.joblib"))

    print(f"Saved model to {MODEL_ARTIFACTS_DIR}/")


if __name__ == "__main__":
    train()
