"""
UPI Fraud Guard - Backend API
--------------------------------
FastAPI service exposing:
  POST /predict        -> ML risk score + SHAP explanation for one transaction
  POST /agent/evaluate  -> full pipeline: model score -> Tavily enrich -> n8n alert
  GET  /health         -> liveness check (used by Render)

Run locally:  uvicorn main:app --reload --port 8000
"""

import sys
import os
import joblib
import shap
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv  # noqa: E402

# Load variables from the .env file at the project root (one level up from backend/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from integrations.swytchcode_agent import run_fraud_agent  # noqa: E402

app = FastAPI(title="UPI Fraud Guard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts", "fraud_model.joblib")
EXPLAINER_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts", "shap_explainer.joblib")

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

_model = None
_explainer = None


def get_model():
    global _model, _explainer
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Model artifacts not found. Run `python ml/train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
        # Built fresh from the model instead of loaded from disk — a pickled
        # SHAP explainer isn't portable across Python versions, which was
        # exactly the crash we just saw. Rebuilding costs milliseconds.
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


class Transaction(BaseModel):
    payee_upi_id: str = Field(..., example="merchant@upi")
    amount: float = Field(..., example=15000)
    avg_amount_7d: float = Field(..., example=800)
    txn_count_1h: int = Field(..., example=1)
    txn_count_24h: int = Field(..., example=3)
    hour_of_day: int = Field(..., ge=0, le=23, example=2)
    is_new_payee: int = Field(..., ge=0, le=1, example=1)
    distance_from_usual_location_km: float = Field(..., example=450.0)
    account_age_days: int = Field(..., example=200)


def _build_features(txn: Transaction) -> pd.DataFrame:
    row = txn.dict()
    row["amount_to_avg_ratio"] = min(row["amount"] / max(row["avg_amount_7d"], 1), 200)
    row["is_night_txn"] = 1 if (row["hour_of_day"] >= 23 or row["hour_of_day"] <= 4) else 0
    return pd.DataFrame([{col: row[col] for col in FEATURE_COLUMNS}])


def _top_reasons(explainer, X_row: pd.DataFrame, top_n: int = 3) -> list:
    shap_values = explainer.shap_values(X_row)
    vals = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    contributions = list(zip(FEATURE_COLUMNS, vals))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    reasons = []
    for feature, val in contributions[:top_n]:
        direction = "increased" if val > 0 else "decreased"
        reasons.append(f"{feature} {direction} risk (impact={val:.3f})")
    return reasons


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(txn: Transaction):
    model, explainer = get_model()
    X_row = _build_features(txn)
    risk_score = float(model.predict_proba(X_row)[0, 1])
    reasons = _top_reasons(explainer, X_row)
    return {
        "risk_score": round(risk_score, 4),
        "is_fraud_predicted": risk_score >= 0.5,
        "top_reasons": reasons,
    }


@app.post("/agent/evaluate")
def agent_evaluate(txn: Transaction):
    """Full pipeline: ML score -> Swytchcode agent (Tavily enrich + n8n alert)."""
    model, explainer = get_model()
    X_row = _build_features(txn)
    risk_score = float(model.predict_proba(X_row)[0, 1])
    reasons = _top_reasons(explainer, X_row)

    agent_result = run_fraud_agent(
        transaction=txn.dict(),
        model_risk_score=risk_score,
        model_reasons=reasons,
    )
    return agent_result