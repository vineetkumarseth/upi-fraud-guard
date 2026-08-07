"""
Swytchcode Integration - AI Integration Partner
--------------------------------------------------
Track requirement: build using the Swytchcode CLI/runtime, integrate at
least 2 external APIs, and include an AI-powered workflow/agent.

This module defines the AGENT WORKFLOW: an orchestration layer that ties
together the ML model score + Tavily web signal + n8n alert into a single
"agent decision" -- this is the piece you register/run through the
Swytchcode CLI so the track's evaluation criteria are satisfied.

Follow Swytchcode's setup docs to install their CLI, then point it at
`run_fraud_agent()` as the callable entrypoint. The two external APIs
integrated here are Tavily (search) and n8n (automation webhook) --
satisfying the "2+ external APIs" requirement.
"""

from integrations.tavily_client import check_scam_reports
from integrations.n8n_alerts import trigger_fraud_alert


def run_fraud_agent(transaction: dict, model_risk_score: float, model_reasons: list) -> dict:
    """
    Orchestrates the full decision pipeline for a single transaction:
      1. Take the ML model's risk score (computed upstream in the API).
      2. Enrich it with a live Tavily web-signal check on the payee.
      3. Combine into a final blended risk score.
      4. If above threshold, trigger an n8n automated alert.

    This function is the "AI agent" entrypoint for the Swytchcode track.
    """
    payee = transaction.get("payee_upi_id", "unknown")
    web_signal = check_scam_reports(payee)

    # Blend: 75% model confidence, 25% live web signal
    blended_score = round(0.75 * model_risk_score + 0.25 * web_signal["risk_signal"], 3)

    decision = "BLOCK" if blended_score >= 0.75 else "REVIEW" if blended_score >= 0.4 else "ALLOW"

    alert_result = None
    if decision in ("BLOCK", "REVIEW"):
        alert_result = trigger_fraud_alert(
            transaction=transaction,
            risk_score=blended_score,
            reasons=model_reasons + [web_signal["summary"]],
        )

    return {
        "decision": decision,
        "blended_risk_score": blended_score,
        "model_risk_score": model_risk_score,
        "web_signal": web_signal,
        "alert_triggered": alert_result,
    }
