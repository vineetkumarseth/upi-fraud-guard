"""
n8n Integration - Automation Partner
--------------------------------------
When a transaction is flagged as fraud, this fires a webhook into an n8n
workflow, which can then fan out to email / Slack / SMS / dashboard alerts
without any of that fan-out logic living in our own backend.

Setup:
1. In n8n, create a workflow starting with a "Webhook" trigger node.
2. Copy its Production URL and set it as N8N_WEBHOOK_URL below (env var).
3. Downstream nodes in n8n can send Email / Slack / Telegram / SMS alerts,
   log to a Google Sheet, or open a ticket -- all configurable without
   touching this backend code again.
"""

import os
import requests
from datetime import datetime, timezone

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "")


def trigger_fraud_alert(transaction: dict, risk_score: float, reasons: list) -> dict:
    if not N8N_WEBHOOK_URL:
        return {"sent": False, "reason": "N8N_WEBHOOK_URL not configured"}

    payload = {
        "event": "fraud_alert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction": transaction,
        "risk_score": risk_score,
        "reasons": reasons,
    }

    try:
        resp = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=5)
        return {"sent": resp.ok, "status_code": resp.status_code}
    except Exception as e:
        return {"sent": False, "reason": str(e)}
