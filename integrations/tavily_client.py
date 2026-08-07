"""
Tavily Integration - AI Search Partner
----------------------------------------
Used to pull real-time context on whether a UPI ID / payee name / pattern
has recent scam reports, enriching the model's risk score with live web
signal instead of relying purely on historical transaction features.

Get your API key from: https://tavily.com  (participant credits via
Decode SIH referral link -> Tavily gives 8,000 API credits per member)

Set env var: TAVILY_API_KEY
"""

import os
from typing import Optional
from tavily import TavilyClient  # pip install tavily-python


def check_scam_reports(payee_identifier: str) -> dict:
    """
    Searches for recent scam/fraud reports associated with a UPI ID or
    payee name. Returns a lightweight risk signal to blend into the model
    score, plus the sources for transparency in the dashboard.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "risk_signal": 0.0,
            "summary": "Tavily API key not configured (set TAVILY_API_KEY).",
            "sources": [],
        }

    query = f'"{payee_identifier}" UPI scam OR fraud report complaint'
    try:
        client = TavilyClient(api_key=api_key)
        result = client.search(query=query, search_depth="basic", max_results=5)
        hits = result.get("results", [])

        # crude heuristic: more matching results = higher live risk signal
        risk_signal = min(len(hits) / 5.0, 1.0)

        return {
            "enabled": True,
            "risk_signal": round(risk_signal, 2),
            "summary": f"Found {len(hits)} web result(s) referencing this identifier "
                       f"alongside scam/fraud-related terms.",
            "sources": [{"title": h.get("title"), "url": h.get("url")} for h in hits],
        }
    except Exception as e:
        # Catches: invalid key, SDK version mismatches, network errors, rate limits.
        # Never let a Tavily hiccup crash the whole fraud-check request.
        return {"enabled": True, "risk_signal": 0.0, "summary": f"Tavily lookup failed: {e}", "sources": []}