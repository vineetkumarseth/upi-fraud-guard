# UPI Fraud Guard

**Decode SIH 2026 — Bharat Pragati Track (PS2: Intelligent UPI Fraud Prevention)**

An explainable, real-time UPI transaction fraud detection system with an
AI agent pipeline that blends a trained ML model, live web-signal search,
and automated alerting.

---

## What this does

1. A transaction comes in (amount, payee, time, location, account history).
2. An **XGBoost classifier** scores it for fraud risk and explains *why*
   using **SHAP** (top contributing features, not just a black-box number).
3. A **Swytchcode AI agent** orchestrates the decision: it enriches the
   model score with a **Tavily** real-time web search (checking if the
   payee has recent scam reports) and blends both into a final score.
4. If the blended score crosses a threshold, an **n8n** webhook fires an
   automated alert (email/Slack/SMS — configurable entirely inside n8n,
   no backend code changes needed).
5. Everything is deployed on **Render** as two services (API + dashboard).
6. **Codemate.ai** was used during development for debugging/review.
7. **Startuped.ai** is used for GTM/idea-validation — see `docs/gtm.md`.

## Project structure

```
upi-fraud-guard/
├── ml/
│   └── train_model.py        # trains the XGBoost fraud model on synthetic data
├── backend/
│   ├── main.py                # FastAPI app: /predict, /agent/evaluate, /health
│   ├── requirements.txt
│   └── model_artifacts/       # created after training (model + SHAP explainer)
├── integrations/
│   ├── tavily_client.py       # Tavily — AI Search Partner
│   ├── n8n_alerts.py          # n8n — Automation Partner
│   └── swytchcode_agent.py    # Swytchcode — AI Integration Partner (agent orchestration)
├── frontend/
│   └── index.html             # live dashboard (single file, no build step)
├── render.yaml                 # Render Blueprint — 2 services (web + static)
├── .env.example
└── README.md
```

## Running it locally

```bash
# 1. Install dependencies
cd upi-fraud-guard
pip install -r backend/requirements.txt

# 2. Train the model (creates backend/model_artifacts/)
cd ml
python train_model.py
cd ..

# 3. Set your API keys (optional — app works without them, just skips
#    the live web-signal and alert steps)
cp .env.example .env
# edit .env and add TAVILY_API_KEY / N8N_WEBHOOK_URL

# 4. Start the backend
cd backend
uvicorn main:app --reload --port 8000

# 5. Open the dashboard
# just open frontend/index.html in your browser (or serve it with
# `python -m http.server 5500` from inside the frontend/ folder)
```

The dashboard's API_BASE defaults to `http://localhost:8000` — update the
line near the top of the `<script>` in `index.html` once you deploy to
Render.

## Deploying to Render

1. Push this repo to GitHub.
2. In Render: **New +** → **Blueprint** → connect your repo. Render will
   read `render.yaml` and create both services automatically.
3. Add your `TAVILY_API_KEY` and `N8N_WEBHOOK_URL` as environment
   variables on the `upi-fraud-guard-api` service (Render dashboard →
   service → Environment).
4. Once deployed, update `API_BASE` in `frontend/index.html` to your
   Render API URL, and redeploy the static site.

## Swapping in a real dataset

`ml/train_model.py` currently generates a synthetic dataset because real
UPI transaction data isn't public. To use a real dataset (e.g. Kaggle's
IEEE-CIS Fraud Detection, or your own CSV):

1. Load your CSV into a DataFrame with the same columns listed in
   `FEATURE_COLUMNS` (or adjust the list to match your columns).
2. Replace the call to `generate_synthetic_data()` in `train()` with your
   loader.
3. Re-run `python train_model.py`.

## What to say in your PPT / demo

- **Differentiator #1:** not just "fraud/not fraud" — pattern-based
  features (spending spike vs. 7-day average, sudden location jump,
  transaction burst) plus SHAP explainability so judges see *why* each
  call was made.
- **Differentiator #2:** live web-signal enrichment via Tavily, so the
  system isn't only reacting to historical patterns — it checks if a
  payee has active scam reports right now.
- **Differentiator #3:** automated response loop via n8n — detection
  triggers action, not just a dashboard flag.

## Sponsor integration checklist

| Sponsor | Where it's used | Track requirement met |
|---|---|---|
| Swytchcode | `integrations/swytchcode_agent.py` — orchestrates the AI agent workflow | AI agent + 2 external APIs (Tavily, n8n) ✅ |
| Codemate.ai | Used during development (debugging/refactoring) | Used throughout dev lifecycle |
| Tavily | `integrations/tavily_client.py` — real-time scam/fraud web search | Real-time retrieval integrated ✅ |
| n8n | `integrations/n8n_alerts.py` — webhook-triggered alert workflow | Automation demonstrably improves UX ✅ |
| Render | `render.yaml` — Web Service + Static Site | 2+ service types ✅ |
| Startuped.ai | GTM validation — see `docs/gtm.md` | Idea validation + GTM plan ✅ |

---

Built for Decode SIH 2026.
