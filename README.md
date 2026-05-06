---
title: Fraud Decision Explainer
emoji: 🔍
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.28.0
app_file: app.py
pinned: false
license: mit
---

# Fraud Decision Explainer

XGBoost fraud classifier with a three-layer explainability stack:

| Layer | What it does |
|---|---|
| **XGBoost** | Binary fraud classifier (AUC ~0.98 on Kaggle CC dataset) |
| **SHAP** | Per-feature attribution — which signals drove this decision |
| **RAG (FAISS)** | Retrieves the 3 most similar confirmed fraud cases from training history |
| **LLM (Zephyr-7B)** | Synthesizes all signals into a plain-language analyst narrative |

## Quick start

```bash
pip install -r requirements.txt

# Option A: use synthetic data (no Kaggle account)
make demo

# Option B: use real Kaggle data
kaggle datasets download mlg-ulb/creditcardfraud -p data --unzip
make train
make app
```

## API (FastAPI / Render)

```bash
make api
# → http://localhost:8000/docs
```

**POST /predict**
```json
{
  "Amount": 149.62,
  "Time": 0,
  "V1": -1.3598,
  "V14": -0.3112,
  ...
}
```

Response:
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9741,
  "risk_level": "HIGH",
  "shap_explanation": { "features": [...] },
  "similar_cases": [...],
  "llm_explanation": "This $149.62 transaction..."
}
```

## Deployment

### Hugging Face Spaces (free)

```bash
# Create a new Space at huggingface.co/new-space (SDK: Gradio)
git remote add space https://huggingface.co/spaces/<your-username>/<space-name>

# Train locally, commit artifacts, push
make train
git add artifacts/ -f
git commit -m "add trained artifacts"
git push space main
```

Set `HF_TOKEN` in Space → Settings → Secrets to enable generative LLM explanations.

### Render (free tier)

1. Connect your GitHub repo to Render
2. New Web Service → Docker → port `8000`
3. Add env var `HF_TOKEN` in the Render dashboard

## LLM without a token

If `HF_TOKEN` is not set, the app falls back to a deterministic template that still incorporates SHAP values and similar cases — good enough for demos.

## Dataset

- **Real data:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284k transactions, 0.17% fraud)
- **Synthetic data:** `python scripts/generate_demo_data.py` — mimics the V1–V28 PCA structure with injected fraud signal in V14/V12
