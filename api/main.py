"""FastAPI service: fraud prediction + SHAP + RAG + LLM explanation."""
import json
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.drift import check_drift
from src.explain import FEATURE_COLS, SHAPExplainer
from src.llm import generate_explanation
from src.rag import FraudCaseRetriever

ARTIFACTS_DIR = Path("artifacts")

# Singletons populated at startup
_model: xgb.XGBClassifier | None = None
_scaler = None
_explainer: SHAPExplainer | None = None
_retriever: FraudCaseRetriever | None = None
_threshold: float = 0.5


def _load_artifacts() -> None:
    global _model, _scaler, _explainer, _retriever, _threshold
    model_path = ARTIFACTS_DIR / "xgboost_model.json"
    if not model_path.exists():
        raise RuntimeError(
            "Artifacts not found. Run `python src/train.py` first "
            "or `python scripts/generate_demo_data.py && python src/train.py`."
        )
    _model = xgb.XGBClassifier()
    _model.load_model(str(model_path))

    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)

    threshold_path = ARTIFACTS_DIR / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path) as f:
            _threshold = json.load(f)["threshold"]

    _explainer = SHAPExplainer(_model)
    _retriever = FraudCaseRetriever()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Fraud Decision Explainer",
    description="XGBoost + SHAP + RAG + LLM fraud detection with drift monitoring",
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Shared request/response models
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0
    Amount: float = Field(..., gt=0, description="Transaction amount in USD")
    Time: float = Field(default=0.0, ge=0, description="Seconds elapsed since first transaction in dataset")


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    decision_threshold: float
    shap_explanation: dict[str, Any]
    similar_cases: list[dict[str, Any]]
    llm_explanation: str


class BatchPrediction(BaseModel):
    index: int
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    top_shap_features: list[dict[str, Any]]  # top 5 only — no LLM/RAG in batch mode


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=500)
    check_drift: bool = Field(default=True, description="Run PSI drift report on this batch")


class BatchResponse(BaseModel):
    count: int
    fraud_count: int
    fraud_rate: float
    decision_threshold: float
    predictions: list[BatchPrediction]
    drift_report: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale(features: np.ndarray) -> np.ndarray:
    """Scale Amount (col 28) and Time (col 29)."""
    out = features.copy()
    out[:, 28:30] = _scaler.transform(features[:, 28:30])
    return out


def _risk_level(prob: float) -> str:
    return "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.3 else "LOW")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: TransactionRequest) -> PredictionResponse:
    tx_dict = tx.model_dump()
    raw = np.array([[tx_dict[f] for f in FEATURE_COLS]], dtype=np.float32)
    scaled = _scale(raw)

    prob = float(_model.predict_proba(scaled)[0, 1])
    is_fraud = prob >= _threshold

    shap_exp = _explainer.explain(scaled)[0]
    similar = _retriever.retrieve(scaled[0])
    explanation = generate_explanation(tx.Amount, prob, shap_exp, similar)

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(prob, 4),
        risk_level=_risk_level(prob),
        decision_threshold=round(_threshold, 4),
        shap_explanation=shap_exp,
        similar_cases=similar,
        llm_explanation=explanation,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(req: BatchRequest) -> BatchResponse:
    n = len(req.transactions)
    raw = np.array(
        [[tx.model_dump()[f] for f in FEATURE_COLS] for tx in req.transactions],
        dtype=np.float32,
    )
    scaled = _scale(raw)

    probs = _model.predict_proba(scaled)[:, 1]
    shap_exps = _explainer.explain(scaled, top_n=5)

    predictions = []
    for i, (prob, shap_exp) in enumerate(zip(probs, shap_exps)):
        predictions.append(
            BatchPrediction(
                index=i,
                is_fraud=bool(prob >= _threshold),
                fraud_probability=round(float(prob), 4),
                risk_level=_risk_level(float(prob)),
                top_shap_features=shap_exp["features"],
            )
        )

    fraud_count = sum(1 for p in predictions if p.is_fraud)
    drift_report = check_drift(scaled, FEATURE_COLS) if req.check_drift else None

    return BatchResponse(
        count=n,
        fraud_count=fraud_count,
        fraud_rate=round(fraud_count / n, 4),
        decision_threshold=round(_threshold, 4),
        predictions=predictions,
        drift_report=drift_report,
    )


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "decision_threshold": _threshold,
    }


@app.get("/")
async def root() -> dict:
    return {"service": "Fraud Decision Explainer v2", "docs": "/docs"}
