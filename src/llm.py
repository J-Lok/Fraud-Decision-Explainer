"""LLM explanation layer: HF Inference API with template fallback."""
import os
from typing import Any

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")


def _build_prompt(
    amount: float,
    fraud_prob: float,
    shap_features: list[dict],
    similar_cases: list[dict],
) -> str:
    feature_lines = "\n".join(
        f"  - {f['label'].title()}: {'+' if f['shap_value'] > 0 else ''}"
        f"{f['shap_value']:.3f} ({f['direction']} fraud risk)"
        for f in shap_features[:5]
    )

    similar_lines = "\n".join(
        f"  - Case {i + 1}: Amount=${c['amount']:.2f}, "
        f"primary signal={c['top_features'][0] if c['top_features'] else 'N/A'}, "
        f"similarity={c['similarity']:.0%}"
        for i, c in enumerate(similar_cases[:3])
    ) or "  None found."

    return f"""<|system|>
You are a concise fraud analyst assistant. Write only the explanation — no preamble.
<|user|>
A transaction of ${amount:.2f} was scored {fraud_prob:.1%} fraud probability by our ML model.

SHAP risk factors (positive = increases fraud likelihood):
{feature_lines}

Similar confirmed fraud cases retrieved from our database:
{similar_lines}

Write a 2–3 sentence plain-language explanation for a fraud analyst: why was this flagged, what pattern matches known fraud, and what action to take.
<|assistant|>"""


def _template_fallback(
    amount: float,
    fraud_prob: float,
    shap_features: list[dict],
    similar_cases: list[dict],
) -> str:
    top = [f["label"] for f in shap_features[:3]]
    risk = "HIGH" if fraud_prob > 0.7 else "MODERATE"
    action = (
        "Block and escalate to the fraud team immediately."
        if fraud_prob > 0.7
        else "Flag for manual review before processing."
    )
    return (
        f"This ${amount:.2f} transaction was flagged with {fraud_prob:.1%} fraud probability "
        f"due to {risk} anomaly signals in {', '.join(top[:2])}. "
        f"The behavioral pattern matches {len(similar_cases)} confirmed fraud case(s) in "
        f"the historical database, particularly around {top[0] if top else 'risk pattern 14'}. "
        f"{action}"
    )


def generate_explanation(
    amount: float,
    fraud_prob: float,
    shap_explanation: dict[str, Any],
    similar_cases: list[dict],
) -> str:
    features = shap_explanation.get("features", [])

    if not HF_TOKEN:
        return _template_fallback(amount, fraud_prob, features, similar_cases)

    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)
        prompt = _build_prompt(amount, fraud_prob, features, similar_cases)
        response = client.text_generation(
            prompt,
            max_new_tokens=180,
            temperature=0.3,
            repetition_penalty=1.1,
            stop_sequences=["<|user|>", "<|system|>", "\n\n\n"],
        )
        explanation = response.strip()
        return explanation if explanation else _template_fallback(amount, fraud_prob, features, similar_cases)
    except Exception:
        return _template_fallback(amount, fraud_prob, features, similar_cases)
