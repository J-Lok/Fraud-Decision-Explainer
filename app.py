"""Gradio UI for Hugging Face Spaces deployment."""
import json
import pickle
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from src.explain import FEATURE_COLS, SHAPExplainer, plot_waterfall
from src.feedback import log as log_feedback
from src.feedback import stats as feedback_stats
from src.llm import generate_explanation
from src.rag import FraudCaseRetriever

ARTIFACTS_DIR = Path("artifacts")

# ---------------------------------------------------------------------------
# Load artifacts (once at startup)
# ---------------------------------------------------------------------------
model = xgb.XGBClassifier()
model.load_model(str(ARTIFACTS_DIR / "xgboost_model.json"))
with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

threshold = 0.5
threshold_path = ARTIFACTS_DIR / "threshold.json"
if threshold_path.exists():
    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

explainer = SHAPExplainer(model)
retriever = FraudCaseRetriever()

# First real fraud transaction from the Kaggle dataset (public knowledge)
DEMO_FRAUD = {
    "Amount": 149.62, "Time": 0.0,
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
    "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
    "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211,
}
DEMO_LEGIT = {k: 0.0 for k in DEMO_FRAUD}
DEMO_LEGIT["Amount"] = 12.50
DEMO_LEGIT["Time"] = 86400.0


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _scale(features: np.ndarray) -> np.ndarray:
    out = features.copy()
    out[:, 28:30] = scaler.transform(features[:, 28:30])
    return out


def _shap_bar(value: float) -> str:
    filled = min(int(abs(value) * 30), 30)
    return ("█" if value > 0 else "░") * filled


def analyze(amount: float, time_val: float, *v_vals: float) -> tuple:
    features_dict = {f"V{i + 1}": float(v_vals[i]) for i in range(28)}
    features_dict["Amount"] = float(amount)
    features_dict["Time"] = float(time_val)

    raw = np.array([[features_dict[f] for f in FEATURE_COLS]], dtype=np.float32)
    scaled = _scale(raw)

    prob = float(model.predict_proba(scaled)[0, 1])
    is_fraud = prob >= threshold
    risk = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.3 else "LOW")

    shap_exp = explainer.explain(scaled)[0]
    similar = retriever.retrieve(scaled[0])
    llm_text = generate_explanation(amount, prob, shap_exp, similar)
    waterfall = plot_waterfall(shap_exp, top_n=12)

    # Verdict
    emoji = "🚨" if is_fraud else "✅"
    verdict = (
        f"## {emoji} {'FRAUD DETECTED' if is_fraud else 'LEGITIMATE'}\n"
        f"**Probability:** {prob:.1%}  |  **Risk:** {risk}  |  "
        f"**Threshold:** {threshold:.2f} (calibrated F2)"
    )

    # SHAP markdown table (top 6)
    shap_md = "### Top Risk Factors\n| Feature | Impact | Direction |\n|---|---|---|\n"
    for feat in shap_exp["features"][:6]:
        shap_md += (
            f"| {feat['label'].title()} | `{feat['shap_value']:+.3f}` {_shap_bar(feat['shap_value'])} "
            f"| {feat['direction']} risk |\n"
        )

    # Similar cases
    if similar:
        sim_md = "### Similar Past Fraud Cases\n"
        for i, c in enumerate(similar, 1):
            sim_md += (
                f"**Case {i}:** Amount=${c['amount']:.2f} | "
                f"Primary signal: _{c['top_features'][0] if c['top_features'] else 'N/A'}_ | "
                f"Similarity: {c['similarity']:.0%}\n\n"
            )
    else:
        sim_md = "_No similar cases found in the index._"

    # State dict stored in gr.State for feedback
    state = {
        "amount": float(amount),
        "prob": float(prob),
        "is_fraud": bool(is_fraud),
        "features": features_dict,
    }

    return verdict, shap_md, sim_md, llm_text, waterfall, state


def give_feedback(correct: bool, state: dict | None) -> str:
    if not state:
        return "_Analyze a transaction first, then give feedback._"
    log_feedback(state["amount"], state["prob"], state["is_fraud"], correct, state["features"])
    s = feedback_stats()
    label = "Correct ✓" if correct else "Wrong ✗"
    rate = f"{s['agreement_rate']:.0%}" if s["agreement_rate"] is not None else "N/A"
    return f"**Feedback recorded: {label}** — Total: {s['total_feedback']} | Agreement rate: {rate}"


def load_demo(kind: str) -> list:
    src = DEMO_FRAUD if kind == "fraud" else DEMO_LEGIT
    return [src["Amount"], src["Time"]] + [src[f"V{i}"] for i in range(1, 29)]


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Fraud Decision Explainer", theme=gr.themes.Soft()) as demo:
    prediction_state = gr.State(None)

    gr.Markdown(
        "# 🔍 Fraud Decision Explainer\n"
        "XGBoost · SHAP waterfall · RAG similar-case retrieval · LLM narrative · Analyst feedback loop"
    )

    with gr.Row():
        # --- Left panel: inputs ---
        with gr.Column(scale=1):
            amount = gr.Number(label="Amount ($)", value=149.62, minimum=0.01)
            time_val = gr.Number(label="Time (seconds)", value=0.0, minimum=0)

            with gr.Accordion("PCA Features V1–V28 (advanced)", open=False):
                v_inputs = [gr.Number(label=f"V{i}", value=0.0, step=0.001) for i in range(1, 29)]

            with gr.Row():
                btn_fraud = gr.Button("Load demo fraud", variant="secondary")
                btn_legit = gr.Button("Load demo legit", variant="secondary")

            analyze_btn = gr.Button("Analyze Transaction", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### Was the model correct?")
            with gr.Row():
                btn_correct = gr.Button("✓ Correct", variant="primary")
                btn_wrong = gr.Button("✗ Wrong", variant="stop")
            feedback_status = gr.Markdown("_Analyze a transaction first._")

        # --- Right panel: outputs ---
        with gr.Column(scale=2):
            verdict_out = gr.Markdown()
            shap_plot_out = gr.Image(label="SHAP Waterfall", type="pil")
            shap_out = gr.Markdown()
            similar_out = gr.Markdown()
            llm_out = gr.Textbox(label="LLM Explanation", lines=5, interactive=False)

    all_inputs = [amount, time_val] + v_inputs
    all_outputs = [verdict_out, shap_out, similar_out, llm_out, shap_plot_out, prediction_state]

    analyze_btn.click(analyze, inputs=all_inputs, outputs=all_outputs)
    btn_fraud.click(lambda: load_demo("fraud"), outputs=all_inputs)
    btn_legit.click(lambda: load_demo("legit"), outputs=all_inputs)
    btn_correct.click(lambda s: give_feedback(True, s), inputs=[prediction_state], outputs=[feedback_status])
    btn_wrong.click(lambda s: give_feedback(False, s), inputs=[prediction_state], outputs=[feedback_status])

    gr.Markdown(
        "---\n*No HF_TOKEN → LLM uses a deterministic template. "
        "Set `HF_TOKEN` in Space secrets for generative explanations via Zephyr-7B.*"
    )

if __name__ == "__main__":
    demo.launch()
