"""SHAP-based feature attribution for individual transactions."""
from pathlib import Path
from typing import Any

import numpy as np
import shap
import xgboost as xgb

# PIL and matplotlib are imported lazily inside plot_waterfall to keep module load fast.

ARTIFACTS_DIR = Path("artifacts")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

# Human-readable aliases for the anonymized PCA features.
# V14 and V12 are well-known strong fraud signals in the CC dataset.
_ALIASES: dict[str, str] = {
    "V1": "behavioral pattern 1",
    "V2": "behavioral pattern 2",
    "V3": "velocity pattern",
    "V4": "merchant pattern",
    "V9": "card usage pattern",
    "V10": "session pattern",
    "V11": "frequency pattern",
    "V12": "transaction signature",
    "V14": "card authentication signal",
    "V16": "location pattern",
    "V17": "device fingerprint",
    "Amount": "transaction amount",
    "Time": "time of day",
}


def feature_label(name: str) -> str:
    return _ALIASES.get(name, name.lower().replace("v", "risk pattern "))


class SHAPExplainer:
    def __init__(self, model: xgb.XGBClassifier) -> None:
        self._explainer = shap.TreeExplainer(model)

    def explain(self, X: np.ndarray, top_n: int | None = None) -> list[dict[str, Any]]:
        """Return per-sample SHAP attribution dicts, sorted by absolute impact.

        top_n=None returns all features (useful for waterfall plots and batch drift).
        """
        shap_vals = self._explainer.shap_values(X)
        base = float(self._explainer.expected_value)

        results = []
        for i in range(len(X)):
            sv = shap_vals[i]
            sorted_idx = np.abs(sv).argsort()[::-1]
            if top_n is not None:
                sorted_idx = sorted_idx[:top_n]
            results.append(
                {
                    "base_value": base,
                    "features": [
                        {
                            "name": FEATURE_COLS[j],
                            "label": feature_label(FEATURE_COLS[j]),
                            "shap_value": float(sv[j]),
                            "raw_value": float(X[i, j]),
                            "direction": "increases" if sv[j] > 0 else "decreases",
                        }
                        for j in sorted_idx
                    ],
                }
            )
        return results


def plot_waterfall(shap_exp: dict, top_n: int = 12) -> Any:
    """Render a horizontal bar waterfall chart for one transaction's SHAP values.

    Returns a PIL Image (PNG) ready for display in Gradio or saving to disk.
    """
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    features = shap_exp["features"][:top_n]
    labels = [f["label"].title() for f in features]
    values = [f["shap_value"] for f in features]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(4, len(features) * 0.6)))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor="white", height=0.65)
    ax.axvline(0, color="#333333", linewidth=0.9, zorder=3)
    ax.set_xlabel("SHAP value  (+  increases fraud probability)", fontsize=10)
    ax.set_title("Feature Attribution — Impact on Fraud Score", pad=10, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, values[::-1]):
        pad = max(abs(val) * 0.03, 0.002)
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + (pad if val >= 0 else -pad),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center", ha=ha, fontsize=8.5, color="#111111",
        )

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()
