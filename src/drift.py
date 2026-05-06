"""PSI-based feature drift detection.

PSI < 0.10  → stable
PSI 0.10–0.20 → warning (monitor)
PSI ≥ 0.20  → drift (consider retraining)
"""
import json
from pathlib import Path

import numpy as np

ARTIFACTS_DIR = Path("artifacts")
PSI_WARNING = 0.10
PSI_DRIFT = 0.20


def _psi_for_feature(expected: list[float], breakpoints: list[float], actual: np.ndarray) -> float:
    edges = np.concatenate([[-np.inf], breakpoints, [np.inf]])
    exp = np.array(expected, dtype=float)
    act_counts, _ = np.histogram(actual, bins=edges)
    act = act_counts / max(act_counts.sum(), 1)
    exp = np.where(exp == 0, 1e-4, exp)
    act = np.where(act == 0, 1e-4, act)
    return float(np.sum((act - exp) * np.log(act / exp)))


def check_drift(feature_matrix: np.ndarray, feature_names: list[str]) -> dict:
    """Compute per-feature PSI against the training reference distribution.

    Args:
        feature_matrix: shape (n_samples, n_features) — scaled features.
        feature_names:  ordered list matching columns of feature_matrix.
    """
    stats_path = ARTIFACTS_DIR / "feature_stats.json"
    if not stats_path.exists():
        return {"error": "feature_stats.json missing — re-run src/train.py to generate it."}

    with open(stats_path) as f:
        ref: dict = json.load(f)

    results: dict[str, dict] = {}
    flagged: list[str] = []

    for i, name in enumerate(feature_names):
        if name not in ref:
            continue
        psi = _psi_for_feature(ref[name]["expected"], ref[name]["breakpoints"], feature_matrix[:, i])
        status = "stable" if psi < PSI_WARNING else ("warning" if psi < PSI_DRIFT else "drift")
        results[name] = {"psi": round(psi, 4), "status": status}
        if status == "drift":
            flagged.append(name)

    overall = "stable" if not flagged else "DRIFT"
    return {
        "overall_status": overall,
        "flagged_features": flagged,
        "sample_size": int(feature_matrix.shape[0]),
        "features": results,
    }
