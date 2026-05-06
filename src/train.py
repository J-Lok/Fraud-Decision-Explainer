"""Train XGBoost fraud classifier, compute SHAP values, build FAISS index."""
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import classification_report, fbeta_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
TARGET_COL = "Class"
MAX_FRAUD_INDEX = 2000  # cap FAISS index size for speed


def train(data_path: str = "data/creditcard.csv") -> None:
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows | {int(df[TARGET_COL].sum())} fraud cases")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist",
    )

    print("Training XGBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nTest AUC: {auc:.4f}")

    # --- Threshold calibration (F2 weights recall 2x over precision — right for fraud) ---
    print("Calibrating decision threshold (F2 score)...")
    thresholds = np.linspace(0.05, 0.95, 181)
    f2_scores = [fbeta_score(y_test, y_prob >= t, beta=2, zero_division=0) for t in thresholds]
    best_i = int(np.argmax(f2_scores))
    opt_threshold = float(thresholds[best_i])
    opt_f2 = float(f2_scores[best_i])
    print(f"  Optimal threshold: {opt_threshold:.2f}  (F2={opt_f2:.4f} vs {fbeta_score(y_test, y_prob >= 0.5, beta=2, zero_division=0):.4f} at 0.50)")

    print(classification_report(y_test, (y_prob >= opt_threshold).astype(int), digits=4))

    print("Computing SHAP values for fraud index...")
    explainer = shap.TreeExplainer(model)

    fraud_train = X_train[y_train == 1]
    if len(fraud_train) > MAX_FRAUD_INDEX:
        fraud_train = fraud_train.sample(MAX_FRAUD_INDEX, random_state=42)

    fraud_arr = np.ascontiguousarray(fraud_train.values, dtype=np.float32)
    shap_vals = explainer.shap_values(fraud_train)

    # Build FAISS index (L2 on raw scaled features)
    index = faiss.IndexFlatL2(fraud_arr.shape[1])
    faiss.normalize_L2(fraud_arr)
    index.add(fraud_arr)

    # Fraud case metadata for RAG context
    original_df = df.loc[fraud_train.index]
    metadata = []
    for i in range(len(fraud_train)):
        top_idx = np.abs(shap_vals[i]).argsort()[::-1][:5]
        metadata.append(
            {
                "amount": float(original_df.iloc[i]["Amount"]),
                "top_features": [FEATURE_COLS[j] for j in top_idx],
                "shap_contributions": [float(shap_vals[i][j]) for j in top_idx],
            }
        )

    # --- Feature distribution stats for drift detection (PSI) ---
    print("Computing feature distribution stats for drift detection...")
    N_BINS = 10
    feature_stats: dict = {}
    for col in FEATURE_COLS:
        vals = X_train[col].values
        breaks_full = np.percentile(vals, np.linspace(0, 100, N_BINS + 1))
        internal = breaks_full[1:-1].tolist()
        edges = np.concatenate([[-np.inf], internal, [np.inf]])
        counts, _ = np.histogram(vals, bins=edges)
        expected = (counts / counts.sum()).tolist()
        feature_stats[col] = {"breakpoints": internal, "expected": expected}

    # Persist artifacts
    model.save_model(str(ARTIFACTS_DIR / "xgboost_model.json"))
    with open(ARTIFACTS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    faiss.write_index(index, str(ARTIFACTS_DIR / "faiss_index.bin"))
    with open(ARTIFACTS_DIR / "fraud_metadata.json", "w") as f:
        json.dump(metadata, f)
    with open(ARTIFACTS_DIR / "threshold.json", "w") as f:
        json.dump({"threshold": opt_threshold, "f2_score": opt_f2, "metric": "fbeta(beta=2)"}, f, indent=2)
    with open(ARTIFACTS_DIR / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    artifacts = ["xgboost_model.json", "scaler.pkl", "faiss_index.bin",
                 "fraud_metadata.json", "threshold.json", "feature_stats.json"]
    for name in artifacts:
        size_kb = (ARTIFACTS_DIR / name).stat().st_size / 1024
        print(f"  {name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    train(path)
