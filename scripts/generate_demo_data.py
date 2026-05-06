"""Generate synthetic credit card fraud data that mimics the Kaggle dataset structure.

Run this if you don't have a Kaggle account / access to creditcard.csv.
The fraud signal is injected into V14 and V12 (dominant features in the real dataset).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Path("data").mkdir(exist_ok=True)


def generate(n_normal: int = 20_000, n_fraud: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Normal transactions: all 30 features from standard normal
    normal = rng.standard_normal((n_normal, 28))
    normal_amount = rng.exponential(scale=60, size=n_normal)
    normal_time = np.sort(rng.uniform(0, 172_800, n_normal))

    # Fraud: inject strong signal into V14 (index 13) and V12 (index 11)
    fraud = rng.standard_normal((n_fraud, 28))
    fraud[:, 13] -= 6.0   # V14 strongly negative in real fraud cases
    fraud[:, 11] -= 3.5   # V12 moderately negative
    fraud[:, 3] += 2.0    # V4 slightly elevated
    fraud[:, 9] -= 2.5    # V10 slightly negative
    fraud_amount = rng.choice([199, 499, 999, 1_999, 9_999], size=n_fraud).astype(float)
    fraud_time = rng.uniform(0, 172_800, n_fraud)

    v_cols = [f"V{i}" for i in range(1, 29)]
    df_normal = pd.DataFrame(normal, columns=v_cols)
    df_normal["Amount"] = normal_amount
    df_normal["Time"] = normal_time
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(fraud, columns=v_cols)
    df_fraud["Amount"] = fraud_amount
    df_fraud["Time"] = fraud_time
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    out = Path("data/creditcard.csv")
    df.to_csv(out, index=False)
    fraud_rate = n_fraud / (n_normal + n_fraud) * 100
    print(f"Generated {len(df):,} transactions ({n_fraud} fraud, {n_normal} normal, {fraud_rate:.2f}% fraud rate)")
    print(f"Saved → {out}")
    return df


if __name__ == "__main__":
    n_normal = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    n_fraud = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    generate(n_normal, n_fraud)
