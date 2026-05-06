"""FAISS-based retrieval of similar past fraud cases."""
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

ARTIFACTS_DIR = Path("artifacts")


class FraudCaseRetriever:
    def __init__(self) -> None:
        self._index = faiss.read_index(str(ARTIFACTS_DIR / "faiss_index.bin"))
        with open(ARTIFACTS_DIR / "fraud_metadata.json") as f:
            self._metadata: list[dict] = json.load(f)

    def retrieve(self, features: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
        """Return k nearest fraud cases with similarity scores."""
        vec = features.astype(np.float32).reshape(1, -1).copy()
        faiss.normalize_L2(vec)
        distances, indices = self._index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self._metadata):
                case = dict(self._metadata[idx])
                # Convert L2 distance on unit-normalized vectors to cosine similarity
                case["similarity"] = float(1.0 - dist / 2.0)
                results.append(case)
        return results
