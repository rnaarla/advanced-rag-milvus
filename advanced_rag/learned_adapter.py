from typing import List, Tuple, Dict
import math

class LearnedHybridAdapter:
    """
    Lightweight learned adapter for dense/sparse weighting.
    Trains simple running statistics from feedback logs (thumbs up/down)
    and adjusts weights based on feature signals:
      - query_length
      - prior sparse success rate
      - prior dense success rate
    This is intentionally simple to avoid heavy dependencies.
    """
    def __init__(self):
        self.total = 0
        self.sparse_success = 0
        self.dense_success = 0

    def fit_from_feedback(self, rows: List[Dict]):
        """
        rows: list of { 'method': 'sparse'|'semantic', 'vote': 'up'|'down' }
        """
        for r in rows:
            self.total += 1
            if r.get("vote") == "up":
                if r.get("method") == "sparse":
                    self.sparse_success += 1
                elif r.get("method") == "semantic":
                    self.dense_success += 1

    def __call__(self, query: str) -> Tuple[float, float]:
        qlen = len(query.split())
        # Base priors from running success rates
        sparse_rate = (self.sparse_success + 1) / (self.total + 2)
        dense_rate = (self.dense_success + 1) / (self.total + 2)
        # Heuristic nudges
        # Short queries -> slight bump to sparse; longer -> dense
        sparse_bias = 0.1 if qlen <= 4 else 0.0
        dense_bias = 0.1 if qlen > 8 else 0.0
        s = sparse_rate + sparse_bias
        d = dense_rate + dense_bias
        # Normalize to [0,1] and sum <=1 ; keep both non-zero
        total = s + d
        if total <= 0:
            return (0.5, 0.5)
        s = s / total
        d = d / total
        # Clamp
        s = max(0.05, min(0.95, s))
        d = max(0.05, min(0.95, d))
        # Re-normalize
        total = s + d
        return (d / total, s / total)  # returns (dense_weight, sparse_weight)


