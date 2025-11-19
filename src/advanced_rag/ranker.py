"""
Learned ranking utilities for hybrid retrieval.

This module provides a lightweight LearnedRanker that can be wired into
HybridRetriever to re-score fused results based on simple features.
In production this would be backed by a trained model; here we keep a
transparent and deterministic implementation suitable for unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np


@dataclass
class LearnedRankerConfig:
    """
    Configuration for the LearnedRanker.

    The current implementation uses a very small linear model over
    hand-crafted features; the config exposes the main weights so tests
    can reason about ordering without opaque randomness.
    """

    base_weight: float = 1.0
    method_bonus: float = 0.1  # bonus when result comes from multiple methods
    diversity_penalty: float = 0.0  # reserved for future use
    recency_weight: float = 0.0  # optional boost for more recent content


@dataclass
class TrainingExample:
    """Simple container for offline training examples."""

    query: str
    doc_id: str
    features: Dict[str, float]
    label: float


class LearnedRanker:
    """
    Extremely lightweight ranking model.

    Features:
    - Keeps track of training examples (for offline use).
    - Exposes a deterministic score() method for lists of results.
    - Can be dropped into HybridRetriever as an alternative reranker.
    """

    def __init__(self, config: LearnedRankerConfig | None = None) -> None:
        self.config = config or LearnedRankerConfig()
        self.training_examples: List[TrainingExample] = []

    def featurize(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract a small, deterministic feature set from a retrieval result.

        Expected keys in `result`:
        - "score" (float)
        - "retrieval_methods" (optional list[str])
        - "metadata" (optional dict)
        """
        base_score = float(result.get("score", 0.0))
        methods = result.get("retrieval_methods") or []
        method_count = float(len(methods))
        meta = result.get("metadata") or {}
        # Recency is treated as a pre-computed, unitless score in [0,1] where
        # larger values indicate "more recent". This keeps the implementation
        # deterministic and avoids dependence on wall-clock time.
        recency = float(meta.get("recency", 0.0)) if isinstance(meta, dict) else 0.0
        return {
            "base_score": base_score,
            "method_count": method_count,
            "recency": recency,
        }

    def update_from_feedback(
        self,
        query: str,
        results: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]],
    ) -> None:
        """
        Collect training examples from feedback signals.

        Args:
            query: Original query string.
            results: Retrieved results (must include 'id' and 'score').
            feedback: List of dicts with at least:
                - "id": doc_id
                - "label": numeric label (e.g. 1.0 for good, 0.0 for bad)

        For the purposes of this implementation we only store examples;
        learning a real model is left to production deployments.
        """
        label_map = {fb["id"]: float(fb.get("label", 0.0)) for fb in feedback}
        for r in results:
            doc_id = r.get("id")
            if doc_id not in label_map:
                continue
            feats = self.featurize(r)
            self.training_examples.append(
                TrainingExample(query=query, doc_id=doc_id, features=feats, label=label_map[doc_id])
            )

    async def score(self, query: str, results: List[Dict[str, Any]]) -> List[float]:
        """
        Score query-document pairs.

        The scoring rule is intentionally simple and transparent:
            score = base_weight * base_score + method_bonus * method_count
        """
        scores: List[float] = []
        for r in results:
            feats = self.featurize(r)
            s = (
                self.config.base_weight * feats["base_score"]
                + self.config.method_bonus * feats["method_count"]
                + self.config.recency_weight * feats.get("recency", 0.0)
            )
            scores.append(float(s))
        return scores



