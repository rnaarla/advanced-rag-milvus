"""
Query decomposition utilities for plan-and-execute RAG.

This module provides a lightweight, deterministic QueryDecomposer that
can split complex queries into sub-questions. In production this could
be backed by an LLM, but here we keep the logic simple and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class DecompositionResult:
    """Result of decomposing a complex query."""

    sub_queries: List[str]
    strategy: str


class QueryDecomposer:
    """
    Heuristic query decomposer.

    Rules:
    - If the query is short (< N chars), return it as-is.
    - Otherwise, split on the first " and " / " AND " into at most two
      sub-queries, trimming whitespace.
    - If no heuristic applies, fall back to the original query.
    """

    def __init__(self, min_length: int = 60) -> None:
        self.min_length = min_length

    def decompose(self, query: str) -> DecompositionResult:
        if not query:
            return DecompositionResult(sub_queries=[], strategy="empty")

        q = query.strip()
        if len(q) < self.min_length:
            return DecompositionResult(sub_queries=[q], strategy="single")

        lowered = q.lower()
        for token in (" and ", " AND "):
            if token.strip().lower() in lowered:
                parts = [p.strip() for p in q.split(token.strip(), 1) if p.strip()]
                if len(parts) >= 2:
                    return DecompositionResult(sub_queries=parts, strategy="split_and")

        return DecompositionResult(sub_queries=[q], strategy="fallback")



