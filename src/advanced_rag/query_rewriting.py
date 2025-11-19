"""
Lightweight query rewriting utilities.

This module provides a small, deterministic QueryRewriter that can be wired into
the pipeline to support intent-aware query expansion without relying on
external LLM calls. In production this can be swapped for a more capable
rewriter while preserving the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class QueryRewriterConfig:
    """
    Configuration for QueryRewriter.

    The defaults are intentionally conservative so behaviour is essentially
    a no-op unless callers opt into more aggressive rewriting.
    """

    enable_expansion: bool = True


class QueryRewriter:
    """
    Simple, deterministic query rewriter.

    Current behaviour:
    - If the query contains well-known abbreviations, append an explicit
      expansion phrase to make intent clearer for retrieval.
    - Otherwise, return the query unchanged.
    """

    def __init__(self, config: QueryRewriterConfig | None = None) -> None:
        self.config = config or QueryRewriterConfig()

    def rewrite(self, query: str, context: Dict[str, Any] | None = None) -> str:
        if not self.config.enable_expansion:
            return query

        if not query:
            return query

        q = query.strip()
        if not q:
            return q

        lower = q.lower()
        # Very small, explicit expansions to keep behaviour transparent.
        if "rag" in lower and "retrieval augmented generation" not in lower:
            return f"{q} (retrieval augmented generation)"

        if "llm" in lower and "large language model" not in lower:
            return f"{q} (large language model)"

        return q



