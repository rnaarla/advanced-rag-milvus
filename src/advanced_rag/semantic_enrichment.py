"""
Lightweight semantic enrichment utilities for ETL and retrieval.

The goal of this module is to attach simple, deterministic semantic
signals (entities, topics) to chunks at ingest time so that downstream
retrieval and analytics can use them without heavy external NLP
dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set
import re
from collections import Counter


@dataclass
class EnrichmentResult:
    """Structured enrichment output."""

    entities: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


class SemanticEnricher:
    """
    Simple semantic enricher for texts.

    Heuristics:
    - Entities: tokens that look like proper nouns (capitalized, alphabetic).
    - Topics: the most frequent non-stopword tokens of length > 3.
    """

    def __init__(self, max_topics: int = 5) -> None:
        self.max_topics = max_topics
        # Minimal English stopword list to keep behavior deterministic.
        self._stopwords: Set[str] = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "about",
            "your",
            "their",
            "have",
            "been",
            "are",
            "was",
            "were",
            "will",
            "shall",
            "can",
            "could",
            "would",
            "should",
            "a",
            "an",
            "of",
            "in",
            "on",
            "at",
            "to",
            "by",
            "is",
            "it",
            "as",
        }

    def enrich(self, text: str) -> EnrichmentResult:
        """Extract entities and topics from text."""
        if not text:
            return EnrichmentResult()

        # Tokenize while preserving original casing for entity detection.
        raw_tokens = re.findall(r"\b\w+\b", text)
        lower_tokens = [t.lower() for t in raw_tokens]

        entities: List[str] = []
        seen_entities: Set[str] = set()
        for tok in raw_tokens:
            if tok.isalpha() and tok[0].isupper():
                key = tok
                if key not in seen_entities:
                    seen_entities.add(key)
                    entities.append(key)

        counter = Counter(lower_tokens)
        topic_candidates = [
            (tok, freq)
            for tok, freq in counter.items()
            if tok not in self._stopwords and len(tok) > 3
        ]
        topic_candidates.sort(key=lambda x: (-x[1], x[0]))
        topics = [tok for tok, _ in topic_candidates[: self.max_topics]]

        return EnrichmentResult(entities=entities, topics=topics)



