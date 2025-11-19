"""
Hybrid Retrieval Engine
Combines dense vector search, sparse BM25, and cross-encoder reranking
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import asyncio
from typing import Callable, Tuple
import re
import logging

from .ranker import LearnedRanker
from .constants import RetrievalConstants
from datetime import datetime

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Lightweight query classifier to route requests to retrieval profiles.
    Heuristics are intentionally simple and deterministic so they are safe
    to run synchronously on the hot path.
    """

    def __init__(self, max_faq_len: int = 80, long_query_len: int = 200):
        self.max_faq_len = max_faq_len
        self.long_query_len = long_query_len

    def classify(self, query: str) -> str:
        """
        Classify the query into a coarse profile:
        - faq: short question-style lookups
        - troubleshooting: mentions errors/exceptions/failures
        - summary: explicit summarization/overview requests
        - analysis: long-form, open-ended analysis
        - default: everything else
        """
        if not query:
            return "default"

        q = query.strip()
        if not q:
            return "default"

        q_lower = q.lower()

        # Troubleshooting-style queries
        if any(token in q_lower for token in ("error", "exception", "stack trace", "failed", "failure", "bug")):
            return "troubleshooting"

        # Explicit summarization
        if any(token in q_lower for token in ("summarize", "summary", "tl;dr", "overview")):
            return "summary"

        # Short question-style lookups
        if len(q) <= self.max_faq_len and q.endswith("?"):
            return "faq"

        # Long-form analytical queries
        if len(q) >= self.long_query_len:
            return "analysis"

        return "default"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    hybrid_alpha: float = 0.7  # Weight for dense vs sparse (0=sparse only, 1=dense only)
    top_k: int = 20
    rerank_top_k: int = 5
    enable_reranking: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    enable_mmr: bool = False
    mmr_lambda: float = 0.7  # 1.0 favors relevance; 0.0 favors diversity
    # Learned ranker flag (applies within rerank())
    enable_learned_ranker: bool = False
    # Adaptive hybrid weighting hook (query -> (dense_weight, sparse_weight))
    # Set via HybridRetriever(weight_adapter=...)
    # Not serialized; runtime only.
    
    # Search parameters
    semantic_search_params: Dict = None
    sparse_search_params: Dict = None
    
    def __post_init__(self):
        if self.semantic_search_params is None:
            self.semantic_search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }
        if self.sparse_search_params is None:
            self.sparse_search_params = {
                "metric_type": "IP",
                "params": {"drop_ratio_search": 0.2}
            }


class HybridRetriever:
    """
    Implements hybrid retrieval combining:
    1. Dense semantic search (vector similarity)
    2. Sparse keyword search (BM25-style)
    3. Optional domain-specific search
    4. Cross-encoder reranking
    """
    
    def __init__(
        self,
        index_manager: 'MilvusIndexManager',
        config: Optional[RetrievalConfig] = None,
        weight_adapter: Optional[Callable[[str], Tuple[float, float]]] = None,
        classifier: Optional[QueryClassifier] = None,
        profiles: Optional[Dict[str, RetrievalConfig]] = None,
        learned_ranker: Optional[LearnedRanker] = None,
    ):
        """
        Args:
            index_manager: MilvusIndexManager instance
            config: Base retrieval configuration
            weight_adapter: Optional adaptive weight hook
            classifier: Optional query classifier to choose retrieval profiles
            profiles: Optional map of profile_name -> RetrievalConfig
        """
        self.index_manager = index_manager
        base_config = config or RetrievalConfig()
        self.config = base_config
        self.weight_adapter = weight_adapter
        self.classifier = classifier or QueryClassifier()
        # Build default profiles if none provided
        self.profiles: Dict[str, RetrievalConfig] = profiles or self._build_default_profiles(base_config)

        # Rerankers
        self.reranker = None  # Cross-encoder or other reranker; set externally
        self.learned_ranker: Optional[LearnedRanker] = learned_ranker

    @staticmethod
    def _build_default_profiles(base_config: RetrievalConfig) -> Dict[str, RetrievalConfig]:
        """
        Construct a small set of retrieval profiles.
        The base_config is used for the 'default' profile so existing
        behavior is preserved when classification is not in use.
        """
        profiles: Dict[str, RetrievalConfig] = {}
        profiles["default"] = base_config

        # Helper to clamp top_k and rerank_top_k to safe limits.
        def _clamp_top_k(value: int) -> int:
            return max(
                1,
                min(int(value), getattr(RetrievalConstants, "MAX_TOP_K", value)),
            )

        def _clamp_rerank_k(value: int) -> int:
            return max(1, min(int(value), _clamp_top_k(value)))

        # Short FAQ-style lookups: smaller top_k, reranking enabled, modest MMR.
        faq_top_k = _clamp_top_k(min(base_config.top_k, 10))
        profiles["faq"] = RetrievalConfig(
            hybrid_alpha=base_config.hybrid_alpha,
            top_k=faq_top_k,
            rerank_top_k=_clamp_rerank_k(base_config.rerank_top_k),
            enable_reranking=True,
            dense_weight=base_config.dense_weight,
            sparse_weight=base_config.sparse_weight,
            enable_mmr=False,
            mmr_lambda=0.7,
        )

        # Troubleshooting: over-retrieve, use MMR to diversify similar snippets.
        trouble_top_k = _clamp_top_k(max(base_config.top_k, 30))
        profiles["troubleshooting"] = RetrievalConfig(
            hybrid_alpha=base_config.hybrid_alpha,
            top_k=trouble_top_k,
            rerank_top_k=_clamp_rerank_k(10),
            enable_reranking=True,
            dense_weight=base_config.dense_weight,
            sparse_weight=base_config.sparse_weight,
            enable_mmr=True,
            mmr_lambda=0.5,
        )

        # Summaries: pull more context but often aggregate, reranking optional.
        summary_top_k = _clamp_top_k(max(base_config.top_k, 40))
        profiles["summary"] = RetrievalConfig(
            hybrid_alpha=base_config.hybrid_alpha,
            top_k=summary_top_k,
            rerank_top_k=_clamp_rerank_k(10),
            enable_reranking=False,
            dense_weight=base_config.dense_weight,
            sparse_weight=base_config.sparse_weight,
            enable_mmr=False,
            mmr_lambda=0.7,
        )

        # Long-form analysis: more context and diversification.
        analysis_top_k = _clamp_top_k(max(base_config.top_k, 30))
        profiles["analysis"] = RetrievalConfig(
            hybrid_alpha=base_config.hybrid_alpha,
            top_k=analysis_top_k,
            rerank_top_k=_clamp_rerank_k(10),
            enable_reranking=True,
            dense_weight=base_config.dense_weight,
            sparse_weight=base_config.sparse_weight,
            enable_mmr=True,
            mmr_lambda=0.8,
        )
        return profiles

    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_domain_index: bool = False,
        domain: Optional[str] = None,
        profile_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Public entrypoint for hybrid retrieval with an end-to-end timeout.

        The core retrieval logic lives in _retrieve_inner(); this wrapper applies
        an overall latency budget so slow downstreams (Milvus, embeddings) do not
        exceed the configured SLA.
        """
        timeout_seconds = float(getattr(RetrievalConstants, "TIMEOUT_SECONDS", 0.3))
        try:
            return await asyncio.wait_for(
                self._retrieve_inner(
                    query=query,
                    filters=filters,
                    use_domain_index=use_domain_index,
                    domain=domain,
                    profile_hint=profile_hint,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "HybridRetriever.retrieve timed out after %.3f seconds", timeout_seconds
            )
            # On timeout we degrade gracefully to "no results" instead of raising.
            return []

    async def _retrieve_inner(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_domain_index: bool = False,
        domain: Optional[str] = None,
        profile_hint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining multiple search strategies
        
        Args:
            query: Search query
            filters: Optional metadata filters
            use_domain_index: Whether to include domain-specific index
            domain: Domain identifier
            
        Returns:
            List of retrieved documents with scores. Each result will include
            the retrieval profile used for this query in its metadata.
        """
        # Choose retrieval profile for this query (safe default on any error)
        profile_name = "default"
        try:
            # Allow explicit profile override first, then fall back to classifier.
            if profile_hint and profile_hint in self.profiles:
                profile_name = profile_hint
            elif self.classifier:
                profile_name = self.classifier.classify(query) or "default"
        except Exception:
            profile_name = "default"

        profile_config = self.profiles.get(profile_name, self.config)
        # Use the selected profile config for this request. This is mutable by
        # design (e.g., weight_adapter may update dense/sparse weights).
        self.config = profile_config

        # Generate query embeddings
        semantic_emb = await self._get_semantic_embedding(query)
        sparse_emb = await self._get_sparse_embedding(query)
        
        # Build filter expression
        filter_expr = self._build_filter_expression(filters) if filters else None
        
        # Parallel retrieval from different indexes
        retrieval_tasks = [
            self._search_semantic(semantic_emb, filter_expr),
            self._search_sparse(sparse_emb, filter_expr)
        ]
        
        if use_domain_index and domain:
            domain_emb = await self._get_domain_embedding(query, domain)
            retrieval_tasks.append(
                self._search_domain(domain_emb, filter_expr)
            )
        
        # Execute searches in parallel
        results_list = await asyncio.gather(*retrieval_tasks)
        
        # Optionally adapt dense/sparse weights per query
        if self.weight_adapter:
            try:
                dense_w, sparse_w = self.weight_adapter(query)
                # Clamp and normalize conservatively
                dense_w = max(0.0, min(1.0, float(dense_w)))
                sparse_w = max(0.0, min(1.0, float(sparse_w)))
                if dense_w + sparse_w > 0:
                    self.config.dense_weight = dense_w
                    self.config.sparse_weight = sparse_w
            except Exception:
                # Fallback silently to configured weights
                pass
        # Combine and fuse results
        fused_results = self._fuse_results(
            semantic_results=results_list[0],
            sparse_results=results_list[1],
            domain_results=results_list[2] if len(results_list) > 2 else []
        )

        # Attach profile metadata for downstream logging/analysis
        for result in fused_results:
            metadata = result.get("metadata")
            if isinstance(metadata, dict):
                metadata.setdefault("retrieval_profile", profile_name)
                result["metadata"] = metadata
            else:
                # Fallback when tests or callers don't use metadata dicts
                result["retrieval_profile"] = profile_name

        # Return top-k for the active profile
        return fused_results[:self.config.top_k]
    
    async def _search_semantic(
        self,
        embedding: np.ndarray,
        filters: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search semantic index"""
        try:
            results = await self.index_manager.search(
                query_embedding=embedding,
                collection_name="semantic_index",
                top_k=self.config.top_k * 2,  # Over-retrieve for fusion
                filters=filters,
                search_params=self.config.semantic_search_params
            )
        except Exception:  # pragma: no cover - defensive Milvus error handling
            # If Milvus search fails (schema/partition issues, etc.), degrade gracefully
            # to "no semantic hits" instead of surfacing a 500 to the caller.
            return []
        
        # Tag results with search method
        for result in results:
            result["method"] = "semantic"
            result["original_score"] = result["score"]
        
        return results
    
    async def _search_sparse(
        self,
        embedding: np.ndarray,
        filters: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search sparse index (BM25-style)"""
        # Skip if sparse collection is not available on real Milvus managers.
        # For lightweight fakes used in tests (without .collections), we still
        # call through so they can provide canned results.
        collections = getattr(self.index_manager, "collections", None)
        if collections is not None and "sparse_index" not in collections:
            return []
        try:
            results = await self.index_manager.search(
                query_embedding=embedding,
                collection_name="sparse_index",
                top_k=self.config.top_k * 2,
                filters=filters,
                search_params=self.config.sparse_search_params
            )
        except Exception:  # pragma: no cover - defensive sparse index error handling
            # Sparse is best-effort; on failure, fall back to dense-only results.
            return []
        
        for result in results:
            result["method"] = "sparse"
            result["original_score"] = result["score"]
        
        return results
    
    async def _search_domain(
        self,
        embedding: np.ndarray,
        filters: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Search domain-specific index"""
        try:
            results = await self.index_manager.search(
                query_embedding=embedding,
                collection_name="domain_index",
                top_k=self.config.top_k,
                filters=filters,
                search_params=self.config.semantic_search_params
            )
        except Exception:  # pragma: no cover - defensive domain index error handling
            # Domain index is optional; on failure, simply omit domain hits.
            return []
        
        for result in results:
            result["method"] = "domain"
            result["original_score"] = result["score"]
        
        return results
    
    def _fuse_results(
        self,
        semantic_results: List[Dict],
        sparse_results: List[Dict],
        domain_results: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple retrieval methods using Reciprocal Rank Fusion
        
        This addresses the challenge of combining scores from different metrics
        """
        # Reciprocal Rank Fusion (RRF)
        k = 60  # RRF parameter
        fused_scores = defaultdict(lambda: {"score": 0.0, "data": None, "methods": []})
        
        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank)
            fused_scores[doc_id]["score"] += rrf_score * self.config.dense_weight
            fused_scores[doc_id]["data"] = result
            fused_scores[doc_id]["methods"].append("semantic")
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result["id"]
            rrf_score = 1.0 / (k + rank)
            fused_scores[doc_id]["score"] += rrf_score * self.config.sparse_weight
            if fused_scores[doc_id]["data"] is None:
                fused_scores[doc_id]["data"] = result
            fused_scores[doc_id]["methods"].append("sparse")
        
        # Process domain results (if available)
        if domain_results:
            domain_weight = 0.2
            for rank, result in enumerate(domain_results, start=1):
                doc_id = result["id"]
                rrf_score = 1.0 / (k + rank)
                fused_scores[doc_id]["score"] += rrf_score * domain_weight
                if fused_scores[doc_id]["data"] is None:
                    fused_scores[doc_id]["data"] = result
                fused_scores[doc_id]["methods"].append("domain")
        
        # Convert to list and sort by fused score
        fused_list = []
        now = datetime.utcnow()
        for doc_id, info in fused_scores.items():
            result = info["data"]
            result["score"] = info["score"]
            result["retrieval_methods"] = list(set(info["methods"]))
            # Best-effort recency annotation for downstream rankers:
            # compute a simple [0,1] recency score from timestamp metadata when present.
            meta = result.get("metadata")
            if isinstance(meta, dict) and "timestamp" in meta and "recency" not in meta:
                try:
                    ts = datetime.fromisoformat(str(meta["timestamp"]))
                    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                    recency = 1.0 / (1.0 + age_days)
                    meta["recency"] = float(recency)
                    result["metadata"] = meta
                except Exception:  # pragma: no cover - defensive timestamp parsing
                    # If parsing fails, skip recency annotation.
                    pass
            fused_list.append(result)
        
        # Sort by fused score
        fused_list.sort(key=lambda x: x["score"], reverse=True)
        # Optional diversification via MMR
        if self.config.enable_mmr and fused_list:
            return self._mmr_diversify(fused_list, self.config.top_k, self.config.mmr_lambda)
        return fused_list

    def _mmr_diversify(self, ranked: List[Dict[str, Any]], k: int, mmr_lambda: float) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance diversification on token Jaccard similarity."""
        selected: List[Dict[str, Any]] = []
        candidates = ranked.copy()
        get_tokens = lambda r: set((r.get("content") or "").lower().split())
        while candidates and len(selected) < k:
            best = None
            best_score = -1e9
            for r in candidates:
                rel = r["score"]
                if not selected:
                    score = rel
                else:
                    sim = max(
                        (len(get_tokens(r) & get_tokens(s)) / (len(get_tokens(r) | get_tokens(s)) or 1) for s in selected),
                        default=0.0
                    )
                    score = mmr_lambda * rel - (1 - mmr_lambda) * sim
                if score > best_score:
                    best_score = score
                    best = r
            selected.append(best)
            candidates.remove(best)
        return selected
    
    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder
        
        Args:
            query: Original query
            results: Initial retrieval results
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked results
        """
        if not self.config.enable_reranking or not results:
            return results[:top_k] if top_k else results
        
        top_k = top_k or self.config.rerank_top_k
        
        # Generate pairs for reranking
        pairs = [(query, result["content"]) for result in results]

        # Get reranking scores
        if self.learned_ranker and self.config.enable_learned_ranker:
            rerank_scores = await self.learned_ranker.score(query, results)
        elif self.reranker:
            rerank_scores = await self.reranker.score(pairs)
        else:
            # Placeholder: use existing scores with small perturbation
            rerank_scores = [
                result["score"] + np.random.normal(0, 0.01)
                for result in results
            ]
        
        # Update scores and sort
        for result, score in zip(results, rerank_scores):
            result["rerank_score"] = score
            result["original_retrieval_score"] = result["score"]
            result["score"] = score
        
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results[:top_k]
    
    # Whitelist of allowed filter fields for security
    ALLOWED_FILTER_FIELDS: Set[str] = {
        'doc_id', 'chunk_id', 'domain_density', 'timestamp', 
        'entropy', 'redundancy', 'chunk_index', 'token_count'
    }
    
    ALLOWED_OPERATORS: Set[str] = {'$gte', '$lte', '$gt', '$lt', '$eq', '$ne'}
    
    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """
        Build Milvus filter expression from filter dict with security validation
        
        Example filters:
        {
            "doc_id": "doc123",
            "domain_density": {"$gte": 0.5},
            "timestamp": {"$gte": "2024-01-01"}
        }
        """
        expressions = []
        
        for field, value in filters.items():
            # SECURITY: Whitelist validation
            if field not in self.ALLOWED_FILTER_FIELDS:
                logger.warning(f"Invalid filter field attempted: {field}")
                raise ValueError(f"Invalid filter field: {field}")
            
            # SECURITY: Field name format validation (prevent injection via field names)
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', field):
                logger.warning(f"Invalid field name format: {field}")
                raise ValueError(f"Invalid field name format: {field}")
            
            if isinstance(value, dict):
                # Handle comparison operators
                for op, val in value.items():
                    # SECURITY: Validate operator
                    if op not in self.ALLOWED_OPERATORS:
                        logger.warning(f"Invalid operator attempted: {op}")
                        raise ValueError(f"Invalid operator: {op}")
                    
                    # SECURITY: Type validation
                    if not isinstance(val, (int, float, str, bool)):
                        raise ValueError(f"Invalid value type for {field}: {type(val)}")
                    
                    op_map = {
                        '$gte': '>=', '$lte': '<=', 
                        '$gt': '>', '$lt': '<',
                        '$eq': '==', '$ne': '!='
                    }
                    
                    if isinstance(val, str):
                        # Escape special characters properly
                        safe_val = val.replace("\\", "\\\\").replace('"', '\\"')
                        expressions.append(f'{field} {op_map[op]} "{safe_val}"')
                    else:
                        expressions.append(f'{field} {op_map[op]} {val}')
            else:
                # Direct equality
                if isinstance(value, str):
                    # Escape special characters
                    safe_val = value.replace("\\", "\\\\").replace('"', '\\"')
                    expressions.append(f'{field} == "{safe_val}"')
                elif isinstance(value, (int, float, bool)):
                    expressions.append(f'{field} == {value}')
                else:
                    raise ValueError(f"Unsupported value type for {field}: {type(value)}")
        
        return " and ".join(expressions) if expressions else None
    
    async def _get_semantic_embedding(self, text: str) -> np.ndarray:
        """Get semantic embedding for query"""
        return await self.index_manager._generate_semantic_embedding(text)
    
    async def _get_sparse_embedding(self, text: str) -> np.ndarray:
        """Get sparse embedding for query"""
        return await self.index_manager._generate_sparse_embedding(text)
    
    async def _get_domain_embedding(
        self,
        text: str,
        domain: str
    ) -> np.ndarray:
        """Get domain-specific embedding for query"""
        return await self.index_manager._generate_domain_embedding(text, domain)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for fine-grained relevance scoring
    Placeholder for integration with actual models (e.g., ms-marco-MiniLM)
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None  # Load actual model in production
    
    async def score(self, pairs: List[tuple]) -> List[float]:
        """
        Score query-document pairs
        
        Args:
            pairs: List of (query, document) tuples
            
        Returns:
            List of relevance scores
        """
        if self.model:  # pragma: no cover
            # Use actual cross-encoder model
            scores = self.model.predict(pairs)
            return scores.tolist()
        
        # Placeholder: return dummy scores
        return [0.5 + np.random.randn() * 0.1 for _ in pairs]
