"""
Hybrid Retrieval Engine
Combines dense vector search, sparse BM25, and cross-encoder reranking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import asyncio
from typing import Callable, Tuple


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
        weight_adapter: Optional[Callable[[str], Tuple[float, float]]] = None
    ):
        """
        Args:
            index_manager: MilvusIndexManager instance
            config: Retrieval configuration
        """
        self.index_manager = index_manager
        self.config = config or RetrievalConfig()
        self.weight_adapter = weight_adapter
        
        # Reranker (placeholder - integrate actual cross-encoder)
        self.reranker = None  # Will be set externally
    
    async def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        use_domain_index: bool = False,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining multiple search strategies
        
        Args:
            query: Search query
            filters: Optional metadata filters
            use_domain_index: Whether to include domain-specific index
            domain: Domain identifier
            
        Returns:
            List of retrieved documents with scores
        """
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
        
        # Return top-k
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        for doc_id, info in fused_scores.items():
            result = info["data"]
            result["score"] = info["score"]
            result["retrieval_methods"] = list(set(info["methods"]))
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
        if self.reranker:
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
    
    def _build_filter_expression(self, filters: Dict[str, Any]) -> str:
        """
        Build Milvus filter expression from filter dict
        
        Example filters:
        {
            "doc_id": "doc123",
            "domain_density": {"$gte": 0.5},
            "timestamp": {"$gte": "2024-01-01"}
        }
        """
        expressions = []
        
        for field, value in filters.items():
            if isinstance(value, dict):
                # Handle comparison operators
                for op, val in value.items():
                    if op == "$gte":
                        expressions.append(f'{field} >= {val}')
                    elif op == "$lte":
                        expressions.append(f'{field} <= {val}')
                    elif op == "$gt":
                        expressions.append(f'{field} > {val}')
                    elif op == "$lt":
                        expressions.append(f'{field} < {val}')
                    elif op == "$eq":
                        expressions.append(f'{field} == {val}')
                    elif op == "$ne":
                        expressions.append(f'{field} != {val}')
            else:
                # Direct equality
                if isinstance(value, str):
                    # Escape double quotes and backslashes in strings
                    safe_val = value.replace("\\", "\\\\").replace('"', '\\"')
                    expressions.append(f'{field} == "{safe_val}"')
                else:
                    expressions.append(f'{field} == {value}')
        
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
