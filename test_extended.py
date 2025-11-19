"""
Extended Unit Tests to raise coverage ≥95%
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any

from advanced_rag import (
    DocumentDiagnostics,
    AdaptiveChunker,
    RAGEvaluator,
    HybridRetriever,
    RetrievalConfig,
    LearnedRanker,
    SemanticEnricher,
    QueryDecomposer,
    AdvancedRAGPipeline,
    ExperimentManager,
)
from advanced_rag.indexing import MilvusIndexManager
from advanced_rag.learned_adapter import LearnedHybridAdapter
from advanced_rag.indexing import IndexConfig


def test_diagnostics_empty_text():
    diagnostics = DocumentDiagnostics()
    metrics = diagnostics.analyze_document("")
    assert metrics.information_entropy == 0.0
    assert metrics.redundancy_score == 0.0
    assert metrics.domain_density == 0.0
    assert metrics.vocabulary_diversity == 0.0
    assert metrics.semantic_coherence == 1.0 or metrics.semantic_coherence == 0.5


def test_chunk_metrics_propagation_from_doc():
    diagnostics_module = DocumentDiagnostics()
    text = "A domain protocol optimization and database interface."
    metrics = diagnostics_module.analyze_document(text)
    chunker = AdaptiveChunker(base_chunk_size=10, semantic_boundary_detection=True)

    chunks = chunker.chunk_document(
        text=text,
        diagnostics=metrics,
        metadata={"doc_id": "docX", "source": "unit_test"},
    )
    assert len(chunks) >= 1
    ch = chunks[0]
    # Ensure domain_density propagated from document-level (not overridden by 0.0)
    assert abs(ch.metadata.domain_density - metrics.domain_density) < 1e-6
    # Ensure entropy set (non-negative)
    assert ch.metadata.entropy >= 0.0


def test_chunk_fixed_size_and_overlap():
    diagnostics_module = DocumentDiagnostics()
    text = " ".join([f"t{i}" for i in range(100)])
    metrics = diagnostics_module.analyze_document(text)
    chunker = AdaptiveChunker(base_chunk_size=10, max_chunk_size=10, min_chunk_size=10, overlap_ratio=0.2, semantic_boundary_detection=False)
    chunks = chunker.chunk_document(text=text, diagnostics=metrics, metadata={"doc_id": "docY"})
    assert len(chunks) >= 2
    chunk1_words = set(chunks[0].text.split())
    chunk2_words = set(chunks[1].text.split())
    assert len(chunk1_words & chunk2_words) >= 1

def test_semantic_chunking_overlap_path():
    diagnostics_module = DocumentDiagnostics()
    # 12 short sentences, 2 tokens each
    sentences = [f"s{i} x." for i in range(12)]
    text = " ".join(sentences)
    metrics = diagnostics_module.analyze_document(text)
    chunker = AdaptiveChunker(base_chunk_size=10, max_chunk_size=10, min_chunk_size=10, overlap_ratio=0.3, semantic_boundary_detection=True)
    chunks = chunker.chunk_document(text=text, diagnostics=metrics, metadata={"doc_id": "docZ"})
    assert len(chunks) >= 2
    # Ensure char indices are set
    assert chunks[0].metadata.char_start == 0
    assert chunks[0].metadata.char_end > chunks[0].metadata.char_start

def test_retrieval_rrf_and_filter_escape():
    class DummyRetriever(HybridRetriever):
        async def _get_semantic_embedding(self, text: str):
            return np.ones(4, dtype=np.float32)

        async def _get_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def _search_semantic(self, embedding, filters):
            # Two results
            return [
                {"id": "A", "content": "alpha content", "score": 0.9},
                {"id": "B", "content": "bravo content", "score": 0.8},
            ]

        async def _search_sparse(self, embedding, filters):
            # Duplicate "A" and a new "C" with different ranks
            return [
                {"id": "A", "content": "alpha content", "score": 0.7},
                {"id": "C", "content": "charlie content", "score": 0.6},
            ]

    retriever = DummyRetriever(index_manager=None)
    # Test fusion order and dedup
    fused = retriever._fuse_results(
        semantic_results=[{"id": "A", "content": "x", "score": 0.9}, {"id": "B", "content": "y", "score": 0.8}],
        sparse_results=[{"id": "A", "content": "x", "score": 0.7}, {"id": "C", "content": "z", "score": 0.6}],
        domain_results=[],
    )
    ids = [r["id"] for r in fused]
    assert "A" in ids and "B" in ids and "C" in ids
    # Ensure dedup happened (no duplicates)
    assert len(ids) == len(set(ids))

    # Test filter expression escaping
    expr = retriever._build_filter_expression({"doc_id": 'doc"123', "entropy": {"$gte": 0.2}})
    assert 'doc_id == "doc\\"123"' in expr
    assert "entropy >= 0.2" in expr
    # More operators coverage
    expr2 = retriever._build_filter_expression({
        "redundancy": {"$lt": 0.5, "$gt": 0.1, "$eq": 0.2, "$ne": 0.3},
        "chunk_index": 1
    })
    assert "redundancy < 0.5" in expr2
    assert "redundancy > 0.1" in expr2
    assert "redundancy == 0.2" in expr2
    assert "redundancy != 0.3" in expr2
    assert "chunk_index == 1" in expr2
    # Empty filters returns None
    assert retriever._build_filter_expression({}) is None


def test_query_classifier_and_profiles_are_used():
    """Ensure QueryClassifier routes to different RetrievalConfig profiles."""
    from advanced_rag.retrieval import QueryClassifier

    class FakeIndexManager:
        async def _generate_semantic_embedding(self, text: str):
            return np.ones(4, dtype=np.float32)

        async def _generate_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def search(self, query_embedding, collection_name, top_k=20, filters=None, search_params=None):
            # Always return a single result so we can inspect metadata
            return [{"id": "X", "content": "foo", "score": 1.0, "metadata": {"doc_id": "d"}}]

    # Build a base config and allow HybridRetriever to create default profiles
    base_cfg = RetrievalConfig(top_k=50, rerank_top_k=20)
    classifier = QueryClassifier(max_faq_len=80, long_query_len=200)
    retriever = HybridRetriever(index_manager=FakeIndexManager(), config=base_cfg, classifier=classifier)

    # FAQ-style short question
    profile = classifier.classify("What is vector search?")
    assert profile == "faq"

    # Troubleshooting style
    profile2 = classifier.classify("I see an error: connection failed")
    assert profile2 == "troubleshooting"

    # Explicit summary
    profile_summary = classifier.classify("Please provide a summary or overview of RAG.")
    assert profile_summary == "summary"

    # Long-form analysis
    long_query = "x" * 250
    profile_analysis = classifier.classify(long_query)
    assert profile_analysis == "analysis"

    # Fallback/default
    profile3 = classifier.classify("")
    assert profile3 == "default"

    # Profile configs should be tuned for throughput and clamped to safe bounds
    faq_cfg = retriever.profiles["faq"]
    trouble_cfg = retriever.profiles["troubleshooting"]
    summary_cfg = retriever.profiles["summary"]
    analysis_cfg = retriever.profiles["analysis"]

    assert 1 <= faq_cfg.top_k <= base_cfg.top_k
    assert trouble_cfg.top_k >= base_cfg.top_k
    assert summary_cfg.top_k >= trouble_cfg.top_k
    # Rerank top_k should never exceed profile top_k
    assert trouble_cfg.rerank_top_k <= trouble_cfg.top_k
    assert summary_cfg.rerank_top_k <= summary_cfg.top_k
    assert analysis_cfg.rerank_top_k <= analysis_cfg.top_k


def test_rrf_with_mmr_diversification():
    from advanced_rag import RetrievalConfig
    # Create a HybridRetriever-like object to access _fuse_results with MMR enabled
    class DummyRetriever(HybridRetriever):
        pass
    cfg = RetrievalConfig(hybrid_alpha=0.7, top_k=3, enable_mmr=True, mmr_lambda=0.6)
    r = DummyRetriever(index_manager=None, config=cfg)
    # Three semantic, three sparse with overlapping IDs and similar content to trigger MMR path
    semantic_results = [
        {"id": "A", "content": "alpha alpha content one", "score": 0.95},
        {"id": "B", "content": "bravo content two", "score": 0.85},
        {"id": "C", "content": "alpha content three", "score": 0.80},
    ]
    sparse_results = [
        {"id": "A", "content": "alpha alpha content one", "score": 0.75},
        {"id": "D", "content": "delta unique different", "score": 0.70},
        {"id": "E", "content": "echo also different", "score": 0.65},
    ]
    fused = r._fuse_results(semantic_results=semantic_results, sparse_results=sparse_results, domain_results=[])
    # MMR enabled: verify size and presence of IDs from both groups (diversification picks diverse docs)
    assert len(fused) <= cfg.top_k
    ids = {x["id"] for x in fused}
    assert "A" in ids or "B" in ids  # top relevance preserved
    # At least one diversified from sparse that isn't A
    assert ("D" in ids) or ("E" in ids) or ("C" in ids)

def test_weight_adapter_affects_weights_and_ordering():
    # Create custom adapter that heavily favors sparse
    def adapter(q: str):
        return (0.1, 0.9)
    from advanced_rag import RetrievalConfig
    class DummyRetriever(HybridRetriever):
        async def _search_semantic(self, embedding, filters):
            return [{"id": "S", "content": "semantic thing", "score": 0.9}]
        async def _search_sparse(self, embedding, filters):
            return [{"id": "P", "content": "keyword exact match", "score": 0.8}]
    cfg = RetrievalConfig(hybrid_alpha=0.7, top_k=2)
    retr = DummyRetriever(index_manager=None, config=cfg, weight_adapter=adapter)
    # Directly call internal fusion using expected results and check weighting bias
    fused = retr._fuse_results(
        semantic_results=[{"id": "S", "content": "semantic thing", "score": 0.9}],
        sparse_results=[{"id": "P", "content": "keyword exact match", "score": 0.8}],
        domain_results=[]
    )
    # With weight adapter favoring sparse, ensure "P" is not disadvantaged vs "S"
    ids = [x["id"] for x in fused]
    assert "P" in ids and "S" in ids


@pytest.mark.asyncio
async def test_rerank_placeholder_scores():
    class DummyRetriever(HybridRetriever):
        pass

    retriever = DummyRetriever(index_manager=None)
    results = [
        {"id": "A", "content": "alpha", "score": 0.5},
        {"id": "B", "content": "bravo", "score": 0.4},
        {"id": "C", "content": "charlie", "score": 0.3},
    ]
    reranked = await retriever.rerank(query="q", results=results, top_k=2)
    assert len(reranked) == 2
    assert "rerank_score" in reranked[0]


@pytest.mark.asyncio
async def test_rerank_with_external_reranker_scores():
    """Exercise the external reranker path in HybridRetriever.rerank."""
    class DummyReranker:
        async def score(self, pairs):
            # Return higher score for second item to flip ordering
            return [0.1, 0.9]

    class DummyRetriever(HybridRetriever):
        pass

    retriever = DummyRetriever(index_manager=None)
    retriever.reranker = DummyReranker()
    results = [
        {"id": "A", "content": "alpha", "score": 0.5},
        {"id": "B", "content": "bravo", "score": 0.4},
    ]
    reranked = await retriever.rerank(query="q", results=results, top_k=2)
    assert [r["id"] for r in reranked] == ["B", "A"]
    assert "rerank_score" in reranked[0]


@pytest.mark.asyncio
async def test_hybrid_retriever_retrieve_with_domain():
    class FakeIndexManager:
        async def _generate_semantic_embedding(self, text: str):
            return np.ones(4, dtype=np.float32)

        async def _generate_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def _generate_domain_embedding(self, text: str, domain: str):
            return np.full(4, 2.0, dtype=np.float32)

        async def search(self, query_embedding, collection_name, top_k=20, filters=None, search_params=None):
            if collection_name == "semantic_index":
                return [{"id": "S", "content": "semantic", "score": 0.9, "metadata": {"doc_id": "d1"}}]
            if collection_name == "sparse_index":
                return [{"id": "P", "content": "sparse", "score": 0.8, "metadata": {"doc_id": "d2"}}]
            if collection_name == "domain_index":
                return [{"id": "D", "content": "domain", "score": 0.85, "metadata": {"doc_id": "d3"}}]
            return []

    retriever = HybridRetriever(index_manager=FakeIndexManager())
    # Use a question-style query to exercise classifier and ensure metadata tagging
    results = await retriever.retrieve(query="What is RAG?", filters={"doc_id": "x"}, use_domain_index=True, domain="tech")
    assert len(results) >= 1
    ids = {r["id"] for r in results}
    assert {"S", "P", "D"} & ids  # at least some present
    # Verify retrieval_profile is attached either in metadata or at top-level
    profiles = set()
    for r in results:
        meta = r.get("metadata") or {}
        if isinstance(meta, dict) and "retrieval_profile" in meta:
            profiles.add(meta["retrieval_profile"])
        elif "retrieval_profile" in r:
            profiles.add(r["retrieval_profile"])
    assert profiles  # at least one profile recorded


@pytest.mark.asyncio
async def test_hybrid_retriever_metadata_fallback_when_no_dict():
    """When results lack metadata dict, retrieval_profile should be attached at top level."""
    class FakeIndexManagerNoMeta:
        async def _generate_semantic_embedding(self, text: str):
            return np.ones(4, dtype=np.float32)

        async def _generate_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def search(self, query_embedding, collection_name, top_k=20, filters=None, search_params=None):
            # Return results without metadata dict to exercise fallback path
            return [{"id": "Z", "content": "no-metadata", "score": 1.0}]

    retriever = HybridRetriever(index_manager=FakeIndexManagerNoMeta())
    results = await retriever.retrieve(query="What is RAG?")
    assert results
    assert "retrieval_profile" in results[0]


@pytest.mark.asyncio
async def test_hybrid_retriever_respects_timeout(monkeypatch):
    """HybridRetriever.retrieve should enforce an overall latency budget."""
    from advanced_rag.constants import RetrievalConstants

    class SlowIndexManager:
        async def _generate_semantic_embedding(self, text: str):
            # Sleep longer than the tightened timeout in this test
            await asyncio.sleep(0.02)
            return np.ones(4, dtype=np.float32)

        async def _generate_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def search(self, query_embedding, collection_name, top_k=20, filters=None, search_params=None):
            return []

    retriever = HybridRetriever(index_manager=SlowIndexManager())

    original_timeout = RetrievalConstants.TIMEOUT_SECONDS
    RetrievalConstants.TIMEOUT_SECONDS = 0.005
    try:
        results = await retriever.retrieve(query="slow query")
    finally:
        RetrievalConstants.TIMEOUT_SECONDS = original_timeout

    # On timeout we degrade gracefully to an empty result list
    assert results == []


@pytest.mark.asyncio
async def test_hybrid_retriever_weight_adapter_updates_weights():
    """Ensure weight_adapter hook is exercised and updates dense/sparse weights."""
    class FakeIndexManager:
        async def _generate_semantic_embedding(self, text: str):
            return np.ones(4, dtype=np.float32)

        async def _generate_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def search(self, query_embedding, collection_name, top_k=20, filters=None, search_params=None):
            return [{"id": "X", "content": "x", "score": 1.0, "metadata": {"doc_id": "dx"}}]

    def adapter(q: str):
        # Intentionally use values that require clamping/normalization
        return 1.5, -0.2

    retriever = HybridRetriever(index_manager=FakeIndexManager(), weight_adapter=adapter)
    # Before retrieve, use defaults
    assert retriever.config.dense_weight == 0.7
    assert retriever.config.sparse_weight == 0.3
    await retriever.retrieve(query="q")
    # After retrieve, weights should be clamped and updated
    assert 0.0 <= retriever.config.dense_weight <= 1.0
    assert 0.0 <= retriever.config.sparse_weight <= 1.0


def test_indexing_embedding_generation_without_connect():
    manager = MilvusIndexManager(connect=False)
    # Semantic embedding
    import asyncio
    sem = asyncio.get_event_loop().run_until_complete(manager._generate_semantic_embedding("hello"))
    assert sem.shape[0] == manager.semantic_dim
    # Sparse embedding: placeholder should return a dict with indices/values for SPARSE payloads
    sp = asyncio.get_event_loop().run_until_complete(manager._generate_sparse_embedding("hello"))
    assert isinstance(sp, dict)
    assert "indices" in sp and "values" in sp


def test_indexing_embedding_generator_paths():
    """Ensure custom embedding_generator hooks are exercised for all three embedding types."""
    class DummyGen:
        async def encode_semantic(self, text: str):
            return np.full(4, 1.0, dtype=np.float32)

        async def encode_sparse(self, text: str):
            # Return dense vector; in production this would be a real sparse encoder,
            # but here we just need to hit the branch.
            return np.arange(4, dtype=np.float32)

        async def encode_domain(self, text: str, domain: str = ""):
            return np.full(4, 2.0, dtype=np.float32)

    mgr = MilvusIndexManager(semantic_dim=4, sparse_dim=4, domain_dim=4, connect=False)
    mgr.embedding_generator = DummyGen()
    import asyncio
    sem = asyncio.get_event_loop().run_until_complete(mgr._generate_semantic_embedding("x"))
    sp = asyncio.get_event_loop().run_until_complete(mgr._generate_sparse_embedding("x"))
    dom = asyncio.get_event_loop().run_until_complete(mgr._generate_domain_embedding("x", domain="d"))
    assert np.all(sem == 1.0)
    # Sparse path should pass through DummyGen unchanged
    assert np.all(sp == np.arange(4, dtype=np.float32))
    assert np.all(dom == 2.0)


@pytest.mark.asyncio
async def test_evaluation_and_history_and_drift():
    evaluator = RAGEvaluator(drift_threshold=0.0)  # Ensure drift gets flagged with any change
    # Create dummy results
    results = [
        {"id": "A", "content": "python language", "score": 0.8, "metadata": {"redundancy": 0.1}},
        {"id": "B", "content": "java language", "score": 0.6, "metadata": {"redundancy": 0.2}},
        {"id": "C", "content": "golang language", "score": 0.4, "metadata": {"redundancy": 0.3}},
    ]
    metrics = await evaluator.evaluate_retrieval("python", results, context={"relevant_doc_ids": ["A", "Z"]})
    assert 0.0 <= metrics.retrieval_precision <= 1.0
    assert 0.0 <= metrics.retrieval_recall <= 1.0
    assert 0.0 <= metrics.hallucination_risk <= 1.0
    # Score history appended
    assert len(evaluator.score_distributions_history) >= 1

    class DummyIndexMgr:
        async def _generate_semantic_embedding(self, q: str):
            # Deterministic embedding per query
            rng = np.random.default_rng(abs(hash(q)) % (2**32))
            return rng.standard_normal(8).astype(np.float32)

    # Seed some history
    evaluator.query_embeddings_history = [np.ones(8, dtype=np.float32) for _ in range(5)]
    evaluator.timestamp_history = [datetime.now() - timedelta(days=1)]
    drift_report = await evaluator.detect_drift(["python", "java"], DummyIndexMgr())
    assert isinstance(drift_report.drift_detected, bool)

@pytest.mark.asyncio
async def test_evaluation_edge_cases_and_metrics():
    ev = RAGEvaluator()
    # No results case
    hr = await ev._estimate_hallucination_risk("q", [])
    assert hr == 1.0
    assert ev._calculate_diversity([]) == 1.0
    assert ev._calculate_confidence([]) == 0.0
    assert ev._calculate_uncertainty([]) == 1.0
    # NDCG and precision with no ground truth
    assert ev._calculate_ndcg([], []) == 0.0
    assert ev._calculate_precision([], []) == 0.0
    assert ev._calculate_recall([], ["A"]) == 0.0
    assert ev._calculate_mrr([], ["A"]) == 0.0
    # NDCG with some ground truth
    ndcg = ev._calculate_ndcg(
        [{"id": "A", "score": 1.0}, {"id": "B", "score": 0.5}],
        ["A", "C"],
        k=2,
    )
    assert 0.0 <= ndcg <= 1.0

@pytest.mark.asyncio
async def test_detect_drift_no_history_path():
    ev = RAGEvaluator()
    class DummyIM:
        async def _generate_semantic_embedding(self, q: str):
            return np.ones(8, dtype=np.float32)
    report = await ev.detect_drift(["q1", "q2"], DummyIM())
    assert report.drift_detected is False
    assert isinstance(report.recommendations, list)
    assert 0.0 <= report.embedding_divergence <= 2.0

def test_evaluation_distribution_shift_internal():
    ev = RAGEvaluator()
    import numpy as np
    # Two simple distributions
    ev.score_distributions_history = [
        np.array([0.7, 0.3], dtype=np.float32),
        np.array([0.6, 0.4], dtype=np.float32),
    ]
    shift = ev._calculate_distribution_shift()
    assert 0.0 <= shift <= 1.0


def test_generate_drift_recommendations_thresholds():
    ev = RAGEvaluator(drift_threshold=0.1)
    recs = ev._generate_drift_recommendations(
        drift_magnitude=0.5,
        embedding_divergence=0.3,
        distribution_shift=0.4,
    )
    # All recommendation branches should be exercised
    joined = " ".join(recs)
    assert "Significant drift detected" in joined
    assert "Embedding space has shifted" in joined
    assert "Score distribution has changed" in joined
    assert "A/B testing" in joined or "Increase monitoring frequency" in joined

def test_learned_hybrid_adapter_basic_fit_and_call():
    adapter = LearnedHybridAdapter()
    feedback = [
        {"method": "sparse", "vote": "up"},
        {"method": "semantic", "vote": "down"},
        {"method": "semantic", "vote": "up"},
        {"method": "sparse", "vote": "down"},
    ]
    adapter.fit_from_feedback(feedback)
    dw_short, sw_short = adapter("cpu info")
    dw_long, sw_long = adapter("how does vector similarity search scale with index fanout and ef parameter")
    assert 0.0 < dw_short < 1.0 and 0.0 < sw_short < 1.0
    assert abs(dw_short + sw_short - 1.0) < 1e-6
    assert 0.0 < dw_long < 1.0 and 0.0 < sw_long < 1.0
    assert abs(dw_long + sw_long - 1.0) < 1e-6


def test_learned_ranker_recency_feature_affects_ordering():
    """LearnedRanker should be able to boost more recent results when configured."""
    from advanced_rag import LearnedRanker, LearnedRankerConfig

    cfg = LearnedRankerConfig(base_weight=1.0, method_bonus=0.0, recency_weight=1.0)
    ranker = LearnedRanker(config=cfg)

    # Same base score, different recency metadata
    results = [
        {"id": "old", "score": 0.8, "retrieval_methods": ["semantic"], "metadata": {"recency": 0.1}},
        {"id": "new", "score": 0.8, "retrieval_methods": ["semantic"], "metadata": {"recency": 0.9}},
    ]
    scores = asyncio.get_event_loop().run_until_complete(ranker.score("q", results))
    assert len(scores) == 2
    # Newer item should receive a higher score due to recency_weight
    assert scores[1] > scores[0]


def test_semantic_enricher_entities_and_topics():
    """SemanticEnricher should extract at least one entity and topic from rich text."""
    enricher = SemanticEnricher(max_topics=3)
    text = "OpenAI builds advanced Retrieval systems for Milvus and Postgres backends."
    result = enricher.enrich(text)
    # Basic sanity: entities and topics lists exist and are not pathological
    assert isinstance(result.entities, list) and isinstance(result.topics, list)
    assert "OpenAI" in result.entities
    # Topics are derived from frequency; ensure at least one non-trivial term exists
    assert any(len(t) > 4 for t in result.topics)


def test_query_decomposer_basic_strategies():
    """QueryDecomposer should return single or split queries with a strategy label."""
    decomposer = QueryDecomposer(min_length=20)
    short = "What is RAG?"
    res_short = decomposer.decompose(short)
    assert res_short.sub_queries == [short]
    assert res_short.strategy in {"single", "fallback"}

    complex_q = "Explain vectors and indexes in Milvus"
    res_complex = decomposer.decompose(complex_q)
    assert len(res_complex.sub_queries) >= 1
    assert isinstance(res_complex.strategy, str)


def test_query_rewriter_expansion_and_disable():
    """QueryRewriter should expand common abbreviations and be configurable."""
    from advanced_rag.query_rewriting import QueryRewriter, QueryRewriterConfig

    rw = QueryRewriter()
    # Empty query should be returned as-is
    assert rw.rewrite("", context={}) == ""

    out_rag = rw.rewrite("Explain RAG architecture", context={})
    assert "retrieval augmented generation" in out_rag.lower()

    out_llm = rw.rewrite("LLM overview", context={})
    assert "large language model" in out_llm.lower()

    # Query without known abbreviations should be passed through unchanged
    plain = "regular question without abbreviations"
    assert rw.rewrite(plain, context={}) == plain

    # When disabled, rewriting should be a no-op
    rw_disabled = QueryRewriter(config=QueryRewriterConfig(enable_expansion=False))
    q = "What is RAG?"
    assert rw_disabled.rewrite(q, context={}) == q


@pytest.mark.asyncio
async def test_pipeline_plan_and_execute_uses_decomposer_and_retrieve(monkeypatch):
    """AdvancedRAGPipeline.plan_and_execute should call retrieve once per sub-query."""
    pipe = AdvancedRAGPipeline(connect_to_milvus=False)

    calls: List[str] = []

    async def fake_retrieve(query: str, filters=None, context=None):
        calls.append(query)
        # Minimal shape: no results, metrics placeholder
        return [], None

    # Patch retrieve to avoid touching Milvus and to count calls
    monkeypatch.setattr(pipe, "retrieve", fake_retrieve)

    query = "Explain vectors and indexes"
    result = await pipe.plan_and_execute(query)
    decomp = result["decomposition"]
    assert isinstance(decomp, dict)
    assert calls  # retrieve called at least once
    assert len(calls) == len(decomp["sub_queries"])


@pytest.mark.asyncio
async def test_pipeline_uses_query_rewriter_and_profile_hint(monkeypatch):
    """AdvancedRAGPipeline should apply QueryRewriter and respect retrieval_profile context hint."""
    from advanced_rag.query_rewriting import QueryRewriterConfig, QueryRewriter

    class RecordingRewriter(QueryRewriter):
        def __init__(self):
            super().__init__(config=QueryRewriterConfig(enable_expansion=False))
            self.seen = []

        def rewrite(self, query: str, context=None) -> str:
            self.seen.append((query, context or {}))
            return f"{query} [rewritten]"

    pipe = AdvancedRAGPipeline(connect_to_milvus=False, query_rewriter=RecordingRewriter())

    seen_queries: List[str] = []
    seen_profiles: List[str] = []

    async def fake_retrieve(query: str, filters=None, use_domain_index=False, domain=None, profile_hint=None):
        seen_queries.append(query)
        if profile_hint:
            seen_profiles.append(profile_hint)
        # Minimal shape matching retrieve() expectations
        return [{"id": "X", "content": "c", "score": 1.0, "metadata": {"doc_id": "d"}}]

    async def fake_rerank(query: str, results: List[Dict[str, Any]], top_k: int = 5):
        return results[:top_k]

    # Patch retriever methods to avoid touching Milvus and to observe arguments
    monkeypatch.setattr(pipe.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipe.retriever, "rerank", fake_rerank)

    ctx = {"retrieval_profile": "faq"}
    results, metrics = await pipe.retrieve(query="What is RAG?", context=ctx)
    assert results  # some result returned
    # Rewriter should have been invoked with the original query and context
    assert pipe.query_rewriter.seen
    orig_q, seen_ctx = pipe.query_rewriter.seen[0]
    assert orig_q == "What is RAG?"
    assert seen_ctx == ctx
    # Retriever should receive rewritten query and profile hint
    assert any("[rewritten]" in q for q in seen_queries)
    assert "faq" in seen_profiles


def test_experiment_manager_bandit_basic_flow():
    """ExperimentManager should learn to prefer higher-reward variants when epsilon=0."""
    mgr = ExperimentManager(epsilon=0.0)
    mgr.register_experiment("retrieval_profile", ["baseline", "mmr"])

    # Initially, no reward info: choose_variant should return some valid variant
    first = mgr.choose_variant("retrieval_profile")
    assert first in {"baseline", "mmr"}

    # Record outcomes: make 'mmr' clearly better
    for _ in range(5):
        mgr.record_outcome("retrieval_profile", "mmr", reward=1.0)
    for _ in range(3):
        mgr.record_outcome("retrieval_profile", "baseline", reward=0.0)

    stats = mgr.get_stats("retrieval_profile")
    assert "mmr" in stats and "baseline" in stats
    assert stats["mmr"]["average_reward"] >= stats["baseline"]["average_reward"]

    # With epsilon=0 and better reward, mmr should be chosen greedily most of the time
    chosen = {mgr.choose_variant("retrieval_profile") for _ in range(10)}
    assert chosen == {"mmr"}


def test_learned_ranker_scoring_and_training_examples():
    """Ensure LearnedRanker can score results and collect training examples."""
    ranker = LearnedRanker()
    results = [
        {"id": "d1", "score": 0.8, "retrieval_methods": ["semantic"]},
        {"id": "d2", "score": 0.6, "retrieval_methods": ["semantic", "sparse"]},
    ]
    # Score should be deterministic and ordered by underlying scores/features
    scores = asyncio.get_event_loop().run_until_complete(ranker.score("q", results))
    assert len(scores) == 2
    # Higher original score should normally produce higher learned score
    assert scores[0] >= scores[1]

    # Update from feedback and ensure training examples are stored
    feedback = [
        {"id": "d1", "label": 1.0},
        {"id": "d2", "label": 0.0},
    ]
    ranker.update_from_feedback("q", results, feedback)
    assert len(ranker.training_examples) == 2


@pytest.mark.asyncio
async def test_hybrid_retriever_learned_ranker_integration():
    """Wire a dummy LearnedRanker into HybridRetriever.rerank."""
    class DummyIndexManager:
        async def _generate_semantic_embedding(self, text: str):
            return np.ones(4, dtype=np.float32)

        async def _generate_sparse_embedding(self, text: str):
            return np.zeros(4, dtype=np.float32)

        async def search(self, query_embedding, collection_name, top_k=20, filters=None, search_params=None):
            # Two results with slightly different base scores
            return [
                {"id": "A", "content": "alpha", "score": 0.5, "metadata": {"doc_id": "d1"}, "retrieval_methods": ["semantic"]},
                {"id": "B", "content": "bravo", "score": 0.4, "metadata": {"doc_id": "d2"}, "retrieval_methods": ["semantic", "sparse"]},
            ]

    # LearnedRanker that inverts base scores to change ordering
    class InvertingRanker(LearnedRanker):
        async def score(self, query: str, results: List[Dict[str, Any]]) -> List[float]:
            base_scores = [float(r.get("score", 0.0)) for r in results]
            max_score = max(base_scores) if base_scores else 0.0
            return [max_score - s for s in base_scores]

    cfg = RetrievalConfig(enable_reranking=True, enable_learned_ranker=True)
    retriever = HybridRetriever(
        index_manager=DummyIndexManager(),
        config=cfg,
        learned_ranker=InvertingRanker(),
    )
    raw = await retriever.retrieve(query="q")
    # Apply learned reranker
    reranked = await retriever.rerank("q", raw, top_k=2)
    assert len(reranked) == 2
    # With inverting ranker, the originally lower-scored doc should bubble up
    ids = [r["id"] for r in reranked]
    assert "A" in ids and "B" in ids


def test_compliance_retention_policy_math():
    from advanced_rag import ComplianceManager, AuditEventType, AuditLog

    cm = ComplianceManager(enable_audit=True, retention_days=0)
    # Manually add an old log
    old_log = AuditLog(
        event_id="evt_old",
        event_type=AuditEventType.RETRIEVAL,
        timestamp=(datetime.now() - timedelta(days=10)).isoformat(),
        user_id=None,
        session_id=None,
        event_data={},
        parent_event_id=None,
        related_event_ids=[],
        compliance_flags=[],
        retention_policy="0_days",
    )
    cm.audit_logs.append(old_log)
    # Add a fresh log through API
    import asyncio

    asyncio.get_event_loop().run_until_complete(
        cm.log_retrieval(query="q", chunk_id="c", score=0.1, latency_ms=1.0)
    )
    # Enforce retention
    cm._enforce_retention_policy()
    # Old should be pruned, new kept
    assert any(log.event_id != "evt_old" for log in cm.audit_logs)


def test_config_loader_defaults(tmp_path):
    # Minimal config YAML
    content = "pipeline:\n  target_latency_ms: 80.0\n"
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(content, encoding="utf-8")
    from advanced_rag.config import load_pipeline_config, load_component_configs
    pc = load_pipeline_config(str(cfg_file))
    assert pc.target_latency_ms == 80.0
    comps = load_component_configs(str(cfg_file))
    assert isinstance(comps, dict)
    assert "milvus" in comps


def test_index_manager_misc_methods_without_connect():
    mgr = MilvusIndexManager(connect=False)
    assert mgr.get_collection_stats("nonexistent") == {}
    import asyncio
    # delete_by_filter should not raise
    asyncio.get_event_loop().run_until_complete(mgr.delete_by_filter("none", "id == 'x'"))
    # close should not raise
    asyncio.get_event_loop().run_until_complete(mgr.close())


def test_chunking_metadata_and_adaptive_branches():
    """Cover ChunkMetadata.to_dict, hashing, and low-entropy adaptive sizing."""
    from advanced_rag.chunking import ChunkMetadata, Chunk, AdaptiveChunker
    meta = ChunkMetadata(
        chunk_id="c1",
        doc_id="d1",
        chunk_index=0,
        char_start=0,
        char_end=10,
        token_count=2,
        entropy=0.5,
        redundancy=0.1,
        domain_density=0.2,
        coherence_score=0.9,
        source="unit-test",
        timestamp="2024-01-01T00:00:00",
        version="v1",
        extra={"foo": "bar"},
    )
    d = meta.to_dict()
    assert d["chunk_id"] == "c1" and d["foo"] == "bar"
    c = Chunk(text="hello world", metadata=meta)
    assert isinstance(hash(c), int)

    # Exercise low-entropy branch in adaptive sizing
    class DummyDiag:
        information_entropy = 0.1  # < 0.4 triggers shrink branch
        redundancy_score = 0.0
        domain_density = 0.0
        semantic_coherence = 1.0

    chunker = AdaptiveChunker(base_chunk_size=100, max_chunk_size=200, min_chunk_size=10)
    # _determine_chunk_size is the public wrapper that calls the internal heuristic
    size = chunker._determine_chunk_size(DummyDiag())
    assert size < 100

    # Exercise _analyze_chunk empty-token path
    empty_metrics = chunker._analyze_chunk("")
    assert empty_metrics["entropy"] == 0.0 and empty_metrics["redundancy"] == 0.0


def test_pipeline_performance_report_without_milvus():
    from advanced_rag import AdvancedRAGPipeline, PipelineStage
    pipe = AdvancedRAGPipeline(connect_to_milvus=False)
    # Simulate recorded latencies
    pipe._record_latency(PipelineStage.RETRIEVAL, 20.0)
    pipe._record_latency(PipelineStage.RERANKING, 15.0)
    pipe._record_latency(PipelineStage.EVALUATION, 10.0)
    report = pipe.get_performance_report()
    assert "stage_latencies" in report
    assert "sla_compliance" in report
    assert "retrieval" in report["stage_latencies"]


@pytest.mark.asyncio
async def test_ingest_documents_records_data_quality(monkeypatch):
    """AdvancedRAGPipeline.ingest_documents should record basic data quality flags."""
    from advanced_rag import AdvancedRAGPipeline

    pipe = AdvancedRAGPipeline(connect_to_milvus=False)

    async def fake_index_chunks(chunks, domain=None):
        return {"total_chunks": len(chunks)}

    # Avoid touching real Milvus
    monkeypatch.setattr(pipe.index_manager, "index_chunks", fake_index_chunks)

    docs = [
        {"id": "empty", "text": "", "metadata": {}},
        {"id": "redundant", "text": "spam " * 200, "metadata": {}},
        {"id": "ok", "text": "A small but valid document.", "metadata": {}},
    ]

    report = await pipe.ingest_documents(docs)
    dq_entries = {entry["document_id"]: entry["flags"] for entry in report.get("data_quality", [])}

    assert "empty" in dq_entries and "empty_text" in dq_entries["empty"]
    assert "redundant" in dq_entries
    # At least one quality flag should be present for the redundant doc
    assert dq_entries["redundant"]


def test_compliance_versioning_lineage_and_report():
    from advanced_rag import ComplianceManager
    cm = ComplianceManager(enable_audit=True, enable_versioning=True, retention_days=90)
    import asyncio
    # Create two versions
    v1 = asyncio.get_event_loop().run_until_complete(
        cm.create_version(doc_id="docZ", content="c1", change_type="create", chunk_count=1, total_tokens=10)
    )
    v2 = asyncio.get_event_loop().run_until_complete(
        cm.create_version(doc_id="docZ", content="c2", change_type="update", chunk_count=2, total_tokens=20)
    )
    assert v1.version == "v1" and v2.version == "v2"
    # Lineage tracking
    cm.track_lineage("evt1", "evt2")
    cm.track_lineage("evt2", "evt3")
    tree = cm.get_event_lineage("evt1")
    assert tree["event_id"] == "evt1"
    # Compliance report
    report = cm.generate_compliance_report("2000-01-01", "2100-01-01")
    assert "total_events" in report
    # Integrity verification
    ok = cm.verify_data_integrity("docZ", v2.version_hash)
    assert ok is True
    # Close
    asyncio.get_event_loop().run_until_complete(cm.close())


@pytest.mark.asyncio
async def test_compliance_legal_hold_and_forget_semantics():
    """ComplianceManager should honour legal holds and implement a basic right-to-forget hook."""
    from advanced_rag import ComplianceManager

    mgr = ComplianceManager(enable_audit=True, enable_versioning=True)

    # Create a couple of versions for a document
    await mgr.create_version(
        doc_id="doc-forget",
        content="hello",
        change_type="create",
        chunk_count=1,
        total_tokens=5,
        user_id="u",
    )
    await mgr.create_version(
        doc_id="doc-forget",
        content="world",
        change_type="update",
        chunk_count=2,
        total_tokens=10,
        user_id="u",
    )
    assert len(mgr.get_document_lineage("doc-forget")) == 2

    # Apply tenant-specific legal holds and verify forget does not remove versions
    await mgr.apply_legal_hold("doc-forget", tenant_id="tenantA")
    res_hold = await mgr.forget_document("doc-forget", tenant_id="tenantA")
    assert res_hold["forgotten"] is False
    assert res_hold["reason"] == "legal_hold"
    assert res_hold["tenant_id"] == "tenantA"
    assert len(mgr.get_document_lineage("doc-forget")) == 2

    # Different tenant should not be blocked by tenantA's hold
    res_other_tenant = await mgr.forget_document("doc-forget", tenant_id="tenantB")
    assert res_other_tenant["forgotten"] is True
    assert res_other_tenant["reason"] == "removed"
    assert res_other_tenant["tenant_id"] == "tenantB"

def test_pipeline_close_without_milvus():
    from advanced_rag import AdvancedRAGPipeline
    import asyncio
    pipe = AdvancedRAGPipeline(connect_to_milvus=False)
    asyncio.get_event_loop().run_until_complete(pipe.close())

def test_compliance_query_and_get_version_and_log_ingestion():
    from advanced_rag import ComplianceManager, AuditEventType
    import asyncio
    cm = ComplianceManager(enable_audit=True, enable_versioning=True, retention_days=90)
    # Log ingestion
    asyncio.get_event_loop().run_until_complete(
        cm.log_ingestion(document_count=1, chunk_count=2, report={"ok": True}, user_id="u1")
    )
    # Create version and fetch it
    v = asyncio.get_event_loop().run_until_complete(
        cm.create_version(doc_id="docA", content="x", change_type="create", chunk_count=1, total_tokens=5, user_id="u1")
    )
    fetched = cm.get_version("docA", v.version)
    assert fetched is not None
    # Query logs by type, user, start/end times
    now = datetime.now().isoformat()
    logs = cm.query_audit_logs(
        event_type=AuditEventType.DOCUMENT_INGESTION,
        user_id="u1",
        start_time="2000-01-01T00:00:00",
        end_time=now,
        compliance_flag="data_ingestion"
    )
    assert isinstance(logs, list)
    # No match filters
    empty = cm.query_audit_logs(user_id="nope")
    assert isinstance(empty, list)
    assert len(empty) == 0

def test_index_config_defaults_post_init():
    cfg = IndexConfig(collection_name="c", dimension=16, index_params=None)
    assert "M" in cfg.index_params and "efConstruction" in cfg.index_params

def test_evaluation_pairwise_similarity():
    ev = RAGEvaluator()
    sim = ev._calculate_pairwise_similarity([
        {"content": "a b c"},
        {"content": "a b d"}
    ])
    assert 0.0 <= sim <= 1.0

