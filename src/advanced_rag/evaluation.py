"""
RAG Evaluation Framework
Quantifies hallucination risk, faithfulness, and retrieval drift
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import asyncio


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for RAG retrieval"""
    # Retrieval quality
    retrieval_precision: float
    retrieval_recall: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    
    # Hallucination risk
    hallucination_risk: float
    faithfulness_score: float
    
    # Coverage and diversity
    coverage_score: float
    diversity_score: float
    
    # Confidence metrics
    confidence_score: float
    uncertainty_estimate: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "retrieval_precision": self.retrieval_precision,
            "retrieval_recall": self.retrieval_recall,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "ndcg_at_k": self.ndcg_at_k,
            "hallucination_risk": self.hallucination_risk,
            "faithfulness_score": self.faithfulness_score,
            "coverage_score": self.coverage_score,
            "diversity_score": self.diversity_score,
            "confidence_score": self.confidence_score,
            "uncertainty_estimate": self.uncertainty_estimate
        }


@dataclass
class DriftReport:
    """Report on retrieval drift detection"""
    drift_detected: bool
    drift_magnitude: float
    embedding_divergence: float
    distribution_shift: float
    temporal_decay: float
    affected_queries: List[str]
    recommendations: List[str]


class RAGEvaluator:
    """
    Evaluates RAG system quality through multiple lenses:
    - Retrieval quality metrics
    - Hallucination risk quantification
    - Faithfulness to source documents
    - Drift detection in embedding space (cosine-based divergence)
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.15,
        hallucination_threshold: float = 0.2
    ):
        """
        Args:
            drift_threshold: Threshold for drift detection
            hallucination_threshold: Threshold for hallucination risk
        """
        self.drift_threshold = drift_threshold
        self.hallucination_threshold = hallucination_threshold
        
        # Historical data for drift detection - use deque with maxlen to prevent memory leak
        self.query_embeddings_history = deque(maxlen=1000)
        self.score_distributions_history = deque(maxlen=1000)
        self.timestamp_history = deque(maxlen=1000)
        
        # NLI model for faithfulness (placeholder)
        self.nli_model = None
    
    async def evaluate_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate retrieval quality comprehensively
        
        Args:
            query: Search query
            results: Retrieved results
            context: Optional context with ground truth, expected docs, etc.
            
        Returns:
            EvaluationMetrics object
        """
        # Extract ground truth if available
        ground_truth_ids = context.get("relevant_doc_ids", []) if context else []
        
        # Calculate retrieval metrics
        precision = self._calculate_precision(results, ground_truth_ids)
        recall = self._calculate_recall(results, ground_truth_ids)
        mrr = self._calculate_mrr(results, ground_truth_ids)
        ndcg = self._calculate_ndcg(results, ground_truth_ids)
        
        # Calculate hallucination risk
        hallucination_risk = await self._estimate_hallucination_risk(
            query, results
        )
        
        # Calculate faithfulness
        faithfulness = await self._calculate_faithfulness(results)
        
        # Calculate coverage and diversity
        coverage = self._calculate_coverage(results, query)
        diversity = self._calculate_diversity(results)
        
        # Calculate confidence metrics
        confidence = self._calculate_confidence(results)
        uncertainty = self._calculate_uncertainty(results)
        
        # Track score distribution history for drift detection
        if results:
            scores = np.array([r["score"] for r in results], dtype=np.float32)
            # Convert scores to a probability-like distribution using softmax
            exp_scores = np.exp(scores - np.max(scores))
            prob_dist = exp_scores / (np.sum(exp_scores) + 1e-12)
            self.score_distributions_history.append(prob_dist)
        
        return EvaluationMetrics(
            retrieval_precision=precision,
            retrieval_recall=recall,
            mean_reciprocal_rank=mrr,
            ndcg_at_k=ndcg,
            hallucination_risk=hallucination_risk,
            faithfulness_score=faithfulness,
            coverage_score=coverage,
            diversity_score=diversity,
            confidence_score=confidence,
            uncertainty_estimate=uncertainty
        )
    
    def _calculate_precision(
        self,
        results: List[Dict],
        ground_truth: List[str]
    ) -> float:
        """Calculate precision at k"""
        if not ground_truth or not results:
            return 0.0
        
        retrieved_ids = {r["id"] for r in results}
        relevant_retrieved = len(retrieved_ids & set(ground_truth))
        
        return relevant_retrieved / len(results)
    
    def _calculate_recall(
        self,
        results: List[Dict],
        ground_truth: List[str]
    ) -> float:
        """Calculate recall"""
        if not ground_truth or not results:
            return 0.0
        
        retrieved_ids = {r["id"] for r in results}
        relevant_retrieved = len(retrieved_ids & set(ground_truth))
        
        return relevant_retrieved / len(ground_truth)
    
    def _calculate_mrr(
        self,
        results: List[Dict],
        ground_truth: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank"""
        if not ground_truth or not results:
            return 0.0
        
        for rank, result in enumerate(results, start=1):
            if result["id"] in ground_truth:
                return 1.0 / rank
        
        return 0.0
    
    def _calculate_ndcg(
        self,
        results: List[Dict],
        ground_truth: List[str],
        k: Optional[int] = None
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if not ground_truth or not results:
            return 0.0
        
        k = k or len(results)
        results = results[:k]
        
        # Calculate DCG
        dcg = 0.0
        for rank, result in enumerate(results, start=1):
            relevance = 1.0 if result["id"] in ground_truth else 0.0
            dcg += relevance / np.log2(rank + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = sorted([1.0] * min(len(ground_truth), k), reverse=True)
        idcg = sum(
            rel / np.log2(rank + 1)
            for rank, rel in enumerate(ideal_relevances, start=1)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    async def _estimate_hallucination_risk(
        self,
        query: str,
        results: List[Dict]
    ) -> float:
        """
        Estimate hallucination risk using multiple signals:
        1. Score distribution consistency
        2. Content similarity across retrieved documents
        3. Semantic coherence
        4. Negative sampling evaluation
        """
        if not results:
            return 1.0  # Maximum risk if no results
        
        # Signal 1: Score distribution
        scores = [r["score"] for r in results]
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        
        # High variance in scores suggests uncertainty
        score_risk = min(score_std / (score_mean + 1e-10), 1.0)
        
        # Signal 2: Content similarity (low diversity might indicate poor retrieval)
        content_similarity = self._calculate_pairwise_similarity(results)
        diversity_risk = 1.0 - content_similarity if content_similarity > 0.8 else 0.0
        
        # Signal 3: Top score magnitude (low top score = high risk)
        top_score_risk = 1.0 - min(scores[0], 1.0) if scores else 1.0
        
        # Signal 4: Coverage of query terms
        query_terms = set(query.lower().split())
        coverage_scores = []
        for result in results[:5]:  # Check top 5
            content_terms = set(result["content"].lower().split())
            coverage = len(query_terms & content_terms) / len(query_terms)
            coverage_scores.append(coverage)
        
        coverage_risk = 1.0 - np.mean(coverage_scores) if coverage_scores else 1.0
        
        # Weighted combination
        hallucination_risk = (
            0.25 * score_risk +
            0.2 * diversity_risk +
            0.3 * top_score_risk +
            0.25 * coverage_risk
        )
        
        return hallucination_risk
    
    async def _calculate_faithfulness(self, results: List[Dict]) -> float:
        """
        Calculate faithfulness score using NLI or semantic similarity
        Measures if retrieved content is faithful to source
        """
        if not results:
            return 0.0
        
        # If NLI model available, use it for entailment checking
        if self.nli_model:  # pragma: no cover - requires external NLI model
            faithfulness_scores = []
            for result in results:
                # Check if result content is faithful (entailment)
                score = await self._nli_entailment(result["content"])
                faithfulness_scores.append(score)
            return np.mean(faithfulness_scores)
        
        # Fallback: use metadata signals
        # Documents with high redundancy might be less faithful
        redundancy_scores = [
            1.0 - result["metadata"].get("redundancy", 0.5)
            for result in results
        ]
        
        return np.mean(redundancy_scores)
    
    def _calculate_coverage(self, results: List[Dict], query: str) -> float:  # pragma: no cover
        """Calculate query term coverage in results"""
        if not results:
            return 0.0
        
        query_terms = set(query.lower().split())
        
        # Check coverage across all results
        all_content = " ".join([r["content"] for r in results])
        content_terms = set(all_content.lower().split())
        
        coverage = len(query_terms & content_terms) / len(query_terms)
        return coverage
    
    def _calculate_diversity(self, results: List[Dict]) -> float:
        """Calculate diversity of retrieved results"""
        if len(results) < 2:
            return 1.0
        
        # Calculate pairwise similarity and return inverse
        similarity = self._calculate_pairwise_similarity(results)
        diversity = 1.0 - similarity
        
        return diversity
    
    def _calculate_pairwise_similarity(self, results: List[Dict]) -> float:
        """Calculate average pairwise similarity between results"""
        if len(results) < 2:
            return 0.0
        
        similarities = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                # Simple Jaccard similarity on tokens
                tokens_i = set(results[i]["content"].lower().split())
                tokens_j = set(results[j]["content"].lower().split())
                
                if tokens_i and tokens_j:
                    similarity = len(tokens_i & tokens_j) / len(tokens_i | tokens_j)
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_confidence(self, results: List[Dict]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not results:
            return 0.0
        
        scores = [r["score"] for r in results]
        
        # Confidence based on top score and score gap
        top_score = scores[0]
        score_gap = scores[0] - scores[1] if len(scores) > 1 else 0.0
        
        # High top score and large gap = high confidence
        confidence = top_score * (1 + score_gap)
        
        return min(confidence, 1.0)
    
    def _calculate_uncertainty(self, results: List[Dict]) -> float:
        """Calculate uncertainty estimate"""
        if not results:
            return 1.0
        
        scores = [r["score"] for r in results]
        
        # Uncertainty based on score variance
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        
        # Coefficient of variation as uncertainty
        uncertainty = score_std / (score_mean + 1e-10)
        
        return min(uncertainty, 1.0)
    
    async def detect_drift(
        self,
        queries: List[str],
        index_manager: 'MilvusIndexManager'
    ) -> DriftReport:
        """
        Detect retrieval drift using embedding space divergence
        
        Args:
            queries: Sample queries for drift analysis
            index_manager: Index manager for generating embeddings
            
        Returns:
            DriftReport with drift analysis
        """
        # Generate embeddings for queries
        current_embeddings = []
        for query in queries:
            emb = await index_manager._generate_semantic_embedding(query)
            current_embeddings.append(emb)
        
        current_embeddings = np.array(current_embeddings)
        
        # Calculate drift if we have historical data
        if len(self.query_embeddings_history) > 0:
            # Calculate embedding space divergence
            historical_embeddings = np.array(self.query_embeddings_history[-100:])
            
            # KL divergence approximation in embedding space
            embedding_divergence = self._calculate_embedding_divergence(
                historical_embeddings, current_embeddings
            )
            
            # Distribution shift in scores
            if len(self.score_distributions_history) > 0:
                distribution_shift = self._calculate_distribution_shift()
            else:
                distribution_shift = 0.0
            
            # Temporal decay (how old is historical data)
            if self.timestamp_history:
                time_diff = (datetime.now() - self.timestamp_history[-1]).total_seconds()
                temporal_decay = min(time_diff / (30 * 24 * 3600), 1.0)  # Normalize to 30 days
            else:
                temporal_decay = 0.0
            
            # Combine signals for drift magnitude
            drift_magnitude = (
                0.5 * float(embedding_divergence) +
                0.3 * float(distribution_shift) +
                0.2 * float(temporal_decay)
            )
            
            drift_detected = bool(drift_magnitude > self.drift_threshold)
            
            # Identify affected queries
            affected_queries = []
            if drift_detected:
                # Find queries with high divergence
                for i, query in enumerate(queries):
                    query_emb = current_embeddings[i]
                    divergence = self._point_divergence(
                        query_emb, historical_embeddings
                    )
                    if divergence > self.drift_threshold:
                        affected_queries.append(query)
            
            # Generate recommendations
            recommendations = self._generate_drift_recommendations(
                drift_magnitude, embedding_divergence, distribution_shift
            )
            
        else:
            # No historical data - cannot detect drift
            drift_detected = False
            drift_magnitude = 0.0
            embedding_divergence = 0.0
            distribution_shift = 0.0
            temporal_decay = 0.0
            affected_queries = []
            recommendations = ["Insufficient historical data for drift detection"]
        
        # Store current data
        self.query_embeddings_history.extend(current_embeddings.tolist())
        self.timestamp_history.append(datetime.now())
        
        # Keep rolling window
        if len(self.query_embeddings_history) > 1000:
            self.query_embeddings_history = self.query_embeddings_history[-1000:]
            self.timestamp_history = self.timestamp_history[-1000:]
        
        return DriftReport(
            drift_detected=bool(drift_detected),
            drift_magnitude=float(drift_magnitude),
            embedding_divergence=float(embedding_divergence),
            distribution_shift=float(distribution_shift),
            temporal_decay=float(temporal_decay),
            affected_queries=affected_queries,
            recommendations=recommendations
        )
    
    def _calculate_embedding_divergence(
        self,
        historical: np.ndarray,
        current: np.ndarray
    ) -> float:
        """Calculate divergence between historical and current embeddings (cosine)"""
        # Use mean shift via cosine distance as divergence measure
        historical_mean = np.mean(historical, axis=0)
        current_mean = np.mean(current, axis=0)
        
        # Cosine distance
        divergence = 1 - np.dot(historical_mean, current_mean) / (
            np.linalg.norm(historical_mean) * np.linalg.norm(current_mean)
        )
        
        return divergence
    
    def _calculate_distribution_shift(self) -> float:
        """Calculate shift in score distributions"""
        if len(self.score_distributions_history) < 2:
            return 0.0
        
        # Compare last two distributions using KL divergence
        recent = self.score_distributions_history[-1]
        previous = self.score_distributions_history[-2]
        
        # Add small epsilon for numerical stability
        recent = recent + 1e-10
        previous = previous + 1e-10
        
        kl_div = np.sum(recent * np.log(recent / previous))
        
        return min(kl_div, 1.0)
    
    def _point_divergence(
        self,
        point: np.ndarray,
        distribution: np.ndarray
    ) -> float:
        """Calculate divergence of a point from a distribution"""
        # Average cosine distance to distribution
        distances = []
        for hist_point in distribution:
            dist = 1 - np.dot(point, hist_point) / (
                np.linalg.norm(point) * np.linalg.norm(hist_point)
            )
            distances.append(dist)
        
        return np.mean(distances)
    
    def _generate_drift_recommendations(
        self,
        drift_magnitude: float,
        embedding_divergence: float,
        distribution_shift: float
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        if drift_magnitude > 0.3:
            recommendations.append("CRITICAL: Significant drift detected - consider retraining embeddings")
        
        if embedding_divergence > 0.2:
            recommendations.append("Embedding space has shifted - review query distribution")
        
        if distribution_shift > 0.25:
            recommendations.append("Score distribution has changed - recalibrate retrieval thresholds")
        
        if drift_magnitude > self.drift_threshold:
            recommendations.append("Consider A/B testing new vs old embeddings")
            recommendations.append("Increase monitoring frequency")
        
        return recommendations
    
    async def _nli_entailment(self, text: str) -> float:  # pragma: no cover
        """Check entailment using NLI model (placeholder)"""
        # In production, use actual NLI model
        return 0.8  # Placeholder
