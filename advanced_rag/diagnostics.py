"""
Document Diagnostics Module
Quantifies information entropy, redundancy, and domain density
"""

import re
import math
from typing import Dict, List, Set, Tuple
from typing import Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np
from scipy.stats import entropy as scipy_entropy


@dataclass
class DiagnosticMetrics:
    """Diagnostic metrics for a document"""
    information_entropy: float  # Shannon entropy of token distribution
    redundancy_score: float  # Normalized repetition measure (0-1)
    domain_density: float  # Concentration of domain-specific terms (0-1)
    vocabulary_diversity: float  # Type-token ratio
    semantic_coherence: float  # Estimated coherence score
    avg_sentence_complexity: float  # Average tokens per sentence
    
    # Detailed breakdowns
    token_distribution: Dict[str, float]
    n_gram_redundancy: Dict[int, float]  # Redundancy by n-gram size
    domain_terms: Set[str]
    
    def to_dict(self) -> Dict:
        return {
            "information_entropy": self.information_entropy,
            "redundancy_score": self.redundancy_score,
            "domain_density": self.domain_density,
            "vocabulary_diversity": self.vocabulary_diversity,
            "semantic_coherence": self.semantic_coherence,
            "avg_sentence_complexity": self.avg_sentence_complexity,
            "n_gram_redundancy": self.n_gram_redundancy,
            "num_domain_terms": len(self.domain_terms)
        }


class DocumentDiagnostics:
    """
    Analyze document characteristics to inform chunking and retrieval strategies
    """
    
    def __init__(self, domain_lexicons: Optional[Dict[str, Set[str]]] = None):
        """
        Args:
            domain_lexicons: Optional mapping of domain names to term sets
        """
        self.domain_lexicons = domain_lexicons or self._default_domain_lexicons()
        
        # Common English words for filtering (simplified stopwords)
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they'
        }
    
    def analyze_document(self, text: str) -> DiagnosticMetrics:
        """
        Perform comprehensive document diagnostics
        
        Args:
            text: Document text to analyze
            
        Returns:
            DiagnosticMetrics object with all measurements
        """
        # Tokenization
        tokens = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        # Core metrics
        entropy = self._calculate_entropy(tokens)
        redundancy = self._calculate_redundancy(tokens)
        domain_density, domain_terms = self._calculate_domain_density(tokens)
        vocab_diversity = self._calculate_vocabulary_diversity(tokens)
        coherence = self._estimate_coherence(tokens, sentences)
        complexity = self._calculate_sentence_complexity(tokens, sentences)
        
        # Detailed analysis
        token_dist = self._get_token_distribution(tokens)
        ngram_redundancy = self._analyze_ngram_redundancy(tokens)
        
        return DiagnosticMetrics(
            information_entropy=entropy,
            redundancy_score=redundancy,
            domain_density=domain_density,
            vocabulary_diversity=vocab_diversity,
            semantic_coherence=coherence,
            avg_sentence_complexity=complexity,
            token_distribution=token_dist,
            n_gram_redundancy=ngram_redundancy,
            domain_terms=domain_terms
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple word tokenization with normalization
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_entropy(self, tokens: List[str]) -> float:
        """
        Calculate Shannon entropy of token distribution
        Higher entropy = more information diversity
        """
        if not tokens:
            return 0.0
        
        # Get token frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate probabilities
        probabilities = [count / total_tokens for count in token_counts.values()]
        
        # Shannon entropy
        entropy = scipy_entropy(probabilities, base=2)
        
        # Normalize by theoretical maximum (log2 of vocabulary size)
        max_entropy = math.log2(len(token_counts)) if len(token_counts) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_redundancy(self, tokens: List[str]) -> float:
        """
        Calculate document redundancy based on repetition patterns
        0 = no redundancy, 1 = maximum redundancy
        """
        if len(tokens) < 2:
            return 0.0
        
        # Calculate various redundancy measures
        
        # 1. Token-level repetition
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        token_redundancy = 1 - (unique_tokens / total_tokens)
        
        # 2. Bigram repetition
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        bigram_redundancy = 1 - (unique_bigrams / total_bigrams) if total_bigrams > 0 else 0
        
        # 3. Trigram repetition
        if len(tokens) >= 3:
            trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
            unique_trigrams = len(set(trigrams))
            total_trigrams = len(trigrams)
            trigram_redundancy = 1 - (unique_trigrams / total_trigrams) if total_trigrams > 0 else 0
        else:
            trigram_redundancy = 0
        
        # Weighted average of redundancy measures
        redundancy = (
            0.4 * token_redundancy +
            0.35 * bigram_redundancy +
            0.25 * trigram_redundancy
        )
        
        return redundancy
    
    def _calculate_domain_density(self, tokens: List[str]) -> Tuple[float, Set[str]]:
        """
        Calculate concentration of domain-specific terminology
        
        Returns:
            Tuple of (density score, set of identified domain terms)
        """
        # Filter out common words
        content_tokens = [t for t in tokens if t not in self.common_words]
        
        if not content_tokens:
            return 0.0, set()
        
        # Identify domain terms
        domain_terms = set()
        for token in content_tokens:
            for domain, lexicon in self.domain_lexicons.items():
                if token in lexicon:
                    domain_terms.add(token)
        
        # Calculate density
        density = len(domain_terms) / len(content_tokens)
        
        return density, domain_terms
    
    def _calculate_vocabulary_diversity(self, tokens: List[str]) -> float:
        """
        Calculate type-token ratio (vocabulary diversity)
        """
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        # Use square root to reduce impact of document length
        # (Moving-Average Type-Token Ratio variant)
        diversity = unique_tokens / math.sqrt(total_tokens)
        
        # Normalize to 0-1 range (heuristic)
        normalized = min(diversity / 10, 1.0)
        
        return normalized
    
    def _estimate_coherence(self, tokens: List[str], sentences: List[str]) -> float:
        """
        Estimate semantic coherence using lexical cohesion
        """
        if len(sentences) < 2:
            return 1.0
        
        # Calculate sentence overlap as a proxy for coherence
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            sent1_tokens = set(self._tokenize(sentences[i]))
            sent2_tokens = set(self._tokenize(sentences[i + 1]))
            
            # Jaccard similarity
            if sent1_tokens and sent2_tokens:
                overlap = len(sent1_tokens & sent2_tokens)
                union = len(sent1_tokens | sent2_tokens)
                coherence = overlap / union if union > 0 else 0
                coherence_scores.append(coherence)
        
        # Average coherence
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        
        return avg_coherence
    
    def _calculate_sentence_complexity(
        self,
        tokens: List[str],
        sentences: List[str]
    ) -> float:
        """Calculate average tokens per sentence"""
        if not sentences:
            return 0.0
        
        return len(tokens) / len(sentences)
    
    def _get_token_distribution(self, tokens: List[str]) -> Dict[str, float]:
        """Get normalized token frequency distribution"""
        if not tokens:
            return {}
        
        token_counts = Counter(tokens)
        total = len(tokens)
        
        # Return top 20 tokens
        top_tokens = dict(token_counts.most_common(20))
        return {token: count / total for token, count in top_tokens.items()}
    
    def _analyze_ngram_redundancy(self, tokens: List[str]) -> Dict[int, float]:
        """Analyze redundancy for different n-gram sizes"""
        redundancy = {}
        
        for n in range(1, 5):  # Unigrams through 4-grams
            if len(tokens) < n:
                redundancy[n] = 0.0
                continue
            
            # Generate n-grams
            ngrams = [
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]
            
            if ngrams:
                unique_ngrams = len(set(ngrams))
                total_ngrams = len(ngrams)
                redundancy[n] = 1 - (unique_ngrams / total_ngrams)
            else:
                redundancy[n] = 0.0
        
        return redundancy
    
    def _default_domain_lexicons(self) -> Dict[str, Set[str]]:
        """
        Default domain lexicons for common technical domains
        In production, load from external knowledge bases
        """
        return {
            "technical": {
                "algorithm", "architecture", "binary", "cache", "compile",
                "database", "encryption", "framework", "hardware", "interface",
                "kernel", "latency", "memory", "network", "optimization",
                "protocol", "query", "runtime", "server", "thread",
                "variable", "vector", "webhook", "xml", "yaml"
            },
            "medical": {
                "diagnosis", "treatment", "symptom", "patient", "clinical",
                "therapy", "medication", "disease", "syndrome", "pathology",
                "prognosis", "epidemiology", "etiology", "pharmacology"
            },
            "financial": {
                "investment", "portfolio", "equity", "bond", "derivative",
                "dividend", "capital", "asset", "liability", "revenue",
                "margin", "valuation", "liquidity", "leverage", "hedging"
            },
            "legal": {
                "contract", "statute", "regulation", "compliance", "liability",
                "jurisdiction", "plaintiff", "defendant", "testimony", "evidence",
                "precedent", "litigation", "arbitration", "settlement"
            }
        }
