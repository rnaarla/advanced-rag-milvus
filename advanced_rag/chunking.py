"""
Adaptive Chunking Module
Diagnostic-informed chunking with variable granularity
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk"""
    chunk_id: str
    doc_id: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    
    # Diagnostic-informed metadata
    entropy: float
    redundancy: float
    domain_density: float
    coherence_score: float
    
    # Source tracking
    source: str
    timestamp: str
    version: str
    
    # Additional metadata
    extra: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "entropy": self.entropy,
            "redundancy": self.redundancy,
            "domain_density": self.domain_density,
            "coherence_score": self.coherence_score,
            "source": self.source,
            "timestamp": self.timestamp,
            "version": self.version,
            **self.extra
        }


@dataclass
class Chunk:
    """Document chunk with text and metadata"""
    text: str
    metadata: ChunkMetadata
    
    def __hash__(self):
        return hash(self.metadata.chunk_id)


class ChunkingStrategy:
    """Base strategy for chunking"""
    
    def __init__(self, target_size: int, overlap: int):
        self.target_size = target_size
        self.overlap = overlap


class AdaptiveChunker:
    """
    Adaptive document chunker that adjusts granularity based on
    diagnostic metrics to optimize retrieval precision
    """
    
    def __init__(
        self,
        base_chunk_size: int = 512,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 128,
        overlap_ratio: float = 0.15,
        semantic_boundary_detection: bool = True
    ):
        """
        Args:
            base_chunk_size: Default chunk size in tokens
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
            overlap_ratio: Ratio of overlap between chunks
            semantic_boundary_detection: Use semantic boundaries for splitting
        """
        self.base_chunk_size = base_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_ratio = overlap_ratio
        self.semantic_boundary_detection = semantic_boundary_detection
    
    def chunk_document(
        self,
        text: str,
        diagnostics: 'DiagnosticMetrics',
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk document with adaptive granularity based on diagnostics
        
        Args:
            text: Document text to chunk
            diagnostics: DiagnosticMetrics from document analysis
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        metadata = metadata or {}
        
        # Determine optimal chunk size based on diagnostics
        chunk_size = self._determine_chunk_size(diagnostics)
        overlap = int(chunk_size * self.overlap_ratio)
        
        # Select chunking method
        if self.semantic_boundary_detection:
            chunks = self._semantic_chunking(
                text, chunk_size, overlap, diagnostics
            )
        else:
            chunks = self._fixed_size_chunking(text, chunk_size, overlap)
        
        # Create chunk objects with metadata
        chunk_objects = []
        doc_id = metadata.get("doc_id", self._generate_id(text))
        version = metadata.get("version", "1.0")
        
        for idx, (chunk_text, char_start, char_end, chunk_metrics) in enumerate(chunks):
            chunk_id = self._generate_chunk_id(doc_id, idx, chunk_text)
            token_count = len(self._tokenize(chunk_text))
            
            chunk_metadata = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=idx,
                char_start=char_start,
                char_end=char_end,
                token_count=token_count,
                entropy=chunk_metrics.get("entropy", diagnostics.information_entropy),
                redundancy=chunk_metrics.get("redundancy", diagnostics.redundancy_score),
                domain_density=chunk_metrics.get("domain_density", diagnostics.domain_density),
                coherence_score=chunk_metrics.get("coherence", diagnostics.semantic_coherence),
                source=metadata.get("source", "unknown"),
                timestamp=metadata.get("timestamp", datetime.now().isoformat()),
                version=version,
                extra={k: v for k, v in metadata.items() 
                       if k not in ["doc_id", "source", "timestamp", "version"]}
            )
            
            chunk_objects.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata
            ))
        
        return chunk_objects
    
    def _determine_chunk_size(self, diagnostics: 'DiagnosticMetrics') -> int:
        """
        Determine optimal chunk size based on document characteristics
        
        Heuristics:
        - High entropy → larger chunks (more diverse information)
        - High redundancy → smaller chunks (avoid repetition)
        - High domain density → smaller chunks (preserve precision)
        - Low coherence → smaller chunks (maintain semantic units)
        """
        # Start with base size
        chunk_size = self.base_chunk_size
        
        # Adjust for entropy (high entropy = more info per token)
        if diagnostics.information_entropy > 0.8:
            chunk_size = int(chunk_size * 1.3)
        elif diagnostics.information_entropy < 0.4:
            chunk_size = int(chunk_size * 0.8)
        
        # Adjust for redundancy (high redundancy = smaller chunks)
        if diagnostics.redundancy_score > 0.6:
            chunk_size = int(chunk_size * 0.7)
        
        # Adjust for domain density (high density = smaller chunks for precision)
        if diagnostics.domain_density > 0.3:
            chunk_size = int(chunk_size * 0.85)
        
        # Adjust for coherence (low coherence = smaller chunks)
        if diagnostics.semantic_coherence < 0.3:
            chunk_size = int(chunk_size * 0.75)
        
        # Apply bounds
        chunk_size = max(self.min_chunk_size, min(chunk_size, self.max_chunk_size))
        
        return chunk_size
    
    def _semantic_chunking(
        self,
        text: str,
        target_size: int,
        overlap: int,
        diagnostics: 'DiagnosticMetrics'
    ) -> List[tuple]:
        """
        Chunk text along semantic boundaries (sentences, paragraphs)
        
        Returns:
            List of tuples: (chunk_text, char_start, char_end, metrics_dict)
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        char_position = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence_tokens = len(self._tokenize(sentence))
            
            # Check if adding this sentence exceeds target
            if current_tokens + sentence_tokens > target_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunk_metrics = self._analyze_chunk(chunk_text)
                chunks.append((
                    chunk_text,
                    chunk_start,
                    char_position,
                    chunk_metrics
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, overlap
                )
                current_chunk = overlap_sentences
                current_tokens = sum(len(self._tokenize(s)) for s in current_chunk)
                chunk_start = char_position - sum(len(s) + 1 for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
            char_position += len(sentence) + 1  # +1 for space
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_metrics = self._analyze_chunk(chunk_text)
            chunks.append((
                chunk_text,
                chunk_start,
                len(text),
                chunk_metrics
            ))
        
        return chunks
    
    def _fixed_size_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[tuple]:
        """
        Chunk text using fixed token windows
        
        Returns:
            List of tuples: (chunk_text, char_start, char_end, metrics_dict)
        """
        tokens = self._tokenize(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            
            # Estimate character positions (approximate)
            char_start = len(" ".join(tokens[:i]))
            char_end = char_start + len(chunk_text)
            
            chunk_metrics = self._analyze_chunk(chunk_text)
            chunks.append((
                chunk_text,
                char_start,
                min(char_end, len(text)),
                chunk_metrics
            ))
        
        return chunks
    
    def _analyze_chunk(self, chunk_text: str) -> Dict[str, float]:
        """Quick analysis of chunk characteristics"""
        tokens = self._tokenize(chunk_text)
        
        if not tokens:
            return {
                "entropy": 0.0,
                "redundancy": 0.0,
                # Intentionally omit domain_density and coherence so document-level metrics propagate
            }
        
        # Simple entropy calculation
        from collections import Counter
        token_counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in token_counts.values()
        ) / math.log2(len(token_counts)) if len(token_counts) > 1 else 0
        
        # Simple redundancy
        unique_ratio = len(set(tokens)) / len(tokens)
        redundancy = 1 - unique_ratio
        
        return {
            "entropy": entropy,
            "redundancy": redundancy,
            # Domain density and coherence not computed here; rely on document-level diagnostics
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Enhanced sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_tokens: int
    ) -> List[str]:
        """Get sentences for overlap window"""
        overlap_sentences = []
        token_count = 0
        
        # Take sentences from the end
        for sentence in reversed(sentences):
            sentence_tokens = len(self._tokenize(sentence))
            if token_count + sentence_tokens > overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            token_count += sentence_tokens
        
        return overlap_sentences
    
    def _generate_id(self, content: str) -> str:
        """Generate deterministic ID from content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _generate_chunk_id(self, doc_id: str, index: int, content: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"{doc_id}::{index}::{content_hash}"


import math  # Need to import at module level
