"""
Milvus Index Manager
Multi-layered indexing with semantic, sparse, and domain-specific embeddings
"""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
import os

# Prevent python-dotenv (pulled in by pymilvus) from trying to read .env files
# outside the project root, which can cause permission errors in constrained
# environments (CI sandboxes, containers). Configuration is provided via real
# environment variables instead.
os.environ.setdefault("PYTHON_DOTENV_DISABLE", "true")
os.environ.setdefault("DOTENV_DISABLE", "true")

# As an extra guard, monkey-patch dotenv.load_dotenv to a no-op before pymilvus
# imports it. This makes tests and containers independent of any filesystem-level
# .env files outside the project.
try:  # pragma: no cover - defensive patching
    import dotenv
    from dotenv import main as dotenv_main

    def _noop_load_dotenv(*args, **kwargs):
        return False

    dotenv.load_dotenv = _noop_load_dotenv
    dotenv_main.load_dotenv = _noop_load_dotenv
except Exception:
    pass

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
import asyncio
from datetime import datetime
import logging
from scipy.sparse import csr_matrix, issparse
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .embedding_cache import get_semantic_cache


class IndexType(Enum):
    """Types of indexes maintained"""
    SEMANTIC = "semantic"  # Dense vector embeddings
    SPARSE = "sparse"  # BM25-style sparse vectors
    DOMAIN = "domain"  # Domain-specific ontological embeddings
    HYBRID = "hybrid"  # Combined index


@dataclass
class IndexConfig:
    """Configuration for index creation"""
    collection_name: str
    dimension: int
    index_type: str = "HNSW"  # HNSW, IVF_FLAT, etc.
    metric_type: str = "L2"  # L2, IP, COSINE
    index_params: Optional[Dict] = None
    enable_dynamic_field: bool = True
    
    def __post_init__(self):
        if self.index_params is None:
            # Default HNSW parameters for low-latency
            self.index_params = {
                "M": 16,  # Number of bi-directional links
                "efConstruction": 200  # Build-time search scope
            }


class MilvusIndexManager:
    """
    Manages multiple Milvus collections for layered indexing strategy
    Implements hierarchical indexing and locality-aware sharding
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        enable_sharding: bool = True,
        num_shards: int = 4,
        semantic_dim: int = 1536,  # OpenAI ada-002 dimension
        sparse_dim: int = 10000,  # Sparse vector dimension
        domain_dim: int = 768,  # Domain-specific embedding dimension
        connect: bool = True
    ):
        """
        Args:
            host: Milvus host
            port: Milvus port
            enable_sharding: Enable sharding for scalability
            semantic_dim: Dimension for semantic embeddings
            sparse_dim: Dimension for sparse vectors
            domain_dim: Dimension for domain embeddings
        """
        self.host = host
        self.port = port
        self.enable_sharding = enable_sharding
        self.num_shards = num_shards
        
        # Embedding dimensions
        self.semantic_dim = semantic_dim
        self.sparse_dim = sparse_dim
        self.domain_dim = domain_dim
        
        # Collection references
        self.collections: Dict[str, Collection] = {}
        
        # Embedding generators (placeholders - integrate with actual models)
        self.embedding_generator = None  # Will be set externally
        
        # Thread pool for CPU-intensive embedding operations
        self.embedding_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="embedding-"
        )
        
        # Connect to Milvus
        if connect:
            self._connect()
            # Initialize collections
            self._initialize_collections()
    
    def _connect(self):  # pragma: no cover
        """Establish connection to Milvus"""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        print(f"Connected to Milvus at {self.host}:{self.port}")
    
    def _initialize_collections(self):  # pragma: no cover
        """Create collections for different index types"""
        # Semantic index collection
        self._create_collection(
            IndexConfig(
                collection_name="semantic_index",
                dimension=self.semantic_dim,
                index_type="HNSW",
                metric_type="COSINE",
                index_params={"M": 16, "efConstruction": 200}
            )
        )
        
        # Sparse index collection (for BM25-style retrieval). This path is guarded
        # by an env flag because SPARSE_FLOAT_VECTOR support can differ by Milvus version.
        if os.getenv("ENABLE_SPARSE", "1") == "1":
            self._create_collection(
                IndexConfig(
                    collection_name="sparse_index",
                    dimension=self.sparse_dim,
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",  # Inner product for sparse
                    index_params={}  # use safe defaults
                )
            )
        
        # Domain-specific index collection
        self._create_collection(
            IndexConfig(
                collection_name="domain_index",
                dimension=self.domain_dim,
                index_type="HNSW",
                metric_type="COSINE",
                index_params={"M": 12, "efConstruction": 150}
            )
        )
        
        print("Initialized multi-layered index collections")
    
    def _create_collection(self, config: IndexConfig):  # pragma: no cover
        """Create a Milvus collection with specified configuration"""
        # Check if collection exists
        if utility.has_collection(config.collection_name):
            self.collections[config.collection_name] = Collection(config.collection_name)
            print(f"Loaded existing collection: {config.collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="token_count", dtype=DataType.INT64),
            
            # Diagnostic metrics
            FieldSchema(name="entropy", dtype=DataType.FLOAT),
            FieldSchema(name="redundancy", dtype=DataType.FLOAT),
            FieldSchema(name="domain_density", dtype=DataType.FLOAT),
        ]
        
        # Embedding field:
        # - For dense vectors (FLOAT_VECTOR), Milvus requires a 'dim'
        # - For sparse vectors (SPARSE_FLOAT_VECTOR), 'dim' MUST NOT be provided
        if config.index_type == "SPARSE_INVERTED_INDEX":
            embedding_field = FieldSchema(
                name="embedding",
                dtype=DataType.SPARSE_FLOAT_VECTOR
            )
        else:
            embedding_field = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=config.dimension
            )
        fields.append(embedding_field)
        
        # Metadata (using JSON for flexibility)
        fields.extend([
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=32),
        ])
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Collection for {config.collection_name}",
            enable_dynamic_field=config.enable_dynamic_field
        )
        
        # Create collection with sharding
        num_shards = self.num_shards if self.enable_sharding else 1
        collection = Collection(
            name=config.collection_name,
            schema=schema,
            num_shards=num_shards
        )
        
        # Create index on embedding field
        index_params = {
            "index_type": config.index_type,
            "metric_type": config.metric_type,
            "params": config.index_params
        }
        
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
        
        # Create scalar indexes for filtering
        collection.create_index(field_name="doc_id")
        collection.create_index(field_name="domain_density")
        collection.create_index(field_name="timestamp")
        
        # Load collection into memory
        collection.load()
        
        self.collections[config.collection_name] = collection
        print(f"Created collection: {config.collection_name} with {num_shards} shards")
    
    async def index_chunks(  # pragma: no cover
        self,
        chunks: List['Chunk'],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index chunks across multiple collection types
        
        Args:
            chunks: List of Chunk objects to index
            domain: Optional domain identifier for specialized embeddings
            
        Returns:
            Indexing summary with statistics
        """
        summary = {
            "total_chunks": len(chunks),
            "indexed_semantic": 0,
            "indexed_sparse": 0,
            "indexed_domain": 0,
            "errors": []
        }
        
        # Prepare data for batch insertion
        semantic_data = []
        domain_data = []
        # For sparse vectors, use column-wise insert to satisfy Milvus SPARSE payload requirements.
        # Guarded by presence of collection and feature flag so we can disable at runtime.
        sparse_columns = None
        if "sparse_index" in self.collections and os.getenv("ENABLE_SPARSE", "1") == "1":
            sparse_columns = {
                "id": [],
                "chunk_id": [],
                "doc_id": [],
                "content": [],
                "chunk_index": [],
                "token_count": [],
                "entropy": [],
                "redundancy": [],
                "domain_density": [],
                "embedding": [],
                "metadata_json": [],
                "timestamp": [],
            }
        
        # Batch generate semantic embeddings for all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        semantic_embeddings = await self._generate_semantic_embeddings_batch(chunk_texts)
        
        for i, chunk in enumerate(chunks):
            try:
                # Use pre-generated semantic embedding
                semantic_emb = semantic_embeddings[i]
                
                # Sparse and domain are best‑effort; failures should not break ingest.
                sparse_emb = None
                try:
                    if sparse_columns is not None:
                        sparse_emb = await self._generate_sparse_embedding(chunk.text)
                except Exception as e:
                    summary["errors"].append({
                        "chunk_id": chunk.metadata.chunk_id,
                        "error": f"sparse_embedding_failed: {e}"
                    })
                domain_emb = await self._generate_domain_embedding(chunk.text, domain)
                
                # Prepare common fields
                base_data = {
                    "id": chunk.metadata.chunk_id,
                    "chunk_id": chunk.metadata.chunk_id,
                    "doc_id": chunk.metadata.doc_id,
                    "content": chunk.text[:65535],  # Truncate to max length
                    "chunk_index": chunk.metadata.chunk_index,
                    "token_count": chunk.metadata.token_count,
                    "entropy": chunk.metadata.entropy,
                    "redundancy": chunk.metadata.redundancy,
                    "domain_density": chunk.metadata.domain_density,
                    "metadata_json": str(chunk.metadata.to_dict()),
                    "timestamp": chunk.metadata.timestamp
                }
                
                # Add to respective collections
                semantic_data.append({**base_data, "embedding": semantic_emb})
                domain_data.append({**base_data, "embedding": domain_emb})
                # Column-wise accumulation for sparse (if enabled)
                if sparse_columns is not None and sparse_emb is not None:
                    sparse_columns["id"].append(base_data["id"])
                    sparse_columns["chunk_id"].append(base_data["chunk_id"])
                    sparse_columns["doc_id"].append(base_data["doc_id"])
                    sparse_columns["content"].append(base_data["content"])
                    sparse_columns["chunk_index"].append(base_data["chunk_index"])
                    sparse_columns["token_count"].append(base_data["token_count"])
                    sparse_columns["entropy"].append(base_data["entropy"])
                    sparse_columns["redundancy"].append(base_data["redundancy"])
                    sparse_columns["domain_density"].append(base_data["domain_density"])
                    sparse_columns["embedding"].append(sparse_emb)
                    sparse_columns["metadata_json"].append(base_data["metadata_json"])
                    sparse_columns["timestamp"].append(base_data["timestamp"])
                
            except Exception as e:
                summary["errors"].append({
                    "chunk_id": chunk.metadata.chunk_id,
                    "error": str(e)
                })
        
        # Batch insert into collections
        try:
            if semantic_data:
                await asyncio.to_thread(self.collections["semantic_index"].insert, semantic_data)
                summary["indexed_semantic"] = len(semantic_data)
            
            # Insert sparse using column-wise order; embedding must be a scipy.sparse CSR matrix.
            # Any Milvus‑level error here should be captured and surfaced via summary, not crash the request.
            if sparse_columns is not None and sparse_columns["id"]:
                try:
                    rows = len(sparse_columns["id"])
                    # Build CSR matrix for embeddings from per-row indices/values
                    row_entries = sparse_columns["embedding"]
                    data_parts = []
                    index_parts = []
                    indptr = [0]
                    nnz_total = 0
                    for entry in row_entries:
                        idx = np.array(entry["indices"], dtype=np.int64)
                        vals = np.array(entry["values"], dtype=np.float32)
                        data_parts.append(vals)
                        index_parts.append(idx)
                        nnz_total += len(vals)
                        indptr.append(nnz_total)
                    if nnz_total == 0:
                        # Create an empty CSR with correct shape
                        embedding_matrix = csr_matrix((rows, self.sparse_dim), dtype=np.float32)
                    else:
                        data = np.concatenate(data_parts, axis=0)
                        indices = np.concatenate(index_parts, axis=0)
                        indptr_arr = np.array(indptr, dtype=np.int64)
                        embedding_matrix = csr_matrix(
                            (data, indices, indptr_arr),
                            shape=(rows, self.sparse_dim),
                            dtype=np.float32,
                        )
                    ordered = [
                        sparse_columns["id"],
                        sparse_columns["chunk_id"],
                        sparse_columns["doc_id"],
                        sparse_columns["content"],
                        sparse_columns["chunk_index"],
                        sparse_columns["token_count"],
                        sparse_columns["entropy"],
                        sparse_columns["redundancy"],
                        sparse_columns["domain_density"],
                        embedding_matrix,
                        sparse_columns["metadata_json"],
                        sparse_columns["timestamp"],
                    ]
                    await asyncio.to_thread(self.collections["sparse_index"].insert, ordered)
                    summary["indexed_sparse"] = len(sparse_columns["id"])
                except Exception as e:
                    logging.warning("Sparse insert failed; continuing without sparse index: %s", e)
                    summary["errors"].append({"insert_sparse_error": str(e)})
            
            if domain_data:
                await asyncio.to_thread(self.collections["domain_index"].insert, domain_data)
                summary["indexed_domain"] = len(domain_data)
            
            # Flush to ensure persistence
            for collection in self.collections.values():
                await asyncio.to_thread(collection.flush)
                
        except Exception as e:
            # Capture insert failure; caller can inspect this for diagnostics.
            summary["errors"].append({"insert_error": str(e)})
        
        return summary
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def search(  # pragma: no cover
        self,
        query_embedding: np.ndarray,
        collection_name: str,
        top_k: int = 20,
        filters: Optional[str] = None,
        search_params: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search in a specific collection with timeout and retry
        
        Args:
            query_embedding: Query vector
            collection_name: Name of collection to search
            top_k: Number of results to return
            filters: Optional filter expression
            search_params: Search parameters (ef, nprobe, etc.)
            
        Returns:
            List of search results with metadata
        """
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")
        
        collection = self.collections[collection_name]
        
        # Default search parameters (depend on collection type)
        is_sparse = (collection_name == "sparse_index")
        if search_params is None:
            if is_sparse:
                # Sparse inverted index typically uses Inner Product; no extra params needed
                search_params = {
                    "metric_type": "IP"
                }
            else:
                # HNSW dense defaults
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": 64}
                }
        
        # Prepare query vector format
        if is_sparse:
            # Accept dict payload or scipy sparse; normalize to CSR matrix (1, dim)
            if isinstance(query_embedding, dict):
                q_idx = np.array(query_embedding.get("indices", []), dtype=np.int64)
                q_vals = np.array(query_embedding.get("values", []), dtype=np.float32)
                indptr = np.array([0, len(q_vals)], dtype=np.int64)
                q_mat = csr_matrix((q_vals, q_idx, indptr), shape=(1, self.sparse_dim), dtype=np.float32)
                query_data = q_mat
            elif issparse(query_embedding):
                query_data = query_embedding
            else:
                raise ValueError("Sparse query embedding must be dict with indices/values or a scipy.sparse matrix")
        else:
            query_data = [query_embedding.tolist()]
        
        # Perform search with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    collection.search,
                    query_data,
                    "embedding",
                    search_params,
                    top_k,
                    expr=filters,
                    output_fields=[
                        "chunk_id",
                        "doc_id",
                        "content",
                        "chunk_index",
                        "entropy",
                        "redundancy",
                        "domain_density",
                        "metadata_json",
                        "timestamp",
                    ],
                ),
                timeout=5.0  # 5 second timeout
            )
        except asyncio.TimeoutError:
            logging.error(f"Milvus search timeout for collection {collection_name}")
            raise Exception(f"Search timeout for collection {collection_name}")
        except Exception as e:
            logging.error(f"Milvus search failed for {collection_name}: {e}")
            raise
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.entity.get("chunk_id"),
                    "content": hit.entity.get("content"),
                    "score": hit.score,
                    "metadata": {
                        "doc_id": hit.entity.get("doc_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "entropy": hit.entity.get("entropy"),
                        "redundancy": hit.entity.get("redundancy"),
                        "domain_density": hit.entity.get("domain_density"),
                        "timestamp": hit.entity.get("timestamp")
                    }
                })
        
        return formatted_results
    
    async def _generate_semantic_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate semantic embeddings in batch for improved throughput
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        cache = get_semantic_cache()
        embeddings = []
        
        # Check which texts are already in cache
        cache_misses = []
        cache_miss_indices = []
        
        for i, text in enumerate(texts):
            cached = await cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                cache_misses.append(text)
                cache_miss_indices.append(i)
        
        # Generate embeddings for cache misses in batch
        if cache_misses and self.embedding_generator:
            loop = asyncio.get_event_loop()
            
            # Batch encode in thread pool
            batch_embeddings = await loop.run_in_executor(
                self.embedding_executor,
                lambda: [self.embedding_generator.encode_semantic(text) for text in cache_misses]
            )
            
            # Store in cache and update results
            for i, emb in zip(cache_miss_indices, batch_embeddings):
                await cache.put(cache_misses[cache_miss_indices.index(i)], emb)
                embeddings[i] = emb
        elif cache_misses:
            # Fallback: random embeddings
            for i in cache_miss_indices:
                emb = np.random.randn(self.semantic_dim).astype(np.float32)
                embeddings[i] = emb
        
        return embeddings
    
    async def _generate_semantic_embedding(self, text: str) -> np.ndarray:
        """
        Generate dense semantic embedding with async execution and caching
        In production, integrate with actual embedding model (OpenAI, Cohere, etc.)
        """
        # Check cache first
        cache = get_semantic_cache()
        
        async def compute_embedding_inner() -> np.ndarray:
            if self.embedding_generator:
                # Respect async embedding_generator implementations in tests
                if asyncio.iscoroutinefunction(self.embedding_generator.encode_semantic):
                    return await self.embedding_generator.encode_semantic(text)
                # Offload CPU-intensive sync work to thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.embedding_executor,
                    self.embedding_generator.encode_semantic,
                    text,
                )

            # Placeholder: random embedding for demonstration
            return np.random.randn(self.semantic_dim).astype(np.float32)

        # Get from cache or compute
        embedding = await cache.get_or_compute(text, compute_embedding_inner)
        return embedding
    
    async def _generate_sparse_embedding(self, text: str):
        """
        Generate sparse embedding (BM25-style) with async execution.
        In production, use BM25 or SPLADE.
        """
        if self.embedding_generator:
            # Support async encode_sparse implementations (used in tests)
            if asyncio.iscoroutinefunction(self.embedding_generator.encode_sparse):
                return await self.embedding_generator.encode_sparse(text)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.embedding_executor,
                self.embedding_generator.encode_sparse,
                text,
            )
        
        # Placeholder: construct a valid SPARSE_FLOAT_VECTOR payload
        # Milvus expects a dict with 'indices' and 'values'
        nnz = min(100, self.sparse_dim)  # number of non-zeros
        rand_indices = np.random.choice(self.sparse_dim, size=nnz, replace=False)
        rand_values = np.abs(np.random.randn(nnz).astype(np.float32))  # positive TF-IDF-like values
        # Sort by index for determinism
        order = np.argsort(rand_indices)
        indices = rand_indices[order].tolist()
        values = rand_values[order].astype(float).tolist()
        return {"indices": indices, "values": values}
    
    async def _generate_domain_embedding(
        self,
        text: str,
        domain: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate domain-specific embedding with async execution.
        In production, use domain-adapted models.
        """
        if self.embedding_generator:
            # Support async encode_domain implementations (used in tests)
            if asyncio.iscoroutinefunction(self.embedding_generator.encode_domain):
                return await self.embedding_generator.encode_domain(text, domain or "")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.embedding_executor,
                lambda: self.embedding_generator.encode_domain(text, domain),
            )
        
        # Placeholder: random embedding
        return np.random.randn(self.domain_dim).astype(np.float32)
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        if collection_name not in self.collections:
            return {}
        
        collection = self.collections[collection_name]
        
        return {
            "name": collection_name,
            "num_entities": collection.num_entities,
            "schema": str(collection.schema),
            "indexes": [str(idx) for idx in collection.indexes]
        }
    
    async def delete_by_filter(self, collection_name: str, expr: str):
        """Delete entities matching filter expression"""
        if collection_name in self.collections:
            self.collections[collection_name].delete(expr)
    
    async def close(self):
        """Close connections and release resources"""
        for collection in self.collections.values():
            try:
                collection.release()
            except Exception:
                pass
        try:
            connections.disconnect("default")
        except Exception:
            pass
        
        # Shutdown thread pool executor
        if hasattr(self, 'embedding_executor'):
            self.embedding_executor.shutdown(wait=True)
        
        print("Disconnected from Milvus")
