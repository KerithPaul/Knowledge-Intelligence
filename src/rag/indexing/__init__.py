"""Phase 2 indexing: passage embeddings into a Qdrant dense and sparse collection."""

from rag.indexing.indexer import IndexStats, index_processed, index_records
from rag.indexing.records import ChunkRecord, load_chunk_records

__all__ = [
    "ChunkRecord",
    "IndexStats",
    "index_processed",
    "index_records",
    "load_chunk_records",
]
