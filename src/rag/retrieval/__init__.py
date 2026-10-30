"""Phase 3 retrieval: dense search, BM25, RRF, and a cross-encoder rerank."""

from rag.retrieval.filters import MetadataFilter
from rag.retrieval.pipeline import Retriever, open_retriever
from rag.retrieval.types import RetrievedChunk

__all__ = [
    "MetadataFilter",
    "Retriever",
    "RetrievedChunk",
    "open_retriever",
]
