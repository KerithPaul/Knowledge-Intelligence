"""One retrieved chunk, with the scores that put it there."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned to the caller, best first.

    `rank` is the final position after reranking, starting at 1.
    `index` is the chunk's position inside its source document.
    Dense and BM25 fields stay empty when that list did not contain the chunk.
    `rerank_score` is the cross-encoder probability. Higher is more relevant.
    """

    rank: int
    chunk_id: str
    source: str
    doc_title: str
    heading_path: tuple[str, ...]
    text: str
    token_count: int
    index: int
    oversized: bool
    orphan: bool
    dense_score: float | None
    dense_rank: int | None
    bm25_score: float | None
    bm25_rank: int | None
    rrf_score: float
    rerank_score: float
