"""Dense search, BM25, RRF, then a cross-encoder rerank.

Passage text comes from the chunk file indexing reads. The Qdrant payload
is used to read ``chunk_id`` and the dense score. A point whose chunk id
is missing from that file is a stale index.

`top_n` is the candidate cut for each list and for the fused list. The
reranker scores those candidates. `top_k` is how many it returns.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from rag.core.config import PROJECT_ROOT, Settings
from rag.core.errors import RetrievalError
from rag.core.interfaces import EmbeddingClient, Reranker
from rag.core.trace import RequestTrace
from rag.indexing.records import ChunkRecord, load_chunk_records, validate_records
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.dense import dense_search, require_collection
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.types import RetrievedChunk


@dataclass(frozen=True)
class _Fused:
    record: ChunkRecord
    rrf_score: float
    dense_score: float | None
    dense_rank: int | None
    bm25_score: float | None
    bm25_rank: int | None


class Retriever:
    """Search one indexed corpus. Build it once and call `retrieve` per query."""

    def __init__(
        self,
        records: Sequence[ChunkRecord],
        *,
        embedder: EmbeddingClient,
        client: QdrantClient,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        validate_records(records)
        if embedder.dimension != settings.embeddings.dimension:
            raise RetrievalError(
                f"embedder dimension is {embedder.dimension}, "
                f"settings say {settings.embeddings.dimension}"
            )
        self._records = list(records)
        self._by_id = {record.chunk_id: record for record in self._records}
        retrieval = settings.retrieval
        self._bm25 = BM25Index.build(
            self._records,
            k1=retrieval.bm25_k1,
            b=retrieval.bm25_b,
        )
        self._embedder = embedder
        self._client = client
        self._reranker = reranker
        self._settings = settings

    def retrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        top_n: int | None = None,
        top_k: int | None = None,
        trace: RequestTrace | None = None,
    ) -> list[RetrievedChunk]:
        """Return reranked chunks, best first. An unknown query term still searches densely."""

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query text is empty")
        pool = self._settings.retrieval.top_n if top_n is None else top_n
        keep = self._settings.retrieval.top_k if top_k is None else top_k
        if pool <= 0 or keep <= 0:
            raise ValueError("top_n and top_k must be positive")
        if keep > pool:
            raise ValueError("top_k must be less than or equal to top_n")
        allowed = self._allowed(metadata_filter)

        started = time.perf_counter()
        vector = self._embedder.embed_query(cleaned, trace=trace)
        dense_hits = dense_search(
            self._client,
            self._settings.qdrant,
            vector,
            limit=pool,
            metadata_filter=metadata_filter,
        )
        for hit in dense_hits:
            if hit.chunk_id not in self._by_id:
                raise RetrievalError(
                    f"Qdrant returned chunk {hit.chunk_id}, which is not in the chunk file. "
                    "Run python -m rag.indexing before searching."
                )
        if trace is not None:
            trace.record(
                "retrieval.dense",
                (time.perf_counter() - started) * 1000,
                count=len(dense_hits),
                collection=self._settings.qdrant.collection,
            )

        started = time.perf_counter()
        bm25_hits = self._bm25.search(cleaned, limit=pool, allowed=allowed)
        if trace is not None:
            trace.record(
                "retrieval.bm25",
                (time.perf_counter() - started) * 1000,
                count=len(bm25_hits),
            )

        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            [
                [hit.chunk_id for hit in dense_hits],
                [hit.chunk_id for hit in bm25_hits],
            ],
            k=self._settings.retrieval.rrf_k,
            limit=pool,
        )
        if trace is not None:
            trace.record(
                "retrieval.rrf",
                (time.perf_counter() - started) * 1000,
                count=len(fused),
                k=self._settings.retrieval.rrf_k,
            )
        if not fused:
            if trace is not None:
                trace.record("retrieval.rerank", 0.0, count=0)
            return []

        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
        candidates = [
            _fuse_one(chunk_id, score, dense_by_id, bm25_by_id, self._by_id)
            for chunk_id, score in fused
        ]
        rerank_scores = self._reranker.score(
            cleaned,
            [candidate.record.text for candidate in candidates],
            trace=trace,
        )
        if len(rerank_scores) != len(candidates):
            raise RetrievalError(
                f"reranker returned {len(rerank_scores)} scores for {len(candidates)} candidates"
            )
        order = sorted(
            zip(candidates, rerank_scores, strict=True),
            key=lambda item: (-item[1], -item[0].rrf_score, item[0].record.chunk_id),
        )
        chosen = order[:keep]
        return [
            _to_result(rank, candidate, score)
            for rank, (candidate, score) in enumerate(chosen, start=1)
        ]

    def _allowed(self, metadata_filter: MetadataFilter | None) -> set[str] | None:
        if metadata_filter is None or metadata_filter.is_empty():
            return None
        return {record.chunk_id for record in self._records if metadata_filter.matches(record)}


def open_retriever(
    settings: Settings,
    *,
    processed_dir: Path | None = None,
    records: Sequence[ChunkRecord] | None = None,
    embedder: EmbeddingClient | None = None,
    client: QdrantClient | None = None,
    reranker: Reranker | None = None,
) -> Retriever:
    """Load the chunk file, check it matches Qdrant, and build a retriever.

    The weights for the embedder and the reranker load on the first query.
    """

    from rag.core.clients import build_embedding_client, build_reranker
    from rag.indexing.store import build_qdrant_client

    directory = processed_dir if processed_dir is not None else PROJECT_ROOT / "data" / "processed"
    loaded = (
        list(records) if records is not None else load_chunk_records(directory / "chunks.jsonl")
    )
    vector_client = embedder if embedder is not None else build_embedding_client(settings)
    qdrant = client if client is not None else build_qdrant_client(settings.qdrant)
    ranker = reranker if reranker is not None else build_reranker(settings)
    require_collection(qdrant, settings.qdrant, len(loaded))
    return Retriever(
        loaded,
        embedder=vector_client,
        client=qdrant,
        reranker=ranker,
        settings=settings,
    )


def _fuse_one(chunk_id, rrf_score, dense_by_id, bm25_by_id, by_id) -> _Fused:
    record = by_id.get(chunk_id)
    if record is None:
        raise RetrievalError(
            f"fused chunk {chunk_id} is not in the chunk file. "
            "Run python -m rag.indexing before searching."
        )
    dense = dense_by_id.get(chunk_id)
    bm25 = bm25_by_id.get(chunk_id)
    return _Fused(
        record=record,
        rrf_score=rrf_score,
        dense_score=None if dense is None else dense.score,
        dense_rank=None if dense is None else dense.rank,
        bm25_score=None if bm25 is None else bm25.score,
        bm25_rank=None if bm25 is None else bm25.rank,
    )


def _to_result(rank: int, candidate: _Fused, rerank_score: float) -> RetrievedChunk:
    record = candidate.record
    return RetrievedChunk(
        rank=rank,
        chunk_id=record.chunk_id,
        source=record.source,
        doc_title=record.doc_title,
        heading_path=record.heading_path,
        text=record.text,
        token_count=record.token_count,
        index=record.index,
        oversized=record.oversized,
        orphan=record.orphan,
        dense_score=candidate.dense_score,
        dense_rank=candidate.dense_rank,
        bm25_score=candidate.bm25_score,
        bm25_rank=candidate.bm25_rank,
        rrf_score=candidate.rrf_score,
        rerank_score=rerank_score,
    )
