"""Embed chunk records and upsert them into Qdrant.

The embedding client applies the passage prefix. This module passes chunk
text through unchanged, so a passage prefix is not stored in the payload
and is not counted in the sparse vector.

Chunks past the embedding window are still stored in full. The embedding
model drops tokens past that window when it encodes them. `truncated_chunks`
counts those rows. It does not cut the text.
"""

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from rag.core.config import Settings
from rag.core.errors import IndexingError
from rag.core.interfaces import EmbeddingClient
from rag.core.trace import RequestTrace
from rag.indexing.records import ChunkRecord, load_chunk_records, validate_records
from rag.indexing.sparse import build_sparse_vectors
from rag.indexing.store import (
    build_qdrant_client,
    count_points,
    ensure_collection,
    exact_self_hit,
    hnsw_self_hit,
    is_local_client,
    prepare_point,
    prune_points,
    upsert_points,
    wait_until_green,
)


@dataclass(frozen=True)
class IndexStats:
    """What one index run wrote, and whether exact search can find each point."""

    collection: str
    embedding_model: str
    points: int
    upserted: int
    pruned: int
    schema_recreated: bool
    dense_dimension: int
    distance: str
    vocabulary_size: int
    sparse_nnz_min: int | None
    sparse_nnz_median: float | None
    sparse_nnz_max: int | None
    oversized_chunks: int
    truncated_chunks: int
    self_hit_at_1: float | None
    self_hit_at_1_hnsw: float | None
    indexed_vectors: int
    search_mode: str


def index_records(
    records: Sequence[ChunkRecord],
    *,
    embedder: EmbeddingClient,
    client: QdrantClient,
    settings: Settings,
    trace: RequestTrace | None = None,
) -> IndexStats:
    """Embed `records` as passages and make the collection match them exactly."""

    validate_records(records)
    dimension = settings.embeddings.dimension
    if embedder.dimension != dimension:
        raise IndexingError(f"embedder dimension is {embedder.dimension}, settings say {dimension}")
    texts = [record.text for record in records]
    dense_rows = embedder.embed_passages(texts, trace=trace) if texts else []
    if len(dense_rows) != len(records):
        raise IndexingError(
            f"embedder returned {len(dense_rows)} vectors for {len(records)} chunks"
        )
    for row in dense_rows:
        if len(row) != dimension:
            raise IndexingError(f"embedder returned dimension {len(row)}, settings say {dimension}")
    vocabulary, sparse_rows = build_sparse_vectors(texts)
    points = [
        prepare_point(record, dense, sparse)
        for record, dense, sparse in zip(records, dense_rows, sparse_rows, strict=True)
    ]
    point_ids = [point.point_id for point in points]
    if len(set(point_ids)) != len(point_ids):
        raise IndexingError("two chunk ids produced the same Qdrant point id")

    started = time.perf_counter()
    qdrant = settings.qdrant
    recreated = ensure_collection(
        client,
        qdrant,
        dimension=dimension,
        embedding_model=embedder.model_name,
    )
    upsert_points(client, qdrant, points)
    pruned = prune_points(client, qdrant.collection, keep=set(point_ids))
    info = wait_until_green(client, qdrant)
    point_count = count_points(client, qdrant.collection)
    if point_count != len(points):
        raise IndexingError(
            f"collection {qdrant.collection} has {point_count} points "
            f"after indexing {len(points)} chunks"
        )
    self_hit = exact_self_hit(client, qdrant, points) if points else None
    hnsw_hit = hnsw_self_hit(client, qdrant, points) if points else None
    if trace is not None:
        trace.record(
            "indexing.upsert",
            (time.perf_counter() - started) * 1000,
            points=len(points),
            pruned=pruned,
            collection=qdrant.collection,
            dimension=dimension,
        )
    nnz = [row.nnz for row in sparse_rows]
    window = settings.chunking.chunk_size_tokens
    return IndexStats(
        collection=qdrant.collection,
        embedding_model=embedder.model_name,
        points=point_count,
        upserted=len(points),
        pruned=pruned,
        schema_recreated=recreated,
        dense_dimension=dimension,
        distance=qdrant.distance,
        vocabulary_size=len(vocabulary),
        sparse_nnz_min=min(nnz) if nnz else None,
        sparse_nnz_median=float(statistics.median(nnz)) if nnz else None,
        sparse_nnz_max=max(nnz) if nnz else None,
        oversized_chunks=sum(1 for record in records if record.oversized),
        truncated_chunks=sum(1 for record in records if record.token_count > window),
        self_hit_at_1=self_hit,
        self_hit_at_1_hnsw=hnsw_hit,
        # Qdrant counts each named vector, so dense plus sparse is two per point.
        indexed_vectors=int(info.indexed_vectors_count or 0),
        search_mode="local_exact" if is_local_client(client) else "server",
    )


def index_processed(
    processed_dir: Path,
    settings: Settings,
    *,
    embedder: EmbeddingClient | None = None,
    client: QdrantClient | None = None,
    trace: RequestTrace | None = None,
) -> IndexStats:
    """Index `processed_dir/chunks.jsonl` and write `index_stats.json` beside it."""

    records = load_chunk_records(processed_dir / "chunks.jsonl")
    vector_client = embedder if embedder is not None else _build_embedder(settings)
    qdrant_client = client if client is not None else build_qdrant_client(settings.qdrant)
    stats = index_records(
        records,
        embedder=vector_client,
        client=qdrant_client,
        settings=settings,
        trace=trace,
    )
    write_index_stats(stats, processed_dir / "index_stats.json")
    return stats


def index_stats_record(stats: IndexStats) -> dict[str, object]:
    """JSON object for index_stats.json."""

    return {
        "collection": stats.collection,
        "embedding_model": stats.embedding_model,
        "points": stats.points,
        "upserted": stats.upserted,
        "pruned": stats.pruned,
        "schema_recreated": stats.schema_recreated,
        "dense_dimension": stats.dense_dimension,
        "distance": stats.distance,
        "vocabulary_size": stats.vocabulary_size,
        "sparse_nnz_min": stats.sparse_nnz_min,
        "sparse_nnz_median": stats.sparse_nnz_median,
        "sparse_nnz_max": stats.sparse_nnz_max,
        "oversized_chunks": stats.oversized_chunks,
        "truncated_chunks": stats.truncated_chunks,
        "self_hit_at_1": stats.self_hit_at_1,
        "self_hit_at_1_hnsw": stats.self_hit_at_1_hnsw,
        "indexed_vectors": stats.indexed_vectors,
        "search_mode": stats.search_mode,
    }


def write_index_stats(stats: IndexStats, path: Path) -> None:
    """Write index_stats.json. Overwrites the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(index_stats_record(stats), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")


def _build_embedder(settings: Settings) -> EmbeddingClient:
    from rag.core.clients import build_embedding_client

    return build_embedding_client(settings)
