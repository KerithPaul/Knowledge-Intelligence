"""Dense search on the Qdrant collection Phase 2 wrote.

The query vector comes from `EmbeddingClient.embed_query`, which applies
the BGE query prefix. This module sends that vector to the named dense
channel. On a server it sets ``hnsw_ef`` from settings. Local Qdrant
brute-forces and warns if search params are set, so those are omitted.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.core.config import QdrantSettings
from rag.core.errors import IndexingError, RetrievalError
from rag.indexing.store import count_points, is_local_client
from rag.retrieval.filters import MetadataFilter, to_qdrant_filter

_QDRANT_ERRORS = (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError, OSError)


@dataclass(frozen=True)
class DenseHit:
    """One dense hit. `rank` starts at 1, in the order Qdrant returned."""

    chunk_id: str
    score: float
    rank: int


def dense_search(
    client: QdrantClient,
    qdrant: QdrantSettings,
    vector: Sequence[float],
    *,
    limit: int,
    metadata_filter: MetadataFilter | None = None,
) -> list[DenseHit]:
    """Return up to `limit` nearest chunks on the dense channel."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if not vector:
        raise ValueError("query vector is empty")
    search_params = None
    if not is_local_client(client):
        search_params = models.SearchParams(hnsw_ef=qdrant.hnsw_ef_search, exact=False)
    try:
        response = client.query_points(
            collection_name=qdrant.collection,
            query=[float(value) for value in vector],
            using=qdrant.dense_vector,
            query_filter=to_qdrant_filter(metadata_filter),
            search_params=search_params,
            limit=limit,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
    except _QDRANT_ERRORS as exc:
        raise RetrievalError(
            f"Qdrant at {_endpoint(client)} could not run dense search: {exc}"
        ) from exc
    hits: list[DenseHit] = []
    seen: set[str] = set()
    for point in response.points:
        chunk_id = _chunk_id(point)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        hits.append(DenseHit(chunk_id=chunk_id, score=float(point.score), rank=len(hits) + 1))
    return hits


def require_collection(client: QdrantClient, qdrant: QdrantSettings, expected_points: int) -> None:
    """Raise when the collection is missing or its point count differs."""

    name = qdrant.collection
    try:
        exists = client.collection_exists(name)
    except _QDRANT_ERRORS as exc:
        raise RetrievalError(f"Qdrant at {qdrant.url} is not reachable: {exc}") from exc
    if not exists:
        raise RetrievalError(
            f"Qdrant collection {name} does not exist. Run python -m rag.indexing first."
        )
    try:
        count = count_points(client, name)
    except IndexingError as exc:
        raise RetrievalError(str(exc)) from exc
    if count != expected_points:
        raise RetrievalError(
            f"Qdrant collection {name} has {count} points and the chunk file has "
            f"{expected_points}. Run python -m rag.indexing before searching."
        )


def _chunk_id(point: models.ScoredPoint) -> str:
    payload = point.payload or {}
    chunk_id = payload.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise RetrievalError(f"Qdrant point {point.id} payload has no chunk_id")
    return chunk_id


def _endpoint(client: QdrantClient) -> str:
    options = client.init_options
    return str(options.get("url") or options.get("location") or options.get("path") or "qdrant")
