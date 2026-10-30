"""Qdrant write path for one dense vector and one sparse vector per chunk.

Point ids are a uuid5 of chunk_id. A second run upserts those same ids,
then deletes ids that are no longer in the corpus. The collection is
recreated only when the vector schema or the embedding model no longer
matches. Local Qdrant does not store the collection-level HNSW and
optimizer thresholds, so those fields are checked on a server only.
"""

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.core.config import QdrantSettings
from rag.core.errors import IndexingError
from rag.indexing.records import ChunkRecord
from rag.indexing.sparse import SPARSE_KIND, SparseVector

# Stable across processes. uuid5(namespace, chunk_id) is the Qdrant point id.
_POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "knowledge-intelligence/chunk")

_DISTANCES = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}

_QDRANT_ERRORS = (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError, OSError)


@dataclass(frozen=True)
class PreparedPoint:
    """One chunk after embedding, ready to upsert."""

    point_id: str
    chunk_id: str
    dense: list[float]
    sparse: SparseVector
    payload: dict[str, object]


def point_id(chunk_id: str) -> str:
    """Return the stable Qdrant id for a chunk id."""

    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


def build_qdrant_client(qdrant: QdrantSettings) -> QdrantClient:
    """Connect to the Qdrant URL in settings. The call does not create a collection."""

    return QdrantClient(url=qdrant.url, timeout=qdrant.timeout_seconds)


def is_local_client(client: QdrantClient) -> bool:
    """True for `:memory:` and for a path client. Those clients brute-force search."""

    options = client.init_options
    return bool(options.get("location") or options.get("path"))


def prepare_point(
    record: ChunkRecord,
    dense: Sequence[float],
    sparse: SparseVector,
) -> PreparedPoint:
    """Pack one chunk. The payload keeps the full text, including oversized chunks."""

    return PreparedPoint(
        point_id=point_id(record.chunk_id),
        chunk_id=record.chunk_id,
        dense=[float(value) for value in dense],
        sparse=sparse,
        payload={
            "chunk_id": record.chunk_id,
            "source": record.source,
            "doc_title": record.doc_title,
            "heading_path": list(record.heading_path),
            "text": record.text,
            "token_count": record.token_count,
            "index": record.index,
            "oversized": record.oversized,
            "orphan": record.orphan,
        },
    )


def ensure_collection(
    client: QdrantClient,
    qdrant: QdrantSettings,
    *,
    dimension: int,
    embedding_model: str,
) -> bool:
    """Create the collection, or replace it when the schema does not match.

    Returns True only when an existing collection was deleted and created again.
    """

    name = qdrant.collection
    if _run(client, "check the collection", lambda: client.collection_exists(name)):
        info = _run(client, "read the collection", lambda: client.get_collection(name))
        if _schema_matches(
            info,
            qdrant,
            dimension=dimension,
            embedding_model=embedding_model,
            server=not is_local_client(client),
        ):
            _ensure_payload_indexes(client, name)
            return False
        _run(client, "delete the collection", lambda: client.delete_collection(name))
        _create_collection(client, qdrant, dimension=dimension, embedding_model=embedding_model)
        return True
    _create_collection(client, qdrant, dimension=dimension, embedding_model=embedding_model)
    return False


def upsert_points(
    client: QdrantClient,
    qdrant: QdrantSettings,
    points: Sequence[PreparedPoint],
) -> None:
    """Upsert in batches. `wait` makes each batch durable before the next starts."""

    if not points:
        return
    name = qdrant.collection
    dense_name = qdrant.dense_vector
    sparse_name = qdrant.sparse_vector
    batch_size = qdrant.upsert_batch_size
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        structs = [
            models.PointStruct(
                id=point.point_id,
                vector={
                    dense_name: point.dense,
                    sparse_name: models.SparseVector(
                        indices=list(point.sparse.indices),
                        values=list(point.sparse.values),
                    ),
                },
                payload=point.payload,
            )
            for point in batch
        ]
        _run(
            client,
            "upsert points",
            lambda structs=structs: client.upsert(
                collection_name=name,
                points=structs,
                wait=True,
            ),
        )


def prune_points(
    client: QdrantClient,
    collection: str,
    *,
    keep: set[str],
) -> int:
    """Delete points whose ids are not in `keep`. Returns how many were deleted."""

    stale = sorted(_list_point_ids(client, collection) - keep)
    if not stale:
        return 0
    for start in range(0, len(stale), 256):
        batch = stale[start : start + 256]
        _run(
            client,
            "delete stale points",
            lambda batch=batch: client.delete(
                collection_name=collection,
                points_selector=batch,
                wait=True,
            ),
        )
    return len(stale)


def count_points(client: QdrantClient, collection: str) -> int:
    """Exact point count."""

    result = _run(client, "count points", lambda: client.count(collection, exact=True))
    return int(result.count)


def wait_until_green(client: QdrantClient, qdrant: QdrantSettings) -> models.CollectionInfo:
    """Wait until Qdrant finishes optimizing. Local mode is already green."""

    deadline = time.perf_counter() + qdrant.timeout_seconds
    while True:
        info = _run(
            client,
            "read collection status",
            lambda: client.get_collection(qdrant.collection),
        )
        status = _status_value(info.status)
        if status == "green":
            return info
        if status == "red":
            raise IndexingError(f"Qdrant collection {qdrant.collection} status is red")
        if time.perf_counter() >= deadline:
            raise IndexingError(
                f"Qdrant collection {qdrant.collection} was still {status} "
                f"after {qdrant.timeout_seconds} seconds"
            )
        time.sleep(0.25)


def exact_self_hit(
    client: QdrantClient,
    qdrant: QdrantSettings,
    points: Sequence[PreparedPoint],
) -> float:
    """Fraction of points whose own dense vector is an exact nearest neighbor.

    A tie with an identical vector still counts. A miss raises, because the
    point we just wrote must be findable without the ANN graph.
    """

    if not points:
        raise ValueError("no points to check")
    misses: list[str] = []
    for point in points:
        if not _is_nearest(client, qdrant, point, exact=True):
            misses.append(point.chunk_id)
    if misses:
        shown = ", ".join(misses[:5])
        raise IndexingError(
            f"exact self-hit failed for {len(misses)} of {len(points)} chunks: {shown}"
        )
    return 1.0


def hnsw_self_hit(
    client: QdrantClient,
    qdrant: QdrantSettings,
    points: Sequence[PreparedPoint],
) -> float | None:
    """Same check through the HNSW graph. Local mode has no graph, so it returns None."""

    if not points or is_local_client(client):
        return None
    hits = sum(1 for point in points if _is_nearest(client, qdrant, point, exact=False))
    return hits / len(points)


def _create_collection(
    client: QdrantClient,
    qdrant: QdrantSettings,
    *,
    dimension: int,
    embedding_model: str,
) -> None:
    hnsw = _hnsw(qdrant)
    _run(
        client,
        "create the collection",
        lambda: client.create_collection(
            collection_name=qdrant.collection,
            vectors_config={
                qdrant.dense_vector: models.VectorParams(
                    size=dimension,
                    distance=_DISTANCES[qdrant.distance],
                    hnsw_config=hnsw,
                )
            },
            sparse_vectors_config={
                qdrant.sparse_vector: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                )
            },
            hnsw_config=hnsw,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=qdrant.indexing_threshold_kb,
            ),
            metadata={
                "embedding_model": embedding_model,
                "sparse": SPARSE_KIND,
            },
        ),
    )
    _ensure_payload_indexes(client, qdrant.collection)


def _hnsw(qdrant: QdrantSettings) -> models.HnswConfigDiff:
    return models.HnswConfigDiff(
        m=qdrant.hnsw_m,
        ef_construct=qdrant.hnsw_ef_construct,
        full_scan_threshold=qdrant.full_scan_threshold_kb,
    )


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    # Local mode warns that payload indexes do nothing there. The server uses
    # them when Phase 3 filters on source or chunk id.
    if is_local_client(client):
        return
    for field_name in ("chunk_id", "source"):
        _run(
            client,
            f"index payload field {field_name}",
            lambda field_name=field_name: client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            ),
        )


def _schema_matches(
    info: models.CollectionInfo,
    qdrant: QdrantSettings,
    *,
    dimension: int,
    embedding_model: str,
    server: bool,
) -> bool:
    vectors = info.config.params.vectors
    if not isinstance(vectors, dict):
        return False
    dense = vectors.get(qdrant.dense_vector)
    if not isinstance(dense, models.VectorParams):
        return False
    if dense.size != dimension or dense.distance != _DISTANCES[qdrant.distance]:
        return False
    hnsw = dense.hnsw_config
    if (
        hnsw is None
        or hnsw.m != qdrant.hnsw_m
        or hnsw.ef_construct != qdrant.hnsw_ef_construct
        or hnsw.full_scan_threshold != qdrant.full_scan_threshold_kb
    ):
        return False
    sparse_config = info.config.params.sparse_vectors or {}
    sparse = sparse_config.get(qdrant.sparse_vector)
    if sparse is None or sparse.modifier not in (None, models.Modifier.NONE):
        return False
    metadata = info.config.metadata or {}
    if metadata.get("embedding_model") != embedding_model or metadata.get("sparse") != SPARSE_KIND:
        return False
    if not server:
        return True
    collection_hnsw = info.config.hnsw_config
    if (
        collection_hnsw.m != qdrant.hnsw_m
        or collection_hnsw.ef_construct != qdrant.hnsw_ef_construct
        or collection_hnsw.full_scan_threshold != qdrant.full_scan_threshold_kb
    ):
        return False
    optimizer = info.config.optimizer_config
    return optimizer.indexing_threshold == qdrant.indexing_threshold_kb


def _list_point_ids(client: QdrantClient, collection: str) -> set[str]:
    found: set[str] = set()
    offset = None
    seen_offsets: set[str] = set()
    while True:
        points, offset = _run(
            client,
            "scroll points",
            lambda offset=offset: client.scroll(
                collection_name=collection,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            ),
        )
        found.update(str(point.id) for point in points)
        if not points or offset is None:
            break
        marker = str(offset)
        if marker in seen_offsets:
            break
        seen_offsets.add(marker)
    return found


def _is_nearest(
    client: QdrantClient,
    qdrant: QdrantSettings,
    point: PreparedPoint,
    *,
    exact: bool,
) -> bool:
    top = _query(client, qdrant, point.dense, exact=exact, only_id=None)
    if not top:
        return False
    if str(top[0].id) == point.point_id:
        return True
    mine = _query(client, qdrant, point.dense, exact=exact, only_id=point.point_id)
    if not mine:
        return False
    return abs(mine[0].score - top[0].score) <= 1e-4


def _query(
    client: QdrantClient,
    qdrant: QdrantSettings,
    vector: Sequence[float],
    *,
    exact: bool,
    only_id: str | None,
):
    query_filter = None
    if only_id is not None:
        query_filter = models.Filter(must=[models.HasIdCondition(has_id=[only_id])])
    # Local mode always brute-forces and warns if search_params is set.
    search_params = None
    if not is_local_client(client):
        search_params = models.SearchParams(
            exact=exact,
            hnsw_ef=None if exact else qdrant.hnsw_ef_search,
        )
    response = _run(
        client,
        "query points",
        lambda: client.query_points(
            collection_name=qdrant.collection,
            query=list(vector),
            using=qdrant.dense_vector,
            limit=1,
            query_filter=query_filter,
            search_params=search_params,
            with_payload=False,
        ),
    )
    return response.points


def _status_value(status: object) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def _endpoint(client: QdrantClient) -> str:
    options = client.init_options
    return str(options.get("url") or options.get("location") or options.get("path") or "qdrant")


def _run(client: QdrantClient, action: str, func):
    try:
        return func()
    except _QDRANT_ERRORS as exc:
        raise IndexingError(f"Qdrant at {_endpoint(client)} could not {action}: {exc}") from exc
