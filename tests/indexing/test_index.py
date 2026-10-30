"""Index writes, a second run, and exact self-hit against in-memory Qdrant."""

import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from rag.core.config import load_settings
from rag.core.errors import IndexingError
from rag.core.trace import RequestTrace
from rag.embeddings.bge import BGEEmbeddingClient
from rag.indexing.indexer import index_processed, index_records
from rag.indexing.records import ChunkRecord, load_chunk_records
from rag.indexing.store import point_id

_YAML_ENV = ""


def _settings(tmp_path: Path, **overrides):
    env_file = tmp_path / ".env"
    env_file.write_text(_YAML_ENV, encoding="utf-8")
    qdrant = {
        "url": "http://localhost:6333",
        "collection": "knowledge",
        "upsert_batch_size": 1,
        "timeout_seconds": 5,
    }
    qdrant.update(overrides.pop("qdrant", {}))
    embeddings = {"model": "hash-test", "dimension": 4}
    embeddings.update(overrides.pop("embeddings", {}))
    return load_settings(
        env_file=env_file,
        qdrant=qdrant,
        embeddings=embeddings,
        **overrides,
    )


def _vector(text: str, dimension: int = 4) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[index] / 255 for index in range(dimension)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


class HashEmbedder:
    """Deterministic stand-in. The same text always returns the same vector."""

    def __init__(self, dimension: int = 4, model_name: str = "hash-test") -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.texts: list[str] = []

    def embed_query(self, text: str, *, trace: RequestTrace | None = None) -> list[float]:
        raise AssertionError("indexing embeds passages")

    def embed_passages(self, texts, *, trace: RequestTrace | None = None) -> list[list[float]]:
        self.texts.extend(texts)
        if trace is not None:
            trace.record("embedding.encode", 1.0, kind="passage", count=len(texts))
        return [_vector(text, self.dimension) for text in texts]


class _Encoder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        self.texts = list(texts)
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]


def _record(chunk_id: str = "doc.md#0000", text: str = "alpha beta", **kwargs) -> ChunkRecord:
    fields = dict(
        chunk_id=chunk_id,
        source="doc.md",
        doc_title="Doc",
        heading_path=("Doc", "Section"),
        text=text,
        token_count=10,
        index=0,
        oversized=False,
        orphan=False,
    )
    fields.update(kwargs)
    return ChunkRecord(**fields)


def _index(tmp_path: Path, records, **kwargs):
    client = kwargs.pop("client", None) or QdrantClient(":memory:")
    embedder = kwargs.pop("embedder", None) or HashEmbedder()
    settings = kwargs.pop("settings", None) or _settings(tmp_path)
    trace = kwargs.pop("trace", None)
    stats = index_records(
        records,
        embedder=embedder,
        client=client,
        settings=settings,
        trace=trace,
    )
    return client, embedder, stats


def _payload(client: QdrantClient, chunk_id: str) -> dict:
    records = client.retrieve(
        collection_name="knowledge",
        ids=[point_id(chunk_id)],
        with_payload=True,
        with_vectors=True,
    )
    assert len(records) == 1
    return records[0]


def test_point_ids_are_stable_uuids() -> None:
    first = point_id("hermes/architecture.md#0000")
    assert first == point_id("hermes/architecture.md#0000")
    assert first != point_id("hermes/architecture.md#0001")
    assert len(first) == 36


def test_index_writes_dense_and_sparse_vectors(tmp_path: Path) -> None:
    embedder = HashEmbedder()
    client, _embedder, stats = _index(
        tmp_path,
        [
            _record("a.md#0000", "Cat cat dog"),
            _record("a.md#0001", "bird", index=1),
        ],
        embedder=embedder,
    )
    assert embedder.texts == ["Cat cat dog", "bird"]
    assert stats.points == 2
    assert stats.upserted == 2
    assert stats.pruned == 0
    assert stats.schema_recreated is False
    assert stats.vocabulary_size == 3
    assert stats.sparse_nnz_min == 1
    assert stats.sparse_nnz_max == 2
    assert stats.self_hit_at_1 == 1.0
    assert stats.self_hit_at_1_hnsw is None
    assert stats.search_mode == "local_exact"
    assert stats.truncated_chunks == 0

    stored = _payload(client, "a.md#0000")
    assert stored.payload["text"] == "Cat cat dog"
    assert stored.payload["heading_path"] == ["Doc", "Section"]
    assert stored.payload["doc_title"] == "Doc"
    dense = stored.vector["dense"]
    assert dense == pytest.approx(_vector("Cat cat dog"))
    sparse = stored.vector["sparse"]
    # Joint vocabulary is bird, cat, dog. This chunk has cat twice and dog once.
    assert list(sparse.indices) == [1, 2]
    assert list(sparse.values) == pytest.approx([2.0, 1.0])


def test_second_run_updates_and_prunes(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _index(
        tmp_path,
        [_record("a.md#0000", "alpha"), _record("a.md#0001", "beta", index=1)],
        client=client,
    )
    _client, _embedder, stats = _index(
        tmp_path,
        [_record("a.md#0000", "alpha alpha beta")],
        client=client,
    )
    assert stats.points == 1
    assert stats.pruned == 1
    assert stats.schema_recreated is False
    stored = _payload(client, "a.md#0000")
    assert stored.payload["text"] == "alpha alpha beta"
    assert list(stored.vector["sparse"].indices) == [0, 1]
    assert list(stored.vector["sparse"].values) == pytest.approx([2.0, 1.0])
    removed = client.retrieve(
        collection_name="knowledge",
        ids=[point_id("a.md#0001")],
        with_payload=False,
    )
    assert removed == []


def test_empty_corpus_drops_existing_points(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    _index(tmp_path, [_record()], client=client)
    _client, _embedder, stats = _index(tmp_path, [], client=client)
    assert stats.points == 0
    assert stats.pruned == 1
    assert stats.self_hit_at_1 is None
    assert stats.vocabulary_size == 0
    assert stats.sparse_nnz_min is None


def test_identical_texts_still_count_as_a_self_hit(tmp_path: Path) -> None:
    _client, _embedder, stats = _index(
        tmp_path,
        [
            _record("a.md#0000", "same words"),
            _record("a.md#0001", "same words", index=1),
        ],
    )
    assert stats.self_hit_at_1 == 1.0
    assert stats.points == 2


def test_punctuation_only_text_gets_an_empty_sparse_vector(tmp_path: Path) -> None:
    client, _embedder, stats = _index(tmp_path, [_record(text="???")])
    assert stats.sparse_nnz_min == 0
    stored = _payload(client, "doc.md#0000")
    assert list(stored.vector["sparse"].indices) == []


def test_oversized_chunk_is_counted_and_stored_whole(tmp_path: Path) -> None:
    text = "table " * 50
    client, embedder, stats = _index(
        tmp_path,
        [_record(text=text, token_count=900, oversized=True)],
    )
    assert stats.oversized_chunks == 1
    assert stats.truncated_chunks == 1
    assert embedder.texts == [text]
    assert _payload(client, "doc.md#0000").payload["text"] == text


def test_passage_prefix_is_applied_by_the_embedding_client(tmp_path: Path) -> None:
    encoder = _Encoder()
    embedder = BGEEmbeddingClient(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=4,
        query_prefix="query: ",
        passage_prefix="passage: ",
        batch_size=8,
        normalize=True,
        device="cpu",
        encoder=encoder,
    )
    settings = _settings(tmp_path, embeddings={"model": embedder.model_name, "dimension": 4})
    client, _embedder, _stats = _index(
        tmp_path,
        [_record(text="alpha beta")],
        embedder=embedder,
        settings=settings,
    )
    assert encoder.texts == ["passage: alpha beta"]
    assert _payload(client, "doc.md#0000").payload["text"] == "alpha beta"


def test_schema_mismatch_recreates_the_collection(tmp_path: Path) -> None:
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="knowledge",
        vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE),
    )
    _client, _embedder, stats = _index(tmp_path, [_record()], client=client)
    assert stats.schema_recreated is True
    info = client.get_collection("knowledge")
    assert info.config.params.vectors["dense"].size == 4
    assert info.config.metadata["embedding_model"] == "hash-test"
    assert info.config.metadata["sparse"] == "term_frequency"
    _client, _embedder, again = _index(tmp_path, [_record()], client=client)
    assert again.schema_recreated is False


def test_duplicate_chunk_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate chunk_id"):
        _index(tmp_path, [_record("a.md#0000"), _record("a.md#0000", index=1)])


def test_empty_text_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty text"):
        _index(tmp_path, [_record(text="   ")])


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(IndexingError, match="dimension"):
        _index(tmp_path, [_record()], embedder=HashEmbedder(dimension=8))


def test_trace_records_the_upsert(tmp_path: Path) -> None:
    trace = RequestTrace()
    _index(tmp_path, [_record()], trace=trace)
    names = [span.name for span in trace.spans]
    assert names == ["embedding.encode", "indexing.upsert"]
    assert trace.spans[1].attributes["points"] == 1
    assert trace.spans[1].attributes["dimension"] == 4


def test_index_processed_writes_stats(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    row = {
        "chunk_id": "doc.md#0000",
        "source": "doc.md",
        "doc_title": "",
        "heading_path": ["Title"],
        "text": "alpha beta",
        "token_count": 4,
        "index": 0,
        "oversized": False,
        "orphan": False,
    }
    (processed / "chunks.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    stats = index_processed(
        processed,
        _settings(tmp_path),
        embedder=HashEmbedder(),
        client=QdrantClient(":memory:"),
    )
    written = json.loads((processed / "index_stats.json").read_text(encoding="utf-8"))
    assert stats.points == 1
    assert written["points"] == 1
    assert written["search_mode"] == "local_exact"
    assert written["embedding_model"] == "hash-test"


def test_load_chunk_records_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    row = {
        "chunk_id": "doc.md#0000",
        "source": "doc.md",
        "doc_title": "Doc",
        "heading_path": [],
        "text": "alpha",
        "token_count": 1,
        "index": 0,
        "oversized": False,
        "orphan": False,
    }
    path.write_text(json.dumps(row) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunk_records(path)


def test_load_allows_an_empty_doc_title(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    row = {
        "chunk_id": "hermes/adding-platform-adapters.md#0000",
        "source": "hermes/adding-platform-adapters.md",
        "doc_title": "",
        "heading_path": [],
        "text": "adapter text",
        "token_count": 3,
        "index": 0,
        "oversized": False,
        "orphan": False,
    }
    path.write_text(json.dumps(row) + "\n\n", encoding="utf-8")
    records = load_chunk_records(path)
    assert records[0].doc_title == ""
    assert records[0].heading_path == ()


def test_missing_chunk_file_names_ingestion(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ingestion"):
        load_chunk_records(tmp_path / "chunks.jsonl")


def test_vector_names_must_differ(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="different names"):
        _settings(tmp_path, qdrant={"dense_vector": "same", "sparse_vector": "same"})
