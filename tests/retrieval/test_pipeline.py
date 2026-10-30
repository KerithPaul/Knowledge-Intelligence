"""The retriever fuses dense and BM25, then reranks, against in-memory Qdrant."""

import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from rag.core.config import load_settings
from rag.core.errors import RetrievalError
from rag.core.trace import RequestTrace
from rag.indexing.indexer import index_records
from rag.indexing.records import ChunkRecord
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.pipeline import Retriever, open_retriever
from rag.retrieval.types import RetrievedChunk


def _settings(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    return load_settings(
        env_file=env_file,
        embeddings={"model": "scripted", "dimension": 4},
        qdrant={
            "url": "http://localhost:6333",
            "collection": "knowledge",
            "upsert_batch_size": 8,
            "timeout_seconds": 5,
        },
    )


def _record(
    chunk_id: str,
    text: str,
    *,
    source: str = "doc.md",
    title: str = "Doc",
    heading: tuple[str, ...] = ("Doc",),
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source=source,
        doc_title=title,
        heading_path=heading,
        text=text,
        token_count=10,
        index=0,
        oversized=False,
        orphan=False,
    )


class ScriptedEmbedder:
    """Query vectors are chosen by the test. Passage vectors are indexed as given."""

    def __init__(self, query_vector: list[float], passage_vectors: list[list[float]]) -> None:
        self.dimension = len(query_vector)
        self.model_name = "scripted"
        self.query_vector = query_vector
        self.passage_vectors = passage_vectors
        self.queries: list[str] = []

    def embed_query(self, text: str, *, trace=None) -> list[float]:
        self.queries.append(text)
        return list(self.query_vector)

    def embed_passages(self, texts, *, trace=None) -> list[list[float]]:
        assert len(texts) == len(self.passage_vectors)
        return [list(vector) for vector in self.passage_vectors]


class ScoreReranker:
    """Scores by an exact passage map. Unlisted passages score 0."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.model_name = "fake-reranker"
        self.scores = scores
        self.queries: list[str] = []
        self.seen: list[str] = []

    def score(self, query: str, passages, *, trace=None) -> list[float]:
        self.queries.append(query)
        self.seen = list(passages)
        if trace is not None:
            trace.record("retrieval.rerank", 1.0, model=self.model_name, count=len(passages))
        return [self.scores.get(passage, 0.0) for passage in passages]


def _build(
    tmp_path: Path,
    records: list[ChunkRecord],
    vectors: list[list[float]],
    query_vector: list[float],
):
    settings = _settings(tmp_path)
    embedder = ScriptedEmbedder(query_vector, vectors)
    client = QdrantClient(":memory:")
    index_records(records, embedder=embedder, client=client, settings=settings)
    reranker = ScoreReranker({})
    retriever = Retriever(
        records,
        embedder=embedder,
        client=client,
        reranker=reranker,
        settings=settings,
    )
    return retriever, reranker, client, settings


def test_rerank_can_outrank_a_higher_rrf_score(tmp_path: Path) -> None:
    records = [
        _record("a.md#0000", "zebra zebra zebra"),
        _record("b.md#0000", "zebra other"),
    ]
    retriever, reranker, _client, _settings_obj = _build(
        tmp_path,
        records,
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [1, 0, 0, 0],
    )
    # a is rank 1 for both dense and BM25, so its RRF score is higher.
    # The reranker is allowed to put b first anyway.
    reranker.scores = {"zebra zebra zebra": 0.05, "zebra other": 0.95}
    trace = RequestTrace()
    hits = retriever.retrieve("zebra", top_n=2, top_k=2, trace=trace)
    assert hits[1].rrf_score > hits[0].rrf_score
    assert [hit.chunk_id for hit in hits] == ["b.md#0000", "a.md#0000"]
    assert hits[0].rerank_score == pytest.approx(0.95)
    assert hits[0].bm25_rank == 2
    assert hits[0].dense_rank == 2
    assert hits[1].dense_rank == 1
    assert hits[1].bm25_rank == 1
    assert reranker.queries == ["zebra"]
    assert retriever._embedder.queries == ["zebra"]
    names = [span.name for span in trace.spans]
    assert "retrieval.dense" in names
    assert "retrieval.bm25" in names
    assert "retrieval.rrf" in names
    assert "retrieval.rerank" in names
    for span in trace.spans:
        assert "zebra" not in span.attributes.values()


def test_equal_rerank_scores_keep_the_higher_rrf_score(tmp_path: Path) -> None:
    records = [
        _record("b.md#0000", "alpha only"),
        _record("a.md#0000", "zebra only"),
    ]
    retriever, reranker, _client, _settings_obj = _build(
        tmp_path,
        records,
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [1, 0, 0, 0],
    )
    reranker.scores = {"alpha only": 0.4, "zebra only": 0.4}
    hits = retriever.retrieve("zebra", top_n=2, top_k=2)
    # a is the BM25 hit and also a dense neighbor, so its fused score is higher.
    assert [hit.chunk_id for hit in hits] == ["a.md#0000", "b.md#0000"]
    assert hits[0].rrf_score > hits[1].rrf_score


def test_source_and_heading_filters_apply_to_both_lists(tmp_path: Path) -> None:
    records = [
        _record("a.md#0000", "zebra alpha", source="a.md", heading=("Keep",)),
        _record("b.md#0000", "zebra beta", source="a.md", heading=("Drop",)),
        _record("c.md#0000", "zebra gamma", source="c.md", heading=("Keep",)),
    ]
    retriever, reranker, _client, _settings_obj = _build(
        tmp_path,
        records,
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        [1, 0, 0, 0],
    )
    reranker.scores = {record.text: 0.2 for record in records}
    hits = retriever.retrieve(
        "zebra",
        metadata_filter=MetadataFilter(sources=("a.md",), heading="Keep"),
        top_n=3,
        top_k=3,
    )
    assert [hit.chunk_id for hit in hits] == ["a.md#0000"]


def test_a_filter_that_matches_nothing_skips_the_reranker(tmp_path: Path) -> None:
    records = [_record("a.md#0000", "zebra")]
    retriever, reranker, _client, _settings_obj = _build(
        tmp_path,
        records,
        [[1, 0, 0, 0]],
        [1, 0, 0, 0],
    )
    trace = RequestTrace()
    hits = retriever.retrieve(
        "zebra",
        metadata_filter=MetadataFilter(sources=("missing.md",)),
        top_n=1,
        top_k=1,
        trace=trace,
    )
    assert hits == []
    assert reranker.seen == []
    rerank = [span for span in trace.spans if span.name == "retrieval.rerank"]
    assert rerank[-1].attributes["count"] == 0


def test_a_query_with_no_tokens_still_searches_densely(tmp_path: Path) -> None:
    records = [_record("a.md#0000", "alpha semantic")]
    retriever, reranker, _client, _settings_obj = _build(
        tmp_path,
        records,
        [[1, 0, 0, 0]],
        [1, 0, 0, 0],
    )
    reranker.scores = {"alpha semantic": 0.7}
    hits = retriever.retrieve("???", top_n=1, top_k=1)
    assert [hit.chunk_id for hit in hits] == ["a.md#0000"]
    assert hits[0].bm25_score is None
    assert hits[0].dense_rank == 1


def test_empty_query_and_a_top_k_past_the_pool_are_rejected(tmp_path: Path) -> None:
    records = [_record("a.md#0000", "alpha")]
    retriever, _reranker, _client, _settings_obj = _build(
        tmp_path,
        records,
        [[1, 0, 0, 0]],
        [1, 0, 0, 0],
    )
    with pytest.raises(ValueError):
        retriever.retrieve("   ")
    with pytest.raises(ValueError):
        retriever.retrieve("alpha", top_n=1, top_k=2)


def test_a_point_without_chunk_id_is_an_error(tmp_path: Path) -> None:
    records = [_record("a.md#0000", "alpha")]
    retriever, _reranker, client, _settings_obj = _build(
        tmp_path,
        records,
        [[0, 1, 0, 0]],
        [1, 0, 0, 0],
    )
    client.upsert(
        collection_name="knowledge",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": [1, 0, 0, 0],
                    "sparse": models.SparseVector(indices=[], values=[]),
                },
                payload={},
            )
        ],
        wait=True,
    )
    with pytest.raises(RetrievalError, match="chunk_id"):
        retriever.retrieve("alpha", top_n=2, top_k=1)


def test_open_retriever_requires_a_matching_collection(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    embedder = ScriptedEmbedder([1, 0, 0, 0], [[1, 0, 0, 0]])
    reranker = ScoreReranker({})
    client = QdrantClient(":memory:")
    with pytest.raises(RetrievalError, match="does not exist"):
        open_retriever(
            settings,
            records=[_record("a.md#0000", "alpha")],
            embedder=embedder,
            client=client,
            reranker=reranker,
        )
    index_records(
        [_record("a.md#0000", "alpha")],
        embedder=embedder,
        client=client,
        settings=settings,
    )
    with pytest.raises(RetrievalError, match="1 point"):
        open_retriever(
            settings,
            records=[],
            embedder=embedder,
            client=client,
            reranker=reranker,
        )


def test_cli_prints_the_top_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from rag.retrieval.__main__ import main

    class _Fake:
        def retrieve(self, query, **kwargs) -> list[RetrievedChunk]:
            assert query == "how does forking work"
            assert kwargs["metadata_filter"].sources == ("hermes/acp-internals.md",)
            return [
                RetrievedChunk(
                    rank=1,
                    chunk_id="hermes/acp-internals.md#0002",
                    source="hermes/acp-internals.md",
                    doc_title="ACP Internals",
                    heading_path=("ACP Internals", "Forking"),
                    text="fork_session copies history",
                    token_count=4,
                    index=2,
                    oversized=False,
                    orphan=False,
                    dense_score=None,
                    dense_rank=None,
                    bm25_score=1.5,
                    bm25_rank=1,
                    rrf_score=0.016,
                    rerank_score=0.812,
                )
            ]

    monkeypatch.setattr("rag.retrieval.__main__.get_settings", lambda: object())
    monkeypatch.setattr("rag.retrieval.__main__.open_retriever", lambda settings: _Fake())
    main(["how does forking work", "--source", "hermes/acp-internals.md"])
    printed = capsys.readouterr().out
    assert "hermes/acp-internals.md#0002" in printed
    assert "rerank=0.812" in printed
    assert "dense=-" in printed


def test_cli_rejects_a_query_together_with_eval() -> None:
    from rag.retrieval.__main__ import main

    with pytest.raises(SystemExit, match="not both"):
        main(["hello", "--eval"])
    with pytest.raises(SystemExit, match="pass a query or --eval"):
        main([])
