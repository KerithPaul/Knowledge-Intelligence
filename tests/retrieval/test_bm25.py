"""BM25 scores the chunk text with the sparse tokenizer."""

import pytest

from rag.indexing.records import ChunkRecord
from rag.retrieval.bm25 import BM25Index


def _record(chunk_id: str, text: str, token_count: int = 10) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source="doc.md",
        doc_title="Doc",
        heading_path=("Doc",),
        text=text,
        token_count=token_count,
        index=0,
        oversized=False,
        orphan=False,
    )


def test_lucene_idf_matches_a_hand_worked_example() -> None:
    index = BM25Index.build(
        [
            _record("d1", "cat cat dog"),
            _record("d2", "cat bird"),
            _record("d3", "bird bird"),
        ],
        k1=1.2,
        b=0.75,
    )
    hits = {hit.chunk_id: hit for hit in index.search("cat dog", limit=3)}
    assert list(hits) == ["d1", "d2"]
    assert hits["d1"].score == pytest.approx(1.4763707684067633)
    assert hits["d2"].score == pytest.approx(0.49917626830236755)
    assert hits["d1"].rank == 1
    assert hits["d2"].rank == 2


def test_a_repeated_query_token_is_counted_once() -> None:
    index = BM25Index.build([_record("d1", "cat")], k1=1.2, b=0.75)
    once = index.search("cat", limit=1)[0].score
    twice = index.search("cat cat", limit=1)[0].score
    assert once == pytest.approx(twice)


def test_more_repetitions_rank_above_a_single_mention() -> None:
    index = BM25Index.build(
        [_record("once", "cat pad"), _record("twice", "cat cat")],
        k1=1.2,
        b=0.75,
    )
    hits = index.search("cat", limit=2)
    assert [hit.chunk_id for hit in hits] == ["twice", "once"]


def test_a_short_chunk_beats_a_long_chunk_with_the_same_term() -> None:
    index = BM25Index.build(
        [_record("short", "zebra"), _record("long", "zebra " + "pad " * 20)],
        k1=1.2,
        b=0.75,
    )
    hits = index.search("zebra", limit=2)
    assert [hit.chunk_id for hit in hits] == ["short", "long"]


def test_a_rare_term_outranks_a_common_term() -> None:
    index = BM25Index.build(
        [_record("rare", "zebra"), _record("common-a", "cat"), _record("common-b", "cat")],
        k1=1.2,
        b=0.75,
    )
    hits = index.search("zebra cat", limit=3)
    assert hits[0].chunk_id == "rare"


def test_document_length_ignores_the_embedding_token_count() -> None:
    index = BM25Index.build(
        [_record("a", "zebra", token_count=1), _record("b", "zebra", token_count=5000)],
        k1=1.2,
        b=0.75,
    )
    hits = {hit.chunk_id: hit.score for hit in index.search("zebra", limit=2)}
    assert hits["a"] == pytest.approx(hits["b"])


def test_a_filter_does_not_recompute_idf() -> None:
    index = BM25Index.build(
        [_record("a", "zebra"), _record("b", "zebra")],
        k1=1.2,
        b=0.75,
    )
    full = {hit.chunk_id: hit.score for hit in index.search("zebra", limit=2)}
    only_a = {hit.chunk_id: hit.score for hit in index.search("zebra", limit=2, allowed={"a"})}
    assert list(only_a) == ["a"]
    assert only_a["a"] == pytest.approx(full["a"])


def test_punctuation_and_an_empty_index_return_nothing() -> None:
    index = BM25Index.build([_record("d1", "???")], k1=1.2, b=0.75)
    assert index.search("cat", limit=1) == []
    assert index.search("???", limit=1) == []
    assert BM25Index.build([], k1=1.2, b=0.75).search("cat", limit=1) == []


def test_limit_must_be_positive() -> None:
    index = BM25Index.build([_record("d1", "cat")], k1=1.2, b=0.75)
    with pytest.raises(ValueError):
        index.search("cat", limit=0)
