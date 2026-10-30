"""RRF adds 1/(k+rank) and breaks ties on chunk id."""

import pytest

from rag.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_sums_the_lists_and_keeps_the_limit() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60, limit=3)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 62)
    assert [chunk_id for chunk_id, _score in fused] == ["b", "a", "c"]


def test_a_duplicate_id_in_one_list_keeps_its_best_rank() -> None:
    fused = reciprocal_rank_fusion([["a", "a"]], k=60, limit=2)
    assert fused == [("a", pytest.approx(1 / 61))]


def test_equal_scores_break_on_chunk_id() -> None:
    fused = reciprocal_rank_fusion([["b"], ["a"]], k=60, limit=2)
    assert [chunk_id for chunk_id, _score in fused] == ["a", "b"]


def test_the_limit_cuts_the_tail() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60, limit=2)
    assert [chunk_id for chunk_id, _score in fused] == ["a", "b"]


def test_bad_constants_are_rejected() -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0, limit=1)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=60, limit=0)
