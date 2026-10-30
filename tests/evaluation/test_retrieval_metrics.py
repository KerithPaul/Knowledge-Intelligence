"""Retrieval metrics follow the usual definitions, including a short list."""

import math

import pytest

from rag.evaluation.retrieval import (
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_queries,
    write_retrieval_report,
)


def test_a_perfect_top_scores_one_except_precision_when_k_is_longer() -> None:
    ranking = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert precision_at_k(ranking, relevant, 3) == pytest.approx(2 / 3)
    assert recall_at_k(ranking, relevant, 3) == 1
    assert hit_at_k(ranking, relevant, 3) == 1
    assert reciprocal_rank(ranking, relevant) == 1
    assert ndcg_at_k(ranking, relevant, 3) == pytest.approx(1)
    assert average_precision(ranking, relevant) == pytest.approx(1)


def test_precision_divides_by_k_when_the_list_is_short() -> None:
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(1 / 5)
    assert recall_at_k(["a"], {"a"}, 5) == 1
    assert hit_at_k(["a"], {"a"}, 5) == 1


def test_a_miss_scores_zero() -> None:
    ranking = ["x", "y"]
    relevant = {"a"}
    assert precision_at_k(ranking, relevant, 2) == 0
    assert recall_at_k(ranking, relevant, 2) == 0
    assert hit_at_k(ranking, relevant, 2) == 0
    assert reciprocal_rank(ranking, relevant) == 0
    assert ndcg_at_k(ranking, relevant, 2) == 0
    assert average_precision(ranking, relevant) == 0


def test_ndcg_and_average_precision_for_a_partial_hit() -> None:
    ranking = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert precision_at_k(ranking, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(ranking, relevant, 3) == pytest.approx(1 / 2)
    assert reciprocal_rank(ranking, relevant) == pytest.approx(1 / 2)
    # b at rank 2 and d at rank 4. Two relevant ids, so AP is (1/2 + 2/4) / 2.
    assert average_precision(ranking, relevant) == pytest.approx(0.5)
    dcg = 1 / math.log2(3)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(ranking, relevant, 3) == pytest.approx(dcg / idcg)


def test_macro_averages_and_the_report_file(tmp_path) -> None:
    report = score_queries(
        [
            ("first", ["a", "b"], {"a"}),
            ("second", ["x"], {"a"}),
        ],
        k=1,
    )
    assert report.queries == 2
    assert report.hit_at_k == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)
    assert report.per_query[1].hit_at_k == 0
    path = tmp_path / "retrieval_report.json"
    write_retrieval_report(report, path)
    text = path.read_text(encoding="utf-8")
    assert '"map"' in text
    assert "first" in text


def test_bad_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], set(), 1)
    with pytest.raises(ValueError):
        precision_at_k(["a", "a"], {"a"}, 1)
    with pytest.raises(ValueError):
        score_queries([], k=1)
