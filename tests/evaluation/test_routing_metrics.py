"""Routing rates compare the gate's route with the hand label."""

import json

import pytest

from rag.evaluation.routing import (
    RoutingObservation,
    score_routing,
    write_routing_report,
)


def test_false_fallback_and_missed_fallback_use_their_own_classes() -> None:
    report = score_routing(
        [
            _row("kept", label=True, route="knowledge_base"),
            _row("also kept", label=True, route="knowledge_base"),
            _row("refused", label=True, route="insufficient", reason="below_threshold"),
            _row("stopped", label=False, route="insufficient", reason="model"),
            _row("answered anyway", label=False, route="knowledge_base"),
        ],
        threshold=0.35,
    )
    assert report.queries == 5
    assert report.answerable == 3
    assert report.unanswerable == 2
    assert report.false_fallbacks == 1
    assert report.missed_fallbacks == 1
    assert report.false_fallback_rate == pytest.approx(1 / 3)
    assert report.missed_fallback_rate == pytest.approx(1 / 2)
    assert report.accuracy == pytest.approx(3 / 5)
    assert report.threshold == pytest.approx(0.35)
    assert [row.correct for row in report.per_query] == [True, True, False, True, False]


def test_a_missing_class_has_no_rate(tmp_path) -> None:
    report = score_routing(
        [_row("only", label=True, route="knowledge_base")],
        threshold=0.35,
    )
    assert report.false_fallback_rate == 0
    assert report.missed_fallback_rate is None
    assert report.unanswerable == 0
    path = tmp_path / "routing_report.json"
    write_routing_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["missed_fallback_rate"] is None
    assert payload["accuracy"] == 1
    assert payload["per_query"][0]["query"] == "only"


def test_bad_routing_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="no queries"):
        score_routing([], threshold=0.35)
    with pytest.raises(ValueError, match="threshold"):
        score_routing([_row("q", label=True, route="knowledge_base")], threshold=2)
    with pytest.raises(ValueError, match="route"):
        score_routing(
            [_row("q", label=True, route="web")],
            threshold=0.35,
        )


def _row(
    query: str,
    *,
    label: bool,
    route: str,
    reason: str | None = None,
) -> RoutingObservation:
    decided_by = "retrieval_score" if reason == "below_threshold" else "groundedness"
    return RoutingObservation(
        query=query,
        corpus_can_answer=label,
        route=route,
        decided_by=decided_by,
        reason=reason,
        best_rerank_score=None if reason == "empty_context" else 0.8,
    )
