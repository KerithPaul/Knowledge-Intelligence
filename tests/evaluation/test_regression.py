"""The gate fails a missing metric, a low floor, and a high ceiling."""

import pytest

from rag.evaluation.gate import DEFAULT_BASELINES, check_gate, load_baselines


def test_bounds_are_inclusive() -> None:
    metrics = {"routing.accuracy": 1.0, "routing.false_fallback_rate": 0.0}
    assert (
        check_gate(metrics, {"routing.accuracy": 1.0}, {"routing.false_fallback_rate": 0.0}) == ()
    )
    low = check_gate(metrics, {"routing.accuracy": 1.0, "win_rate": 0.9}, {})
    assert [item.metric for item in low] == ["win_rate"]
    assert low[0].actual is None
    high = check_gate(
        {"routing.false_fallback_rate": 0.2},
        {},
        {"routing.false_fallback_rate": 0.0},
    )
    assert high[0].direction == "max"
    assert high[0].actual == pytest.approx(0.2)
    nulls = check_gate({"routing.accuracy": None}, {"routing.accuracy": 0.5}, {})
    assert nulls[0].actual is None


def test_the_checked_in_baselines_name_the_starter_floors() -> None:
    floors, ceilings = load_baselines(DEFAULT_BASELINES)
    assert floors["retrieval.hit_at_k"] == 1.0
    assert floors["retrieval.recall_at_k"] == 1.0
    assert floors["routing.accuracy"] == 1.0
    assert ceilings["routing.false_fallback_rate"] == 0.0
    assert ceilings["routing.missed_fallback_rate"] == 0.0
    assert "faithfulness" not in floors
