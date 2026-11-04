"""Stage times stay separate from the clock, and a win needs the whole path."""

import pytest

from rag.core.trace import RequestTrace
from rag.evaluation.operational import (
    case_won,
    cost_usd,
    mean_stages,
    stage_times,
    tokens_from_trace,
)


def test_tokens_ignore_other_spans_and_missing_usage() -> None:
    trace = RequestTrace()
    trace.record("retrieval.dense", 10.0, count=5)
    trace.record("llm.complete", 4.0, prompt_tokens=100, completion_tokens=20)
    trace.record("llm.complete", 4.0, prompt_tokens=None, completion_tokens=None)
    usage = tokens_from_trace(trace)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.unpriced_calls == 1
    assert stage_times(trace)["llm.complete"] == 8.0
    stages = mean_stages([{"llm.complete": 8.0}, {"llm.complete": 2.0, "routing.decision": 1.0}])
    by_name = {stage.name: stage for stage in stages}
    assert by_name["llm.complete"].questions == 2
    assert by_name["llm.complete"].mean_ms == 5.0
    assert by_name["routing.decision"].questions == 1


def test_cost_uses_the_configured_rates() -> None:
    assert cost_usd(
        1_000_000, 2_000_000, prompt_usd_per_million=0.075, completion_usd_per_million=0.30
    ) == pytest.approx(0.675)
    assert cost_usd(10, 10, prompt_usd_per_million=0, completion_usd_per_million=0) == 0.0
    with pytest.raises(ValueError, match="prompt tokens"):
        cost_usd(-1, 0, prompt_usd_per_million=0, completion_usd_per_million=0)


def test_a_win_is_a_refusal_or_a_hit_plus_a_correct_answer() -> None:
    assert case_won(
        corpus_can_answer=False,
        route="insufficient",
        retrieved=(),
        relevant=(),
        sufficient=False,
        correctness=None,
        correctness_floor=0.5,
    )
    assert not case_won(
        corpus_can_answer=False,
        route="knowledge_base",
        retrieved=("a",),
        relevant=(),
        sufficient=True,
        correctness=1.0,
        correctness_floor=0.5,
    )
    assert case_won(
        corpus_can_answer=True,
        route="knowledge_base",
        retrieved=("other", "a"),
        relevant=("a",),
        sufficient=True,
        correctness=0.5,
        correctness_floor=0.5,
    )
    assert not case_won(
        corpus_can_answer=True,
        route="knowledge_base",
        retrieved=("other",),
        relevant=("a",),
        sufficient=True,
        correctness=1.0,
        correctness_floor=0.5,
    )
    assert not case_won(
        corpus_can_answer=True,
        route="insufficient",
        retrieved=("a",),
        relevant=("a",),
        sufficient=False,
        correctness=1.0,
        correctness_floor=0.5,
    )
    assert not case_won(
        corpus_can_answer=True,
        route="knowledge_base",
        retrieved=("a",),
        relevant=("a",),
        sufficient=True,
        correctness=None,
        correctness_floor=0.5,
    )
