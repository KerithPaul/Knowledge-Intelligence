"""The span sink and the Langfuse gate. These tests do not call Langfuse."""

import logging
from pathlib import Path

import pytest

from memory_sink import MemorySink
from rag.core.config import load_settings
from rag.core.errors import LLMError
from rag.core.observability import NullTracer, observation_fields, open_tracer
from rag.core.trace import RequestTrace, open_span
from rag.evaluation.operational import cost_usd

_PROMPT_RATE = 0.075
_COMPLETION_RATE = 0.30


def test_span_writes_the_same_row_the_sink_sees() -> None:
    sink = MemorySink()
    trace = RequestTrace(sink=sink)
    with open_span(trace, "routing.score", observation="span") as span:
        span.set(passed=False, threshold=0.35)
    assert trace.spans[0].name == "routing.score"
    assert trace.spans[0].attributes == {"passed": False, "threshold": 0.35}
    assert trace.spans[0].latency_ms >= 0
    assert sink.rows[0].name == "routing.score"
    assert sink.rows[0].observation == "span"
    assert sink.rows[0].attributes == trace.spans[0].attributes
    assert sink.rows[0].level == "DEFAULT"


def test_a_raised_error_is_a_span_and_the_sink_hides_the_key() -> None:
    sink = MemorySink()
    trace = RequestTrace(sink=sink)
    with pytest.raises(LLMError):
        with open_span(trace, "llm.stream", observation="generation"):
            raise LLMError("Bearer gsk-unit-secret")
    assert trace.spans[0].name == "llm.stream"
    assert trace.spans[0].attributes["error"] == "LLMError"
    assert "gsk-" not in str(trace.spans[0].attributes)
    row = sink.rows[0]
    assert row.level == "ERROR"
    assert row.status_message is not None
    assert "gsk-" not in row.status_message
    assert "[REDACTED]" in row.status_message


def test_a_sink_failure_keeps_the_result_and_logs_only_the_class(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="rag.serving")
    opening = MemorySink(fail_open=True)
    trace = RequestTrace(sink=opening)
    with open_span(trace, "routing.score") as span:
        span.set(passed=True)
    assert trace.spans[0].attributes["passed"] is True
    assert opening.rows == []

    closing = MemorySink(fail_close=True)
    trace = RequestTrace(sink=closing)
    with open_span(trace, "routing.decision") as span:
        span.set(route="insufficient")
    assert trace.spans[0].name == "routing.decision"
    assert closing.rows == []
    assert "RuntimeError" in caplog.text
    assert "sink open failed" not in caplog.text
    assert "sink close failed" not in caplog.text


def test_priced_generations_split_input_and_output() -> None:
    fields = observation_fields(
        "llm.complete",
        "generation",
        {
            "model": "openai/gpt-oss-20b",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "finish_reason": "stop",
        },
        prompt_usd_per_million=_PROMPT_RATE,
        completion_usd_per_million=_COMPLETION_RATE,
    )
    assert fields["model"] == "openai/gpt-oss-20b"
    assert fields["usage_details"] == {"input": 100, "output": 20}
    assert fields["cost_details"]["input"] == cost_usd(
        100,
        0,
        prompt_usd_per_million=_PROMPT_RATE,
        completion_usd_per_million=_COMPLETION_RATE,
    )
    assert fields["cost_details"]["output"] == cost_usd(
        0,
        20,
        prompt_usd_per_million=_PROMPT_RATE,
        completion_usd_per_million=_COMPLETION_RATE,
    )
    assert fields["metadata"] == {"finish_reason": "stop"}
    assert "prompt_tokens" not in fields["metadata"]


def test_missing_usage_omits_cost_and_local_steps_cost_zero() -> None:
    missing = observation_fields(
        "llm.stream",
        "generation",
        {"model": "openai/gpt-oss-20b", "prompt_tokens": None, "completion_tokens": 2},
        prompt_usd_per_million=_PROMPT_RATE,
        completion_usd_per_million=_COMPLETION_RATE,
    )
    assert "usage_details" not in missing
    assert "cost_details" not in missing

    embed = observation_fields(
        "embedding.encode",
        "embedding",
        {"model": "BAAI/bge-small-en-v1.5", "kind": "query", "count": 1},
        prompt_usd_per_million=_PROMPT_RATE,
        completion_usd_per_million=_COMPLETION_RATE,
    )
    assert embed["cost_details"] == {"total": 0.0}
    assert embed["metadata"]["kind"] == "query"


def test_open_tracer_stays_null_until_the_flag_and_both_keys_are_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "LANGFUSE__ENABLED",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE__BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")

    def forbid(_settings: object) -> NullTracer:
        raise AssertionError("the builder must stay unused")

    monkeypatch.setattr("rag.core.observability._build_langfuse_tracer", forbid)
    off = load_settings(
        env_file=env_file,
        langfuse={"enabled": False},
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    missing = load_settings(
        env_file=env_file,
        langfuse={"enabled": True},
        langfuse_public_key=None,
        langfuse_secret_key="sk-test",
    )
    blank = load_settings(
        env_file=env_file,
        langfuse={"enabled": True},
        langfuse_public_key="pk-test",
        langfuse_secret_key="   ",
    )
    assert isinstance(open_tracer(off), NullTracer)
    assert isinstance(open_tracer(missing), NullTracer)
    assert isinstance(open_tracer(blank), NullTracer)

    sentinel = NullTracer()
    monkeypatch.setattr(
        "rag.core.observability._build_langfuse_tracer",
        lambda _settings: sentinel,
    )
    ready = load_settings(
        env_file=env_file,
        langfuse={"enabled": True},
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
    )
    assert open_tracer(ready) is sentinel
