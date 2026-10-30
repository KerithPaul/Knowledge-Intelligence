"""Request traces store timings, not secrets."""

import pytest

from rag.core.trace import RequestTrace


def test_record_appends_a_span() -> None:
    trace = RequestTrace()
    trace.record("llm.complete", 12.5, model="openai/gpt-oss-20b", prompt_tokens=3)
    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.name == "llm.complete"
    assert span.latency_ms == 12.5
    assert span.attributes["model"] == "openai/gpt-oss-20b"
    assert span.attributes["prompt_tokens"] == 3


def test_record_rejects_secret_attribute_names() -> None:
    trace = RequestTrace()
    with pytest.raises(ValueError, match="secrets"):
        trace.record("llm.complete", 1.0, api_key="should-not-land")
    assert trace.spans == []
