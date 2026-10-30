"""Groq client maps messages and hides the API key when a call fails."""

from types import SimpleNamespace

import pytest

from rag.core.errors import LLMError, MissingSecretError, UnknownProviderError
from rag.core.interfaces import LLMMessage
from rag.core.trace import RequestTrace
from rag.generation.groq import GroqLLMClient


class _Completions:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _response(content: str = "pong", prompt_tokens: int = 4, completion_tokens: int = 1):
    message = SimpleNamespace(content=content, reasoning="hidden thoughts")
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage, model="openai/gpt-oss-20b")


def _client(completions: _Completions, **overrides) -> GroqLLMClient:
    settings = dict(
        api_key="gsk_unit_test_key",
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        temperature=0.2,
        max_tokens=64,
        timeout_seconds=30,
        reasoning_effort="low",
        include_reasoning=False,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    settings.update(overrides)
    return GroqLLMClient(**settings)


def test_complete_maps_messages_and_records_a_span() -> None:
    completions = _Completions(_response())
    client = _client(completions)
    trace = RequestTrace()
    result = client.complete(
        [
            LLMMessage(role="system", content="Be brief."),
            LLMMessage(role="user", content="Say pong"),
        ],
        trace=trace,
    )
    assert result.content == "pong"
    assert result.model == "openai/gpt-oss-20b"
    assert result.prompt_tokens == 4
    assert result.completion_tokens == 1
    assert result.latency_ms >= 0
    sent = completions.kwargs
    assert sent["model"] == "openai/gpt-oss-20b"
    assert sent["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say pong"},
    ]
    assert sent["temperature"] == 0.2
    assert sent["max_completion_tokens"] == 64
    assert sent["extra_body"] == {"reasoning_effort": "low", "include_reasoning": False}
    assert trace.spans[0].name == "llm.complete"
    assert trace.spans[0].attributes["finish_reason"] == "stop"
    assert "gsk_unit_test_key" not in repr(client)


def test_call_overrides_replace_defaults() -> None:
    completions = _Completions(_response())
    client = _client(completions)
    client.complete(
        [LLMMessage(role="user", content="ping")],
        temperature=0.5,
        max_tokens=16,
    )
    assert completions.kwargs["temperature"] == 0.5
    assert completions.kwargs["max_completion_tokens"] == 16


def test_reasoning_fields_are_omitted_when_unset() -> None:
    completions = _Completions(_response())
    client = _client(completions, reasoning_effort=None, include_reasoning=None)
    client.complete([LLMMessage(role="user", content="ping")])
    assert "extra_body" not in completions.kwargs


def test_empty_answer_is_an_error() -> None:
    client = _client(_Completions(_response(content="  ")))
    with pytest.raises(LLMError, match="empty"):
        client.complete([LLMMessage(role="user", content="ping")])


def test_api_error_is_redacted() -> None:
    secret = "gsk_unit_test_key"
    boom = RuntimeError(f"unauthorized Bearer {secret} and also {secret}")
    client = _client(_Completions(error=boom))
    with pytest.raises(LLMError) as caught:
        client.complete([LLMMessage(role="user", content="ping")])
    message = str(caught.value)
    assert secret not in message
    assert "Bearer [REDACTED]" in message
    assert caught.value.__cause__ is None


def test_missing_key_and_empty_messages() -> None:
    with pytest.raises(MissingSecretError):
        GroqLLMClient(
            api_key="  ",
            model="openai/gpt-oss-20b",
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
            max_tokens=64,
            timeout_seconds=30,
        )
    client = _client(_Completions(_response()))
    with pytest.raises(ValueError, match="empty"):
        client.complete([])


def test_factory_rejects_unknown_provider(tmp_path, monkeypatch) -> None:
    from rag.core.clients import build_llm_client
    from rag.core.config import load_settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text(
        """
llm:
  provider: other
  model: x
  base_url: https://example.test/v1
  temperature: 0.2
  max_tokens: 32
  timeout_seconds: 10
embeddings:
  provider: bge
  model: BAAI/bge-small-en-v1.5
  dimension: 384
  query_prefix: ""
  passage_prefix: ""
  batch_size: 4
  normalize: true
  device: cpu
reranker:
  model: BAAI/bge-reranker-base
chunking:
  chunk_size_tokens: 100
  chunk_overlap_tokens: 0
retrieval:
  top_n: 10
  top_k: 3
  rrf_k: 60
routing:
  sufficiency_threshold: 0.5
qdrant:
  url: http://localhost:6333
  collection: knowledge
""",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=present\n", encoding="utf-8")
    settings = load_settings(yaml_file=yaml_file, env_file=env_file)
    with pytest.raises(UnknownProviderError, match="other"):
        build_llm_client(settings)
