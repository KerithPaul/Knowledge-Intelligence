"""Groq chat client, through the OpenAI-compatible API."""

import time
from collections.abc import Iterator, Sequence
from typing import Any

from openai import OpenAI

from rag.core.errors import LLMError, MissingSecretError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse, LLMStreamEvent
from rag.core.redact import redact_secrets
from rag.core.trace import RequestTrace

_ROLES = frozenset({"system", "user", "assistant"})


def _usage_count(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            else:
                text = getattr(part, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


class GroqLLMClient(LLMClient):
    """Chat completions against Groq's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
        timeout_seconds: float,
        reasoning_effort: str | None = None,
        include_reasoning: bool | None = None,
        client: Any | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise MissingSecretError("GROQ_API_KEY is not set. Add it to .env.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._include_reasoning = include_reasoning
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=2,
        )

    def __repr__(self) -> str:
        return f"GroqLLMClient(model={self._model!r}, base_url={self._base_url!r})"

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: RequestTrace | None = None,
    ) -> LLMResponse:
        kwargs = self._request_kwargs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMError(self._safe(exc)) from None
        latency_ms = (time.perf_counter() - started) * 1000

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("model returned no choices")
        message = choices[0].message
        content = _message_text(message).strip()
        if not content:
            raise LLMError("model returned an empty answer")

        usage = getattr(response, "usage", None)
        prompt_tokens = _usage_count(usage, "prompt_tokens")
        completion_tokens = _usage_count(usage, "completion_tokens")
        model_name = str(getattr(response, "model", None) or self._model)
        result = LLMResponse(
            content=content,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        if trace is not None:
            trace.record(
                "llm.complete",
                latency_ms,
                model=result.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=getattr(choices[0], "finish_reason", None),
            )
        return result

    def stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: RequestTrace | None = None,
    ) -> Iterator[LLMStreamEvent]:
        """Yield provider text deltas. The trace span is `llm.stream`.

        The joined text is checked the same way as `complete`: an empty reply
        is an error, and the error text cannot contain the API key. Token
        counts ride on the last event.
        """

        kwargs = self._request_kwargs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        started = time.perf_counter()
        raw: list[str] = []
        pieces: list[LLMStreamEvent] = []
        model_name = self._model
        finish_reason: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            response = self._client.chat.completions.create(**kwargs)
            for chunk in response:
                model_name = str(getattr(chunk, "model", None) or model_name)
                usage = getattr(chunk, "usage", None)
                seen_prompt = _usage_count(usage, "prompt_tokens")
                seen_completion = _usage_count(usage, "completion_tokens")
                if seen_prompt is not None:
                    prompt_tokens = seen_prompt
                if seen_completion is not None:
                    completion_tokens = seen_completion
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                reason = getattr(choice, "finish_reason", None)
                if reason:
                    finish_reason = str(reason)
                text = _message_text(getattr(choice, "delta", None))
                if not text:
                    continue
                raw.append(text)
                pieces.append(LLMStreamEvent(text=text, done=False, model=model_name))
        except Exception as exc:
            raise LLMError(self._safe(exc)) from None
        # Hold the pieces until the call finishes. A failed call must not
        # look like a partial answer, and an empty reply is an error.
        if not "".join(raw).strip():
            raise LLMError("model returned an empty answer")
        latency_ms = (time.perf_counter() - started) * 1000
        if trace is not None:
            trace.record(
                "llm.stream",
                latency_ms,
                model=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
            )
        yield from pieces
        yield LLMStreamEvent(
            text="",
            done=True,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def _safe(self, exc: Exception) -> str:
        return redact_secrets(str(exc), self._api_key)

    def _request_kwargs(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        if not messages:
            raise ValueError("messages must not be empty")
        payload: list[dict[str, str]] = []
        for message in messages:
            if message.role not in _ROLES:
                raise ValueError(f"unsupported message role: {message.role}")
            if not isinstance(message.content, str):
                raise ValueError("message content must be a string")
            payload.append({"role": message.role, "content": message.content})
        extra: dict[str, Any] = {}
        if self._reasoning_effort is not None:
            extra["reasoning_effort"] = self._reasoning_effort
        if self._include_reasoning is not None:
            extra["include_reasoning"] = self._include_reasoning
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": payload,
            "temperature": self._temperature if temperature is None else temperature,
            "max_completion_tokens": self._max_tokens if max_tokens is None else max_tokens,
        }
        if stream:
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        if extra:
            kwargs["extra_body"] = extra
        return kwargs
