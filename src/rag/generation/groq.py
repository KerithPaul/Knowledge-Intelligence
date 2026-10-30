"""Groq chat client, through the OpenAI-compatible API."""

import re
import time
from collections.abc import Sequence
from typing import Any

from openai import OpenAI

from rag.core.errors import LLMError, MissingSecretError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_KEY_RE = re.compile(r"\b(?:gsk|sk)-[A-Za-z0-9_\-]+")
_ROLES = frozenset({"system", "user", "assistant"})


def _redact(text: str, secret: str) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _KEY_RE.sub("[REDACTED]", text)
    return text


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
        if extra:
            kwargs["extra_body"] = extra

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            safe = _redact(str(exc), self._api_key)
            raise LLMError(safe) from None
        latency_ms = (time.perf_counter() - started) * 1000

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("model returned no choices")
        message = choices[0].message
        content = _message_text(message).strip()
        if not content:
            raise LLMError("model returned an empty answer")

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        model_name = getattr(response, "model", None) or self._model
        result = LLMResponse(
            content=content,
            model=str(model_name),
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
