"""A small trace of one request.

Spans store timings and plain attributes. They do not store prompts, answers,
or secrets.
"""

from dataclasses import dataclass, field
from typing import Any

_SECRET_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "groq_api_key",
        "password",
        "secret",
        "tavily_api_key",
    }
)


@dataclass
class Span:
    name: str
    latency_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    spans: list[Span] = field(default_factory=list)

    def record(self, name: str, latency_ms: float, **attributes: Any) -> None:
        blocked = sorted(key for key in attributes if key.lower() in _SECRET_NAMES)
        if blocked:
            joined = ", ".join(blocked)
            raise ValueError(f"trace attributes must not carry secrets: {joined}")
        self.spans.append(Span(name=name, latency_ms=latency_ms, attributes=dict(attributes)))
