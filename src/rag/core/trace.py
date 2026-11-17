"""A small trace of one request.

Spans store timings and plain attributes. They do not store prompts, answers,
or secrets. `span` closes the row even when the block raises, and can tell an
optional sink. The sink is how a later exporter sees the same step. A sink
error is logged and does not change the result of the block.
"""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag.core.redact import redact_secrets

_LOG = logging.getLogger("rag.serving")

_SECRET_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "groq_api_key",
        "langfuse_public_key",
        "langfuse_secret_key",
        "password",
        "secret",
        "tavily_api_key",
    }
)

# Langfuse observation types this repo uses. `span` is the plain one.
_OBSERVATIONS = frozenset(
    {
        "chain",
        "embedding",
        "evaluator",
        "generation",
        "retriever",
        "span",
    }
)


@dataclass
class Span:
    name: str
    latency_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)


class SpanCloser(Protocol):
    """Finishes one sink observation. Implementations must not raise into the app."""

    def close(
        self,
        *,
        attributes: dict[str, Any],
        level: str,
        status_message: str | None,
        output: Any,
    ) -> None:
        """Record the finished step. `output` is for the sink, not the span row."""


class SpanSink(Protocol):
    """Opens one observation around a `RequestTrace.span` block."""

    def open(self, name: str, *, observation: str) -> SpanCloser:
        """Start the observation. The caller closes it."""


class _SpanScope:
    """Attributes and the sink output for one open span."""

    def __init__(self) -> None:
        self._attributes: dict[str, Any] = {}
        self._output: Any = None
        self._started = time.perf_counter()
        self._error: BaseException | None = None

    def set(self, **attributes: Any) -> None:
        """Add attributes stored on the span row. Secret-like names are refused."""

        _reject_secrets(attributes)
        self._attributes.update(attributes)

    def set_output(self, output: Any) -> None:
        """Text for the sink only. It is not copied onto the span row."""

        self._output = output


class _SilentSpan:
    """Used when the caller has no trace. `set` keeps nothing."""

    def set(self, **attributes: Any) -> None:
        del attributes

    def set_output(self, output: Any) -> None:
        del output


@dataclass
class RequestTrace:
    spans: list[Span] = field(default_factory=list)
    sink: SpanSink | None = None

    def record(self, name: str, latency_ms: float, **attributes: Any) -> None:
        _reject_secrets(attributes)
        self.spans.append(Span(name=name, latency_ms=latency_ms, attributes=dict(attributes)))

    @contextmanager
    def span(self, name: str, *, observation: str = "span") -> Iterator[_SpanScope]:
        """Time one step, then `record` it.

        On an exception the row still lands. `error` is the exception class
        name unless `set` already stored `error`. The sink sees `level` ERROR
        and a redacted message. The exception still propagates.
        """

        if observation not in _OBSERVATIONS:
            allowed = ", ".join(sorted(_OBSERVATIONS))
            raise ValueError(f"observation must be one of: {allowed}")
        closer = _open_sink(self.sink, name, observation)
        scope = _SpanScope()
        try:
            yield scope
        except BaseException as exc:
            scope._error = exc
            raise
        finally:
            attributes = dict(scope._attributes)
            level = "DEFAULT"
            status_message = None
            if scope._error is not None:
                attributes.setdefault("error", type(scope._error).__name__)
                level = "ERROR"
                status_message = redact_secrets(str(scope._error))
            latency_ms = (time.perf_counter() - scope._started) * 1000
            try:
                self.record(name, latency_ms, **attributes)
            finally:
                _close_sink(
                    closer,
                    attributes=attributes,
                    level=level,
                    status_message=status_message,
                    output=scope._output,
                )


@contextmanager
def open_span(
    trace: RequestTrace | None,
    name: str,
    *,
    observation: str = "span",
) -> Iterator[_SpanScope | _SilentSpan]:
    """`trace.span` when a trace was passed, otherwise a span that stores nothing."""

    if trace is None:
        yield _SilentSpan()
        return
    with trace.span(name, observation=observation) as scope:
        yield scope


def _reject_secrets(attributes: dict[str, Any]) -> None:
    blocked = sorted(key for key in attributes if key.lower() in _SECRET_NAMES)
    if blocked:
        joined = ", ".join(blocked)
        raise ValueError(f"trace attributes must not carry secrets: {joined}")


def _open_sink(sink: SpanSink | None, name: str, observation: str) -> SpanCloser | None:
    if sink is None:
        return None
    try:
        return sink.open(name, observation=observation)
    except Exception as exc:
        _LOG.warning("trace sink failed: %s", type(exc).__name__)
        return None


def _close_sink(
    closer: SpanCloser | None,
    *,
    attributes: dict[str, Any],
    level: str,
    status_message: str | None,
    output: Any,
) -> None:
    if closer is None:
        return
    try:
        closer.close(
            attributes=attributes,
            level=level,
            status_message=status_message,
            output=output,
        )
    except Exception as exc:
        _LOG.warning("trace sink failed: %s", type(exc).__name__)
