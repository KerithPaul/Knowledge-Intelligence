"""Optional Langfuse export for one request trace.

The in-process `RequestTrace` stays the source for the API and the harness.
This module is a second sink. It stays off unless `langfuse.enabled` is true
and both keys are set. Missing keys are not an error.

Library code does not call `get_client()`. The client is built with the keys
from settings and its own tracer, so other libraries are not exported. The
Langfuse import happens only in that builder.

A generation observation is priced with `cost_usd` and the evaluation rates.
`openai/gpt-oss-20b` is a Groq model id, so the bill is the one in
`settings.yaml`, not a guess from Langfuse's model list. A call with a missing
token count is left unpriced. Local embed and rerank steps cost zero.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from rag.core.config import Settings
from rag.core.redact import redact_secrets
from rag.core.trace import RequestTrace, SpanCloser, SpanSink
from rag.evaluation.operational import cost_usd
from rag.routing.types import RoutedAnswer

_LOG = logging.getLogger("rag.serving")
_METADATA_LIMIT = 180
_STATUS_LIMIT = 500
_ZERO_COST_SPANS = frozenset({"embedding.encode", "retrieval.rerank"})
_GENERATION_SPANS = frozenset({"llm.complete", "llm.stream"})


class RootObservation:
    """The top observation for one request. `trace_id` is missing when export is off."""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id
        self._output: Any = None
        self._failed: str | None = None

    def set_output(self, output: Any) -> None:
        """The final public result. Chunk bodies do not belong here."""

        self._output = output

    def fail(self, message: str) -> None:
        """Mark the root as an error. The message is redacted."""

        self._failed = redact_secrets(message)


class Tracer:
    """One process exporter. `sink` is what a `RequestTrace` should use."""

    sink: SpanSink | None

    def trace(self) -> RequestTrace:
        return RequestTrace(sink=self.sink)

    def root(
        self,
        name: str,
        *,
        seed: str,
        tags: tuple[str, ...],
        environment: str,
        input: Any = None,
    ) -> Iterator[RootObservation]:
        raise NotImplementedError

    def flush(self) -> None:
        """Send buffered observations. Short-lived commands must call this."""


class NullTracer(Tracer):
    """Used when Langfuse is off. `trace_id` stays missing."""

    sink = None

    @contextmanager
    def root(
        self,
        name: str,
        *,
        seed: str,
        tags: tuple[str, ...],
        environment: str,
        input: Any = None,
    ) -> Iterator[RootObservation]:
        del name, seed, tags, environment, input
        yield RootObservation(None)

    def flush(self) -> None:
        return None


class _LangfuseTracer(Tracer):
    def __init__(
        self,
        client: Any,
        *,
        prompt_usd_per_million: float,
        completion_usd_per_million: float,
    ) -> None:
        self._client = client
        self.sink = _LangfuseSink(
            client,
            prompt_usd_per_million=prompt_usd_per_million,
            completion_usd_per_million=completion_usd_per_million,
        )

    @contextmanager
    def root(
        self,
        name: str,
        *,
        seed: str,
        tags: tuple[str, ...],
        environment: str,
        input: Any = None,
    ) -> Iterator[RootObservation]:
        from langfuse import Langfuse, propagate_attributes

        manager = None
        attributes = None
        observation = None
        entered_observation = False
        entered_attributes = False
        try:
            trace_id = Langfuse.create_trace_id(seed=seed)
            manager = self._client.start_as_current_observation(
                name=name,
                as_type=_root_type(name),
                input=input,
                trace_context={"trace_id": trace_id},
            )
            observation = manager.__enter__()
            entered_observation = True
            attributes = propagate_attributes(
                trace_name=name,
                tags=list(tags),
                environment=environment,
                metadata={"request_id": _clip(seed)},
            )
            attributes.__enter__()
            entered_attributes = True
        except Exception as exc:
            _LOG.warning("trace sink failed: %s", type(exc).__name__)
            if entered_attributes:
                _exit_quietly(attributes)
            if entered_observation:
                _exit_quietly(manager)
            yield RootObservation(None)
            return

        holder = RootObservation(trace_id)
        try:
            try:
                yield holder
            except (KeyboardInterrupt, GeneratorExit):
                raise
            except BaseException as exc:
                if holder._failed is None:
                    holder.fail(str(exc))
                raise
            finally:
                _finish_root(observation, holder)
        finally:
            _exit_quietly(attributes)
            _exit_quietly(manager)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            _LOG.warning("trace sink failed: %s", type(exc).__name__)


class _LangfuseSink:
    def __init__(
        self,
        client: Any,
        *,
        prompt_usd_per_million: float,
        completion_usd_per_million: float,
    ) -> None:
        self._client = client
        self._prompt_rate = prompt_usd_per_million
        self._completion_rate = completion_usd_per_million

    def open(self, name: str, *, observation: str) -> SpanCloser:
        manager = self._client.start_as_current_observation(name=name, as_type=observation)
        observation_handle = manager.__enter__()
        return _LangfuseCloser(
            manager,
            observation_handle,
            name=name,
            observation=observation,
            prompt_usd_per_million=self._prompt_rate,
            completion_usd_per_million=self._completion_rate,
        )


class _LangfuseCloser:
    def __init__(
        self,
        manager: Any,
        observation_handle: Any,
        *,
        name: str,
        observation: str,
        prompt_usd_per_million: float,
        completion_usd_per_million: float,
    ) -> None:
        self._manager = manager
        self._observation = observation_handle
        self._name = name
        self._kind = observation
        self._prompt_rate = prompt_usd_per_million
        self._completion_rate = completion_usd_per_million

    def close(
        self,
        *,
        attributes: dict[str, Any],
        level: str,
        status_message: str | None,
        output: Any,
    ) -> None:
        fields = observation_fields(
            self._name,
            self._kind,
            attributes,
            prompt_usd_per_million=self._prompt_rate,
            completion_usd_per_million=self._completion_rate,
        )
        if output is not None:
            fields["output"] = output
        if level == "ERROR":
            fields["level"] = "ERROR"
        if status_message:
            fields["status_message"] = status_message[:_STATUS_LIMIT]
        try:
            if fields:
                self._observation.update(**fields)
        finally:
            self._manager.__exit__(None, None, None)


def open_tracer(settings: Settings) -> Tracer:
    """Return a live exporter only when the flag and both keys are set.

    Either missing piece returns a null tracer. This does not import Langfuse
    on that path, and it does not talk to the network.
    """

    if not settings.langfuse.enabled:
        return NullTracer()
    public = settings.langfuse_public_key
    secret = settings.langfuse_secret_key
    if public is None or secret is None:
        return NullTracer()
    if not public.get_secret_value().strip() or not secret.get_secret_value().strip():
        return NullTracer()
    try:
        return _build_langfuse_tracer(settings)
    except Exception as exc:
        _LOG.warning("trace sink failed: %s", type(exc).__name__)
        return NullTracer()


@contextmanager
def tracing(settings: Settings) -> Iterator[Tracer]:
    """Open a tracer and flush it when the command ends."""

    tracer = open_tracer(settings)
    try:
        yield tracer
    finally:
        tracer.flush()


def observation_fields(
    name: str,
    observation: str,
    attributes: Mapping[str, Any],
    *,
    prompt_usd_per_million: float,
    completion_usd_per_million: float,
) -> dict[str, Any]:
    """Langfuse fields for one finished span. Missing usage is left unpriced."""

    fields: dict[str, Any] = {}
    model = attributes.get("model")
    if isinstance(model, str) and model:
        fields["model"] = model
    if name in _GENERATION_SPANS or observation == "generation":
        prompt = attributes.get("prompt_tokens")
        completion = attributes.get("completion_tokens")
        if _is_count(prompt) and _is_count(completion):
            fields["usage_details"] = {"input": prompt, "output": completion}
            fields["cost_details"] = {
                "input": cost_usd(
                    prompt,
                    0,
                    prompt_usd_per_million=prompt_usd_per_million,
                    completion_usd_per_million=completion_usd_per_million,
                ),
                "output": cost_usd(
                    0,
                    completion,
                    prompt_usd_per_million=prompt_usd_per_million,
                    completion_usd_per_million=completion_usd_per_million,
                ),
            }
    elif name in _ZERO_COST_SPANS or observation == "embedding":
        fields["cost_details"] = {"total": 0.0}
    metadata = _metadata(attributes)
    if metadata:
        fields["metadata"] = metadata
    return fields


def routed_output(routed: RoutedAnswer) -> dict[str, Any]:
    """Public ask result for the root observation. Chunk text stays out."""

    payload: dict[str, Any] = {
        "route": routed.route,
        "decided_by": routed.decided_by,
        "reason": routed.reason,
        "answer": routed.text,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "rank": chunk.rank,
                "rerank_score": chunk.rerank_score,
            }
            for chunk in routed.chunks
        ],
    }
    answer = routed.answer
    if answer is None:
        return payload
    payload["model"] = answer.model or None
    payload["prompt_tokens"] = answer.prompt_tokens
    payload["completion_tokens"] = answer.completion_tokens
    payload["citations"] = [
        {
            "number": item.number,
            "chunk_id": item.chunk_id,
            "source": item.source,
            "doc_title": item.doc_title,
            "heading_path": list(item.heading_path),
        }
        for item in answer.citations
    ]
    return payload


def mask_secrets(*, data: Any, **kwargs: Any) -> Any:
    """Langfuse mask hook. Strings pass through `redact_secrets`."""

    del kwargs
    return _walk(data)


def _build_langfuse_tracer(settings: Settings) -> Tracer:
    from langfuse import Langfuse
    from opentelemetry.sdk.trace import TracerProvider

    from rag import __version__

    public = settings.langfuse_public_key
    secret = settings.langfuse_secret_key
    if public is None or secret is None:
        return NullTracer()
    client = Langfuse(
        public_key=public.get_secret_value(),
        secret_key=secret.get_secret_value(),
        base_url=settings.langfuse.base_url,
        tracer_provider=TracerProvider(),
        mask=mask_secrets,
        release=__version__,
        environment=settings.serving.profile,
    )
    return _LangfuseTracer(
        client,
        prompt_usd_per_million=settings.evaluation.prompt_usd_per_million,
        completion_usd_per_million=settings.evaluation.completion_usd_per_million,
    )


def _exit_quietly(manager: Any) -> None:
    if manager is None:
        return
    try:
        manager.__exit__(None, None, None)
    except Exception as exc:
        _LOG.warning("trace sink failed: %s", type(exc).__name__)


def _finish_root(observation: Any, holder: RootObservation) -> None:
    fields: dict[str, Any] = {}
    if holder._output is not None:
        fields["output"] = holder._output
    if holder._failed:
        fields["level"] = "ERROR"
        fields["status_message"] = holder._failed[:_STATUS_LIMIT]
    if not fields:
        return
    try:
        observation.update(**fields)
    except Exception as exc:
        _LOG.warning("trace sink failed: %s", type(exc).__name__)


def _root_type(name: str) -> str:
    if name == "retrieval":
        return "retriever"
    if name in {"ingest", "indexing"}:
        return "span"
    return "chain"


def _metadata(attributes: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in attributes.items():
        if key in {"model", "prompt_tokens", "completion_tokens"}:
            continue
        packed = _metadata_value(value)
        if packed is not None:
            metadata[key] = packed
    return metadata


def _metadata_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value if isinstance(value, bool) else None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return _clip(value)
    return None


def _clip(value: str) -> str:
    if len(value) <= _METADATA_LIMIT:
        return value
    return value[:_METADATA_LIMIT]


def _is_count(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _walk(data: Any) -> Any:
    if isinstance(data, str):
        return redact_secrets(data)
    if isinstance(data, dict):
        return {str(key): _walk(value) for key, value in data.items()}
    if isinstance(data, list | tuple):
        return [_walk(item) for item in data]
    return data
