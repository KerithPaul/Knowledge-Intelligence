"""Decide whether the indexed docs are enough, then answer.

Stage 1 compares the highest rerank probability to
``routing.sufficiency_threshold``. That probability is the cross-encoder
score. The retriever already returns chunks best first, so the highest
score is the first chunk when this retriever built the list. A score
below the threshold skips the model. No chunks is the same stop: there
is nothing to score.

Stage 2 is the grounded answer from Phase 4. ``GroundedAnswer.sufficient``
is the groundedness check. This module does not send a second judge
prompt, and it does not search the web. An insufficient result uses the
Phase 4 sentence. A later agent can treat that result as the moment to
call its own web-search tool.

The trace records ``routing.score`` after retrieval and ``routing.decision``
once a route is chosen. Spans do not store the answer text.
"""

import math
import time
from collections.abc import Iterator, Sequence
from typing import Protocol

from rag.core.config import Settings
from rag.core.trace import RequestTrace
from rag.generation.pipeline import GroundedGenerator, open_generator
from rag.generation.types import ContextSource, GroundedAnswer
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.pipeline import Retriever, open_retriever
from rag.retrieval.types import RetrievedChunk
from rag.routing.events import AskDelta, AskResult, AskStatus, iter_text_deltas
from rag.routing.types import DecidedBy, RoutedAnswer, RoutingReason


class _Searcher(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        top_n: int | None = None,
        top_k: int | None = None,
        trace: RequestTrace | None = None,
    ) -> list[RetrievedChunk]: ...


class _Answerer(Protocol):
    def generate(
        self,
        query: str,
        chunks: Sequence[ContextSource],
        *,
        trace: RequestTrace | None = None,
    ) -> GroundedAnswer: ...


class Router:
    """One gated answer per query. Build it once and reuse the clients."""

    def __init__(
        self,
        retriever: _Searcher,
        generator: _Answerer,
        *,
        sufficiency_threshold: float,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._threshold = _require_threshold(sufficiency_threshold)

    def ask(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        top_n: int | None = None,
        top_k: int | None = None,
        trace: RequestTrace | None = None,
    ) -> RoutedAnswer:
        """Retrieve, apply the score gate, and generate only when the score passes."""

        cleaned = _clean_query(query)
        chosen, best, passed = self._retrieve(
            cleaned,
            metadata_filter=metadata_filter,
            top_n=top_n,
            top_k=top_k,
            trace=trace,
        )
        routed = self._route(cleaned, chosen, best, passed, trace=trace, stream=False)
        _record_decision(trace, routed)
        return routed

    def ask_stream(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        top_n: int | None = None,
        top_k: int | None = None,
        trace: RequestTrace | None = None,
    ) -> Iterator[AskStatus | AskDelta | AskResult]:
        """Same gate as `ask`. Text events are the grounded answer.

        A score refusal does not call the model. A passing score streams from
        the provider when the generator supports it, then emits the accepted
        text. The raw provider tokens are not events.
        """

        cleaned = _clean_query(query)
        yield AskStatus(stage="retrieving")
        chosen, best, passed = self._retrieve(
            cleaned,
            metadata_filter=metadata_filter,
            top_n=top_n,
            top_k=top_k,
            trace=trace,
        )
        yield AskStatus(
            stage="gate",
            best_rerank_score=best,
            threshold=self._threshold,
            chunk_count=len(chosen),
            passed=passed,
        )
        if passed:
            yield AskStatus(stage="generating")
        routed = self._route(cleaned, chosen, best, passed, trace=trace, stream=True)
        _record_decision(trace, routed)
        yield from iter_text_deltas(routed.text)
        yield AskResult(routed=routed)

    def _retrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None,
        top_n: int | None,
        top_k: int | None,
        trace: RequestTrace | None,
    ) -> tuple[tuple[RetrievedChunk, ...], float | None, bool]:
        chunks = self._retriever.retrieve(
            query,
            metadata_filter=metadata_filter,
            top_n=top_n,
            top_k=top_k,
            trace=trace,
        )
        chosen = tuple(chunks)
        started = time.perf_counter()
        best = _best_score(chosen)
        passed = best is not None and best >= self._threshold
        _record_score(
            trace,
            latency_ms=(time.perf_counter() - started) * 1000,
            threshold=self._threshold,
            chunk_count=len(chosen),
            best=best,
            passed=passed,
        )
        return chosen, best, passed

    def _route(
        self,
        query: str,
        chosen: tuple[RetrievedChunk, ...],
        best: float | None,
        passed: bool,
        *,
        trace: RequestTrace | None,
        stream: bool,
    ) -> RoutedAnswer:
        if not passed:
            reason: RoutingReason = "empty_context" if best is None else "below_threshold"
            return _refused(
                query,
                decided_by="retrieval_score",
                reason=reason,
                best=best,
                threshold=self._threshold,
                chunks=chosen,
            )
        answer = _generate(self._generator, query, chosen, trace=trace, stream=stream)
        if answer.sufficient:
            return RoutedAnswer(
                query=query,
                route="knowledge_base",
                decided_by="groundedness",
                reason=None,
                best_rerank_score=best,
                threshold=self._threshold,
                chunks=chosen,
                answer=answer,
            )
        if answer.insufficient_reason is None:
            raise ValueError("an insufficient answer needs a reason")
        return _refused(
            query,
            decided_by="groundedness",
            reason=answer.insufficient_reason,
            best=best,
            threshold=self._threshold,
            chunks=chosen,
            answer=answer,
        )


def open_router(
    settings: Settings,
    *,
    retriever: Retriever | None = None,
    generator: GroundedGenerator | None = None,
) -> Router:
    """Build the ask path from settings.

    Pass ``retriever`` and ``generator`` in tests to skip Qdrant and Groq.
    A missing retriever opens the Phase 3 retriever. A missing generator
    opens the Phase 4 generator.
    """

    searcher = retriever if retriever is not None else open_retriever(settings)
    writer = generator if generator is not None else open_generator(settings)
    return Router(
        searcher,
        writer,
        sufficiency_threshold=settings.routing.sufficiency_threshold,
    )


def _clean_query(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("query text is empty")
    return cleaned


def _generate(
    generator: _Answerer,
    query: str,
    chunks: tuple[RetrievedChunk, ...],
    *,
    trace: RequestTrace | None,
    stream: bool,
) -> GroundedAnswer:
    """Call the streaming generator when this ask is a stream and it has one.

    A test double that only implements `generate` still works. The non-stream
    ask path always uses `generate`, so a one-shot completion stays one shot.
    """

    if stream:
        generate_stream = getattr(generator, "generate_stream", None)
        if generate_stream is not None:
            return generate_stream(query, chunks, trace=trace)
    return generator.generate(query, chunks, trace=trace)


def _require_threshold(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("sufficiency_threshold must be between 0 and 1")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("sufficiency_threshold must be between 0 and 1")
    return threshold


def _best_score(chunks: Sequence[RetrievedChunk]) -> float | None:
    if not chunks:
        return None
    best = max(chunk.rerank_score for chunk in chunks)
    if isinstance(best, bool) or not isinstance(best, int | float) or not math.isfinite(best):
        raise ValueError("best rerank score must be a finite number")
    return float(best)


def _refused(
    query: str,
    *,
    decided_by: DecidedBy,
    reason: RoutingReason,
    best: float | None,
    threshold: float,
    chunks: tuple[RetrievedChunk, ...],
    answer: GroundedAnswer | None = None,
) -> RoutedAnswer:
    return RoutedAnswer(
        query=query,
        route="insufficient",
        decided_by=decided_by,
        reason=reason,
        best_rerank_score=best,
        threshold=threshold,
        chunks=chunks,
        answer=answer,
    )


def _record_score(
    trace: RequestTrace | None,
    *,
    latency_ms: float,
    threshold: float,
    chunk_count: int,
    best: float | None,
    passed: bool,
) -> None:
    if trace is None:
        return
    attributes: dict[str, object] = {
        "threshold": threshold,
        "chunk_count": chunk_count,
        "passed": passed,
    }
    if best is not None:
        attributes["best_rerank_score"] = best
    trace.record("routing.score", latency_ms, **attributes)


def _record_decision(trace: RequestTrace | None, routed: RoutedAnswer) -> None:
    if trace is None:
        return
    started = time.perf_counter()
    attributes: dict[str, object] = {
        "route": routed.route,
        "decided_by": routed.decided_by,
        "threshold": routed.threshold,
        "chunk_count": len(routed.chunks),
    }
    if routed.best_rerank_score is not None:
        attributes["best_rerank_score"] = routed.best_rerank_score
    if routed.reason is not None:
        attributes["reason"] = routed.reason
    latency_ms = (time.perf_counter() - started) * 1000
    trace.record("routing.decision", latency_ms, **attributes)
