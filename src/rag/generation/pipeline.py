"""Turn retrieved chunks into a grounded answer.

The model sees only the numbered blocks from `assemble_context`. The reply
must be the JSON object described in the system prompt. Citation numbers
that do not match a block are removed. Three outcomes share one sentence,
`INSUFFICIENT_CONTEXT`:

- the retrieval returned no chunks, and the model is not called
- the model sets sufficient to false
- the model claims the context is sufficient but cites no real block

Phase 5 reads `sufficient`. This module does not decide to search the web.
"""

import time
from collections.abc import Sequence
from typing import Any

from rag.core.config import Settings
from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace
from rag.generation.context import assemble_context, render_context
from rag.generation.parse import ParsedDraft, interpret_answer, request_json
from rag.generation.prompt import SYSTEM_PROMPT, user_message
from rag.generation.types import (
    INSUFFICIENT_CONTEXT,
    Citation,
    ContextBlock,
    ContextSource,
    GroundedAnswer,
)


class GroundedGenerator:
    """One grounded answer per query. Build it once and reuse the client."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_context_chunks: int,
        max_tokens: int,
    ) -> None:
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be positive")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._llm = llm
        self._limit = max_context_chunks
        self._max_tokens = max_tokens

    def generate(
        self,
        query: str,
        chunks: Sequence[ContextSource],
        *,
        trace: RequestTrace | None = None,
    ) -> GroundedAnswer:
        """Answer `query` from `chunks`. The chunks should already be best first."""

        return self._generate(query, chunks, trace=trace, stream=False)

    def generate_stream(
        self,
        query: str,
        chunks: Sequence[ContextSource],
        *,
        trace: RequestTrace | None = None,
    ) -> GroundedAnswer:
        """Same rules as `generate`. The provider stream is joined first.

        The returned text is the grounded answer, which can replace the
        model's own words. Callers that stream to a person should send this
        text, not the provider deltas.
        """

        return self._generate(query, chunks, trace=trace, stream=True)

    def _generate(
        self,
        query: str,
        chunks: Sequence[ContextSource],
        *,
        trace: RequestTrace | None,
        stream: bool,
    ) -> GroundedAnswer:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query text is empty")
        blocks = assemble_context(chunks, limit=self._limit)
        if not blocks:
            answer = _empty_answer(cleaned)
            _record(trace, answer, retries=0, latency_ms=0.0)
            return answer

        started = time.perf_counter()
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_message(cleaned, render_context(blocks))),
        ]

        def interpret(payload: dict[str, Any]) -> ParsedDraft:
            return interpret_answer(payload, context_size=len(blocks))

        try:
            draft, responses = request_json(
                self._llm,
                messages,
                interpret,
                max_tokens=self._max_tokens,
                trace=trace,
                stream=stream,
            )
        except GenerationError:
            if trace is not None:
                trace.record(
                    "generation.answer",
                    (time.perf_counter() - started) * 1000,
                    context_chunks=len(blocks),
                    retries=1,
                    error="unreadable",
                )
            raise
        answer = _from_draft(
            cleaned,
            blocks,
            draft,
            responses,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        _record(trace, answer, retries=len(responses) - 1, latency_ms=answer.latency_ms)
        return answer


def open_generator(settings: Settings, *, llm: LLMClient | None = None) -> GroundedGenerator:
    """Build the generator from settings. Pass `llm` in tests to skip the network."""

    from rag.core.clients import build_llm_client

    client = llm if llm is not None else build_llm_client(settings)
    generation = settings.generation
    return GroundedGenerator(
        client,
        max_context_chunks=generation.max_context_chunks,
        max_tokens=generation.max_tokens,
    )


def _empty_answer(query: str) -> GroundedAnswer:
    return GroundedAnswer(
        query=query,
        answer=INSUFFICIENT_CONTEXT,
        sufficient=False,
        insufficient_reason="empty_context",
        citations=(),
        context=(),
        model="",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
    )


def _from_draft(
    query: str,
    blocks: Sequence[ContextBlock],
    draft: ParsedDraft,
    responses: Sequence[LLMResponse],
    *,
    latency_ms: float,
) -> GroundedAnswer:
    by_number = {block.number: block for block in blocks}
    citations = tuple(_citation(by_number[number]) for number in draft.citation_numbers)
    last = responses[-1]
    return GroundedAnswer(
        query=query,
        answer=draft.answer,
        sufficient=draft.sufficient,
        insufficient_reason=draft.insufficient_reason,
        citations=citations,
        context=tuple(blocks),
        model=last.model,
        prompt_tokens=_sum_tokens(responses, "prompt_tokens"),
        completion_tokens=_sum_tokens(responses, "completion_tokens"),
        latency_ms=latency_ms,
    )


def _citation(block: ContextBlock) -> Citation:
    return Citation(
        number=block.number,
        chunk_id=block.chunk_id,
        source=block.source,
        doc_title=block.doc_title,
        heading_path=block.heading_path,
    )


def _sum_tokens(responses: Sequence[LLMResponse], name: str) -> int | None:
    total = 0
    for response in responses:
        value = getattr(response, name)
        if not isinstance(value, int):
            return None
        total += value
    return total


def _record(
    trace: RequestTrace | None,
    answer: GroundedAnswer,
    *,
    retries: int,
    latency_ms: float,
) -> None:
    if trace is None:
        return
    attributes: dict[str, object] = {
        "context_chunks": len(answer.context),
        "citations": len(answer.citations),
        "retries": retries,
        "sufficient": answer.sufficient,
    }
    if answer.insufficient_reason is not None:
        attributes["reason"] = answer.insufficient_reason
    trace.record("generation.answer", latency_ms, **attributes)
