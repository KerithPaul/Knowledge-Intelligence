"""A streamed ask emits the grounded answer, not the raw model tokens."""

import json

import pytest

from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMResponse, LLMStreamEvent
from rag.core.trace import RequestTrace
from rag.generation.pipeline import GroundedGenerator
from rag.generation.types import INSUFFICIENT_CONTEXT
from rag.retrieval.types import RetrievedChunk
from rag.routing.events import AskDelta, AskResult, AskStatus
from rag.routing.pipeline import Router


def test_a_low_score_streams_the_refusal_and_skips_the_model() -> None:
    llm = _StreamingLLM([['{"sufficient": true, "answer": "nope [1].", "citations": [1]}']])
    events = _events(
        Router(
            _Retriever([_chunk("a", 0.1)]),
            GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
            sufficiency_threshold=0.35,
        )
    )
    assert isinstance(events[0], AskStatus)
    assert events[0].stage == "retrieving"
    assert isinstance(events[1], AskStatus)
    assert events[1].stage == "gate"
    assert events[1].passed is False
    assert all(isinstance(event, AskDelta) for event in events[2:-1])
    assert events[2:-1]
    assert llm.stream_calls == 0
    assert llm.complete_calls == 0
    result = events[-1]
    assert isinstance(result, AskResult)
    assert result.routed.route == "insufficient"
    assert result.routed.reason == "below_threshold"
    assert _text(events) == INSUFFICIENT_CONTEXT


def test_stream_hides_an_uncited_model_answer() -> None:
    raw = "Secret uncited fact that must not stream."
    llm = _StreamingLLM([[json.dumps({"sufficient": True, "answer": raw, "citations": []})]])
    trace = RequestTrace()
    events = _events(
        Router(
            _Retriever([_chunk("a", 0.9)]),
            GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
            sufficiency_threshold=0.35,
        ),
        trace=trace,
    )
    stages = [event.stage for event in events if isinstance(event, AskStatus)]
    assert stages == ["retrieving", "gate", "generating"]
    assert llm.stream_calls == 1
    assert llm.complete_calls == 0
    assert raw not in _text(events)
    result = events[-1]
    assert isinstance(result, AskResult)
    assert result.routed.reason == "uncited"
    assert result.routed.text == INSUFFICIENT_CONTEXT
    assert _text(events) == INSUFFICIENT_CONTEXT
    names = [span.name for span in trace.spans]
    assert "llm.stream" in names
    assert "generation.answer" in names
    assert raw not in str(trace.spans)


def test_a_cited_answer_is_the_stream_text() -> None:
    reply = json.dumps(
        {"sufficient": True, "answer": "Forking copies the session [1].", "citations": [1]}
    )
    llm = _StreamingLLM([[reply[:20], reply[20:]]])
    trace = RequestTrace()
    events = _events(
        Router(
            _Retriever([_chunk("a", 0.8)]),
            GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
            sufficiency_threshold=0.35,
        ),
        trace=trace,
    )
    result = events[-1]
    assert isinstance(result, AskResult)
    assert result.routed.route == "knowledge_base"
    assert result.routed.answer is not None
    assert [item.chunk_id for item in result.routed.answer.citations] == ["a"]
    assert _text(events) == "Forking copies the session [1]."
    assert result.routed.answer.prompt_tokens == 8
    assert [span.name for span in trace.spans].index("routing.score") < [
        span.name for span in trace.spans
    ].index("routing.decision")


def test_one_shot_ask_does_not_use_the_stream() -> None:
    reply = json.dumps(
        {"sufficient": True, "answer": "Forking copies the session [1].", "citations": [1]}
    )
    llm = _StreamingLLM([[reply]])
    routed = Router(
        _Retriever([_chunk("a", 0.8)]),
        GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
        sufficiency_threshold=0.35,
    ).ask("How does forking work?")
    assert routed.route == "knowledge_base"
    assert llm.complete_calls == 1
    assert llm.stream_calls == 0


def test_an_unreadable_stream_raises_before_any_answer_text() -> None:
    llm = _StreamingLLM([["not json"], ["still not json"]])
    events = []
    with pytest.raises(GenerationError, match="could not be read"):
        for event in Router(
            _Retriever([_chunk("a", 0.8)]),
            GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
            sufficiency_threshold=0.35,
        ).ask_stream("How does forking work?"):
            events.append(event)
    assert llm.stream_calls == 2
    assert not any(isinstance(event, AskDelta) for event in events)
    assert [event.stage for event in events if isinstance(event, AskStatus)] == [
        "retrieving",
        "gate",
        "generating",
    ]


def _events(router: Router, *, trace: RequestTrace | None = None) -> list[object]:
    return list(router.ask_stream("How does forking work?", trace=trace))


def _text(events: list[object]) -> str:
    return "".join(event.text for event in events if isinstance(event, AskDelta))


class _Retriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query, *, metadata_filter=None, top_n=None, top_k=None, trace=None):
        if trace is not None:
            trace.record("retrieval.dense", 1.0, count=len(self._chunks))
        return list(self._chunks)


class _StreamingLLM(LLMClient):
    """Each entry is one model attempt, already split into deltas."""

    def __init__(self, attempts: list[list[str]]) -> None:
        self._attempts = [list(attempt) for attempt in attempts]
        self.stream_calls = 0
        self.complete_calls = 0
        self.trace = RequestTrace()

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        del messages, temperature, max_tokens
        self.complete_calls += 1
        content = "".join("".join(attempt) for attempt in self._attempts)
        self._attempts.clear()
        if trace is not None:
            trace.record("llm.complete", 1.0, model="scripted")
        return LLMResponse(
            content=content,
            model="scripted",
            prompt_tokens=8,
            completion_tokens=4,
            latency_ms=1.0,
        )

    def stream(self, messages, *, temperature=None, max_tokens=None, trace=None):
        del messages, temperature, max_tokens
        self.stream_calls += 1
        if not self._attempts:
            raise GenerationError("model reply could not be read")
        pieces = self._attempts.pop(0)
        if trace is not None:
            self.trace = trace
            trace.record(
                "llm.stream",
                1.0,
                model="scripted",
                prompt_tokens=8,
                completion_tokens=4,
                finish_reason="stop",
            )
        for index, text in enumerate(pieces):
            last = index == len(pieces) - 1
            yield LLMStreamEvent(
                text=text,
                done=last,
                model="scripted",
                prompt_tokens=8 if last else None,
                completion_tokens=4 if last else None,
                finish_reason="stop" if last else None,
            )


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk_id,
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text=f"text for {chunk_id}",
        token_count=20,
        index=0,
        oversized=False,
        orphan=False,
        dense_score=0.5,
        dense_rank=1,
        bm25_score=1.0,
        bm25_rank=1,
        rrf_score=0.02,
        rerank_score=score,
    )
