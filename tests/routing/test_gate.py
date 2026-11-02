"""The score gate skips the model, then the grounded answer decides the route."""

import json
from pathlib import Path

import pytest

from rag.core.config import load_settings
from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace
from rag.generation.pipeline import GroundedGenerator
from rag.generation.types import INSUFFICIENT_CONTEXT, GroundedAnswer, InsufficientReason
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.types import RetrievedChunk
from rag.routing.__main__ import main
from rag.routing.pipeline import Router, open_router


def test_a_score_under_the_threshold_skips_the_model() -> None:
    retriever = _ScriptedRetriever([_chunk("a", 0.349)])
    generator = _ScriptedGenerator(sufficient=True)
    trace = RequestTrace()
    routed = Router(retriever, generator, sufficiency_threshold=0.35).ask(
        "  How does forking work? ",
        trace=trace,
    )
    assert generator.calls == []
    assert retriever.calls[0]["query"] == "How does forking work?"
    assert routed.route == "insufficient"
    assert routed.decided_by == "retrieval_score"
    assert routed.reason == "below_threshold"
    assert routed.answer is None
    assert routed.text == INSUFFICIENT_CONTEXT
    assert routed.best_rerank_score == pytest.approx(0.349)
    assert [span.name for span in trace.spans] == [
        "retrieval.dense",
        "routing.score",
        "routing.decision",
    ]
    assert trace.spans[1].attributes["passed"] is False
    blob = " ".join(str(span.attributes) for span in trace.spans)
    assert INSUFFICIENT_CONTEXT not in blob
    assert "text for a" not in blob


def test_a_score_equal_to_the_threshold_calls_the_model() -> None:
    generator = _ScriptedGenerator(sufficient=True)
    routed = Router(
        _ScriptedRetriever([_chunk("low", 0.1), _chunk("tied", 0.35, rank=2)]),
        generator,
        sufficiency_threshold=0.35,
    ).ask("How does forking work?")
    assert routed.route == "knowledge_base"
    assert routed.decided_by == "groundedness"
    assert routed.reason is None
    assert routed.best_rerank_score == pytest.approx(0.35)
    assert routed.answer is not None
    assert routed.answer.sufficient is True
    query, chunks, _trace = generator.calls[0]
    assert query == "How does forking work?"
    assert [chunk.chunk_id for chunk in chunks] == ["low", "tied"]


def test_no_chunks_skip_the_model() -> None:
    generator = _ScriptedGenerator(sufficient=True)
    routed = Router(_ScriptedRetriever([]), generator, sufficiency_threshold=0.35).ask(
        "How does forking work?"
    )
    assert generator.calls == []
    assert routed.route == "insufficient"
    assert routed.reason == "empty_context"
    assert routed.best_rerank_score is None
    assert routed.chunks == ()
    assert routed.text == INSUFFICIENT_CONTEXT


def test_a_model_refusal_stays_insufficient() -> None:
    routed = Router(
        _ScriptedRetriever([_chunk("a", 0.9)]),
        _ScriptedGenerator(sufficient=False, reason="model"),
        sufficiency_threshold=0.35,
    ).ask("How does forking work?")
    assert routed.route == "insufficient"
    assert routed.decided_by == "groundedness"
    assert routed.reason == "model"
    assert routed.answer is not None
    assert routed.text == INSUFFICIENT_CONTEXT


def test_an_uncited_answer_stays_insufficient() -> None:
    routed = Router(
        _ScriptedRetriever([_chunk("a", 0.9)]),
        _ScriptedGenerator(sufficient=False, reason="uncited"),
        sufficiency_threshold=0.35,
    ).ask("How does forking work?")
    assert routed.route == "insufficient"
    assert routed.decided_by == "groundedness"
    assert routed.reason == "uncited"


def test_a_passing_score_uses_the_grounded_generator() -> None:
    reply = json.dumps(
        {
            "sufficient": True,
            "answer": "Forking copies the session [1].",
            "citations": [1],
        }
    )
    llm = _ScriptedLLM([reply])
    trace = RequestTrace()
    routed = Router(
        _ScriptedRetriever([_chunk("a", 0.8)]),
        GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
        sufficiency_threshold=0.35,
    ).ask("How does forking work?", trace=trace)
    assert llm.calls
    assert routed.route == "knowledge_base"
    assert routed.text == "Forking copies the session [1]."
    assert routed.answer is not None
    assert [item.chunk_id for item in routed.answer.citations] == ["a"]
    names = [span.name for span in trace.spans]
    assert names.index("routing.score") < names.index("generation.answer")
    assert names.index("generation.answer") < names.index("routing.decision")
    blob = " ".join(str(span.attributes) for span in trace.spans)
    assert "Forking copies the session" not in blob


def test_a_generation_error_does_not_invent_a_route() -> None:
    trace = RequestTrace()
    with pytest.raises(GenerationError, match="could not be read"):
        Router(
            _ScriptedRetriever([_chunk("a", 0.8)]),
            _ScriptedGenerator(sufficient=True, error=GenerationError("could not be read")),
            sufficiency_threshold=0.35,
        ).ask("How does forking work?", trace=trace)
    assert [span.name for span in trace.spans] == ["retrieval.dense", "routing.score"]
    assert trace.spans[1].attributes["passed"] is True


def test_filters_and_cuts_reach_retrieval() -> None:
    retriever = _ScriptedRetriever([_chunk("a", 0.2)])
    metadata = MetadataFilter(sources=("hermes/acp-internals.md",))
    Router(retriever, _ScriptedGenerator(sufficient=True), sufficiency_threshold=0.35).ask(
        "How does forking work?",
        metadata_filter=metadata,
        top_n=8,
        top_k=2,
    )
    call = retriever.calls[0]
    assert call["metadata_filter"] is metadata
    assert call["top_n"] == 8
    assert call["top_k"] == 2


def test_a_non_finite_score_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        Router(
            _ScriptedRetriever([_chunk("a", float("nan"))]),
            _ScriptedGenerator(sufficient=True),
            sufficiency_threshold=0.35,
        ).ask("How does forking work?")


def test_bad_thresholds_and_empty_queries_are_rejected() -> None:
    retriever = _ScriptedRetriever([])
    generator = _ScriptedGenerator(sufficient=True)
    for value in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError, match="threshold"):
            Router(retriever, generator, sufficiency_threshold=value)
    with pytest.raises(ValueError, match="query"):
        Router(retriever, generator, sufficiency_threshold=0.35).ask("   ")


def test_open_router_uses_the_configured_threshold(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    settings = load_settings(env_file=env_file, routing={"sufficiency_threshold": 0.42})
    routed = open_router(
        settings,
        retriever=_ScriptedRetriever([_chunk("a", 0.42)]),
        generator=_ScriptedGenerator(sufficient=True),
    ).ask("How does forking work?")
    assert routed.threshold == pytest.approx(0.42)
    assert routed.route == "knowledge_base"


def test_cli_requires_a_query_or_eval() -> None:
    with pytest.raises(SystemExit, match="pass a query or --eval"):
        main([])
    with pytest.raises(SystemExit, match="pass a query or --eval"):
        main(["   "])
    with pytest.raises(SystemExit, match="not both"):
        main(["--eval", "How does forking work?"])


class _ScriptedRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query, *, metadata_filter=None, top_n=None, top_k=None, trace=None):
        self.calls.append(
            {
                "query": query,
                "metadata_filter": metadata_filter,
                "top_n": top_n,
                "top_k": top_k,
            }
        )
        if trace is not None:
            trace.record("retrieval.dense", 1.0, count=len(self._chunks))
        return list(self._chunks)


class _ScriptedGenerator:
    def __init__(
        self,
        *,
        sufficient: bool,
        reason: InsufficientReason | None = None,
        error: Exception | None = None,
    ) -> None:
        self._sufficient = sufficient
        self._reason = reason
        self._error = error
        self.calls: list[tuple[str, tuple[RetrievedChunk, ...], RequestTrace | None]] = []

    def generate(self, query, chunks, *, trace=None):
        self.calls.append((query, tuple(chunks), trace))
        if self._error is not None:
            raise self._error
        text = "Forking copies the session." if self._sufficient else INSUFFICIENT_CONTEXT
        return GroundedAnswer(
            query=query,
            answer=text,
            sufficient=self._sufficient,
            insufficient_reason=self._reason,
            citations=(),
            context=(),
            model="scripted",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )


class _ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[LLMMessage]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        self.calls.append(list(messages))
        if trace is not None:
            trace.record("llm.complete", 1.0, model="scripted")
        return LLMResponse(
            content=self._replies.pop(0),
            model="scripted",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )


def _chunk(chunk_id: str, score: float, *, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
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
        dense_rank=rank,
        bm25_score=1.0,
        bm25_rank=rank,
        rrf_score=0.02,
        rerank_score=score,
    )
