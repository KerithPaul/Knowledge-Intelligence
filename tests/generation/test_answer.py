"""Grounded answers cite real blocks, or they return the insufficient signal."""

import json
from pathlib import Path

import pytest

from rag.core.config import load_settings
from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace
from rag.generation.__main__ import main
from rag.generation.pipeline import GroundedGenerator, open_generator
from rag.generation.types import INSUFFICIENT_CONTEXT
from rag.retrieval.types import RetrievedChunk


def test_empty_retrieval_does_not_call_the_model() -> None:
    llm = _ScriptedLLM([])
    trace = RequestTrace()
    answer = GroundedGenerator(llm, max_context_chunks=5, max_tokens=64).generate(
        "  How does forking work? ",
        [],
        trace=trace,
    )
    assert llm.calls == []
    assert answer.query == "How does forking work?"
    assert answer.sufficient is False
    assert answer.insufficient_reason == "empty_context"
    assert answer.answer == INSUFFICIENT_CONTEXT
    assert answer.citations == ()
    assert answer.prompt_tokens == 0
    assert answer.completion_tokens == 0
    assert answer.latency_ms == 0.0
    assert answer.model == ""
    span = trace.spans[0]
    assert span.name == "generation.answer"
    assert span.attributes["reason"] == "empty_context"
    assert span.attributes["sufficient"] is False


def test_cited_answer_maps_markers_back_to_chunks() -> None:
    reply = _reply("Forking copies the session [1].", citations=[1])
    llm = _ScriptedLLM([reply])
    trace = RequestTrace()
    answer = GroundedGenerator(llm, max_context_chunks=5, max_tokens=64).generate(
        "How does forking work?",
        [_chunk("a", "The fork copies the session."), _chunk("b", "Cancel stops it.", rank=2)],
        trace=trace,
    )
    assert answer.sufficient is True
    assert answer.answer == "Forking copies the session [1]."
    assert [item.chunk_id for item in answer.citations] == ["a"]
    assert answer.citations[0].number == 1
    assert answer.citations[0].heading_path == ("Forking",)
    assert answer.model == "scripted"
    assert answer.prompt_tokens == 10
    assert answer.completion_tokens == 5
    user = llm.calls[0][0][1].content
    assert "How does forking work?" in user
    assert "The fork copies the session." in user
    assert "Cancel stops it." in user
    assert llm.calls[0][0][0].role == "system"
    span = trace.spans[-1]
    assert span.name == "generation.answer"
    assert span.attributes["sufficient"] is True
    assert span.attributes["citations"] == 1
    assert "reason" not in span.attributes
    blob = " ".join(str(value) for value in span.attributes.values())
    assert "Forking copies the session" not in blob
    assert "The fork copies the session." not in blob


def test_context_limit_drops_later_chunks(tmp_path: Path) -> None:
    reply = _reply("Only the first [1].", citations=[1])
    llm = _ScriptedLLM([reply])
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    settings = load_settings(
        env_file=env_file,
        generation={"max_context_chunks": 1, "max_tokens": 32},
    )
    answer = open_generator(settings, llm=llm).generate(
        "What forks?",
        [_chunk("a", "alpha text"), _chunk("b", "beta text", rank=2)],
    )
    assert [block.chunk_id for block in answer.context] == ["a"]
    assert "beta text" not in llm.calls[0][0][1].content
    assert llm.calls[0][1] == 32


def test_bad_json_is_retried_and_tokens_add_up() -> None:
    good = _reply("Forking copies the session [1].", citations=[1])
    llm = _ScriptedLLM(["nope", good])
    answer = GroundedGenerator(llm, max_context_chunks=2, max_tokens=64).generate(
        "How does forking work?",
        [_chunk("a", "The fork copies the session.")],
    )
    assert answer.sufficient is True
    assert len(llm.calls) == 2
    assert answer.prompt_tokens == 20
    assert answer.completion_tokens == 10


def test_two_bad_replies_raise_and_the_trace_hides_them() -> None:
    llm = _ScriptedLLM(["LEAKED-REPLY", "STILL-LEAKED"])
    trace = RequestTrace()
    with pytest.raises(GenerationError, match="could not be read") as caught:
        GroundedGenerator(llm, max_context_chunks=2, max_tokens=64).generate(
            "How does forking work?",
            [_chunk("a", "The fork copies the session.")],
            trace=trace,
        )
    message = str(caught.value)
    assert "LEAKED-REPLY" not in message
    assert "STILL-LEAKED" not in message
    span = trace.spans[-1]
    assert span.attributes["error"] == "unreadable"
    blob = " ".join(str(value) for value in span.attributes.values())
    assert "LEAKED-REPLY" not in blob


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="query"):
        GroundedGenerator(_ScriptedLLM([]), max_context_chunks=1, max_tokens=8).generate(
            "  ",
            [],
        )


def test_cli_requires_a_query() -> None:
    with pytest.raises(SystemExit, match="pass a query"):
        main([])
    with pytest.raises(SystemExit, match="pass a query"):
        main(["   "])


class _ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[LLMMessage], int | None]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        self.calls.append((list(messages), max_tokens))
        content = self._replies.pop(0)
        if trace is not None:
            trace.record("llm.complete", 1.0, model="scripted")
        return LLMResponse(
            content=content,
            model="scripted",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
        )


def _reply(answer: str, *, citations: list[int]) -> str:
    return json.dumps({"sufficient": True, "answer": answer, "citations": citations})


def _chunk(chunk_id: str, text: str, *, rank: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text=text,
        token_count=20,
        index=0,
        oversized=False,
        orphan=False,
        dense_score=0.5,
        dense_rank=rank,
        bm25_score=1.0,
        bm25_rank=rank,
        rrf_score=0.02,
        rerank_score=0.8,
    )
