"""A proposed question must copy its evidence from the chunk."""

import json

import pytest

from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMResponse
from rag.evaluation.qa import load_qa_cases
from rag.evaluation.synthesize import (
    chunks_to_propose,
    evidence_owners,
    marker_is_absent,
    propose_from_chunk,
    propose_unanswerable,
    write_qa_lines,
)
from rag.indexing.records import ChunkRecord


def test_a_copied_span_becomes_an_answerable_row(tmp_path) -> None:
    chunk = _chunk("hermes/adding-providers.md#0002", "Use HERMES_INFERENCE_PROVIDER for the id.")
    reply = json.dumps(
        {
            "question": "Which variable accepts the provider id?",
            "reference_answer": "Use HERMES_INFERENCE_PROVIDER for the id.",
            "evidence": "HERMES_INFERENCE_PROVIDER",
        }
    )
    case = propose_from_chunk(chunk, _ScriptedLLM([reply]))
    assert case.corpus_can_answer is True
    assert case.relevant_chunk_ids == ("hermes/adding-providers.md#0002",)
    assert case.evidence == "HERMES_INFERENCE_PROVIDER"
    path = tmp_path / "candidates.jsonl"
    write_qa_lines([case], path)
    assert load_qa_cases(path) == [case]


def test_evidence_outside_the_chunk_is_rejected() -> None:
    chunk = _chunk("a", "Only this sentence is in the chunk.")
    reply = json.dumps(
        {
            "question": "What is outside?",
            "reference_answer": "A fact from somewhere else entirely.",
            "evidence": "somewhere else",
        }
    )
    with pytest.raises(GenerationError, match="not copied"):
        propose_from_chunk(chunk, _ScriptedLLM([reply, reply]))


def test_an_outside_question_must_contain_its_marker() -> None:
    reply = json.dumps({"question": "What role do mitochondria play?", "marker": "mitochondria"})
    proposal = propose_unanswerable(_ScriptedLLM([reply]))
    assert proposal.marker == "mitochondria"
    assert marker_is_absent("mitochondria", "Hermes agent internals")
    assert not marker_is_absent("Hermes", "Hermes agent internals")
    missing = json.dumps({"question": "What is water?", "marker": "mitochondria"})
    with pytest.raises(GenerationError, match="marker"):
        propose_unanswerable(_ScriptedLLM([missing, missing]))


def test_propose_skips_covered_chunks_in_file_order() -> None:
    records = [_chunk("a", "one"), _chunk("b", "two"), _chunk("c", "three")]
    chosen = chunks_to_propose(records, {"a"}, limit=1)
    assert [record.chunk_id for record in chosen] == ["b"]
    assert evidence_owners("two", records) == ["b"]


class _ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        del messages, temperature, max_tokens, trace
        return LLMResponse(
            content=self._replies.pop(0),
            model="scripted",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=1.0,
        )


def _chunk(chunk_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source=chunk_id.split("#", 1)[0],
        doc_title="Doc",
        heading_path=("Section",),
        text=text,
        token_count=10,
        index=0,
        oversized=False,
        orphan=False,
    )
