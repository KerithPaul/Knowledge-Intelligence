"""Propose golden questions from a chunk, or from outside the corpus.

The model writes a question, a reference answer, and an evidence span.
The evidence must be copied from that chunk and must appear in the reference.
A second prompt writes a question the corpus should not be able to answer,
plus a marker phrase. The caller checks that the marker is absent from the
corpus before keeping the row.

Candidates are not written into the golden files. A person reviews them
and copies the ones that still look right.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMMessage
from rag.evaluation.qa import QACase
from rag.generation.parse import request_json
from rag.generation.prompt import (
    SYNTHESIZE_SYSTEM,
    UNANSWERABLE_SYSTEM,
    synthesize_user,
    unanswerable_user,
)
from rag.indexing.records import ChunkRecord

_MIN_EVIDENCE = 8


@dataclass(frozen=True)
class UnanswerableProposal:
    """A question and the phrase that must stay out of the corpus."""

    question: str
    marker: str


def propose_from_chunk(
    chunk: ChunkRecord,
    llm: LLMClient,
    *,
    max_tokens: int = 1024,
) -> QACase:
    """Ask for one answerable question grounded in ``chunk``."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not chunk.text.strip():
        raise ValueError("chunk text is empty")
    case, _responses = request_json(
        llm,
        [
            LLMMessage(role="system", content=SYNTHESIZE_SYSTEM),
            LLMMessage(role="user", content=synthesize_user(chunk.text)),
        ],
        _chunk_interpreter(chunk),
        max_tokens=max_tokens,
    )
    return case


def propose_unanswerable(
    llm: LLMClient,
    *,
    max_tokens: int = 1024,
) -> UnanswerableProposal:
    """Ask for one question that should fall outside this corpus."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    proposal, _responses = request_json(
        llm,
        [
            LLMMessage(role="system", content=UNANSWERABLE_SYSTEM),
            LLMMessage(role="user", content=unanswerable_user()),
        ],
        _unanswerable_interpreter,
        max_tokens=max_tokens,
    )
    return proposal


def chunks_to_propose(
    records: Sequence[ChunkRecord],
    covered_ids: set[str],
    *,
    limit: int,
) -> list[ChunkRecord]:
    """The first uncovered chunks, in file order, up to ``limit``."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    chosen: list[ChunkRecord] = []
    for record in records:
        if record.chunk_id in covered_ids:
            continue
        chosen.append(record)
        if len(chosen) >= limit:
            break
    return chosen


def evidence_owners(evidence: str, records: Sequence[ChunkRecord]) -> list[str]:
    """Chunk ids whose text contains ``evidence``."""

    if not evidence:
        raise ValueError("evidence is empty")
    return [record.chunk_id for record in records if evidence in record.text]


def marker_is_absent(marker: str, corpus_text: str) -> bool:
    """True when ``marker`` does not appear in ``corpus_text``, ignoring case."""

    if not marker.strip():
        raise ValueError("marker is empty")
    return marker.casefold() not in corpus_text.casefold()


def write_qa_lines(cases: Sequence[QACase], path: Path) -> None:
    """Write Q/A JSONL. Overwrites the file. An empty list still writes a file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_record(case), ensure_ascii=True) for case in cases]
    body = "\n".join(lines)
    if body:
        body += "\n"
    path.write_text(body, encoding="utf-8")


def _chunk_interpreter(chunk: ChunkRecord):
    def interpret(payload: dict[str, Any]) -> QACase:
        question = payload.get("question")
        reference = payload.get("reference_answer")
        evidence = payload.get("evidence")
        if not isinstance(question, str) or not question.strip():
            raise GenerationError("question text is empty")
        if not isinstance(reference, str) or not reference.strip():
            raise GenerationError("reference answer is empty")
        if not isinstance(evidence, str) or len(evidence.strip()) < _MIN_EVIDENCE:
            raise GenerationError("evidence must be a short span copied from the chunk")
        span = evidence.strip()
        if span not in chunk.text:
            raise GenerationError("evidence is not copied from the chunk")
        answer = reference.strip()
        if span not in answer:
            raise GenerationError("reference answer does not contain the evidence")
        return QACase(
            query=question.strip(),
            corpus_can_answer=True,
            relevant_chunk_ids=(chunk.chunk_id,),
            evidence=span,
            reference_answer=answer,
        )

    return interpret


def _unanswerable_interpreter(payload: dict[str, Any]) -> UnanswerableProposal:
    question = payload.get("question")
    marker = payload.get("marker")
    if not isinstance(question, str) or not question.strip():
        raise GenerationError("question text is empty")
    if not isinstance(marker, str) or len(marker.strip()) < 4:
        raise GenerationError("marker must be a short phrase")
    phrase = marker.strip()
    if phrase.casefold() not in question.casefold():
        raise GenerationError("marker must appear in the question")
    return UnanswerableProposal(question=question.strip(), marker=phrase)


def _record(case: QACase) -> dict[str, object]:
    return {
        "query": case.query,
        "corpus_can_answer": case.corpus_can_answer,
        "relevant_chunk_ids": list(case.relevant_chunk_ids),
        "evidence": case.evidence,
        "reference_answer": case.reference_answer,
    }
