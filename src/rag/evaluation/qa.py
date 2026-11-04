"""The Phase 6 question file.

Each row is one question. Answerable rows name the chunks that contain
``evidence`` and give a reference answer that contains that same evidence.
Unanswerable rows have an empty chunk list, empty evidence, and an empty
reference. Routing labels and retrieval labels are repeated here so the
harness can score them together.

``retrieval_golden.jsonl`` and ``routing_golden.jsonl`` stay the inputs for
their own commands. This file must carry the same questions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag.core.config import PROJECT_ROOT

DEFAULT_QA = PROJECT_ROOT / "data" / "evaluation" / "qa_golden.jsonl"

_FIELDS = frozenset(
    {
        "query",
        "corpus_can_answer",
        "relevant_chunk_ids",
        "evidence",
        "reference_answer",
    }
)


@dataclass(frozen=True)
class QACase:
    """One golden question. The reference is empty when the corpus cannot answer."""

    query: str
    corpus_can_answer: bool
    relevant_chunk_ids: tuple[str, ...]
    evidence: str
    reference_answer: str


def load_qa_cases(path: Path) -> list[QACase]:
    """Read the Q/A JSONL. A bad line names its number."""

    if not path.is_file():
        raise FileNotFoundError(f"qa golden file does not exist: {path}")
    cases: list[QACase] = []
    seen_queries: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_number} is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} line {line_number} must be a JSON object")
        case = _parse_case(raw, line_number=line_number, file_name=path.name)
        if case.query in seen_queries:
            raise ValueError(f"duplicate query on line {line_number}")
        seen_queries.add(case.query)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path.name} has no questions")
    return cases


def _parse_case(raw: dict[str, object], *, line_number: int, file_name: str) -> QACase:
    where = f"{file_name} line {line_number}"
    if set(raw) != _FIELDS:
        missing = sorted(_FIELDS - set(raw))
        extra = sorted(set(raw) - _FIELDS)
        raise ValueError(f"{where} fields are wrong (missing {missing}, extra {extra})")
    query = raw["query"]
    label = raw["corpus_can_answer"]
    relevant = raw["relevant_chunk_ids"]
    evidence = raw["evidence"]
    reference = raw["reference_answer"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{where} has an empty query")
    if not isinstance(label, bool):
        raise ValueError(f"{where} corpus_can_answer must be true or false")
    if not isinstance(relevant, list):
        raise ValueError(f"{where} relevant chunk ids must be a list")
    if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in relevant):
        raise ValueError(f"{where} relevant chunk ids must be non-empty strings")
    if len(set(relevant)) != len(relevant):
        raise ValueError(f"{where} repeats a chunk id")
    if not isinstance(evidence, str) or not isinstance(reference, str):
        raise ValueError(f"{where} evidence and reference_answer must be strings")
    if label:
        if not relevant:
            raise ValueError(f"{where} needs at least one relevant chunk id")
        if not evidence:
            raise ValueError(f"{where} has empty evidence")
        if not reference.strip():
            raise ValueError(f"{where} has an empty reference answer")
        if evidence not in reference:
            raise ValueError(f"{where} reference answer does not contain the evidence")
    elif relevant or evidence or reference:
        raise ValueError(f"{where} unanswerable rows keep chunks, evidence, and reference empty")
    return QACase(
        query=query.strip(),
        corpus_can_answer=label,
        relevant_chunk_ids=tuple(relevant),
        evidence=evidence,
        reference_answer=reference.strip(),
    )
