"""The hand-checked retrieval questions for Phase 3.

Each row names the chunks that contain `evidence`. The query is the
question a person would ask. The Phase 6 file ``qa_golden.jsonl`` repeats
these rows and adds a reference answer. Reports written next to this file
are generated and stay out of git.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag.core.config import PROJECT_ROOT

DEFAULT_GOLDEN = PROJECT_ROOT / "data" / "evaluation" / "retrieval_golden.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "evaluation" / "retrieval_report.json"

_FIELDS = frozenset({"query", "relevant_chunk_ids", "evidence"})


@dataclass(frozen=True)
class RetrievalCase:
    """One labeled question. `evidence` is a string found in every relevant chunk."""

    query: str
    relevant_chunk_ids: tuple[str, ...]
    evidence: str


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """Read the golden JSONL. A bad line names its number."""

    if not path.is_file():
        raise FileNotFoundError(f"retrieval golden file does not exist: {path}")
    cases: list[RetrievalCase] = []
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


def _parse_case(raw: dict[str, object], *, line_number: int, file_name: str) -> RetrievalCase:
    where = f"{file_name} line {line_number}"
    if set(raw) != _FIELDS:
        missing = sorted(_FIELDS - set(raw))
        extra = sorted(set(raw) - _FIELDS)
        raise ValueError(f"{where} fields are wrong (missing {missing}, extra {extra})")
    query = raw["query"]
    relevant = raw["relevant_chunk_ids"]
    evidence = raw["evidence"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{where} has an empty query")
    if not isinstance(relevant, list) or not relevant:
        raise ValueError(f"{where} needs at least one relevant chunk id")
    if not all(isinstance(chunk_id, str) and chunk_id for chunk_id in relevant):
        raise ValueError(f"{where} relevant chunk ids must be non-empty strings")
    if len(set(relevant)) != len(relevant):
        raise ValueError(f"{where} repeats a chunk id")
    if not isinstance(evidence, str) or not evidence:
        raise ValueError(f"{where} has empty evidence")
    return RetrievalCase(
        query=query.strip(),
        relevant_chunk_ids=tuple(relevant),
        evidence=evidence,
    )
