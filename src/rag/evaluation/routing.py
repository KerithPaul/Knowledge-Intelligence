"""Routing metrics from IMPLEMENTATION_PLAN.md section 5.

Each question is labeled: the indexed docs can answer it, or they cannot.
The gate chooses ``knowledge_base`` or ``insufficient``. There is no web
call in this build. "Fallback" means the insufficient route, which is
the decision that would have left the knowledge base.

Accuracy is the share of questions whose route matches the label.
False fallback is the share of answerable questions the gate refused.
Missed fallback is the share of unanswerable questions the gate still
answered from the knowledge base. A rate is missing when that class
has no questions, because the ratio would divide by zero.

The labels live in ``routing_golden.jsonl``. The answerable questions are
the retrieval questions. The unanswerable questions were checked against
``data/raw`` and are outside this corpus. ``qa_golden.jsonl`` repeats both
classes and adds a reference answer for the answerable ones.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.core.config import PROJECT_ROOT
from rag.routing.types import DecidedBy, RouteName, RoutingReason

DEFAULT_ROUTING_GOLDEN = PROJECT_ROOT / "data" / "evaluation" / "routing_golden.jsonl"
DEFAULT_ROUTING_REPORT = PROJECT_ROOT / "data" / "evaluation" / "routing_report.json"

_FIELDS = frozenset({"query", "corpus_can_answer"})
_ROUTES = frozenset({"knowledge_base", "insufficient"})


@dataclass(frozen=True)
class RoutingCase:
    """One labeled question. ``corpus_can_answer`` is the hand label."""

    query: str
    corpus_can_answer: bool


@dataclass(frozen=True)
class RoutingObservation:
    """What the gate did with one labeled question."""

    query: str
    corpus_can_answer: bool
    route: RouteName
    decided_by: DecidedBy
    reason: RoutingReason | None
    best_rerank_score: float | None


@dataclass(frozen=True)
class RoutingQueryScore:
    """One question after it has been compared with its label."""

    query: str
    corpus_can_answer: bool
    route: RouteName
    decided_by: DecidedBy
    reason: RoutingReason | None
    best_rerank_score: float | None
    correct: bool


@dataclass(frozen=True)
class RoutingReport:
    """Rates over a set of labeled questions."""

    threshold: float
    queries: int
    answerable: int
    unanswerable: int
    accuracy: float
    false_fallback_rate: float | None
    missed_fallback_rate: float | None
    false_fallbacks: int
    missed_fallbacks: int
    per_query: tuple[RoutingQueryScore, ...]


def load_routing_cases(path: Path) -> list[RoutingCase]:
    """Read the routing JSONL. A bad line names its number."""

    if not path.is_file():
        raise FileNotFoundError(f"routing golden file does not exist: {path}")
    cases: list[RoutingCase] = []
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


def score_routing(
    rows: Sequence[RoutingObservation],
    *,
    threshold: float,
) -> RoutingReport:
    """Score gate decisions against hand labels.

    ``threshold`` is recorded on the report. It is the knob the gate used.
    It does not change the rates. The rates come from the routes.
    """

    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ValueError("threshold must be between 0 and 1")
    knob = float(threshold)
    if not math.isfinite(knob) or not 0.0 <= knob <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    pending = tuple(rows)
    if not pending:
        raise ValueError("no queries")
    per_query: list[RoutingQueryScore] = []
    false_fallbacks = 0
    missed_fallbacks = 0
    answerable = 0
    for row in pending:
        _check_row(row)
        correct = (row.route == "knowledge_base") == row.corpus_can_answer
        if row.corpus_can_answer:
            answerable += 1
            if row.route != "knowledge_base":
                false_fallbacks += 1
        elif row.route == "knowledge_base":
            missed_fallbacks += 1
        per_query.append(
            RoutingQueryScore(
                query=row.query,
                corpus_can_answer=row.corpus_can_answer,
                route=row.route,
                decided_by=row.decided_by,
                reason=row.reason,
                best_rerank_score=row.best_rerank_score,
                correct=correct,
            )
        )
    unanswerable = len(per_query) - answerable
    correct_count = sum(1 for row in per_query if row.correct)
    return RoutingReport(
        threshold=knob,
        queries=len(per_query),
        answerable=answerable,
        unanswerable=unanswerable,
        accuracy=correct_count / len(per_query),
        false_fallback_rate=None if answerable == 0 else false_fallbacks / answerable,
        missed_fallback_rate=None if unanswerable == 0 else missed_fallbacks / unanswerable,
        false_fallbacks=false_fallbacks,
        missed_fallbacks=missed_fallbacks,
        per_query=tuple(per_query),
    )


def routing_report_record(report: RoutingReport) -> dict[str, object]:
    """JSON object for routing_report.json."""

    return {
        "threshold": report.threshold,
        "queries": report.queries,
        "answerable": report.answerable,
        "unanswerable": report.unanswerable,
        "accuracy": report.accuracy,
        "false_fallback_rate": report.false_fallback_rate,
        "missed_fallback_rate": report.missed_fallback_rate,
        "false_fallbacks": report.false_fallbacks,
        "missed_fallbacks": report.missed_fallbacks,
        "per_query": [
            {
                "query": row.query,
                "corpus_can_answer": row.corpus_can_answer,
                "route": row.route,
                "decided_by": row.decided_by,
                "reason": row.reason,
                "best_rerank_score": row.best_rerank_score,
                "correct": row.correct,
            }
            for row in report.per_query
        ],
    }


def write_routing_report(report: RoutingReport, path: Path) -> None:
    """Write routing_report.json. Overwrites the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(routing_report_record(report), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")


def _parse_case(raw: dict[str, object], *, line_number: int, file_name: str) -> RoutingCase:
    where = f"{file_name} line {line_number}"
    if set(raw) != _FIELDS:
        missing = sorted(_FIELDS - set(raw))
        extra = sorted(set(raw) - _FIELDS)
        raise ValueError(f"{where} fields are wrong (missing {missing}, extra {extra})")
    query = raw["query"]
    label = raw["corpus_can_answer"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{where} has an empty query")
    if not isinstance(label, bool):
        raise ValueError(f"{where} corpus_can_answer must be true or false")
    return RoutingCase(query=query.strip(), corpus_can_answer=label)


def _check_row(row: RoutingObservation) -> None:
    if not isinstance(row, RoutingObservation):
        raise ValueError("each row must be a RoutingObservation")
    if not row.query.strip():
        raise ValueError("query text is empty")
    if row.route not in _ROUTES:
        raise ValueError("route must be knowledge_base or insufficient")
