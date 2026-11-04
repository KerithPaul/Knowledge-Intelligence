"""Join retrieval, routing, generation, and cost into one report.

The live command collects a route for every golden question, then judges
the answerable ones. This module turns those rows into the report and does
not open Qdrant or Groq itself.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag.core.interfaces import EmbeddingClient, LLMClient
from rag.core.trace import RequestTrace
from rag.evaluation.crosscheck import DIVERGENCES, Divergence
from rag.evaluation.generation import score_answer
from rag.evaluation.operational import (
    StageLatency,
    case_won,
    cost_usd,
    mean_stages,
    stage_times,
    tokens_from_trace,
)
from rag.evaluation.qa import QACase
from rag.evaluation.quality import score_quality
from rag.evaluation.retrieval import RetrievalReport, score_queries
from rag.evaluation.routing import (
    RoutingObservation,
    RoutingReport,
    score_routing,
)
from rag.generation.context import assemble_context
from rag.generation.types import (
    INSUFFICIENT_CONTEXT,
    ContextBlock,
    ContextSource,
    GroundedAnswer,
    InsufficientReason,
)
from rag.routing.types import DecidedBy, RoutedAnswer, RouteName, RoutingReason


@dataclass(frozen=True)
class CaseResult:
    """One golden question after retrieval, routing, and the judges."""

    query: str
    corpus_can_answer: bool
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    route: RouteName
    decided_by: DecidedBy
    reason: RoutingReason | None
    best_rerank_score: float | None
    sufficient: bool
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None
    answer_correctness: float | None
    prompt_tokens: int
    completion_tokens: int
    unpriced_calls: int
    elapsed_ms: float
    stages_ms: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class HarnessReport:
    """Macro scores, cost, and the per-question rows."""

    k: int
    threshold: float
    factual_weight: float
    win_correctness: float
    queries: int
    retrieval: RetrievalReport | None
    routing: RoutingReport
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None
    answer_correctness: float | None
    wins: int
    win_rate: float
    prompt_tokens: int
    completion_tokens: int
    unpriced_calls: int
    cost_usd: float
    mean_elapsed_ms: float
    stages: tuple[StageLatency, ...]
    per_query: tuple[CaseResult, ...]
    divergences: tuple[Divergence, ...]


def judging_context(
    *,
    answer_context: Sequence[ContextBlock] | None,
    chunks: Sequence[ContextSource],
    limit: int,
) -> tuple[ContextBlock, ...]:
    """The blocks a judge should see.

    When the model was called, that is the context it was shown. When it
    was not called, it is the retrieved chunks the prompt would have used.
    """

    if answer_context is not None:
        return tuple(answer_context)
    return assemble_context(chunks, limit=limit)


def context_from_route(routed: RoutedAnswer, *, limit: int) -> tuple[ContextBlock, ...]:
    """Blocks for the judges, taken from the route the gate returned."""

    if routed.answer is not None:
        return judging_context(answer_context=routed.answer.context, chunks=(), limit=limit)
    return judging_context(answer_context=None, chunks=routed.chunks, limit=limit)


def run_case(
    case: QACase,
    *,
    ask: Callable[[str, RequestTrace], RoutedAnswer],
    llm: LLMClient,
    embedder: EmbeddingClient,
    relevance_questions: int,
    max_context_chunks: int,
    max_tokens: int,
    factual_weight: float,
) -> CaseResult:
    """Ask one question, judge it when a reference exists, and pack the row.

    ``ask`` is ``router.ask`` without the extra keyword arguments. The trace
    it receives is the same trace the judges write on, so token totals include
    both the answer and the judges.
    """

    if relevance_questions <= 0 or max_context_chunks <= 0 or max_tokens <= 0:
        raise ValueError("judge limits must be positive")
    trace = RequestTrace()
    started = time.perf_counter()
    routed = ask(case.query, trace)
    if routed.query != case.query:
        raise ValueError("the route query must match the golden question")
    context = context_from_route(routed, limit=max_context_chunks)
    faithfulness = None
    answer_relevance = None
    context_precision = None
    context_recall = None
    answer_correctness = None
    if case.corpus_can_answer:
        answer = routed.answer if routed.answer is not None else _refusal_answer(routed, context)
        generation = score_answer(
            answer,
            llm=llm,
            embedder=embedder,
            relevance_questions=relevance_questions,
            max_tokens=max_tokens,
            trace=trace,
        )
        quality = score_quality(
            query=case.query,
            answer=answer.answer,
            reference=case.reference_answer,
            context=context,
            llm=llm,
            embedder=embedder,
            factual_weight=factual_weight,
            max_tokens=max_tokens,
            trace=trace,
        )
        faithfulness = generation.faithfulness.score
        answer_relevance = generation.answer_relevance.score
        context_precision = quality.context_precision.score
        context_recall = quality.context_recall.score
        answer_correctness = quality.answer_correctness.score
    tokens = tokens_from_trace(trace)
    return CaseResult(
        query=case.query,
        corpus_can_answer=case.corpus_can_answer,
        relevant_chunk_ids=case.relevant_chunk_ids,
        retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in routed.chunks),
        route=routed.route,
        decided_by=routed.decided_by,
        reason=routed.reason,
        best_rerank_score=routed.best_rerank_score,
        sufficient=routed.route == "knowledge_base",
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
        answer_correctness=answer_correctness,
        prompt_tokens=tokens.prompt_tokens,
        completion_tokens=tokens.completion_tokens,
        unpriced_calls=tokens.unpriced_calls,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        stages_ms=tuple(sorted(stage_times(trace).items())),
    )


def build_harness_report(
    rows: Sequence[CaseResult],
    *,
    k: int,
    threshold: float,
    factual_weight: float,
    win_correctness: float,
    prompt_usd_per_million: float,
    completion_usd_per_million: float,
) -> HarnessReport:
    """Average one finished run. ``rows`` must already be judged."""

    pending = tuple(rows)
    if not pending:
        raise ValueError("no questions")
    if k <= 0:
        raise ValueError("k must be positive")
    retrieval_rows = []
    routing_rows = []
    for row in pending:
        _check_row(row)
        if row.corpus_can_answer:
            retrieval_rows.append((row.query, row.retrieved_chunk_ids, set(row.relevant_chunk_ids)))
        routing_rows.append(
            RoutingObservation(
                query=row.query,
                corpus_can_answer=row.corpus_can_answer,
                route=row.route,
                decided_by=row.decided_by,
                reason=row.reason,
                best_rerank_score=row.best_rerank_score,
            )
        )
    retrieval = score_queries(retrieval_rows, k=k) if retrieval_rows else None
    routing = score_routing(routing_rows, threshold=threshold)
    wins = sum(
        1
        for row in pending
        if case_won(
            corpus_can_answer=row.corpus_can_answer,
            route=row.route,
            retrieved=row.retrieved_chunk_ids,
            relevant=row.relevant_chunk_ids,
            sufficient=row.sufficient,
            correctness=row.answer_correctness,
            correctness_floor=win_correctness,
        )
    )
    prompt_tokens = sum(row.prompt_tokens for row in pending)
    completion_tokens = sum(row.completion_tokens for row in pending)
    return HarnessReport(
        k=k,
        threshold=routing.threshold,
        factual_weight=factual_weight,
        win_correctness=win_correctness,
        queries=len(pending),
        retrieval=retrieval,
        routing=routing,
        faithfulness=_mean([row.faithfulness for row in pending if row.corpus_can_answer]),
        answer_relevance=_mean([row.answer_relevance for row in pending if row.corpus_can_answer]),
        context_precision=_mean(
            [row.context_precision for row in pending if row.corpus_can_answer]
        ),
        context_recall=_mean([row.context_recall for row in pending if row.corpus_can_answer]),
        answer_correctness=_mean(
            [row.answer_correctness for row in pending if row.corpus_can_answer]
        ),
        wins=wins,
        win_rate=wins / len(pending),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        unpriced_calls=sum(row.unpriced_calls for row in pending),
        cost_usd=cost_usd(
            prompt_tokens,
            completion_tokens,
            prompt_usd_per_million=prompt_usd_per_million,
            completion_usd_per_million=completion_usd_per_million,
        ),
        mean_elapsed_ms=sum(row.elapsed_ms for row in pending) / len(pending),
        stages=mean_stages([dict(row.stages_ms) for row in pending]),
        per_query=pending,
        divergences=DIVERGENCES,
    )


def gate_metrics(report: HarnessReport) -> dict[str, float | None]:
    """The flat numbers the regression gate reads."""

    retrieval = report.retrieval
    return {
        "retrieval.hit_at_k": None if retrieval is None else retrieval.hit_at_k,
        "retrieval.recall_at_k": None if retrieval is None else retrieval.recall_at_k,
        "retrieval.precision_at_k": None if retrieval is None else retrieval.precision_at_k,
        "retrieval.mrr": None if retrieval is None else retrieval.mrr,
        "retrieval.ndcg_at_k": None if retrieval is None else retrieval.ndcg_at_k,
        "retrieval.map": None if retrieval is None else retrieval.mean_average_precision,
        "routing.accuracy": report.routing.accuracy,
        "routing.false_fallback_rate": report.routing.false_fallback_rate,
        "routing.missed_fallback_rate": report.routing.missed_fallback_rate,
        "faithfulness": report.faithfulness,
        "answer_relevance": report.answer_relevance,
        "context_precision": report.context_precision,
        "context_recall": report.context_recall,
        "answer_correctness": report.answer_correctness,
        "win_rate": report.win_rate,
    }


def harness_report_record(report: HarnessReport) -> dict[str, object]:
    """JSON object for harness_report.json."""

    retrieval = report.retrieval
    return {
        "k": report.k,
        "threshold": report.threshold,
        "factual_weight": report.factual_weight,
        "win_correctness": report.win_correctness,
        "queries": report.queries,
        "wins": report.wins,
        "win_rate": report.win_rate,
        "prompt_tokens": report.prompt_tokens,
        "completion_tokens": report.completion_tokens,
        "unpriced_calls": report.unpriced_calls,
        "cost_usd": report.cost_usd,
        "mean_elapsed_ms": report.mean_elapsed_ms,
        "metrics": gate_metrics(report),
        "retrieval": None
        if retrieval is None
        else {
            "queries": retrieval.queries,
            "precision_at_k": retrieval.precision_at_k,
            "recall_at_k": retrieval.recall_at_k,
            "hit_at_k": retrieval.hit_at_k,
            "mrr": retrieval.mrr,
            "ndcg_at_k": retrieval.ndcg_at_k,
            "map": retrieval.mean_average_precision,
        },
        "routing": {
            "queries": report.routing.queries,
            "answerable": report.routing.answerable,
            "unanswerable": report.routing.unanswerable,
            "accuracy": report.routing.accuracy,
            "false_fallback_rate": report.routing.false_fallback_rate,
            "missed_fallback_rate": report.routing.missed_fallback_rate,
            "false_fallbacks": report.routing.false_fallbacks,
            "missed_fallbacks": report.routing.missed_fallbacks,
        },
        "stages": [
            {
                "name": stage.name,
                "questions": stage.questions,
                "total_ms": stage.total_ms,
                "mean_ms": stage.mean_ms,
            }
            for stage in report.stages
        ],
        "divergences": [
            {"metric": item.metric, "library": item.library, "detail": item.detail}
            for item in report.divergences
        ],
        "per_query": [_case_record(row) for row in report.per_query],
    }


def write_harness_report(report: HarnessReport, path: Path) -> None:
    """Write harness_report.json. Overwrites the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(harness_report_record(report), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")


def load_report_metrics(path: Path) -> dict[str, float | None]:
    """Read the ``metrics`` object from a saved harness report."""

    if not path.is_file():
        raise FileNotFoundError(f"harness report does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("metrics"), dict):
        raise ValueError(f"{path.name} is missing metrics")
    metrics: dict[str, float | None] = {}
    for key, value in raw["metrics"].items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path.name} has an empty metric name")
        if value is None:
            metrics[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{path.name} metric {key} must be a number or null")
        metrics[key] = float(value)
    return metrics


def answerable_cases(cases: Sequence[QACase]) -> list[QACase]:
    """Questions the corpus is supposed to answer."""

    return [case for case in cases if case.corpus_can_answer]


def _case_record(row: CaseResult) -> dict[str, object]:
    return {
        "query": row.query,
        "corpus_can_answer": row.corpus_can_answer,
        "relevant_chunk_ids": list(row.relevant_chunk_ids),
        "retrieved_chunk_ids": list(row.retrieved_chunk_ids),
        "route": row.route,
        "decided_by": row.decided_by,
        "reason": row.reason,
        "best_rerank_score": row.best_rerank_score,
        "sufficient": row.sufficient,
        "faithfulness": row.faithfulness,
        "answer_relevance": row.answer_relevance,
        "context_precision": row.context_precision,
        "context_recall": row.context_recall,
        "answer_correctness": row.answer_correctness,
        "prompt_tokens": row.prompt_tokens,
        "completion_tokens": row.completion_tokens,
        "unpriced_calls": row.unpriced_calls,
        "elapsed_ms": row.elapsed_ms,
        "stages_ms": {name: value for name, value in row.stages_ms},
    }


def _mean(values: Sequence[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _check_row(row: CaseResult) -> None:
    if not isinstance(row, CaseResult):
        raise ValueError("each row must be a CaseResult")
    if not row.query.strip():
        raise ValueError("query text is empty")
    if isinstance(row.elapsed_ms, bool) or not isinstance(row.elapsed_ms, int | float):
        raise ValueError("elapsed time must be a number")
    if row.elapsed_ms < 0:
        raise ValueError("elapsed time must be zero or more")
    for name, value in row.stages_ms:
        if not isinstance(name, str) or not name:
            raise ValueError("stage times must be named")
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ValueError("stage times must be zero or more")


def _refusal_answer(
    routed: RoutedAnswer,
    context: tuple[ContextBlock, ...],
) -> GroundedAnswer:
    """A grounded answer for a route that never called the model.

    The judges read the sentence and the retrieved blocks. The score gate's
    own reason is not one of the generator reasons, so this object records
    empty context instead.
    """

    reason: InsufficientReason = "empty_context"
    if routed.reason in {"model", "uncited", "empty_context"}:
        reason = routed.reason
    return GroundedAnswer(
        query=routed.query,
        answer=routed.text if routed.text.strip() else INSUFFICIENT_CONTEXT,
        sufficient=False,
        insufficient_reason=reason,
        citations=(),
        context=context,
        model="",
        prompt_tokens=0,
        completion_tokens=0,
        latency_ms=0.0,
    )
