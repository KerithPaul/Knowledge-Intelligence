"""Retrieval metrics from IMPLEMENTATION_PLAN.md section 5.

Precision@k divides by k, including when the list is shorter than k.
Recall, hit, nDCG, and average precision treat a chunk id as relevant or
not. nDCG uses the binary gain ``2^rel - 1``, which is 1 for a hit.
Average precision walks the whole ranking it is given, so a relevant id
past the cutoff lowers the score. Mean average precision is the mean of
that value across queries.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryScore:
    """Metrics for one question at one k."""

    query: str
    retrieved: tuple[str, ...]
    relevant: tuple[str, ...]
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    average_precision: float


@dataclass(frozen=True)
class RetrievalReport:
    """Macro averages over a set of questions."""

    k: int
    queries: int
    precision_at_k: float
    recall_at_k: float
    hit_at_k: float
    mrr: float
    ndcg_at_k: float
    mean_average_precision: float
    per_query: tuple[QueryScore, ...]


def precision_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranking, relevant, k)
    return sum(1 for item in ranking[:k] if item in relevant) / k


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranking, relevant, k)
    return sum(1 for item in ranking[:k] if item in relevant) / len(relevant)


def hit_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranking, relevant, k)
    return 1.0 if any(item in relevant for item in ranking[:k]) else 0.0


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    _check(ranking, relevant, k=1)
    for index, item in enumerate(ranking, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    _check(ranking, relevant, k)
    dcg = 0.0
    for index, item in enumerate(ranking[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def average_precision(ranking: Sequence[str], relevant: set[str]) -> float:
    """Average precision of this ranking. Unretrieved relevant ids count against it."""

    _check(ranking, relevant, k=1)
    hits = 0
    total = 0.0
    for index, item in enumerate(ranking, start=1):
        if item in relevant:
            hits += 1
            total += hits / index
    return total / len(relevant)


def score_queries(
    rows: Sequence[tuple[str, Sequence[str], set[str]]],
    *,
    k: int,
) -> RetrievalReport:
    """Score ``(query, ranking, relevant ids)`` rows and average them."""

    if not rows:
        raise ValueError("no queries")
    if k <= 0:
        raise ValueError("k must be positive")
    per_query: list[QueryScore] = []
    for query, ranking, relevant in rows:
        if not query.strip():
            raise ValueError("query text is empty")
        per_query.append(
            QueryScore(
                query=query,
                retrieved=tuple(ranking),
                relevant=tuple(sorted(relevant)),
                precision_at_k=precision_at_k(ranking, relevant, k),
                recall_at_k=recall_at_k(ranking, relevant, k),
                hit_at_k=hit_at_k(ranking, relevant, k),
                reciprocal_rank=reciprocal_rank(ranking, relevant),
                ndcg_at_k=ndcg_at_k(ranking, relevant, k),
                average_precision=average_precision(ranking, relevant),
            )
        )
    return RetrievalReport(
        k=k,
        queries=len(per_query),
        precision_at_k=_mean([row.precision_at_k for row in per_query]),
        recall_at_k=_mean([row.recall_at_k for row in per_query]),
        hit_at_k=_mean([row.hit_at_k for row in per_query]),
        mrr=_mean([row.reciprocal_rank for row in per_query]),
        ndcg_at_k=_mean([row.ndcg_at_k for row in per_query]),
        mean_average_precision=_mean([row.average_precision for row in per_query]),
        per_query=tuple(per_query),
    )


def retrieval_report_record(report: RetrievalReport) -> dict[str, object]:
    """JSON object for retrieval_report.json."""

    return {
        "k": report.k,
        "queries": report.queries,
        "precision_at_k": report.precision_at_k,
        "recall_at_k": report.recall_at_k,
        "hit_at_k": report.hit_at_k,
        "mrr": report.mrr,
        "ndcg_at_k": report.ndcg_at_k,
        "map": report.mean_average_precision,
        "per_query": [
            {
                "query": row.query,
                "retrieved": list(row.retrieved),
                "relevant": list(row.relevant),
                "precision_at_k": row.precision_at_k,
                "recall_at_k": row.recall_at_k,
                "hit_at_k": row.hit_at_k,
                "reciprocal_rank": row.reciprocal_rank,
                "ndcg_at_k": row.ndcg_at_k,
                "average_precision": row.average_precision,
            }
            for row in report.per_query
        ],
    }


def write_retrieval_report(report: RetrievalReport, path: Path) -> None:
    """Write retrieval_report.json. Overwrites the file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(retrieval_report_record(report), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _check(ranking: Sequence[str], relevant: set[str], k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant set is empty")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains a duplicate id")
