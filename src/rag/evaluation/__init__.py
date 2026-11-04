"""Retrieval, routing, and generation metrics, plus the Phase 6 harness.

Faithfulness and answer relevance are the Phase 4 scores. Context precision,
context recall, and answer correctness are the Phase 6 scores. The harness
joins those with retrieval, routing, latency, token cost, and a win rate.
"""

from rag.evaluation.generation import (
    GenerationScore,
    answer_relevance,
    faithfulness,
    score_answer,
)
from rag.evaluation.golden import RetrievalCase, load_retrieval_cases
from rag.evaluation.quality import (
    QualityScore,
    answer_correctness,
    context_precision,
    context_recall,
    factual_f1,
    score_quality,
)
from rag.evaluation.retrieval import RetrievalReport, score_queries
from rag.evaluation.routing import RoutingCase, RoutingReport, load_routing_cases, score_routing

__all__ = [
    "GenerationScore",
    "QualityScore",
    "RetrievalCase",
    "RetrievalReport",
    "RoutingCase",
    "RoutingReport",
    "answer_correctness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "factual_f1",
    "faithfulness",
    "load_retrieval_cases",
    "load_routing_cases",
    "score_answer",
    "score_quality",
    "score_queries",
    "score_routing",
]
