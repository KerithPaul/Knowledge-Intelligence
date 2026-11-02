"""Retrieval metrics from Phase 3. Faithfulness and answer relevance from Phase 4.

Routing accuracy, false fallback, and missed fallback are from Phase 5.
Context precision, context recall, and answer correctness stay in Phase 6.
"""

from rag.evaluation.generation import (
    GenerationScore,
    answer_relevance,
    faithfulness,
    score_answer,
)
from rag.evaluation.golden import RetrievalCase, load_retrieval_cases
from rag.evaluation.retrieval import RetrievalReport, score_queries
from rag.evaluation.routing import RoutingCase, RoutingReport, load_routing_cases, score_routing

__all__ = [
    "GenerationScore",
    "RetrievalCase",
    "RetrievalReport",
    "RoutingCase",
    "RoutingReport",
    "answer_relevance",
    "faithfulness",
    "load_retrieval_cases",
    "load_routing_cases",
    "score_answer",
    "score_queries",
    "score_routing",
]
