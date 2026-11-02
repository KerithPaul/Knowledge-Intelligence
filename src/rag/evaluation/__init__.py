"""Retrieval metrics from Phase 3. Faithfulness and answer relevance from Phase 4.

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

__all__ = [
    "GenerationScore",
    "RetrievalCase",
    "RetrievalReport",
    "answer_relevance",
    "faithfulness",
    "load_retrieval_cases",
    "score_answer",
    "score_queries",
]
