"""The RAGAS epsilon is the only numeric gap in context precision."""

from rag.evaluation.crosscheck import (
    DIVERGENCES,
    library_status,
    precision_gap,
    ragas_context_precision,
)
from rag.evaluation.quality import context_precision


def test_one_useful_block_is_exact_here_and_just_under_one_in_ragas() -> None:
    assert context_precision((True,)) == 1.0
    assert ragas_context_precision((True,)) == 1.0 / (1.0 + 1e-10)
    assert precision_gap((True,), 1.0) > 0
    assert context_precision((False, True)) == 0.5
    assert ragas_context_precision((False, True)) == 0.5 / (1.0 + 1e-10)
    assert ragas_context_precision((False, False)) == 0.0


def test_the_notes_name_the_input_differences() -> None:
    metrics = {item.metric for item in DIVERGENCES}
    assert "answer_relevance" in metrics
    assert "answer_correctness" in metrics
    status = library_status()
    assert set(status) == {"ragas", "deepeval"}
    assert status["ragas"]["installed"] is False
    assert status["deepeval"]["installed"] is False
