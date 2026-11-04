"""The routing labels match the retrieval questions, plus out-of-corpus ones."""

import pytest

from rag.core.config import PROJECT_ROOT
from rag.evaluation.golden import load_retrieval_cases
from rag.evaluation.routing import DEFAULT_ROUTING_GOLDEN, load_routing_cases

# Each unanswerable question is tied to a phrase that must stay out of data/raw.
# If the corpus grows to include one of these, change the label before trusting --eval.
_ABSENT = {
    "What is the boiling point of water at sea level in degrees Celsius?": "boiling point",
    "Which planet is the largest in Earth's solar system?": "solar system",
    "How many players are on the field for one soccer team during a standard match?": "soccer",
    "What year was the Eiffel Tower completed?": "eiffel",
    "What role do mitochondria play in a living cell?": "mitochondria",
    "How does photosynthesis store energy from sunlight?": "photosynthesis",
    "Which plays did Shakespeare write besides the tragedies?": "shakespeare",
    "What does the Pythagorean theorem say about the sides of a right triangle?": "pythagorean",
}


def test_answerable_questions_are_the_retrieval_set() -> None:
    routing = load_routing_cases(DEFAULT_ROUTING_GOLDEN)
    retrieval_path = PROJECT_ROOT / "data" / "evaluation" / "retrieval_golden.jsonl"
    retrieval = {case.query for case in load_retrieval_cases(retrieval_path)}
    positives = {case.query for case in routing if case.corpus_can_answer}
    negatives = {case.query for case in routing if not case.corpus_can_answer}
    assert positives == retrieval
    assert negatives == set(_ABSENT)


def test_unanswerable_markers_are_absent_from_the_corpus() -> None:
    blob = "\n".join(
        path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / "data" / "raw").rglob("*.md")
    ).lower()
    for marker in _ABSENT.values():
        assert marker not in blob


def test_a_bad_routing_line_names_itself(tmp_path) -> None:
    path = tmp_path / "routing_golden.jsonl"
    path.write_text('{"query": "only"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_routing_cases(path)
    path.write_text('{"query": "q", "corpus_can_answer": "yes"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="true or false"):
        load_routing_cases(path)
    path.write_text(
        '{"query": "same", "corpus_can_answer": true}\n'
        '{"query": "same", "corpus_can_answer": false}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_routing_cases(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no questions"):
        load_routing_cases(path)
