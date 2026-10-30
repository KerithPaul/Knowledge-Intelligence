"""The reranker returns one finite score per passage and loads lazily."""

from pathlib import Path

import pytest

from rag.core.clients import build_reranker
from rag.core.config import load_settings
from rag.core.errors import RetrievalError
from rag.core.trace import RequestTrace
from rag.retrieval.rerank import BGEReranker


class _Scorer:
    def __init__(self, raw) -> None:
        self.raw = raw
        self.pairs = None

    def predict(self, inputs, *, batch_size, show_progress_bar):
        self.pairs = list(inputs)
        self.batch_size = batch_size
        assert show_progress_bar is False
        return self.raw


def test_scores_follow_the_passage_order_and_record_a_span() -> None:
    scorer = _Scorer([0.2, 0.8])
    reranker = BGEReranker(
        model_name="BAAI/bge-reranker-base",
        device="cpu",
        batch_size=4,
        max_length=512,
        scorer=scorer,
    )
    trace = RequestTrace()
    scores = reranker.score("how does forking work", ["one", "two"], trace=trace)
    assert scores == [0.2, 0.8]
    assert scorer.pairs == [("how does forking work", "one"), ("how does forking work", "two")]
    assert scorer.batch_size == 4
    assert trace.spans[0].name == "retrieval.rerank"
    assert trace.spans[0].attributes["count"] == 2
    assert trace.spans[0].attributes["model"] == "BAAI/bge-reranker-base"


def test_nested_and_scalar_scores_are_flattened() -> None:
    nested = BGEReranker(
        model_name="fake",
        device="cpu",
        batch_size=2,
        max_length=32,
        scorer=_Scorer([[0.25], [0.75]]),
    )
    assert nested.score("q", ["a", "b"]) == [0.25, 0.75]
    # The class stores batch_size. This scorer ignores it.
    single = BGEReranker(
        model_name="fake",
        device="cpu",
        batch_size=2,
        max_length=32,
        scorer=_Scorer(0.5),
    )
    assert single.score("q", ["a"]) == [0.5]


def test_empty_input_does_not_score() -> None:
    scorer = _Scorer([0.1])
    reranker = BGEReranker(
        model_name="fake",
        device="cpu",
        batch_size=2,
        max_length=32,
        scorer=scorer,
    )
    assert reranker.score("q", []) == []
    assert scorer.pairs is None
    with pytest.raises(ValueError):
        reranker.score("  ", ["a"])
    with pytest.raises(ValueError):
        reranker.score("q", [" "])


def test_bad_score_shapes_are_errors() -> None:
    reranker = BGEReranker(
        model_name="fake",
        device="cpu",
        batch_size=2,
        max_length=32,
        scorer=_Scorer([0.2]),
    )
    with pytest.raises(RetrievalError, match="1 score"):
        reranker.score("q", ["a", "b"])
    reranker = BGEReranker(
        model_name="fake",
        device="cpu",
        batch_size=2,
        max_length=32,
        scorer=_Scorer([float("nan")]),
    )
    with pytest.raises(RetrievalError, match="non-finite"):
        reranker.score("q", ["a"])


def test_factory_does_not_load_the_model(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    settings = load_settings(env_file=env_file)
    reranker = build_reranker(settings)
    assert isinstance(reranker, BGEReranker)
    assert reranker.model_name == "BAAI/bge-reranker-base"
    assert reranker._scorer is None
    assert reranker._max_length == 512
