"""Context precision is average precision. Correctness mixes F1 and cosine."""

import json

import pytest

from rag.core.interfaces import EmbeddingClient, LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace
from rag.evaluation.quality import (
    answer_correctness,
    context_precision,
    context_recall,
    factual_f1,
    score_quality,
)
from rag.generation.types import ContextBlock


def test_context_precision_rewards_a_useful_block_at_rank_one() -> None:
    assert context_precision(()) == 0.0
    assert context_precision((False, False)) == 0.0
    assert context_precision((True, False)) == 1.0
    assert context_precision((False, True)) == 0.5
    assert context_precision((True, False, True)) == pytest.approx(5 / 6)
    with pytest.raises(ValueError, match="booleans"):
        context_precision((True, 1))


def test_context_recall_and_factual_f1() -> None:
    assert context_recall((True, False)) == 0.5
    with pytest.raises(ValueError, match="no reference claims"):
        context_recall(())
    assert factual_f1(1, 1, 1) == 0.5
    with pytest.raises(ValueError, match="no statements"):
        factual_f1(0, 0, 0)
    with pytest.raises(ValueError, match="fact counts"):
        factual_f1(True, 0, 0)


def test_answer_correctness_uses_the_factual_weight_and_can_be_negative() -> None:
    assert answer_correctness(0.5, 1.0, factual_weight=0.75) == pytest.approx(0.625)
    assert answer_correctness(0.0, -1.0, factual_weight=0.75) == pytest.approx(-0.25)
    assert answer_correctness(1.0, 0.0, factual_weight=1) == 1.0
    with pytest.raises(ValueError, match="factual_weight"):
        answer_correctness(1.0, 1.0, factual_weight=1.5)
    with pytest.raises(ValueError, match="between -1 and 1"):
        answer_correctness(1.0, 1.1)


def test_score_quality_judges_the_reference_and_embeds_passages() -> None:
    precision = json.dumps(
        {"verdicts": [{"number": 1, "useful": True}, {"number": 2, "useful": False}]}
    )
    claims = json.dumps({"claims": ["The fork copies the session."]})
    verdicts = json.dumps(
        {"verdicts": [{"claim": "The fork copies the session.", "supported": True}]}
    )
    facts = json.dumps(
        {
            "true_positives": ["The fork copies the session."],
            "false_positives": ["Cancel stops it."],
            "false_negatives": [],
        }
    )
    llm = _ScriptedLLM([precision, claims, verdicts, facts])
    embedder = _Embedder([[1.0, 0.0], [1.0, 0.0]])
    trace = RequestTrace()
    blocks = (
        _block(1, "The fork copies the session."),
        _block(2, "Unrelated note."),
    )
    score = score_quality(
        query="How does forking work?",
        answer="The fork copies the session. Cancel stops it.",
        reference="The fork copies the session.",
        context=blocks,
        llm=llm,
        embedder=embedder,
        factual_weight=0.75,
        max_tokens=64,
        trace=trace,
    )
    assert score.context_precision.score == 1.0
    assert score.context_precision.useful == (True, False)
    assert score.context_recall.score == 1.0
    assert score.answer_correctness.factual_score == pytest.approx(1 / 1.5)
    assert score.answer_correctness.semantic_similarity == 1.0
    assert score.answer_correctness.score == pytest.approx(0.75 * (1 / 1.5) + 0.25)
    precision_user = llm.calls[0][0][1].content
    assert "The fork copies the session." in precision_user
    assert "Unrelated note." in precision_user
    assert embedder.queries == []
    assert embedder.passages == [
        "The fork copies the session. Cancel stops it.",
        "The fork copies the session.",
    ]
    names = [span.name for span in trace.spans]
    assert "evaluation.context_precision" in names
    assert "evaluation.context_recall" in names
    assert "evaluation.answer_correctness" in names


def test_empty_context_skips_the_precision_judge_and_scores_zero() -> None:
    claims = json.dumps({"claims": []})
    facts = json.dumps({"true_positives": [], "false_positives": [], "false_negatives": []})
    llm = _ScriptedLLM([claims, facts])
    embedder = _Embedder([[1.0, 0.0], [0.0, 1.0]])
    score = score_quality(
        query="How does forking work?",
        answer="The fork copies the session.",
        reference="The fork copies the session.",
        context=(),
        llm=llm,
        embedder=embedder,
        max_tokens=32,
    )
    assert score.context_precision.score == 0.0
    assert score.context_precision.useful == ()
    assert score.context_recall.score is None
    assert score.answer_correctness.score is None
    assert score.answer_correctness.semantic_similarity == 0.0
    assert len(llm.calls) == 2


class _ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[LLMMessage], int | None]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        self.calls.append((list(messages), max_tokens))
        return LLMResponse(
            content=self._replies.pop(0),
            model="scripted",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1.0,
        )


class _Embedder(EmbeddingClient):
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = list(vectors)
        self.queries: list[str] = []
        self.passages: list[str] = []

    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "scripted"

    def embed_query(self, text: str, *, trace=None) -> list[float]:
        self.queries.append(text)
        raise AssertionError("answer correctness embeds both texts as passages")

    def embed_passages(self, texts, *, trace=None) -> list[list[float]]:
        self.passages.extend(texts)
        if len(self._vectors) < len(texts):
            raise AssertionError("not enough vectors")
        chosen = self._vectors[: len(texts)]
        self._vectors = self._vectors[len(texts) :]
        return chosen


def _block(number: int, text: str) -> ContextBlock:
    return ContextBlock(
        number=number,
        chunk_id=f"chunk-{number}",
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text=text,
    )
