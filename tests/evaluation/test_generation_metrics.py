"""Faithfulness is supported claims over claims. Relevance is a mean cosine."""

import json
import math

import pytest

from rag.core.interfaces import EmbeddingClient, LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace
from rag.evaluation.generation import (
    answer_relevance,
    cosine_similarity,
    faithfulness,
    score_answer,
)
from rag.generation.types import INSUFFICIENT_CONTEXT, GroundedAnswer


def test_faithfulness_is_the_share_of_supported_claims() -> None:
    assert faithfulness([True, False]) == 0.5
    assert faithfulness([True, True]) == 1.0
    with pytest.raises(ValueError, match="no claims"):
        faithfulness([])
    with pytest.raises(ValueError, match="booleans"):
        faithfulness([True, 1])  # 1 is an int, not a supported flag


def test_cosine_and_the_noncommittal_penalty() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1 / math.sqrt(2))
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="same non-zero length"):
        cosine_similarity([1.0], [1.0, 0.0])
    assert answer_relevance([1.0, 0.0, 0.5], noncommittal=False) == pytest.approx(0.5)
    assert answer_relevance([1.0, 1.0], noncommittal=True) == 0.0
    with pytest.raises(ValueError, match="between -1 and 1"):
        answer_relevance([1.1], noncommittal=False)
    with pytest.raises(ValueError, match="number"):
        answer_relevance([True], noncommittal=False)


def test_score_uses_the_context_the_model_saw_and_embeds_questions_as_queries() -> None:
    claims = json.dumps({"claims": ["Forking copies the session.", "Cancel stops it."]})
    verdicts = json.dumps(
        {
            "verdicts": [
                {"claim": "Forking copies the session.", "supported": True},
                {"claim": "Cancel stops it.", "supported": False},
            ]
        }
    )
    questions = json.dumps(
        {
            "questions": [
                {"question": "What does forking do?", "noncommittal": False},
                {"question": "How does cancel work?", "noncommittal": False},
                {"question": "extra question", "noncommittal": False},
            ]
        }
    )
    llm = _ScriptedLLM([claims, verdicts, questions])
    embedder = _Embedder(
        {
            "How does forking work?": [1.0, 0.0],
            "What does forking do?": [1.0, 0.0],
            "How does cancel work?": [0.0, 1.0],
        }
    )
    trace = RequestTrace()
    answer = _answer(
        "Forking copies the session [1]. Cancel stops it [2].",
        context_text="The fork copies the session.",
    )
    score = score_answer(
        answer,
        llm=llm,
        embedder=embedder,
        relevance_questions=2,
        max_tokens=64,
        trace=trace,
    )
    assert score.faithfulness.score == 0.5
    assert [item.supported for item in score.faithfulness.claims] == [True, False]
    assert score.answer_relevance.score == pytest.approx(0.5)
    assert [item.question for item in score.answer_relevance.questions] == [
        "What does forking do?",
        "How does cancel work?",
    ]
    verdict_user = llm.calls[1][0][1].content
    assert "The fork copies the session." in verdict_user
    assert embedder.passages == []
    assert embedder.queries == [
        "How does forking work?",
        "What does forking do?",
        "How does cancel work?",
    ]
    assert llm.calls[0][1] == 64
    names = [span.name for span in trace.spans]
    assert "generation.faithfulness" in names
    assert "generation.answer_relevance" in names


def test_no_claims_skips_the_verdict_call() -> None:
    claims = json.dumps({"claims": []})
    questions = json.dumps({"questions": [{"question": "What is missing?", "noncommittal": True}]})
    llm = _ScriptedLLM([claims, questions])
    embedder = _Embedder(
        {
            "How does forking work?": [1.0, 0.0],
            "What is missing?": [1.0, 0.0],
        }
    )
    answer = _answer(INSUFFICIENT_CONTEXT, context_text="UNIQUE-CONTEXT")
    score = score_answer(answer, llm=llm, embedder=embedder, relevance_questions=1)
    assert score.faithfulness.score is None
    assert score.faithfulness.claims == ()
    assert score.answer_relevance.score == 0.0
    assert len(llm.calls) == 2
    sent = " ".join(message.content for call, _limit in llm.calls for message in call)
    assert "UNIQUE-CONTEXT" not in sent


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
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
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
        return self._vectors[text]

    def embed_passages(self, texts, *, trace=None) -> list[list[float]]:
        self.passages.extend(texts)
        raise AssertionError("answer relevance embeds questions with embed_query")


def _answer(text: str, *, context_text: str) -> GroundedAnswer:
    from rag.generation.types import ContextBlock

    block = ContextBlock(
        number=1,
        chunk_id="a",
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text=context_text,
    )
    return GroundedAnswer(
        query="How does forking work?",
        answer=text,
        sufficient=True,
        insufficient_reason=None,
        citations=(),
        context=(block,),
        model="scripted",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
    )
