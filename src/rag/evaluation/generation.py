"""Faithfulness and answer relevance, from the formulas in the plan.

Faithfulness is the share of atomic claims in the answer that the context
supports. One judge call extracts the claims. A second call says whether a
careful reader could infer each claim from the context alone. The score is
supported claims divided by all claims. Zero claims has no score, because
that ratio would divide by zero.

Answer relevance is the mean cosine similarity between the user question and
questions written from the answer. When every generated question is marked
noncommittal, the score is 0. That is the penalty for an evasive answer.
Both sides are embedded with ``embed_query`` so they share one prefix. The
common library version embeds the generated questions as documents. That
split assumes the embedder does not add a query instruction. BGE does.

A judge that returns more questions than requested is cut to the requested
count. Fewer questions are scored as they are, as long as there is one.

Context precision, context recall, and answer correctness need a golden
answer or labeled chunks. Those stay in Phase 6.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rag.core.errors import GenerationError
from rag.core.interfaces import EmbeddingClient, LLMClient, LLMMessage
from rag.core.trace import RequestTrace
from rag.generation.context import render_context
from rag.generation.parse import request_json
from rag.generation.prompt import (
    CLAIMS_SYSTEM,
    QUESTIONS_SYSTEM,
    VERDICTS_SYSTEM,
    claims_user,
    questions_user,
    verdicts_user,
)
from rag.generation.types import GroundedAnswer


@dataclass(frozen=True)
class ClaimVerdict:
    """One atomic claim and whether the context supports it."""

    claim: str
    supported: bool


@dataclass(frozen=True)
class FaithfulnessResult:
    """`score` is None when the answer produced no claims."""

    score: float | None
    claims: tuple[ClaimVerdict, ...]


@dataclass(frozen=True)
class GeneratedQuestion:
    """A question written from the answer. `noncommittal` describes the answer."""

    question: str
    noncommittal: bool


@dataclass(frozen=True)
class AnswerRelevanceResult:
    """Mean cosine, or 0 when every generated question is noncommittal."""

    score: float
    questions: tuple[GeneratedQuestion, ...]
    similarities: tuple[float, ...]


@dataclass(frozen=True)
class GenerationScore:
    """The two Phase 4 measurements for one answer."""

    faithfulness: FaithfulnessResult
    answer_relevance: AnswerRelevanceResult


def faithfulness(supported: Sequence[bool]) -> float:
    """Supported claims divided by the number of claims."""

    if not supported:
        raise ValueError("no claims")
    for item in supported:
        if not isinstance(item, bool):
            raise ValueError("supported flags must be booleans")
    return sum(1 for item in supported if item) / len(supported)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of two vectors. A zero vector has no direction, so it raises."""

    if len(left) != len(right) or not left:
        raise ValueError("vectors must be the same non-zero length")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        a = _component(left_value)
        b = _component(right_value)
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("cosine is undefined for a zero vector")
    value = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value


def answer_relevance(similarities: Sequence[float], *, noncommittal: bool) -> float:
    """Mean cosine. An all-noncommittal answer scores 0.

    ``similarities`` are cosines, each between -1 and 1. ``noncommittal`` is
    true only when every question written from the answer was flagged evasive.
    """

    if not isinstance(noncommittal, bool):
        raise ValueError("noncommittal must be a boolean")
    if not similarities:
        raise ValueError("no generated questions")
    checked: list[float] = []
    for value in similarities:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("similarity must be a number")
        number = float(value)
        if not -1.0 <= number <= 1.0:
            raise ValueError("similarity must be between -1 and 1")
        checked.append(number)
    if noncommittal:
        return 0.0
    return sum(checked) / len(checked)


def score_answer(
    answer: GroundedAnswer,
    *,
    llm: LLMClient,
    embedder: EmbeddingClient,
    relevance_questions: int = 3,
    max_tokens: int = 1024,
    trace: RequestTrace | None = None,
) -> GenerationScore:
    """Measure one grounded answer. The judge is `llm`. The embedder is BGE today."""

    if relevance_questions <= 0:
        raise ValueError("relevance_questions must be positive")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not answer.query.strip():
        raise ValueError("query text is empty")
    if not answer.answer.strip():
        raise ValueError("answer text is empty")
    faith = _measure_faithfulness(answer, llm=llm, max_tokens=max_tokens, trace=trace)
    relevance = _measure_relevance(
        answer,
        llm=llm,
        embedder=embedder,
        count=relevance_questions,
        max_tokens=max_tokens,
        trace=trace,
    )
    return GenerationScore(faithfulness=faith, answer_relevance=relevance)


def _measure_faithfulness(
    answer: GroundedAnswer,
    *,
    llm: LLMClient,
    max_tokens: int,
    trace: RequestTrace | None,
) -> FaithfulnessResult:
    started = time.perf_counter()
    claims, _responses = request_json(
        llm,
        [
            LLMMessage(role="system", content=CLAIMS_SYSTEM),
            LLMMessage(role="user", content=claims_user(answer.answer)),
        ],
        _interpret_claims,
        max_tokens=max_tokens,
        trace=trace,
    )
    if not claims:
        result = FaithfulnessResult(score=None, claims=())
    else:
        flags, _responses = request_json(
            llm,
            [
                LLMMessage(role="system", content=VERDICTS_SYSTEM),
                LLMMessage(
                    role="user",
                    content=verdicts_user(render_context(answer.context), claims),
                ),
            ],
            _verdict_interpreter(claims),
            max_tokens=max_tokens,
            trace=trace,
        )
        verdicts = tuple(
            ClaimVerdict(claim=claim, supported=flag)
            for claim, flag in zip(claims, flags, strict=True)
        )
        result = FaithfulnessResult(score=faithfulness(flags), claims=verdicts)
    if trace is not None:
        supported = sum(1 for claim in result.claims if claim.supported)
        trace.record(
            "generation.faithfulness",
            (time.perf_counter() - started) * 1000,
            score=result.score,
            claims=len(result.claims),
            supported=supported,
        )
    return result


def _measure_relevance(
    answer: GroundedAnswer,
    *,
    llm: LLMClient,
    embedder: EmbeddingClient,
    count: int,
    max_tokens: int,
    trace: RequestTrace | None,
) -> AnswerRelevanceResult:
    started = time.perf_counter()
    questions, _responses = request_json(
        llm,
        [
            LLMMessage(role="system", content=QUESTIONS_SYSTEM),
            LLMMessage(role="user", content=questions_user(answer.answer, count)),
        ],
        _question_interpreter(count),
        max_tokens=max_tokens,
        trace=trace,
    )
    origin = embedder.embed_query(answer.query, trace=trace)
    vectors = [embedder.embed_query(item.question, trace=trace) for item in questions]
    similarities = tuple(cosine_similarity(origin, vector) for vector in vectors)
    noncommittal = all(item.noncommittal for item in questions)
    result = AnswerRelevanceResult(
        score=answer_relevance(similarities, noncommittal=noncommittal),
        questions=questions,
        similarities=similarities,
    )
    if trace is not None:
        trace.record(
            "generation.answer_relevance",
            (time.perf_counter() - started) * 1000,
            score=result.score,
            questions=len(result.questions),
            noncommittal=noncommittal,
        )
    return result


def _interpret_claims(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("claims")
    if not isinstance(raw, list):
        raise GenerationError("claims must be a list")
    claims: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise GenerationError("each claim must be a non-empty string")
        claims.append(item.strip())
    return tuple(claims)


def _verdict_interpreter(claims: Sequence[str]) -> Callable[[dict[str, Any]], tuple[bool, ...]]:
    expected = len(claims)

    def interpret(payload: dict[str, Any]) -> tuple[bool, ...]:
        raw = payload.get("verdicts")
        if not isinstance(raw, list) or len(raw) != expected:
            raise GenerationError("verdicts must match the claims")
        flags: list[bool] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GenerationError("each verdict must be an object")
            supported = item.get("supported")
            if not isinstance(supported, bool):
                raise GenerationError("supported must be a boolean")
            flags.append(supported)
        return tuple(flags)

    return interpret


def _question_interpreter(
    expected: int,
) -> Callable[[dict[str, Any]], tuple[GeneratedQuestion, ...]]:
    def interpret(payload: dict[str, Any]) -> tuple[GeneratedQuestion, ...]:
        raw = payload.get("questions")
        if not isinstance(raw, list) or not raw:
            raise GenerationError("questions must be a non-empty list")
        parsed: list[GeneratedQuestion] = []
        for item in raw:
            if not isinstance(item, dict):
                raise GenerationError("each question must be an object")
            question = item.get("question")
            flag = item.get("noncommittal")
            if not isinstance(question, str) or not question.strip():
                raise GenerationError("question text is empty")
            if not isinstance(flag, bool):
                raise GenerationError("noncommittal must be a boolean")
            parsed.append(GeneratedQuestion(question=question.strip(), noncommittal=flag))
        return tuple(parsed[:expected])

    return interpret


def _component(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("vector values must be numbers")
    return float(value)
