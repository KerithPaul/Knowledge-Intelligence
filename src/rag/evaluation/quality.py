"""Context precision, context recall, and answer correctness.

These are the Phase 6 generation metrics from the plan. Faithfulness and
answer relevance stay in ``generation.py``.

Context precision asks whether the useful chunks are ranked first. A judge
marks each retrieved block as useful or not for the reference answer.
The score is the average of precision@k at each useful rank:

    sum(precision@k * v_k) / (number of useful blocks)

``v_k`` is 1 when block k is useful. Precision@k is the useful blocks in
the first k divided by k. No useful block scores 0. An empty retrieval
scores 0 and does not call the judge.

The published library formula adds ``1e-10`` to the denominator, so a
perfect one-block case prints as 0.9999999999. This module divides by the
useful count and returns exactly 0 or 1. ``crosscheck.py`` keeps the
library form so the two numbers can be compared.

Context recall is the share of claims in the reference answer that the
retrieved context supports. It is the faithfulness ratio applied to the
reference, not to the model's answer. No reference claims has no score.

Answer correctness mixes two comparisons with the reference. Factual
overlap is an F1 over atomic facts:

    true positives / (true positives + 0.5 * (false positives + false negatives))

Semantic similarity is the cosine of the answer and the reference. Both
are embedded with ``embed_passages``, so neither receives the BGE query
instruction. The score is ``factual_weight * F1 + (1 - factual_weight) *
cosine``. The default weight 0.75 is the usual library default. The cosine
can be negative, so the weighted score can be negative. No facts on either
side leaves the factual score empty, and the weighted score is empty too.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rag.core.errors import GenerationError
from rag.core.interfaces import EmbeddingClient, LLMClient, LLMMessage
from rag.core.trace import RequestTrace
from rag.evaluation.generation import cosine_similarity, faithfulness
from rag.generation.context import render_context
from rag.generation.parse import request_json
from rag.generation.prompt import (
    CLAIMS_SYSTEM,
    CLASSIFY_SYSTEM,
    CONTEXT_PRECISION_SYSTEM,
    VERDICTS_SYSTEM,
    claims_user,
    classify_user,
    context_precision_user,
    verdicts_user,
)
from rag.generation.types import ContextBlock


@dataclass(frozen=True)
class ContextPrecisionResult:
    """Usefulness of each block, best first, and the average-precision score."""

    score: float
    useful: tuple[bool, ...]


@dataclass(frozen=True)
class ContextRecallResult:
    """`score` is None when the reference answer produced no claims."""

    score: float | None
    claims: tuple[str, ...]
    attributed: tuple[bool, ...]


@dataclass(frozen=True)
class FactOverlap:
    """Atomic facts sorted against the reference answer."""

    true_positives: tuple[str, ...]
    false_positives: tuple[str, ...]
    false_negatives: tuple[str, ...]

    @property
    def factual_score(self) -> float | None:
        total = len(self.true_positives) + len(self.false_positives) + len(self.false_negatives)
        if total == 0:
            return None
        return factual_f1(
            len(self.true_positives),
            len(self.false_positives),
            len(self.false_negatives),
        )


@dataclass(frozen=True)
class AnswerCorrectnessResult:
    """Weighted factual overlap and semantic similarity. `score` is None
    when there are no facts to compare.
    """

    score: float | None
    factual_score: float | None
    semantic_similarity: float
    facts: FactOverlap
    factual_weight: float


@dataclass(frozen=True)
class QualityScore:
    """The three Phase 6 measurements for one answer and its reference."""

    context_precision: ContextPrecisionResult
    context_recall: ContextRecallResult
    answer_correctness: AnswerCorrectnessResult


def context_precision(useful: Sequence[bool]) -> float:
    """Average precision of usefulness flags. The first flag is rank 1.

    An empty flag list is an empty retrieval and scores 0. A list with no
    useful block also scores 0.
    """

    if not useful:
        return 0.0
    hits = 0
    total = 0.0
    for index, flag in enumerate(useful, start=1):
        if not isinstance(flag, bool):
            raise ValueError("useful flags must be booleans")
        if flag:
            hits += 1
            total += hits / index
    if hits == 0:
        return 0.0
    return total / hits


def context_recall(attributed: Sequence[bool]) -> float:
    """Attributed reference claims divided by the number of reference claims."""

    if not attributed:
        raise ValueError("no reference claims")
    return faithfulness(attributed)


def factual_f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """F1 of fact counts. The false counts are weighted by one half."""

    counts = (true_positives, false_positives, false_negatives)
    for count in counts:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("fact counts must be zero or more")
    denominator = true_positives + 0.5 * (false_positives + false_negatives)
    if denominator == 0:
        raise ValueError("no statements")
    return true_positives / denominator


def answer_correctness(
    factual_score: float,
    semantic_similarity: float,
    *,
    factual_weight: float = 0.75,
) -> float:
    """Weighted sum of factual F1 and semantic cosine.

    ``factual_weight`` is the share given to the F1. The rest goes to the
    cosine. The cosine is between -1 and 1, and the result is not clamped.
    """

    weight = _unit_interval(factual_weight, name="factual_weight")
    factual = _plain_number(factual_score, name="factual score")
    semantic = _plain_number(semantic_similarity, name="semantic similarity")
    if not -1.0 <= semantic <= 1.0:
        raise ValueError("semantic similarity must be between -1 and 1")
    return weight * factual + (1.0 - weight) * semantic


def score_quality(
    *,
    query: str,
    answer: str,
    reference: str,
    context: Sequence[ContextBlock],
    llm: LLMClient,
    embedder: EmbeddingClient,
    factual_weight: float = 0.75,
    max_tokens: int = 1024,
    trace: RequestTrace | None = None,
) -> QualityScore:
    """Measure one answer against a reference. The judge is ``llm``."""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not query.strip():
        raise ValueError("query text is empty")
    if not answer.strip():
        raise ValueError("answer text is empty")
    if not reference.strip():
        raise ValueError("reference answer is empty")
    weight = _unit_interval(factual_weight, name="factual_weight")
    blocks = tuple(context)
    precision = _measure_precision(
        query=query,
        reference=reference,
        blocks=blocks,
        llm=llm,
        max_tokens=max_tokens,
        trace=trace,
    )
    recall = _measure_recall(
        reference=reference,
        blocks=blocks,
        llm=llm,
        max_tokens=max_tokens,
        trace=trace,
    )
    correctness = _measure_correctness(
        answer=answer,
        reference=reference,
        llm=llm,
        embedder=embedder,
        factual_weight=weight,
        max_tokens=max_tokens,
        trace=trace,
    )
    return QualityScore(
        context_precision=precision,
        context_recall=recall,
        answer_correctness=correctness,
    )


def _measure_precision(
    *,
    query: str,
    reference: str,
    blocks: Sequence[ContextBlock],
    llm: LLMClient,
    max_tokens: int,
    trace: RequestTrace | None,
) -> ContextPrecisionResult:
    started = time.perf_counter()
    if not blocks:
        result = ContextPrecisionResult(score=0.0, useful=())
    else:
        flags, _responses = request_json(
            llm,
            [
                LLMMessage(role="system", content=CONTEXT_PRECISION_SYSTEM),
                LLMMessage(
                    role="user",
                    content=context_precision_user(query, reference, render_context(blocks)),
                ),
            ],
            _precision_interpreter(len(blocks)),
            max_tokens=max_tokens,
            trace=trace,
        )
        result = ContextPrecisionResult(score=context_precision(flags), useful=flags)
    if trace is not None:
        trace.record(
            "evaluation.context_precision",
            (time.perf_counter() - started) * 1000,
            score=result.score,
            blocks=len(result.useful),
            useful=sum(1 for flag in result.useful if flag),
        )
    return result


def _measure_recall(
    *,
    reference: str,
    blocks: Sequence[ContextBlock],
    llm: LLMClient,
    max_tokens: int,
    trace: RequestTrace | None,
) -> ContextRecallResult:
    started = time.perf_counter()
    claims, _responses = request_json(
        llm,
        [
            LLMMessage(role="system", content=CLAIMS_SYSTEM),
            LLMMessage(role="user", content=claims_user(reference)),
        ],
        _interpret_claims,
        max_tokens=max_tokens,
        trace=trace,
    )
    if not claims:
        result = ContextRecallResult(score=None, claims=(), attributed=())
    else:
        flags, _responses = request_json(
            llm,
            [
                LLMMessage(role="system", content=VERDICTS_SYSTEM),
                LLMMessage(
                    role="user",
                    content=verdicts_user(render_context(blocks), claims),
                ),
            ],
            _verdict_interpreter(claims),
            max_tokens=max_tokens,
            trace=trace,
        )
        result = ContextRecallResult(
            score=context_recall(flags),
            claims=claims,
            attributed=flags,
        )
    if trace is not None:
        trace.record(
            "evaluation.context_recall",
            (time.perf_counter() - started) * 1000,
            score=result.score,
            claims=len(result.claims),
            attributed=sum(1 for flag in result.attributed if flag),
        )
    return result


def _measure_correctness(
    *,
    answer: str,
    reference: str,
    llm: LLMClient,
    embedder: EmbeddingClient,
    factual_weight: float,
    max_tokens: int,
    trace: RequestTrace | None,
) -> AnswerCorrectnessResult:
    started = time.perf_counter()
    facts, _responses = request_json(
        llm,
        [
            LLMMessage(role="system", content=CLASSIFY_SYSTEM),
            LLMMessage(role="user", content=classify_user(answer, reference)),
        ],
        _interpret_facts,
        max_tokens=max_tokens,
        trace=trace,
    )
    vectors = embedder.embed_passages([answer, reference], trace=trace)
    if len(vectors) != 2:
        raise ValueError("the embedder must return one vector per passage")
    semantic = cosine_similarity(vectors[0], vectors[1])
    factual = facts.factual_score
    weighted = None
    if factual is not None:
        weighted = answer_correctness(factual, semantic, factual_weight=factual_weight)
    result = AnswerCorrectnessResult(
        score=weighted,
        factual_score=factual,
        semantic_similarity=semantic,
        facts=facts,
        factual_weight=factual_weight,
    )
    if trace is not None:
        trace.record(
            "evaluation.answer_correctness",
            (time.perf_counter() - started) * 1000,
            score=result.score,
            factual_score=result.factual_score,
            semantic_similarity=result.semantic_similarity,
        )
    return result


def _precision_interpreter(expected: int) -> Callable[[dict[str, Any]], tuple[bool, ...]]:
    def interpret(payload: dict[str, Any]) -> tuple[bool, ...]:
        raw = payload.get("verdicts")
        if not isinstance(raw, list) or len(raw) != expected:
            raise GenerationError("verdicts must match the context blocks")
        flags: list[bool] = []
        for number, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                raise GenerationError("each verdict must be an object")
            if item.get("number") != number:
                raise GenerationError("verdict numbers must follow the blocks")
            useful = item.get("useful")
            if not isinstance(useful, bool):
                raise GenerationError("useful must be a boolean")
            flags.append(useful)
        return tuple(flags)

    return interpret


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


def _interpret_facts(payload: dict[str, Any]) -> FactOverlap:
    grouped: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for name in ("true_positives", "false_positives", "false_negatives"):
        raw = payload.get(name)
        if not isinstance(raw, list):
            raise GenerationError(f"{name} must be a list")
        facts: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise GenerationError("each fact must be a non-empty string")
            fact = item.strip()
            if fact in seen:
                raise GenerationError("a fact is listed twice")
            seen.add(fact)
            facts.append(fact)
        grouped[name] = tuple(facts)
    return FactOverlap(
        true_positives=grouped["true_positives"],
        false_positives=grouped["false_positives"],
        false_negatives=grouped["false_negatives"],
    )


def _unit_interval(value: object, *, name: str) -> float:
    number = _plain_number(value, name=name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _plain_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)
