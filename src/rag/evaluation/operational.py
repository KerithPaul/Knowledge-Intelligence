"""Latency, token cost, and the end-to-end win from one harness run.

Stage times come from the request trace. A later span can include the model
calls recorded inside it, so the stage table is not added up into the
end-to-end time. End-to-end time is the clock around the whole question.

Token totals count ``llm.complete`` spans that reported both token numbers.
A call that omitted usage is counted as unpriced and is left out of the cost.
Cost is ``tokens / 1_000_000 * usd_per_million``. The prices live in config.
They are the uncached rates. This report does not see a cached-token split.

A question the corpus cannot answer is a win when the gate refuses it.
A question the corpus can answer is a win when three things are true: the
labeled chunk is in the retrieved list, the gate accepted the answer, and
answer correctness reaches ``win_correctness``. A missing correctness score
is not a win.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.core.trace import RequestTrace


@dataclass(frozen=True)
class StageLatency:
    """One span name across the questions that recorded it."""

    name: str
    questions: int
    total_ms: float
    mean_ms: float


@dataclass(frozen=True)
class TokenUse:
    """Token counts from model calls, and how many calls omitted usage."""

    prompt_tokens: int
    completion_tokens: int
    unpriced_calls: int


def tokens_from_trace(trace: RequestTrace) -> TokenUse:
    """Sum usage on ``llm.complete`` spans. Missing usage is unpriced."""

    prompt = 0
    completion = 0
    unpriced = 0
    for span in trace.spans:
        if span.name != "llm.complete":
            continue
        prompt_tokens = span.attributes.get("prompt_tokens")
        completion_tokens = span.attributes.get("completion_tokens")
        if _is_count(prompt_tokens) and _is_count(completion_tokens):
            prompt += prompt_tokens
            completion += completion_tokens
        else:
            unpriced += 1
    return TokenUse(
        prompt_tokens=prompt,
        completion_tokens=completion,
        unpriced_calls=unpriced,
    )


def stage_times(trace: RequestTrace) -> dict[str, float]:
    """Total milliseconds per span name for one question."""

    totals: dict[str, float] = {}
    for span in trace.spans:
        if isinstance(span.latency_ms, bool) or not isinstance(span.latency_ms, int | float):
            raise ValueError("span latency must be a number")
        totals[span.name] = totals.get(span.name, 0.0) + float(span.latency_ms)
    return totals


def mean_stages(per_question: Sequence[Mapping[str, float]]) -> tuple[StageLatency, ...]:
    """Average each stage over the questions that recorded it."""

    if not per_question:
        raise ValueError("no questions")
    names: set[str] = set()
    for row in per_question:
        names.update(row)
    stages: list[StageLatency] = []
    for name in sorted(names):
        values = [float(row[name]) for row in per_question if name in row]
        total = sum(values)
        stages.append(
            StageLatency(
                name=name,
                questions=len(values),
                total_ms=total,
                mean_ms=total / len(values),
            )
        )
    return tuple(stages)


def cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    prompt_usd_per_million: float,
    completion_usd_per_million: float,
) -> float:
    """USD for these tokens at the configured uncached rates."""

    prompt = _count(prompt_tokens, name="prompt tokens")
    completion = _count(completion_tokens, name="completion tokens")
    prompt_rate = _rate(prompt_usd_per_million, name="prompt price")
    completion_rate = _rate(completion_usd_per_million, name="completion price")
    return prompt * prompt_rate / 1_000_000 + completion * completion_rate / 1_000_000


def case_won(
    *,
    corpus_can_answer: bool,
    route: str,
    retrieved: Sequence[str],
    relevant: Sequence[str],
    sufficient: bool,
    correctness: float | None,
    correctness_floor: float,
) -> bool:
    """Whether this question counts as an end-to-end win."""

    if not isinstance(corpus_can_answer, bool) or not isinstance(sufficient, bool):
        raise ValueError("labels must be booleans")
    if route not in {"knowledge_base", "insufficient"}:
        raise ValueError("route must be knowledge_base or insufficient")
    floor = _rate(correctness_floor, name="win correctness")
    if floor > 1.0:
        raise ValueError("win correctness must be between 0 and 1")
    if not corpus_can_answer:
        return route == "insufficient"
    if not relevant:
        raise ValueError("an answerable question needs a relevant chunk id")
    relevant_ids = set(relevant)
    hit = any(chunk_id in relevant_ids for chunk_id in retrieved)
    if route != "knowledge_base" or not hit or not sufficient:
        return False
    if correctness is None:
        return False
    if isinstance(correctness, bool) or not isinstance(correctness, int | float):
        raise ValueError("correctness must be a number")
    return float(correctness) >= floor


def _is_count(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _count(value: int, *, name: str) -> int:
    if not _is_count(value):
        raise ValueError(f"{name} must be zero or more")
    return value


def _rate(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be zero or more")
    return number
