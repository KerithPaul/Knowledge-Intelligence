"""Read the JSON object the model was asked to return.

A fenced block is accepted. So is a JSON object with a short preface, as long
as the object itself parses. The citation check then drops any marker that
does not name a context block. An answer that claims to be sufficient and
cites nothing is turned into the insufficient-context signal, and the model's
uncited text is not kept.
"""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from rag.core.errors import GenerationError
from rag.core.interfaces import LLMClient, LLMMessage, LLMResponse
from rag.core.trace import RequestTrace
from rag.generation.types import INSUFFICIENT_CONTEXT, InsufficientReason

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
_MARKER = re.compile(r"\[(\d+)\]")
_PUNCT = re.compile(r" +([.,;:!?])")


@dataclass(frozen=True)
class ParsedDraft:
    """A model reply after the citation rules have been applied."""

    sufficient: bool
    answer: str
    insufficient_reason: InsufficientReason | None
    citation_numbers: tuple[int, ...]


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the JSON object in a model reply."""

    cleaned = text.strip()
    if not cleaned:
        raise GenerationError("model reply is not a JSON object")
    if cleaned.startswith("```"):
        match = _FENCE.search(cleaned)
        if match is None:
            raise GenerationError("model reply is not a JSON object")
        cleaned = match.group(1).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = _slice_object(cleaned)
    if not isinstance(value, dict):
        raise GenerationError("model reply is not a JSON object")
    return value


def interpret_answer(payload: dict[str, Any], *, context_size: int) -> ParsedDraft:
    """Apply the grounding rules to one parsed object.

    `context_size` is how many blocks the model was shown. Citation numbers
    outside `1 .. context_size` are dropped. Markers for those numbers are
    removed from the answer text.
    """

    if context_size < 0:
        raise ValueError("context_size must be zero or more")
    sufficient, answer, declared = _fields(payload)
    valid = set(range(1, context_size + 1))
    ordered = _citation_order(answer, declared, valid)
    cleaned = _strip_invalid_markers(answer, valid)
    if not sufficient:
        return ParsedDraft(
            sufficient=False,
            answer=INSUFFICIENT_CONTEXT,
            insufficient_reason="model",
            citation_numbers=(),
        )
    if not ordered or not cleaned:
        # An uncited answer is not safe to show. The signal replaces it.
        return ParsedDraft(
            sufficient=False,
            answer=INSUFFICIENT_CONTEXT,
            insufficient_reason="uncited",
            citation_numbers=(),
        )
    return ParsedDraft(
        sufficient=True,
        answer=cleaned,
        insufficient_reason=None,
        citation_numbers=tuple(ordered),
    )


def request_json[T](
    llm: LLMClient,
    messages: Sequence[LLMMessage],
    interpret: Callable[[dict[str, Any]], T],
    *,
    max_tokens: int,
    trace: RequestTrace | None = None,
) -> tuple[T, tuple[LLMResponse, ...]]:
    """Ask for one JSON object and check it. A bad reply is retried once.

    The retry shows the model its own reply and the short reason it failed.
    The second failure raises GenerationError. The reason does not include
    the reply text.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not messages:
        raise ValueError("messages must not be empty")
    responses: list[LLMResponse] = []
    conversation = list(messages)
    last_error = "model reply could not be read"
    for attempt in range(2):
        response = llm.complete(conversation, max_tokens=max_tokens, trace=trace)
        responses.append(response)
        try:
            payload = extract_json_object(response.content)
            return interpret(payload), tuple(responses)
        except GenerationError as exc:
            last_error = str(exc)
            if attempt == 1:
                break
            conversation = [
                *messages,
                LLMMessage(role="assistant", content=response.content),
                LLMMessage(
                    role="user",
                    content=(
                        f"That reply could not be parsed ({last_error}). "
                        "Reply with one JSON object and no other text."
                    ),
                ),
            ]
    raise GenerationError(f"model reply could not be read: {last_error}") from None


def _fields(payload: dict[str, Any]) -> tuple[bool, str, list[int]]:
    if not isinstance(payload, dict):
        raise GenerationError("model reply is not a JSON object")
    missing = [name for name in ("sufficient", "answer", "citations") if name not in payload]
    if missing:
        joined = ", ".join(missing)
        raise GenerationError(f"model reply is missing {joined}")
    sufficient = payload["sufficient"]
    answer = payload["answer"]
    citations = payload["citations"]
    if not isinstance(sufficient, bool):
        raise GenerationError("sufficient must be a boolean")
    if not isinstance(answer, str) or not answer.strip():
        raise GenerationError("answer must be a non-empty string")
    if not isinstance(citations, list):
        raise GenerationError("citations must be a list")
    declared: list[int] = []
    for item in citations:
        if isinstance(item, bool) or not isinstance(item, int):
            raise GenerationError("citations must be integers")
        declared.append(item)
    return sufficient, answer.strip(), declared


def _citation_order(answer: str, declared: Sequence[int], valid: set[int]) -> list[int]:
    markers = [int(match) for match in _MARKER.findall(answer)]
    ordered: list[int] = []
    seen: set[int] = set()
    for number in [*markers, *declared]:
        if number not in valid or number in seen:
            continue
        seen.add(number)
        ordered.append(number)
    return ordered


def _strip_invalid_markers(answer: str, valid: set[int]) -> str:
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number in valid:
            return match.group(0)
        return ""

    cleaned = _MARKER.sub(replace, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = _PUNCT.sub(r"\1", cleaned)
    return cleaned.strip()


def _slice_object(cleaned: str) -> object:
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise GenerationError("model reply is not a JSON object")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        raise GenerationError("model reply is not a JSON object") from None
