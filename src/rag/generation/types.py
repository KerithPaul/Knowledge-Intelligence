"""A grounded answer and the context blocks it was allowed to use."""

from dataclasses import dataclass
from typing import Literal, Protocol


class ContextSource(Protocol):
    """What the prompt needs from a retrieved chunk.

    Retrieval scores are not part of this. A later web snippet can satisfy
    the same fields without pretending it has a BM25 score.
    """

    chunk_id: str
    source: str
    doc_title: str
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class ContextBlock:
    """One chunk placed in the prompt. `number` is the citation marker."""

    number: int
    chunk_id: str
    source: str
    doc_title: str
    heading_path: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Citation:
    """A context block the answer pointed at."""

    number: int
    chunk_id: str
    source: str
    doc_title: str
    heading_path: tuple[str, ...]


# The machine-readable signal is GroundedAnswer.sufficient is False.
# This sentence is what a person sees in that case. It does not add facts.
INSUFFICIENT_CONTEXT = (
    "The indexed documentation does not contain enough information to answer this."
)

InsufficientReason = Literal["empty_context", "model", "uncited"]


@dataclass(frozen=True)
class GroundedAnswer:
    """One grounded reply.

    `sufficient` is false when there were no chunks, when the model said the
    context is not enough, or when a claimed answer cited no real block.
    In all three cases `answer` is `INSUFFICIENT_CONTEXT`.
    `context` is the blocks the model was allowed to see, in prompt order.
    Token counts are zero when the model was not called. They are None when
    a call did not report usage.
    """

    query: str
    answer: str
    sufficient: bool
    insufficient_reason: InsufficientReason | None
    citations: tuple[Citation, ...]
    context: tuple[ContextBlock, ...]
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
