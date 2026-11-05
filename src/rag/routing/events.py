"""Events from one streaming ask.

The text events are the grounded answer, split into short pieces. They are
not the raw model tokens. The citation rules can replace those tokens, and
this stream starts only after that check.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from rag.routing.types import RoutedAnswer

DELTA_CHARS = 48


@dataclass(frozen=True)
class AskStatus:
    """A stage boundary. `gate` is the only stage that carries scores."""

    stage: Literal["retrieving", "gate", "generating"]
    best_rerank_score: float | None = None
    threshold: float | None = None
    chunk_count: int | None = None
    passed: bool | None = None


@dataclass(frozen=True)
class AskDelta:
    """A piece of the answer a person should read."""

    text: str


@dataclass(frozen=True)
class AskResult:
    """The finished route. The same object `Router.ask` would return."""

    routed: RoutedAnswer


def iter_text_deltas(text: str, *, size: int = DELTA_CHARS) -> Iterator[AskDelta]:
    """Split `text` into pieces of at most `size` characters."""

    if size <= 0:
        raise ValueError("delta size must be positive")
    for start in range(0, len(text), size):
        yield AskDelta(text=text[start : start + size])
