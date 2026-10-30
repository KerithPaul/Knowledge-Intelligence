"""Client seams. Provider code stays behind these types."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag.core.trace import RequestTrace


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class LLMClient(ABC):
    """One chat completion. Streaming arrives with the API in Phase 7."""

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        trace: RequestTrace | None = None,
    ) -> LLMResponse:
        """Return the assistant text and the token and timing counts."""


class EmbeddingClient(ABC):
    """Turn text into vectors.

    Query text and passage text are separate calls because BGE adds an
    instruction to queries only.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of one vector, as declared in config."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model id, for traces and for the vector collection."""

    @abstractmethod
    def embed_query(self, text: str, *, trace: RequestTrace | None = None) -> list[float]:
        """Embed one search query."""

    @abstractmethod
    def embed_passages(
        self,
        texts: Sequence[str],
        *,
        trace: RequestTrace | None = None,
    ) -> list[list[float]]:
        """Embed passages in batches. An empty input returns an empty list."""


class Reranker(ABC):
    """Score query-passage pairs. A higher score is more relevant.

    The locked model returns one probability in (0, 1) per passage, after
    sigmoid. A later sufficiency check can compare that score to a threshold.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model id, for traces."""

    @abstractmethod
    def score(
        self,
        query: str,
        passages: Sequence[str],
        *,
        trace: RequestTrace | None = None,
    ) -> list[float]:
        """Return one score per passage, in the same order. Empty in, empty out."""
