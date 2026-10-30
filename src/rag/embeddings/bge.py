"""Local BGE embeddings.

Query text gets the BGE instruction prefix. Passage text does not.
sentence-transformers batches the forward pass at batch_size.
"""

import time
from collections.abc import Sequence
from typing import Any, Protocol

from rag.core.errors import EmbeddingError
from rag.core.interfaces import EmbeddingClient
from rag.core.trace import RequestTrace


class Encoder(Protocol):
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Return one vector per text. A numpy array or a list of lists."""


def _with_prefix(text: str, prefix: str) -> str:
    if not prefix or text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def _to_vectors(raw: Any) -> list[list[float]]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return [[float(value) for value in row] for row in raw]


class BGEEmbeddingClient(EmbeddingClient):
    """sentence-transformers wrapper for BAAI/bge-small-en-v1.5."""

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        query_prefix: str,
        passage_prefix: str,
        batch_size: int,
        normalize: bool,
        device: str,
        encoder: Encoder | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._dimension = dimension
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size
        self._normalize = normalize
        self._device = device
        self._encoder = encoder

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_query(self, text: str, *, trace: RequestTrace | None = None) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("query text is empty")
        prepared = _with_prefix(cleaned, self._query_prefix)
        vectors = self._embed([prepared], kind="query", trace=trace)
        return vectors[0]

    def embed_passages(
        self,
        texts: Sequence[str],
        *,
        trace: RequestTrace | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        prepared: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"passage at index {index} is empty")
            prepared.append(_with_prefix(text.strip(), self._passage_prefix))
        return self._embed(prepared, kind="passage", trace=trace)

    def _load(self) -> Encoder:
        if self._encoder is None:
            import os

            # Windows without Developer Mode cannot symlink the model cache.
            # The files still download; this only hides that warning.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model_name, device=self._device)
        return self._encoder

    def _embed(
        self,
        texts: list[str],
        *,
        kind: str,
        trace: RequestTrace | None,
    ) -> list[list[float]]:
        encoder = self._load()
        started = time.perf_counter()
        try:
            raw = encoder.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(str(exc)) from None
        latency_ms = (time.perf_counter() - started) * 1000
        vectors = _to_vectors(raw)
        if len(vectors) != len(texts):
            raise EmbeddingError(f"model returned {len(vectors)} vectors for {len(texts)} texts")
        for vector in vectors:
            if len(vector) != self._dimension:
                raise EmbeddingError(
                    f"model {self._model_name} returned dimension {len(vector)}, "
                    f"config says {self._dimension}"
                )
        if trace is not None:
            trace.record(
                "embedding.encode",
                latency_ms,
                model=self._model_name,
                count=len(texts),
                dimension=self._dimension,
                kind=kind,
            )
        return vectors
