"""Local BGE cross-encoder reranker.

Each query-passage pair gets one relevance probability. sentence-transformers
would apply sigmoid for this one-label model on its own. This class passes
sigmoid explicitly so the score stays in (0, 1) for the model named in
settings. Higher is more relevant. The query is the raw question. The BGE
bi-encoder prefix belongs on the dense query vector only.
"""

import math
import time
from collections.abc import Sequence
from typing import Any, Protocol

from rag.core.errors import RetrievalError
from rag.core.interfaces import Reranker
from rag.core.trace import RequestTrace


class PairScorer(Protocol):
    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        """Return one score per pair."""


class BGEReranker(Reranker):
    """sentence-transformers CrossEncoder for BAAI/bge-reranker-base."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
        scorer: PairScorer | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if not model_name.strip():
            raise ValueError("reranker model name is empty")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._scorer = scorer

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(
        self,
        query: str,
        passages: Sequence[str],
        *,
        trace: RequestTrace | None = None,
    ) -> list[float]:
        if not passages:
            return []
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("query text is empty")
        for index, passage in enumerate(passages):
            if not isinstance(passage, str) or not passage.strip():
                raise ValueError(f"passage at index {index} is empty")
        pairs = [(cleaned, passage) for passage in passages]
        started = time.perf_counter()
        try:
            raw = self._load().predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(str(exc)) from None
        scores = _as_scores(raw, expected=len(passages))
        if trace is not None:
            trace.record(
                "retrieval.rerank",
                (time.perf_counter() - started) * 1000,
                model=self._model_name,
                count=len(passages),
                max_length=self._max_length,
            )
        return scores

    def _load(self) -> PairScorer:
        if self._scorer is None:
            import os

            # Windows without Developer Mode cannot symlink the model cache.
            # The files still download; this only hides that warning.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import torch.nn as nn
            from sentence_transformers import CrossEncoder

            self._scorer = CrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
                activation_fn=nn.Sigmoid(),
            )
        return self._scorer


def _as_scores(raw: Any, *, expected: int) -> list[float]:
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, int | float):
        raw = [raw]
    if not isinstance(raw, list):
        raise RetrievalError("reranker returned an unexpected score shape")
    scores: list[float] = []
    for item in raw:
        if isinstance(item, list):
            if len(item) != 1:
                raise RetrievalError("reranker returned an unexpected score shape")
            item = item[0]
        try:
            scores.append(float(item))
        except (TypeError, ValueError) as exc:
            raise RetrievalError("reranker returned an unexpected score shape") from exc
    if len(scores) != expected:
        raise RetrievalError(f"reranker returned {len(scores)} scores for {expected} passages")
    if not all(math.isfinite(score) for score in scores):
        raise RetrievalError("reranker returned a non-finite score")
    return scores
