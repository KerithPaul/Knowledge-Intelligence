"""The route chosen for one question.

`knowledge_base` means the grounded answer was allowed to stand.
`insufficient` means the ask path stopped. This build does not search
the web. The text in that case is the Phase 4 insufficient-context sentence.
"""

import math
from dataclasses import dataclass
from typing import Literal

from rag.generation.types import INSUFFICIENT_CONTEXT, GroundedAnswer
from rag.retrieval.types import RetrievedChunk

RouteName = Literal["knowledge_base", "insufficient"]
DecidedBy = Literal["retrieval_score", "groundedness"]
# `below_threshold` is only produced by the score gate. The other three
# are the generator's reasons. `empty_context` is also used when retrieval
# returned nothing, which is decided before the model is called.
RoutingReason = Literal["below_threshold", "empty_context", "model", "uncited"]


@dataclass(frozen=True)
class RoutedAnswer:
    """One ask-path result after the sufficiency gate.

    `best_rerank_score` is the highest cross-encoder probability among
    the retrieved chunks. It is missing when retrieval returned nothing.
    `answer` is missing when the model was not called. `text` is the
    sentence a person should read in either case.
    """

    query: str
    route: RouteName
    decided_by: DecidedBy
    reason: RoutingReason | None
    best_rerank_score: float | None
    threshold: float
    chunks: tuple[RetrievedChunk, ...]
    answer: GroundedAnswer | None

    def __post_init__(self) -> None:
        if not self.query.strip() or self.query != self.query.strip():
            raise ValueError("query text is empty or has surrounding whitespace")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("sufficiency_threshold must be between 0 and 1")
        if self.best_rerank_score is not None and not math.isfinite(self.best_rerank_score):
            raise ValueError("best rerank score must be a finite number")
        if self.answer is not None and self.answer.query != self.query:
            raise ValueError("the answer query must match the routed query")
        if self.route == "knowledge_base":
            self._check_accepted()
        else:
            self._check_refused()

    @property
    def text(self) -> str:
        if self.answer is None:
            return INSUFFICIENT_CONTEXT
        return self.answer.answer

    def _check_accepted(self) -> None:
        if self.decided_by != "groundedness":
            raise ValueError("only the groundedness check can accept an answer")
        if self.reason is not None:
            raise ValueError("an accepted answer has no refusal reason")
        if self.answer is None or not self.answer.sufficient:
            raise ValueError("a knowledge-base route needs a sufficient answer")
        if not self.chunks or self.best_rerank_score is None:
            raise ValueError("a knowledge-base route needs a rerank score")
        if self.best_rerank_score < self.threshold:
            raise ValueError("a knowledge-base route needs a score at or above the threshold")

    def _check_refused(self) -> None:
        if self.reason is None:
            raise ValueError("an insufficient route needs a reason")
        if self.decided_by == "retrieval_score":
            self._check_score_refusal()
            return
        if self.answer is None or self.answer.sufficient:
            raise ValueError("a groundedness refusal needs the generator's insufficient answer")
        if self.reason == "below_threshold":
            raise ValueError("the generator does not decide the score threshold")
        if self.answer.insufficient_reason != self.reason:
            raise ValueError("the route reason must match the generator")
        if not self.chunks or self.best_rerank_score is None:
            raise ValueError("a groundedness refusal needs a rerank score")
        if self.best_rerank_score < self.threshold:
            raise ValueError("a groundedness refusal needs a score at or above the threshold")

    def _check_score_refusal(self) -> None:
        if self.answer is not None:
            raise ValueError("a retrieval-score refusal does not call the model")
        if self.reason == "empty_context":
            if self.chunks or self.best_rerank_score is not None:
                raise ValueError("an empty retrieval has no chunks and no rerank score")
            return
        if self.reason != "below_threshold":
            raise ValueError("a retrieval-score refusal is empty or below the threshold")
        if not self.chunks or self.best_rerank_score is None:
            raise ValueError("a below-threshold refusal needs a rerank score")
        if self.best_rerank_score >= self.threshold:
            raise ValueError("a below-threshold refusal needs a score under the threshold")
