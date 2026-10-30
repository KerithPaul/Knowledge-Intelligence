"""Phase 3 reranker check.

What this measures:
- BAAI/bge-reranker-base returns one probability per passage.
- A passage that answers the question scores above an unrelated passage.

Run with: pytest -m smoke -o addopts=
The default pytest run skips this module. The first call downloads the model.
"""

import pytest

from rag.core.clients import build_reranker
from rag.core.config import get_settings
from rag.core.trace import RequestTrace

pytestmark = pytest.mark.smoke


def test_bge_reranker_prefers_the_matching_passage() -> None:
    get_settings.cache_clear()
    reranker = build_reranker(get_settings())
    trace = RequestTrace()
    scores = reranker.score(
        "How many people live in Berlin?",
        [
            "Berlin had a population of 3,520,031 in 2019.",
            "The recipe uses two eggs and a cup of flour.",
        ],
        trace=trace,
    )
    assert len(scores) == 2
    assert all(0.0 < score < 1.0 for score in scores)
    assert scores[0] > scores[1]
    assert trace.spans[-1].name == "retrieval.rerank"
    assert trace.spans[-1].attributes["model"] == "BAAI/bge-reranker-base"
    get_settings.cache_clear()
