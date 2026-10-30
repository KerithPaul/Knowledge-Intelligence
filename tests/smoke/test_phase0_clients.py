"""Phase 0 exit check.

What this measures:
- Groq returns a non-empty answer, token counts, and a latency span.
- BGE returns 384-d unit vectors, and the query prefix changes the vector.

Run with: pytest -m smoke -o addopts=
The default pytest run skips this module. It needs GROQ_API_KEY and it
downloads BAAI/bge-small-en-v1.5 on the first call.
"""

import math

import pytest

from rag.core.clients import build_embedding_client, build_llm_client
from rag.core.config import get_settings
from rag.core.interfaces import LLMMessage
from rag.core.trace import RequestTrace

pytestmark = pytest.mark.smoke


def test_groq_completion_smoke() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    client = build_llm_client(settings)
    trace = RequestTrace()
    response = client.complete(
        [
            LLMMessage(
                role="user",
                content="Reply with exactly this one word and nothing else: pong",
            )
        ],
        max_tokens=256,
        trace=trace,
    )
    assert response.content.strip()
    assert "pong" in response.content.lower()
    assert response.model
    assert response.latency_ms >= 0
    assert response.prompt_tokens is not None
    assert response.completion_tokens is not None
    assert trace.spans[0].name == "llm.complete"
    assert trace.spans[0].attributes["model"] == response.model


def test_bge_embedding_smoke() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    client = build_embedding_client(settings)
    trace = RequestTrace()
    text = "How does the agent loop call tools?"
    query_vector = client.embed_query(text, trace=trace)
    passage_vectors = client.embed_passages([text], trace=trace)

    assert len(query_vector) == settings.embeddings.dimension
    assert len(passage_vectors) == 1
    assert len(passage_vectors[0]) == settings.embeddings.dimension
    norm = math.sqrt(sum(value * value for value in query_vector))
    assert norm == pytest.approx(1.0, abs=1e-3)
    # Same words, different prefix, so the vectors must differ.
    assert query_vector != passage_vectors[0]
    assert [span.attributes["kind"] for span in trace.spans] == ["query", "passage"]
