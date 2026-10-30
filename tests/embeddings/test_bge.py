"""BGE prefixing and batching, with a fake encoder so the model is not loaded."""

import math

import pytest

from rag.core.errors import EmbeddingError, UnknownProviderError
from rag.core.trace import RequestTrace
from rag.embeddings.bge import BGEEmbeddingClient

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FakeEncoder:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.texts: list[str] = []
        self.kwargs = None

    def encode(self, texts, *, batch_size, normalize_embeddings, show_progress_bar):
        self.texts = list(texts)
        self.kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
        }
        return [[1.0 / math.sqrt(self.dimension)] * self.dimension for _ in texts]


def _client(encoder: FakeEncoder | None = None, **overrides) -> BGEEmbeddingClient:
    encoder = encoder or FakeEncoder()
    settings = dict(
        model_name="BAAI/bge-small-en-v1.5",
        dimension=encoder.dimension,
        query_prefix=QUERY_PREFIX,
        passage_prefix="",
        batch_size=8,
        normalize=True,
        device="cpu",
        encoder=encoder,
    )
    settings.update(overrides)
    return BGEEmbeddingClient(**settings)


def test_query_gets_the_prefix_and_passage_does_not() -> None:
    encoder = FakeEncoder()
    client = _client(encoder)
    trace = RequestTrace()
    text = "How does the agent loop work?"
    query = client.embed_query(text, trace=trace)
    assert encoder.texts == [f"{QUERY_PREFIX}{text}"]
    passages = client.embed_passages([text, "second passage"], trace=trace)
    assert encoder.texts == [text, "second passage"]
    assert len(query) == 4
    assert len(passages) == 2
    assert encoder.kwargs["batch_size"] == 8
    assert encoder.kwargs["normalize_embeddings"] is True
    assert encoder.kwargs["show_progress_bar"] is False
    kinds = [span.attributes["kind"] for span in trace.spans]
    assert kinds == ["query", "passage"]
    assert trace.spans[1].attributes["count"] == 2


def test_prefix_is_not_applied_twice() -> None:
    encoder = FakeEncoder()
    client = _client(encoder)
    already = f"{QUERY_PREFIX}already prefixed"
    client.embed_query(already)
    assert encoder.texts == [already]


def test_empty_inputs() -> None:
    client = _client()
    with pytest.raises(ValueError, match="empty"):
        client.embed_query("   ")
    with pytest.raises(ValueError, match="index 1"):
        client.embed_passages(["ok", "  "])
    assert client.embed_passages([]) == []


def test_dimension_mismatch_raises() -> None:
    client = _client(FakeEncoder(dimension=4), dimension=8)
    with pytest.raises(EmbeddingError, match="dimension"):
        client.embed_query("hello")


def test_factory_builds_bge_without_loading_the_model(tmp_path, monkeypatch) -> None:
    from rag.core.clients import build_embedding_client
    from rag.core.config import load_settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text(
        """
llm:
  provider: groq
  model: openai/gpt-oss-20b
  base_url: https://api.groq.com/openai/v1
  temperature: 0.2
  max_tokens: 32
  timeout_seconds: 10
embeddings:
  provider: bge
  model: BAAI/bge-small-en-v1.5
  dimension: 384
  query_prefix: "Represent this sentence for searching relevant passages: "
  passage_prefix: ""
  batch_size: 4
  normalize: true
  device: cpu
reranker:
  model: BAAI/bge-reranker-base
chunking:
  chunk_size_tokens: 100
  chunk_overlap_tokens: 0
retrieval:
  top_n: 10
  top_k: 3
  rrf_k: 60
routing:
  sufficiency_threshold: 0.5
qdrant:
  url: http://localhost:6333
  collection: knowledge
""",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    client = build_embedding_client(load_settings(yaml_file=yaml_file, env_file=env_file))
    assert client.model_name == "BAAI/bge-small-en-v1.5"
    assert client.dimension == 384
    assert client._encoder is None


def test_factory_rejects_unknown_embedding_provider(tmp_path, monkeypatch) -> None:
    from rag.core.clients import build_embedding_client
    from rag.core.config import load_settings

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    yaml_file = tmp_path / "settings.yaml"
    yaml_file.write_text(
        """
llm:
  provider: groq
  model: x
  base_url: https://example.test/v1
  temperature: 0.2
  max_tokens: 32
  timeout_seconds: 10
embeddings:
  provider: other
  model: x
  dimension: 8
  query_prefix: ""
  passage_prefix: ""
  batch_size: 2
  normalize: true
  device: cpu
reranker:
  model: x
chunking:
  chunk_size_tokens: 100
  chunk_overlap_tokens: 0
retrieval:
  top_n: 10
  top_k: 3
  rrf_k: 60
routing:
  sufficiency_threshold: 0.5
qdrant:
  url: http://localhost:6333
  collection: knowledge
""",
        encoding="utf-8",
    )
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    settings = load_settings(yaml_file=yaml_file, env_file=env_file)
    with pytest.raises(UnknownProviderError, match="other"):
        build_embedding_client(settings)
