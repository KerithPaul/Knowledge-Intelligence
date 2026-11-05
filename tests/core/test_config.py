"""Settings load from YAML, then .env, then process env, then kwargs."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.core.config import get_settings, load_settings

_YAML = """
llm:
  provider: groq
  model: from-yaml
  base_url: https://example.test/v1
  temperature: 0.2
  max_tokens: 128
  timeout_seconds: 30
  reasoning_effort: low
  include_reasoning: false
embeddings:
  provider: bge
  model: BAAI/bge-small-en-v1.5
  dimension: 384
  query_prefix: "Represent this sentence for searching relevant passages: "
  passage_prefix: ""
  batch_size: 8
  normalize: true
  device: cpu
reranker:
  model: BAAI/bge-reranker-base
chunking:
  chunk_size_tokens: 512
  chunk_overlap_tokens: 64
retrieval:
  top_n: 20
  top_k: 5
  rrf_k: 60
routing:
  sufficiency_threshold: 0.35
qdrant:
  url: http://localhost:6333
  collection: knowledge
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GROQ_API_KEY", "TAVILY_API_KEY", "LLM__MODEL", "LLM__TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)


def test_repository_yaml_sets_the_evaluation_prices() -> None:
    settings = load_settings()
    assert settings.serving.profile == "local"
    assert settings.serving.host == "127.0.0.1"
    assert settings.serving.port == 8000
    assert settings.serving.log_level == "info"
    assert settings.evaluation.factual_weight == 0.75
    assert settings.evaluation.win_correctness == 0.5
    assert settings.evaluation.prompt_usd_per_million == 0.075
    assert settings.evaluation.completion_usd_per_million == 0.30
    assert settings.evaluation.synthesize_limit == 5


def test_yaml_values_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    yaml_file = _write(tmp_path, "settings.yaml", _YAML)
    env_file = _write(tmp_path, ".env", "")
    settings = load_settings(yaml_file=yaml_file, env_file=env_file)
    assert settings.llm.model == "from-yaml"
    assert settings.embeddings.dimension == 384
    assert settings.embeddings.passage_prefix == ""
    assert settings.retrieval.top_k == 5
    assert settings.retrieval.bm25_k1 == 1.2
    assert settings.retrieval.bm25_b == 0.75
    assert settings.groq_api_key is None
    assert settings.tavily_api_key is None


def test_dotenv_and_env_override_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    yaml_file = _write(tmp_path, "settings.yaml", _YAML)
    env_file = _write(
        tmp_path,
        ".env",
        "GROQ_API_KEY=dotenv-key-value\nTAVILY_API_KEY=tavily-dotenv\nLLM__MODEL=from-dotenv\n",
    )
    monkeypatch.setenv("LLM__MODEL", "from-env")
    settings = load_settings(yaml_file=yaml_file, env_file=env_file)
    assert settings.llm.model == "from-env"
    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "dotenv-key-value"
    assert settings.tavily_api_key is not None
    assert settings.tavily_api_key.get_secret_value() == "tavily-dotenv"
    assert "dotenv-key-value" not in repr(settings)
    assert "tavily-dotenv" not in repr(settings)


def test_serving_profile_can_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.delenv("SERVING__PROFILE", raising=False)
    monkeypatch.delenv("SERVING__HOST", raising=False)
    yaml_file = _write(tmp_path, "settings.yaml", _YAML)
    env_file = _write(tmp_path, ".env", "")
    monkeypatch.setenv("SERVING__PROFILE", "docker")
    monkeypatch.setenv("SERVING__HOST", "0.0.0.0")
    settings = load_settings(yaml_file=yaml_file, env_file=env_file)
    assert settings.serving.profile == "docker"
    assert settings.serving.host == "0.0.0.0"


def test_keyword_override_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    yaml_file = _write(tmp_path, "settings.yaml", _YAML)
    env_file = _write(tmp_path, ".env", "LLM__MODEL=from-dotenv\n")
    settings = load_settings(
        yaml_file=yaml_file,
        env_file=env_file,
        llm={
            "provider": "groq",
            "model": "from-kwargs",
            "base_url": "https://example.test/v1",
            "temperature": 0.2,
            "max_tokens": 128,
            "timeout_seconds": 30,
        },
    )
    assert settings.llm.model == "from-kwargs"


def test_repo_yaml_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    env_file = _write(tmp_path, ".env", "")
    settings = load_settings(env_file=env_file)
    assert settings.llm.provider == "groq"
    assert settings.llm.model == "openai/gpt-oss-20b"
    assert settings.llm.base_url == "https://api.groq.com/openai/v1"
    assert settings.llm.reasoning_effort == "low"
    assert settings.llm.include_reasoning is False
    assert settings.embeddings.provider == "bge"
    assert settings.embeddings.model == "BAAI/bge-small-en-v1.5"
    assert settings.embeddings.dimension == 384
    assert settings.embeddings.query_prefix.startswith("Represent this sentence")
    assert settings.embeddings.passage_prefix == ""
    assert settings.embeddings.device == "cpu"
    assert settings.qdrant.url == "http://localhost:6333"
    assert settings.qdrant.collection == "knowledge"
    assert settings.qdrant.dense_vector == "dense"
    assert settings.qdrant.sparse_vector == "sparse"
    assert settings.qdrant.distance == "cosine"
    assert settings.qdrant.hnsw_m == 16
    assert settings.qdrant.hnsw_ef_construct == 100
    assert settings.qdrant.hnsw_ef_search == 128
    assert settings.qdrant.indexing_threshold_kb == 1
    assert settings.qdrant.full_scan_threshold_kb == 10
    assert settings.qdrant.upsert_batch_size == 64
    assert settings.chunking.chunk_size_tokens == 512
    assert settings.chunking.chunk_overlap_tokens == 64
    assert settings.retrieval.top_n == 20
    assert settings.retrieval.top_k == 5
    assert settings.retrieval.rrf_k == 60
    assert settings.retrieval.bm25_k1 == 1.2
    assert settings.retrieval.bm25_b == 0.75
    assert settings.generation.max_context_chunks == 5
    assert settings.generation.max_tokens == 1024
    assert settings.generation.relevance_questions == 3
    assert settings.reranker.model == "BAAI/bge-reranker-base"
    assert settings.reranker.device == "cpu"
    assert settings.reranker.batch_size == 16
    assert settings.reranker.max_length == 512
    assert settings.groq_api_key is None


def test_overlap_must_be_smaller_than_chunk_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch)
    bad_yaml = _YAML.replace("chunk_overlap_tokens: 64", "chunk_overlap_tokens: 512")
    yaml_file = _write(tmp_path, "settings.yaml", bad_yaml)
    env_file = _write(tmp_path, ".env", "")
    with pytest.raises(ValidationError):
        load_settings(yaml_file=yaml_file, env_file=env_file)


def test_top_k_cannot_exceed_the_candidate_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch)
    bad_yaml = _YAML.replace("top_k: 5", "top_k: 21")
    yaml_file = _write(tmp_path, "settings.yaml", bad_yaml)
    env_file = _write(tmp_path, ".env", "")
    with pytest.raises(ValidationError):
        load_settings(yaml_file=yaml_file, env_file=env_file)


def test_context_chunk_limit_must_be_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_env(monkeypatch)
    yaml_file = _write(tmp_path, "settings.yaml", _YAML)
    env_file = _write(tmp_path, ".env", "")
    with pytest.raises(ValidationError):
        load_settings(
            yaml_file=yaml_file,
            env_file=env_file,
            generation={"max_context_chunks": 0},
        )


def test_overlap_cannot_be_negative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    bad_yaml = _YAML.replace("chunk_overlap_tokens: 64", "chunk_overlap_tokens: -1")
    yaml_file = _write(tmp_path, "settings.yaml", bad_yaml)
    env_file = _write(tmp_path, ".env", "")
    with pytest.raises(ValidationError):
        load_settings(yaml_file=yaml_file, env_file=env_file)


def test_get_settings_reads_real_files(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.embeddings.dimension == 384
    assert settings.llm.model == "openai/gpt-oss-20b"
    # The key is present or absent. Either way it must not show up as plain text.
    dumped = repr(settings)
    if settings.groq_api_key is not None:
        assert settings.groq_api_key.get_secret_value() not in dumped
    get_settings.cache_clear()
