"""Load tunables from configs/settings.yaml and secrets from .env.

Source order, highest priority first:
1. Keyword arguments to load_settings (tests and one-off overrides)
2. Process environment
3. .env file
4. YAML file
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# src/rag/core/config.py -> parents[3] is the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YAML = PROJECT_ROOT / "configs" / "settings.yaml"
DEFAULT_ENV = PROJECT_ROOT / ".env"


class LLMSettings(BaseModel):
    provider: str = "groq"
    model: str
    base_url: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    reasoning_effort: str | None = None
    include_reasoning: bool | None = None


class EmbeddingSettings(BaseModel):
    provider: str = "bge"
    model: str
    dimension: int = Field(gt=0)
    query_prefix: str = ""
    passage_prefix: str = ""
    batch_size: int = Field(gt=0)
    normalize: bool = True
    device: str = "cpu"


class RerankerSettings(BaseModel):
    """Local cross-encoder. The score is one probability per query-chunk pair."""

    model: str
    device: str = "cpu"
    batch_size: int = Field(default=16, gt=0)
    # The reranker drops tokens past this window. bge-reranker-base is 512.
    max_length: int = Field(default=512, gt=0)


class ChunkingSettings(BaseModel):
    chunk_size_tokens: int = Field(gt=0)
    chunk_overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def overlap_is_smaller_than_size(self) -> "ChunkingSettings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


class RetrievalSettings(BaseModel):
    """Candidate pool, final cut, RRF constant, and BM25 knobs.

    Dense search and BM25 each return `top_n` hits. RRF keeps `top_n` of
    the merged list for the reranker, and the reranker returns `top_k`.
    """

    top_n: int = Field(gt=0)
    top_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    # Lucene defaults. k1 saturates repeated terms. b is the length penalty.
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def top_k_fits_in_the_candidate_pool(self) -> "RetrievalSettings":
        if self.top_k > self.top_n:
            raise ValueError("top_k must be less than or equal to top_n")
        return self


class GenerationSettings(BaseModel):
    """How many chunks go into the prompt, and how answer relevance is scored.

    `max_context_chunks` is how many retrieved chunks are copied into the
    prompt, best first. `max_tokens` is the completion budget for that JSON
    answer. On gpt-oss, reasoning tokens count against the same budget, so
    this is larger than the short-call default on the LLM client.

    Answer relevance writes `relevance_questions` questions from the answer
    and averages their cosine similarity to the user question.
    """

    max_context_chunks: int = Field(default=5, gt=0)
    max_tokens: int = Field(default=1024, gt=0)
    relevance_questions: int = Field(default=3, gt=0)


class RoutingSettings(BaseModel):
    """When the best rerank probability is too low to call the model.

    ``sufficiency_threshold`` is compared with that probability. A lower
    score returns the insufficient-context sentence and does not call Groq.
    The value is from 0 to 1 because the rerank score is a probability.
    """

    sufficiency_threshold: float = Field(ge=0.0, le=1.0)


class EvaluationSettings(BaseModel):
    """Weights, the win bar, and the token prices for the Phase 6 harness.

    ``factual_weight`` is the share of answer correctness that comes from
    fact overlap. The rest is the cosine of the two passages.
    ``win_correctness`` is how high that score must be before an answerable
    question counts as a win.

    Prices are USD per 1,000,000 tokens. The default is 0 so a missing
    block does not invent a bill. ``settings.yaml`` sets the uncached
    Groq rates for the configured model. Cached input is cheaper on that
    price page, and the report does not split it out.
    """

    factual_weight: float = Field(default=0.75, ge=0.0, le=1.0)
    win_correctness: float = Field(default=0.5, ge=0.0, le=1.0)
    prompt_usd_per_million: float = Field(default=0.0, ge=0.0)
    completion_usd_per_million: float = Field(default=0.0, ge=0.0)
    synthesize_limit: int = Field(default=5, gt=0)


class QdrantSettings(BaseModel):
    url: str
    collection: str
    dense_vector: str = Field(default="dense", min_length=1)
    sparse_vector: str = Field(default="sparse", min_length=1)
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    # m is edges per HNSW node. ef_construct is the build-time search width.
    # ef_search is that width at query time.
    hnsw_m: int = Field(default=16, gt=0)
    hnsw_ef_construct: int = Field(default=100, gt=0)
    hnsw_ef_search: int = Field(default=128, gt=0)
    # Kilobytes, not point counts. indexing_threshold_kb of 0 turns indexing off.
    # The server rejects a full-scan threshold below 10.
    indexing_threshold_kb: int = Field(default=1, gt=0)
    full_scan_threshold_kb: int = Field(default=10, ge=10)
    upsert_batch_size: int = Field(default=64, gt=0)
    timeout_seconds: int = Field(default=30, gt=0)

    @model_validator(mode="after")
    def vector_names_differ(self) -> "QdrantSettings":
        if self.dense_vector == self.sparse_vector:
            raise ValueError("dense_vector and sparse_vector must use different names")
        return self


def _settings_config(yaml_file: Path, env_file: Path | None) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_file=str(env_file) if env_file is not None else None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        yaml_file=str(yaml_file),
        yaml_file_encoding="utf-8",
        extra="ignore",
    )


class Settings(BaseSettings):
    """Process-wide settings. Build one with load_settings, not by hand."""

    model_config = _settings_config(
        DEFAULT_YAML,
        DEFAULT_ENV if DEFAULT_ENV.is_file() else None,
    )

    groq_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    llm: LLMSettings
    embeddings: EmbeddingSettings
    reranker: RerankerSettings
    chunking: ChunkingSettings
    retrieval: RetrievalSettings
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    routing: RoutingSettings
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    qdrant: QdrantSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


def load_settings(
    *,
    yaml_file: Path | None = None,
    env_file: Path | None = None,
    **overrides: Any,
) -> Settings:
    """Load settings from the given files.

    Pass env_file only when you want a specific dotenv file. The default is
    the repository .env when that file exists. Pass env_file to a missing path
    and the call fails, so tests should point at a real temp file.
    """

    yaml_path = yaml_file or DEFAULT_YAML
    if env_file is None:
        env_path = DEFAULT_ENV if DEFAULT_ENV.is_file() else None
    else:
        env_path = env_file

    class _LoadedSettings(Settings):
        model_config = _settings_config(yaml_path, env_path)

    return _LoadedSettings(**overrides)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the repository settings. Call cache_clear() after env changes."""

    return load_settings()
