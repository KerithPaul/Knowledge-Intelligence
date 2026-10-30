"""Build the clients named in settings.

This is the only place that picks a provider. Callers depend on LLMClient,
EmbeddingClient, and Reranker, not on Groq or BGE.
"""

from rag.core.config import Settings
from rag.core.errors import MissingSecretError, UnknownProviderError
from rag.core.interfaces import EmbeddingClient, LLMClient, Reranker


def build_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm.provider
    if provider == "groq":
        from rag.generation.groq import GroqLLMClient

        key = settings.groq_api_key
        if key is None or not key.get_secret_value().strip():
            raise MissingSecretError("GROQ_API_KEY is not set. Add it to .env.")
        return GroqLLMClient(
            api_key=key.get_secret_value(),
            model=settings.llm.model,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            timeout_seconds=settings.llm.timeout_seconds,
            reasoning_effort=settings.llm.reasoning_effort,
            include_reasoning=settings.llm.include_reasoning,
        )
    raise UnknownProviderError(f"unknown LLM provider: {provider}")


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    provider = settings.embeddings.provider
    if provider == "bge":
        from rag.embeddings.bge import BGEEmbeddingClient

        embeddings = settings.embeddings
        return BGEEmbeddingClient(
            model_name=embeddings.model,
            dimension=embeddings.dimension,
            query_prefix=embeddings.query_prefix,
            passage_prefix=embeddings.passage_prefix,
            batch_size=embeddings.batch_size,
            normalize=embeddings.normalize,
            device=embeddings.device,
        )
    raise UnknownProviderError(f"unknown embedding provider: {provider}")


def build_reranker(settings: Settings) -> Reranker:
    """Build the local cross-encoder. The weights load on the first score call."""

    from rag.retrieval.rerank import BGEReranker

    reranker = settings.reranker
    return BGEReranker(
        model_name=reranker.model,
        device=reranker.device,
        batch_size=reranker.batch_size,
        max_length=reranker.max_length,
    )
