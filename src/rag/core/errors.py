"""Errors for config, the model clients, indexing, and retrieval."""


class ConfigError(Exception):
    """The settings or the chosen provider cannot be used."""


class MissingSecretError(ConfigError):
    """A required API key is missing. The message must not include the key."""


class UnknownProviderError(ConfigError):
    """Settings name a provider this build does not know."""


class LLMError(Exception):
    """The language-model call failed. The message must not include the API key."""


class EmbeddingError(Exception):
    """The embedding call failed or returned an unexpected vector."""


class IndexingError(Exception):
    """The vector index could not be written or failed its own check."""


class RetrievalError(Exception):
    """Search could not run, or the collection does not match the chunk file."""
