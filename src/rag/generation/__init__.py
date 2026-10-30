"""Answer generation. Phase 0 ships the Groq client. Prompting lands in Phase 4."""

from rag.generation.groq import GroqLLMClient

__all__ = ["GroqLLMClient"]
