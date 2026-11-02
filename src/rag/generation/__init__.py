"""Grounded answers. Phase 0 shipped the Groq client. Phase 4 adds the prompt."""

from rag.generation.groq import GroqLLMClient
from rag.generation.pipeline import GroundedGenerator, open_generator
from rag.generation.types import (
    INSUFFICIENT_CONTEXT,
    Citation,
    ContextBlock,
    GroundedAnswer,
)

__all__ = [
    "INSUFFICIENT_CONTEXT",
    "Citation",
    "ContextBlock",
    "GroundedAnswer",
    "GroundedGenerator",
    "GroqLLMClient",
    "open_generator",
]
