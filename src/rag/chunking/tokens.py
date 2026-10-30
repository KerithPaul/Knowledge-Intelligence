"""Token counters for the chunk budget.

The embedder truncates at its own window. The counter used for real ingestion
must be that model's tokenizer, or a 512-token chunk is not really 512.
Tests use WordTokenCounter so they stay offline.
"""

import re
from typing import Any, Protocol


class TokenCounter(Protocol):
    """How many tokens the embedder will spend on a string."""

    def count(self, text: str) -> int:
        """Return the token count. Empty text may still cost special tokens."""


_WORD_RE = re.compile(r"\S+")


class WordTokenCounter:
    """One token per whitespace-separated word. No special tokens.

    This is for tests and for a quick structural pass. It is not the BGE budget.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(_WORD_RE.findall(text))


class HuggingFaceTokenCounter:
    """Count with the embedding model's tokenizer, special tokens included.

    bge-small-en-v1.5 has a 512-token window. [CLS] and [SEP] use two of those
    positions, so a chunk that counts as 512 here fits and a longer one does not.
    """

    def __init__(self, model_name: str) -> None:
        if not model_name.strip():
            raise ValueError("model_name is empty")
        self.model_name = model_name
        self._tokenizer: Any = None

    def count(self, text: str) -> int:
        tokenizer = self._load()
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    def _load(self) -> Any:
        if self._tokenizer is None:
            import os

            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Long code fences are counted on purpose. Leave the model window
            # check to the chunker, and do not warn on the way there.
            tokenizer.model_max_length = 10**12
            self._tokenizer = tokenizer
        return self._tokenizer


def build_token_counter(model_name: str) -> HuggingFaceTokenCounter:
    """Counter for the embedding model named in settings."""

    return HuggingFaceTokenCounter(model_name)
