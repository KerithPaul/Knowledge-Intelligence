"""Strip API keys and bearer tokens from text that might be logged."""

import re

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_KEY_RE = re.compile(r"\b(?:gsk|sk)-[A-Za-z0-9_\-]+")


def redact_secrets(text: str, secret: str = "") -> str:
    """Return `text` with known secret shapes replaced.

    `secret` is an extra exact value, such as the key this process loaded.
    The replacement is the word `[REDACTED]`. The input is not logged here.
    """

    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _KEY_RE.sub("[REDACTED]", text)
    return text
