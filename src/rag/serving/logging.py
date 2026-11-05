"""One JSON line per request. Lines do not carry answers or API keys."""

import json
import logging
import time
from typing import Any

from rag.core.redact import redact_secrets

LOGGER_NAME = "rag.serving"
_SECRET_FIELDS = frozenset(
    {"api_key", "apikey", "authorization", "groq_api_key", "password", "secret", "tavily_api_key"}
)


class _JsonHandler(logging.StreamHandler):
    """The marker lets configure_logging attach this handler only once."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(JsonFormatter())
        self._rag_json = True


class JsonFormatter(logging.Formatter):
    """A single-line JSON object. String fields are redacted."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_secrets(record.getMessage()),
        }
        fields = getattr(record, "rag", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if str(key).lower() in _SECRET_FIELDS:
                    continue
                payload[str(key)] = redact_secrets(value) if isinstance(value, str) else value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> logging.Logger:
    """Attach one JSON handler to the serving logger. Safe to call again."""

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False
    if not any(getattr(handler, "_rag_json", False) for handler in logger.handlers):
        handler = _JsonHandler()
        logger.addHandler(handler)
    return logger


def log_request(logger: logging.Logger, **fields: Any) -> None:
    """Write one request line. The answer text is not a field.

    `None` values are left out so a normal ask does not log an empty error.
    """

    present = {key: value for key, value in fields.items() if value is not None}
    logger.info("request", extra={"rag": present})
