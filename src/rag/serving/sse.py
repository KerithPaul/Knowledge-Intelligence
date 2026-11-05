"""Server-sent events for `/ask`."""

import json
from typing import Any


def sse(event: str, payload: dict[str, Any]) -> str:
    """One SSE block. Callers choose which fields go in `payload`."""

    data = json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n"
