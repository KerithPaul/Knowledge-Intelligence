"""Phase 5 routing: a score gate, then the grounded answer.

Web search is not part of this package. The insufficient route is the
signal a later agent can use to call its own web-search tool.
"""

from rag.routing.pipeline import Router, open_router
from rag.routing.types import RoutedAnswer

__all__ = [
    "RoutedAnswer",
    "Router",
    "open_router",
]
