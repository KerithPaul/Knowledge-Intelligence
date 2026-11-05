"""Readiness of this process and the Qdrant it searches."""

from dataclasses import dataclass
from typing import Any, Literal

from rag.core.config import Settings
from rag.indexing.store import build_qdrant_client, count_points


@dataclass(frozen=True)
class HealthReport:
    """`ok` means Qdrant answered. The collection may still be missing.

    A missing collection is not a down server. `/ask` reports that case
    when someone searches before ingest. `down` means the probe failed.
    """

    status: Literal["ok", "down"]
    profile: str
    qdrant: Literal["ok", "down"]
    collection: str
    collection_ready: bool
    points: int | None


def check_health(settings: Settings, *, client: Any | None = None) -> HealthReport:
    """Probe Qdrant. Pass `client` in tests to skip the network."""

    qdrant = settings.qdrant
    try:
        store = client if client is not None else build_qdrant_client(qdrant)
        ready = bool(store.collection_exists(qdrant.collection))
        points = count_points(store, qdrant.collection) if ready else None
    except Exception:
        return HealthReport(
            status="down",
            profile=settings.serving.profile,
            qdrant="down",
            collection=qdrant.collection,
            collection_ready=False,
            points=None,
        )
    return HealthReport(
        status="ok",
        profile=settings.serving.profile,
        qdrant="ok",
        collection=qdrant.collection,
        collection_ready=ready,
        points=points,
    )
