"""One process-wide lock around the models and the index.

The embedding model and the reranker are not safe to call from two
requests at once. Ingest writes the same collection `/ask` reads, so it
takes the same lock.
"""

import threading
from collections.abc import Callable, Iterator

from rag.core.config import Settings
from rag.core.observability import Tracer, open_tracer
from rag.core.trace import RequestTrace
from rag.retrieval.filters import MetadataFilter
from rag.routing.events import AskDelta, AskResult, AskStatus
from rag.routing.pipeline import Router, open_router
from rag.routing.types import RoutedAnswer
from rag.serving.health import HealthReport, check_health
from rag.serving.ingest import IngestReport, run_ingest


class Service:
    """The objects the routes call. Tests pass fakes for each one."""

    def __init__(
        self,
        settings: Settings,
        *,
        router: Router | None = None,
        ingest: Callable[[], IngestReport] | None = None,
        health: Callable[[], HealthReport] | None = None,
    ) -> None:
        self.settings = settings
        self._router = router
        self._ingest = ingest
        self._health = health if health is not None else (lambda: check_health(settings))
        self._lock = threading.Lock()
        self._tracer: Tracer | None = None

    @property
    def tracer(self) -> Tracer:
        """The exporter for this process. The first ask or ingest builds it.

        ``/health`` does not call this. A missing flag or a missing key stays
        a null tracer and does not import Langfuse.
        """

        if self._tracer is None:
            self._tracer = open_tracer(self.settings)
        return self._tracer

    def flush_tracer(self) -> None:
        """Send buffered observations when the process stops.

        A process that never asked or ingested has nothing to send.
        """

        if self._tracer is not None:
            self._tracer.flush()

    def health(self) -> HealthReport:
        return self._health()

    def ingest(self, *, trace: RequestTrace | None = None) -> IngestReport:
        with self._lock:
            if self._ingest is not None:
                return self._ingest()
            return run_ingest(self.settings, trace=trace)

    def ask(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None,
        top_n: int | None,
        top_k: int | None,
        trace: RequestTrace,
    ) -> RoutedAnswer:
        with self._lock:
            return self._router_or_open().ask(
                query,
                metadata_filter=metadata_filter,
                top_n=top_n,
                top_k=top_k,
                trace=trace,
            )

    def ask_stream(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None,
        top_n: int | None,
        top_k: int | None,
        trace: RequestTrace,
    ) -> Iterator[AskStatus | AskDelta | AskResult]:
        with self._lock:
            yield from self._router_or_open().ask_stream(
                query,
                metadata_filter=metadata_filter,
                top_n=top_n,
                top_k=top_k,
                trace=trace,
            )

    def _router_or_open(self) -> Router:
        if self._router is None:
            self._router = open_router(self.settings)
        return self._router
