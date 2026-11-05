"""FastAPI routes for `/ask`, `/ingest`, and `/health`.

`/ask` uses the Phase 5 gate. `stream: true` is server-sent events.
The text events are the grounded answer. Provider tokens stay inside
the process until the citation rules accept or replace them.

A stream that fails after the first event still uses HTTP 200 and sends
an `error` event. The status line was already sent. A non-stream failure
uses the HTTP status below.
"""

import logging
import time
import uuid
from collections.abc import Callable, Iterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag import __version__
from rag.core.config import Settings, get_settings
from rag.core.errors import (
    ConfigError,
    EmbeddingError,
    GenerationError,
    IndexingError,
    LLMError,
    MissingSecretError,
    RetrievalError,
)
from rag.core.redact import redact_secrets
from rag.core.trace import RequestTrace
from rag.retrieval.filters import MetadataFilter
from rag.routing.events import AskDelta, AskResult, AskStatus
from rag.routing.pipeline import Router
from rag.serving.health import HealthReport
from rag.serving.ingest import IngestReport
from rag.serving.logging import configure_logging, log_request
from rag.serving.runtime import Service
from rag.serving.schemas import (
    AskRequest,
    ask_payload,
    error_payload,
    health_payload,
    ingest_payload,
    metadata_from,
    status_payload,
    trace_payload,
)
from rag.serving.sse import sse

# Status, machine code, and the sentence a client may show.
# MissingSecretError is a ConfigError, so it has to be matched first.
_ERRORS: tuple[tuple[type[Exception], int, str, str], ...] = (
    (MissingSecretError, 500, "config_error", "A required API key is not set."),
    (ConfigError, 500, "config_error", "The server configuration cannot be used."),
    (GenerationError, 502, "generation_failed", "The model reply could not be read."),
    (LLMError, 502, "model_failed", "The model call failed."),
    (RetrievalError, 503, "retrieval_failed", "Search could not run."),
    (EmbeddingError, 503, "embedding_failed", "The embedding call failed."),
    (IndexingError, 503, "index_failed", "The index could not be written."),
    (FileNotFoundError, 400, "corpus_missing", "The corpus directory does not exist."),
)
_UNEXPECTED = (500, "request_failed", "The request could not be completed.")


class RequestIdMiddleware:
    """Copy `X-Request-ID` onto the response. Generate one when it is absent.

    This does not read or buffer the body, so a streamed `/ask` still flushes
    each event.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = _header(scope, b"x-request-id") or uuid.uuid4().hex
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message.setdefault("headers", []))
                headers["x-request-id"] = request_id
            await send(message)

        await self.app(scope, receive, send_with_id)


def create_app(
    settings: Settings | None = None,
    *,
    router: Router | None = None,
    ingest: Callable[[], IngestReport] | None = None,
    health: Callable[[], HealthReport] | None = None,
) -> FastAPI:
    """Build the API. Pass `router`, `ingest`, or `health` in tests.

    The real router loads the embedding model and the reranker on the first
    `/ask`. `/health` does not load them.
    """

    loaded = settings if settings is not None else get_settings()
    logger = configure_logging(loaded.serving.log_level)
    service = Service(loaded, router=router, ingest=ingest, health=health)
    app = FastAPI(title="Knowledge Intelligence", version=__version__)
    app.add_middleware(RequestIdMiddleware)
    app.state.service = service
    app.state.logger = logger

    @app.get("/health")
    def health_route() -> JSONResponse:
        report = service.health()
        status_code = 200 if report.status == "ok" else 503
        return JSONResponse(status_code=status_code, content=health_payload(report))

    @app.post("/ingest")
    def ingest_route(request: Request) -> JSONResponse:
        request_id = _request_id(request)
        started = time.perf_counter()
        try:
            report = service.ingest()
        except Exception as exc:
            return _failed(request, exc, request_id=request_id, started=started, query_chars=None)
        latency_ms = (time.perf_counter() - started) * 1000
        log_request(
            logger,
            request_id=request_id,
            method="POST",
            path="/ingest",
            status_code=200,
            latency_ms=latency_ms,
            documents=report.documents,
            chunks=report.chunks,
            points=report.points,
            profile=loaded.serving.profile,
        )
        return JSONResponse(content=ingest_payload(report))

    @app.post("/ask", response_model=None)
    def ask_route(body: AskRequest, request: Request) -> JSONResponse | StreamingResponse:
        request_id = _request_id(request)
        metadata = metadata_from(body)
        if body.stream:
            return StreamingResponse(
                _ask_events(
                    service,
                    body,
                    request_id=request_id,
                    metadata=metadata,
                    logger=logger,
                    profile=loaded.serving.profile,
                ),
                media_type="text/event-stream",
                headers={"cache-control": "no-cache", "x-accel-buffering": "no"},
            )
        trace = RequestTrace()
        started = time.perf_counter()
        try:
            routed = service.ask(
                body.query,
                metadata_filter=metadata,
                top_n=body.top_n,
                top_k=body.top_k,
                trace=trace,
            )
        except Exception as exc:
            return _failed(
                request,
                exc,
                request_id=request_id,
                started=started,
                query_chars=len(body.query),
            )
        latency_ms = (time.perf_counter() - started) * 1000
        log_request(
            logger,
            request_id=request_id,
            method="POST",
            path="/ask",
            status_code=200,
            latency_ms=latency_ms,
            query_chars=len(body.query),
            route=routed.route,
            decided_by=routed.decided_by,
            reason=routed.reason,
            profile=loaded.serving.profile,
        )
        return JSONResponse(
            content=ask_payload(routed, request_id=request_id, latency_ms=latency_ms, trace=trace)
        )

    return app


def _ask_events(
    service: Service,
    body: AskRequest,
    *,
    request_id: str,
    metadata: MetadataFilter | None,
    logger: logging.Logger,
    profile: str,
) -> Iterator[str]:
    trace = RequestTrace()
    started = time.perf_counter()
    route = None
    decided_by = None
    reason = None
    error_code = None
    error_detail = None
    try:
        for event in service.ask_stream(
            body.query,
            metadata_filter=metadata,
            top_n=body.top_n,
            top_k=body.top_k,
            trace=trace,
        ):
            if isinstance(event, AskStatus):
                yield sse("status", status_payload(event, request_id))
            elif isinstance(event, AskDelta):
                yield sse("delta", {"request_id": request_id, "text": event.text})
            elif isinstance(event, AskResult):
                routed = event.routed
                route = routed.route
                decided_by = routed.decided_by
                reason = routed.reason
                latency_ms = (time.perf_counter() - started) * 1000
                yield sse(
                    "answer",
                    ask_payload(routed, request_id=request_id, latency_ms=latency_ms),
                )
                yield sse(
                    "trace",
                    {"request_id": request_id, "spans": trace_payload(trace)},
                )
                yield sse("done", {"request_id": request_id})
    except Exception as exc:
        _status_code, error_code, message = _public(exc)
        del _status_code
        error_detail = redact_secrets(str(exc))
        yield sse("error", error_payload(request_id, error_code, message))
    latency_ms = (time.perf_counter() - started) * 1000
    log_request(
        logger,
        request_id=request_id,
        method="POST",
        path="/ask",
        status_code=200,
        latency_ms=latency_ms,
        query_chars=len(body.query),
        route=route,
        decided_by=decided_by,
        reason=reason,
        stream_error=error_code,
        error_detail=error_detail,
        profile=profile,
    )


def _failed(
    request: Request,
    exc: Exception,
    *,
    request_id: str,
    started: float,
    query_chars: int | None,
) -> JSONResponse:
    status_code, code, message = _public(exc)
    log_request(
        request.app.state.logger,
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=status_code,
        latency_ms=(time.perf_counter() - started) * 1000,
        query_chars=query_chars,
        error=code,
        error_detail=redact_secrets(str(exc)),
        profile=request.app.state.service.settings.serving.profile,
    )
    return JSONResponse(
        status_code=status_code,
        content=error_payload(request_id, code, message),
    )


def _public(exc: Exception) -> tuple[int, str, str]:
    for error_type, status_code, code, message in _ERRORS:
        if isinstance(exc, error_type):
            return status_code, code, message
    return _UNEXPECTED


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return uuid.uuid4().hex


def _header(scope: Scope, name: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            text = value.decode("latin-1").strip()
            if text:
                return text
    return ""
