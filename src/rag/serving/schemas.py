"""Request and response bodies for the API.

The JSON body and the SSE `answer` event share one shape. The SSE body
then adds a `trace` event and a `done` event. Chunk text stays out of
both. The trace already refuses secrets.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.core.trace import RequestTrace
from rag.generation.types import Citation
from rag.retrieval.filters import MetadataFilter
from rag.routing.events import AskStatus
from rag.routing.types import RoutedAnswer
from rag.serving.health import HealthReport
from rag.serving.ingest import IngestReport


class AskRequest(BaseModel):
    """One question. `stream` true is the server-sent event response."""

    model_config = ConfigDict(extra="forbid")

    query: str
    stream: bool = False
    source: list[str] = Field(default_factory=list)
    doc_title: list[str] = Field(default_factory=list)
    chunk_id: list[str] = Field(default_factory=list)
    heading: str | None = None
    top_n: int | None = Field(default=None, gt=0)
    top_k: int | None = Field(default=None, gt=0)

    @field_validator("query")
    @classmethod
    def query_has_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query text is empty")
        return cleaned


def metadata_from(body: AskRequest) -> MetadataFilter | None:
    """The same filter fields as `python -m rag.routing`. Empty means no filter."""

    metadata = MetadataFilter(
        sources=tuple(body.source),
        doc_titles=tuple(body.doc_title),
        chunk_ids=tuple(body.chunk_id),
        heading=body.heading,
    )
    if metadata.is_empty():
        return None
    return metadata


def ask_payload(
    routed: RoutedAnswer,
    *,
    request_id: str,
    latency_ms: float,
    trace: RequestTrace | None = None,
) -> dict[str, Any]:
    """Fields a client can render. `trace` is omitted on the SSE answer event."""

    answer = routed.answer
    citations = () if answer is None else answer.citations
    payload: dict[str, Any] = {
        "request_id": request_id,
        "query": routed.query,
        "route": routed.route,
        "decided_by": routed.decided_by,
        "reason": routed.reason,
        "best_rerank_score": routed.best_rerank_score,
        "threshold": routed.threshold,
        "answer": routed.text,
        "sufficient": answer is not None and answer.sufficient,
        "citations": [_citation_payload(item) for item in citations],
        "model": answer.model if answer is not None and answer.model else None,
        "prompt_tokens": None if answer is None else answer.prompt_tokens,
        "completion_tokens": None if answer is None else answer.completion_tokens,
        "latency_ms": latency_ms,
    }
    if trace is not None:
        payload["trace"] = trace_payload(trace)
    return payload


def status_payload(event: AskStatus, request_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"request_id": request_id, "stage": event.stage}
    if event.stage == "gate":
        payload["best_rerank_score"] = event.best_rerank_score
        payload["threshold"] = event.threshold
        payload["chunk_count"] = event.chunk_count
        payload["passed"] = event.passed
    return payload


def trace_payload(trace: RequestTrace) -> list[dict[str, Any]]:
    return [
        {"name": span.name, "latency_ms": span.latency_ms, "attributes": dict(span.attributes)}
        for span in trace.spans
    ]


def health_payload(report: HealthReport) -> dict[str, Any]:
    return {
        "status": report.status,
        "profile": report.profile,
        "qdrant": report.qdrant,
        "collection": report.collection,
        "collection_ready": report.collection_ready,
        "points": report.points,
    }


def ingest_payload(report: IngestReport) -> dict[str, Any]:
    return {
        "documents": report.documents,
        "chunks": report.chunks,
        "orphan_rate": report.orphan_rate,
        "code_split_rate": report.code_split_rate,
        "points": report.points,
        "upserted": report.upserted,
        "pruned": report.pruned,
        "schema_recreated": report.schema_recreated,
        "self_hit_at_1": report.self_hit_at_1,
        "latency_ms": report.latency_ms,
        "trace": trace_payload(report.trace),
    }


def error_payload(request_id: str, code: str, message: str) -> dict[str, str]:
    return {"request_id": request_id, "error": code, "message": message}


def _citation_payload(citation: Citation) -> dict[str, Any]:
    return {
        "number": citation.number,
        "chunk_id": citation.chunk_id,
        "source": citation.source,
        "doc_title": citation.doc_title,
        "heading_path": list(citation.heading_path),
    }
