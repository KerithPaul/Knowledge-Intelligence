"""HTTP /ask, /ingest, and /health. No network and no model weights."""

import json
import logging
from typing import Literal

from fastapi.testclient import TestClient

from rag.core.config import load_settings
from rag.core.errors import GenerationError, IndexingError, RetrievalError
from rag.core.interfaces import LLMClient, LLMResponse
from rag.core.trace import RequestTrace
from rag.generation.pipeline import GroundedGenerator
from rag.generation.types import INSUFFICIENT_CONTEXT
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.types import RetrievedChunk
from rag.routing.pipeline import Router
from rag.serving.app import create_app
from rag.serving.health import HealthReport
from rag.serving.ingest import IngestReport
from rag.serving.logging import LOGGER_NAME, JsonFormatter


def test_health_ok_and_down() -> None:
    up = _client(health=_health(status="ok", points=4, ready=True))
    with TestClient(up) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant"] == "ok"
    assert body["collection_ready"] is True
    assert body["points"] == 4

    down = _client(health=_health(status="down", points=None, ready=False, qdrant="down"))
    with TestClient(down) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "down"


def test_ask_returns_a_cited_answer_and_a_trace() -> None:
    app = _client(retriever=_Retriever([_chunk("a", 0.8)]), llm=_JsonLLM(_cited()))
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            headers={"X-Request-ID": "req-1"},
            json={
                "query": "  How does forking work? ",
                "source": ["hermes/acp-internals.md"],
                "top_k": 2,
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["x-request-id"] == "req-1"
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["query"] == "How does forking work?"
    assert body["route"] == "knowledge_base"
    assert body["sufficient"] is True
    assert body["answer"] == "Forking copies the session [1]."
    assert body["citations"][0]["chunk_id"] == "a"
    assert body["citations"][0]["heading_path"] == ["Forking"]
    assert body["model"] == "scripted"
    names = [span["name"] for span in body["trace"]]
    assert "routing.decision" in names
    assert "text for a" not in response.text


def test_a_low_score_does_not_call_the_model() -> None:
    llm = _JsonLLM(_cited())
    app = _client(retriever=_Retriever([_chunk("a", 0.1)]), llm=llm)
    with TestClient(app) as client:
        response = client.post("/ask", json={"query": "How does forking work?"})
    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "insufficient"
    assert body["reason"] == "below_threshold"
    assert body["answer"] == INSUFFICIENT_CONTEXT
    assert body["citations"] == []
    assert body["model"] is None
    assert body["prompt_tokens"] is None
    assert llm.calls == 0


def test_stream_sends_the_grounded_text_and_not_an_uncited_draft() -> None:
    raw = "Secret uncited fact that must not stream."
    llm = _JsonLLM(json.dumps({"sufficient": True, "answer": raw, "citations": []}))
    app = _client(retriever=_Retriever([_chunk("a", 0.9)]), llm=llm)
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"query": "How does forking work?", "stream": True},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(response.text)
    names = [name for name, _payload in events]
    assert names[0] == "status"
    assert names[-3] == "answer"
    assert names[-2] == "trace"
    assert names[-1] == "done"
    stages = [payload["stage"] for name, payload in events if name == "status"]
    assert stages == ["retrieving", "gate", "generating"]
    deltas = "".join(payload["text"] for name, payload in events if name == "delta")
    answer = next(payload for name, payload in events if name == "answer")
    assert deltas == INSUFFICIENT_CONTEXT
    assert answer["reason"] == "uncited"
    assert answer["answer"] == INSUFFICIENT_CONTEXT
    assert raw not in response.text
    spans = next(payload["spans"] for name, payload in events if name == "trace")
    assert "routing.decision" in [span["name"] for span in spans]


def test_a_stream_error_is_an_event_and_the_log_hides_the_key() -> None:
    secret = "gsk_stream_secret"
    app = _client(
        retriever=_Retriever([_chunk("a", 0.9)]),
        llm=_BrokenLLM(GenerationError(f"failed Bearer {secret}")),
    )
    records = _capture_logs()
    with TestClient(app) as client:
        response = client.post("/ask", json={"query": "unique-zebra-query", "stream": True})
    assert response.status_code == 200
    events = _parse_sse(response.text)
    error = next(payload for name, payload in events if name == "error")
    assert error["error"] == "generation_failed"
    assert secret not in response.text
    assert "unique-zebra-query" not in "\n".join(records)
    assert secret not in "\n".join(records)
    assert "Bearer [REDACTED]" in "\n".join(records)


def test_errors_use_a_fixed_message_and_redact_logs() -> None:
    secret = "gsk_index_secret"
    app = _client(ingest=_raise(IndexingError(f"upsert Bearer {secret}")))
    records = _capture_logs()
    with TestClient(app) as client:
        response = client.post("/ingest")
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "index_failed"
    assert body["message"] == "The index could not be written."
    assert secret not in response.text
    assert secret not in "\n".join(records)
    assert "Bearer [REDACTED]" in "\n".join(records)

    missing = _client(ingest=_raise(FileNotFoundError("corpus directory does not exist")))
    with TestClient(missing) as client:
        response = client.post("/ingest")
    assert response.status_code == 400
    assert response.json()["error"] == "corpus_missing"

    search = _client(retriever=_Retriever([_chunk("a", 0.2)], error=RetrievalError("down")))
    with TestClient(search) as client:
        response = client.post("/ask", json={"query": "How does forking work?"})
    assert response.status_code == 503
    assert response.json()["error"] == "retrieval_failed"


def test_ingest_returns_counts_and_blank_queries_are_rejected() -> None:
    trace = RequestTrace()
    trace.record("ingest.corpus", 1.5, documents=2, chunks=5)
    report = IngestReport(
        documents=2,
        chunks=5,
        orphan_rate=0.0,
        code_split_rate=0.0,
        points=5,
        upserted=5,
        pruned=1,
        schema_recreated=False,
        self_hit_at_1=1.0,
        latency_ms=3.0,
        trace=trace,
    )
    calls = {"ingest": 0}

    def ingest() -> IngestReport:
        calls["ingest"] += 1
        return report

    app = _client(ingest=ingest)
    with TestClient(app) as client:
        response = client.post("/ingest")
        blank = client.post("/ask", json={"query": "  "})
        extra = client.post("/ask", json={"query": "hello", "web": True})
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 2
    assert body["chunks"] == 5
    assert body["pruned"] == 1
    assert body["self_hit_at_1"] == 1.0
    assert body["trace"][0]["name"] == "ingest.corpus"
    assert calls["ingest"] == 1
    assert blank.status_code == 422
    assert extra.status_code == 422


def test_filters_reach_retrieval() -> None:
    retriever = _Retriever([_chunk("a", 0.2)])
    app = _client(retriever=retriever, llm=_JsonLLM(_cited()))
    with TestClient(app) as client:
        client.post(
            "/ask",
            json={
                "query": "How does forking work?",
                "source": ["hermes/acp-internals.md"],
                "doc_title": ["ACP Internals"],
                "chunk_id": ["a"],
                "heading": "Forking",
                "top_n": 8,
                "top_k": 2,
            },
        )
    call = retriever.calls[0]
    metadata = call["metadata_filter"]
    assert isinstance(metadata, MetadataFilter)
    assert metadata.sources == ("hermes/acp-internals.md",)
    assert metadata.doc_titles == ("ACP Internals",)
    assert metadata.chunk_ids == ("a",)
    assert metadata.heading == "Forking"
    assert call["top_n"] == 8
    assert call["top_k"] == 2


def _client(**overrides):
    settings = load_settings()
    retriever = overrides.get("retriever", _Retriever([_chunk("a", 0.8)]))
    llm = overrides.get("llm", _JsonLLM(_cited()))
    router = Router(
        retriever,
        GroundedGenerator(llm, max_context_chunks=5, max_tokens=64),
        sufficiency_threshold=0.35,
    )
    return create_app(
        settings,
        router=router,
        ingest=overrides.get("ingest", _ingest_report),
        health=overrides.get("health", _health()),
    )


def _health(
    *,
    status: Literal["ok", "down"] = "ok",
    points: int | None = 1,
    ready: bool = True,
    qdrant: Literal["ok", "down"] = "ok",
):
    def health() -> HealthReport:
        return HealthReport(
            status=status,
            profile="local",
            qdrant=qdrant,
            collection="knowledge",
            collection_ready=ready,
            points=points,
        )

    return health


def _ingest_report() -> IngestReport:
    return IngestReport(
        documents=1,
        chunks=1,
        orphan_rate=0.0,
        code_split_rate=0.0,
        points=1,
        upserted=1,
        pruned=0,
        schema_recreated=False,
        self_hit_at_1=1.0,
        latency_ms=1.0,
        trace=RequestTrace(),
    )


def _raise(exc: Exception):
    def call():
        raise exc

    return call


def _cited() -> str:
    return json.dumps(
        {"sufficient": True, "answer": "Forking copies the session [1].", "citations": [1]}
    )


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        name = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((name, json.loads(data)))
    return events


def _capture_logs() -> list[str]:
    logger = logging.getLogger(LOGGER_NAME)
    handler = _ListHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return handler.lines


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


class _Retriever:
    def __init__(
        self,
        chunks: list[RetrievedChunk],
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query, *, metadata_filter=None, top_n=None, top_k=None, trace=None):
        self.calls.append(
            {
                "query": query,
                "metadata_filter": metadata_filter,
                "top_n": top_n,
                "top_k": top_k,
            }
        )
        if self._error is not None:
            raise self._error
        if trace is not None:
            trace.record("retrieval.dense", 1.0, count=len(self._chunks))
        return list(self._chunks)


class _JsonLLM(LLMClient):
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        self.calls += 1
        if trace is not None:
            trace.record(
                "llm.complete", 1.0, model="scripted", prompt_tokens=3, completion_tokens=2
            )
        return LLMResponse(
            content=self._content,
            model="scripted",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1.0,
        )


class _BrokenLLM(LLMClient):
    def __init__(self, error: Exception) -> None:
        self._error = error

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        raise self._error

    def stream(self, messages, *, temperature=None, max_tokens=None, trace=None):
        raise self._error
        yield  # pragma: no cover


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk_id,
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text=f"text for {chunk_id}",
        token_count=20,
        index=0,
        oversized=False,
        orphan=False,
        dense_score=0.5,
        dense_rank=1,
        bm25_score=1.0,
        bm25_rank=1,
        rrf_score=0.02,
        rerank_score=score,
    )
