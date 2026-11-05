"""Ingest wiring, the Qdrant probe, and the Compose profile."""

from types import SimpleNamespace

import yaml

from rag.core.config import PROJECT_ROOT, load_settings
from rag.indexing.indexer import IndexStats
from rag.serving.__main__ import _parser
from rag.serving.health import check_health
from rag.serving.ingest import run_ingest


def test_run_ingest_writes_chunks_before_it_indexes(tmp_path, monkeypatch) -> None:
    settings = load_settings()
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    order: list[str] = []

    def fake_ingest(raw_dir, got_settings, counter=None):
        assert raw_dir == raw
        assert got_settings is settings
        assert counter is None
        order.append("ingest")
        return SimpleNamespace(
            documents=(1, 2),
            chunks=(1, 2, 3),
            stats=SimpleNamespace(orphan_rate=0.0, code_split_rate=0.0),
        )

    def fake_write(result, directory):
        assert directory == processed
        assert len(result.chunks) == 3
        order.append("write")

    def fake_index(directory, got_settings, embedder=None, client=None, trace=None):
        assert directory == processed
        assert got_settings is settings
        assert trace is not None
        trace.record("indexing.upsert", 2.0, points=3)
        order.append("index")
        return _stats()

    monkeypatch.setattr("rag.serving.ingest.ingest_corpus", fake_ingest)
    monkeypatch.setattr("rag.serving.ingest.write_ingest_result", fake_write)
    monkeypatch.setattr("rag.serving.ingest.index_processed", fake_index)
    report = run_ingest(settings, raw_dir=raw, processed_dir=processed)
    assert order == ["ingest", "write", "index"]
    assert report.documents == 2
    assert report.chunks == 3
    assert report.points == 3
    assert report.pruned == 1
    assert report.self_hit_at_1 == 1.0
    assert [span.name for span in report.trace.spans] == ["ingest.corpus", "indexing.upsert"]


def test_health_probe_uses_the_collection_and_hides_failures() -> None:
    settings = load_settings()
    name = settings.qdrant.collection
    ready = check_health(settings, client=_Qdrant(name=name, exists=True, points=6))
    assert ready.status == "ok"
    assert ready.collection_ready is True
    assert ready.points == 6
    assert ready.profile == settings.serving.profile

    missing = check_health(settings, client=_Qdrant(name=name, exists=False, points=0))
    assert missing.status == "ok"
    assert missing.collection_ready is False
    assert missing.points is None

    down = check_health(
        settings, client=_Qdrant(name=name, exists=True, points=1, error=RuntimeError("nope"))
    )
    assert down.status == "down"
    assert down.qdrant == "down"
    assert down.points is None
    assert "nope" not in str(down)


def test_compose_profile_runs_the_app_beside_qdrant() -> None:
    document = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = document["services"]
    assert "qdrant" in services
    app = services["app"]
    assert app["profiles"] == ["docker"]
    assert "qdrant" in app["depends_on"]
    assert app["env_file"] == [".env", "configs/profiles/docker.env"]
    profile = (PROJECT_ROOT / "configs" / "profiles" / "docker.env").read_text(encoding="utf-8")
    assert "QDRANT__URL=http://qdrant:6333" in profile
    assert "SERVING__HOST=0.0.0.0" in profile
    assert "SERVING__PROFILE=docker" in profile
    assert "GROQ" not in profile
    assert "API_KEY" not in profile
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.12" in dockerfile
    assert "download.pytorch.org/whl/cpu" in dockerfile
    assert "COPY .env" not in dockerfile
    assert "-e ." in dockerfile


def test_cli_defaults_come_from_settings() -> None:
    args = _parser("127.0.0.1", 8000).parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    custom = _parser("127.0.0.1", 8000).parse_args(["--host", "0.0.0.0", "--port", "9000"])
    assert custom.host == "0.0.0.0"
    assert custom.port == 9000


class _Qdrant:
    def __init__(
        self,
        *,
        name: str,
        exists: bool,
        points: int,
        error: Exception | None = None,
    ) -> None:
        self._name = name
        self._exists = exists
        self._points = points
        self._error = error

    def collection_exists(self, name: str) -> bool:
        if self._error is not None:
            raise self._error
        assert name == self._name
        return self._exists

    def count(self, collection: str, exact: bool = False):
        assert collection == self._name
        assert exact is True
        return SimpleNamespace(count=self._points)


def _stats() -> IndexStats:
    return IndexStats(
        collection="knowledge",
        embedding_model="BAAI/bge-small-en-v1.5",
        points=3,
        upserted=3,
        pruned=1,
        schema_recreated=False,
        dense_dimension=384,
        distance="cosine",
        vocabulary_size=10,
        sparse_nnz_min=1,
        sparse_nnz_median=2.0,
        sparse_nnz_max=4,
        oversized_chunks=0,
        truncated_chunks=0,
        self_hit_at_1=1.0,
        self_hit_at_1_hnsw=None,
        indexed_vectors=6,
        search_mode="server",
    )
