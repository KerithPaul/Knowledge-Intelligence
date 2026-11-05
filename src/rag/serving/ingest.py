"""Read the corpus, write chunks, and refresh the Qdrant collection.

`/ingest` does both steps. Writing chunks without indexing would leave
`/ask` on the previous collection.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from rag.core.config import PROJECT_ROOT, Settings
from rag.core.trace import RequestTrace
from rag.indexing.indexer import IndexStats, index_processed
from rag.ingestion import IngestResult, ingest_corpus, write_ingest_result


@dataclass(frozen=True)
class IngestReport:
    """Counts from one ingest-and-index run. The trace has no chunk text."""

    documents: int
    chunks: int
    orphan_rate: float
    code_split_rate: float
    points: int
    upserted: int
    pruned: int
    schema_recreated: bool
    self_hit_at_1: float | None
    latency_ms: float
    trace: RequestTrace


def run_ingest(
    settings: Settings,
    *,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> IngestReport:
    """Ingest `data/raw` and index `data/processed` unless paths are passed."""

    raw = raw_dir if raw_dir is not None else PROJECT_ROOT / "data" / "raw"
    processed = processed_dir if processed_dir is not None else PROJECT_ROOT / "data" / "processed"
    started = time.perf_counter()
    trace = RequestTrace()
    corpus_started = time.perf_counter()
    result = ingest_corpus(raw, settings)
    trace.record(
        "ingest.corpus",
        (time.perf_counter() - corpus_started) * 1000,
        documents=len(result.documents),
        chunks=len(result.chunks),
        orphan_rate=result.stats.orphan_rate,
        code_split_rate=result.stats.code_split_rate,
    )
    write_ingest_result(result, processed)
    stats = index_processed(processed, settings, trace=trace)
    return _report(result, stats, trace, latency_ms=(time.perf_counter() - started) * 1000)


def _report(
    result: IngestResult,
    stats: IndexStats,
    trace: RequestTrace,
    *,
    latency_ms: float,
) -> IngestReport:
    return IngestReport(
        documents=len(result.documents),
        chunks=len(result.chunks),
        orphan_rate=result.stats.orphan_rate,
        code_split_rate=result.stats.code_split_rate,
        points=stats.points,
        upserted=stats.upserted,
        pruned=stats.pruned,
        schema_recreated=stats.schema_recreated,
        self_hit_at_1=stats.self_hit_at_1,
        latency_ms=latency_ms,
        trace=trace,
    )
