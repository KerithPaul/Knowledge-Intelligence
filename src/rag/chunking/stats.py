"""Phase 1 measurements: chunk-size distribution, orphan rate, code-split rate.

An orphan chunk has no body, only a heading or a break. A code split is a
fenced block whose full text did not land in any one chunk. Both rates should
stay at zero. Oversized chunks are atomic blocks that were kept whole past
the budget; they are counted, not treated as splits.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rag.chunking.chunker import Chunk
from rag.parsing.markdown import ParsedDocument

_BUCKETS = (
    ("1-128", 1, 128),
    ("129-256", 129, 256),
    ("257-384", 257, 384),
    ("385-512", 385, 512),
    ("513+", 513, None),
)


@dataclass(frozen=True)
class ChunkStats:
    documents: int
    chunks: int
    blocks: int
    heading_blocks: int
    code_blocks: int
    table_blocks: int
    admonition_blocks: int
    token_min: int
    token_max: int
    token_mean: float
    token_p50: int
    token_p95: int
    orphan_chunks: int
    orphan_rate: float
    code_splits: int
    code_split_rate: float
    atomic_splits: int
    atomic_split_rate: float
    oversized_chunks: int
    histogram: tuple[tuple[str, int], ...]


def measure_chunks(documents: list[ParsedDocument], chunks: list[Chunk]) -> ChunkStats:
    """Measure one ingest pass. Atomic blocks are checked per source file."""

    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.source].append(chunk)

    code_blocks = 0
    code_splits = 0
    atomic_blocks = 0
    atomic_splits = 0
    headings = 0
    tables = 0
    admonitions = 0
    block_count = 0
    for document in documents:
        owned = by_source.get(document.source, [])
        for block in document.blocks:
            block_count += 1
            if block.kind == "heading":
                headings += 1
            elif block.kind == "table":
                tables += 1
            elif block.kind == "admonition":
                admonitions += 1
            if not block.atomic or not block.text:
                continue
            atomic_blocks += 1
            found = any(block.text in chunk.text for chunk in owned)
            if block.kind == "code":
                code_blocks += 1
                if not found:
                    code_splits += 1
            if not found:
                atomic_splits += 1

    counts = [chunk.token_count for chunk in chunks]
    ordered = sorted(counts)
    orphans = sum(1 for chunk in chunks if chunk.orphan)
    oversized = sum(1 for chunk in chunks if chunk.oversized)
    histogram = tuple((label, _bucket(counts, low, high)) for label, low, high in _BUCKETS)
    return ChunkStats(
        documents=len(documents),
        chunks=len(chunks),
        blocks=block_count,
        heading_blocks=headings,
        code_blocks=code_blocks,
        table_blocks=tables,
        admonition_blocks=admonitions,
        token_min=ordered[0] if ordered else 0,
        token_max=ordered[-1] if ordered else 0,
        token_mean=(sum(counts) / len(counts)) if counts else 0.0,
        token_p50=_percentile(ordered, 50),
        token_p95=_percentile(ordered, 95),
        orphan_chunks=orphans,
        orphan_rate=_rate(orphans, len(chunks)),
        code_splits=code_splits,
        code_split_rate=_rate(code_splits, code_blocks),
        atomic_splits=atomic_splits,
        atomic_split_rate=_rate(atomic_splits, atomic_blocks),
        oversized_chunks=oversized,
        histogram=histogram,
    )


def format_chunk_stats(stats: ChunkStats) -> str:
    """Plain-text report for a terminal or a saved log."""

    lines = [
        f"documents: {stats.documents}",
        f"blocks: {stats.blocks}",
        f"headings: {stats.heading_blocks}",
        f"code blocks: {stats.code_blocks}",
        f"tables: {stats.table_blocks}",
        f"admonitions: {stats.admonition_blocks}",
        f"chunks: {stats.chunks}",
        (
            "tokens min/p50/p95/max: "
            f"{stats.token_min}/{stats.token_p50}/{stats.token_p95}/{stats.token_max}"
        ),
        f"tokens mean: {stats.token_mean:.1f}",
        f"orphan chunks: {stats.orphan_chunks} ({stats.orphan_rate:.4f})",
        f"code splits: {stats.code_splits} ({stats.code_split_rate:.4f})",
        f"atomic splits: {stats.atomic_splits} ({stats.atomic_split_rate:.4f})",
        f"oversized chunks: {stats.oversized_chunks}",
        "histogram:",
    ]
    for label, count in stats.histogram:
        lines.append(f"  {label}: {count}")
    return "\n".join(lines)


def stats_record(stats: ChunkStats) -> dict[str, Any]:
    """JSON-ready stats. Histogram stays in bucket order."""

    return {
        "documents": stats.documents,
        "chunks": stats.chunks,
        "blocks": stats.blocks,
        "heading_blocks": stats.heading_blocks,
        "code_blocks": stats.code_blocks,
        "table_blocks": stats.table_blocks,
        "admonition_blocks": stats.admonition_blocks,
        "token_min": stats.token_min,
        "token_max": stats.token_max,
        "token_mean": stats.token_mean,
        "token_p50": stats.token_p50,
        "token_p95": stats.token_p95,
        "orphan_chunks": stats.orphan_chunks,
        "orphan_rate": stats.orphan_rate,
        "code_splits": stats.code_splits,
        "code_split_rate": stats.code_split_rate,
        "atomic_splits": stats.atomic_splits,
        "atomic_split_rate": stats.atomic_split_rate,
        "oversized_chunks": stats.oversized_chunks,
        "histogram": {label: count for label, count in stats.histogram},
    }


def _rate(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return part / whole


def _percentile(ordered: list[int], percent: int) -> int:
    if not ordered:
        return 0
    rank = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def _bucket(counts: list[int], low: int, high: int | None) -> int:
    if high is None:
        return sum(1 for count in counts if count >= low)
    return sum(1 for count in counts if low <= count <= high)
