"""Read the raw corpus, parse it, and chunk it.

Phase 2 indexes the chunks this module writes. The default token counter is
the embedding model's tokenizer, because the chunk budget is that model's window.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from rag.chunking.chunker import Chunk, chunk_document
from rag.chunking.stats import ChunkStats, measure_chunks, stats_record
from rag.chunking.tokens import TokenCounter, build_token_counter
from rag.core.config import Settings
from rag.parsing.markdown import ParsedDocument, parse_markdown


@dataclass(frozen=True)
class IngestResult:
    documents: tuple[ParsedDocument, ...]
    chunks: tuple[Chunk, ...]
    stats: ChunkStats


def ingest_directory(
    raw_dir: Path,
    *,
    counter: TokenCounter,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    passage_prefix: str = "",
) -> IngestResult:
    """Parse and chunk every `*.md` file under `raw_dir`."""

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"corpus directory does not exist: {raw_dir}")
    paths = sorted(raw_dir.rglob("*.md"), key=lambda path: path.relative_to(raw_dir).as_posix())
    documents: list[ParsedDocument] = []
    chunks: list[Chunk] = []
    for path in paths:
        source = path.relative_to(raw_dir).as_posix()
        document = parse_markdown(path.read_text(encoding="utf-8"), source)
        documents.append(document)
        chunks.extend(
            chunk_document(
                document,
                counter=counter,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                passage_prefix=passage_prefix,
            )
        )
    stats = measure_chunks(documents, chunks)
    return IngestResult(documents=tuple(documents), chunks=tuple(chunks), stats=stats)


def ingest_corpus(
    raw_dir: Path,
    settings: Settings,
    counter: TokenCounter | None = None,
) -> IngestResult:
    """Ingest with the chunk budget and passage prefix from settings.

    When `counter` is omitted, this loads the embedding tokenizer. That reads
    the local model cache and can download the tokenizer if it is missing.
    """

    token_counter = (
        counter if counter is not None else build_token_counter(settings.embeddings.model)
    )
    return ingest_directory(
        raw_dir,
        counter=token_counter,
        chunk_size_tokens=settings.chunking.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunking.chunk_overlap_tokens,
        passage_prefix=settings.embeddings.passage_prefix,
    )


def chunk_record(chunk: Chunk) -> dict[str, object]:
    """One JSON object for `chunks.jsonl`."""

    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "doc_title": chunk.doc_title,
        "heading_path": list(chunk.heading_path),
        "text": chunk.text,
        "token_count": chunk.token_count,
        "index": chunk.index,
        "oversized": chunk.oversized,
        "orphan": chunk.orphan,
    }


def write_ingest_result(result: IngestResult, directory: Path) -> None:
    """Write `chunks.jsonl` and `chunk_stats.json`. Overwrites both files."""

    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(chunk_record(chunk), ensure_ascii=False) for chunk in result.chunks]
    body = "\n".join(lines)
    if body:
        body += "\n"
    (directory / "chunks.jsonl").write_text(body, encoding="utf-8")
    payload = json.dumps(stats_record(result.stats), indent=2) + "\n"
    (directory / "chunk_stats.json").write_text(payload, encoding="utf-8")
