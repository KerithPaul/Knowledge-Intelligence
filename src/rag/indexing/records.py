"""Load the chunk records Phase 1 wrote to chunks.jsonl."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChunkRecord:
    """One row of chunks.jsonl. `text` is the passage the embedder will see."""

    chunk_id: str
    source: str
    doc_title: str
    heading_path: tuple[str, ...]
    text: str
    token_count: int
    index: int
    oversized: bool
    orphan: bool


def load_chunk_records(path: Path) -> list[ChunkRecord]:
    """Read chunks.jsonl. Blank lines are skipped. A bad line names its number."""

    if not path.is_file():
        raise FileNotFoundError(
            f"chunk file does not exist: {path}. Run ingestion before indexing."
        )
    records: list[ChunkRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name} line {line_number} is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} line {line_number} must be a JSON object")
        record = _parse_record(raw, line_number=line_number, file_name=path.name)
        if record.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {record.chunk_id} on line {line_number}")
        seen.add(record.chunk_id)
        records.append(record)
    return records


def validate_records(records: Sequence[ChunkRecord]) -> None:
    """Reject ids and text that cannot be upserted as one point each."""

    seen: set[str] = set()
    for record in records:
        if not isinstance(record.chunk_id, str) or not record.chunk_id:
            raise ValueError("chunk_id is empty")
        if record.chunk_id in seen:
            raise ValueError(f"duplicate chunk_id {record.chunk_id}")
        seen.add(record.chunk_id)
        if not isinstance(record.source, str) or not record.source:
            raise ValueError(f"chunk {record.chunk_id} has an empty source")
        if not isinstance(record.text, str) or not record.text.strip():
            raise ValueError(f"chunk {record.chunk_id} has empty text")
        if isinstance(record.token_count, bool) or not isinstance(record.token_count, int):
            raise ValueError(f"chunk {record.chunk_id} has a bad token_count")
        if record.token_count < 0:
            raise ValueError(f"chunk {record.chunk_id} has a negative token_count")
        if isinstance(record.index, bool) or not isinstance(record.index, int) or record.index < 0:
            raise ValueError(f"chunk {record.chunk_id} has a bad index")


def _parse_record(raw: Mapping[str, object], *, line_number: int, file_name: str) -> ChunkRecord:
    def required(name: str) -> object:
        if name not in raw:
            raise ValueError(f"{file_name} line {line_number} is missing {name}")
        return raw[name]

    chunk_id = required("chunk_id")
    source = required("source")
    doc_title = required("doc_title")
    heading_path = required("heading_path")
    text = required("text")
    token_count = required("token_count")
    index = required("index")
    oversized = required("oversized")
    orphan = required("orphan")
    where = f"{file_name} line {line_number}"
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"{where} has an empty chunk_id")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{where} has an empty source")
    if not isinstance(doc_title, str):
        raise ValueError(f"{where} doc_title must be a string")
    if not isinstance(heading_path, list) or not all(
        isinstance(part, str) for part in heading_path
    ):
        raise ValueError(f"{where} heading_path must be a list of strings")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{where} has empty text")
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise ValueError(f"{where} has a bad token_count")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"{where} has a bad index")
    if not isinstance(oversized, bool) or not isinstance(orphan, bool):
        raise ValueError(f"{where} oversized and orphan must be booleans")
    return ChunkRecord(
        chunk_id=chunk_id,
        source=source,
        doc_title=doc_title,
        heading_path=tuple(heading_path),
        text=text,
        token_count=token_count,
        index=index,
        oversized=oversized,
        orphan=orphan,
    )
