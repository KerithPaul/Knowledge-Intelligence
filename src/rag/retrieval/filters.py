"""Metadata filters shared by dense search and BM25.

Several values in one field match if any of them match. Different fields
all have to match. An empty filter keeps every chunk.

`source` is the path under data/raw, such as ``hermes/acp-internals.md``.
`heading` matches one segment of ``heading_path`` exactly.
"""

from dataclasses import dataclass

from qdrant_client import models

from rag.indexing.records import ChunkRecord


@dataclass(frozen=True)
class MetadataFilter:
    """Narrow the chunks both retrievers are allowed to return."""

    sources: tuple[str, ...] = ()
    doc_titles: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    heading: str | None = None

    def is_empty(self) -> bool:
        return (
            not self.sources and not self.doc_titles and not self.chunk_ids and self.heading is None
        )

    def matches(self, record: ChunkRecord) -> bool:
        """True when this chunk is allowed through."""

        if self.sources and record.source not in self.sources:
            return False
        if self.doc_titles and record.doc_title not in self.doc_titles:
            return False
        if self.chunk_ids and record.chunk_id not in self.chunk_ids:
            return False
        if self.heading is not None and self.heading not in record.heading_path:
            return False
        return True


def to_qdrant_filter(metadata: MetadataFilter | None) -> models.Filter | None:
    """The same predicate as `MetadataFilter.matches`, for Qdrant payload fields."""

    if metadata is None or metadata.is_empty():
        return None
    must: list[models.Condition] = []
    if metadata.sources:
        must.append(
            models.FieldCondition(
                key="source",
                match=models.MatchAny(any=list(metadata.sources)),
            )
        )
    if metadata.doc_titles:
        must.append(
            models.FieldCondition(
                key="doc_title",
                match=models.MatchAny(any=list(metadata.doc_titles)),
            )
        )
    if metadata.chunk_ids:
        must.append(
            models.FieldCondition(
                key="chunk_id",
                match=models.MatchAny(any=list(metadata.chunk_ids)),
            )
        )
    if metadata.heading is not None:
        # heading_path is stored as a list. Qdrant matches when any element equals.
        must.append(
            models.FieldCondition(
                key="heading_path",
                match=models.MatchValue(value=metadata.heading),
            )
        )
    return models.Filter(must=must)
