"""Metadata filters are AND across fields and OR inside one field."""

from rag.indexing.records import ChunkRecord
from rag.retrieval.filters import MetadataFilter


def _record(chunk_id: str, source: str, title: str, heading: tuple[str, ...]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        source=source,
        doc_title=title,
        heading_path=heading,
        text="alpha",
        token_count=1,
        index=0,
        oversized=False,
        orphan=False,
    )


def test_empty_filter_matches_everything() -> None:
    record = _record("a", "a.md", "A", ("A", "Boot"))
    assert MetadataFilter().is_empty()
    assert MetadataFilter().matches(record)


def test_fields_combine_and_values_inside_a_field_are_or() -> None:
    keep = _record("keep", "hermes/a.md", "Alpha", ("Alpha", "Boot"))
    other_source = _record("other", "openai_sdk/b.md", "Alpha", ("Alpha", "Boot"))
    other_heading = _record("heading", "hermes/a.md", "Alpha", ("Alpha", "Other"))
    metadata = MetadataFilter(sources=("hermes/a.md", "hermes/c.md"), heading="Boot")
    assert metadata.matches(keep)
    assert not metadata.matches(other_source)
    assert not metadata.matches(other_heading)


def test_doc_title_and_chunk_id_match_exactly() -> None:
    untitled = _record("openai_sdk/api.md#0000", "openai_sdk/api.md", "", ("Shared Types",))
    titled = _record("hermes/agent-loop.md#0000", "hermes/agent-loop.md", "Agent Loop", ("Intro",))
    assert MetadataFilter(doc_titles=("",)).matches(untitled)
    assert not MetadataFilter(doc_titles=("",)).matches(titled)
    assert MetadataFilter(chunk_ids=("hermes/agent-loop.md#0000",)).matches(titled)
