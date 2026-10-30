"""Structural checks on the real corpus. Token counts here are words, not BGE tokens."""

from pathlib import Path

from rag.chunking.tokens import WordTokenCounter
from rag.core.config import PROJECT_ROOT
from rag.ingestion import ingest_directory

RAW = PROJECT_ROOT / "data" / "raw"
SIZE = 512
OVERLAP = 64


def _body_lines(text: str) -> list[str]:
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    start = 0
    if lines and lines[0].strip() == "---":
        for index in range(1, len(lines)):
            if lines[index].strip() == "---":
                start = index + 1
                break
    return [line for line in lines[start:] if line.strip()]


def test_corpus_parses_and_chunks_without_splits() -> None:
    result = ingest_directory(
        RAW,
        counter=WordTokenCounter(),
        chunk_size_tokens=SIZE,
        chunk_overlap_tokens=OVERLAP,
    )
    stats = result.stats
    assert stats.documents == 13
    assert stats.code_blocks > 0
    assert stats.table_blocks >= 10
    assert stats.admonition_blocks == 11
    assert stats.code_splits == 0
    assert stats.code_split_rate == 0
    assert stats.atomic_splits == 0
    assert stats.orphan_chunks == 0

    by_source = {document.source: document for document in result.documents}
    assert by_source["hermes/architecture.md"].title == "Architecture"
    assert by_source["openai_sdk/security-model.md"].title == ""
    assert by_source["openai_sdk/api.md"].title == ""
    hermes = [document for document in result.documents if document.source.startswith("hermes/")]
    openai = [
        document for document in result.documents if document.source.startswith("openai_sdk/")
    ]
    assert len(hermes) == 9
    assert len(openai) == 4
    # This file has frontmatter, but no title field. The H1 is not the title.
    untitled = by_source["hermes/adding-platform-adapters.md"]
    assert untitled.title == ""
    assert untitled.blocks[0].text == "# Adding a Platform Adapter"
    titled = [document for document in hermes if document.source != untitled.source]
    assert len(titled) == 8
    assert all(document.title for document in titled)
    assert all(document.title == "" for document in openai)

    skills = by_source["hermes/creating-skills.md"]
    assert not any(
        block.kind == "heading" and block.text == "# Skill Title" for block in skills.blocks
    )
    fenced = next(
        block for block in skills.blocks if block.kind == "code" and "# Skill Title" in block.text
    )
    assert fenced.atomic is True

    architecture = by_source["hermes/architecture.md"]
    cli = next(
        block
        for block in architecture.blocks
        if block.kind == "heading" and block.text == "### CLI Session"
    )
    assert cli.heading_path == ("Architecture", "Data Flow", "CLI Session")

    mermaid = [
        block
        for document in result.documents
        for block in document.blocks
        if block.kind == "code" and block.language == "mermaid"
    ]
    assert len(mermaid) == 1

    chunks_by_source: dict[str, list] = {}
    for chunk in result.chunks:
        chunks_by_source.setdefault(chunk.source, []).append(chunk)

    for path in sorted(RAW.rglob("*.md")):
        source = path.relative_to(RAW).as_posix()
        document = by_source[source]
        got: list[str] = []
        for block in document.blocks:
            got.extend(line for line in block.text.split("\n") if line.strip())
        assert got == _body_lines(path.read_text(encoding="utf-8")), source

        owned = chunks_by_source[source]
        assert [chunk.index for chunk in owned] == list(range(len(owned)))
        assert len({chunk.chunk_id for chunk in owned}) == len(owned)
        for chunk in owned:
            assert chunk.text.strip()
            assert chunk.token_count == WordTokenCounter().count(chunk.text)
            assert chunk.doc_title == document.title
            if chunk.oversized:
                assert chunk.token_count > SIZE
                assert any(block.atomic and block.text in chunk.text for block in document.blocks)
            else:
                assert chunk.token_count <= SIZE

    assert stats.chunks == sum(len(owned) for owned in chunks_by_source.values())
    assert Path(RAW).is_dir()
