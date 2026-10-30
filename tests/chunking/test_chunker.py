"""Chunk packing, overlap, and the rule that atomic blocks stay whole."""

import re

import pytest

from rag.chunking.chunker import chunk_document
from rag.chunking.stats import measure_chunks
from rag.chunking.tokens import WordTokenCounter
from rag.parsing.markdown import parse_markdown


def _words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _chunk(text: str, *, size: int = 10, overlap: int = 3, prefix: str = ""):
    document = parse_markdown(text, source="doc.md")
    return chunk_document(
        document,
        counter=WordTokenCounter(),
        chunk_size_tokens=size,
        chunk_overlap_tokens=overlap,
        passage_prefix=prefix,
    )


def test_short_document_is_one_chunk_with_metadata() -> None:
    text = """---
title: "Architecture"
---

# Architecture

## Data Flow

A short note.
"""
    document = parse_markdown(text, source="hermes/architecture.md")
    chunks = chunk_document(
        document,
        counter=WordTokenCounter(),
        chunk_size_tokens=50,
        chunk_overlap_tokens=5,
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.chunk_id == "hermes/architecture.md#0000"
    assert chunk.source == "hermes/architecture.md"
    assert chunk.doc_title == "Architecture"
    assert chunk.heading_path == ("Architecture",)
    assert "A short note." in chunk.text
    assert chunk.oversized is False
    assert chunk.orphan is False


def test_openai_doc_title_stays_empty() -> None:
    chunks = _chunk("# Security Model\n\nCaller-owned client.\n", size=40, overlap=4)
    assert len(chunks) == 1
    assert chunks[0].doc_title == ""
    assert chunks[0].heading_path == ("Security Model",)


def test_prose_overlap_repeats_the_tail() -> None:
    text = _words(30)
    chunks = _chunk(text, size=10, overlap=3)
    assert len(chunks) >= 3
    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert chunks[1].text.startswith("w7 w8 w9")
    words = re.findall(r"\S+", text)
    for word in words:
        assert any(word in chunk.text for chunk in chunks)


def test_zero_overlap_does_not_repeat_words() -> None:
    chunks = _chunk(_words(25), size=10, overlap=0)
    assert [chunk.token_count for chunk in chunks] == [10, 10, 5]
    assert "w9" in chunks[0].text
    assert "w9" not in chunks[1].text
    assert chunks[1].text.startswith("w10")


def test_code_fence_bigger_than_the_budget_is_not_split() -> None:
    fence = "```python\n" + "\n".join(f"line{index} = {index}" for index in range(30)) + "\n```"
    text = f"# Title\n\nIntro words here.\n\n{fence}\n\nAfter the fence.\n"
    chunks = _chunk(text, size=20, overlap=4)
    code_chunks = [chunk for chunk in chunks if "line0 = 0" in chunk.text]
    assert len(code_chunks) == 1
    assert fence in code_chunks[0].text
    assert code_chunks[0].oversized is True
    after = next(chunk for chunk in chunks if "After the fence." in chunk.text)
    assert "line15 = 15" not in after.text
    document = parse_markdown(text, source="doc.md")
    stats = measure_chunks([document], chunks)
    assert stats.code_splits == 0
    assert stats.atomic_splits == 0


def test_heading_stays_with_the_following_paragraph() -> None:
    text = f"{_words(10)}\n\n## Next\n\nbody words here now\n"
    chunks = _chunk(text, size=12, overlap=2)
    owner = next(chunk for chunk in chunks if "## Next" in chunk.text)
    assert "body words here now" in owner.text
    assert all(not chunk.text.endswith("## Next") for chunk in chunks)
    assert all(not chunk.orphan for chunk in chunks)


def test_table_and_admonition_stay_intact() -> None:
    text = """# Docs

| Name | Role |
| --- | --- |
| alpha | one |
| beta | two |
| gamma | three |
| delta | four |

:::tip Keep this together
Do not cut this note in half just to hit the budget.
The note is short but the budget below is shorter.
:::
"""
    chunks = _chunk(text, size=8, overlap=2)
    table = next(chunk for chunk in chunks if "| alpha | one |" in chunk.text)
    assert "| delta | four |" in table.text
    note = next(chunk for chunk in chunks if ":::tip Keep this together" in chunk.text)
    assert note.text.strip().endswith(":::")
    document = parse_markdown(text, source="doc.md")
    stats = measure_chunks([document], chunks)
    assert stats.atomic_splits == 0
    assert stats.orphan_chunks == 0


def test_passage_prefix_uses_part_of_the_budget() -> None:
    prefix = "INS INS INS INS "
    chunks = _chunk(_words(10), size=10, overlap=0, prefix=prefix)
    assert chunks[0].token_count <= 10
    body_words = re.findall(r"\S+", chunks[0].text)
    assert len(body_words) <= 6


def test_overlap_must_be_smaller_than_the_chunk() -> None:
    document = parse_markdown("Hello world.\n", source="doc.md")
    with pytest.raises(ValueError, match="overlap"):
        chunk_document(
            document,
            counter=WordTokenCounter(),
            chunk_size_tokens=10,
            chunk_overlap_tokens=10,
        )


def test_empty_document_yields_no_chunks() -> None:
    document = parse_markdown("", source="empty.md")
    chunks = chunk_document(
        document,
        counter=WordTokenCounter(),
        chunk_size_tokens=32,
        chunk_overlap_tokens=4,
    )
    assert chunks == []


class _PlusTwo:
    """Word count plus two special tokens, so sums are not the joined count."""

    def count(self, text: str) -> int:
        if not text:
            return 2
        return 2 + len(re.findall(r"\S+", text))


def test_budget_uses_the_joined_string_not_the_sum_of_parts() -> None:
    document = parse_markdown(_words(20), source="doc.md")
    chunks = chunk_document(
        document,
        counter=_PlusTwo(),
        chunk_size_tokens=8,
        chunk_overlap_tokens=2,
    )
    assert chunks
    assert all(chunk.token_count <= 8 for chunk in chunks)
    assert all(chunk.oversized is False for chunk in chunks)
