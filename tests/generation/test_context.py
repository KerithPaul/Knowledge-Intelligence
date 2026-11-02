"""Context blocks are numbered in retrieval order and rendered for the prompt."""

from dataclasses import dataclass

import pytest

from rag.generation.context import assemble_context, render_context
from rag.retrieval.types import RetrievedChunk


def test_blocks_are_numbered_and_cut_at_the_limit() -> None:
    chunks = [
        _chunk("a", "alpha text", rank=1),
        _chunk("b", "beta text", rank=2),
        _chunk("c", "gamma text", rank=3),
    ]
    blocks = assemble_context(chunks, limit=2)
    assert [block.number for block in blocks] == [1, 2]
    assert [block.chunk_id for block in blocks] == ["a", "b"]
    assert blocks[0].text == "alpha text"
    rendered = render_context(blocks)
    assert "[1] source: hermes/acp-internals.md" in rendered
    assert "chunk_id: a" in rendered
    assert "alpha text" in rendered
    assert "gamma text" not in rendered


def test_empty_title_and_heading_use_placeholders() -> None:
    chunk = _chunk("a", "alpha text", title="", heading=())
    rendered = render_context(assemble_context([chunk], limit=5))
    assert "title: (no title)" in rendered
    assert "heading: (no heading)" in rendered


def test_heading_path_is_joined_in_order() -> None:
    chunk = _chunk("a", "alpha text", heading=("ACP", "Forking"))
    rendered = render_context(assemble_context([chunk], limit=1))
    assert "heading: ACP > Forking" in rendered


def test_empty_text_and_bad_heading_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty text"):
        assemble_context([_chunk("a", "  ")], limit=1)
    with pytest.raises(ValueError, match="heading_path"):
        assemble_context([_Heading("a", "alpha", "Forking")], limit=1)


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="limit"):
        assemble_context([], limit=0)


@dataclass
class _Heading:
    chunk_id: str
    text: str
    heading_path: str
    source: str = "doc.md"
    doc_title: str = "Doc"


def _chunk(
    chunk_id: str,
    text: str,
    *,
    rank: int = 1,
    title: str = "ACP Internals",
    heading: tuple[str, ...] = ("Forking",),
) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=chunk_id,
        source="hermes/acp-internals.md",
        doc_title=title,
        heading_path=heading,
        text=text,
        token_count=20,
        index=0,
        oversized=False,
        orphan=False,
        dense_score=0.5,
        dense_rank=rank,
        bm25_score=1.0,
        bm25_rank=rank,
        rrf_score=0.02,
        rerank_score=0.8,
    )
