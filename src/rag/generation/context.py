"""Copy retrieved chunks into numbered context blocks."""

from collections.abc import Sequence

from rag.generation.types import ContextBlock, ContextSource


def assemble_context(
    chunks: Sequence[ContextSource],
    *,
    limit: int,
) -> tuple[ContextBlock, ...]:
    """Take the first `limit` chunks, in the order given.

    Retrieval already returns the best chunk first. Numbers start at 1 and
    are the only citation ids the model is allowed to use.
    """

    if limit <= 0:
        raise ValueError("limit must be positive")
    blocks: list[ContextBlock] = []
    for chunk in chunks:
        if len(blocks) >= limit:
            break
        blocks.append(_block(chunk, number=len(blocks) + 1))
    return tuple(blocks)


def render_context(blocks: Sequence[ContextBlock]) -> str:
    """The context section of the user message. An empty list renders as ''."""

    parts: list[str] = []
    for block in blocks:
        title = block.doc_title if block.doc_title else "(no title)"
        if block.heading_path:
            heading = " > ".join(block.heading_path)
        else:
            heading = "(no heading)"
        parts.append(
            f"[{block.number}] source: {block.source}\n"
            f"title: {title}\n"
            f"heading: {heading}\n"
            f"chunk_id: {block.chunk_id}\n"
            f"\n"
            f"{block.text}"
        )
    return "\n\n".join(parts)


def _block(chunk: ContextSource, *, number: int) -> ContextBlock:
    chunk_id = chunk.chunk_id
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("chunk_id is empty")
    source = chunk.source
    if not isinstance(source, str) or not source:
        raise ValueError(f"chunk {chunk_id} has an empty source")
    title = chunk.doc_title
    if not isinstance(title, str):
        raise ValueError(f"chunk {chunk_id} has a bad doc_title")
    heading = chunk.heading_path
    if isinstance(heading, str) or not isinstance(heading, Sequence):
        raise ValueError(f"chunk {chunk_id} heading_path must be a sequence of strings")
    if not all(isinstance(part, str) for part in heading):
        raise ValueError(f"chunk {chunk_id} heading_path must be a sequence of strings")
    heading_path = tuple(heading)
    text = chunk.text
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"chunk {chunk_id} has empty text")
    return ContextBlock(
        number=number,
        chunk_id=chunk_id,
        source=source,
        doc_title=title,
        heading_path=heading_path,
        text=text.strip(),
    )
