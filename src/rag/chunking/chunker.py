"""Pack parsed blocks into retrieval chunks.

Code fences, tables, and admonitions are never cut. A block that is bigger
than the budget is its own chunk and is marked oversized. Prose splits on a
token boundary. The next chunk repeats a tail of the previous one, up to the
overlap, and that tail does not slice an atomic block. A heading that would
land at the end of a full chunk moves onto the chunk that holds the text it
introduces.
"""

from dataclasses import dataclass

from rag.chunking.tokens import TokenCounter
from rag.parsing.markdown import ParsedDocument

_SPLITTABLE = frozenset({"paragraph", "list", "blockquote"})
_CONTEXT = frozenset({"heading", "thematic_break"})
_BODY = frozenset({"paragraph", "code", "table", "admonition", "list", "blockquote"})


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit. `heading_path` is the path in effect at the start."""

    chunk_id: str
    source: str
    doc_title: str
    heading_path: tuple[str, ...]
    text: str
    token_count: int
    index: int
    oversized: bool
    orphan: bool


@dataclass(frozen=True)
class _Segment:
    text: str
    kind: str
    splittable: bool
    heading_path: tuple[str, ...]


def chunk_document(
    document: ParsedDocument,
    *,
    counter: TokenCounter,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    passage_prefix: str = "",
) -> list[Chunk]:
    """Split one parsed document. Token counts include `passage_prefix`."""

    packer = _Packer(
        counter=counter,
        chunk_size=chunk_size_tokens,
        overlap=chunk_overlap_tokens,
        passage_prefix=passage_prefix,
    )
    return packer.pack(document)


class _Packer:
    def __init__(
        self,
        *,
        counter: TokenCounter,
        chunk_size: int,
        overlap: int,
        passage_prefix: str,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        self.counter = counter
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.passage_prefix = passage_prefix

    def pack(self, document: ParsedDocument) -> list[Chunk]:
        segments = [
            _Segment(
                text=block.text,
                kind=block.kind,
                splittable=block.kind in _SPLITTABLE,
                heading_path=block.heading_path,
            )
            for block in document.blocks
            if block.text.strip()
        ]
        chunks: list[Chunk] = []
        carry: list[_Segment] = []
        pos = 0
        guard = max(64, sum(len(segment.text) for segment in segments) + 1)
        steps = 0
        while pos < len(segments):
            steps += 1
            if steps > guard:
                raise RuntimeError(f"chunker stopped making progress on {document.source}")
            carry_text = _join(carry)
            current: list[_Segment] = list(carry) if carry and self._fits(carry_text) else []
            consumed = False
            while pos < len(segments):
                segment = segments[pos]
                if self._fits(_join(current + [segment])):
                    current.append(segment)
                    pos += 1
                    consumed = True
                    continue
                if segment.splittable:
                    split = self._take_prefix(current, segment)
                    if split is not None:
                        prefix_text, rest_text = split
                        current.append(_replace(segment, prefix_text))
                        if rest_text:
                            segments[pos] = _replace(segment, rest_text)
                        else:
                            pos += 1
                        consumed = True
                        break
                    if current and not _context_only(current):
                        _shift_trailing_context(current, segments, pos)
                        break
                    if not current:
                        current.append(segment)
                        pos += 1
                        consumed = True
                        break
                    break
                if not current or _context_only(current):
                    current.append(segment)
                    pos += 1
                    consumed = True
                    break
                _shift_trailing_context(current, segments, pos)
                break

            grew = consumed or (_join(current) != carry_text and bool(current))
            if not grew:
                placed, pos = self._force_one(segments, pos)
                current = placed
            text = _join(current)
            if text.strip():
                chunks.append(self._make_chunk(document, len(chunks), current, text))
            if pos < len(segments) and self.overlap > 0:
                carry = self._overlap(current)
            else:
                carry = []
        return chunks

    def _make_chunk(
        self,
        document: ParsedDocument,
        index: int,
        segments: list[_Segment],
        text: str,
    ) -> Chunk:
        count = self._count(text)
        path = segments[0].heading_path if segments else ()
        return Chunk(
            chunk_id=f"{document.source}#{index:04d}",
            source=document.source,
            doc_title=document.title,
            heading_path=path,
            text=text,
            token_count=count,
            index=index,
            oversized=count > self.chunk_size,
            orphan=not any(segment.kind in _BODY for segment in segments),
        )

    def _prepared(self, text: str) -> str:
        prefix = self.passage_prefix
        if not prefix or text.startswith(prefix):
            return text
        return f"{prefix}{text}"

    def _count(self, text: str) -> int:
        return self.counter.count(self._prepared(text))

    def _fits(self, text: str) -> bool:
        if not text.strip():
            return True
        return self._count(text) <= self.chunk_size

    def _prefix_fits(self, current: list[_Segment], prefix: str) -> bool:
        if not prefix.strip():
            return False
        return self._fits(_join(current + [_Segment(prefix, "paragraph", True, ())]))

    def _take_prefix(self, current: list[_Segment], segment: _Segment) -> tuple[str, str] | None:
        text = segment.text
        best = 0
        low = 1
        high = len(text)
        while low <= high:
            mid = (low + high) // 2
            if self._prefix_fits(current, text[:mid]):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best <= 0:
            return None
        end = _snap_prefix_end(text, best)
        candidate = text[:end].rstrip(" \t")
        if not self._prefix_fits(current, candidate):
            end = best
            candidate = text[:end].rstrip(" \t")
        if not candidate or not self._prefix_fits(current, candidate):
            return None
        rest = text[end:].lstrip(" \t\n")
        if rest == text:
            return None
        return candidate, rest

    def _overlap(self, segments: list[_Segment]) -> list[_Segment]:
        if self.overlap <= 0 or not segments:
            return []
        chosen: list[_Segment] = []
        for segment in reversed(segments):
            trial = [segment, *chosen]
            if self._count(_join(trial)) <= self.overlap:
                chosen = trial
                continue
            if segment.splittable:
                suffix = self._suffix(segment.text, chosen)
                if suffix:
                    chosen = [_replace(segment, suffix), *chosen]
            break
        return chosen

    def _suffix(self, text: str, following: list[_Segment]) -> str:
        best = len(text)
        low = 0
        high = len(text) - 1
        while low <= high:
            mid = (low + high) // 2
            suffix = text[mid:]
            trial = _join([_Segment(suffix, "paragraph", True, ()), *following])
            if suffix.strip() and self._count(trial) <= self.overlap:
                best = mid
                high = mid - 1
            else:
                low = mid + 1
        if best >= len(text):
            return ""
        start = _snap_suffix_start(text, best)
        suffix = text[start:].lstrip(" \t\n")
        if suffix and self._suffix_fits(suffix, following):
            return suffix
        fallback = text[best:].lstrip(" \t\n")
        if fallback and self._suffix_fits(fallback, following):
            return fallback
        return ""

    def _suffix_fits(self, suffix: str, following: list[_Segment]) -> bool:
        trial = _join([_Segment(suffix, "paragraph", True, ()), *following])
        return self._count(trial) <= self.overlap

    def _force_one(self, segments: list[_Segment], pos: int) -> tuple[list[_Segment], int]:
        segment = segments[pos]
        if segment.splittable:
            split = self._take_prefix([], segment)
            if split is not None:
                prefix_text, rest_text = split
                if rest_text and rest_text != segment.text:
                    segments[pos] = _replace(segment, rest_text)
                    return [_replace(segment, prefix_text)], pos
                if not rest_text:
                    return [_replace(segment, prefix_text)], pos + 1
        return [segment], pos + 1


def _join(segments: list[_Segment]) -> str:
    return "\n\n".join(segment.text for segment in segments if segment.text)


def _replace(segment: _Segment, text: str) -> _Segment:
    return _Segment(
        text=text,
        kind=segment.kind,
        splittable=segment.splittable,
        heading_path=segment.heading_path,
    )


def _context_only(segments: list[_Segment]) -> bool:
    return bool(segments) and all(segment.kind in _CONTEXT for segment in segments)


def _shift_trailing_context(current: list[_Segment], segments: list[_Segment], pos: int) -> None:
    """Keep a heading with the block that follows it."""

    moved: list[_Segment] = []
    while len(current) > 1 and current[-1].kind in _CONTEXT:
        moved.append(current.pop())
    if not moved:
        return
    moved.reverse()
    segments[pos:pos] = moved


def _snap_prefix_end(text: str, best: int) -> int:
    if best <= 0 or best > len(text):
        return max(0, best)
    if best == len(text) or text[best - 1].isspace():
        return best
    cut = max(text.rfind(" ", 0, best), text.rfind("\n", 0, best))
    if cut > 0 and best - cut <= 48:
        return cut
    return best


def _snap_suffix_start(text: str, start: int) -> int:
    if start <= 0 or start >= len(text):
        return start
    if text[start - 1].isspace() or text[start].isspace():
        return start
    window = text[start : start + 48]
    for offset, char in enumerate(window):
        if char.isspace():
            return start + offset
    return start
