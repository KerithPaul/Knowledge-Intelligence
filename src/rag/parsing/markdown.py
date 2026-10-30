"""Parse markdown into blocks.

Frontmatter is metadata, and only when the file opens with a closing --- pair.
A later --- line is a break in the body, or text inside a code fence.
Heading lines update the heading path. Code fences, tables, and admonitions
stay whole so the chunker can refuse to cut them.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

import yaml

BlockKind = Literal[
    "heading",
    "paragraph",
    "code",
    "table",
    "admonition",
    "list",
    "blockquote",
    "thematic_break",
]

_ATOMIC_KINDS = frozenset({"code", "table", "admonition"})

_FENCE_OPEN_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})[ \t]*(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*#*\s*$")
_ADMONITION_OPEN_RE = re.compile(
    r"^:::(tip|note|warning|caution|danger|info)(?:[ \t]+(.*))?$",
    re.IGNORECASE,
)
_ADMONITION_CLOSE_RE = re.compile(r"^:::[ \t]*$")
_THEMATIC_RE = re.compile(r"^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})[ \t]*$")
_LIST_ITEM_RE = re.compile(r"^ {0,3}(?:[-+*]|\d{1,9}[.)])[ \t]+\S")
_BLOCKQUOTE_RE = re.compile(r"^ {0,3}>")
_TABLE_DELIM_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$"
)


class ParseError(ValueError):
    """The file is not markdown this parser can read. The message names the source."""


@dataclass(frozen=True)
class Block:
    """One region of a document. `text` is the source lines, joined by newlines."""

    kind: BlockKind
    text: str
    heading_path: tuple[str, ...]
    atomic: bool
    language: str = ""
    level: int = 0


@dataclass(frozen=True)
class ParsedDocument:
    """A parsed file. `title` comes from frontmatter only, never from the first heading."""

    source: str
    title: str
    metadata: dict[str, Any]
    blocks: tuple[Block, ...]


def parse_markdown(text: str, source: str) -> ParsedDocument:
    """Parse one markdown document.

    `source` is the corpus-relative path, such as `hermes/architecture.md`.
    """

    if not source.strip():
        raise ValueError("source must be a non-empty path")
    normalized = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    metadata, body_start = _frontmatter(lines, source)
    raw_title = metadata.get("title", "")
    title = raw_title if isinstance(raw_title, str) else ""
    blocks = tuple(_blocks(lines, body_start))
    return ParsedDocument(source=source, title=title, metadata=metadata, blocks=blocks)


def _frontmatter(lines: list[str], source: str) -> tuple[dict[str, Any], int]:
    if not lines or lines[0].strip() != "---":
        return {}, 0
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        raise ParseError(f"{source}: opening --- has no closing ---")
    raw = "\n".join(lines[1:end])
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ParseError(f"{source}: frontmatter is not valid YAML") from exc
    if loaded is None:
        return {}, end + 1
    if not isinstance(loaded, dict):
        raise ParseError(f"{source}: frontmatter must be a mapping")
    return dict(loaded), end + 1


def _blocks(lines: list[str], start: int) -> list[Block]:
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []
    index = start
    total = len(lines)
    while index < total:
        if not lines[index].strip():
            index += 1
            continue
        if _fence_open(lines[index]) is not None:
            end = _consume_fence(lines, index)
            kind_line = lines[index]
            language = _fence_language(kind_line)
            blocks.append(_block("code", lines, index, end, _path(stack), language=language))
            index = end
            continue
        if _ADMONITION_OPEN_RE.match(lines[index]):
            end = _consume_admonition(lines, index)
            blocks.append(_block("admonition", lines, index, end, _path(stack)))
            index = end
            continue
        heading = _HEADING_RE.match(lines[index])
        if heading is not None:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            blocks.append(_block("heading", lines, index, index + 1, _path(stack), level=level))
            index += 1
            continue
        if _THEMATIC_RE.match(lines[index]):
            blocks.append(_block("thematic_break", lines, index, index + 1, _path(stack)))
            index += 1
            continue
        if _is_table_start(lines, index):
            end = _consume_table(lines, index)
            blocks.append(_block("table", lines, index, end, _path(stack)))
            index = end
            continue
        if _LIST_ITEM_RE.match(lines[index]):
            end = _consume_list(lines, index)
            blocks.append(_block("list", lines, index, end, _path(stack)))
            index = end
            continue
        if _BLOCKQUOTE_RE.match(lines[index]):
            end = _consume_while(lines, index, _BLOCKQUOTE_RE.match)
            blocks.append(_block("blockquote", lines, index, end, _path(stack)))
            index = end
            continue
        end = _consume_paragraph(lines, index)
        blocks.append(_block("paragraph", lines, index, end, _path(stack)))
        index = end
    return blocks


def _block(
    kind: BlockKind,
    lines: list[str],
    start: int,
    end: int,
    heading_path: tuple[str, ...],
    *,
    language: str = "",
    level: int = 0,
) -> Block:
    text = "\n".join(lines[start:end])
    return Block(
        kind=kind,
        text=text,
        heading_path=heading_path,
        atomic=kind in _ATOMIC_KINDS,
        language=language,
        level=level,
    )


def _path(stack: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(title for _, title in stack)


def _fence_open(line: str) -> re.Match[str] | None:
    return _FENCE_OPEN_RE.match(line)


def _fence_language(line: str) -> str:
    match = _fence_open(line)
    if match is None:
        return ""
    info = match.group(3).strip()
    if not info:
        return ""
    return info.split()[0]


def _is_fence_close(line: str, char: str, min_len: int) -> bool:
    pattern = rf"^ {{0,3}}{re.escape(char)}{{{min_len},}}[ \t]*$"
    return re.match(pattern, line) is not None


def _consume_fence(lines: list[str], start: int) -> int:
    match = _fence_open(lines[start])
    if match is None:
        return start + 1
    marker = match.group(2)
    index = start + 1
    while index < len(lines):
        if _is_fence_close(lines[index], marker[0], len(marker)):
            return index + 1
        index += 1
    return index


def _consume_admonition(lines: list[str], start: int) -> int:
    index = start + 1
    while index < len(lines):
        if _ADMONITION_CLOSE_RE.match(lines[index]):
            return index + 1
        index += 1
    return index


def _is_table_delimiter(line: str) -> bool:
    return "|" in line and _TABLE_DELIM_RE.match(line) is not None


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index]
    if "|" not in header or not header.strip():
        return False
    if _fence_open(header) or _HEADING_RE.match(header) or _ADMONITION_OPEN_RE.match(header):
        return False
    return _is_table_delimiter(lines[index + 1])


def _consume_table(lines: list[str], start: int) -> int:
    index = start + 2
    while index < len(lines):
        line = lines[index]
        if not line.strip() or "|" not in line:
            break
        if _fence_open(line) or _HEADING_RE.match(line) or _ADMONITION_OPEN_RE.match(line):
            break
        index += 1
    return index


def _starts_structure(lines: list[str], index: int) -> bool:
    line = lines[index]
    if not line.strip():
        return False
    if _fence_open(line) or _ADMONITION_OPEN_RE.match(line):
        return True
    if _HEADING_RE.match(line) or _THEMATIC_RE.match(line):
        return True
    if _LIST_ITEM_RE.match(line) or _BLOCKQUOTE_RE.match(line):
        return True
    return _is_table_start(lines, index)


def _consume_paragraph(lines: list[str], start: int) -> int:
    index = start + 1
    while index < len(lines):
        if not lines[index].strip() or _starts_structure(lines, index):
            break
        index += 1
    return index


def _consume_list(lines: list[str], start: int) -> int:
    index = start + 1
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            nxt = index + 1
            if nxt < len(lines) and (
                _LIST_ITEM_RE.match(lines[nxt])
                or (lines[nxt].startswith("  ") and not _starts_structure(lines, nxt))
            ):
                index += 1
                continue
            break
        if _LIST_ITEM_RE.match(line):
            index += 1
            continue
        if line.startswith("  ") and not _starts_structure(lines, index):
            index += 1
            continue
        break
    return index


def _consume_while(lines: list[str], start: int, predicate: Any) -> int:
    index = start + 1
    while index < len(lines) and predicate(lines[index]):
        index += 1
    return index
