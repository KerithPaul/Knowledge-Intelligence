"""Parser behavior on the markdown shapes in this corpus."""

import pytest

from rag.parsing.markdown import ParseError, parse_markdown


def _lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line.strip()]


def _block_lines(source: str) -> list[str]:
    document = parse_markdown(source, source="fixture.md")
    lines: list[str] = []
    for block in document.blocks:
        lines.extend(line for line in block.text.split("\n") if line.strip())
    return lines


def test_frontmatter_is_metadata_and_not_a_heading() -> None:
    source = """---
sidebar_position: 1
title: "Architecture"
description: "A map"
---

# Architecture

Intro paragraph.
"""
    document = parse_markdown(source, source="hermes/architecture.md")
    assert document.title == "Architecture"
    assert document.metadata["sidebar_position"] == 1
    assert document.metadata["description"] == "A map"
    assert document.blocks[0].kind == "heading"
    assert document.blocks[0].text == "# Architecture"
    assert document.blocks[0].heading_path == ("Architecture",)
    assert "sidebar_position" not in document.blocks[0].text


def test_missing_frontmatter_leaves_the_title_empty() -> None:
    source = "# Security Model\n\nThe body.\n"
    document = parse_markdown(source, source="openai_sdk/security-model.md")
    assert document.title == ""
    assert document.metadata == {}
    assert document.blocks[0].heading_path == ("Security Model",)


def test_body_divider_is_not_frontmatter() -> None:
    source = """---
title: "Skills"
---

# Skills

Before the break.

---

## After
"""
    document = parse_markdown(source, source="hermes/creating-skills.md")
    assert document.title == "Skills"
    kinds = [block.kind for block in document.blocks]
    assert kinds == ["heading", "paragraph", "thematic_break", "heading"]
    assert document.blocks[-1].heading_path == ("Skills", "After")


def test_heading_path_resets_at_a_sibling() -> None:
    source = "# Top\n\n## Child\n\n### Deep\n\n## Other\n"
    document = parse_markdown(source, source="doc.md")
    paths = [block.heading_path for block in document.blocks if block.kind == "heading"]
    assert paths == [("Top",), ("Top", "Child"), ("Top", "Child", "Deep"), ("Top", "Other")]


def test_code_fence_hides_headings_and_frontmatter_markers() -> None:
    source = """# Real

```markdown
---
name: my-skill
---

# Skill Title
```

## Also real
"""
    document = parse_markdown(source, source="hermes/creating-skills.md")
    headings = [block.text for block in document.blocks if block.kind == "heading"]
    assert headings == ["# Real", "## Also real"]
    code = next(block for block in document.blocks if block.kind == "code")
    assert code.atomic is True
    assert code.language == "markdown"
    assert "# Skill Title" in code.text
    assert "---" in code.text
    assert code.heading_path == ("Real",)


def test_mermaid_fence_stays_one_code_block() -> None:
    source = """# Security Model

```mermaid
flowchart LR
  app --> sdk
```
"""
    document = parse_markdown(source, source="openai_sdk/security-model.md")
    code = next(block for block in document.blocks if block.kind == "code")
    assert code.language == "mermaid"
    assert code.atomic is True
    assert "flowchart LR" in code.text
    assert code.text.startswith("```mermaid")
    assert code.text.endswith("```")


def test_unclosed_fence_runs_to_the_end() -> None:
    source = "# Title\n\n```python\nprint('hi')\n"
    document = parse_markdown(source, source="doc.md")
    code = document.blocks[-1]
    assert code.kind == "code"
    assert "print('hi')" in code.text
    assert code.language == "python"


def test_admonition_with_a_title_stays_whole() -> None:
    source = """# Tools

:::warning Built-in Core Tools Only
This page is for adding a tool.

- plugins instead
:::

After.
"""
    document = parse_markdown(source, source="hermes/adding-tools.md")
    admonition = next(block for block in document.blocks if block.kind == "admonition")
    assert admonition.atomic is True
    assert admonition.text.startswith(":::warning Built-in Core Tools Only")
    assert admonition.text.endswith(":::")
    assert "- plugins instead" in admonition.text
    assert document.blocks[-1].kind == "paragraph"


def test_table_is_one_block() -> None:
    source = """# Models

| API mode | Used for |
|----------|----------|
| `chat_completions` | OpenAI-compatible endpoints |

After the table.
"""
    document = parse_markdown(source, source="hermes/agent-loop.md")
    table = next(block for block in document.blocks if block.kind == "table")
    assert table.atomic is True
    assert "`chat_completions`" in table.text
    assert document.blocks[-1].text == "After the table."


def test_every_nonblank_line_is_kept_in_order() -> None:
    source = """---
title: "Architecture"
---

# Architecture

Intro.

## Directory Structure

```text
hermes-agent/
```

- one
- two

> a quote
"""
    document = parse_markdown(source, source="hermes/architecture.md")
    got: list[str] = []
    for block in document.blocks:
        got.extend(line for line in block.text.split("\n") if line.strip())
    body = source.split("---", 2)[-1]
    assert got == _lines(body)
    assert document.title == "Architecture"


def test_unclosed_frontmatter_is_an_error() -> None:
    with pytest.raises(ParseError, match="no closing"):
        parse_markdown("---\ntitle: broken\n# Heading\n", source="broken.md")


def test_frontmatter_must_be_a_mapping() -> None:
    source = "---\n- just\n- a list\n---\n\n# Title\n"
    with pytest.raises(ParseError, match="mapping"):
        parse_markdown(source, source="broken.md")


def test_empty_file_has_no_blocks_and_no_title() -> None:
    document = parse_markdown("", source="empty.md")
    assert document.title == ""
    assert document.blocks == ()


def test_nonempty_source_is_required() -> None:
    with pytest.raises(ValueError, match="source"):
        parse_markdown("# Hi\n", source="  ")
