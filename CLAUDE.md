# CLAUDE.md

This file is auto-loaded by Claude Code every session. The real content lives in
the harness documents below — read them before doing non-trivial work.

@AGENT.md

## Harness map
- **@AGENT.md** — operating manual: golden rules, environment, working agreement, phase checklist. **(canonical)**
- **SOUL.md** — the *why*: learning-first philosophy and non-negotiable principles.
- **MAP.md** — file index + module layout + end-to-end query flow. Consult before editing to avoid guessing.
- **IMPLEMENTATION_PLAN.md** — the spec. Stack, flow, phased plan, evaluation methods.

## Session reflexes
- **Don't hallucinate.** Verify file paths, symbols, and API shapes with tools (CodeGraph for structure, Read/Grep for text, WebSearch for external APIs) before asserting them.
- **Use Python 3.12 venv** — never system 3.15.1 or Hermes' 3.14.9.
- **Secrets stay in `.env`** — never print, log, or commit them.
- When you add a module, update MAP.md; when you add a command, update AGENT.md §3.
