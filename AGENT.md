# AGENT.md — Operating Manual

> Canonical instructions for any AI agent (and human) working in this repo.
> `CLAUDE.md` imports this file; `SOUL.md` explains *why*; `MAP.md` is the file index.
> If these three documents and the code ever disagree, **the code wins** — then fix the doc.

---

## 1. What this project is

A production-grade, **learning-focused RAG system** over a documentation corpus
(Hermes internals + OpenAI SDK docs). It answers **only from the indexed docs**,
and **falls back to the web (Tavily)** when the knowledge base is insufficient.

The authoritative spec is **`IMPLEMENTATION_PLAN.md`** — stack, query flow, module
layout, phased plan, and evaluation methods. Read it before any non-trivial change.

**Current state:** Phase 0, Phase 1, and Phase 2 are done. Python 3.12 venv, config,
`LLMClient` (Groq), `EmbeddingClient` (local BGE), and a Qdrant Compose file are
in place. Groq and BGE smoke tests passed. Phase 1 parses `data/raw` and writes
`data/processed/chunks.jsonl` plus `chunk_stats.json` through `ingest_corpus`
in `src/rag/ingestion.py`. Chunking uses the BGE tokenizer and does not cut
code fences, tables, or admonitions. Phase 2 reads that JSONL, embeds passages,
and upserts Qdrant through `python -m rag.indexing`. Dense vectors are BGE.
Sparse vectors are raw term counts. A second run upserts the same point ids
and deletes chunks that left the corpus. Phase 3 searches that collection:
dense Qdrant search, hand-built BM25, RRF, then `BAAI/bge-reranker-base`.
`python -m rag.retrieval` answers one question. `python -m rag.retrieval --eval`
scores `data/evaluation/retrieval_golden.jsonl`. The API is still empty.

---

## 2. Golden rules (read before writing code)

1. **No hallucinated facts.** If you don't know, look it up with a tool or say so and aslo do not use complex words explain in simple daily used words.
   Never invent file paths, function names, config keys, API shapes, or doc content.
2. **Verify before you assert.** Use CodeGraph (`codegraph_*`) for structural questions
   (what calls what, where a symbol lives, blast radius), native Read/Grep for literal
   text. Don't claim something exists without confirming it.
3. **Plan is the source of truth for scope.** Build in the order and shape defined by
   `IMPLEMENTATION_PLAN.md §4`. Don't jump phases without saying so.
4. **Evaluation is first-class at *every* phase**, not just Phase 6. Any retrieval/
   generation/routing change must come with (or update) a way to measure it.
5. **Build core techniques from first principles, then cross-check** against established
   frameworks (RAGAS/DeepEval/LangChain). The point is to *understand* every method.
6. **Swap-ability is a hard constraint.** LLM, embeddings, and the reranker live
   behind client interfaces (`LLMClient`, `EmbeddingClient`, `Reranker`) so providers
   change in exactly one place.
7. **Secrets never touch code or chat.** Keys come from `.env` (GROQ, TAVILY). Never
   print, log, or echo them; never commit `.env`.
8. **Small, verifiable steps.** Prefer a working slice with a test over a big unproven
   change. State what you ran and what the output was — faithfully, including failures.

---

## 3. Environment & runtime (non-negotiable)

- **Use a dedicated Python 3.12 venv.** Do **not** use system Python 3.15.1 (no ML
  wheels yet) or Hermes' bundled 3.14.9 (unrelated interpreter).
- Platform is **Windows 11 / PowerShell**. A Bash tool is also available for POSIX scripts.
  Prefer the dedicated file/search tools over shell for file ops.
- Tooling: `ruff` (lint/format) + `pytest` (tests) + pre-commit. Config in YAML under
  `configs/` + `.env` for secrets, loaded via `pydantic-settings`.

### Commands
```text
# one-time: install 3.12   uv python install 3.12
# create the venv          uv python find 3.12
#                          that exe -m venv .venv
# activate (PowerShell)    .\.venv\Scripts\Activate.ps1
# activate (cmd)           .\.venv\Scripts\activate.bat
# CPU torch, then the app  .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
#                          .\.venv\Scripts\python.exe -m pip install -e ".[dev]" --upgrade-strategy only-if-needed
# lint                     .\.venv\Scripts\python.exe -m ruff check .
# format                   .\.venv\Scripts\python.exe -m ruff format .
# tests (no network)       .\.venv\Scripts\python.exe -m pytest
# smoke (Groq + BGE)       .\.venv\Scripts\python.exe -m pytest -m smoke -o addopts=
# pre-commit (needs git)   .\.venv\Scripts\pre-commit.exe install
# qdrant                   docker compose up -d qdrant
# index into Qdrant        .\.venv\Scripts\python.exe -m rag.indexing
# search one question      .\.venv\Scripts\python.exe -m rag.retrieval "your question"
# score the retrieval set  .\.venv\Scripts\python.exe -m rag.retrieval --eval
# run API (Phase 7)        not built yet
```
> Keep this block in sync with reality. An agent that can't find the real command
> will guess — that's the failure mode this file exists to prevent.

---

## 4. Where things live

See **`MAP.md`** for the full index. The short version:

- `IMPLEMENTATION_PLAN.md` — the spec. Scope, stack, flow, phases, eval methods.
- `src/rag/` — all application code, one subpackage per RAG stage (see `MAP.md §2`).
- `configs/` — `*.yaml`: chunk sizes, model names, `top_k`, thresholds. No secrets.
- `data/raw/` — the corpus. `hermes/` (9 docs, YAML frontmatter) + `openai_sdk/`
  (4 docs, **plain markdown, no frontmatter** — handle both cases in parsing).
- `data/processed/` — parsed/chunked artifacts (generated; don't hand-edit).
- `data/evaluation/` — `retrieval_golden.jsonl` is the hand-checked Phase 3 set.
  Reports from `--eval` are generated and gitignored. Phase 6 grows the set.
- `tests/` — pytest suite, mirrors `src/rag/` layout.

---

## 5. Working agreement

- **Before coding:** restate which phase/module you're in and what "done" means for it
  (its exit criteria from the plan). Check `MAP.md` so you edit the right file.
- **While coding:** match surrounding style; keep provider-specific code behind the
  client interfaces; add/adjust the test and the eval hook alongside the change.
- **After coding:** run `ruff` + `pytest`; report results honestly. Update `MAP.md` when
  you add a module and this file's command block when you add a command.
- **When unsure about the corpus or an API:** read the actual doc in `data/raw/` or use
  WebSearch/WebFetch — do not guess its contents.
- **Confirm before irreversible/outward actions** (deleting data, network publishes,
  installing heavy deps). Approval for one step is not approval for the next.

---

## 6. Phase checklist (from IMPLEMENTATION_PLAN.md §4)

- [x] **P0 Foundations** — venv, deps, ruff/pytest, config, `LLMClient`+`EmbeddingClient`,
      Docker skeleton. *Exit:* Groq completion + local BGE embedding both smoke-tested.
- [x] **P1 Ingestion** — markdown parse (frontmatter→metadata, heading paths, never split
      code fences) + structure-aware chunking w/ overlap + rich chunk metadata.
      *Exit:* corpus chunks in `data/processed/`, orphan rate 0, code-split rate 0.
- [x] **P2 Indexing** — batch embed w/ query/passage prefixes; Qdrant dense **and** sparse;
      idempotent re-index.
      *Exit:* one point per chunk id, exact self-hit@1 is 1, a second run prunes
      removed chunks. Sparse values are term counts.
- [x] **P3 Retrieval** — dense + hand-built BM25 + RRF + cross-encoder rerank + filters.
      *Exit:* `python -m rag.retrieval --eval` scores `data/evaluation/retrieval_golden.jsonl`
      with Precision@k, Recall@k, Hit@k, MRR, nDCG@k, and MAP. The starter set is
      hand-checked. Phase 6 grows it.
- [ ] **P4 Generation** — grounded context + citations + explicit "insufficient" signal.
- [ ] **P5 Routing** — sufficiency gate (score threshold → LLM groundedness) + Tavily fallback.
- [ ] **P6 Evaluation** — golden set, from-scratch metrics, cross-check vs RAGAS/DeepEval.
- [ ] **P7 Serving** — FastAPI `/ask` `/ingest` `/health`, tracing, Dockerfile+Compose.
- [ ] **P8 Advanced (optional)** — HyDE/multi-query, semantic cache, guardrails, Langfuse.

Tick these as phases complete; they are the contract for "progress."
