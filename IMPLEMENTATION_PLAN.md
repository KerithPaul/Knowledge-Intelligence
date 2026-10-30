# Knowledge Intelligence — Implementation Plan

> A production-grade, learning-focused RAG system over a documentation corpus
> (Hermes internals + OpenAI SDK docs). Answers **from the indexed docs only**,
> and **falls back to the web (Tavily)** when the knowledge base can't answer.

**Status:** Phase 0 through Phase 3 are in the tree. Phases 4–8 are not started.
**Created:** 2026-10-26
**Primary goal:** Understand *every method* at each RAG stage — with special
emphasis on **evaluation** and **production-grade implementation**. Core
techniques are built from first principles, then cross-checked against
established frameworks.

---

## 1. Locked Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Generation** | Groq API (OpenAI-compatible), behind a swappable `LLMClient` | Free tier + very fast → survives heavy eval loops; swap models/providers in one place |
| **Embeddings** | Local `bge-small-en-v1.5` (sentence-transformers), swappable `EmbeddingClient` | Free, offline, unlimited re-indexing; upgrade to `bge-base` later |
| **Reranker** | Local cross-encoder `bge-reranker-base` | Teaches the retrieve → rerank production pattern |
| **Vector store** | **Qdrant** (Docker) | Dense + sparse (hybrid) in one engine, payload filtering, fits Docker target |
| **Keyword search** | Hand-built BM25, fused with dense via **Reciprocal Rank Fusion (RRF)** | Learn sparse vs dense and fusion from first principles |
| **Web fallback** | Tavily + content fetch/clean | LLM-optimized search API |
| **Serving** | FastAPI + Docker Compose (app + Qdrant) | Canonical production shape |
| **Evaluation** | From-scratch metrics, cross-checked against RAGAS/DeepEval | The centerpiece of the learning goal |
| **Runtime** | Dedicated **Python 3.12** venv | System Python 3.15.1 has no ML wheels yet; Hermes runs on its own bundled 3.14.9 (unrelated) |

> **Open objection point:** Qdrant vs. a simpler FAISS/Chroma start. Default is
> Qdrant; can begin FAISS-first and defer Docker to Phase 7 if preferred.

---

## 2. End-to-End Query Flow

```text
query
  → embed (BGE)  ──────────────┐
  → dense search (Qdrant)      │→ RRF fusion → top-N
  → BM25 search ───────────────┘
      → cross-encoder rerank → top-k
          → SUFFICIENCY GATE (retrieval score + LLM groundedness check)
              ├─ sufficient → generate grounded answer + citations (Groq)
              └─ insufficient → Tavily web search → fetch/clean → generate → label source=web
  → trace object (chunks, scores, routing decision, tokens, latency) → logged
```

---

## 3. Module Layout

Maps onto the existing `src/rag/` scaffold; `(new)` marks subpackages to add.

```text
src/rag/
  parsing/     markdown-aware parsing, frontmatter→metadata, heading paths
  chunking/    structure-aware + token chunking, code-fence-safe
  embeddings/  EmbeddingClient (BGE) + batching/prefixing
  retrieval/   dense, bm25, hybrid(RRF), rerank, filters
  generation/  LLMClient (Groq), prompt assembly, citations, grounding
  routing/     sufficiency gate + Tavily web fallback            (new)
  evaluation/  datasets, retrieval metrics, gen metrics, runners, reports
  serving/     FastAPI app, schemas, tracing/logging             (new)
  core/        config (pydantic-settings), trace model, interfaces (new)
configs/       *.yaml — chunk sizes, model names, top_k, thresholds
data/raw|processed|evaluation
tests/
```

---

## 4. Phased Build Plan

Evaluation is a first-class concern at **every** phase, not only Phase 6.

### Phase 0 — Foundations
- Python 3.12 venv, dependency management, `ruff` + `pytest` + pre-commit.
- `pydantic-settings` config loaded from YAML + `.env` (GROQ/TAVILY keys).
- `LLMClient` / `EmbeddingClient` interfaces + Groq and BGE implementations.
- Docker Compose skeleton.
- **Exit criteria:** working Groq completion + local BGE embedding, both smoke-tested.

### Phase 1 — Ingestion
- Markdown-aware parsing: keep frontmatter as metadata, preserve heading
  hierarchy, never split code fences.
- Structure-aware chunking with token overlap.
- Rich chunk metadata: source, doc title, heading-path, chunk id.
- **Learn:** chunking strategies & tradeoffs. **Measure:** chunk-size
  distribution, orphan/code-split rate.

### Phase 2 — Indexing
- Batch-embed with correct query/passage prefixing.
- Qdrant collection with dense **and** sparse vectors; idempotent re-index.
- **Learn:** ANN index internals, dense vs sparse.

### Phase 3 — Retrieval
- Dense + BM25 + RRF fusion + cross-encoder rerank + metadata filters.
- **Learn:** hybrid retrieval & reranking. **Measure:** retrieval metrics (§5)
  on `data/evaluation/retrieval_golden.jsonl` (`python -m rag.retrieval --eval`).
  That file is a hand-checked starter. Phase 6 grows it into the full golden set.

### Phase 4 — Generation & Grounding
- Context assembly with citations; structured answer that can emit an explicit
  "insufficient context" signal; anti-hallucination prompting.
- **Learn:** grounded prompting & citation. **Measure:** faithfulness, answer
  relevance.

### Phase 5 — Routing & Web Fallback
- Two-stage sufficiency gate: cheap retrieval-score threshold → LLM groundedness
  check.
- Tavily search → fetch + clean → re-ground on web content → source labeling.
- **Learn:** adaptive/corrective RAG. **Measure:** router accuracy,
  false-fallback & missed-fallback rates.

### Phase 6 — Evaluation Harness (centerpiece)
- Synthetic golden Q/A generated from the corpus + hand-written "needs-web"
  questions (to test routing).
- From-scratch metrics (§5); then cross-check against RAGAS/DeepEval and
  reconcile divergences.
- Eval CLI + reports written to `data/evaluation/`; regression gating.

### Phase 7 — Serving & Production
- FastAPI: `/ask` (streaming), `/ingest`, `/health`.
- Per-request tracing/structured logs, error handling.
- Dockerfile + Compose (app + Qdrant); config profiles.

### Phase 8 — Advanced (optional)
- Query transforms (HyDE, multi-query, decomposition), semantic cache,
  guardrails, Langfuse tracing, CI eval-gate.

---

## 5. Evaluation Methods (built by hand, then validated)

- **Retrieval:** Precision@k, Recall@k, Hit@k, MRR, nDCG@k, MAP.
- **Generation:** Faithfulness/groundedness (claim-level LLM-judge), Answer
  Relevance, Context Precision, Context Recall, Answer Correctness vs golden.
- **Routing:** KB-vs-web decision accuracy, false-fallback & missed-fallback rates.
- **Operational:** per-stage latency, tokens/cost, end-to-end win-rate.
- **Cross-check:** same data through RAGAS/DeepEval → explain any divergence.

---

## 6. Prerequisites to Start Phase 0

1. **Groq API key** and **Tavily API key** → wired via `.env` + config (values
   pasted by the user, never revealed to the assistant).
2. **Docker Desktop** running for Qdrant — or decision to start **FAISS-first**
   and defer Docker to Phase 7.
3. Confirm **Qdrant** vs FAISS/Chroma, and greenlight starting Phase 0.

---

## 7. Notes & Constraints

- Build RAG app in a **Python 3.12** venv — do **not** use system Python 3.15.1
  (ML wheels unavailable) or Hermes' bundled 3.14.9.
- Keep LLM and embedding behind **client interfaces** so providers swap in one
  place (Groq today; subscription/other providers possible later).
- Corpus is technical markdown with a clear heading hierarchy and fenced code
  blocks. Hermes docs open with YAML frontmatter. OpenAI SDK docs do not.
  `hermes/adding-platform-adapters.md` has frontmatter and no `title`. Parsing
  must handle both shapes.
