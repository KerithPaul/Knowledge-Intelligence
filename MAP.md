# MAP.md — Repository Map

> Navigation index so the agent edits the right file instead of guessing.
> Update this whenever you add/move a module. `(empty)` = scaffolded, no code yet.

## 1. Top level

```text
knowledge-intelligence/
├── IMPLEMENTATION_PLAN.md   ← THE SPEC: stack, query flow, phases, eval methods
├── AGENT.md                 ← operating manual (canonical harness doc)
├── CLAUDE.md                ← auto-loaded; imports AGENT.md
├── SOUL.md                  ← philosophy / non-negotiable principles
├── MAP.md                   ← this file
├── pyproject.toml           ← package + ruff + pytest
├── docker-compose.yml       ← Qdrant only (app service arrives in Phase 7)
├── .env.example             ← secret names, no values
├── .pre-commit-config.yaml
├── configs/
│   └── settings.yaml        ← tunables (models, chunk size, top_k, thresholds). No secrets.
├── data/
│   ├── raw/                 ← the corpus (source of truth for answers)
│   │   ├── hermes/          ← 9 docs, YAML frontmatter (title/description/sidebar_position)
│   │   └── openai_sdk/      ← 4 docs, PLAIN markdown, NO frontmatter  ⚠ parse both cases
│   ├── processed/           ← chunks.jsonl, chunk_stats.json, index_stats.json (generated)
│   └── evaluation/          ← retrieval_golden.jsonl and routing_golden.jsonl (hand-checked). --eval reports are generated.
├── src/rag/                 ← application code (see §2)
└── tests/                   ← pytest suite, mirrors src/rag/
```

## 2. `src/rag/` module layout (planned — IMPLEMENTATION_PLAN.md §3)

| Subpackage | Responsibility | Phase |
|---|---|---|
| `core/` | `config.py` (YAML + `.env`), `interfaces.py`, `clients.py` (the provider switch), `trace.py`, `errors.py` | P0 |
| `ingestion.py` | `ingest_corpus` reads `data/raw`, chunks, and can write `data/processed` | P1 |
| `parsing/` | `parsing/markdown.py` — frontmatter, heading paths, fenced blocks | P1 |
| `chunking/` | `chunking/chunker.py`, `tokens.py`, `stats.py` — pack blocks and measure splits | P1 |
| `embeddings/` | `bge.py` — `BGEEmbeddingClient` (query prefix, passage prefix, batch size) | P0; indexing calls it |
| `indexing/` | `indexer.py`, `store.py`, `sparse.py`, `records.py` — batch-embed and upsert Qdrant | P2 |
| `retrieval/` | `pipeline.py`, `dense.py`, `bm25.py`, `fusion.py`, `rerank.py`, `filters.py` — dense search, BM25, RRF, rerank | P3 |
| `generation/` | `groq.py`, `pipeline.py`, `context.py`, `prompt.py`, `parse.py`, `types.py` — Groq client, numbered context, citations, insufficient-context signal | P0 client; P4 grounding |
| `routing/` | `pipeline.py`, `types.py` — best rerank score, then the grounded answer. No web search | P5 |
| `evaluation/` | `retrieval.py`, `golden.py`, `generation.py` — retrieval metrics, the Phase 3 question file, faithfulness, and answer relevance. Context precision, context recall, and answer correctness remain P6 | P3 retrieval; P4 generation scores; P6 runners |
| `serving/` *(new)* | FastAPI app (`/ask` `/ingest` `/health`), schemas, tracing/logging | P7 |

> When you create one of these, replace its row's phase note with the key entry-point
> file(s), e.g. `core/config.py`, so the next agent lands directly on the code.

## 3. End-to-end query flow (IMPLEMENTATION_PLAN.md §2)

```text
query
  → embed (BGE) ──────────────┐
  → dense search (Qdrant)     │→ RRF fusion → top-N
  → BM25 search ──────────────┘
      → cross-encoder rerank → top-k
          → SUFFICIENCY GATE (best rerank probability, then the grounded answer)
              ├─ score below threshold → insufficient-context sentence (no model call)
              └─ score passes → grounded answer + citations (Groq)
                    └─ model says not enough, or cites nothing → same sentence
  → trace (chunks, scores, routing decision, tokens, latency) → logged
```

## 4. Key facts an agent needs (and often gets wrong)

- **Corpus is mixed-format**: Hermes docs have YAML frontmatter; OpenAI SDK docs don't.
  Parsing must not assume frontmatter exists. The doc title is the frontmatter
  `title` string only. `hermes/adding-platform-adapters.md` has frontmatter and
  no title, so its doc title is empty. The first heading is not the title.
- **Chunk budget is the embedding tokenizer**, special tokens included. Code
  fences, tables, and admonitions are not cut to fit. A block past 512 tokens
  stays whole and is marked oversized. The latest counts are in
  `data/processed/chunk_stats.json`.
- **Two client interfaces are the seams**: `LLMClient` (Groq today) and `EmbeddingClient`
  (local BGE). Provider swaps happen only behind these.
- **Qdrant holds a dense vector and a sparse vector on each point.** Dense is BGE.
  Sparse is a raw term count from `indexing/sparse.py`. BM25 is hand-built in
  `retrieval/bm25.py` from those same tokens and fused with dense search by RRF.
  The Qdrant IDF modifier stays off. `python -m rag.retrieval` reranks the fused
  list with `BAAI/bge-reranker-base`. Each score is a probability after sigmoid.
- **Filters** narrow both lists: `source`, `doc_title`, `chunk_id`, and one
  exact `heading_path` segment. Values inside one field are OR. Fields are AND.
- **Retrieval questions** live in `data/evaluation/retrieval_golden.jsonl`.
  Each row has an `evidence` string that must appear only in the labeled chunks.
  `python -m rag.retrieval --eval` writes `retrieval_report.json` beside it.
- **Grounded answers** cite numbered context blocks. `sufficient` is false when
  retrieval returns nothing, when the model says the context is not enough, or
  when a claimed answer cites no real block. Those three cases use one fixed
  sentence and do not keep the model's extra text. `--score` reports
  faithfulness and answer relevance. Both questions in answer relevance are
  embedded with the query prefix.
- **The ask path** is `python -m rag.routing`. It calls the model only when
  the best rerank probability is at least `routing.sufficiency_threshold`
  (settings.yaml: 0.35). Otherwise it returns the same insufficient-context
  sentence. A passing score still returns that sentence when the grounded
  answer says the context is not enough. `python -m rag.generation` does
  not apply this score gate. Web search is not called. `--eval` scores
  `routing_golden.jsonl`: accuracy, false-fallback, and missed-fallback.
- **HNSW thresholds in `configs/settings.yaml` are kilobytes.** Full scan is
  `10` because the server rejects anything smaller, and that is still under
  this corpus, so it builds a graph. Qdrant's own defaults would brute-force it.
  `python -m rag.indexing` writes `data/processed/index_stats.json`, including
  exact self-hit@1. Local Qdrant has no HNSW graph, so that run leaves the
  HNSW self-hit empty.
- **Runtime is Python 3.12 venv** — not system 3.15.1, not Hermes' 3.14.9.
- **`configs/*.yaml` = tunables, `.env` = secrets.** Never mix them.
