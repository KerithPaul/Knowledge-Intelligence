"""Search the indexed corpus, or score the retrieval golden set.

Qdrant must already be running and the corpus indexed:
    docker compose up -d qdrant
    python -m rag.indexing

Examples:
    python -m rag.retrieval "How does ACP forking work?"
    python -m rag.retrieval --source hermes/acp-internals.md "How does cancel work?"
    python -m rag.retrieval --eval
"""

import argparse

from rag.core.config import get_settings
from rag.evaluation.golden import DEFAULT_GOLDEN, DEFAULT_REPORT, load_retrieval_cases
from rag.evaluation.retrieval import (
    QueryScore,
    RetrievalReport,
    score_queries,
    write_retrieval_report,
)
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.pipeline import open_retriever
from rag.retrieval.types import RetrievedChunk


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.eval and args.query:
        raise SystemExit("pass a query or --eval, not both")
    if not args.eval and not args.query:
        raise SystemExit("pass a query or --eval")
    settings = get_settings()
    retriever = open_retriever(settings)
    if args.eval:
        cases = load_retrieval_cases(DEFAULT_GOLDEN)
        rows = []
        for case in cases:
            hits = retriever.retrieve(case.query)
            rows.append(
                (
                    case.query,
                    [hit.chunk_id for hit in hits],
                    set(case.relevant_chunk_ids),
                )
            )
        report = score_queries(rows, k=settings.retrieval.top_k)
        write_retrieval_report(report, DEFAULT_REPORT)
        _print_report(report)
        return
    metadata = MetadataFilter(
        sources=tuple(args.source),
        doc_titles=tuple(args.doc_title),
        chunk_ids=tuple(args.chunk_id),
        heading=args.heading,
    )
    hits = retriever.retrieve(
        args.query,
        metadata_filter=None if metadata.is_empty() else metadata,
        top_n=args.top_n,
        top_k=args.top_k,
    )
    if not hits:
        print("no chunks")
        return
    for hit in hits:
        _print_hit(hit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.retrieval",
        description="Search the indexed documentation corpus.",
    )
    parser.add_argument("query", nargs="?", help="Question to search for.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Score data/evaluation/retrieval_golden.jsonl and write a report.",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Keep chunks from this source path. Repeat to allow several.",
    )
    parser.add_argument(
        "--doc-title",
        action="append",
        default=[],
        help="Keep chunks with this doc title. Repeat to allow several.",
    )
    parser.add_argument(
        "--chunk-id",
        action="append",
        default=[],
        help="Keep this chunk id. Repeat to allow several.",
    )
    parser.add_argument(
        "--heading",
        default=None,
        help="Keep chunks whose heading path contains this exact segment.",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Candidate pool before rerank.")
    parser.add_argument("--top-k", type=int, default=None, help="How many chunks to print.")
    return parser


def _print_hit(hit: RetrievedChunk) -> None:
    heading = " > ".join(hit.heading_path) if hit.heading_path else hit.source
    print(
        f"{hit.rank}. {hit.chunk_id}  "
        f"rerank={hit.rerank_score:.3f} rrf={hit.rrf_score:.4f} "
        f"dense={_score(hit.dense_score)} bm25={_score(hit.bm25_score)}"
    )
    print(f"   {heading}")
    print(f"   {_preview(hit.text)}")


def _print_report(report: RetrievalReport) -> None:
    k = report.k
    print(f"retrieval golden set: {report.queries} queries at k={k}")
    print(f"precision@{k} {report.precision_at_k:.3f}")
    print(f"recall@{k} {report.recall_at_k:.3f}")
    print(f"hit@{k} {report.hit_at_k:.3f}")
    print(f"mrr {report.mrr:.3f}")
    print(f"ndcg@{k} {report.ndcg_at_k:.3f}")
    print(f"map {report.mean_average_precision:.3f}")
    misses = [row for row in report.per_query if row.hit_at_k == 0]
    for row in misses:
        _print_miss(row)
    print(f"wrote {DEFAULT_REPORT}")


def _print_miss(row: QueryScore) -> None:
    retrieved = ", ".join(row.retrieved) if row.retrieved else "(none)"
    relevant = ", ".join(row.relevant)
    print(f"miss: {row.query}")
    print(f"  retrieved: {retrieved}")
    print(f"  relevant: {relevant}")


def _score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _preview(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "..."


if __name__ == "__main__":
    main()
