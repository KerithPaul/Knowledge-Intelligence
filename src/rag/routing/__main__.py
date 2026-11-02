"""Ask one question through the sufficiency gate, or score the routing set.

Qdrant must already be running and the corpus indexed. ``--eval`` also
calls Groq for every question whose best rerank score reaches the threshold.

    docker compose up -d qdrant
    python -m rag.indexing

Examples:
    python -m rag.routing "How does ACP forking work?"
    python -m rag.routing --source hermes/acp-internals.md "How does cancel work?"
    python -m rag.routing --eval
"""

import argparse

from rag.core.config import get_settings
from rag.core.errors import GenerationError
from rag.evaluation.routing import (
    DEFAULT_ROUTING_GOLDEN,
    DEFAULT_ROUTING_REPORT,
    RoutingObservation,
    RoutingReport,
    load_routing_cases,
    score_routing,
    write_routing_report,
)
from rag.generation.types import Citation
from rag.retrieval.filters import MetadataFilter
from rag.routing.pipeline import open_router
from rag.routing.types import RoutedAnswer


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.eval and args.query:
        raise SystemExit("pass a query or --eval, not both")
    if not args.eval and (args.query is None or not str(args.query).strip()):
        raise SystemExit("pass a query or --eval")
    settings = get_settings()
    router = open_router(settings)
    if args.eval:
        cases = load_routing_cases(DEFAULT_ROUTING_GOLDEN)
        rows: list[RoutingObservation] = []
        for case in cases:
            try:
                routed = router.ask(case.query)
            except GenerationError as exc:
                raise SystemExit(f"could not read the model reply: {exc}") from None
            rows.append(_observation(case.query, case.corpus_can_answer, routed))
        report = score_routing(rows, threshold=settings.routing.sufficiency_threshold)
        write_routing_report(report, DEFAULT_ROUTING_REPORT)
        _print_report(report)
        return
    metadata = MetadataFilter(
        sources=tuple(args.source),
        doc_titles=tuple(args.doc_title),
        chunk_ids=tuple(args.chunk_id),
        heading=args.heading,
    )
    try:
        routed = router.ask(
            args.query,
            metadata_filter=None if metadata.is_empty() else metadata,
            top_n=args.top_n,
            top_k=args.top_k,
        )
    except GenerationError as exc:
        raise SystemExit(f"could not read the model reply: {exc}") from None
    _print_routed(routed)


def _observation(query: str, corpus_can_answer: bool, routed: RoutedAnswer) -> RoutingObservation:
    return RoutingObservation(
        query=query,
        corpus_can_answer=corpus_can_answer,
        route=routed.route,
        decided_by=routed.decided_by,
        reason=routed.reason,
        best_rerank_score=routed.best_rerank_score,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.routing",
        description=(
            "Answer a question when the indexed docs look sufficient. "
            "A weak retrieval score returns the insufficient-context sentence "
            "without calling the model."
        ),
    )
    parser.add_argument("query", nargs="?", help="Question to ask.")
    parser.add_argument(
        "--eval",
        action="store_true",
        help=(
            "Score data/evaluation/routing_golden.jsonl and write a report. "
            "Questions that pass the score gate call the model."
        ),
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="How many chunks to retrieve. The prompt keeps at most generation.max_context_chunks.",
    )
    return parser


def _print_routed(routed: RoutedAnswer) -> None:
    print(f"route: {routed.route}")
    print(f"decided_by: {routed.decided_by}")
    if routed.reason is None:
        print("reason: -")
    else:
        print(f"reason: {routed.reason}")
    print(f"best_rerank_score: {_score(routed.best_rerank_score)}")
    print(f"threshold: {routed.threshold:.3f}")
    if routed.answer is not None and routed.answer.sufficient:
        print("sufficient: yes")
    else:
        print("sufficient: no")
    citations = () if routed.answer is None else routed.answer.citations
    if not citations:
        print("citations: none")
    for citation in citations:
        _print_citation(citation)
    model = "-"
    if routed.answer is not None and routed.answer.model:
        model = routed.answer.model
    print(f"model: {model}")
    print()
    print(routed.text)


def _print_citation(citation: Citation) -> None:
    if citation.heading_path:
        heading = " > ".join(citation.heading_path)
    else:
        heading = citation.source
    print(f"[{citation.number}] {citation.chunk_id}  {heading}")


def _print_report(report: RoutingReport) -> None:
    print(f"routing golden set: {report.queries} queries")
    print(f"threshold: {report.threshold:.3f}")
    print(f"accuracy: {report.accuracy:.3f}")
    print(
        "false_fallback: "
        f"{_rate(report.false_fallback_rate)} "
        f"({report.false_fallbacks}/{report.answerable} answerable refused)"
    )
    print(
        "missed_fallback: "
        f"{_rate(report.missed_fallback_rate)} "
        f"({report.missed_fallbacks}/{report.unanswerable} unanswerable answered)"
    )
    for row in report.per_query:
        if row.correct:
            continue
        label = "answerable" if row.corpus_can_answer else "unanswerable"
        print(f"miss: {row.query}")
        print(f"  label: {label}")
        print(f"  route: {row.route} ({row.decided_by}, reason={row.reason or '-'})")
        print(f"  best_rerank_score: {_score(row.best_rerank_score)}")
    print(f"wrote {DEFAULT_ROUTING_REPORT}")


def _score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def _rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
