"""Run the Phase 6 evaluation harness.

The full run needs Qdrant, an index, and Groq. It asks every question in
``qa_golden.jsonl``, judges the answerable ones, and writes
``harness_report.json``.

    python -m rag.evaluation
    python -m rag.evaluation --gate
    python -m rag.evaluation --check
    python -m rag.evaluation --synthesize --limit 5
    python -m rag.evaluation --cross-check

``--gate`` runs the harness and exits with an error when a metric misses
``baselines.json``. ``--check`` gates the report already on disk.
``--synthesize`` proposes new rows and does not change the golden files.
``--cross-check`` prints the formula gaps against RAGAS and DeepEval. It
does not install those packages.
"""

import argparse
import sys
from pathlib import Path

from rag.core.clients import build_embedding_client, build_llm_client
from rag.core.config import PROJECT_ROOT, get_settings
from rag.core.errors import GenerationError
from rag.core.trace import RequestTrace
from rag.evaluation.crosscheck import (
    DIVERGENCES,
    library_status,
    precision_gap,
    ragas_context_precision,
)
from rag.evaluation.gate import DEFAULT_BASELINES, check_gate, load_baselines
from rag.evaluation.harness import (
    build_harness_report,
    gate_metrics,
    load_report_metrics,
    run_case,
    write_harness_report,
)
from rag.evaluation.qa import DEFAULT_QA, QACase, load_qa_cases
from rag.evaluation.quality import context_precision
from rag.evaluation.synthesize import (
    chunks_to_propose,
    evidence_owners,
    marker_is_absent,
    propose_from_chunk,
    propose_unanswerable,
    write_qa_lines,
)
from rag.generation.pipeline import open_generator
from rag.indexing.records import load_chunk_records
from rag.retrieval.pipeline import open_retriever
from rag.routing.pipeline import open_router
from rag.routing.types import RoutedAnswer

DEFAULT_REPORT = PROJECT_ROOT / "data" / "evaluation" / "harness_report.json"
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "evaluation" / "synthetic_candidates.jsonl"
_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
_RAW = PROJECT_ROOT / "data" / "raw"


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _reject_mixed_modes(args)
    if args.cross_check:
        _print_cross_check()
        return
    if args.check:
        try:
            _check(DEFAULT_REPORT, DEFAULT_BASELINES)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from None
        return
    if args.synthesize:
        _synthesize(limit=args.limit, outside=args.outside)
        return
    report_path = _run()
    if args.gate:
        _check(report_path, DEFAULT_BASELINES)


def _run() -> Path:
    settings = get_settings()
    cases = load_qa_cases(DEFAULT_QA)
    llm = build_llm_client(settings)
    embedder = build_embedding_client(settings)
    router = open_router(
        settings,
        retriever=open_retriever(settings, embedder=embedder),
        generator=open_generator(settings, llm=llm),
    )

    def ask(query: str, trace: RequestTrace) -> RoutedAnswer:
        try:
            return router.ask(query, trace=trace)
        except GenerationError as exc:
            raise SystemExit(f"could not read the model reply: {exc}") from None

    rows = []
    for case in cases:
        try:
            rows.append(
                run_case(
                    case,
                    ask=ask,
                    llm=llm,
                    embedder=embedder,
                    relevance_questions=settings.generation.relevance_questions,
                    max_context_chunks=settings.generation.max_context_chunks,
                    max_tokens=settings.generation.max_tokens,
                    factual_weight=settings.evaluation.factual_weight,
                )
            )
        except GenerationError as exc:
            raise SystemExit(f"could not score {case.query!r}: {exc}") from None
    report = build_harness_report(
        rows,
        k=settings.retrieval.top_k,
        threshold=settings.routing.sufficiency_threshold,
        factual_weight=settings.evaluation.factual_weight,
        win_correctness=settings.evaluation.win_correctness,
        prompt_usd_per_million=settings.evaluation.prompt_usd_per_million,
        completion_usd_per_million=settings.evaluation.completion_usd_per_million,
    )
    write_harness_report(report, DEFAULT_REPORT)
    _print_report(report)
    return DEFAULT_REPORT


def _check(report_path: Path, baseline_path: Path) -> None:
    metrics = load_report_metrics(report_path)
    floors, ceilings = load_baselines(baseline_path)
    failures = check_gate(metrics, floors, ceilings)
    if not failures:
        print(f"gate passed ({len(floors)} floors, {len(ceilings)} ceilings)")
        return
    for failure in failures:
        actual = "missing" if failure.actual is None else f"{failure.actual:.3f}"
        print(f"gate failed: {failure.metric} is {actual}, {failure.direction} {failure.bound:.3f}")
    raise SystemExit(1)


def _synthesize(*, limit: int | None, outside: int) -> None:
    settings = get_settings()
    count = settings.evaluation.synthesize_limit if limit is None else limit
    if count < 0 or outside < 0:
        raise SystemExit("--limit and --outside must be zero or more")
    if count == 0 and outside == 0:
        raise SystemExit("pass a positive --limit or --outside")
    records = load_chunk_records(_CHUNKS)
    covered = {
        chunk_id
        for case in load_qa_cases(DEFAULT_QA)
        if case.corpus_can_answer
        for chunk_id in case.relevant_chunk_ids
    }
    chosen = [] if count == 0 else chunks_to_propose(records, covered, limit=count)
    llm = build_llm_client(settings)
    kept: list[QACase] = []
    failures = 0
    for chunk in chosen:
        try:
            case = propose_from_chunk(chunk, llm, max_tokens=settings.generation.max_tokens)
        except (GenerationError, ValueError) as exc:
            print(f"skip {chunk.chunk_id}: {exc}")
            failures += 1
            continue
        owners = evidence_owners(case.evidence, records)
        if owners != [chunk.chunk_id]:
            print(f"skip {chunk.chunk_id}: evidence also appears in {', '.join(owners)}")
            failures += 1
            continue
        kept.append(case)
        print(f"kept {chunk.chunk_id}: {case.query}")
    corpus = _corpus_text()
    for _index in range(outside):
        try:
            proposal = propose_unanswerable(llm, max_tokens=settings.generation.max_tokens)
        except (GenerationError, ValueError) as exc:
            print(f"skip outside question: {exc}")
            failures += 1
            continue
        if not marker_is_absent(proposal.marker, corpus):
            print(f"skip outside question: {proposal.marker!r} is in the corpus")
            failures += 1
            continue
        kept.append(
            QACase(
                query=proposal.question,
                corpus_can_answer=False,
                relevant_chunk_ids=(),
                evidence="",
                reference_answer="",
            )
        )
        print(f"kept outside question: {proposal.question}")
    if not kept:
        print("no candidates kept")
        if failures:
            raise SystemExit(1)
        return
    write_qa_lines(kept, DEFAULT_CANDIDATES)
    print(f"wrote {len(kept)} rows to {DEFAULT_CANDIDATES}")
    print("These rows are not in the golden set. Review them before copying any in.")


def _print_cross_check() -> None:
    print("Formula gaps. The arithmetic below uses the same usefulness flags.")
    for flags in ((True,), (True, False), (False, True), (False, False)):
        ours = context_precision(flags)
        library = ragas_context_precision(flags)
        gap = precision_gap(flags, ours)
        shown = ", ".join("useful" if flag else "not useful" for flag in flags)
        print(f"  [{shown}] ours {ours:.10f}  ragas {library:.10f}  gap {gap:.10f}")
    print()
    for item in DIVERGENCES:
        print(f"{item.metric} ({item.library}): {item.detail}")
    print()
    for name, status in library_status().items():
        if status["installed"]:
            print(f"{name}: installed {status['version']}")
        else:
            print(f"{name}: not installed")


def _print_report(report) -> None:
    metrics = gate_metrics(report)
    print(
        f"harness: {report.queries} questions, {report.wins} wins, win_rate {report.win_rate:.3f}"
    )
    print(f"routing accuracy {report.routing.accuracy:.3f}")
    retrieval = report.retrieval
    if retrieval is None:
        print("retrieval: no answerable questions")
    else:
        print(
            f"retrieval hit@{report.k} {retrieval.hit_at_k:.3f}  "
            f"recall@{report.k} {retrieval.recall_at_k:.3f}"
        )
    print(f"faithfulness {_metric(metrics['faithfulness'])}")
    print(f"answer_relevance {_metric(metrics['answer_relevance'])}")
    print(f"context_precision {_metric(metrics['context_precision'])}")
    print(f"context_recall {_metric(metrics['context_recall'])}")
    print(f"answer_correctness {_metric(metrics['answer_correctness'])}")
    print(
        f"tokens prompt {report.prompt_tokens} completion {report.completion_tokens} "
        f"unpriced_calls {report.unpriced_calls}"
    )
    print(f"cost_usd {report.cost_usd:.6f}  mean_elapsed_ms {report.mean_elapsed_ms:.1f}")
    print(f"wrote {DEFAULT_REPORT}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.evaluation",
        description="Score the golden questions and write a harness report.",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Run the harness, then fail if the report misses baselines.json.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Gate the harness report already on disk. Does not call the model.",
    )
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Propose new golden rows from chunks. Does not edit the golden files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="How many uncovered chunks to propose. The default is evaluation.synthesize_limit.",
    )
    parser.add_argument(
        "--outside",
        type=int,
        default=0,
        help="How many out-of-corpus questions to propose alongside --synthesize.",
    )
    parser.add_argument(
        "--cross-check",
        action="store_true",
        help="Print formula gaps versus RAGAS and DeepEval, and whether those packages import.",
    )
    return parser


def _reject_mixed_modes(args: argparse.Namespace) -> None:
    chosen = [args.gate, args.check, args.synthesize, args.cross_check]
    if args.gate and (args.check or args.synthesize or args.cross_check):
        raise SystemExit("pass --gate alone, or one of --check, --synthesize, --cross-check")
    if sum(1 for flag in chosen if flag) > 1:
        raise SystemExit("pass only one of --check, --synthesize, --cross-check")
    if (args.limit is not None or args.outside) and not args.synthesize:
        raise SystemExit("--limit and --outside belong with --synthesize")


def _corpus_text() -> str:
    parts = [path.read_text(encoding="utf-8") for path in sorted(_RAW.rglob("*.md"))]
    return "\n".join(parts)


def _metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


if __name__ == "__main__":
    main()
    sys.exit(0)
