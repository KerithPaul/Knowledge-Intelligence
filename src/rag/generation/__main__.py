"""Answer one question from the indexed corpus.

Qdrant must already be running and the corpus indexed:
    docker compose up -d qdrant
    python -m rag.indexing

Examples:
    python -m rag.generation "How does ACP forking work?"
    python -m rag.generation --source hermes/acp-internals.md "How does cancel work?"
    python -m rag.generation --score "How does ACP forking work?"
"""

import argparse

from rag.core.clients import build_embedding_client, build_llm_client
from rag.core.config import Settings, get_settings
from rag.core.errors import GenerationError
from rag.core.interfaces import EmbeddingClient, LLMClient
from rag.evaluation.generation import GenerationScore, score_answer
from rag.generation.pipeline import open_generator
from rag.generation.types import GroundedAnswer
from rag.retrieval.filters import MetadataFilter
from rag.retrieval.pipeline import open_retriever


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.query is None or not str(args.query).strip():
        raise SystemExit("pass a query")
    settings = get_settings()
    llm = build_llm_client(settings)
    generator = open_generator(settings, llm=llm)
    if args.score:
        embedder = build_embedding_client(settings)
        retriever = open_retriever(settings, embedder=embedder)
    else:
        embedder = None
        retriever = open_retriever(settings)
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
    try:
        answer = generator.generate(args.query, hits)
    except GenerationError as exc:
        raise SystemExit(f"could not read the model reply: {exc}") from None
    _print_answer(answer)
    if args.score:
        _print_score(_score(answer, llm=llm, embedder=embedder, settings=settings))


def _score(
    answer: GroundedAnswer,
    *,
    llm: LLMClient,
    embedder: EmbeddingClient | None,
    settings: Settings,
) -> GenerationScore:
    if embedder is None:
        raise RuntimeError("scoring requires an embedder")
    try:
        return score_answer(
            answer,
            llm=llm,
            embedder=embedder,
            relevance_questions=settings.generation.relevance_questions,
            max_tokens=settings.generation.max_tokens,
        )
    except GenerationError as exc:
        raise SystemExit(f"could not score the answer: {exc}") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.generation",
        description="Answer a question from the indexed documentation, with citations.",
    )
    parser.add_argument("query", nargs="?", help="Question to answer.")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Also report faithfulness and answer relevance for this answer.",
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


def _print_answer(answer: GroundedAnswer) -> None:
    if answer.sufficient:
        print("sufficient: yes")
    else:
        print(f"sufficient: no ({answer.insufficient_reason})")
    if not answer.citations:
        print("citations: none")
    for citation in answer.citations:
        if citation.heading_path:
            heading = " > ".join(citation.heading_path)
        else:
            heading = citation.source
        print(f"[{citation.number}] {citation.chunk_id}  {heading}")
    model = answer.model if answer.model else "-"
    print(f"model: {model}")
    print()
    print(answer.answer)


def _print_score(score: GenerationScore) -> None:
    faith = score.faithfulness
    if faith.score is None:
        faith_text = "n/a"
    else:
        faith_text = f"{faith.score:.3f}"
    supported = sum(1 for claim in faith.claims if claim.supported)
    print()
    print(f"faithfulness: {faith_text} ({supported}/{len(faith.claims)} claims)")
    print(f"answer_relevance: {score.answer_relevance.score:.3f}")


if __name__ == "__main__":
    main()
