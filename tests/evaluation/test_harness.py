"""The harness averages judged rows and can judge one routed question offline."""

import json
from pathlib import Path

import pytest

from rag.core.interfaces import EmbeddingClient, LLMClient, LLMResponse
from rag.core.trace import RequestTrace
from rag.evaluation.harness import (
    CaseResult,
    build_harness_report,
    context_from_route,
    load_report_metrics,
    run_case,
    write_harness_report,
)
from rag.evaluation.qa import QACase
from rag.generation.types import INSUFFICIENT_CONTEXT, ContextBlock, GroundedAnswer
from rag.retrieval.types import RetrievedChunk
from rag.routing.types import RoutedAnswer


def test_the_report_averages_answerable_scores_and_counts_wins(tmp_path: Path) -> None:
    rows = (
        _row("answered", corpus_can_answer=True, route="knowledge_base", sufficient=True, hit=True),
        _row(
            "refused",
            corpus_can_answer=True,
            route="insufficient",
            sufficient=False,
            hit=True,
            correctness=None,
            faithfulness=None,
        ),
        _row("outside", corpus_can_answer=False, route="insufficient", sufficient=False, hit=False),
    )
    report = build_harness_report(
        rows,
        k=1,
        threshold=0.35,
        factual_weight=0.75,
        win_correctness=0.5,
        prompt_usd_per_million=0.075,
        completion_usd_per_million=0.30,
    )
    assert report.queries == 3
    assert report.wins == 2
    assert report.win_rate == pytest.approx(2 / 3)
    assert report.retrieval is not None
    assert report.retrieval.queries == 2
    assert report.retrieval.hit_at_k == 1.0
    assert report.routing.unanswerable == 1
    assert report.routing.accuracy == pytest.approx(2 / 3)
    assert report.faithfulness == pytest.approx(1.0)
    assert report.answer_correctness == pytest.approx(0.8)
    assert report.prompt_tokens == 30
    assert report.cost_usd == pytest.approx(30 * 0.075 / 1_000_000 + 15 * 0.30 / 1_000_000)
    assert report.stages[0].name == "routing.decision"
    path = tmp_path / "harness_report.json"
    write_harness_report(report, path)
    assert load_report_metrics(path)["win_rate"] == pytest.approx(2 / 3)


def test_a_score_refusal_is_judged_against_the_retrieved_chunk() -> None:
    case = QACase(
        query="How does forking work?",
        corpus_can_answer=True,
        relevant_chunk_ids=("a",),
        evidence="fork copies",
        reference_answer="The fork copies the session.",
    )
    chunk = _chunk("a", "The fork copies the session.")

    def ask(query: str, trace: RequestTrace) -> RoutedAnswer:
        trace.record("llm.complete", 1.0, prompt_tokens=4, completion_tokens=5)
        return RoutedAnswer(
            query=query,
            route="insufficient",
            decided_by="retrieval_score",
            reason="below_threshold",
            best_rerank_score=0.1,
            threshold=0.35,
            chunks=(chunk,),
            answer=None,
        )

    claims = json.dumps({"claims": []})
    questions = json.dumps({"questions": [{"question": "What is missing?", "noncommittal": True}]})
    precision = json.dumps({"verdicts": [{"number": 1, "useful": True}]})
    reference_claims = json.dumps({"claims": ["The fork copies the session."]})
    verdicts = json.dumps({"verdicts": [{"supported": True}]})
    facts = json.dumps(
        {
            "true_positives": [],
            "false_positives": [],
            "false_negatives": ["The fork copies the session."],
        }
    )
    llm = _ScriptedLLM([claims, questions, precision, reference_claims, verdicts, facts])
    embedder = _Embedder()
    row = run_case(
        case,
        ask=ask,
        llm=llm,
        embedder=embedder,
        relevance_questions=1,
        max_context_chunks=5,
        max_tokens=64,
        factual_weight=0.75,
    )
    assert row.route == "insufficient"
    assert row.retrieved_chunk_ids == ("a",)
    assert row.faithfulness is None
    assert row.answer_relevance == 0.0
    assert row.context_precision == 1.0
    assert row.context_recall == 1.0
    assert row.answer_correctness == 0.0
    assert row.prompt_tokens == 4 + 6 * 3
    assert row.completion_tokens == 5 + 6 * 2
    precision_user = llm.calls[2][1].content
    assert "The fork copies the session." in precision_user
    assert "fork copies" not in embedder.queries
    routed = ask(case.query, RequestTrace())
    seen = context_from_route(routed, limit=5)
    assert seen[0].chunk_id == "a"
    assert seen[0].text == "The fork copies the session."


def _row(
    query: str,
    *,
    corpus_can_answer: bool,
    route: str,
    sufficient: bool,
    hit: bool,
    correctness: float | None = 0.8,
    faithfulness: float | None = 1.0,
) -> CaseResult:
    return CaseResult(
        query=query,
        corpus_can_answer=corpus_can_answer,
        relevant_chunk_ids=("a",) if corpus_can_answer else (),
        retrieved_chunk_ids=("a",) if hit else ("other",),
        route=route,  # type: ignore[arg-type]
        decided_by="groundedness" if route == "knowledge_base" else "retrieval_score",
        reason=None if route == "knowledge_base" else "below_threshold",
        best_rerank_score=0.9 if route == "knowledge_base" else 0.1,
        sufficient=sufficient,
        faithfulness=faithfulness if corpus_can_answer else None,
        answer_relevance=0.7 if corpus_can_answer else None,
        context_precision=1.0 if corpus_can_answer else None,
        context_recall=1.0 if corpus_can_answer else None,
        answer_correctness=correctness if corpus_can_answer else None,
        prompt_tokens=10,
        completion_tokens=5,
        unpriced_calls=0,
        elapsed_ms=12.0,
        stages_ms=(("routing.decision", 2.0),),
    )


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        rank=1,
        chunk_id=chunk_id,
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text=text,
        token_count=20,
        index=0,
        oversized=False,
        orphan=False,
        dense_score=0.5,
        dense_rank=1,
        bm25_score=1.0,
        bm25_rank=1,
        rrf_score=0.02,
        rerank_score=0.1,
    )


class _ScriptedLLM(LLMClient):
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[object, object]] = []

    def complete(self, messages, *, temperature=None, max_tokens=None, trace=None):
        del temperature
        self.calls.append((max_tokens, messages[1]))
        if trace is not None:
            trace.record("llm.complete", 1.0, prompt_tokens=3, completion_tokens=2)
        return LLMResponse(
            content=self._replies.pop(0),
            model="scripted",
            prompt_tokens=3,
            completion_tokens=2,
            latency_ms=1.0,
        )


class _Embedder(EmbeddingClient):
    def __init__(self) -> None:
        self.queries: list[str] = []

    @property
    def dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "scripted"

    def embed_query(self, text: str, *, trace=None) -> list[float]:
        del trace
        self.queries.append(text)
        if text == "How does forking work?":
            return [1.0, 0.0]
        return [0.0, 1.0]

    def embed_passages(self, texts, *, trace=None) -> list[list[float]]:
        del texts, trace
        return [[1.0, 0.0], [0.0, 1.0]]


def test_context_blocks_keep_their_numbers() -> None:
    block = ContextBlock(
        number=1,
        chunk_id="a",
        source="hermes/acp-internals.md",
        doc_title="ACP Internals",
        heading_path=("Forking",),
        text="The fork copies the session.",
    )
    answer = GroundedAnswer(
        query="How does forking work?",
        answer="The fork copies the session [1].",
        sufficient=True,
        insufficient_reason=None,
        citations=(),
        context=(block,),
        model="scripted",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
    )
    routed = RoutedAnswer(
        query="How does forking work?",
        route="knowledge_base",
        decided_by="groundedness",
        reason=None,
        best_rerank_score=0.9,
        threshold=0.35,
        chunks=(_chunk("a", "The fork copies the session."),),
        answer=answer,
    )
    seen = context_from_route(routed, limit=5)
    assert seen == (block,)
    assert INSUFFICIENT_CONTEXT not in seen[0].text
