"""The Phase 6 file repeats the retrieval and routing labels and adds references."""

import pytest

from rag.core.config import PROJECT_ROOT
from rag.evaluation.golden import DEFAULT_GOLDEN, load_retrieval_cases
from rag.evaluation.qa import DEFAULT_QA, load_qa_cases
from rag.evaluation.routing import DEFAULT_ROUTING_GOLDEN, load_routing_cases
from rag.indexing.records import load_chunk_records


def test_qa_rows_match_the_retrieval_and_routing_files() -> None:
    qa = load_qa_cases(DEFAULT_QA)
    retrieval = load_retrieval_cases(DEFAULT_GOLDEN)
    routing = load_routing_cases(DEFAULT_ROUTING_GOLDEN)
    answerable = [case for case in qa if case.corpus_can_answer]
    unanswerable = [case for case in qa if not case.corpus_can_answer]
    assert len(answerable) == 16
    assert len(unanswerable) == 8
    assert [case.query for case in answerable] == [case.query for case in retrieval]
    by_query = {case.query: case for case in answerable}
    for case in retrieval:
        row = by_query[case.query]
        assert row.relevant_chunk_ids == case.relevant_chunk_ids
        assert row.evidence == case.evidence
        assert case.evidence in row.reference_answer
    assert {case.query for case in qa} == {case.query for case in routing}
    labels = {case.query: case.corpus_can_answer for case in routing}
    assert {case.query: case.corpus_can_answer for case in qa} == labels


def test_evidence_is_only_in_the_labeled_chunks() -> None:
    records = load_chunk_records(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
    by_id = {record.chunk_id: record for record in records}
    for case in load_qa_cases(DEFAULT_QA):
        if not case.corpus_can_answer:
            continue
        for chunk_id in case.relevant_chunk_ids:
            assert case.evidence in by_id[chunk_id].text
        extras = [
            record.chunk_id
            for record in records
            if case.evidence in record.text and record.chunk_id not in case.relevant_chunk_ids
        ]
        assert extras == [], (case.evidence, extras)


def test_a_bad_qa_line_names_itself(tmp_path) -> None:
    path = tmp_path / "qa_golden.jsonl"
    path.write_text('{"query": "only"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_qa_cases(path)
    path.write_text(
        '{"query": "q", "corpus_can_answer": true, "relevant_chunk_ids": ["a"], '
        '"evidence": "EXACT-SPAN", "reference_answer": "no match here"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not contain the evidence"):
        load_qa_cases(path)
    path.write_text(
        '{"query": "q", "corpus_can_answer": false, "relevant_chunk_ids": ["a"], '
        '"evidence": "", "reference_answer": ""}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty"):
        load_qa_cases(path)
