"""Golden labels cite chunks that actually contain the evidence string."""

import pytest

from rag.core.config import PROJECT_ROOT
from rag.evaluation.golden import DEFAULT_GOLDEN, load_retrieval_cases
from rag.indexing.records import load_chunk_records


def test_evidence_is_only_in_the_labeled_chunks() -> None:
    records = load_chunk_records(PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
    by_id = {record.chunk_id: record for record in records}
    cases = load_retrieval_cases(DEFAULT_GOLDEN)
    assert len(cases) >= 8
    for case in cases:
        for chunk_id in case.relevant_chunk_ids:
            assert chunk_id in by_id, chunk_id
            assert case.evidence in by_id[chunk_id].text
        extras = [
            record.chunk_id
            for record in records
            if case.evidence in record.text and record.chunk_id not in case.relevant_chunk_ids
        ]
        assert extras == [], (case.evidence, extras)


def test_a_bad_golden_line_names_itself(tmp_path) -> None:
    path = tmp_path / "retrieval_golden.jsonl"
    path.write_text('{"query": "only"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_retrieval_cases(path)
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no questions"):
        load_retrieval_cases(path)
