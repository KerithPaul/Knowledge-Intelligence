"""Sparse term counts stay deterministic for one batch of texts."""

from rag.indexing.sparse import build_sparse_vectors, tokenize


def test_tokenize_lowercases_and_keeps_identifiers() -> None:
    assert tokenize("Agent_Loop runs C++ and Node.js") == [
        "agent_loop",
        "runs",
        "c",
        "and",
        "node",
        "js",
    ]


def test_term_counts_and_sorted_vocabulary() -> None:
    vocabulary, vectors = build_sparse_vectors(["Cat cat dog", "???"])
    assert vocabulary == {"cat": 0, "dog": 1}
    assert vectors[0].indices == (0, 1)
    assert vectors[0].values == (2.0, 1.0)
    assert vectors[1].indices == ()
    assert vectors[1].nnz == 0


def test_same_texts_keep_the_same_indices() -> None:
    first, _vectors = build_sparse_vectors(["beta cat", "alpha"])
    second, _again = build_sparse_vectors(["beta cat", "alpha"])
    assert first == second


def test_a_new_earlier_token_shifts_later_indices() -> None:
    first, vectors = build_sparse_vectors(["beta cat"])
    second, shifted = build_sparse_vectors(["alpha", "beta cat"])
    assert first["beta"] == 0
    assert vectors[0].indices == (0, 1)
    assert second["beta"] == 1
    assert shifted[1].indices == (1, 2)
