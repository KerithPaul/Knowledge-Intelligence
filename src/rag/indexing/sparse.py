"""Lexical sparse vectors for the Qdrant sparse channel.

A token is a lowercase run of letters, digits, and underscores. The value
is how many times that token appears in the chunk. Indices come from one
vocabulary for the whole batch, sorted, so the same texts always get the
same indices. A new earlier token shifts later indices, so a re-index
rewrites every point instead of patching one chunk.

This is not BM25. Phase 3 scores keywords by hand and fuses that ranking
with dense search. These vectors stay raw counts, and Qdrant's IDF
modifier stays off.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

# Stable name stored on the collection so a future sparse scheme can
# see that these vectors are term counts.
SPARSE_KIND = "term_frequency"

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split text into sparse-channel tokens."""

    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class SparseVector:
    """One sparse vector. Indices are unique and sorted. Values are counts."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def nnz(self) -> int:
        return len(self.indices)


def build_sparse_vectors(
    texts: Sequence[str],
) -> tuple[dict[str, int], list[SparseVector]]:
    """Build one vocabulary and one term-count vector per text."""

    tokens = {token for text in texts for token in tokenize(text)}
    vocabulary = {token: index for index, token in enumerate(sorted(tokens))}
    vectors: list[SparseVector] = []
    for text in texts:
        counts: dict[int, int] = {}
        for token in tokenize(text):
            index = vocabulary[token]
            counts[index] = counts.get(index, 0) + 1
        pairs = sorted(counts.items())
        vectors.append(
            SparseVector(
                indices=tuple(index for index, _count in pairs),
                values=tuple(float(count) for _index, count in pairs),
            )
        )
    return vocabulary, vectors
