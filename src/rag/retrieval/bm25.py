"""Okapi BM25 over the same tokens the sparse index stores.

Document length is how many of those tokens the chunk has. It is not the
BGE ``token_count``. Each distinct query token adds its weight once.

IDF is the Lucene form ``ln(1 + (N - df + 0.5) / (df + 0.5))``. A token
that sits in most chunks still adds a small positive weight. ``k1`` and
``b`` come from settings. The Qdrant sparse channel stays raw term counts,
with its IDF modifier off. This module is the keyword ranker.
"""

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from rag.indexing.records import ChunkRecord
from rag.indexing.sparse import tokenize


@dataclass(frozen=True)
class BM25Hit:
    """One keyword hit. `rank` starts at 1."""

    chunk_id: str
    score: float
    rank: int


class BM25Index:
    """In-memory BM25 over one list of chunk records."""

    def __init__(self, records: Sequence[ChunkRecord], *, k1: float, b: float) -> None:
        self._records = list(records)
        self._k1 = k1
        self._b = b
        self._doc_len: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._build()

    @classmethod
    def build(
        cls,
        records: Sequence[ChunkRecord],
        *,
        k1: float,
        b: float,
    ) -> "BM25Index":
        return cls(records, k1=k1, b=b)

    @property
    def document_count(self) -> int:
        return len(self._records)

    def search(
        self,
        query: str,
        *,
        limit: int,
        allowed: set[str] | None = None,
    ) -> list[BM25Hit]:
        """Return up to `limit` hits with a positive score.

        `allowed` drops chunks before they take a slot. IDF still uses the
        full index, so a filter does not make a common token look rare.
        """

        if limit <= 0:
            raise ValueError("limit must be positive")
        tokens = list(dict.fromkeys(tokenize(query)))
        if not tokens or not self._records or self._avgdl == 0:
            return []
        scores: dict[int, float] = {}
        for token in tokens:
            postings = self._postings.get(token)
            if not postings:
                continue
            idf = self._idf(len(postings))
            for doc_index, term_count in postings:
                record = self._records[doc_index]
                if allowed is not None and record.chunk_id not in allowed:
                    continue
                weight = idf * self._tf_weight(term_count, self._doc_len[doc_index])
                scores[doc_index] = scores.get(doc_index, 0.0) + weight
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._records[item[0]].chunk_id),
        )
        hits: list[BM25Hit] = []
        for doc_index, score in ranked[:limit]:
            if score <= 0:
                continue
            hits.append(
                BM25Hit(
                    chunk_id=self._records[doc_index].chunk_id,
                    score=score,
                    rank=len(hits) + 1,
                )
            )
        return hits

    def _build(self) -> None:
        lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_index, record in enumerate(self._records):
            counts: dict[str, int] = {}
            for token in tokenize(record.text):
                counts[token] = counts.get(token, 0) + 1
            lengths.append(sum(counts.values()))
            for token, term_count in counts.items():
                postings[token].append((doc_index, term_count))
        self._doc_len = lengths
        self._postings = dict(postings)
        self._avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

    def _idf(self, document_frequency: int) -> float:
        total = len(self._records)
        return math.log(1.0 + (total - document_frequency + 0.5) / (document_frequency + 0.5))

    def _tf_weight(self, term_count: int, doc_len: int) -> float:
        normalizer = self._k1 * (1.0 - self._b + self._b * doc_len / self._avgdl)
        return term_count * (self._k1 + 1.0) / (term_count + normalizer)
