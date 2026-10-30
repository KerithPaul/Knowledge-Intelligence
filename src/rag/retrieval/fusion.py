"""Reciprocal Rank Fusion of ordered chunk-id lists.

A chunk's score is the sum of ``1 / (k + rank)`` over every list that
contains it. Rank starts at 1. A chunk that is missing from a list adds
nothing for that list. Ties break on chunk id so the same inputs always
return the same order.
"""

from collections.abc import Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int,
    limit: int,
) -> list[tuple[str, float]]:
    """Merge rankings and return at most `limit` ``(chunk_id, score)`` pairs."""

    if k <= 0:
        raise ValueError("rrf k must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    totals: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            totals[chunk_id] = totals.get(chunk_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit]
