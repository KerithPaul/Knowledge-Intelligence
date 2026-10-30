"""Index data/processed/chunks.jsonl into Qdrant.

Qdrant must already be running: docker compose up -d qdrant
"""

from rag.core.config import PROJECT_ROOT, get_settings
from rag.indexing.indexer import index_processed


def main() -> None:
    settings = get_settings()
    stats = index_processed(PROJECT_ROOT / "data" / "processed", settings)
    self_hit = "n/a" if stats.self_hit_at_1 is None else f"{stats.self_hit_at_1:.3f}"
    hnsw = "n/a" if stats.self_hit_at_1_hnsw is None else f"{stats.self_hit_at_1_hnsw:.3f}"
    print(
        f"indexed {stats.points} points into {stats.collection} "
        f"(upserted {stats.upserted}, pruned {stats.pruned}, "
        f"schema recreated {stats.schema_recreated}, "
        f"exact self-hit@1 {self_hit}, hnsw self-hit@1 {hnsw})"
    )


if __name__ == "__main__":
    main()
