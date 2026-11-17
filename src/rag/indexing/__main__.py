"""Index data/processed/chunks.jsonl into Qdrant.

Qdrant must already be running: docker compose up -d qdrant
"""

import uuid

from rag.core.config import PROJECT_ROOT, get_settings
from rag.core.observability import tracing
from rag.indexing.indexer import index_processed


def main() -> None:
    settings = get_settings()
    with tracing(settings) as tracer:
        with tracer.root(
            "indexing",
            seed=uuid.uuid4().hex,
            tags=("ingest",),
            environment="cli",
        ) as root:
            stats = index_processed(
                PROJECT_ROOT / "data" / "processed",
                settings,
                trace=tracer.trace(),
            )
            root.set_output(
                {
                    "collection": stats.collection,
                    "points": stats.points,
                    "upserted": stats.upserted,
                    "pruned": stats.pruned,
                    "schema_recreated": stats.schema_recreated,
                    "self_hit_at_1": stats.self_hit_at_1,
                }
            )
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
