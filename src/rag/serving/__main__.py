"""Run the API.

Qdrant must already be running. The first `/ask` loads the local models.

    docker compose up -d qdrant
    python -m rag.serving

The Compose app service uses the docker profile:

    docker compose --profile docker up -d --build
"""

import argparse

from rag.core.config import get_settings
from rag.serving.logging import configure_logging


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = _parser(settings.serving.host, settings.serving.port)
    args = parser.parse_args(argv)
    configure_logging(settings.serving.log_level)
    import uvicorn

    uvicorn.run(
        "rag.serving.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=settings.serving.log_level,
    )


def _parser(host: str, port: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag.serving",
        description=(
            "Serve /health, /ask, and /ingest. "
            "A streamed ask sends the grounded answer, not raw model tokens."
        ),
    )
    parser.add_argument("--host", default=host, help="Bind address. Default comes from settings.")
    parser.add_argument(
        "--port", type=int, default=port, help="Bind port. Default comes from settings."
    )
    return parser


if __name__ == "__main__":
    main()
