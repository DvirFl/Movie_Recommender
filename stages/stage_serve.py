"""Stage 10 — Serve: launch the FastAPI REST API with uvicorn."""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ServeResult:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False


def run(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    workers: int = 1,
) -> ServeResult:
    """Launch the FastAPI application via uvicorn.

    Args:
        host:    bind address (default 0.0.0.0).
        port:    bind port (default 8000).
        reload:  enable hot-reload for development (single-worker only).
        workers: number of uvicorn worker processes (ignored when reload=True).

    Returns:
        ServeResult — only returned if uvicorn exits (blocking call).
    """
    import uvicorn

    logger.info("[serve] Starting API on %s:%d  reload=%s", host, port, reload)
    uvicorn.run(
        "serving.api:app",
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,
        log_level="info",
    )
    return ServeResult(host=host, port=port, reload=reload)
