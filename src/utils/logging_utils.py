"""One-line logging setup for scripts (library modules just use logging.getLogger(__name__))."""

from __future__ import annotations

import logging
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Log to the console and, optionally, to a file (parent folders are created)."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper()), format=FORMAT,
                        datefmt="%H:%M:%S", handlers=handlers, force=True)
