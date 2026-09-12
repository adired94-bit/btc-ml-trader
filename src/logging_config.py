"""Structured logging shared by every entry point (API, CLI scripts, dashboard)."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from config import settings

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once with console + rotating file handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.logs_dir / "app.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Quieten noisy third-party loggers.
    for noisy in ("ccxt", "urllib3", "httpx", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
