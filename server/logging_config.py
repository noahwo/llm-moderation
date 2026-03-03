"""
Centralised logging configuration for the LLM Moderation Service.

Call `setup_logging()` once at startup (done in server/__main__.py).
Every other module then obtains its logger with:

    import logging
    logger = logging.getLogger(__name__)
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "server.log"

_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Max 10 MB per log file, keep 5 backups
_MAX_BYTES   = 10 * 1024 * 1024
_BACKUP_COUNT = 5


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root logger with:
      - A RotatingFileHandler  → logs/server.log
      - A StreamHandler        → stdout (mirrors uvicorn console output)

    Subsequent calls are idempotent (handlers are only added once).

    Args:
        level: Log level string, e.g. "INFO", "DEBUG", "WARNING".
               Defaults to the LOG_LEVEL env variable, falling back to "INFO".
    """
    level = os.environ.get("LOG_LEVEL", level).upper()
    numeric = logging.getLevelName(level)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    # Avoid duplicate handlers if setup_logging is called more than once
    if root.handlers:
        return

    root.setLevel(numeric)
    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # --- File handler ---
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        _LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric)
    root.addHandler(file_handler)

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric)
    root.addHandler(console_handler)

    root.info("Logging initialised — level=%s  file=%s", level, _LOG_FILE)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper: returns a named logger."""
    return logging.getLogger(name)
