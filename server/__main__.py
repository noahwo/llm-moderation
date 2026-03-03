"""
Entry point for the LLM Moderation Service.

Usage:
    python -m server                   # default: 0.0.0.0:18084
    PORT=18085 python -m server        # custom port

Environment variables:
    PORT        bind port       (default 18084)
    HOST        bind address    (default 0.0.0.0)
    LOG_LEVEL   uvicorn level   (default info)

All model settings (LG4_*, T5_*) are read in server/app.py.
"""
from __future__ import annotations

import os
import uvicorn

from server.logging_config import setup_logging

PORT      = int(os.environ.get("PORT",      "18084"))
HOST      =     os.environ.get("HOST",      "0.0.0.0")
LOG_LEVEL =     os.environ.get("LOG_LEVEL", "info")

if __name__ == "__main__":
    setup_logging(LOG_LEVEL)
    uvicorn.run(
        "server.app:app",
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL,
    )
