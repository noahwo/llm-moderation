# server/t5_app.py
from __future__ import annotations

import asyncio
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.t5_model_runner import T5ModelRunner


# -----------------------
# Pydantic request/response
# -----------------------
class T5ModerateRequest(BaseModel):
    text: str = Field(..., description="Plain text to classify.")
    max_new_tokens: int = Field(default=10, ge=1, le=64)


class T5ModerateResponse(BaseModel):
    verdict: str      # "toxic" | "non-toxic" | "unknown"
    raw: str          # "positive" | "negative" | ...
    model: str


# -----------------------
# App + global runner
# -----------------------
# Always use the HF hub ID; override with T5_MODEL_ID env var if needed.
MODEL_ID = os.environ.get("T5_MODEL_ID", "lmsys/toxicchat-t5-large-v1.0")
TORCH_DTYPE = os.environ.get("T5_TORCH_DTYPE", "float32")
DEVICE_MAP = os.environ.get("T5_DEVICE_MAP", "auto")
MAX_CONCURRENT = int(os.environ.get("T5_MAX_CONCURRENT", "2"))

_sema = asyncio.Semaphore(MAX_CONCURRENT)

app = FastAPI(title="ToxicChat-T5 Moderation Service", version="1.0.0")

_runner: Optional[T5ModelRunner] = None


@app.on_event("startup")
def _startup() -> None:
    global _runner
    _runner = T5ModelRunner(
        model_id=MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
    )


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    if _runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"ok": True, "model": MODEL_ID}


@app.post("/moderate", response_model=T5ModerateResponse)
async def moderate(req: T5ModerateRequest) -> T5ModerateResponse:
    if _runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    async with _sema:
        try:
            res = _runner.moderate(text=req.text, max_new_tokens=req.max_new_tokens)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"moderation_failed: {type(e).__name__}: {e}")

    return T5ModerateResponse(verdict=res.verdict, raw=res.raw, model=res.model)
