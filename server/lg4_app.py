# server/app.py
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.lg4_model_runner import ModelRunner


# -----------------------
# Pydantic request/response
# -----------------------
class ModerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(
        ...,
        description="HF chat message list; content supports multimodal blocks like {type:text|image,...}",
    )
    excluded_category_keys: Optional[List[str]] = Field(
        default=None,
        description="Optional list of category keys to exclude (e.g. ['S9']).",
    )
    max_new_tokens: int = Field(default=10, ge=1, le=256)
    do_sample: bool = Field(default=False)


class ModerateResponse(BaseModel):
    verdict: str
    categories: List[str]
    raw: str
    model: str


# -----------------------
# App + global runner
# -----------------------
def _default_model_path() -> str:
    """Pick a local cache if present; otherwise fall back to repo id."""

    hf_home = os.environ.get("HF_HOME")
    hf_cache = os.environ.get("TRANSFORMERS_CACHE")
    custom_cache = os.environ.get("LG4_MODEL_CACHE")

    candidates = [
        os.environ.get("LG4_LOCAL_MODEL_PATH"),
        os.environ.get("LG4_MODEL_PATH"),
        os.path.join(custom_cache, "models--meta-llama--Llama-Guard-4-12B") if custom_cache else None,
        os.path.join(hf_cache, "models--meta-llama--Llama-Guard-4-12B") if hf_cache else None,
        os.path.join(hf_home, "hub", "models--meta-llama--Llama-Guard-4-12B") if hf_home else None,
        "/home/wuguangh/.cache/huggingface/hub/models--meta-llama--Llama-Guard-4-12B",
    ]

    for c in candidates:
        if c and os.path.exists(c):
            return c

    return "meta-llama/Llama-Guard-4-12B"


# Default to a local cache path if available to avoid fetching from HF at runtime.
MODEL_ID = os.environ.get("LG4_MODEL_ID", _default_model_path())
TORCH_DTYPE = os.environ.get("LG4_TORCH_DTYPE", "bfloat16")
DEVICE_MAP = os.environ.get("LG4_DEVICE_MAP", "auto")

# Concurrency guard (GPU safety). Increase if you *know* you can handle it.
MAX_CONCURRENT = int(os.environ.get("LG4_MAX_CONCURRENT", "1"))
_sema = asyncio.Semaphore(MAX_CONCURRENT)

# FastAPI app instance
app = FastAPI(title="LlamaGuard4 Moderation Service", version="1.0.0")

_runner: Optional[ModelRunner] = None


@app.on_event("startup")
def _startup() -> None:
    global _runner
    _runner = ModelRunner(
        model_id=MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
    )


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    if _runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"ok": True, "model": MODEL_ID}


@app.post("/moderate", response_model=ModerateResponse)
async def moderate(req: ModerateRequest) -> ModerateResponse:
    if _runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Prevent GPU stampede
    async with _sema:
        try:
            res = _runner.moderate(
                messages=req.messages,
                excluded_category_keys=req.excluded_category_keys,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
            )
        except Exception as e:
            # Bubble up a readable error for debugging
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"moderation_failed: {type(e).__name__}: {e}")

    return ModerateResponse(
        verdict=res.verdict,
        categories=res.categories,
        raw=res.raw,
        model=res.model,
    )