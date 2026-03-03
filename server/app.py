# server/app.py
"""
LLM Moderation Service — FastAPI application.

Routes:
  GET  /healthz          – overall health (both models)
  GET  /lg4/healthz      – LlamaGuard-4 health
  POST /lg4/moderate     – LlamaGuard-4 moderation (text + optional images)
  GET  /t5/healthz       – ToxicChat-T5 health
  POST /t5/moderate      – ToxicChat-T5 moderation (text only)

Configure via environment variables (see server/__main__.py for defaults):
  LG4_MODEL_ID, LG4_TORCH_DTYPE, LG4_DEVICE_MAP, LG4_MAX_CONCURRENT
  T5_MODEL_ID,  T5_TORCH_DTYPE,  T5_DEVICE_MAP,  T5_MAX_CONCURRENT
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.lg4_model_runner import LG4ModelRunner
from server.t5_model_runner import T5ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class LG4ModerateRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "HF chat message list. Content blocks support text and images: "
            "{type:text, text:...} or {type:image, url:...} / {type:image, data:<b64>}"
        ),
    )
    excluded_category_keys: Optional[List[str]] = Field(default=None)
    max_new_tokens: int = Field(default=10, ge=1, le=256)
    do_sample: bool = Field(default=False)


class LG4ModerateResponse(BaseModel):
    verdict: str          # "safe" | "unsafe" | "unknown"
    categories: List[str] # e.g. ["S9"]
    raw: str
    model: str


class T5ModerateRequest(BaseModel):
    text: str = Field(..., description="Plain text to classify.")
    max_new_tokens: int = Field(default=10, ge=1, le=64)


class T5ModerateResponse(BaseModel):
    verdict: str  # "toxic" | "non-toxic" | "unknown"
    raw: str      # "positive" | "negative"
    model: str


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
def _default_lg4_model() -> str:
    hf_home = os.environ.get("HF_HOME")
    hf_cache = os.environ.get("TRANSFORMERS_CACHE")
    candidates = [
        os.environ.get("LG4_LOCAL_MODEL_PATH"),
        os.environ.get("LG4_MODEL_PATH"),
        os.path.join(hf_cache, "models--meta-llama--Llama-Guard-4-12B") if hf_cache else None,
        os.path.join(hf_home, "hub", "models--meta-llama--Llama-Guard-4-12B") if hf_home else None,
        "/home/wuguangh/.cache/huggingface/hub/models--meta-llama--Llama-Guard-4-12B",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return "meta-llama/Llama-Guard-4-12B"


LG4_MODEL_ID   = os.environ.get("LG4_MODEL_ID", _default_lg4_model())
LG4_DTYPE      = os.environ.get("LG4_TORCH_DTYPE", "bfloat16")
LG4_DEVICE_MAP = os.environ.get("LG4_DEVICE_MAP", "cuda")
LG4_MAX_CONC   = int(os.environ.get("LG4_MAX_CONCURRENT", "1"))

T5_MODEL_ID    = os.environ.get("T5_MODEL_ID", "lmsys/toxicchat-t5-large-v1.0")
T5_DTYPE       = os.environ.get("T5_TORCH_DTYPE", "float32")
T5_DEVICE_MAP  = os.environ.get("T5_DEVICE_MAP", "cuda")
T5_MAX_CONC    = int(os.environ.get("T5_MAX_CONCURRENT", "2"))

_lg4_sema = asyncio.Semaphore(LG4_MAX_CONC)
_t5_sema  = asyncio.Semaphore(T5_MAX_CONC)

# ---------------------------------------------------------------------------
# Global model runners (populated at startup)
# ---------------------------------------------------------------------------
_lg4_runner: Optional[LG4ModelRunner] = None
_t5_runner:  Optional[T5ModelRunner]  = None

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="LLM Moderation Service", version="1.0.0")

lg4_router = APIRouter(prefix="/lg4", tags=["LlamaGuard-4"])
t5_router  = APIRouter(prefix="/t5",  tags=["ToxicChat-T5"])


@app.on_event("startup")
def _startup() -> None:
    global _lg4_runner, _t5_runner
    node_ip = os.environ.get("NODE_IP", "unknown")
    logger.info("Node IP: %s", node_ip)
    logger.info("Loading LlamaGuard-4 (model=%s) ...", LG4_MODEL_ID)
    _lg4_runner = LG4ModelRunner(
        model_id=LG4_MODEL_ID,
        torch_dtype=LG4_DTYPE,
        device_map=LG4_DEVICE_MAP,
    )
    logger.info("Loading ToxicChat-T5 (model=%s) ...", T5_MODEL_ID)
    _t5_runner = T5ModelRunner(
        model_id=T5_MODEL_ID,
        torch_dtype=T5_DTYPE,
        device_map=T5_DEVICE_MAP,
    )
    logger.info("Both models ready.")


# ---------------------------------------------------------------------------
# Overall health
# ---------------------------------------------------------------------------
@app.get("/healthz")
def healthz_all() -> Dict[str, Any]:
    return {
        "ok": _lg4_runner is not None and _t5_runner is not None,
        "lg4": {"loaded": _lg4_runner is not None, "model": LG4_MODEL_ID},
        "t5":  {"loaded": _t5_runner  is not None, "model": T5_MODEL_ID},
    }


# ---------------------------------------------------------------------------
# LlamaGuard-4 routes
# ---------------------------------------------------------------------------
@lg4_router.get("/healthz")
def lg4_healthz() -> Dict[str, Any]:
    if _lg4_runner is None:
        raise HTTPException(status_code=503, detail="LG4 model not loaded yet")
    return {"ok": True, "model": LG4_MODEL_ID}


@lg4_router.post("/moderate", response_model=LG4ModerateResponse)
async def lg4_moderate(req: LG4ModerateRequest) -> LG4ModerateResponse:
    if _lg4_runner is None:
        raise HTTPException(status_code=503, detail="LG4 model not loaded yet")
    async with _lg4_sema:
        try:
            res = _lg4_runner.moderate(
                messages=req.messages,
                excluded_category_keys=req.excluded_category_keys,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
            )
        except Exception as e:
            logger.exception("LG4 moderation failed")
            raise HTTPException(status_code=500, detail=f"lg4_failed: {type(e).__name__}: {e}")
    logger.debug("LG4 result: verdict=%s categories=%s", res.verdict, res.categories)
    return LG4ModerateResponse(
        verdict=res.verdict, categories=res.categories, raw=res.raw, model=res.model
    )


# ---------------------------------------------------------------------------
# ToxicChat-T5 routes
# ---------------------------------------------------------------------------
@t5_router.get("/healthz")
def t5_healthz() -> Dict[str, Any]:
    if _t5_runner is None:
        raise HTTPException(status_code=503, detail="T5 model not loaded yet")
    return {"ok": True, "model": T5_MODEL_ID}


@t5_router.post("/moderate", response_model=T5ModerateResponse)
async def t5_moderate(req: T5ModerateRequest) -> T5ModerateResponse:
    if _t5_runner is None:
        raise HTTPException(status_code=503, detail="T5 model not loaded yet")
    async with _t5_sema:
        try:
            res = _t5_runner.moderate(text=req.text, max_new_tokens=req.max_new_tokens)
        except Exception as e:
            logger.exception("T5 moderation failed")
            raise HTTPException(status_code=500, detail=f"t5_failed: {type(e).__name__}: {e}")
    logger.debug("T5 result: verdict=%s raw=%s", res.verdict, res.raw)
    return T5ModerateResponse(verdict=res.verdict, raw=res.raw, model=res.model)


# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------
app.include_router(lg4_router)
app.include_router(t5_router)
