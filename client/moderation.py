# client/moderation.py
"""
Strategy-based moderation layer.

Three concrete strategies share a common ``ModerationStrategy`` interface and
all return a normalised ``ModerationResult``:

  LlamaGuard4Strategy   – backed by the LlamaGuard-4 HTTP server (LGClient)
  ToxicChatT5Strategy   – backed by the ToxicChat-T5 HTTP server (T5Client)
  OpenAIModerationStrategy – backed by the OpenAI Moderation API
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests as _req
from dotenv import load_dotenv
from openai import OpenAI  # type: ignore[import]

try:
    from PIL import Image as _PILImage  # type: ignore[import]
except ImportError:
    _PILImage = None  # type: ignore[assignment]

from backends import LGClient, T5Client

# ---------------------------------------------------------------------------
# Unified result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModerationResult:
    """Normalised result returned by every ``ModerationStrategy``."""
    verdict: str                        # "safe" | "unsafe" | "unknown"
    flagged: bool                       # True if any category is violated
    categories: List[str]               # violated category names
    category_scores: Dict[str, float]   # per-category confidence (0-1); may be empty
    model: str                          # model / backend identifier
    raw: Any                            # raw API / server response for debugging

    def __str__(self) -> str:
        cats = ", ".join(self.categories) if self.categories else "none"
        return (
            f"ModerationResult(verdict={self.verdict!r}, flagged={self.flagged}, "
            f"categories=[{cats}], model={self.model!r})"
        )


# ---------------------------------------------------------------------------
# Abstract Strategy
# ---------------------------------------------------------------------------

class ModerationStrategy(ABC):
    """
    Abstract base for all moderation strategies.

    Concrete subclasses must implement ``moderate`` (text-only) and
    ``moderate_multimodal`` (text + optional images).  Override ``healthz``
    if the backend exposes a health endpoint.
    """

    @abstractmethod
    def moderate(self, text: str, **kwargs) -> ModerationResult:
        """Moderate a text-only input."""

    @abstractmethod
    def moderate_multimodal(
        self,
        text: str = "",
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> ModerationResult:
        """
        Moderate a mixed text + image input.

        ``images`` may contain any of:
          - a public URL string  (http / https)
          - a local file path    (str or pathlib.Path)
          - a PIL Image object
          - a data: URI string
        Strategies that do not support images should fall back to text-only.
        """

    def healthz(self) -> Dict[str, Any]:
        """Optional health-check; override if the backend supports it."""
        raise NotImplementedError(f"{type(self).__name__} does not expose a health endpoint")


# ---------------------------------------------------------------------------
# LlamaGuard-4 Strategy
# ---------------------------------------------------------------------------

class LlamaGuard4Strategy(ModerationStrategy):
    """Strategy backed by the LlamaGuard-4 HTTP server (``LGClient``)."""

    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self._client = LGClient(base_url, timeout_s=timeout_s)

    def healthz(self) -> Dict[str, Any]:
        return self._client.healthz()

    def moderate(self, text: str, **kwargs) -> ModerationResult:
        r = self._client.moderate(text, **kwargs)
        return ModerationResult(
            verdict=r.verdict,
            flagged=r.verdict.lower() == "unsafe",
            categories=r.categories,
            category_scores={},
            model=r.model,
            raw=r.raw,
        )

    def moderate_multimodal(
        self,
        text: str = "",
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> ModerationResult:
        r = self._client.moderate_multimodal(text, images, **kwargs)
        return ModerationResult(
            verdict=r.verdict,
            flagged=r.verdict.lower() == "unsafe",
            categories=r.categories,
            category_scores={},
            model=r.model,
            raw=r.raw,
        )


# ---------------------------------------------------------------------------
# ToxicChat-T5 Strategy
# ---------------------------------------------------------------------------

class ToxicChatT5Strategy(ModerationStrategy):
    """
    Strategy backed by the ToxicChat-T5 HTTP server (``T5Client``).

    This model is text-only; image inputs are silently ignored.
    """

    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self._client = T5Client(base_url, timeout_s=timeout_s)

    def healthz(self) -> Dict[str, Any]:
        return self._client.healthz()

    def moderate(self, text: str, **kwargs) -> ModerationResult:
        r = self._client.moderate(text, **kwargs)
        flagged = r.verdict.lower() == "toxic"
        return ModerationResult(
            verdict=r.verdict,
            flagged=flagged,
            categories=[r.verdict] if flagged else [],
            category_scores={},
            model=r.model,
            raw=r.raw,
        )

    def moderate_multimodal(
        self,
        text: str = "",
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> ModerationResult:
        # T5 is text-only — images are not supported; fall back gracefully
        return self.moderate(text, **kwargs)


# ---------------------------------------------------------------------------
# OpenAI Moderation Strategy
# ---------------------------------------------------------------------------

class OpenAIModerationStrategy(ModerationStrategy):
    """
    Strategy backed by the OpenAI Moderation API.

    Default model: ``omni-moderation-latest`` (supports text + images).
    Falls back gracefully to ``text-moderation-latest`` for text-only use when
    explicitly requested.

    Image inputs are converted to data: URIs (base64) for local files / PIL
    Images, or forwarded as-is for public URLs.

    Parameters
    ----------
    api_key:
        OpenAI API key.  If *None* the ``OPENAI_API_KEY`` environment variable
        is used automatically by the SDK.
    model:
        OpenAI moderation model to use.  Defaults to ``omni-moderation-latest``.
    """

    DEFAULT_MODEL = "omni-moderation-latest"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        load_dotenv()  # populate env from .env if present; SDK reads OPENAI_API_KEY automatically
        self._openai = OpenAI(api_key=api_key)
        self._model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_data_url(source: Any) -> str:
        """
        Convert an image source to a base64 data URI accepted by the OpenAI API.

        - http / https URL  → downloaded client-side, returned as base64 data URI
        - data: URI         → returned as-is
        - PIL Image         → base64 PNG data URI
        - local file path   → base64 data URI (JPEG / PNG / GIF / WebP)
        """
        # PIL Image
        if _PILImage is not None and isinstance(source, _PILImage.Image):
            buf = io.BytesIO()
            source.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"

        # Already a data URI
        if isinstance(source, str) and source.startswith("data:"):
            return source

        # Remote URL — download client-side and encode as base64 data URI so
        # that the OpenAI API receives the raw bytes rather than a URL it must
        # fetch itself (avoids access/firewall issues on the API side).
        # Use requests with a browser User-Agent; bare urllib gets 403 from
        # Wikimedia and many CDNs.
        if isinstance(source, str) and source.startswith(("http://", "https://")):
            resp = _req.get(
                source,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (compatible; llm-moderation/1.0)"},
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
            b64 = base64.b64encode(resp.content).decode()
            return f"data:{content_type};base64,{b64}"

        # Local file path (str or Path)
        path = Path(source)
        suffix = path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"data:{media_type};base64,{b64}"

    def _parse_response(self, response: Any) -> ModerationResult:
        """Normalise an ``openai.types.ModerationCreateResponse`` to ``ModerationResult``."""
        result = response.results[0]

        # categories and category_scores are Pydantic models — convert to dicts
        cats: Dict[str, bool] = result.categories.model_dump()
        scores: Dict[str, float] = result.category_scores.model_dump()

        violated = [k for k, v in cats.items() if v]
        flagged: bool = result.flagged
        verdict = "unsafe" if flagged else "safe"

        return ModerationResult(
            verdict=verdict,
            flagged=flagged,
            categories=violated,
            category_scores=scores,
            model=response.model,
            raw=response.model_dump(),
        )

    # ------------------------------------------------------------------
    # Strategy interface
    # ------------------------------------------------------------------

    def moderate(self, text: str, **kwargs) -> ModerationResult:
        """Moderate a text-only input via the OpenAI Moderation API."""
        response = self._openai.moderations.create(
            model=self._model,
            input=text,
        )
        return self._parse_response(response)

    def moderate_multimodal(
        self,
        text: str = "",
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> ModerationResult:
        """
        Moderate a mixed text + image input via the OpenAI Moderation API.

        If no images are provided, falls back to text-only moderation.
        Requires ``omni-moderation-latest`` (or a compatible snapshot).

        The OpenAI API accepts at most 1 image per request.  When multiple
        images are supplied, one request is issued per image (each paired with
        the same text) and the results are aggregated: flagged=True if any
        image is flagged, categories/scores are the union / max across all.
        """
        if not images:
            return self.moderate(text, **kwargs)

        results: List[ModerationResult] = []
        for img in images:
            url = self._to_data_url(img)
            content: List[Dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": url}},
            ]
            if text:
                content.append({"type": "text", "text": text})
            response = self._openai.moderations.create(
                model=self._model,
                input=content,
            )
            results.append(self._parse_response(response))

        # Aggregate: any flagged → flagged; union categories; max scores
        flagged = any(r.flagged for r in results)
        categories: List[str] = list(dict.fromkeys(
            cat for r in results for cat in r.categories
        ))
        scores: Dict[str, float] = {}
        for r in results:
            for k, v in r.category_scores.items():
                scores[k] = max(scores.get(k, 0.0), v)

        return ModerationResult(
            verdict="unsafe" if flagged else "safe",
            flagged=flagged,
            categories=categories,
            category_scores=scores,
            model=results[0].model,
            raw=[r.raw for r in results],
        )

