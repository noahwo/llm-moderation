# client/backends.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass(frozen=True)
class ModerationResponse:
    verdict: str
    categories: List[str]
    raw: str
    model: str


@dataclass(frozen=True)
class T5ModerationResponse:
    verdict: str   # "toxic" | "non-toxic" | "unknown"
    raw: str       # "positive" | "negative" | ...
    model: str


class LGClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def healthz(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/healthz", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _image_block(source: Any) -> Dict[str, Any]:
        """
        Convert an image source to a JSON-serialisable content block:
          str/Path  -> local file   => base64-encoded
          str       -> http(s) URL  => {"type":"image","url":"..."}   (server fetches)
          PIL Image                 => base64-encoded PNG
        """
        import base64
        import io
        from pathlib import Path as _Path

        # PIL Image
        try:
            from PIL import Image as _PILImage
            if isinstance(source, _PILImage.Image):
                buf = io.BytesIO()
                source.save(buf, format="PNG")
                return {
                    "type": "image",
                    "data": base64.b64encode(buf.getvalue()).decode(),
                    "media_type": "image/png",
                }
        except ImportError:
            pass

        # URL string
        if isinstance(source, str) and source.startswith(("http://", "https://", "data:")):
            return {"type": "image", "url": source}

        # Local file path (str or Path)
        path = _Path(source)
        suffix = path.suffix.lower()
        media_type = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png",  ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(suffix, "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode()
        return {"type": "image", "data": data, "media_type": media_type}

    def moderate(
        self,
        text: str,
        *,
        excluded_category_keys: Optional[List[str]] = None,
        max_new_tokens: int = 10,
        do_sample: bool = False,
    ) -> ModerationResponse:
        """Moderate a text-only message."""
        return self.moderate_multimodal(
            text=text,
            images=None,
            excluded_category_keys=excluded_category_keys,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    def moderate_multimodal(
        self,
        text: str = "",
        images: Optional[List[Any]] = None,
        *,
        excluded_category_keys: Optional[List[str]] = None,
        max_new_tokens: int = 10,
        do_sample: bool = False,
    ) -> ModerationResponse:
        """
        Moderate a message that may include images.

        images: list of local file paths (str/Path), public URLs, or PIL Images.
                Images are inserted before the text in the user content block.
        """
        content: List[Dict[str, Any]] = []
        for img in (images or []):
            content.append(self._image_block(img))
        if text:
            content.append({"type": "text", "text": text})

        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        r = requests.post(
            f"{self.base_url}/moderate",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        j = r.json()
        return ModerationResponse(
            verdict=j.get("verdict", "unknown"),
            categories=j.get("categories", []) or [],
            raw=j.get("raw", "") or "",
            model=j.get("model", "") or "",
        )


class T5Client:
    """HTTP client for the ToxicChat-T5 moderation server (server/t5_app.py)."""

    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def healthz(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/healthz", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def moderate(
        self,
        text: str,
        *,
        max_new_tokens: int = 10,
    ) -> T5ModerationResponse:
        payload: Dict[str, Any] = {
            "text": text,
            "max_new_tokens": max_new_tokens,
        }
        r = requests.post(
            f"{self.base_url}/moderate",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        j = r.json()
        return T5ModerationResponse(
            verdict=j.get("verdict", "unknown"),
            raw=j.get("raw", "") or "",
            model=j.get("model", "") or "",
        )


class CombinedClient:
    """
    Client for the combined moderation server (server/combined_app.py).

    Exposes both models through a single base URL:
      base_url/lg4/healthz   base_url/lg4/moderate
      base_url/t5/healthz    base_url/t5/moderate
      base_url/healthz       (overall health)

    Usage:
        cli = CombinedClient("http://10.251.68.79:18084")
        cli.healthz()                  # overall
        cli.lg4.moderate("some text")  # LlamaGuard-4
        cli.t5.moderate("some text")   # ToxicChat-T5
    """

    def __init__(self, base_url: str, *, lg4_timeout_s: float = 120.0, t5_timeout_s: float = 60.0) -> None:
        root = base_url.rstrip("/")
        self.lg4 = LGClient(f"{root}/lg4", timeout_s=lg4_timeout_s)
        self.t5  = T5Client(f"{root}/t5",  timeout_s=t5_timeout_s)
        self._root = root
        self._timeout = max(lg4_timeout_s, t5_timeout_s)

    def healthz(self) -> Dict[str, Any]:
        r = requests.get(f"{self._root}/healthz", timeout=self._timeout)
        r.raise_for_status()
        return r.json()