# client/client.py
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


class LGClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0) -> None:
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
        excluded_category_keys: Optional[List[str]] = None,
        max_new_tokens: int = 10,
        do_sample: bool = False,
    ) -> ModerationResponse:
        payload: Dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
            # "excluded_category_keys": excluded_category_keys,
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