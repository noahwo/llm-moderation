# server/model_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration


@dataclass(frozen=True)
class ModerationResult:
    verdict: str                 # "safe" | "unsafe" | "unknown"
    categories: List[str]        # e.g. ["S9", "S2"]
    raw: str                     # raw decoded model output
    model: str


def _parse_llamaguard_text(text: str) -> Tuple[str, List[str]]:
    """
    Llama Guard outputs typically look like:
      "safe"
    or
      "unsafe\nS9\nS2"
    We parse conservatively.
    """
    raw = (text or "").strip()
    if not raw:
        return "unknown", []

    # Split by any whitespace/newlines, keep tokens
    tokens = raw.replace("\r", "\n").split()
    first = tokens[0].lower()

    verdict = "unknown"
    if first in ("safe", "unsafe"):
        verdict = first

    # Remaining tokens that look like categories, e.g. S1..S14, etc.
    cats: List[str] = []
    for t in tokens[1:]:
        t = t.strip().upper()
        if t.startswith("S") and len(t) <= 5:   # simple sanity check
            cats.append(t)

    # De-dup while preserving order
    seen = set()
    cats = [c for c in cats if not (c in seen or seen.add(c))]

    return verdict, cats


class ModelRunner:
    """
    Loads Llama Guard 4 once and provides a 'moderate' function.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Guard-4-12B",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(torch_dtype.lower())
        if dtype is None:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype}")

        # Processor handles chat template + multimodal formatting
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Model load (on GPU via device_map="auto" typically)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

        # A quick attribute for later
        self._dtype = dtype

    def moderate(
        self,
        messages: List[Dict[str, Any]],
        excluded_category_keys: Optional[List[str]] = None,
        max_new_tokens: int = 10,
        do_sample: bool = False,
    ) -> ModerationResult:
        """
        messages must match the HF blog structure, e.g.
        [{"role":"user","content":[{"type":"text","text":"..."}]}]
        """
        excluded_category_keys = excluded_category_keys or []

        # Apply template -> tokenized tensors
        # Note: processor.apply_chat_template supports excluded_category_keys in the blog.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            excluded_category_keys=excluded_category_keys,
        )

        # Move inputs to the same device as the model (device_map may shard,
        # but input tensors should go to the first parameter's device).
        # This is a practical approach for device_map="auto".
        model_device = next(self.model.parameters()).device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(model_device)

        with torch.inference_mode():
            # Keep generation deterministic (do_sample=False) like the blog snippet
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=bool(do_sample),
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_len:]
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True)

        verdict, categories = _parse_llamaguard_text(decoded)
        return ModerationResult(
            verdict=verdict,
            categories=categories,
            raw=decoded.strip(),
            model=self.model_id,
        )