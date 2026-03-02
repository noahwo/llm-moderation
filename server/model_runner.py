# server/model_runner.py
from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
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


def _resolve_model_path(model_id: str) -> Tuple[str, bool]:
    """
    If model_id points to a local HF cache directory (e.g. models--.../snapshots/<hash>),
    pick the latest snapshot so we can load without hitting the network. Returns the
    resolved path and whether it is local.
    """

    maybe_path = Path(model_id)
    if not maybe_path.exists():
        return model_id, False

    snapshots_root = maybe_path / "snapshots"
    if snapshots_root.is_dir():
        snapshots = sorted(snapshots_root.iterdir(), reverse=True)
        if snapshots:
            return str(snapshots[0]), True

    return str(maybe_path), True


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
        self.model_ref = model_id  # keep original identifier for reporting
        self.model_path, is_local = _resolve_model_path(model_id)

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(torch_dtype.lower())
        if dtype is None:
            raise ValueError(f"Unsupported torch_dtype={torch_dtype}")

        load_kwargs = {"local_files_only": is_local}

        # Processor handles chat template + multimodal formatting
        self.processor = AutoProcessor.from_pretrained(self.model_path, **load_kwargs)

        # Model load (on GPU via device_map="auto" typically)
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            **load_kwargs,
        )
        self.model.eval()

        # Some cached configs miss attention_chunk_size; set a safe default to avoid mask errors.
        # We need to propagate the value to *all* sub-module configs because
        # create_chunked_causal_mask() reads from the inner Llama4TextModel.config,
        # not from the top-level Llama4ForConditionalGeneration.config.
        chunk_size = getattr(self.model.config, "attention_chunk_size", None)
        if chunk_size is None:
            chunk_size = (
                getattr(self.model.config, "sliding_window", None)
                or getattr(self.model.config, "max_position_embeddings", None)
                or 4096
            )
        # Always set on the top-level config and on text_config (Llama4Config wrapper)
        self.model.config.attention_chunk_size = chunk_size
        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is not None and getattr(text_cfg, "attention_chunk_size", None) is None:
            text_cfg.attention_chunk_size = chunk_size
        # Propagate to every sub-module that carries its own config object
        for module in self.model.modules():
            sub_cfg = getattr(module, "config", None)
            if sub_cfg is not None and sub_cfg is not self.model.config:
                if getattr(sub_cfg, "attention_chunk_size", None) is None:
                    sub_cfg.attention_chunk_size = chunk_size

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
        # Some processor versions take positional `conversation` and may not accept excluded_category_keys.
        apply_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if "excluded_category_keys" in inspect.signature(self.processor.apply_chat_template).parameters:
            apply_kwargs["excluded_category_keys"] = excluded_category_keys

        try:
            inputs = self.processor.apply_chat_template(messages, **apply_kwargs)
        except TypeError:
            # Fallback for older signature; drop excluded_category_keys if unsupported
            apply_kwargs.pop("excluded_category_keys", None)
            inputs = self.processor.apply_chat_template(messages, **apply_kwargs)

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
                use_cache=False,  # avoid StaticCache path that fails when sliding_window is None in this checkpoint
            )

        prompt_len = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_len:]
        decoded = self.processor.decode(generated_tokens, skip_special_tokens=True)

        verdict, categories = _parse_llamaguard_text(decoded)
        return ModerationResult(
            verdict=verdict,
            categories=categories,
            raw=decoded.strip(),
            model=self.model_ref,
        )