# server/model_runner.py
from __future__ import annotations

from dataclasses import dataclass
import base64
import inspect
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration


@dataclass(frozen=True)
class LG4ModerationResult:
    verdict: str                 # "safe" | "unsafe" | "unknown"
    categories: List[str]        # e.g. ["S9", "S2"]
    raw: str                     # raw decoded model output
    model: str


def _extract_images(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """
    Walk the message list and pull out image content blocks.

    Accepted block formats (in the JSON payload):
      {"type": "image", "url": "https://..."}           – public URL
      {"type": "image", "data": "<base64>",
       "media_type": "image/jpeg"}                      – base64-encoded bytes

    Each image block is replaced with the bare {"type": "image"} sentinel
    that Llama 4's processor expects, and the decoded PIL.Image is appended
    to the returned images list (in document order).
    """
    try:
        from PIL import Image
        import requests as _req
    except ImportError as e:
        raise RuntimeError("Pillow and requests are required for image moderation") from e

    cleaned: List[Dict[str, Any]] = []
    images: List[Any] = []

    for msg in messages:
        new_content: List[Dict[str, Any]] = []
        for block in msg.get("content", []):
            if block.get("type") != "image":
                new_content.append(block)
                continue

            # Resolve the image to a PIL Image object
            if "data" in block:
                # base64-encoded bytes
                raw = base64.b64decode(block["data"])
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            elif "url" in block:
                url = block["url"]
                if url.startswith("data:"):
                    # data URI: data:image/jpeg;base64,<data>
                    header, b64 = url.split(",", 1)
                    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                else:
                    resp = _req.get(
                        url,
                        timeout=30,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; llm-moderation/1.0)"},
                    )
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                raise ValueError(f"Image block missing 'data' or 'url': {block}")

            images.append(img)
            # Replace with the bare sentinel the processor expects
            new_content.append({"type": "image"})

        cleaned.append({**msg, "content": new_content})

    return cleaned, images


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


class LG4ModelRunner:
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
    ) -> LG4ModerationResult:
        """
        messages must match the HF blog structure, e.g.
        [{"role":"user","content":[{"type":"text","text":"..."}]}]

        Image blocks are supported inside content:
          {"type": "image", "url": "https://..."}          – public URL
          {"type": "image", "data": "<base64>",
           "media_type": "image/jpeg"}                     – base64-encoded
        """
        excluded_category_keys = excluded_category_keys or []

        # Decode image blocks and replace with bare {"type":"image"} sentinels
        messages, images = _extract_images(messages)

        # Step 1: render the chat template to a plain text string (no tokenization yet).
        # apply_chat_template does NOT accept images= — images are passed to the
        # processor call in step 2.
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if "excluded_category_keys" in inspect.signature(self.processor.apply_chat_template).parameters:
            template_kwargs["excluded_category_keys"] = excluded_category_keys

        try:
            prompt_text: str = self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("excluded_category_keys", None)
            prompt_text = self.processor.apply_chat_template(messages, **template_kwargs)

        # Step 2: tokenize — pass images here so the processor can insert pixel values.
        proc_kwargs: Dict[str, Any] = {
            "return_tensors": "pt",
        }
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(text=prompt_text, **proc_kwargs)

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
        return LG4ModerationResult(
            verdict=verdict,
            categories=categories,
            raw=decoded.strip(),
            model=self.model_ref,
        )