# server/model_runner.py
from __future__ import annotations

from dataclasses import dataclass
import base64
import inspect
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor, Llama4ForConditionalGeneration
try:
    from transformers.models.llama4.modeling_llama4 import Llama4VisionRotaryEmbedding as _Llama4VisionRotaryEmbedding
except ImportError:
    _Llama4VisionRotaryEmbedding = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LG4ModerationResult:
    verdict: str                 # "safe" | "unsafe" | "unknown"
    categories: List[str]        # e.g. ["S9", "S2"]
    raw: str                     # raw decoded model output
    model: str


def _load_images(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Eagerly resolve every image content block to a PIL.Image and store it under
    the ``"image"`` key.  ``apply_chat_template`` extracts values from content
    blocks using the key list ``["image", "url", "path", "base64"]``; by putting
    a real PIL object under ``"image"`` we bypass any network/FS access inside
    the processor and handle all source formats uniformly here.

    Accepted block formats (from the JSON payload):
      {"type": "image", "url": "https://..."}                     – HTTP/S URL
      {"type": "image", "url": "data:image/jpeg;base64,<b64>"}    – data URI
      {"type": "image", "data": "<b64>", "media_type": "..."}     – Anthropic style
      {"type": "image", "path": "/local/file.jpg"}                – local file path
    """
    try:
        from PIL import Image
        import requests as _req
    except ImportError as exc:
        raise RuntimeError(
            "Pillow and requests are required for image moderation"
        ) from exc

    result: List[Dict[str, Any]] = []
    for msg in messages:
        new_content: List[Dict[str, Any]] = []
        for block in msg.get("content", []):
            if block.get("type") != "image":
                new_content.append(block)
                continue

            if "path" in block:
                # Local file path
                img = Image.open(block["path"]).convert("RGB")

            elif "data" in block:
                # Anthropic-style base64: {"data": "<b64>", "media_type": "..."}
                img = Image.open(io.BytesIO(base64.b64decode(block["data"]))).convert("RGB")

            elif "url" in block:
                url = block["url"]
                if url.startswith("data:"):
                    # Data URI: data:image/jpeg;base64,<b64data>
                    _, b64 = url.split(",", 1)
                    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                else:
                    # HTTP/S URL — fetch here so the processor never needs to
                    resp = _req.get(
                        url,
                        timeout=30,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; llm-moderation/1.0)"},
                    )
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")

            else:
                raise ValueError(f"Image block missing 'url', 'data', or 'path': {block}")

            # Store under the "image" key — apply_chat_template extracts this
            # into batch_images which the processor's fetch_images handles natively.
            new_content.append({"type": "image", "image": img})

        result.append({**msg, "content": new_content})

    return result


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
        device_map: str = "cuda",
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

        # Model load onto CUDA explicitly
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=device_map,
            **load_kwargs,
        )
        self.model.eval()

        # Fix meta-device tensors left after device_map loading.
        #
        # Llama4VisionRotaryEmbedding stores `freqs_ci` as a plain Python
        # attribute (self.freqs_ci = ...), NOT via register_buffer().  Under
        # init_empty_weights() / device_map loading accelerate only restores
        # state_dict tensors; non-persistent plain attributes are never moved
        # and stay as meta tensors forever.  The _buffers dict won't contain
        # freqs_ci, so we must handle it separately by re-running __init__.
        _model_device = torch.device("cuda")
        if _Llama4VisionRotaryEmbedding is not None:
            # Llama4VisionRotaryEmbedding.__init__ takes a Llama4VisionConfig but
            # does NOT store it as self.config — so _mod.config doesn't exist.
            # Retrieve it from the top-level model config instead.
            _vision_cfg = (
                getattr(self.model.config, "vision_config", None)
                or getattr(self.model.config, "vision_model_config", None)
            )
            for _mod in self.model.modules():
                if isinstance(_mod, _Llama4VisionRotaryEmbedding):
                    fci = getattr(_mod, "freqs_ci", None)
                    if fci is not None and isinstance(fci, torch.Tensor) and fci.device.type == "meta":
                        if _vision_cfg is None:
                            raise RuntimeError(
                                "Cannot recompute freqs_ci: vision_config not found in model.config"
                            )
                        # Re-run __init__ outside init_empty_weights context so
                        # freqs_ci is computed as real CPU tensors, then move to CUDA.
                        _Llama4VisionRotaryEmbedding.__init__(_mod, _vision_cfg)
                        _mod.freqs_ci = _mod.freqs_ci.to(_model_device)
                        logger.info(
                            "Recomputed Llama4VisionRotaryEmbedding.freqs_ci -> %s",
                            _model_device,
                        )

        # Also sweep registered buffers in _buffers for any other meta tensors
        # (belt-and-suspenders for future transformers changes).
        for _mod in self.model.modules():
            for _buf_name, _buf in list(_mod._buffers.items()):
                if _buf is not None and _buf.device.type == "meta":
                    _mod._buffers[_buf_name] = torch.zeros(
                        _buf.shape, dtype=_buf.dtype, device=_model_device
                    )
                    logger.info(
                        "Materialized meta buffer %s.%s -> %s",
                        type(_mod).__name__, _buf_name, _model_device,
                    )

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
          {"type": "image", "url": "https://..."}                  – HTTP/S URL
          {"type": "image", "url": "data:image/jpeg;base64,<b64>"} – data URI
          {"type": "image", "data": "<b64>", "media_type": "..."}  – Anthropic base64
          {"type": "image", "path": "/local/file.jpg"}             – local file
        """
        excluded_category_keys = excluded_category_keys or []

        # Pre-load all image blocks (URL / data-URI / base64 / local path) into
        # PIL images stored under the "image" key.  apply_chat_template extracts
        # content-block values using keys ["image","url","path","base64"], so a
        # real PIL object under "image" is handled natively by fetch_images
        # without relying on the processor to do any network/FS access itself.
        # Text-only messages are returned unchanged; no pixel_values are produced.
        messages = _load_images(messages)

        # LlamaGuard-4's Jinja chat template inserts <|image|> placeholder tokens
        # only when a text block is also present in the same message.  If images
        # are submitted without any accompanying text (pure-image moderation), the
        # template produces 0 placeholders while the processor still extracts the
        # PIL images, causing a count-mismatch ValueError.  Guard against this by
        # ensuring every message that carries image blocks also has a text block
        # (empty string is sufficient — it just anchors the image tokens).
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                has_img = any(b.get("type") == "image" for b in content)
                has_txt = any(b.get("type") == "text" for b in content)
                if has_img and not has_txt:
                    content.append({"type": "text", "text": ""})

        # Single-call pattern per the Llama Guard 4 model card.
        # apply_chat_template(tokenize=True, return_dict=True) renders the chat
        # template, tokenizes, and injects pixel_values — but ONLY when image
        # blocks with resolvable content are present in messages.  Text-only
        # requests produce no vision keys, so the vision encoder is never called.
        template_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            # Do NOT add images= here — processor extracts from message content.
        }
        if "excluded_category_keys" in inspect.signature(self.processor.apply_chat_template).parameters:
            template_kwargs["excluded_category_keys"] = excluded_category_keys

        try:
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("excluded_category_keys", None)
            inputs = self.processor.apply_chat_template(messages, **template_kwargs)

        # Move all tensors to CUDA.
        inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}

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