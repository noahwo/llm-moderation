# server/t5_model_runner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import logging
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)

# T5 encoder-decoder models do not work correctly with device_map="auto"
# (cross-attention breaks when encoder/decoder land on different devices).
# Always load onto a single CUDA device, exactly as the model card shows.
_DEVICE = "cuda"


def _hf_token() -> str | None:
    """Return HF token from environment, checking both common variable names."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


@dataclass(frozen=True)
class T5ModerationResult:
    verdict: str         # "toxic" | "non-toxic" | "unknown"
    raw: str             # raw decoded output, e.g. "positive" / "negative"
    model: str


def _parse_t5_output(text: str) -> str:
    """
    ToxicChat-T5 outputs:
      "positive"  -> toxic
      "negative"  -> non-toxic
    Returns a normalised verdict.
    """
    t = text.strip().lower()
    if t == "positive":
        return "toxic"
    if t == "negative":
        return "non-toxic"
    return "unknown"


def _resolve_model_path(model_id: str) -> Tuple[str, bool]:
    """Mirror of the LlamaGuard helper: resolve local HF cache snapshots."""
    maybe = Path(model_id)
    if not maybe.exists():
        return model_id, False

    snapshots_root = maybe / "snapshots"
    if snapshots_root.is_dir():
        snapshots = sorted(snapshots_root.iterdir(), reverse=True)
        if snapshots:
            return str(snapshots[0]), True

    return str(maybe), True


_INPUT_PREFIX = "ToxicChat: "

# The model card explicitly loads the tokenizer from the BASE t5-large, not
# from the fine-tuned checkpoint. The checkpoint's tokenizer_config.json sets
# decoder_start_token_id to the span-corruption sentinel <extra_id_0>, which
# makes the model do span-filling instead of classification.
_BASE_TOKENIZER_ID = "t5-large"


def _resolve_t5_tokenizer_path() -> str:
    """
    Find a local t5-large tokenizer cache to avoid hitting the network.
    Falls back to the HF hub id if nothing is cached locally.
    """
    import os
    candidates = []
    for env in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        root = os.environ.get(env)
        if root:
            candidates.append(os.path.join(root, "hub", "models--t5-large"))
            candidates.append(os.path.join(root, "models--t5-large"))
    # common default cache
    candidates.append(os.path.expanduser("~/.cache/huggingface/hub/models--t5-large"))

    for c in candidates:
        p = Path(c)
        snapshots = p / "snapshots"
        if snapshots.is_dir():
            snaps = sorted(snapshots.iterdir(), reverse=True)
            if snaps:
                return str(snaps[0])
        if p.exists():
            return str(p)

    return _BASE_TOKENIZER_ID  # will fetch from HF hub


class T5ModelRunner:
    """
    Loads ToxicChat-T5-Large once and exposes a `moderate` method.

    NOTE: T5 is an encoder-decoder model. device_map="auto" can split the
    encoder and decoder across devices, breaking cross-attention and causing
    the model to output span-corruption sentinels instead of labels. We
    always load onto a single device via .to(), matching the model card.
    """

    def __init__(
        self,
        model_id: str = "lmsys/toxicchat-t5-large-v1.0",
        torch_dtype: str = "float32",  # model weights are native F32; keep as-is
        device_map: str = "cuda",      # ignored for T5; single-device .to("cuda") is used
    ) -> None:
        self.model_ref = model_id
        self.model_path, is_local = _resolve_model_path(model_id)
        self._device = _DEVICE

        token = _hf_token()
        # Only force local_files_only for resolved local snapshot dirs.
        # For hub IDs (is_local=False) we go online and use the token.
        load_kwargs: dict = {"token": token}
        if is_local:
            load_kwargs["local_files_only"] = True

        # Load tokenizer from the BASE t5-large, exactly as the model card:
        #   AutoTokenizer.from_pretrained("t5-large")
        # The fine-tuned checkpoint's tokenizer_config.json sets
        # decoder_start_token_id to <extra_id_0> (32099), hijacking generation.
        tok_path = _resolve_t5_tokenizer_path()
        tok_local = tok_path != _BASE_TOKENIZER_ID
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path,
            local_files_only=tok_local,
            token=token,
        )

        # Load model weights from hub (or local snapshot). Native float32, single device.
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        ).to(self._device)
        self.model.eval()

        # Belt-and-suspenders: reset decoder_start_token_id to pad (0)
        # in case the checkpoint config still carries 32099.
        pad_id = self.tokenizer.pad_token_id or 0
        orig = self.model.config.decoder_start_token_id
        self.model.config.decoder_start_token_id = pad_id
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.decoder_start_token_id = pad_id
            self.model.generation_config.forced_bos_token_id = None
        logger.info("T5 decoder_start_token_id reset: %s -> %s  (device: %s)", orig, pad_id, self._device)

    def moderate(self, text: str, max_new_tokens: int = 10) -> T5ModerationResult:
        """
        Classify a single input string.

        Returns T5ModerationResult with verdict "toxic" | "non-toxic" | "unknown".
        """
        inputs = self.tokenizer.encode(
            _INPUT_PREFIX + text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.inference_mode():
            outputs = self.model.generate(inputs)

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        raw = self.tokenizer.decode(sequences[0], skip_special_tokens=True).strip()
        verdict = _parse_t5_output(raw)

        return T5ModerationResult(verdict=verdict, raw=raw, model=self.model_ref)
