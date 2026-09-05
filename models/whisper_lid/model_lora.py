"""PEFT LoRA variant of the local Whisper token-language-ID model.

LoRA adapters are attached to every linear projection in Whisper's encoder and
decoder.  The parallel ``decoder.language_head`` is deliberately excluded from
LoRA and is retained as a normally trainable/saved module.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn

from .model import DEFAULT_LANGUAGE_LABELS, WhisperTokenLID, load_token_lid_model


def encoder_decoder_linear_targets(model: WhisperTokenLID) -> list[str]:
    """Return all Whisper encoder/decoder ``nn.Linear`` paths except the LID head."""
    targets = []
    for name, module in model.named_modules():
        is_whisper_transformer_projection = name.startswith(("encoder.", "decoder."))
        is_lid_head = name == "decoder.language_head"
        if is_whisper_transformer_projection and not is_lid_head and isinstance(module, nn.Linear):
            targets.append(name)
    if not targets:
        raise RuntimeError("No Whisper encoder/decoder linear projections were found for LoRA")
    return targets


def load_token_lid_lora_model(
    name: str = "small",
    *,
    language_labels: Sequence[str] = DEFAULT_LANGUAGE_LABELS,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    device: Optional[str | torch.device] = None,
    download_root: Optional[str] = None,
):
    """Load Whisper plus token LID, with trainable LoRA on encoder/decoder only.

    The official Whisper weights stay frozen. PEFT trains its low-rank adapter
    weights and the new ``decoder.language_head``; no other base weights are
    trainable. The returned object is a PEFT ``PeftModel`` that delegates the
    local fork API (including ``logits_with_language``) to its base model.
    """
    if rank < 1 or alpha < 1:
        raise ValueError("rank and alpha must both be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - exercised only without the optional runtime dependency.
        raise ImportError(
            "PEFT is required for the LoRA model. Run `uv sync --project processing`."
        ) from error

    base_model = load_token_lid_model(
        name,
        language_labels=language_labels,
        device=device,
        download_root=download_root,
    )
    config = LoraConfig(
        base_model_name_or_path=name,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=encoder_decoder_linear_targets(base_model),
        # This randomly initialised classifier must be optimized and persisted,
        # but must not receive a LoRA adapter itself.
        modules_to_save=["decoder.language_head"],
    )
    lora_model = get_peft_model(base_model, config)
    # OpenAI Whisper does not expose Transformers' ``config._name_or_path``,
    # so PEFT replaces this value with ``None`` during wrapping. Restore it so
    # the serialized adapter remains self-describing.
    lora_model.peft_config["default"].base_model_name_or_path = name
    lora_model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return lora_model


def save_token_lid_lora_adapter(model: Any, path: Path | str) -> None:
    """Save a compact PEFT adapter plus metadata needed by this local fork.

    PEFT serializes only LoRA matrices and ``modules_to_save`` (the language-ID
    head), not a duplicate copy of Whisper's frozen base weights.
    """
    path = Path(path)
    base_model = model.get_base_model()
    if not isinstance(base_model, WhisperTokenLID):
        raise TypeError("Expected a PEFT adapter whose base model is WhisperTokenLID")
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    metadata = {
        "format": "whisper-token-lid-lora-v1",
        "base_model": model.peft_config["default"].base_model_name_or_path,
        "dims": asdict(base_model.dims),
        "language_labels": list(base_model.language_labels),
    }
    (path / "whisper_token_lid_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_token_lid_lora_adapter(
    path: Path | str,
    *,
    device: Optional[str | torch.device] = None,
    download_root: Optional[str] = None,
):
    """Reload a saved local Whisper LoRA adapter for continued training."""
    path = Path(path)
    metadata_path = path / "whisper_token_lid_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing LoRA metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        from peft import PeftModel
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "PEFT is required for the LoRA model. Run `uv sync --project processing`."
        ) from error
    base_model = load_token_lid_model(
        metadata["base_model"],
        language_labels=metadata["language_labels"],
        device=device,
        download_root=download_root,
    )
    model = PeftModel.from_pretrained(base_model, str(path), is_trainable=True)
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model
