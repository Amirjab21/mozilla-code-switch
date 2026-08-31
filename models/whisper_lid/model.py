"""Whisper Small-compatible model with a token-level language-ID classifier."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import Tensor

import whisper
from whisper.model import Linear, ModelDimensions, TextDecoder, Whisper


# These are model outputs, not Whisper vocabulary tokens. Keep this mapping with every checkpoint.
DEFAULT_LANGUAGE_LABELS = ("eng", "spa", "mixed", "other")


class TokenLIDTextDecoder(TextDecoder):
    """The standard decoder plus a classifier over its final hidden state."""

    def __init__(self, *args, language_labels: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        if not language_labels:
            raise ValueError("language_labels must contain at least one class")
        # Whisper's own Linear casts its weight to the activation dtype, which
        # keeps this head compatible with fp16 inference as well as fp32 training.
        self.language_head = Linear(self.positional_embedding.shape[-1], len(language_labels))

    def forward_with_language(
        self, tokens: Tensor, audio_features: Tensor, kv_cache: Optional[dict] = None
    ) -> tuple[Tensor, Tensor]:
        """Return normal next-token logits and parallel language-class logits."""
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        hidden = self.token_embedding(tokens) + self.positional_embedding[
            offset : offset + tokens.shape[-1]
        ]
        hidden = hidden.to(audio_features.dtype)
        for block in self.blocks:
            hidden = block(hidden, audio_features, mask=self.mask, kv_cache=kv_cache)

        hidden = self.ln(hidden)
        token_logits = (hidden @ self.token_embedding.weight.to(hidden.dtype).T).float()
        language_logits = self.language_head(hidden).float()
        return token_logits, language_logits

    def forward(self, tokens: Tensor, audio_features: Tensor, kv_cache: Optional[dict] = None) -> Tensor:
        # Preserve Whisper's ordinary public API, including clip-level language detection.
        token_logits, _ = self.forward_with_language(tokens, audio_features, kv_cache)
        return token_logits


class WhisperTokenLID(Whisper):
    """A Whisper model that retains ASR behaviour and adds token-language logits."""

    def __init__(self, dims: ModelDimensions, language_labels: Sequence[str] = DEFAULT_LANGUAGE_LABELS):
        super().__init__(dims)
        self.language_labels = tuple(language_labels)
        self.decoder = TokenLIDTextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            language_labels=self.language_labels,
        )

    def logits_with_language(self, tokens: Tensor, audio_features: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits for the next transcript token and its language class."""
        return self.decoder.forward_with_language(tokens, audio_features)


def load_token_lid_model(
    name: str = "small",
    *,
    language_labels: Sequence[str] = DEFAULT_LANGUAGE_LABELS,
    device: Optional[str | torch.device] = None,
    download_root: Optional[str] = None,
) -> WhisperTokenLID:
    """Load official Whisper weights, leaving only the new classifier randomly initialised."""
    base_model = whisper.load_model(name, device=device, download_root=download_root)
    model = WhisperTokenLID(base_model.dims, language_labels)
    incompatibility = model.load_state_dict(base_model.state_dict(), strict=False)
    expected = {"decoder.language_head.weight", "decoder.language_head.bias"}
    if set(incompatibility.missing_keys) != expected or incompatibility.unexpected_keys:
        raise RuntimeError(f"Unexpected base-checkpoint mismatch: {incompatibility}")
    del base_model
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def save_token_lid_checkpoint(model: WhisperTokenLID, path: Path | str) -> None:
    """Save a self-describing fine-tuned fork checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "whisper-token-lid-v1",
            "dims": asdict(model.dims),
            "language_labels": model.language_labels,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def load_token_lid_checkpoint(path: Path | str, *, device: Optional[str | torch.device] = None) -> WhisperTokenLID:
    """Load a checkpoint written by :func:`save_token_lid_checkpoint`."""
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    if checkpoint.get("format") != "whisper-token-lid-v1":
        raise ValueError("Not a whisper-token-lid-v1 checkpoint")
    model = WhisperTokenLID(
        ModelDimensions(**checkpoint["dims"]), checkpoint["language_labels"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
