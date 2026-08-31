"""A local Whisper fork with a parallel token-level language-ID head."""

from .decode import TokenLanguageDecodingResult, decode_with_token_language
from .model import (
    DEFAULT_LANGUAGE_LABELS,
    WhisperTokenLID,
    load_token_lid_checkpoint,
    load_token_lid_model,
    save_token_lid_checkpoint,
)

__all__ = [
    "DEFAULT_LANGUAGE_LABELS",
    "TokenLanguageDecodingResult",
    "WhisperTokenLID",
    "decode_with_token_language",
    "load_token_lid_checkpoint",
    "load_token_lid_model",
    "save_token_lid_checkpoint",
]
