"""Greedy Whisper decoding that records a language prediction for every emitted BPE token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from whisper.tokenizer import get_tokenizer

from .model import WhisperTokenLID


@dataclass
class TokenLanguageDecodingResult:
    text: str
    clip_language: str
    token_ids: list[int]
    token_languages: list[str]
    token_language_confidences: list[float]
    words: list[dict[str, object]]


@torch.no_grad()
def decode_with_token_language(
    model: WhisperTokenLID,
    mel: torch.Tensor,
    *,
    language: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> TokenLanguageDecodingResult:
    """Greedily transcribe one <=30-second mel and emit a label for each BPE token.

    This intentionally starts with greedy decoding. Beam search needs additional
    bookkeeping to retain language-label histories when candidate beams are reordered.
    """
    if mel.ndim == 3 and mel.shape[0] != 1:
        raise ValueError("decode_with_token_language accepts one audio clip at a time")

    return decode_batch_with_token_language(
        model, mel, language=language, max_tokens=max_tokens
    )[0]


def _build_result(
    model,
    tokenizer,
    generated: list[int],
    language_probabilities: list[torch.Tensor],
    language: str,
) -> TokenLanguageDecodingResult:
    """Convert one generated sequence into the established review structure."""
    token_languages = [model.language_labels[int(probs.argmax())] for probs in language_probabilities]
    token_confidences = [float(probs.max()) for probs in language_probabilities]
    try:
        words, word_token_groups = tokenizer.split_to_word_tokens(generated)
    except IndexError:
        words = [tokenizer.decode(generated)] if generated else []
        word_token_groups = [generated] if generated else []
    merged_words: list[str] = []
    merged_token_groups: list[list[int]] = []
    for word, group in zip(words, word_token_groups):
        if word.startswith(("-", "‐", "‑", "–", "—")) and merged_words:
            merged_words[-1] += word
            merged_token_groups[-1].extend(group)
        else:
            merged_words.append(word)
            merged_token_groups.append(list(group))
    word_results: list[dict[str, object]] = []
    offset = 0
    for word, group in zip(merged_words, merged_token_groups):
        count = len(group)
        probs = torch.stack(language_probabilities[offset : offset + count]).mean(dim=0)
        word_results.append({
            "word": word.strip(),
            "language_id": model.language_labels[int(probs.argmax())],
            "confidence": float(probs.max()),
            "token_ids": group,
        })
        offset += count
    return TokenLanguageDecodingResult(
        text=tokenizer.decode(generated).strip(),
        clip_language=language,
        token_ids=generated,
        token_languages=token_languages,
        token_language_confidences=token_confidences,
        words=word_results,
    )


@torch.no_grad()
def decode_batch_with_token_language(
    model: WhisperTokenLID,
    mels: torch.Tensor,
    *,
    language: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> list[TokenLanguageDecodingResult]:
    """Greedily transcribe a batch, with independent EOS stopping per clip."""
    if mels.ndim == 2:
        mels = mels.unsqueeze(0)
    if mels.ndim != 3 or mels.shape[0] == 0:
        raise ValueError("Expected mel spectrograms shaped (batch, mels, frames)")

    model.eval()
    mels = mels.to(model.device)
    audio_features = model.encoder(mels)
    probe_tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    if language is None:
        _, probabilities = model.detect_language(audio_features, probe_tokenizer)
        languages = [max(item, key=item.get) for item in probabilities]
    else:
        languages = [language] * mels.shape[0]
    tokenizers = [
        get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=item_language,
            task="transcribe",
        )
        for item_language in languages
    ]

    generated: list[list[int]] = [[] for _ in tokenizers]
    language_probabilities: list[list[torch.Tensor]] = [[] for _ in tokenizers]
    tokens = torch.tensor(
        [tokenizer.sot_sequence_including_notimestamps for tokenizer in tokenizers],
        device=model.device,
    )
    finished = [False] * len(tokenizers)
    eot = tokenizers[0].eot
    limit = max_tokens or model.dims.n_text_ctx // 2
    for _ in range(limit):
        token_logits, language_logits = model.logits_with_language(tokens, audio_features)
        next_token_logits = token_logits[:, -1].clone()
        next_token_logits[:, eot + 1 :] = -torch.inf
        next_tokens = next_token_logits.argmax(dim=-1)
        if any(finished):
            finished_mask = torch.tensor(finished, dtype=torch.bool, device=model.device)
            next_tokens = torch.where(
                finished_mask, torch.full_like(next_tokens, eot), next_tokens
            )
        next_token_values = next_tokens.tolist()
        for index, next_token in enumerate(next_token_values):
            if finished[index] or next_token == eot:
                finished[index] = True
                continue
            generated[index].append(next_token)
            language_probabilities[index].append(
                F.softmax(language_logits[index, -1], dim=-1).cpu()
            )
        if all(finished):
            break
        tokens = torch.cat([tokens, next_tokens.unsqueeze(1)], dim=1)

    return [
        _build_result(model, tokenizer, token_ids, probs, item_language)
        for tokenizer, token_ids, probs, item_language in zip(
            tokenizers, generated, language_probabilities, languages
        )
    ]
