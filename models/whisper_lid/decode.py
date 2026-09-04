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
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)
    if mel.shape[0] != 1:
        raise ValueError("decode_with_token_language accepts one audio clip at a time")

    model.eval()
    mel = mel.to(model.device)
    audio_features = model.encoder(mel)
    probe_tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    if language is None:
        _, probabilities = model.detect_language(audio_features, probe_tokenizer)
        probabilities = probabilities[0]
        language = max(probabilities, key=probabilities.get)
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language=language, task="transcribe")

    generated: list[int] = []
    language_probabilities: list[torch.Tensor] = []
    tokens = torch.tensor([tokenizer.sot_sequence_including_notimestamps], device=model.device)
    limit = max_tokens or model.dims.n_text_ctx // 2
    for _ in range(limit):
        token_logits, language_logits = model.logits_with_language(tokens, audio_features)
        next_token_logits = token_logits[:, -1].clone()
        # `eot` is the first special token; permit it, but never generate a control/timestamp token.
        next_token_logits[:, tokenizer.eot + 1 :] = -torch.inf
        next_token = int(next_token_logits.argmax(dim=-1).item())
        if next_token == tokenizer.eot:
            break
        generated.append(next_token)
        language_probabilities.append(F.softmax(language_logits[0, -1], dim=-1).cpu())
        tokens = torch.cat([tokens, torch.tensor([[next_token]], device=model.device)], dim=-1)

    token_languages = [model.language_labels[int(probs.argmax())] for probs in language_probabilities]
    token_confidences = [float(probs.max()) for probs in language_probabilities]
    try:
        words, word_token_groups = tokenizer.split_to_word_tokens(generated)
    except IndexError:
        # During early fine-tuning, greedy decoding can stop after an incomplete
        # multi-byte UTF-8 token sequence.  openai-whisper's Unicode word
        # splitter assumes the full sequence is valid and indexes beyond the
        # decoded string in that case.  Keep evaluation running; the decoded
        # transcript above is still suitable for WER, while this fallback
        # reports one aggregate language-labelled span in the review output.
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
    words, word_token_groups = merged_words, merged_token_groups
    word_results: list[dict[str, object]] = []
    offset = 0
    for word, group in zip(words, word_token_groups):
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
