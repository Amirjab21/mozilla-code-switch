"""Convert Miami's word-level language annotations into Whisper BPE supervision."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence

from whisper.tokenizer import Tokenizer


IGNORE_INDEX = -100
DEFAULT_CLASS_TO_ID = {"eng": 0, "spa": 1, "mixed": 2, "other": 3}


def canonical_language_id(source_label: str) -> str | None:
    """Map corpus labels to learnable classes; return None for ambiguous supervision."""
    source_label = source_label.strip().casefold()
    if source_label in {"eng", "spa"}:
        return source_label
    if source_label in {"eng+spa", "spa+eng"}:
        return "mixed"
    if source_label in {"other", "unk"}:
        return "other"
    # `eng&spa` is the corpus's undetermined category, not a true bilingual word.
    return None


def normalise_word(text: str) -> str:
    return "".join(char for char in text.strip() if not unicodedata.category(char).startswith("P")).casefold()


def make_token_language_targets(
    transcript: str,
    word_langids: Sequence[dict[str, str]],
    tokenizer: Tokenizer,
    *,
    class_to_id: dict[str, int] = DEFAULT_CLASS_TO_ID,
) -> tuple[list[int], list[int]]:
    """Return BPE IDs and one language target per BPE ID.

    All BPE pieces of a source word share that word's label. Ambiguous words and
    special tokens use ``IGNORE_INDEX`` and therefore contribute no LID loss.
    """
    token_ids = tokenizer.encode(" " + transcript.strip())
    decoded_words, bpe_groups = tokenizer.split_to_word_tokens(token_ids)
    source_words = [normalise_word(item["word"]) for item in word_langids]
    decoded_words = [normalise_word(word) for word in decoded_words]
    if decoded_words != source_words:
        # A uniformly labelled transcript does not require word-boundary
        # reconstruction: every BPE token receives the same class. This is
        # important for corpora such as the Indonesian/Javanese data, whose
        # labels are all `other` and whose punctuation/hyphens Whisper may
        # split into separate word groups.
        languages = [canonical_language_id(item["langid"]) for item in word_langids]
        unique_languages = set(languages)
        if len(unique_languages) == 1 and None not in unique_languages:
            language = languages[0]
            return token_ids, [class_to_id[language]] * len(token_ids)
        raise ValueError(
            "Whisper tokenisation did not round-trip to word_langids; refusing to create misaligned labels."
        )

    targets: list[int] = []
    for source, bpe_group in zip(word_langids, bpe_groups):
        language = canonical_language_id(source["langid"])
        target = class_to_id[language] if language is not None else IGNORE_INDEX
        targets.extend([target] * len(bpe_group))
    if len(token_ids) != len(targets):
        raise RuntimeError("BPE token and language-target counts differ")
    return token_ids, targets
