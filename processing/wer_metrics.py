"""Shared word-error-rate normalization and edit accounting for training."""

from __future__ import annotations

import re


def normalise_for_wer(text: str) -> list[str]:
    """Case-fold, remove punctuation, and split while preserving accents."""
    return re.sub(r"[^\w\s]", "", text.casefold()).split()


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Return substitutions, deletions, insertions, and reference word count."""
    ref, hyp = normalise_for_wer(reference), normalise_for_wer(hypothesis)
    table = [[(0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            candidates = [
                tuple(
                    table[i - 1][j - 1][k] + (1 if k == 0 else 0)
                    for k in range(3)
                ),
                tuple(
                    table[i - 1][j][k] + (1 if k == 1 else 0)
                    for k in range(3)
                ),
                tuple(
                    table[i][j - 1][k] + (1 if k == 2 else 0)
                    for k in range(3)
                ),
            ]
            table[i][j] = min(candidates, key=sum)
    substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions, len(ref)
