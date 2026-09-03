# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "jiwer",
#   "pandas",
#   "typer",
# ]
# ///
"""Shared transcript normalization and standalone submission WER scorer."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


BRACKETED = re.compile(r"\[[^\]]+\]")
UNINTELLIGIBLE_PAREN = re.compile(r"\(\?+\)")
WORD_PAREN = re.compile(r"\(([^()]*)\)")
PUNCTUATION_OTHER = re.compile('[¿¡";:]+')
COMMA = re.compile(r",+")
# [^\W\d_] matches any Unicode letter, including accented capitals.
SENTENCE_INITIAL = re.compile(r"(^\s*|[.!?—]\s*)([^\W\d_])([^\W\d_]?)")
SENTENCE_END = re.compile(r"[!?]+")
MULTISPACE = re.compile(r"  +")


def _lowercase_sentence_initial(match: re.Match[str]) -> str:
    """Lowercase a sentence-initial letter unless it begins an acronym."""
    delimiter, first, second = match.group(1), match.group(2), match.group(3)
    if first.isupper() and not (second and second.isupper()):
        first = first.lower()
    return delimiter + first + second


def normalize_text(text: str) -> str:
    """Normalize a transcript to the canonical form used for WER scoring."""
    text = str(text).replace("~", "")
    text = re.sub(BRACKETED, " ", text)
    text = re.sub(UNINTELLIGIBLE_PAREN, " ", text)
    text = re.sub(WORD_PAREN, r"\1", text)
    text = text.replace("#x27;", "'")
    text = re.sub(PUNCTUATION_OTHER, " ", text)
    text = re.sub(SENTENCE_INITIAL, _lowercase_sentence_initial, text)
    text = text.replace("—", ", ")
    text = re.sub(COMMA, " ", text)
    text = re.sub(SENTENCE_END, " ", text)
    text = text.replace("...", "!ELLIPSIS!").replace(".", " ").replace("!ELLIPSIS!", "...")
    while " ... " in text:
        text = text.replace(" ... ", " ")
    return re.sub(MULTISPACE, " ", text)


def _word_error_rate(predicted, actual, normalize_function: Callable[[str], str]) -> float:
    """Compute corpus WER after normalizing predictions and references."""
    import jiwer

    normalized_predictions = [normalize_function(text) for text in predicted[:, 0]]
    normalized_references = [normalize_function(text) for text in actual[:, 0]]
    return jiwer.wer(normalized_references, normalized_predictions)


def main(
    actual_path: Path,
    predicted_path: Path = Path("runtime/submission/submission.csv"),
) -> None:
    """Score a submission CSV against ground truth, aligned by audio filename."""
    import pandas as pd
    import typer

    predicted = pd.read_csv(predicted_path).set_index("audio_filename").sort_index()
    actual = pd.read_csv(actual_path).set_index("audio_filename").sort_index()
    missing = actual.index.difference(predicted.index)
    if len(missing):
        raise typer.BadParameter(
            f"Submission is missing {len(missing)} rows from ground truth "
            f"(first: {missing[0]})."
        )
    predicted = predicted.loc[actual.index]
    wer = _word_error_rate(
        predicted[["transcript"]].fillna("").to_numpy(),
        actual[["transcript"]].fillna("").to_numpy(),
        normalize_text,
    )
    typer.echo(f"WER: {wer:.6f}")


if __name__ == "__main__":
    import typer

    typer.run(main)
