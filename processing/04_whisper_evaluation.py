#!/usr/bin/env python3
"""Step 4: transcribe VAD clips with Whisper and evaluate word error rate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from text_normalisation import normalize_text


# Model-selection section: extend this registry when another inference backend is added.
MODEL_BACKENDS = {"whisper": {"default_model": "medium", "description": "OpenAI Whisper multilingual ASR"}}
CORPUS_TO_WHISPER_LANGUAGE = {"eng": "en", "spa": "es"}
# Keep batch results reproducible.  Whisper's default fallback schedule samples
# higher temperatures after a low-confidence decode, which can change output
# between otherwise identical CPU requests.
TRANSCRIBE_OPTIONS = {"task": "transcribe", "temperature": 0, "beam_size": 5}


def choose_device(requested: str, torch) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalise_words(text: str) -> list[str]:
    """Normalize a transcript with the shared scoring rules and tokenize it."""
    return normalize_text(text).split()


def reference_labels(row: dict[str, str], reference_words: list[str]) -> list[str]:
    try:
        labels = json.loads(row.get("word_langids", "[]"))
    except json.JSONDecodeError:
        labels = []
    language_ids = [item.get("langid", "unknown") for item in labels if item.get("word")]
    return language_ids if len(language_ids) == len(reference_words) else ["unknown"] * len(reference_words)


def align_words(reference: list[str], hypothesis: list[str], language_ids: list[str]) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Return a Levenshtein alignment and standard WER counts by language ID."""
    rows, columns = len(reference), len(hypothesis)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1): costs[i][0] = i
    for j in range(1, columns + 1): costs[0][j] = j
    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            costs[i][j] = min(costs[i - 1][j] + 1, costs[i][j - 1] + 1,
                              costs[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]))

    alignment: list[dict[str, str]] = []
    i, j = rows, columns
    while i or j:
        if i and j and reference[i - 1] == hypothesis[j - 1] and costs[i][j] == costs[i - 1][j - 1]:
            alignment.append({"operation": "correct", "reference": reference[i - 1], "hypothesis": hypothesis[j - 1], "langid": language_ids[i - 1]})
            i, j = i - 1, j - 1
        elif i and j and costs[i][j] == costs[i - 1][j - 1] + 1:
            alignment.append({"operation": "substitution", "reference": reference[i - 1], "hypothesis": hypothesis[j - 1], "langid": language_ids[i - 1]})
            i, j = i - 1, j - 1
        elif i and costs[i][j] == costs[i - 1][j] + 1:
            alignment.append({"operation": "deletion", "reference": reference[i - 1], "hypothesis": "", "langid": language_ids[i - 1]})
            i -= 1
        else:
            attributed_language = language_ids[i] if i < rows else (language_ids[i - 1] if i else "unknown")
            alignment.append({"operation": "insertion", "reference": "", "hypothesis": hypothesis[j - 1], "langid": attributed_language})
            j -= 1
    alignment.reverse()

    grouped: dict[str, dict[str, int | float]] = defaultdict(lambda: {"reference_words": 0, "substitutions": 0, "deletions": 0, "insertions": 0})
    for language in language_ids:
        grouped[language]["reference_words"] += 1
    for item in alignment:
        if item["operation"] == "substitution": grouped[item["langid"]]["substitutions"] += 1
        elif item["operation"] == "deletion": grouped[item["langid"]]["deletions"] += 1
        elif item["operation"] == "insertion": grouped[item["langid"]]["insertions"] += 1
    for values in grouped.values():
        errors = values["substitutions"] + values["deletions"] + values["insertions"]
        values["wer"] = errors / values["reference_words"] if values["reference_words"] else None

    substitutions = sum(1 for item in alignment if item["operation"] == "substitution")
    deletions = sum(1 for item in alignment if item["operation"] == "deletion")
    insertions = sum(1 for item in alignment if item["operation"] == "insertion")
    summary = {"reference_words": rows, "hypothesis_words": columns, "substitutions": substitutions, "deletions": deletions, "insertions": insertions, "wer": costs[rows][columns] / rows if rows else None, "by_langid": dict(sorted(grouped.items()))}
    return alignment, summary


def dominant_reference(language_ids: list[str]) -> str:
    comparable = Counter(CORPUS_TO_WHISPER_LANGUAGE[label] for label in language_ids if label in CORPUS_TO_WHISPER_LANGUAGE)
    if not comparable:
        return "unknown"
    return "mixed" if len(comparable) > 1 else comparable.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/02_vad.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("processed_indonesia/04_whisper_evaluation.csv"))
    parser.add_argument("--backend", choices=MODEL_BACKENDS, default="whisper")
    parser.add_argument("--model", default="medium", help="Whisper model name; use a multilingual model such as medium.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--download-root", type=Path, default=Path("models/whisper"))
    parser.add_argument("--limit", type=int, help="Run only the first N retained clips.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        import torch
        import whisper
    except ImportError as error:
        raise SystemExit("Install inference dependencies first: uv sync --project processing") from error
    device = choose_device(args.device, torch)
    args.download_root.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(args.model, device=device, download_root=str(args.download_root))
    if not model.is_multilingual:
        parser.error(f"{args.model!r} is English-only and cannot perform multilingual language detection")

    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("filtered_out", "False").casefold() != "true"]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        reference = normalise_words(row["transcript"])
        labels = reference_labels(row, reference)
        audio = whisper.pad_or_trim(whisper.load_audio(row["audio_path"]))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)
        _, probabilities = model.detect_language(mel)
        predicted_language, probability = max(probabilities.items(), key=lambda item: item[1])
        result = model.transcribe(row["audio_path"], fp16=device == "cuda", verbose=False, **TRANSCRIBE_OPTIONS)
        hypothesis = normalise_words(result["text"])
        alignment, scores = align_words(reference, hypothesis, labels)
        output_rows.append({
            "audio_path": row["audio_path"], "transcript": row["transcript"], "whisper_transcript": result["text"].strip(),
            "reference_language_ids": json.dumps(sorted(set(labels))), "reference_dominant_language": dominant_reference(labels),
            "predicted_language": predicted_language, "predicted_language_probability": f"{probability:.6f}",
            "wer": f"{scores['wer']:.6f}" if scores["wer"] is not None else "",
            "wer_by_langid": json.dumps(scores["by_langid"], ensure_ascii=False, sort_keys=True),
            "alignment": json.dumps(alignment, ensure_ascii=False), "model_backend": args.backend, "model_name": args.model, "device": device,
        })
        print(f"[{index}/{len(rows)}] {Path(row['audio_path']).name}: WER={scores['wer']:.1%}, language={predicted_language}")

    fields = ["audio_path", "transcript", "whisper_transcript", "reference_language_ids", "reference_dominant_language", "predicted_language", "predicted_language_probability", "wer", "wer_by_langid", "alignment", "model_backend", "model_name", "device"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} Whisper transcription evaluations to {args.output_csv}")


if __name__ == "__main__":
    main()
