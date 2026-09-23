#!/usr/bin/env python3
"""Evaluate a Whisper token-LID LoRA checkpoint on Common Voice Indonesian and Javanese.

The output CSV can be loaded by ``evaluation_full.html``.  Each row includes
both the viewer's ``ground_truth_transcript`` name and the requested
``original_transcript`` alias.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import whisper


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
PROCESSING_ROOT = Path(__file__).resolve().parent
if str(PROCESSING_ROOT) not in sys.path:
    sys.path.insert(0, str(PROCESSING_ROOT))

from models.whisper_lid import load_token_lid_lora_adapter
from models.whisper_lid.decode import decode_batch_with_token_language
from wer_metrics import word_error_rate


OUTPUT_FIELDS = (
    "dataset",
    "language",
    "audio_file_path",
    "audio_path",
    "original_transcript",
    "ground_truth_transcript",
    "predicted_transcript",
    "wer",
    "reference_words",
    "word_errors",
    "substitutions",
    "deletions",
    "insertions",
)


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def commonvoice_rows(
    *,
    dataset: str,
    language: str,
    manifest: Path,
    audio_dir: Path,
    audio_column: str,
    transcript_column: str,
) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"{dataset} manifest does not exist: {manifest}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"{dataset} audio directory does not exist: {audio_dir}")

    rows: list[dict[str, str]] = []
    for number, source in enumerate(read_tsv(manifest), start=2):
        filename = source.get(audio_column, "").strip()
        transcript = source.get(transcript_column, "").strip()
        if not filename or not transcript:
            continue
        audio_path = audio_dir / filename
        if not audio_path.is_file():
            continue
        try:
            display_path = audio_path.relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError:
            display_path = str(audio_path)
        rows.append(
            {
                "dataset": dataset,
                "language": language,
                "audio_file_path": display_path,
                "resolved_audio_path": str(audio_path),
                "original_transcript": transcript,
                "manifest_row": str(number),
            }
        )
    if not rows:
        raise ValueError(f"{dataset} has no usable rows in {manifest}")
    return rows


def choose_subset(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    if len(rows) < size:
        dataset = rows[0]["dataset"]
        raise ValueError(
            f"Requested {size} {dataset} clips, but only {len(rows)} have both audio and transcripts"
        )
    stable_rows = sorted(rows, key=lambda row: row["audio_file_path"])
    return random.Random(seed).sample(stable_rows, size)


def format_summary(dataset: str, rows: list[dict[str, str | int | float]]) -> str:
    errors = sum(int(row["word_errors"]) for row in rows)
    words = sum(int(row["reference_words"]) for row in rows)
    mean_wer = sum(float(row["wer"]) for row in rows) / len(rows)
    corpus_wer = errors / words if words else math.nan
    return (
        f"{dataset}: {len(rows)} clips | average clip WER {mean_wer:.2%} | "
        f"corpus WER {corpus_wer:.2%} ({errors}/{words} word errors)"
    )


def transcribe_one(model, row: dict[str, str], device: torch.device) -> str:
    """Use Whisper's windowed path so recordings longer than 30 seconds are intact."""
    result = model.transcribe(
        row["resolved_audio_path"],
        language=row["language"],
        task="transcribe",
        temperature=0,
        condition_on_previous_text=True,
        fp16=device.type == "cuda",
        verbose=False,
    )
    return str(result["text"]).strip()


def transcribe_batch(
    model,
    rows: list[dict[str, str]],
    device: torch.device,
) -> list[str]:
    """Greedily decode one same-language batch that fits Whisper's context."""
    language = rows[0]["language"]
    if any(row["language"] != language for row in rows):
        raise ValueError("A decoding batch must use one Whisper language prompt")
    mels = []
    for row in rows:
        audio = whisper.load_audio(row["resolved_audio_path"])
        if len(audio) > whisper.audio.N_SAMPLES:
            raise ValueError("Long clips must use transcribe_one")
        mels.append(whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)))
    # This is the local fork's supported batch decoder. Native Whisper's KV
    # cache is not batch-safe with the token-LID fork.
    results = decode_batch_with_token_language(model, torch.stack(mels), language=language)
    return [result.text.strip() for result in results]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Adapter directory, for example processed_indonesia/.../best_wer.",
    )
    parser.add_argument(
        "--subset-per-dataset",
        type=int,
        default=100,
        help="Number of reproducibly sampled clips per Common Voice dataset (default: 100).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of same-language clips greedily decoded together when they are at "
            "most 30 seconds (default: 1). Longer clips remain windowed and decode singly."
        ),
    )
    parser.add_argument("--download-root", type=Path, default=Path("models/whisper"))
    parser.add_argument(
        "--indonesian-manifest",
        type=Path,
        default=Path("indonesian_data/cv_indonesian/id/test.tsv"),
    )
    parser.add_argument(
        "--indonesian-audio-dir",
        type=Path,
        default=Path("indonesian_data/cv_indonesian/id/clips"),
    )
    parser.add_argument(
        "--javanese-manifest",
        type=Path,
        default=Path("indonesian_data/cv_javanese/ss-corpus-jv.tsv"),
    )
    parser.add_argument(
        "--javanese-audio-dir",
        type=Path,
        default=Path("indonesian_data/cv_javanese/audios"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Defaults to commonvoice_evaluation.csv inside --checkpoint.",
    )
    args = parser.parse_args()

    if args.subset_per_dataset < 1:
        parser.error("--subset-per-dataset must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.checkpoint.is_dir():
        parser.error(f"Checkpoint directory does not exist: {args.checkpoint}")

    datasets = [
        choose_subset(
            commonvoice_rows(
                dataset="cv_indonesian",
                language="id",
                manifest=args.indonesian_manifest,
                audio_dir=args.indonesian_audio_dir,
                audio_column="path",
                transcript_column="sentence",
            ),
            args.subset_per_dataset,
            args.seed,
        ),
        choose_subset(
            commonvoice_rows(
                dataset="cv_javanese",
                # OpenAI Whisper uses its legacy ``jw`` code for Javanese.
                language="jw",
                manifest=args.javanese_manifest,
                audio_dir=args.javanese_audio_dir,
                audio_column="audio_file",
                transcript_column="transcription",
            ),
            args.subset_per_dataset,
            args.seed + 1,
        ),
    ]
    selected = [row for dataset_rows in datasets for row in dataset_rows]

    device = select_device(args.device)
    args.download_root.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.checkpoint} on {device}")
    model = load_token_lid_lora_adapter(
        args.checkpoint, device=device, download_root=str(args.download_root)
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    predictions: list[str] = []
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        # The selected datasets are contiguous by language. Be defensive if a
        # caller later changes their ordering or adds a third language.
        if len({row["language"] for row in batch}) != 1:
            predictions.extend(
                transcribe_one(model, row, device) for row in batch
            )
            continue
        try:
            predictions.extend(transcribe_batch(model, batch, device))
        except ValueError as error:
            if str(error) != "Long clips must use transcribe_one":
                raise
            predictions.extend(
                transcribe_one(model, row, device) for row in batch
            )

    output_rows: list[dict[str, str | int | float]] = []
    for index, (row, prediction) in enumerate(zip(selected, predictions, strict=True), start=1):
        substitutions, deletions, insertions, words = word_error_rate(
            row["original_transcript"], prediction
        )
        errors = substitutions + deletions + insertions
        wer = errors / words if words else math.nan
        output_rows.append(
            {
                "dataset": row["dataset"],
                "language": row["language"],
                "audio_file_path": row["audio_file_path"],
                "audio_path": row["audio_file_path"],
                "original_transcript": row["original_transcript"],
                "ground_truth_transcript": row["original_transcript"],
                "predicted_transcript": prediction,
                "wer": wer,
                "reference_words": words,
                "word_errors": errors,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
            }
        )
        print(f"[{index}/{len(selected)}] {row['dataset']}: {Path(row['audio_file_path']).name} WER={wer:.2%}")

    output_csv = args.output_csv or args.checkpoint / "commonvoice_evaluation.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    by_dataset: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in output_rows:
        by_dataset[str(row["dataset"])].append(row)
    for dataset in ("cv_indonesian", "cv_javanese"):
        print(format_summary(dataset, by_dataset[dataset]))
    print(format_summary("combined", output_rows))
    print(f"Wrote {len(output_rows)} rows to {output_csv}")
    print(
        "Open with evaluation_full.html after serving the repository; set Audio base "
        "to the repository root (.) so the CSV's audio_file_path values resolve."
    )


if __name__ == "__main__":
    main()
