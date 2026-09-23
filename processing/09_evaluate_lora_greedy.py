#!/usr/bin/env python3
"""Evaluate a Whisper LoRA checkpoint with fresh greedy, non-teacher-forced decoding."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.whisper_lid import load_token_lid_lora_adapter
from wer_metrics import word_error_rate


DEFAULT_ADAPTER = Path(
    "processed_indonesia/first_full_run_data_mix_commonvoice_jember_1200/"
    "05_train/best_lora"
)
DEFAULT_EVALUATION_CSV = Path(
    "processed_indonesia/first_full_run_data_mix_commonvoice_jember_1200/"
    "05_train/best_eval_predictions.csv"
)
OUTPUT_FIELDS = (
    "audio_file_path",
    "ground_truth_transcript",
    "language_labels",
    "predicted_transcript",
    "predicted_language_labels",
    "wer",
    "reference_words",
    "word_errors",
    "substitutions",
    "deletions",
    "insertions",
    "decoding",
)


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"audio_file_path", "ground_truth_transcript"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def resolve_audio_path(value: str, csv_path: Path) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend((REPOSITORY_ROOT / requested, csv_path.parent / requested))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio file not found: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--evaluation-csv", type=Path, default=DEFAULT_EVALUATION_CSV)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Defaults to greedy_eval_predictions.csv beside the input CSV.",
    )
    parser.add_argument("--language", default="id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--download-root", type=Path, default=Path("models/whisper"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.adapter_dir.is_dir():
        parser.error(f"Adapter directory not found: {args.adapter_dir}")
    if not args.evaluation_csv.is_file():
        parser.error(f"Evaluation CSV not found: {args.evaluation_csv}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    rows = read_rows(args.evaluation_csv)
    if args.limit is not None:
        rows = rows[: args.limit]
    output_csv = args.output_csv or args.evaluation_csv.with_name(
        "greedy_eval_predictions.csv"
    )
    device = select_device(args.device)
    args.download_root.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.adapter_dir} on {device}")
    model = load_token_lid_lora_adapter(
        args.adapter_dir,
        device=device,
        download_root=str(args.download_root),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_rows: list[dict[str, str | int | float]] = []
    total_substitutions = total_deletions = total_insertions = 0
    total_reference_words = 0
    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row["audio_file_path"], args.evaluation_csv)
        # temperature=0 with beam_size=None is greedy autoregressive inference.
        # No reference tokens are supplied, so this is not teacher forcing.
        result = model.transcribe(
            str(audio_path),
            language=args.language,
            task="transcribe",
            temperature=0,
            beam_size=None,
            best_of=None,
            patience=None,
            condition_on_previous_text=False,
            fp16=device.type == "cuda",
            verbose=False,
        )
        prediction = str(result["text"]).strip()
        substitutions, deletions, insertions, reference_words = word_error_rate(
            row["ground_truth_transcript"], prediction
        )
        errors = substitutions + deletions + insertions
        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions
        total_reference_words += reference_words
        output_rows.append(
            {
                "audio_file_path": row["audio_file_path"],
                "ground_truth_transcript": row["ground_truth_transcript"],
                "language_labels": row.get("language_labels", "[]"),
                "predicted_transcript": prediction,
                "predicted_language_labels": "[]",
                "wer": errors / reference_words if reference_words else math.nan,
                "reference_words": reference_words,
                "word_errors": errors,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "decoding": "greedy (beam_size=None, no teacher forcing)",
            }
        )
        clip_wer = errors / reference_words if reference_words else math.nan
        print(
            f"[{index}/{len(rows)}] WER={clip_wer:.2%} "
            f"S={substitutions} D={deletions} I={insertions} — {audio_path.name}"
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    total_errors = total_substitutions + total_deletions + total_insertions
    aggregate_wer = (
        total_errors / total_reference_words if total_reference_words else math.nan
    )
    metrics_path = output_csv.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "clips": len(output_rows),
                "reference_words": total_reference_words,
                "substitutions": total_substitutions,
                "deletions": total_deletions,
                "insertions": total_insertions,
                "word_errors": total_errors,
                "wer": aggregate_wer,
                "beam_size": None,
                "teacher_forcing": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Aggregate WER: {aggregate_wer:.4f} ({aggregate_wer:.2%}) — "
        f"S={total_substitutions}, D={total_deletions}, I={total_insertions}, "
        f"N={total_reference_words}"
    )
    print(f"Wrote predictions to {output_csv}")
    print(f"Wrote aggregate metrics to {metrics_path}")


if __name__ == "__main__":
    main()
