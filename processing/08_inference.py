#!/usr/bin/env python3
"""Run a trained Whisper LoRA adapter over a CSV manifest and report WER."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from urllib.parse import quote

import torch

# Running this file directly places processing/ rather than the repository root
# first on sys.path, so expose the local model package explicitly.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.whisper_lid import load_token_lid_lora_adapter
from run_config import apply_defaults, load_section
from wer_metrics import word_error_rate


OUTPUT_FIELDS = ("actual_transcript", "audio_path", "predicted_transcript", "wer")
REFERENCE_COLUMNS = (
    "transcript",
    "corrected_transcript",
    "actual_transcript",
    "ground_truth_transcript",
)
AUDIO_COLUMNS = ("audio_path", "audio_file_path")


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_available_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    raise ValueError(
        f"CSV needs one of these columns: {', '.join(candidates)}; "
        f"found: {', '.join(fieldnames)}"
    )


def read_inference_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        audio_column = first_available_column(fields, AUDIO_COLUMNS)
        reference_column = first_available_column(fields, REFERENCE_COLUMNS)
        rows = []
        for number, source_row in enumerate(reader, start=2):
            audio_value = source_row.get(audio_column, "").strip()
            if not audio_value:
                raise ValueError(f"Missing audio path on CSV row {number}")
            reference = source_row.get(reference_column, "").strip()
            if not reference:
                raise ValueError(f"Missing reference transcript on CSV row {number}")
            requested_path = Path(audio_value).expanduser()
            resolved_path = requested_path
            if not resolved_path.is_absolute() and not resolved_path.is_file():
                resolved_path = path.parent / resolved_path
            if not resolved_path.is_file():
                raise FileNotFoundError(
                    f"Audio file on CSV row {number} does not exist: {audio_value}"
                )
            rows.append(
                {
                    "audio_path": audio_value,
                    "resolved_audio_path": str(resolved_path),
                    "actual_transcript": reference,
                }
            )
    if not rows:
        raise ValueError(f"No inference rows found in {path}")
    return rows


def safe_run_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        raise ValueError("Run name must contain at least one filename-safe character")
    return cleaned


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    config_values, config_run_name = load_section(config_args.config, "inference")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="CSV containing audio and reference transcript columns (CLI only).",
    )
    parser.add_argument(
        "--training-run",
        type=Path,
        help="Training output directory containing the selected adapter folder.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best_wer",
        choices=("best_wer", "best_lora", "last_lora"),
        help="Adapter checkpoint to use beneath --training-run.",
    )
    parser.add_argument("--run-name", help="Name used in results_{run_name}.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to <training-run>/08_inference.",
    )
    parser.add_argument("--language", default="id")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--download-root", type=Path, default=Path("models/whisper"))
    parser.add_argument("--beam-size", type=int, default=5)
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    if args.training_run is None:
        parser.error("--training-run is required, either on the CLI or in the inference YAML section")
    if args.beam_size < 1:
        parser.error("--beam-size must be positive")
    run_name = safe_run_name(args.run_name or config_run_name or args.training_run.name)
    output_dir = args.output_dir or args.training_run / "08_inference"
    checkpoint_dir = args.training_run / args.checkpoint
    if not checkpoint_dir.is_dir():
        parser.error(f"Checkpoint directory does not exist: {checkpoint_dir}")

    rows = read_inference_rows(args.csv)
    device = select_device(args.device)
    args.download_root.mkdir(parents=True, exist_ok=True)
    print(f"Loading {checkpoint_dir} on {device}")
    model = load_token_lid_lora_adapter(
        checkpoint_dir,
        device=device,
        download_root=str(args.download_root),
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    output_rows: list[dict[str, str | float]] = []
    total_substitutions = total_deletions = total_insertions = total_reference_words = 0
    clip_wers: list[float] = []
    for index, row in enumerate(rows, start=1):
        # Whisper.transcribe is deliberately used here rather than pad_or_trim +
        # decode: it iterates over long recordings in 30-second sliding windows.
        result = model.transcribe(
            row["resolved_audio_path"],
            language=args.language,
            task="transcribe",
            temperature=0,
            beam_size=args.beam_size,
            condition_on_previous_text=True,
            fp16=device.type == "cuda",
            verbose=False,
        )
        prediction = str(result["text"]).strip()
        substitutions, deletions, insertions, reference_words = word_error_rate(
            row["actual_transcript"], prediction
        )
        errors = substitutions + deletions + insertions
        clip_wer = errors / reference_words if reference_words else math.nan
        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions
        total_reference_words += reference_words
        if not math.isnan(clip_wer):
            clip_wers.append(clip_wer)
        output_rows.append(
            {
                "actual_transcript": row["actual_transcript"],
                "audio_path": row["audio_path"],
                "predicted_transcript": prediction,
                "wer": clip_wer,
            }
        )
        print(
            f"[{index}/{len(rows)}] {Path(row['audio_path']).name}: "
            f"WER={clip_wer:.3%}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"results_{run_name}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    average_wer = sum(clip_wers) / len(clip_wers)
    total_errors = total_substitutions + total_deletions + total_insertions
    corpus_wer = total_errors / total_reference_words
    print(f"Average clip WER: {average_wer:.3%}")
    print(f"Corpus WER: {corpus_wer:.3%} ({total_errors}/{total_reference_words} word errors)")
    print(f"Wrote {len(output_rows)} inference results to {output_path}")
    try:
        relative_output = output_path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        pass
    else:
        csv_parameter = "../" + relative_output.as_posix()
        viewer = (
            "http://127.0.0.1:8000/analysis_indonesia/inference_evaluation.html"
            f"?csv={quote(csv_parameter, safe='/')}"
            "&audio_base=.."
        )
        print(f"View inference errors: {viewer}")


if __name__ == "__main__":
    main()
