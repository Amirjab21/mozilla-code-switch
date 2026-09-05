#!/usr/bin/env python3
"""Add speed/pitch-perturbed copies of every row in a stage-03 manifest."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from urllib.parse import quote

import soundfile as sf
import torch
import torchaudio

from run_config import apply_defaults, load_section


SAMPLE_RATE = 16_000
METADATA_FIELDS = [
    "speed_augmented",
    "speed_factor",
    "speed_augmentation_source_audio",
]
COMPARISON_FIELDS = [
    "comparison_group",
    "variant",
    "audio_path",
    "transcript",
    "noise_augmented",
    "speed_augmented",
    "speed_factor",
    "noise_source",
]


def viewer_url(comparison_csv: Path) -> str | None:
    repository = Path(__file__).resolve().parent.parent
    try:
        relative = comparison_csv.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    return (
        "http://127.0.0.1:8000/analysis_indonesia/"
        "speed_pitch_augmentation_review.html"
        f"?data={quote(f'../{relative}', safe='/')}"
    )


def is_true(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def read_audio(path: Path) -> torch.Tensor:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples.T.copy())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.numel() == 0:
        raise ValueError(f"Empty audio clip: {path}")
    return waveform


def perturb_speed_and_pitch(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    """Change playback speed and pitch together while retaining a 16 kHz file rate."""
    temporary_rate = max(1, round(SAMPLE_RATE / factor))
    perturbed = torchaudio.functional.resample(waveform, SAMPLE_RATE, temporary_rate)
    peak = float(perturbed.abs().max())
    if peak > 0.999:
        perturbed = perturbed * (0.999 / peak)
    return perturbed


def make_speed_copy(
    row: dict[str, str],
    output_dir: Path,
    factor: float,
    copy_number: int,
) -> dict[str, str]:
    source_path = Path(row["audio_path"])
    waveform = perturb_speed_and_pitch(read_audio(source_path), factor)
    destination = output_dir / (
        f"{source_path.stem}__speed_{factor:.4f}_{copy_number:02d}.wav"
    )
    sf.write(destination, waveform.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_16")
    augmented = dict(row)
    augmented.update({
        "audio_path": str(destination),
        "speed_augmented": "true",
        "speed_factor": f"{factor:.4f}",
        "speed_augmentation_source_audio": str(source_path),
    })
    return augmented


def comparison_row(row: dict[str, str]) -> dict[str, str]:
    noise_augmented = is_true(row.get("augmented", "false"))
    speed_augmented = is_true(row.get("speed_augmented", "false"))
    if noise_augmented and speed_augmented:
        variant = "Noise + speed/pitch"
    elif noise_augmented:
        variant = "Noise"
    elif speed_augmented:
        variant = "Speed/pitch"
    else:
        variant = "Original"
    return {
        "comparison_group": row.get("augmentation_source_audio") or row["audio_path"],
        "variant": variant,
        "audio_path": row["audio_path"],
        "transcript": row.get("transcript", ""),
        "noise_augmented": str(noise_augmented).lower(),
        "speed_augmented": str(speed_augmented).lower(),
        "speed_factor": row.get("speed_factor", ""),
        "noise_source": row.get("augmentation_noise", ""),
    }


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    config_values, _ = load_section(config_args.config, "speed_pitch")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    parser.add_argument(
        "--input", type=Path,
        default=Path("processed_indonesia/window_preview_noise/03_dataset_with_augmented.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("processed_indonesia/window_preview_noise/03_1_speed_pertub_added.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("processed_indonesia/window_preview_noise/03_1_speed_audio"),
    )
    parser.add_argument(
        "--comparison-output", type=Path,
        default=Path("processed_indonesia/window_preview_noise/03_1_speed_comparison.csv"),
    )
    parser.add_argument("--copies-per-row", type=int, default=1)
    parser.add_argument("--min-speed", type=float, default=0.7)
    parser.add_argument("--max-speed", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, help="Perturb only the first N stage-03 rows.")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input manifest not found: {args.input}")
    if args.copies_per_row < 1:
        parser.error("--copies-per-row must be at least 1")
    if not 0 < args.min_speed <= args.max_speed:
        parser.error("Speed factors must satisfy 0 < min <= max")
    if args.min_speed <= 1 <= args.max_speed and args.min_speed == args.max_speed == 1:
        parser.error("A speed factor of exactly 1.0 would not augment the audio")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    with args.input.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "audio_path" not in input_fields:
        parser.error("Input manifest must contain at least one row and an audio_path column")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rows_to_augment = rows if args.limit is None else rows[:args.limit]
    original_rows: list[dict[str, str]] = []
    speed_rows: list[dict[str, str]] = []
    for row in rows:
        original = dict(row)
        original.update({
            "speed_augmented": "false",
            "speed_factor": "1.0000",
            "speed_augmentation_source_audio": row["audio_path"],
        })
        original_rows.append(original)
    for index, row in enumerate(rows_to_augment, start=1):
        for copy_number in range(1, args.copies_per_row + 1):
            # Avoid an effectively unchanged copy when the interval crosses 1.0.
            for _ in range(100):
                factor = rng.uniform(args.min_speed, args.max_speed)
                if abs(factor - 1.0) >= 0.01:
                    break
            speed_rows.append(make_speed_copy(row, args.output_dir, factor, copy_number))
        print(f"[{index}/{len(rows_to_augment)}] speed/pitch perturbed {row['audio_path']}")

    all_rows = original_rows + speed_rows
    fieldnames = input_fields + [field for field in METADATA_FIELDS if field not in input_fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    with args.comparison_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_row(row) for row in all_rows)

    print(
        f"Wrote {len(original_rows)} existing + {len(speed_rows)} speed/pitch rows "
        f"to {args.output}"
    )
    print(f"Wrote comparison manifest to {args.comparison_output}")
    link = viewer_url(args.comparison_output)
    if link:
        print(f"View speed/pitch comparisons: {link}")


if __name__ == "__main__":
    main()
