#!/usr/bin/env python3
"""Add room noise to random regions while retaining every clean training row."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from urllib.parse import quote

import librosa
import numpy as np
import soundfile as sf

from run_config import apply_defaults, load_section


SAMPLE_RATE = 16_000
METADATA_FIELDS = [
    "augmented",
    "augmentation_source_audio",
    "augmentation_noise",
    "augmentation_snr_db",
    "augmentation_start_ms",
    "augmentation_end_ms",
]


def viewer_url(manifest: Path) -> str | None:
    repository = Path(__file__).resolve().parent.parent
    try:
        relative = manifest.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    return (
        "http://127.0.0.1:8000/analysis_indonesia/segment_review.html"
        f"?manifest={quote(f'../{relative}', safe='/')}"
    )


def find_noise_files(noise_dir: Path) -> list[Path]:
    """Return additive noises, deliberately excluding room impulse responses."""
    files = [
        path for path in noise_dir.rglob("*.wav")
        if "pointsource_noises" in path.parts
        or ("real_rirs_isotropic_noises" in path.parts and "_noise_" in path.name)
    ]
    files.sort()
    if not files:
        raise ValueError(f"No additive WAV noise files found beneath {noise_dir}")
    return files


def random_noise_region(
    clean: np.ndarray,
    noise: np.ndarray,
    rng: random.Random,
    minimum_fraction: float,
    maximum_fraction: float,
) -> tuple[int, np.ndarray]:
    fraction = rng.uniform(minimum_fraction, maximum_fraction)
    length = max(1, min(len(clean), round(len(clean) * fraction)))
    clean_start = rng.randint(0, len(clean) - length)
    if len(noise) >= length:
        noise_start = rng.randint(0, len(noise) - length)
        region = noise[noise_start : noise_start + length].copy()
    else:
        repeats = (length + len(noise) - 1) // len(noise)
        region = np.tile(noise, repeats)[:length].copy()
    return clean_start, region


def mix_region_at_snr(
    clean: np.ndarray,
    noise_region: np.ndarray,
    start: int,
    snr_db: float,
) -> np.ndarray:
    """Mix one noise region at the requested signal-to-noise ratio."""
    end = start + len(noise_region)
    clean_region = clean[start:end]
    clean_rms = float(np.sqrt(np.mean(np.square(clean_region), dtype=np.float64)))
    noise_rms = float(np.sqrt(np.mean(np.square(noise_region), dtype=np.float64)))
    if noise_rms < 1e-8:
        raise ValueError("Selected noise region is silent")
    target_noise_rms = max(clean_rms, 1e-4) / (10 ** (snr_db / 20))
    scaled_noise = noise_region * (target_noise_rms / noise_rms)

    # Short fades avoid introducing clicks at the random insertion boundaries.
    fade_length = min(round(0.02 * SAMPLE_RATE), len(scaled_noise) // 2)
    if fade_length:
        scaled_noise[:fade_length] *= np.linspace(0, 1, fade_length, dtype=np.float32)
        scaled_noise[-fade_length:] *= np.linspace(1, 0, fade_length, dtype=np.float32)
    mixed = clean.copy()
    mixed[start:end] += scaled_noise
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.999:
        mixed *= 0.999 / peak
    return mixed


def augment_row(
    row: dict[str, str],
    output_dir: Path,
    noise_files: list[Path],
    rng: random.Random,
    minimum_noise_fraction: float,
    maximum_noise_fraction: float,
    minimum_snr_db: float,
    maximum_snr_db: float,
    copy_number: int,
) -> dict[str, str]:
    clean_path = Path(row["audio_path"])
    clean, _ = librosa.load(clean_path, sr=SAMPLE_RATE, mono=True)
    if clean.size == 0:
        raise ValueError(f"Empty audio clip: {clean_path}")

    # Retry a few times because some source-noise files contain silent regions.
    for _ in range(10):
        noise_path = rng.choice(noise_files)
        noise, _ = librosa.load(noise_path, sr=SAMPLE_RATE, mono=True)
        if noise.size == 0:
            continue
        start, noise_region = random_noise_region(
            clean, noise, rng, minimum_noise_fraction, maximum_noise_fraction
        )
        snr_db = rng.uniform(minimum_snr_db, maximum_snr_db)
        try:
            mixed = mix_region_at_snr(clean, noise_region, start, snr_db)
        except ValueError:
            continue
        break
    else:
        raise ValueError(f"Could not find a non-silent noise region for {clean_path}")

    destination = output_dir / f"{clean_path.stem}__noise_{copy_number:02d}.wav"
    sf.write(destination, mixed, SAMPLE_RATE, subtype="PCM_16")
    augmented = dict(row)
    augmented.update({
        "audio_path": str(destination),
        "augmented": "true",
        "augmentation_source_audio": str(clean_path),
        "augmentation_noise": str(noise_path),
        "augmentation_snr_db": f"{snr_db:.3f}",
        "augmentation_start_ms": str(round(start * 1000 / SAMPLE_RATE)),
        "augmentation_end_ms": str(round((start + len(noise_region)) * 1000 / SAMPLE_RATE)),
    })
    return augmented


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    config_values, _ = load_section(config_args.config, "augment")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    parser.add_argument("--input", type=Path, default=Path("processed_indonesia/02_processed.csv"))
    parser.add_argument("--output", type=Path, default=Path("processed_indonesia/03_dataset_with_augmented.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia/03_augmented_audio"))
    parser.add_argument("--noise-dir", type=Path, default=Path("indonesian_data/room_noises"))
    parser.add_argument("--copies-per-clip", type=int, default=1)
    parser.add_argument("--min-noise-fraction", type=float, default=0.25)
    parser.add_argument("--max-noise-fraction", type=float, default=0.75)
    parser.add_argument("--min-snr-db", type=float, default=5.0)
    parser.add_argument("--max-snr-db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int, help="Augment only the first N rows for a preview.")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input manifest not found: {args.input}")
    if not args.noise_dir.is_dir():
        parser.error(f"Noise directory not found: {args.noise_dir}")
    if args.copies_per_clip < 1:
        parser.error("--copies-per-clip must be at least 1")
    if not 0 < args.min_noise_fraction <= args.max_noise_fraction <= 1:
        parser.error("Noise fractions must satisfy 0 < min <= max <= 1")
    if args.min_snr_db > args.max_snr_db:
        parser.error("--min-snr-db cannot exceed --max-snr-db")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    with args.input.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "audio_path" not in input_fields:
        parser.error("Input manifest must contain at least one row and an audio_path column")

    noise_files = find_noise_files(args.noise_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    clean_rows: list[dict[str, str]] = []
    augmented_rows: list[dict[str, str]] = []
    rows_to_augment = rows if args.limit is None else rows[: args.limit]
    for row in rows:
        clean = dict(row)
        clean.update({
            "augmented": "false",
            "augmentation_source_audio": row["audio_path"],
            "augmentation_noise": "",
            "augmentation_snr_db": "",
            "augmentation_start_ms": "",
            "augmentation_end_ms": "",
        })
        clean_rows.append(clean)
    for index, row in enumerate(rows_to_augment, start=1):
        for copy_number in range(1, args.copies_per_clip + 1):
            augmented_rows.append(augment_row(
                row, args.output_dir, noise_files, rng,
                args.min_noise_fraction, args.max_noise_fraction,
                args.min_snr_db, args.max_snr_db, copy_number,
            ))
        print(f"[{index}/{len(rows_to_augment)}] augmented {row['audio_path']}")

    fieldnames = input_fields + [field for field in METADATA_FIELDS if field not in input_fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)
        writer.writerows(augmented_rows)
    print(
        f"Wrote {len(clean_rows)} clean + {len(augmented_rows)} augmented rows "
        f"to {args.output} using {len(noise_files)} noise files"
    )
    link = viewer_url(args.output)
    if link:
        print(f"View augmented dataset: {link}")


if __name__ == "__main__":
    main()
