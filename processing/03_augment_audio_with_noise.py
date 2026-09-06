#!/usr/bin/env python3
"""Add room noise to random regions while retaining every clean training row."""

from __future__ import annotations

import argparse
import csv
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AugmentationJob:
    """One independently reproducible noise-augmentation task."""

    row: dict[str, str]
    output_dir: Path
    noise_files: list[Path]
    seed: int
    minimum_noise_fraction: float
    maximum_noise_fraction: float
    minimum_snr_db: float
    maximum_snr_db: float
    copy_number: int


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


def select_rows_for_augmentation(
    rows: list[dict[str, str]], fraction: float, seed: int
) -> list[dict[str, str]]:
    """Select an exact, deterministic fraction while preserving manifest order."""
    if fraction >= 1:
        return rows[:]
    count = int(len(rows) * fraction + 0.5)
    selected_indices = set(random.Random(seed).sample(range(len(rows)), count))
    return [row for index, row in enumerate(rows) if index in selected_indices]


def dataset_name(row: dict[str, str]) -> str:
    """Map stage-2 dataset labels to the two configurable augmentation groups."""
    value = row.get("dataset", "").strip().lower()
    if value == "jember":
        return "jember"
    if value in {"indonesian_development", "indonesian_dev", "development"}:
        return "indonesian_dev"
    raise ValueError(
        f"Unsupported dataset label {row.get('dataset')!r} for "
        f"{row.get('audio_path', '<unknown audio>')}"
    )


def dataset_augmentation_copy_counts(
    rows: list[dict[str, str]],
    jember_multiplier: float,
    development_multiplier: float,
    seed: int,
) -> list[tuple[dict[str, str], int]]:
    """Return additional-copy counts, including deterministic fractions.

    The integer part of a multiplier is emitted for every row in that dataset.
    For its fractional part, an exact rounded fraction of rows receives one
    additional copy. Thus 5.0 means five noisy copies per original, while 0.5
    means one noisy copy for half of the originals.
    """
    grouped: dict[str, list[dict[str, str]]] = {
        "jember": [],
        "indonesian_dev": [],
    }
    for row in rows:
        grouped[dataset_name(row)].append(row)

    multipliers = {
        "jember": jember_multiplier,
        "indonesian_dev": development_multiplier,
    }
    extra_paths: dict[str, set[str]] = {}
    for offset, name in enumerate(("jember", "indonesian_dev"), start=1):
        fraction = multipliers[name] - math.floor(multipliers[name])
        selected = select_rows_for_augmentation(
            grouped[name], fraction, seed + offset
        ) if fraction else []
        extra_paths[name] = {row["audio_path"] for row in selected}

    planned = []
    for row in rows:
        name = dataset_name(row)
        copies = math.floor(multipliers[name])
        if row["audio_path"] in extra_paths[name]:
            copies += 1
        planned.append((row, copies))
    return planned


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


def run_augmentation_job(job: AugmentationJob) -> dict[str, str]:
    """Run a job in either the main process or a worker process."""
    return augment_row(
        job.row,
        job.output_dir,
        job.noise_files,
        random.Random(job.seed),
        job.minimum_noise_fraction,
        job.maximum_noise_fraction,
        job.minimum_snr_db,
        job.maximum_snr_db,
        job.copy_number,
    )


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
    parser.add_argument(
        "--augmentation-fraction",
        type=float,
        default=1.0,
        help="Fraction of eligible clean rows that receive augmented copies.",
    )
    parser.add_argument(
        "--jember-augmentation-multiplier",
        type=float,
        help=(
            "Additional noisy copies per Jember original; fractions select a "
            "deterministic subset (for example, 0.5 augments half once)."
        ),
    )
    parser.add_argument(
        "--development-augmentation-multiplier",
        type=float,
        help="Additional noisy copies per Indonesian-development original.",
    )
    parser.add_argument("--min-noise-fraction", type=float, default=0.25)
    parser.add_argument("--max-noise-fraction", type=float, default=0.75)
    parser.add_argument("--min-snr-db", type=float, default=5.0)
    parser.add_argument("--max-snr-db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of audio augmentations to process concurrently.",
    )
    parser.add_argument("--limit", type=int, help="Augment only the first N rows for a preview.")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input manifest not found: {args.input}")
    if not args.noise_dir.is_dir():
        parser.error(f"Noise directory not found: {args.noise_dir}")
    if args.copies_per_clip < 1:
        parser.error("--copies-per-clip must be at least 1")
    if not 0 <= args.augmentation_fraction <= 1:
        parser.error("--augmentation-fraction must be between 0 and 1")
    dataset_multipliers = (
        args.jember_augmentation_multiplier,
        args.development_augmentation_multiplier,
    )
    if (dataset_multipliers[0] is None) != (dataset_multipliers[1] is None):
        parser.error(
            "Use --jember-augmentation-multiplier and "
            "--development-augmentation-multiplier together"
        )
    if any(value is not None and value < 0 for value in dataset_multipliers):
        parser.error("Dataset augmentation multipliers cannot be negative")
    if not 0 < args.min_noise_fraction <= args.max_noise_fraction <= 1:
        parser.error("Noise fractions must satisfy 0 < min <= max <= 1")
    if args.min_snr_db > args.max_snr_db:
        parser.error("--min-snr-db cannot exceed --max-snr-db")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    with args.input.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        input_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "audio_path" not in input_fields:
        parser.error("Input manifest must contain at least one row and an audio_path column")

    noise_files = find_noise_files(args.noise_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_rows: list[dict[str, str]] = []
    augmented_rows: list[dict[str, str]] = []
    eligible_rows = rows if args.limit is None else rows[: args.limit]
    multiplier_mode = dataset_multipliers[0] is not None
    if multiplier_mode:
        try:
            row_copy_counts = dataset_augmentation_copy_counts(
                eligible_rows,
                args.jember_augmentation_multiplier,
                args.development_augmentation_multiplier,
                args.seed,
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        rows_to_augment = select_rows_for_augmentation(
            eligible_rows, args.augmentation_fraction, args.seed
        )
        row_copy_counts = [
            (row, args.copies_per_clip) for row in rows_to_augment
        ]
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
    seed_rng = random.Random(args.seed)
    jobs: list[AugmentationJob] = []
    for row, copy_count in row_copy_counts:
        for copy_number in range(1, copy_count + 1):
            jobs.append(AugmentationJob(
                row=row,
                output_dir=args.output_dir,
                noise_files=noise_files,
                seed=seed_rng.getrandbits(64),
                minimum_noise_fraction=args.min_noise_fraction,
                maximum_noise_fraction=args.max_noise_fraction,
                minimum_snr_db=args.min_snr_db,
                maximum_snr_db=args.max_snr_db,
                copy_number=copy_number,
            ))

    if args.workers == 1 or len(jobs) < 2:
        results = map(run_augmentation_job, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)))
        results = executor.map(run_augmentation_job, jobs, chunksize=1)
    try:
        for index, augmented in enumerate(results, start=1):
            augmented_rows.append(augmented)
            print(
                f"[{index}/{len(jobs)}] augmented "
                f"{augmented['augmentation_source_audio']}"
            )
    finally:
        if executor is not None:
            executor.shutdown()

    fieldnames = input_fields + [field for field in METADATA_FIELDS if field not in input_fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_rows)
        writer.writerows(augmented_rows)
    selection_description = (
        f"Jember {args.jember_augmentation_multiplier:g}x additional, "
        f"Indonesian dev {args.development_augmentation_multiplier:g}x additional"
        if multiplier_mode else
        f"{args.augmentation_fraction:.1%} augmentation selection"
    )
    print(
        f"Wrote {len(clean_rows)} clean + {len(augmented_rows)} augmented rows "
        f"to {args.output} using {len(noise_files)} noise files "
        f"({selection_description}, "
        f"{args.workers} worker{'s' if args.workers != 1 else ''})"
    )
    link = viewer_url(args.output)
    if link:
        print(f"View augmented dataset: {link}")


if __name__ == "__main__":
    main()
