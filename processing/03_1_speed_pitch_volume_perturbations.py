"""Add speed, pitch, amplitude, time-dropout, and reverb copies of stage-03 rows."""

from __future__ import annotations

import argparse
import csv
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import soundfile as sf
import torch
import torchaudio
from run_config import apply_defaults, load_section

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class PerturbationJob:
    row: dict[str, str]
    output_dir: Path
    rir_files: tuple[Path, ...]
    copies_per_row: int
    minimum_speed: float
    maximum_speed: float
    minimum_volume_gain_db: float
    maximum_volume_gain_db: float
    minimum_reverb_wet: float
    maximum_reverb_wet: float
    time_dropout_probability: float
    minimum_time_dropout_ms: int
    maximum_time_dropout_ms: int
    minimum_time_dropout_count: int
    maximum_time_dropout_count: int
    seed: int


METADATA_FIELDS = [
    "speed_augmented",
    "speed_factor",
    "speed_augmentation_source_audio",
    "volume_augmented",
    "volume_gain_db",
    "reverb_augmented",
    "reverb_wet",
    "reverb_rir",
    "time_dropout_augmented",
    "time_dropout_chunks",
    "time_dropout_total_ms",
]
COMPARISON_FIELDS = [
    "comparison_group",
    "variant",
    "audio_path",
    "transcript",
    "noise_augmented",
    "speed_augmented",
    "speed_factor",
    "volume_augmented",
    "volume_gain_db",
    "reverb_augmented",
    "reverb_wet",
    "reverb_rir",
    "time_dropout_augmented",
    "time_dropout_total_ms",
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


def find_rir_files(rir_dir: Path) -> list[Path]:
    """Return impulse responses while excluding colocated background noises."""
    files = sorted(
        path for path in rir_dir.rglob("*.wav")
        if "_rir_" in path.name.casefold()
    )
    if not files:
        raise ValueError(f"No '*_rir_*.wav' impulse responses found beneath {rir_dir}")
    return files


def soft_limit(waveform: torch.Tensor) -> torch.Tensor:
    """Keep samples within PCM range without hard clipping."""
    if float(waveform.abs().max()) <= 0.999:
        return waveform
    limited = torch.tanh(waveform)
    return limited * (0.999 / float(limited.abs().max()))


def perturb_speed_and_pitch(waveform: torch.Tensor, factor: float) -> torch.Tensor:
    """Change playback speed and pitch together while retaining a 16 kHz file rate."""
    temporary_rate = max(1, round(SAMPLE_RATE / factor))
    return torchaudio.functional.resample(waveform, SAMPLE_RATE, temporary_rate)


def perturb_random_amplitude(
    waveform: torch.Tensor, requested_gain_db: float
) -> tuple[torch.Tensor, float]:
    """Apply random-amplitude gain and soft-limit if it would clip."""
    perturbed = waveform * (10 ** (requested_gain_db / 20))
    return soft_limit(perturbed), requested_gain_db


def apply_time_dropout(
    waveform: torch.Tensor,
    rng: random.Random,
    probability: float,
    minimum_chunk_ms: int,
    maximum_chunk_ms: int,
    minimum_count: int,
    maximum_count: int,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Replace random consecutive waveform chunks with silence."""
    if rng.random() >= probability:
        return waveform, []
    dropped = waveform.clone()
    chunks: list[tuple[int, int]] = []
    for _ in range(rng.randint(minimum_count, maximum_count)):
        requested_length = round(
            rng.uniform(minimum_chunk_ms, maximum_chunk_ms) * SAMPLE_RATE / 1000
        )
        length = max(1, min(requested_length, waveform.shape[-1]))
        start = rng.randint(0, waveform.shape[-1] - length)
        dropped[:, start:start + length] = 0
        chunks.append((start, start + length))
    return dropped, chunks


def time_dropout_total_samples(chunks: list[tuple[int, int]]) -> int:
    """Count the union of dropped samples without double-counting overlaps."""
    merged: list[list[int]] = []
    for start, end in sorted(chunks):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def apply_room_reverb(
    waveform: torch.Tensor,
    rir_path: Path,
    wet: float,
    channel: int,
) -> torch.Tensor:
    """Convolve speech with one RIR channel and mix it with the dry signal."""
    samples, sample_rate = sf.read(rir_path, dtype="float32", always_2d=True)
    selected_channel = channel % samples.shape[1]
    rir = torch.from_numpy(samples[:, selected_channel].copy()).unsqueeze(0)
    if sample_rate != SAMPLE_RATE:
        rir = torchaudio.functional.resample(rir, sample_rate, SAMPLE_RATE)
    peak = float(rir.abs().max())
    if peak <= 1e-8:
        raise ValueError(f"Silent room impulse response: {rir_path}")

    # Remove propagation silence so convolution does not shift the transcript.
    active = torch.nonzero(rir.abs()[0] >= peak * 0.05)
    rir = rir[:, int(active[0].item()):] if active.numel() else rir
    rir = rir / torch.sqrt(torch.sum(rir.square()))

    output_length = waveform.shape[-1] + rir.shape[-1] - 1
    fft_length = 1 << (output_length - 1).bit_length()
    reverberant = torch.fft.irfft(
        torch.fft.rfft(waveform, n=fft_length)
        * torch.fft.rfft(rir, n=fft_length),
        n=fft_length,
    )[:, :waveform.shape[-1]]

    dry_rms = torch.sqrt(torch.mean(waveform.square()))
    wet_rms = torch.sqrt(torch.mean(reverberant.square()))
    if float(wet_rms) <= 1e-8:
        raise ValueError(f"Room convolution produced silence: {rir_path}")
    reverberant = reverberant * (dry_rms / wet_rms)
    return soft_limit((1 - wet) * waveform + wet * reverberant)


def make_perturbed_copy(
    row: dict[str, str],
    source_waveform: torch.Tensor,
    output_dir: Path,
    factor: float,
    volume_gain_db: float,
    rir_path: Path,
    reverb_wet: float,
    rir_channel: int,
    rng: random.Random,
    time_dropout_probability: float,
    minimum_time_dropout_ms: int,
    maximum_time_dropout_ms: int,
    minimum_time_dropout_count: int,
    maximum_time_dropout_count: int,
    copy_number: int,
) -> dict[str, str]:
    source_path = Path(row["audio_path"])
    waveform = perturb_speed_and_pitch(source_waveform, factor)
    waveform, applied_volume_gain_db = perturb_random_amplitude(
        waveform, volume_gain_db
    )
    waveform = apply_room_reverb(waveform, rir_path, reverb_wet, rir_channel)
    waveform, dropped_chunks = apply_time_dropout(
        waveform,
        rng,
        time_dropout_probability,
        minimum_time_dropout_ms,
        maximum_time_dropout_ms,
        minimum_time_dropout_count,
        maximum_time_dropout_count,
    )
    destination = output_dir / (
        f"{source_path.stem}__speed_{factor:.4f}"
        f"__volume_{applied_volume_gain_db:+.2f}db"
        f"__reverb_{reverb_wet:.2f}_{copy_number:02d}.wav"
    )
    sf.write(destination, waveform.squeeze(0).numpy(), SAMPLE_RATE, subtype="PCM_16")
    augmented = dict(row)
    augmented.update({
        "audio_path": str(destination),
        "speed_augmented": "true",
        "speed_factor": f"{factor:.4f}",
        "speed_augmentation_source_audio": str(source_path),
        "volume_augmented": "true",
        "volume_gain_db": f"{applied_volume_gain_db:.4f}",
        "reverb_augmented": "true",
        "reverb_wet": f"{reverb_wet:.4f}",
        "reverb_rir": str(rir_path),
        "time_dropout_augmented": str(bool(dropped_chunks)).lower(),
        "time_dropout_chunks": json.dumps([
            {
                "start_sample": start,
                "end_sample": end,
                "start_ms": round(start * 1000 / SAMPLE_RATE),
                "end_ms": round(end * 1000 / SAMPLE_RATE),
            }
            for start, end in dropped_chunks
        ]),
        "time_dropout_total_ms": str(round(
            time_dropout_total_samples(dropped_chunks) * 1000 / SAMPLE_RATE
        )),
    })
    return augmented


def sample_away_from(
    rng: random.Random,
    minimum: float,
    maximum: float,
    excluded_value: float,
    minimum_distance: float,
) -> float:
    """Sample a value, avoiding an effectively unchanged augmentation."""
    value = minimum
    for _ in range(100):
        value = rng.uniform(minimum, maximum)
        if abs(value - excluded_value) >= minimum_distance:
            break
    return value


def perturb_row(job: PerturbationJob) -> list[dict[str, str]]:
    """Load one source once and produce all deterministic copies for that row."""
    rng = random.Random(job.seed)
    source_waveform = read_audio(Path(job.row["audio_path"]))
    copies: list[dict[str, str]] = []
    for copy_number in range(1, job.copies_per_row + 1):
        factor = sample_away_from(
            rng, job.minimum_speed, job.maximum_speed, 1.0, 0.01
        )
        volume_gain_db = sample_away_from(
            rng,
            job.minimum_volume_gain_db,
            job.maximum_volume_gain_db,
            0.0,
            0.1,
        )
        copies.append(make_perturbed_copy(
            job.row,
            source_waveform,
            job.output_dir,
            factor,
            volume_gain_db,
            rng.choice(job.rir_files),
            rng.uniform(job.minimum_reverb_wet, job.maximum_reverb_wet),
            rng.randrange(2**31),
            rng,
            job.time_dropout_probability,
            job.minimum_time_dropout_ms,
            job.maximum_time_dropout_ms,
            job.minimum_time_dropout_count,
            job.maximum_time_dropout_count,
            copy_number,
        ))
    return copies


def configure_worker() -> None:
    """Prevent each worker from spawning its own pool of Torch CPU threads."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def comparison_row(row: dict[str, str]) -> dict[str, str]:
    noise_augmented = is_true(row.get("augmented", "false"))
    speed_augmented = is_true(row.get("speed_augmented", "false"))
    volume_augmented = is_true(row.get("volume_augmented", "false"))
    reverb_augmented = is_true(row.get("reverb_augmented", "false"))
    time_dropout_augmented = is_true(row.get("time_dropout_augmented", "false"))
    effects = []
    if noise_augmented:
        effects.append("Noise")
    if speed_augmented:
        effects.append("speed/pitch")
    if volume_augmented:
        effects.append("volume")
    if reverb_augmented:
        effects.append("room reverb")
    if time_dropout_augmented:
        effects.append("time dropout")
    return {
        "comparison_group": row.get("augmentation_source_audio") or row["audio_path"],
        "variant": " + ".join(effects) if effects else "Original",
        "audio_path": row["audio_path"],
        "transcript": row.get("transcript", ""),
        "noise_augmented": str(noise_augmented).lower(),
        "speed_augmented": str(speed_augmented).lower(),
        "speed_factor": row.get("speed_factor", ""),
        "volume_augmented": str(volume_augmented).lower(),
        "volume_gain_db": row.get("volume_gain_db", ""),
        "reverb_augmented": str(reverb_augmented).lower(),
        "reverb_wet": row.get("reverb_wet", ""),
        "reverb_rir": row.get("reverb_rir", ""),
        "time_dropout_augmented": str(time_dropout_augmented).lower(),
        "time_dropout_total_ms": row.get("time_dropout_total_ms", ""),
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
        default=Path("processed_indonesia/03_1_dataset_with_perturbations.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("processed_indonesia/03_1_perturbed_audio"),
    )
    parser.add_argument(
        "--comparison-output", type=Path,
        default=Path("processed_indonesia/03_1_perturbation_comparison.csv"),
    )
    parser.add_argument("--copies-per-row", type=int, default=1)
    parser.add_argument("--min-speed", type=float, default=0.7)
    parser.add_argument("--max-speed", type=float, default=1.3)
    parser.add_argument(
        "--min-volume-gain-db",
        type=float,
        default=-12.0,
        help="Minimum random volume gain in decibels.",
    )
    parser.add_argument(
        "--max-volume-gain-db",
        type=float,
        default=6.0,
        help="Maximum random volume gain in decibels.",
    )
    parser.add_argument(
        "--rir-dir",
        type=Path,
        default=Path("indonesian_data/room_noises"),
        help="Directory containing '*_rir_*.wav' room impulse responses.",
    )
    parser.add_argument("--min-reverb-wet", type=float, default=0.15)
    parser.add_argument("--max-reverb-wet", type=float, default=0.55)
    parser.add_argument("--time-dropout-probability", type=float, default=0.5)
    parser.add_argument("--min-time-dropout-ms", type=int, default=50)
    parser.add_argument("--max-time-dropout-ms", type=int, default=200)
    parser.add_argument("--min-time-dropout-count", type=int, default=1)
    parser.add_argument("--max-time-dropout-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of source rows to perturb concurrently.",
    )
    parser.add_argument("--limit", type=int, help="Perturb only the first N stage-03 rows.")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input manifest not found: {args.input}")
    if not args.rir_dir.is_dir():
        parser.error(f"RIR directory not found: {args.rir_dir}")
    if args.copies_per_row < 1:
        parser.error("--copies-per-row must be at least 1")
    if not 0 < args.min_speed <= args.max_speed:
        parser.error("Speed factors must satisfy 0 < min <= max")
    if args.min_speed <= 1 <= args.max_speed and args.min_speed == args.max_speed == 1:
        parser.error("A speed factor of exactly 1.0 would not augment the audio")
    if args.min_volume_gain_db > args.max_volume_gain_db:
        parser.error("--min-volume-gain-db cannot exceed --max-volume-gain-db")
    if args.min_volume_gain_db == args.max_volume_gain_db == 0:
        parser.error("A volume gain of exactly 0 dB would not augment the audio")
    if not 0 <= args.min_reverb_wet <= args.max_reverb_wet <= 1:
        parser.error("Reverb wet mix must satisfy 0 <= min <= max <= 1")
    if args.min_reverb_wet == args.max_reverb_wet == 0:
        parser.error("A reverb wet mix of exactly 0 would not augment the audio")
    if not 0 <= args.time_dropout_probability <= 1:
        parser.error("--time-dropout-probability must be between 0 and 1")
    if not 0 < args.min_time_dropout_ms <= args.max_time_dropout_ms:
        parser.error("Time-dropout duration must satisfy 0 < min <= max")
    if not 1 <= args.min_time_dropout_count <= args.max_time_dropout_count:
        parser.error("Time-dropout count must satisfy 1 <= min <= max")
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

    try:
        rir_files = find_rir_files(args.rir_dir)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_to_augment = rows if args.limit is None else rows[:args.limit]
    original_rows: list[dict[str, str]] = []
    perturbed_rows: list[dict[str, str]] = []
    for row in rows:
        original = dict(row)
        original.update({
            "speed_augmented": "false",
            "speed_factor": "1.0000",
            "speed_augmentation_source_audio": row["audio_path"],
            "volume_augmented": "false",
            "volume_gain_db": "0.0000",
            "reverb_augmented": "false",
            "reverb_wet": "0.0000",
            "reverb_rir": "",
            "time_dropout_augmented": "false",
            "time_dropout_chunks": "[]",
            "time_dropout_total_ms": "0",
        })
        original_rows.append(original)
    seed_rng = random.Random(args.seed)
    jobs = [
        PerturbationJob(
            row=row,
            output_dir=args.output_dir,
            rir_files=tuple(rir_files),
            copies_per_row=args.copies_per_row,
            minimum_speed=args.min_speed,
            maximum_speed=args.max_speed,
            minimum_volume_gain_db=args.min_volume_gain_db,
            maximum_volume_gain_db=args.max_volume_gain_db,
            minimum_reverb_wet=args.min_reverb_wet,
            maximum_reverb_wet=args.max_reverb_wet,
            time_dropout_probability=args.time_dropout_probability,
            minimum_time_dropout_ms=args.min_time_dropout_ms,
            maximum_time_dropout_ms=args.max_time_dropout_ms,
            minimum_time_dropout_count=args.min_time_dropout_count,
            maximum_time_dropout_count=args.max_time_dropout_count,
            seed=seed_rng.getrandbits(64),
        )
        for row in rows_to_augment
    ]
    if args.workers == 1 or len(jobs) < 2:
        results = map(perturb_row, jobs)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(args.workers, len(jobs)),
            initializer=configure_worker,
        )
        results = executor.map(perturb_row, jobs, chunksize=1)
    try:
        for index, augmented_rows in enumerate(results, start=1):
            perturbed_rows.extend(augmented_rows)
            print(
                f"[{index}/{len(jobs)}] audio perturbed "
                f"{augmented_rows[0]['speed_augmentation_source_audio']}"
            )
    finally:
        if executor is not None:
            executor.shutdown()

    all_rows = original_rows + perturbed_rows
    fieldnames = input_fields + [field for field in METADATA_FIELDS if field not in input_fields]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    args.comparison_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_rows = original_rows[:len(rows_to_augment)] + perturbed_rows
    with args.comparison_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_row(row) for row in comparison_rows)

    print(
        f"Wrote {len(original_rows)} existing + {len(perturbed_rows)} "
        f"augmented rows "
        f"to {args.output} using {args.workers} "
        f"worker{'s' if args.workers != 1 else ''}"
    )
    print(f"Wrote comparison manifest to {args.comparison_output}")
    link = viewer_url(args.comparison_output)
    if link:
        print(f"View augmentation comparisons: {link}")


if __name__ == "__main__":
    main()
