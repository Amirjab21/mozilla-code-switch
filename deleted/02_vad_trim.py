#!/usr/bin/env python3
"""Use Silero VAD to reject silent clips without altering approved audio."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote


def repository_relative(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def viewer_url(config: Path) -> str | None:
    repository = Path(__file__).resolve().parent.parent
    try:
        relative = config.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    value = quote(f"../{relative}", safe="/")
    return (
        "http://127.0.0.1:8000/analysis_indonesia/"
        f"vad_comparison.html?config={value}"
    )


def read_pcm16(ffmpeg: str, source: Path, sample_rate: int):
    """Decode with ffmpeg instead of torchaudio/TorchCodec (reliable on macOS)."""
    torch = __import__("torch")
    process = subprocess.run([ffmpeg, "-v", "error", "-i", str(source), "-ac", "1", "-ar", str(sample_rate),
                              "-f", "s16le", "pipe:1"], check=True, stdout=subprocess.PIPE)
    return torch.frombuffer(bytearray(process.stdout), dtype=torch.int16).float() / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/01_segments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia/02_vad"))
    parser.add_argument("--output-manifest", type=Path, default=Path("processed_indonesia/02_vad.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-words", type=int, default=2, help="Reject transcripts with fewer words.")
    parser.add_argument("--min-seconds", type=float, default=3.0, help="Reject clips shorter than this duration.")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--sample-size", type=int, help="Randomly select this many manifest rows before VAD filtering.")
    sampling.add_argument("--target-approved", type=int, help="Process shuffled rows until this many clips pass all filters.")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed used for repeatable sampling or shuffling.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.target_approved is not None and args.target_approved < 1:
        parser.error("--target-approved must be at least 1")

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as error:
        raise SystemExit("Install stage-2 dependencies first: uv sync --project processing") from error
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg was not found: {args.ffmpeg}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    model, sample_rate = load_silero_vad(), 16000
    with args.manifest.open(encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))
    source_audio_directories = {
        Path(row["audio_path"]).parent.resolve() for row in source_rows
    }
    if len(source_audio_directories) > 1:
        parser.error("The VAD comparison viewer requires one stage-1 audio directory")
    source_audio_directory = next(iter(source_audio_directories), args.manifest.parent)
    if args.sample_size is not None:
        if args.sample_size > len(source_rows):
            parser.error(f"--sample-size {args.sample_size} exceeds the {len(source_rows)} manifest rows")
        source_rows = random.Random(args.random_seed).sample(source_rows, args.sample_size)
    elif args.target_approved is not None:
        if args.target_approved > len(source_rows):
            parser.error(f"--target-approved {args.target_approved} exceeds the {len(source_rows)} manifest rows")
        random.Random(args.random_seed).shuffle(source_rows)
    output_rows: list[dict[str, str]] = []
    filtered_rows: list[dict[str, str]] = []
    for row in source_rows:
        if args.target_approved is not None and len(output_rows) >= args.target_approved:
            break
        waveform = read_pcm16(args.ffmpeg, Path(row["audio_path"]), sample_rate)
        word_count = len(row.get("transcript", "").split())
        duration_seconds = len(waveform) / sample_rate
        row["word_count"] = str(word_count)
        row["source_duration_seconds"] = f"{duration_seconds:.3f}"
        if word_count < args.min_words:
            row["filtered_out"] = "True"
            row["filter_reason"] = "too_few_words"
            filtered_rows.append(row)
            continue
        if duration_seconds < args.min_seconds:
            row["filtered_out"] = "True"
            row["filter_reason"] = "too_short"
            filtered_rows.append(row)
            continue
        regions = get_speech_timestamps(waveform, model, sampling_rate=sample_rate, threshold=args.threshold,
                                        min_silence_duration_ms=20)
        if not regions:
            print(f"Skipping no-speech segment: {row['audio_path']}")
            row["filtered_out"] = "True"
            row["filter_reason"] = "no_vad_speech"
            filtered_rows.append(row)
            continue
        output = args.output_dir / Path(row["audio_path"]).name
        # VAD is a quality gate only.  Concatenating speech regions removes pauses
        # and can make the audio no longer match its transcript timestamps.
        shutil.copy2(row["audio_path"], output)
        row["audio_path"] = str(output)
        row["filtered_out"] = "False"
        row["filter_reason"] = ""
        output_rows.append(row)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [*source_rows[0].keys()] if source_rows else [
            "audio_path", "transcript", "filtered_out"
        ]
        if "filtered_out" not in fieldnames:
            fieldnames.append("filtered_out")
        for name in ("filter_reason", "word_count", "source_duration_seconds"):
            if name not in fieldnames:
                fieldnames.append(name)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows([*output_rows, *filtered_rows])
    print(f"Wrote {len(output_rows)} whole clips and {len(filtered_rows)} filtered segments "
          f"to {args.output_manifest}")
    config = args.output_manifest.parent / "vad_review.json"
    config.write_text(json.dumps({
        "title": f"VAD review — {args.output_manifest.parent.name}",
        "paths": {
            "vad_manifest": repository_relative(
                args.output_manifest, config.parent
            ),
            "review_audio_dir": repository_relative(
                source_audio_directory, config.parent
            ) + "/",
            "vad_audio_dir": repository_relative(
                args.output_dir, config.parent
            ) + "/",
        },
    }, indent=2) + "\n", encoding="utf-8")
    link = viewer_url(config)
    if link:
        print(f"View VAD results: {link}")
    else:
        print("VAD viewer link unavailable: output is outside the repository.")


if __name__ == "__main__":
    main()
