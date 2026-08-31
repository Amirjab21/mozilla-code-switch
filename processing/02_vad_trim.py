#!/usr/bin/env python3
"""Remove non-speech from segmented WAV files with Silero VAD."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import wave
from pathlib import Path


def merge_regions(regions: list[dict], max_gap_samples: int) -> list[dict]:
    merged: list[dict] = []
    for region in regions:
        if merged and region["start"] - merged[-1]["end"] <= max_gap_samples:
            merged[-1]["end"] = region["end"]
        else:
            merged.append(dict(region))
    return merged


def write_pcm16(destination: Path, samples, sample_rate: int) -> None:
    # Silero returns normalized float samples; convert once to mono signed 16-bit PCM.
    pcm = (samples.clamp(-1, 1) * 32767).to(dtype=__import__("torch").int16).numpy().tobytes()
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(sample_rate); output.writeframes(pcm)


def read_pcm16(ffmpeg: str, source: Path, sample_rate: int):
    """Decode with ffmpeg instead of torchaudio/TorchCodec (reliable on macOS)."""
    torch = __import__("torch")
    process = subprocess.run([ffmpeg, "-v", "error", "-i", str(source), "-ac", "1", "-ar", str(sample_rate),
                              "-f", "s16le", "pipe:1"], check=True, stdout=subprocess.PIPE)
    return torch.frombuffer(bytearray(process.stdout), dtype=torch.int16).float() / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed/01_segments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/02_vad"))
    parser.add_argument("--output-manifest", type=Path, default=Path("processed/02_vad.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge-gap-ms", type=int, default=20)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    try:
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as error:
        raise SystemExit("Install stage-2 dependencies first: uv sync --project processing") from error
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg was not found: {args.ffmpeg}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, sample_rate = load_silero_vad(), 16000
    with args.manifest.open(encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))
    output_rows: list[dict[str, str]] = []
    filtered_rows: list[dict[str, str]] = []
    for row in source_rows:
        waveform = read_pcm16(args.ffmpeg, Path(row["audio_path"]), sample_rate)
        regions = get_speech_timestamps(waveform, model, sampling_rate=sample_rate, threshold=args.threshold,
                                        min_silence_duration_ms=args.merge_gap_ms)
        regions = merge_regions(regions, round(sample_rate * args.merge_gap_ms / 1000))
        if not regions:
            print(f"Skipping no-speech segment: {row['audio_path']}")
            row["filtered_out"] = "True"
            filtered_rows.append(row)
            continue
        speech = __import__("torch").cat([waveform[item["start"]:item["end"]] for item in regions])
        output = args.output_dir / Path(row["audio_path"]).name
        write_pcm16(output, speech, sample_rate)
        row["audio_path"] = str(output)
        row["filtered_out"] = "False"
        output_rows.append(row)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [*source_rows[0].keys()] if source_rows else [
            "audio_path", "transcript", "filtered_out"
        ]
        if "filtered_out" not in fieldnames:
            fieldnames.append("filtered_out")
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows([*output_rows, *filtered_rows])
    print(f"Wrote {len(output_rows)} VAD-trimmed and {len(filtered_rows)} filtered segments "
          f"to {args.output_manifest}")


if __name__ == "__main__":
    main()
