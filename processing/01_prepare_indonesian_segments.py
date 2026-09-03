#!/usr/bin/env python3
"""Create overlapping Jember row windows and transcode development clips.

Every output is mono 16 kHz PCM WAV. For each Jember recording, stage 1 emits
every contiguous TSV row window whose duration is within the configured range
and whose combined transcript contains enough words. Windows intentionally
overlap: after exhausting the valid extensions from one starting row, the
start advances by one row and enumeration begins again.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from run_config import apply_defaults, load_section


FIELDNAMES = [
    "audio_path", "transcript", "source_audio", "start_ms", "end_ms",
    "speakers", "word_langids", "language_counts", "utterance_ids",
    "dataset", "alignment_status", "alignment_confidence", "word_timestamps",
]
WORD_PATTERN = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


def viewer_url(viewer: str, parameter: str, artifact: Path) -> str | None:
    """Build a localhost viewer URL when the artifact is inside the repo."""
    repository = Path(__file__).resolve().parent.parent
    try:
        relative = artifact.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    value = quote(f"../{relative}", safe="/")
    return f"http://127.0.0.1:8000/analysis_indonesia/{viewer}?{parameter}={value}"


@dataclass(frozen=True)
class JemberRow:
    source: dict[str, str]
    number: int
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class JemberWindow:
    first_row: int
    last_row: int
    row_numbers: tuple[int, ...]
    start_ms: int
    end_ms: int
    transcript: str


def parse_timestamp(value: str) -> int:
    hours, minutes, seconds = value.strip().split(":")
    return round((int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000)


def transcript_word_count(text: str) -> int:
    return len(WORD_PATTERN.findall(text))


def tokens(transcript: str) -> list[dict[str, str]]:
    return [{"word": word, "langid": "other"} for word in transcript.split()]


def transcode(
    ffmpeg: str,
    source: Path,
    destination: Path,
    *,
    start_ms: int = 0,
    end_ms: int | None = None,
) -> None:
    command = [ffmpeg, "-y", "-v", "error", "-i", str(source)]
    if start_ms:
        command += ["-ss", f"{start_ms / 1000:.3f}"]
    if end_ms is not None:
        command += ["-t", f"{(end_ms - start_ms) / 1000:.3f}"]
    command += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
    subprocess.run(command, check=True)


def duration_ms(ffprobe: str, source: Path) -> int:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(source)],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return round(float(result.stdout.strip()) * 1000)


def make_row(
    output: Path,
    transcript: str,
    source: Path,
    start_ms: int,
    end_ms: int,
    speakers: str,
    utterance_ids: list[str],
    dataset: str,
    status: str,
) -> dict[str, str]:
    word_langids = tokens(transcript)
    return {
        "audio_path": str(output),
        "transcript": transcript,
        "source_audio": str(source),
        "start_ms": str(start_ms),
        "end_ms": str(end_ms),
        "speakers": speakers,
        "word_langids": json.dumps(word_langids, ensure_ascii=False),
        "language_counts": json.dumps(
            Counter(item["langid"] for item in word_langids),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "utterance_ids": json.dumps(utterance_ids),
        "dataset": dataset,
        "alignment_status": status,
        "alignment_confidence": "",
        "word_timestamps": "[]",
    }


def parse_jember_rows(
    raw_rows: list[dict[str, str]], recording_duration_ms: int
) -> list[JemberRow]:
    parsed = []
    for number, row in enumerate(raw_rows, start=1):
        try:
            start_ms = max(0, parse_timestamp(row["start"]))
            end_ms = min(recording_duration_ms, parse_timestamp(row["end"]))
        except (KeyError, TypeError, ValueError) as error:
            print(f"Skipping malformed Jember row {number}: {error}")
            continue
        parsed.append(JemberRow(row, number, start_ms, end_ms))
    return parsed


def enumerate_jember_windows(
    source_rows: list[JemberRow],
    minimum_ms: int,
    maximum_ms: int,
    minimum_words: int,
) -> Iterator[JemberWindow]:
    """Yield every valid contiguous row window in start-row breadth order."""
    for first_index, first in enumerate(source_rows):
        transcript_parts: list[str] = []
        word_count = 0
        for last_index in range(first_index, len(source_rows)):
            last = source_rows[last_index]
            text = last.source.get("text", "").strip()
            if text:
                transcript_parts.append(text)
                word_count += transcript_word_count(text)
            duration = last.end_ms - first.start_ms
            if duration > maximum_ms:
                break
            if duration < minimum_ms or word_count < minimum_words or duration <= 0:
                continue
            yield JemberWindow(
                first_row=first.number,
                last_row=last.number,
                row_numbers=tuple(
                    row.number for row in source_rows[first_index:last_index + 1]
                ),
                start_ms=first.start_ms,
                end_ms=last.end_ms,
                transcript=" ".join(transcript_parts),
            )


def prepare_jember(
    args: argparse.Namespace,
    output_rows: list[dict[str, str]],
    limit: int | None,
) -> None:
    with args.jember_manifest.open(encoding="utf-8", newline="") as file:
        raw_rows = list(csv.DictReader(file, delimiter="\t"))
    by_recording: dict[str, list[dict[str, str]]] = {}
    for row in raw_rows:
        by_recording.setdefault(row["Audio file name"], []).append(row)
    recording_ids = sorted(
        by_recording,
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if args.max_recordings is not None:
        recording_ids = recording_ids[:args.max_recordings]

    minimum_ms = round(args.min_clip_seconds * 1000)
    maximum_ms = round(args.max_clip_seconds * 1000)
    emitted = 0
    for recording_id in recording_ids:
        if limit is not None and emitted >= limit:
            break
        source = args.jember_audio_dir / f"{recording_id}.mp3"
        if not source.exists():
            print(f"Skipping missing Jember audio: {source}")
            continue
        selected_rows = by_recording[recording_id]
        if args.max_segments_per_recording is not None:
            selected_rows = selected_rows[:args.max_segments_per_recording]
        parsed_rows = parse_jember_rows(
            selected_rows, duration_ms(args.ffprobe, source)
        )
        recording_count = 0
        for window in enumerate_jember_windows(
            parsed_rows, minimum_ms, maximum_ms, args.min_words
        ):
            if limit is not None and emitted >= limit:
                break
            output = args.output_dir / (
                f"jember_{int(recording_id):03d}_"
                f"{window.first_row:04d}-{window.last_row:04d}.wav"
            )
            if not (args.reuse_existing and output.is_file()):
                transcode(
                    args.ffmpeg,
                    source,
                    output,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                )
            utterance_ids = [
                f"jember:{recording_id}:{number}"
                for number in window.row_numbers
            ]
            output_rows.append(make_row(
                output,
                window.transcript,
                source,
                window.start_ms,
                window.end_ms,
                "unknown",
                utterance_ids,
                "jember",
                "tsv_row_window",
            ))
            emitted += 1
            recording_count += 1
        print(
            f"Prepared Jember recording {recording_id}: "
            f"{len(parsed_rows)} TSV rows -> {recording_count} overlapping clips"
        )


def prepare_development(
    args: argparse.Namespace,
    output_rows: list[dict[str, str]],
    limit: int | None,
) -> None:
    with args.development_manifest.open(encoding="utf-8", newline="") as file:
        development_rows = list(csv.DictReader(file, delimiter="\t"))
    if limit is not None:
        development_rows = development_rows[:limit]
    minimum_ms = round(args.min_clip_seconds * 1000)
    maximum_ms = round(args.max_clip_seconds * 1000)
    for number, row in enumerate(development_rows, start=1):
        source = args.development_audio_dir / row["audio_filename"]
        if not source.exists():
            print(f"Skipping missing development audio: {source}")
            continue
        transcript = row["transcript"].strip()
        if transcript_word_count(transcript) < args.min_words:
            print(f"Skipping development clip with fewer than {args.min_words} words: {source}")
            continue
        output = args.output_dir / f"indonesian_dev_{source.stem}.wav"
        if not (args.reuse_existing and output.is_file()):
            transcode(args.ffmpeg, source, output)
        end_ms = duration_ms(args.ffprobe, output)
        if not minimum_ms <= end_ms <= maximum_ms:
            print(
                f"Skipping out-of-range development clip {source}: "
                f"{end_ms / 1000:.2f}s"
            )
            continue
        output_rows.append(make_row(
            output,
            transcript,
            source,
            0,
            end_ms,
            row.get("speaker", "unknown"),
            [f"indonesian_dev:{number}"],
            "indonesian_development",
            "source_clip",
        ))


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, help="YAML named-run configuration file."
    )
    config_args, _ = config_parser.parse_known_args()
    config_values, _ = load_section(config_args.config, "prepare")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="YAML named-run configuration file."
    )
    parser.add_argument("--jember-manifest", type=Path, default=Path("indonesian_data/Jember Javanese Spontaneous Speech Corpus/Jember Javanese Spontaneous Speech Corpus - 1-200.tsv"))
    parser.add_argument("--jember-audio-dir", type=Path, default=Path("indonesian_data/Jember Javanese Spontaneous Speech Corpus/mp3 audio"))
    parser.add_argument("--development-manifest", type=Path, default=Path("indonesian_data/indonesian_dev/metadata.tsv"))
    parser.add_argument("--development-audio-dir", type=Path, default=Path("indonesian_data/indonesian_dev/clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia/01_segments"))
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/01_segments.csv"))
    parser.add_argument("--jember-clips", type=int)
    parser.add_argument("--development-clips", type=int)
    parser.add_argument("--preview-per-dataset", type=int, help="Set both dataset limits; 100 creates up to 200 clips.")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--max-segments-per-recording", type=int, help="Read only the first N TSV rows per Jember recording.")
    parser.add_argument("--max-development-clips", type=int, help="Deprecated alias for --development-clips.")
    parser.add_argument("--max-clips", type=int, help="Deprecated final combined cap.")
    parser.add_argument("--min-clip-seconds", type=float, default=3.0)
    parser.add_argument("--max-clip-seconds", type=float, default=40.0)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    if shutil.which(args.ffmpeg) is None or shutil.which(args.ffprobe) is None:
        parser.error("ffmpeg and ffprobe must be installed")
    for path in (
        args.jember_manifest,
        args.jember_audio_dir,
        args.development_manifest,
        args.development_audio_dir,
    ):
        if not path.exists():
            parser.error(f"Input not found: {path}")
    if args.preview_per_dataset is not None:
        args.jember_clips = args.development_clips = args.preview_per_dataset
    if args.max_development_clips is not None and args.development_clips is None:
        args.development_clips = args.max_development_clips
    if args.min_clip_seconds <= 0 or args.max_clip_seconds <= args.min_clip_seconds:
        parser.error("--max-clip-seconds must be greater than --min-clip-seconds > 0")
    if args.min_words < 1:
        parser.error("--min-words must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    prepare_jember(args, output_rows, args.jember_clips)
    prepare_development(args, output_rows, args.development_clips)
    if args.max_clips is not None:
        output_rows = output_rows[:args.max_clips]
    with args.manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)
    counts = Counter(row["dataset"] for row in output_rows)
    print(
        f"Wrote {len(output_rows)} stage-1 segments to "
        f"{args.manifest}: {dict(counts)}"
    )
    link = viewer_url("segment_review.html", "manifest", args.manifest)
    if link:
        print(f"View stage-1 results: {link}")
    else:
        print("Stage-1 viewer link unavailable: manifest is outside the repository.")


if __name__ == "__main__":
    main()
