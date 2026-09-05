#!/usr/bin/env python3
"""Create blockwise expanding Jember windows and transcode development clips.

Every output is mono 16 kHz PCM WAV. From each selected Jember starting row,
stage 1 emits every valid expanding prefix until another row would exceed the
maximum duration. The next block retains a configurable number of rows from
the end of that longest prefix, preserving boundary overlap without restarting
at every row.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

from run_config import apply_defaults, load_section


FIELDNAMES = [
    "audio_path", "transcript", "source_audio", "start_ms", "end_ms",
    "speakers", "word_langids", "language_counts", "utterance_ids",
    "dataset", "alignment_status", "alignment_confidence", "word_timestamps",
]
EXCLUDED_FIELDNAMES = FIELDNAMES + ["exclusion_reason", "duration_ms"]
DEVELOPMENT_MAX_DURATION_MS = 30_000
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


@dataclass(frozen=True)
class JemberRecordingJob:
    recording_id: str
    source: Path
    raw_rows: list[dict[str, str]]
    output_dir: Path
    ffmpeg: str
    ffprobe: str
    minimum_ms: int
    maximum_ms: int
    minimum_words: int
    stride_overlap_rows: int
    reuse_existing: bool
    clip_limit: int | None = None


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


def decode_jember_source(ffmpeg: str, source: Path, destination: Path) -> None:
    """Decode one complete Jember MP3 to canonical PCM exactly once."""
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error", "-i", str(source),
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
        ],
        check=True,
    )


def write_jember_pcm_windows(
    decoded_source: Path,
    outputs: list[tuple[JemberWindow, Path]],
) -> None:
    """Read decoded PCM once and write all requested timestamp slices."""
    with wave.open(str(decoded_source), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        compression = reader.getcomptype()
        frame_count = reader.getnframes()
        pcm = reader.readframes(frame_count)
    if (channels, sample_width, sample_rate, compression) != (1, 2, 16_000, "NONE"):
        raise ValueError(
            f"Unexpected decoded Jember format in {decoded_source}: "
            f"channels={channels}, sample_width={sample_width}, "
            f"sample_rate={sample_rate}, compression={compression}"
        )
    frame_width = channels * sample_width
    for window, destination in outputs:
        start_frame = min(frame_count, round(window.start_ms * sample_rate / 1000))
        end_frame = min(frame_count, round(window.end_ms * sample_rate / 1000))
        if end_frame <= start_frame:
            raise ValueError(
                f"Invalid PCM slice for {destination}: {start_frame}:{end_frame}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(sample_width)
            writer.setframerate(sample_rate)
            writer.setcomptype(compression, "not compressed")
            writer.writeframes(pcm[start_frame * frame_width : end_frame * frame_width])


def prepare_jember_recording(
    job: JemberRecordingJob,
) -> tuple[str, int, list[dict[str, str]]]:
    """Prepare one recording, decoding its compressed audio at most once."""
    parsed_rows = parse_jember_rows(
        job.raw_rows, duration_ms(job.ffprobe, job.source)
    )
    windows = list(
        enumerate_jember_windows(
            parsed_rows,
            job.minimum_ms,
            job.maximum_ms,
            job.minimum_words,
            job.stride_overlap_rows,
        )
    )
    if job.clip_limit is not None:
        windows = windows[: job.clip_limit]

    output_pairs = [
        (
            window,
            job.output_dir
            / (
                f"jember_{int(job.recording_id):03d}_"
                f"{window.first_row:04d}-{window.last_row:04d}.wav"
            ),
        )
        for window in windows
    ]
    missing_outputs = [
        pair
        for pair in output_pairs
        if not (job.reuse_existing and pair[1].is_file())
    ]
    if missing_outputs:
        # Keep the temporary PCM on the same filesystem as the outputs. Each
        # worker owns its directory, and it is removed after all slices finish.
        with tempfile.TemporaryDirectory(
            prefix=f".jember_{job.recording_id}_", dir=job.output_dir
        ) as temporary_directory:
            decoded_source = Path(temporary_directory) / "source_16khz.wav"
            decode_jember_source(job.ffmpeg, job.source, decoded_source)
            write_jember_pcm_windows(decoded_source, missing_outputs)

    records = []
    for window, output in output_pairs:
        utterance_ids = [
            f"jember:{job.recording_id}:{number}"
            for number in window.row_numbers
        ]
        records.append(
            make_row(
                output,
                window.transcript,
                job.source,
                window.start_ms,
                window.end_ms,
                "unknown",
                utterance_ids,
                "jember",
                "tsv_row_window",
            )
        )
    return job.recording_id, len(parsed_rows), records


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
    stride_overlap_rows: int = 2,
) -> Iterator[JemberWindow]:
    """Yield expanding prefixes, retaining rows at each block boundary."""
    first_index = 0
    while first_index < len(source_rows):
        first = source_rows[first_index]
        transcript_parts: list[str] = []
        word_count = 0
        longest_valid_index: int | None = None
        reached_recording_end = False
        for last_index in range(first_index, len(source_rows)):
            last = source_rows[last_index]
            text = last.source.get("text", "").strip()
            if text:
                transcript_parts.append(text)
                word_count += transcript_word_count(text)
            duration = last.end_ms - first.start_ms
            if duration > maximum_ms:
                break
            reached_recording_end = last_index == len(source_rows) - 1
            if duration < minimum_ms or word_count < minimum_words or duration <= 0:
                continue
            longest_valid_index = last_index
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

        # Once a block has expanded through the final row, every remaining
        # suffix would only duplicate audio already present in this tail block.
        if reached_recording_end:
            break
        if longest_valid_index is None:
            # A malformed/overlong starting row must never stall enumeration.
            first_index += 1
            continue
        # If the longest prefix was [start, ..., end], retain the configured
        # number of rows at its end. An overlap of two therefore starts at the
        # second-to-last row. Always advance at least one row.
        next_index = longest_valid_index - stride_overlap_rows + 1
        first_index = max(first_index + 1, next_index)


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
    jobs: list[JemberRecordingJob] = []
    for recording_id in recording_ids:
        source = args.jember_audio_dir / f"{recording_id}.mp3"
        if not source.exists():
            print(f"Skipping missing Jember audio: {source}")
            continue
        selected_rows = by_recording[recording_id]
        if args.max_segments_per_recording is not None:
            selected_rows = selected_rows[:args.max_segments_per_recording]
        jobs.append(
            JemberRecordingJob(
                recording_id=recording_id,
                source=source,
                raw_rows=selected_rows,
                output_dir=args.output_dir,
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
                minimum_ms=minimum_ms,
                maximum_ms=maximum_ms,
                minimum_words=args.min_words,
                stride_overlap_rows=args.jember_stride_overlap_rows,
                reuse_existing=args.reuse_existing,
            )
        )

    if not jobs:
        return
    if limit is not None:
        if args.prepare_workers > 1:
            print(
                "A global Jember clip limit is active; using one preparation "
                "worker so the limit remains deterministic."
            )
        emitted = 0
        results = []
        for job in jobs:
            remaining = limit - emitted
            if remaining <= 0:
                break
            limited_job = replace(job, clip_limit=remaining)
            result = prepare_jember_recording(limited_job)
            results.append(result)
            emitted += len(result[2])
    elif args.prepare_workers == 1:
        results = [prepare_jember_recording(job) for job in jobs]
    else:
        worker_count = min(args.prepare_workers, len(jobs))
        print(f"Preparing {len(jobs)} Jember recordings with {worker_count} workers")
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(prepare_jember_recording, jobs))

    for recording_id, parsed_count, recording_rows in results:
        output_rows.extend(recording_rows)
        print(
            f"Prepared Jember recording {recording_id}: "
            f"{parsed_count} TSV rows -> {len(recording_rows)} overlapping clips"
        )


def prepare_development(
    args: argparse.Namespace,
    output_rows: list[dict[str, str]],
    excluded_rows: list[dict[str, str]],
    limit: int | None,
) -> None:
    with args.development_manifest.open(encoding="utf-8", newline="") as file:
        development_rows = list(csv.DictReader(file, delimiter="\t"))
    if limit is not None:
        development_rows = development_rows[:limit]
    minimum_ms = round(args.min_clip_seconds * 1000)
    for number, row in enumerate(development_rows, start=1):
        source = args.development_audio_dir / row["audio_filename"]
        if not source.exists():
            print(f"Skipping missing development audio: {source}")
            continue
        transcript = row["transcript"].strip()
        source_duration_ms = duration_ms(args.ffprobe, source)
        if transcript_word_count(transcript) < args.min_words:
            print(f"Skipping development clip with fewer than {args.min_words} words: {source}")
            excluded = make_row(
                source, transcript, source, 0, source_duration_ms,
                row.get("speaker", "unknown"), [f"indonesian_dev:{number}"],
                "indonesian_development", "excluded",
            )
            excluded.update({
                "exclusion_reason": "fewer_than_minimum_words",
                "duration_ms": str(source_duration_ms),
            })
            excluded_rows.append(excluded)
            continue
        if source_duration_ms > DEVELOPMENT_MAX_DURATION_MS:
            print(
                f"Excluding development clip over 30 seconds: {source} "
                f"({source_duration_ms / 1000:.3f}s)"
            )
            excluded = make_row(
                source, transcript, source, 0, source_duration_ms,
                row.get("speaker", "unknown"), [f"indonesian_dev:{number}"],
                "indonesian_development", "excluded",
            )
            excluded.update({
                "exclusion_reason": "development_audio_over_30_seconds",
                "duration_ms": str(source_duration_ms),
            })
            excluded_rows.append(excluded)
            continue
        if source_duration_ms < minimum_ms:
            print(
                f"Excluding development clip shorter than {args.min_clip_seconds:g} "
                f"seconds: {source} ({source_duration_ms / 1000:.3f}s)"
            )
            excluded = make_row(
                source, transcript, source, 0, source_duration_ms,
                row.get("speaker", "unknown"), [f"indonesian_dev:{number}"],
                "indonesian_development", "excluded",
            )
            excluded.update({
                "exclusion_reason": "shorter_than_minimum_duration",
                "duration_ms": str(source_duration_ms),
            })
            excluded_rows.append(excluded)
            continue
        output = args.output_dir / f"indonesian_dev_{source.stem}.wav"
        if not (args.reuse_existing and output.is_file()):
            transcode(args.ffmpeg, source, output)
        end_ms = duration_ms(args.ffprobe, output)
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
    parser.add_argument(
        "--excluded-manifest",
        type=Path,
        help="Excluded development clips CSV (default: 01_excluded.csv beside --manifest).",
    )
    parser.add_argument("--jember-clips", type=int)
    parser.add_argument("--development-clips", type=int)
    parser.add_argument("--preview-per-dataset", type=int, help="Set both dataset limits; 100 creates up to 200 clips.")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--max-segments-per-recording", type=int, help="Read only the first N TSV rows per Jember recording.")
    parser.add_argument("--max-development-clips", type=int, help="Deprecated alias for --development-clips.")
    parser.add_argument("--max-clips", type=int, help="Deprecated final combined cap.")
    parser.add_argument("--min-clip-seconds", type=float, default=3.0)
    parser.add_argument("--max-clip-seconds", type=float, default=30.0)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument(
        "--jember-stride-overlap-rows",
        type=int,
        default=2,
        help=(
            "Rows retained from the end of the longest Jember window when "
            "starting the next expanding block (2 starts at the second-last row)."
        ),
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=1,
        help="Number of Jember recordings to decode and slice concurrently.",
    )
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
    if args.jember_stride_overlap_rows < 0:
        parser.error("--jember-stride-overlap-rows cannot be negative")
    if args.prepare_workers < 1:
        parser.error("--prepare-workers must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.excluded_manifest is None:
        args.excluded_manifest = args.manifest.with_name("01_excluded.csv")
    args.excluded_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    excluded_rows: list[dict[str, str]] = []
    prepare_jember(args, output_rows, args.jember_clips)
    prepare_development(
        args, output_rows, excluded_rows, args.development_clips
    )
    if args.max_clips is not None:
        output_rows = output_rows[:args.max_clips]
    with args.manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)
    with args.excluded_manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EXCLUDED_FIELDNAMES)
        writer.writeheader()
        writer.writerows(excluded_rows)
    counts = Counter(row["dataset"] for row in output_rows)
    print(
        f"Wrote {len(output_rows)} stage-1 segments to "
        f"{args.manifest}: {dict(counts)}"
    )
    print(
        f"Wrote {len(excluded_rows)} excluded development clips to "
        f"{args.excluded_manifest}"
    )
    link = viewer_url("segment_review.html", "manifest", args.manifest)
    if link:
        print(f"View stage-1 results: {link}")
    else:
        print("Stage-1 viewer link unavailable: manifest is outside the repository.")


if __name__ == "__main__":
    main()
