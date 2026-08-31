#!/usr/bin/env python3
"""Split Miami recordings into timestamp-aligned WAV snippets from CHAT files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

TIMESTAMP = re.compile("\x15(\\d+)_(\\d+)\x15")
SPEECH = re.compile(r"^\*([^:]+):\s*(.*)$")


def normalise_surface(surface: str) -> list[str]:
    """Apply training-text rules to a TSV token and return its retained words."""
    # Keep words in angle brackets, but discard parenthesised material completely.
    while re.search(r"\([^()]*\)", surface):
        surface = re.sub(r"\([^()]*\)", "", surface)
    surface = surface.replace("<", "").replace(">", "").replace("_", " ")
    # Remove all Unicode punctuation; retain letters, numbers, and combining marks.
    surface = "".join(char for char in surface if not unicodedata.category(char).startswith("P"))
    return [word for word in surface.split() if word and word.casefold() not in {"www", "xxx"}]


def read_tsv_tokens(tsv_file: Path) -> dict[int, list[dict[str, str]]]:
    """Read TSV labels keyed by the CHAT speaker-tier/utterance ID."""
    by_utterance: dict[int, list[dict[str, str]]] = defaultdict(list)
    with tsv_file.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if not row.get("utterance_id"):
                continue
            for word in normalise_surface(row["surface"]):
                by_utterance[int(row["utterance_id"])].append({"word": word, "langid": row["langid"]})
    return by_utterance


def read_utterances(chat_file: Path, tsv_tokens: dict[int, list[dict[str, str]]]) -> list[dict[str, object]]:
    utterances: list[dict[str, object]] = []
    utterance_id = 0
    for line in chat_file.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SPEECH.match(line)
        if not match:
            continue
        utterance_id += 1
        timing = TIMESTAMP.search(match.group(2))
        tokens = tsv_tokens.get(utterance_id, [])
        if timing and tokens:
            utterances.append({
                "speaker": match.group(1), "tokens": tokens, "utterance_id": utterance_id,
                "start_ms": int(timing.group(1)), "end_ms": int(timing.group(2)),
            })
    return utterances


def make_segments(utterances: list[dict[str, object]], max_ms: int) -> list[dict[str, object]]:
    """Combine consecutive utterances, never allowing a segment longer than max_ms."""
    segments: list[dict[str, object]] = []
    current: list[dict[str, object]] = []
    start = 0
    for utterance in utterances:
        candidate_start = start if current else int(utterance["start_ms"])
        if current and int(utterance["end_ms"]) - candidate_start > max_ms:
            segments.append({
                "start_ms": start, "end_ms": int(current[-1]["end_ms"]),
                "transcript": " ".join(token["word"] for item in current for token in item["tokens"]),
                "word_langids": [token for item in current for token in item["tokens"]],
                "language_counts": dict(Counter(token["langid"] for item in current for token in item["tokens"])),
                "utterance_ids": [item["utterance_id"] for item in current],
                "speakers": ",".join(dict.fromkeys(str(item["speaker"]) for item in current)),
            })
            current = []
            candidate_start = int(utterance["start_ms"])
        if not current:
            start = candidate_start
        current.append(utterance)
    if current:
        segments.append({
            "start_ms": start, "end_ms": int(current[-1]["end_ms"]),
            "transcript": " ".join(token["word"] for item in current for token in item["tokens"]),
            "word_langids": [token for item in current for token in item["tokens"]],
            "language_counts": dict(Counter(token["langid"] for item in current for token in item["tokens"])),
            "utterance_ids": [item["utterance_id"] for item in current],
            "speakers": ",".join(dict.fromkeys(str(item["speaker"]) for item in current)),
        })
    return segments


def cut_audio(ffmpeg: str, source: Path, destination: Path, start_ms: int, end_ms: int) -> None:
    duration = (end_ms - start_ms) / 1000
    subprocess.run([ffmpeg, "-y", "-v", "error", "-ss", f"{start_ms / 1000:.3f}", "-i", str(source),
                    "-t", f"{duration:.3f}", "-ac", "1", "-c:a", "pcm_s16le", str(destination)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=Path("miami/audios"))
    parser.add_argument("--chat-dir", type=Path, default=Path("miami/chat"))
    parser.add_argument("--tsv-dir", type=Path, default=Path("miami/word_level_tsvs"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/01_segments"))
    parser.add_argument("--manifest", type=Path, default=Path("processed/01_segments.csv"))
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-recordings", type=int, help="Process only the first N recordings (for previews).")
    parser.add_argument("--max-segments-per-recording", type=int, help="Keep only the first N segments from each recording.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg was not found: {args.ffmpeg}")
    if args.max_recordings is not None and args.max_recordings < 1:
        parser.error("--max-recordings must be at least 1")
    if args.max_segments_per_recording is not None and args.max_segments_per_recording < 1:
        parser.error("--max-segments-per-recording must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    sources = sorted(args.audio_dir.glob("*.mp3"))
    if args.max_recordings is not None:
        sources = sources[:args.max_recordings]
    for source in sources:
        chat_file = args.chat_dir / f"{source.stem}.cha"
        tsv_file = args.tsv_dir / f"{source.stem}_cgwords.tsv"
        if not chat_file.exists() or not tsv_file.exists():
            print(f"Skipping {source.name}: matching CHAT or word-level TSV file is missing")
            continue
        segments = make_segments(read_utterances(chat_file, read_tsv_tokens(tsv_file)), round(args.max_seconds * 1000))
        if args.max_segments_per_recording is not None:
            segments = segments[:args.max_segments_per_recording]
        for number, segment in enumerate(segments, start=1):
            output = args.output_dir / f"{source.stem}_{number:04d}.wav"
            cut_audio(args.ffmpeg, source, output, int(segment["start_ms"]), int(segment["end_ms"]))
            rows.append({"audio_path": str(output), "transcript": segment["transcript"], "source_audio": str(source),
                         "start_ms": segment["start_ms"], "end_ms": segment["end_ms"], "speakers": segment["speakers"],
                         "word_langids": json.dumps(segment["word_langids"], ensure_ascii=False),
                         "language_counts": json.dumps(segment["language_counts"], ensure_ascii=False, sort_keys=True),
                         "utterance_ids": json.dumps(segment["utterance_ids"])})

    with args.manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_path", "transcript", "source_audio", "start_ms", "end_ms", "speakers", "word_langids", "language_counts", "utterance_ids"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} segments and {args.manifest}")


if __name__ == "__main__":
    main()
