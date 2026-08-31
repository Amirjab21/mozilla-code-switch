#!/usr/bin/env python3
"""Resample VAD-trimmed WAV files to 16 kHz and normalize their loudness."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed/02_vad.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/audio"))
    parser.add_argument("--output-csv", type=Path, default=Path("processed/train.csv"))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg was not found: {args.ffmpeg}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    train_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("filtered_out", "False").casefold() == "true":
            continue
        output = args.output_dir / Path(row["audio_path"]).name
        subprocess.run([args.ffmpeg, "-y", "-v", "error", "-i", row["audio_path"], "-ac", "1", "-ar", "16000",
                        "-af", "loudnorm=I=-23:TP=-2:LRA=7", "-c:a", "pcm_s16le", str(output)], check=True)
        train_rows.append({"audio_path": str(output), "transcript": row["transcript"],
                           "word_langids": row.get("word_langids", "[]"),
                           "language_counts": row.get("language_counts", "{}")})
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["audio_path", "transcript", "word_langids", "language_counts"])
        writer.writeheader(); writer.writerows(train_rows)
    print(f"Wrote {len(train_rows)} normalized files and training CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
