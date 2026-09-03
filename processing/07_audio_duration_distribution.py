#!/usr/bin/env python3
"""Plot the probability distribution of training-clip durations from a manifest.

The script uses Python's standard-library ``wave`` module, so it needs no
additional packages.  It expects WAV files and writes a portable SVG histogram.

Example:
    uv run --project processing python processing/07_audio_duration_distribution.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import wave
from pathlib import Path
from xml.sax.saxutils import escape


def read_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        frame_rate = audio.getframerate()
        if frame_rate <= 0:
            raise ValueError("audio has an invalid frame rate")
        return audio.getnframes() / frame_rate


def histogram(values: list[float], bin_count: int) -> tuple[list[float], list[int]]:
    low, high = min(values), max(values)
    if math.isclose(low, high):
        padding = max(low * 0.05, 0.5)
        low, high = max(0.0, low - padding), high + padding
    width = (high - low) / bin_count
    edges = [low + index * width for index in range(bin_count + 1)]
    counts = [0] * bin_count
    for value in values:
        index = min(int((value - low) / width), bin_count - 1)
        counts[index] += 1
    return edges, counts


def write_svg(path: Path, durations: list[float], bin_count: int, title: str) -> None:
    edges, counts = histogram(durations, bin_count)
    probabilities = [count / len(durations) for count in counts]
    width, height = 1000, 620
    left, right, top, bottom = 94, 32, 34, 92
    chart_width, chart_height = width - left - right, height - top - bottom
    max_probability = max(probabilities) or 1.0
    y_max = math.ceil(max_probability * 10) / 10
    y_max = max(y_max, 0.1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        f"<title id=\"title\">{escape(title)}</title>",
        f"<desc id=\"description\">Histogram of {len(durations)} training audio clip durations. The vertical axis shows the probability that a clip falls in each duration bin.</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#162033}.title{font-size:22px;font-weight:650}.axis{font-size:13px}.tick{font-size:12px;fill:#536077}.bar{fill:#2d7ff9}.grid{stroke:#dce3ee;stroke-width:1}.axis-line{stroke:#4d5a70;stroke-width:1.2}</style>',
        f'<text class="title" x="{left}" y="28">{escape(title)}</text>',
        f'<line class="axis-line" x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" y2="{top + chart_height}"/>',
        f'<line class="axis-line" x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}"/>',
    ]
    for tick in range(6):
        probability = y_max * tick / 5
        y = top + chart_height - (probability / y_max) * chart_height
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + chart_width}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.1f}" text-anchor="end">{probability:.0%}</text>')
    bar_width = chart_width / bin_count
    for index, probability in enumerate(probabilities):
        bar_height = probability / y_max * chart_height
        x = left + index * bar_width + 1
        y = top + chart_height - bar_height
        parts.append(f'<rect class="bar" x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 0):.2f}" height="{bar_height:.2f}"><title>{edges[index]:.2f}–{edges[index + 1]:.2f} seconds: {counts[index]} clips ({probability:.1%})</title></rect>')
    for tick in range(6):
        value = edges[0] + (edges[-1] - edges[0]) * tick / 5
        x = left + chart_width * tick / 5
        parts.append(f'<text class="tick" x="{x:.1f}" y="{top + chart_height + 23}" text-anchor="middle">{value:.1f}</text>')
    mean = statistics.mean(durations)
    median = statistics.median(durations)
    parts.extend([
        f'<text class="axis" x="{left + chart_width / 2}" y="{height - 26}" text-anchor="middle">Audio duration (seconds)</text>',
        f'<text class="axis" x="18" y="{top + chart_height / 2}" text-anchor="middle" transform="rotate(-90 18 {top + chart_height / 2})">Probability of clip</text>',
        f'<text class="axis" x="{left}" y="{height - 5}">{len(durations):,} clips · mean {mean:.2f}s · median {median:.2f}s · range {min(durations):.2f}–{max(durations):.2f}s</text>',
        "</svg>",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/train.csv"))
    parser.add_argument("--audio-root", type=Path, default=Path("."), help="Base directory for relative audio_path values.")
    parser.add_argument("--output", type=Path, default=Path("analysis_indonesia/train_audio_duration_distribution.svg"))
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be at least 1")

    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows or "audio_path" not in rows[0]:
        raise ValueError(f"{args.manifest} must contain an audio_path column")

    durations: list[float] = []
    failures: list[str] = []
    for row in rows:
        configured_path = Path(row["audio_path"])
        audio_path = configured_path if configured_path.is_absolute() else args.audio_root / configured_path
        try:
            durations.append(read_duration_seconds(audio_path))
        except (OSError, EOFError, ValueError, wave.Error) as error:
            failures.append(f"{audio_path}: {error}")
    if not durations:
        raise RuntimeError("No readable WAV files were found in the manifest")

    write_svg(args.output, durations, args.bins, "Training audio duration distribution")
    print(
        f"Wrote {args.output} for {len(durations):,} clips "
        f"(mean {statistics.mean(durations):.2f}s; median {statistics.median(durations):.2f}s; "
        f"range {min(durations):.2f}–{max(durations):.2f}s)."
    )
    if failures:
        print(f"Skipped {len(failures):,} unreadable files; first: {failures[0]}")


if __name__ == "__main__":
    main()
