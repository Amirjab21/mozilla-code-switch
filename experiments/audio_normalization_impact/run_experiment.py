#!/usr/bin/env python3
"""Prepare and evaluate the audio-normalization impact experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSING = REPO_ROOT / "processing"
EXPERIMENT = Path(__file__).resolve().parent
DEFAULT_OUT = REPO_ROOT / "processed_indonesia" / "audio_normalization_impact"
EMPTY_JEMBER = EXPERIMENT / "empty_jember.tsv"
TARGET_APPROVED = 300
RANDOM_SEED = 42


def run(script: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-u", str(PROCESSING / script), *arguments], check=True, cwd=REPO_ROOT)


def retained_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row.get("filtered_out", "False").casefold() != "true"
        ]


def waveform_envelope(audio_path: Path, points: int = 700) -> list[list[float]]:
    import librosa
    import numpy as np

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    if not len(audio):
        return []
    boundaries = np.linspace(0, len(audio), min(points, len(audio)) + 1, dtype=int)
    return [
        [round(float(audio[start:end].min()), 4), round(float(audio[start:end].max()), 4)]
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def write_waveforms(output_dir: Path, vad_csv: Path, normalized_csv: Path) -> None:
    unnormalized = retained_rows(vad_csv)
    normalized = {
        Path(row["audio_path"]).name: Path(row["audio_path"])
        for row in retained_rows(normalized_csv)
    }
    payload: dict[str, dict[str, list[list[float]]]] = {}
    for index, row in enumerate(unnormalized, start=1):
        unnormalized_path = Path(row["audio_path"])
        name = unnormalized_path.name
        if name not in normalized:
            raise SystemExit(f"Normalized copy is missing for {name}")
        payload[name] = {
            "unnormalized": waveform_envelope(unnormalized_path),
            "normalized": waveform_envelope(normalized[name]),
        }
        print(f"[waveform {index}/{len(unnormalized)}] {name}")
    (output_dir / "waveforms.json").write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def write_dataset_json(output_dir: Path, clip_count: int) -> None:
    payload = {
        "title": f"Audio normalization impact — {clip_count} Indonesian development clips",
        "clip_count": clip_count,
        "paths": {
            "vad_manifest": "02_vad.csv",
            "review_audio_dir": "02_vad/",
            "vad_audio_dir": "02_vad/",
            "normalized_audio_dir": "03_normalized/",
            "wav2vec2_unnormalized": "04_wav2vec2_unnormalized.csv",
            "wav2vec2_normalized": "04_wav2vec2_normalized.csv",
            "waveforms": "waveforms.json",
        },
    }
    (output_dir / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-approved", type=int, default=TARGET_APPROVED)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-prepare", action="store_true", help="Skip stages 1–3 when outputs already exist.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip Wav2Vec2 evaluation on both audio copies.")
    parser.add_argument("--force-inference", action="store_true", help="Overwrite evaluation CSVs if they already exist.")
    parser.add_argument("--device", default="auto", help="Device for Wav2Vec2: auto, cpu, cuda, or mps.")
    args = parser.parse_args()
    if args.target_approved < 1:
        parser.error("--target-approved must be at least 1")

    output_dir = args.output_dir.resolve()
    segments_dir = output_dir / "01_segments"
    segments_csv = output_dir / "01_segments.csv"
    vad_dir = output_dir / "02_vad"
    vad_csv = output_dir / "02_vad.csv"
    normalized_dir = output_dir / "03_normalized"
    normalized_csv = output_dir / "03_normalized.csv"
    unnormalized_csv = output_dir / "04_wav2vec2_unnormalized.csv"
    normalized_eval_csv = output_dir / "04_wav2vec2_normalized.csv"

    if not args.skip_prepare:
        if not EMPTY_JEMBER.is_file():
            raise SystemExit(f"Missing empty Jember stub: {EMPTY_JEMBER}")
        run(
            "01_prepare_indonesian_segments.py",
            [
                "--jember-manifest", str(EMPTY_JEMBER),
                "--output-dir", str(segments_dir),
                "--manifest", str(segments_csv),
            ],
        )
        run(
            "02_vad_trim.py",
            [
                "--manifest", str(segments_csv),
                "--output-dir", str(vad_dir),
                "--output-manifest", str(vad_csv),
                "--target-approved", str(args.target_approved),
                "--random-seed", str(args.random_seed),
            ],
        )
        run(
            "03_normalize_audio.py",
            [
                "--manifest", str(vad_csv),
                "--output-dir", str(normalized_dir),
                "--output-csv", str(normalized_csv),
            ],
        )
    for required in (vad_csv, normalized_csv):
        if not required.is_file():
            raise SystemExit(f"Required prepared manifest is missing: {required}")
    clip_count = len(retained_rows(vad_csv))
    if clip_count != args.target_approved:
        raise SystemExit(
            f"Expected {args.target_approved} VAD-approved clips, found {clip_count} in {vad_csv}"
        )
    if len(retained_rows(normalized_csv)) != clip_count:
        raise SystemExit("The normalized and unnormalized manifests contain different clip counts")
    write_dataset_json(output_dir, clip_count)
    waveforms = output_dir / "waveforms.json"
    if not waveforms.is_file() or not args.skip_prepare:
        write_waveforms(output_dir, vad_csv, normalized_csv)

    if not args.skip_inference:
        evaluations = (
            (vad_csv, unnormalized_csv),
            (normalized_csv, normalized_eval_csv),
        )
        for manifest, evaluation_csv in evaluations:
            if evaluation_csv.is_file() and not args.force_inference:
                print(f"Reusing existing evaluation: {evaluation_csv}")
                continue
            run(
                "04_wav2vec2_evaluation.py",
                [
                    "--manifest", str(manifest),
                    "--output-csv", str(evaluation_csv),
                    "--device", args.device,
                ],
            )

    print(f"Experiment outputs: {output_dir}")
    print("Open the viewer with a local server from the repository root:")
    print(
        "  python -m http.server 8000\n"
        "  http://localhost:8000/experiments/audio_normalization_impact/comparison.html"
        f"?config=../../processed_indonesia/{output_dir.name}/dataset.json"
    )


if __name__ == "__main__":
    main()
