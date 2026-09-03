#!/usr/bin/env python3
"""Run Indonesian dataset preparation, VAD filtering, and audio normalization."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_config import apply_defaults, load_section


def run(script: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(Path(__file__).with_name(script)), *arguments], check=True)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    config_args, _ = config_parser.parse_known_args()
    config_values, _ = load_section(config_args.config, "pipeline")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    parser.add_argument("--jember-manifest", type=Path, default=Path("indonesian_data/Jember Javanese Spontaneous Speech Corpus/Jember Javanese Spontaneous Speech Corpus - 1-200.tsv"))
    parser.add_argument("--jember-audio-dir", type=Path, default=Path("indonesian_data/Jember Javanese Spontaneous Speech Corpus/mp3 audio"))
    parser.add_argument("--development-manifest", type=Path, default=Path("indonesian_data/indonesian_dev/metadata.tsv"))
    parser.add_argument("--development-audio-dir", type=Path, default=Path("indonesian_data/indonesian_dev/clips"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia"))
    parser.add_argument("--max-recordings", type=int, help="Process only the first N recordings (for previews).")
    parser.add_argument("--max-segments-per-recording", type=int, help="Keep only the first N segments from each recording.")
    parser.add_argument("--max-development-clips", type=int, help="Keep only the first N development clips (for previews).")
    parser.add_argument("--preview-per-dataset", type=int, help="Prepare this many clips from each dataset in stage 1.")
    parser.add_argument("--jember-clips", type=int, help="Limit overlapping Jember TSV-row windows.")
    parser.add_argument("--development-clips", type=int, help="Limit pre-segmented development clips.")
    parser.add_argument("--min-clip-seconds", type=float, default=3.0)
    parser.add_argument("--max-clip-seconds", type=float, default=40.0)
    parser.add_argument("--max-clips", type=int, help="Stop after this many valid clips across both datasets.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-words", type=int, default=2, help="Reject transcripts with fewer words.")
    parser.add_argument("--min-seconds", type=float, default=3.0, help="Reject clips shorter than this duration.")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--skip-inference", action="store_true", help="Skip step 4 Whisper language inference.")
    parser.add_argument("--inference-model", default="medium", help="Whisper model name for step 4.")
    parser.add_argument("--inference-device", default="auto", help="Device for step 4: auto, cpu, cuda, or mps.")
    parser.add_argument("--inference-limit", type=int, help="Limit the number of clips evaluated in step 4.")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()
    segment_manifest = args.output_dir / "01_segments.csv"
    vad_manifest = args.output_dir / "02_vad.csv"
    split_args = ["--jember-manifest", str(args.jember_manifest), "--jember-audio-dir", str(args.jember_audio_dir), "--development-manifest", str(args.development_manifest), "--development-audio-dir", str(args.development_audio_dir), "--output-dir", str(args.output_dir / "01_segments"), "--manifest", str(segment_manifest), "--ffmpeg", args.ffmpeg, "--ffprobe", args.ffprobe, "--min-clip-seconds", str(args.min_clip_seconds), "--max-clip-seconds", str(args.max_clip_seconds), "--min-words", str(args.min_words)]
    if args.max_recordings is not None:
        split_args += ["--max-recordings", str(args.max_recordings)]
    if args.max_segments_per_recording is not None:
        split_args += ["--max-segments-per-recording", str(args.max_segments_per_recording)]
    if args.max_development_clips is not None:
        split_args += ["--max-development-clips", str(args.max_development_clips)]
    if args.preview_per_dataset is not None:
        split_args += ["--preview-per-dataset", str(args.preview_per_dataset)]
    if args.jember_clips is not None:
        split_args += ["--jember-clips", str(args.jember_clips)]
    if args.development_clips is not None:
        split_args += ["--development-clips", str(args.development_clips)]
    if args.max_clips is not None:
        split_args += ["--max-clips", str(args.max_clips)]
    run("01_prepare_indonesian_segments.py", split_args)
    run("02_vad_trim.py", ["--manifest", str(segment_manifest), "--output-dir", str(args.output_dir / "02_vad"), "--output-manifest", str(vad_manifest), "--threshold", str(args.threshold), "--min-words", str(args.min_words), "--min-seconds", str(args.min_seconds), "--ffmpeg", args.ffmpeg])
    run("03_normalize_audio.py", ["--manifest", str(vad_manifest), "--output-dir", str(args.output_dir / "audio"), "--output-csv", str(args.output_dir / "train.csv"), "--ffmpeg", args.ffmpeg])
    if not args.skip_inference:
        inference_args = ["--manifest", str(vad_manifest), "--output-csv", str(args.output_dir / "04_whisper_evaluation.csv"), "--model", args.inference_model, "--device", args.inference_device]
        if args.inference_limit is not None:
            inference_args += ["--limit", str(args.inference_limit)]
        run("04_whisper_evaluation.py", inference_args)
    (args.output_dir / "dataset.json").write_text(json.dumps({
        "title": f"Indonesian processing output — {args.output_dir.name}",
        "paths": {"vad_manifest": "02_vad.csv", "review_audio_dir": "01_segments/", "vad_audio_dir": "02_vad/", "whisper_evaluation": "04_whisper_evaluation.csv"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
