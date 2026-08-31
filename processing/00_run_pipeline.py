#!/usr/bin/env python3
"""Run CHAT segmentation, Silero VAD trimming, and audio normalization in sequence."""

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
    parser.add_argument("--audio-dir", type=Path, default=Path("miami/audios"))
    parser.add_argument("--chat-dir", type=Path, default=Path("miami/chat"))
    parser.add_argument("--tsv-dir", type=Path, default=Path("miami/word_level_tsvs"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed"))
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-recordings", type=int, help="Process only the first N recordings (for previews).")
    parser.add_argument("--max-segments-per-recording", type=int, help="Keep only the first N segments from each recording.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--merge-gap-ms", type=int, default=20)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--skip-inference", action="store_true", help="Skip step 4 Whisper language inference.")
    parser.add_argument("--inference-model", default="medium", help="Whisper model name for step 4.")
    parser.add_argument("--inference-device", default="auto", help="Device for step 4: auto, cpu, cuda, or mps.")
    parser.add_argument("--inference-limit", type=int, help="Limit the number of clips evaluated in step 4.")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()
    segment_manifest = args.output_dir / "01_segments.csv"
    vad_manifest = args.output_dir / "02_vad.csv"
    split_args = ["--audio-dir", str(args.audio_dir), "--chat-dir", str(args.chat_dir), "--tsv-dir", str(args.tsv_dir), "--output-dir", str(args.output_dir / "01_segments"), "--manifest", str(segment_manifest), "--max-seconds", str(args.max_seconds), "--ffmpeg", args.ffmpeg]
    if args.max_recordings is not None:
        split_args += ["--max-recordings", str(args.max_recordings)]
    if args.max_segments_per_recording is not None:
        split_args += ["--max-segments-per-recording", str(args.max_segments_per_recording)]
    run("01_split_chat_segments.py", split_args)
    run("02_vad_trim.py", ["--manifest", str(segment_manifest), "--output-dir", str(args.output_dir / "02_vad"), "--output-manifest", str(vad_manifest), "--threshold", str(args.threshold), "--merge-gap-ms", str(args.merge_gap_ms), "--ffmpeg", args.ffmpeg])
    run("03_normalize_audio.py", ["--manifest", str(vad_manifest), "--output-dir", str(args.output_dir / "audio"), "--output-csv", str(args.output_dir / "train.csv"), "--ffmpeg", args.ffmpeg])
    if not args.skip_inference:
        inference_args = ["--manifest", str(vad_manifest), "--output-csv", str(args.output_dir / "04_whisper_evaluation.csv"), "--model", args.inference_model, "--device", args.inference_device]
        if args.inference_limit is not None:
            inference_args += ["--limit", str(args.inference_limit)]
        run("04_whisper_evaluation.py", inference_args)
    (args.output_dir / "dataset.json").write_text(json.dumps({
        "title": f"Miami processing output — {args.output_dir.name}",
        "paths": {"vad_manifest": "02_vad.csv", "review_audio_dir": "01_segments/", "vad_audio_dir": "02_vad/", "whisper_evaluation": "04_whisper_evaluation.csv"},
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
