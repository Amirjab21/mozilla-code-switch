#!/usr/bin/env python3
"""Run segmentation, start correction, noise augmentation, and ASR training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_config import apply_defaults, load_section


START_MATCH_MODEL = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"


def run(script: str, arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name(script)), *arguments],
        check=True,
    )


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, help="YAML named-run configuration file."
    )
    config_args, _ = config_parser.parse_known_args()
    config_values, _ = load_section(config_args.config, "pipeline")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="YAML named-run configuration file."
    )
    parser.add_argument(
        "--jember-manifest",
        type=Path,
        default=Path(
            "indonesian_data/Jember Javanese Spontaneous Speech Corpus/"
            "Jember Javanese Spontaneous Speech Corpus - 1-200.tsv"
        ),
    )
    parser.add_argument(
        "--jember-audio-dir",
        type=Path,
        default=Path(
            "indonesian_data/Jember Javanese Spontaneous Speech Corpus/mp3 audio"
        ),
    )
    parser.add_argument(
        "--development-manifest",
        type=Path,
        default=Path("indonesian_data/indonesian_dev/metadata.tsv"),
    )
    parser.add_argument(
        "--development-audio-dir",
        type=Path,
        default=Path("indonesian_data/indonesian_dev/clips"),
    )
    parser.add_argument("--development-corrected-long-clips", type=Path)
    parser.add_argument("--development-corrected-long-audio-dir", type=Path)
    parser.add_argument("--commonvoice-indonesian-manifest", type=Path, default=Path("indonesian_data/cv_indonesian/id/test.tsv"))
    parser.add_argument("--commonvoice-indonesian-audio-dir", type=Path, default=Path("indonesian_data/cv_indonesian/id/clips"))
    parser.add_argument("--commonvoice-javanese-manifest", type=Path, default=Path("indonesian_data/cv_javanese/ss-corpus-jv.tsv"))
    parser.add_argument("--commonvoice-javanese-audio-dir", type=Path, default=Path("indonesian_data/cv_javanese/audios"))
    parser.add_argument("--commonvoice-code-switch-samples", type=int)
    parser.add_argument("--commonvoice-clips-per-dataset", type=int)
    parser.add_argument("--commonvoice-seed", type=int, default=1337)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("processed_indonesia")
    )

    # Stage 1: segmentation and 16 kHz conversion.
    parser.add_argument(
        "--max-recordings", type=int,
        help="Process only the first N Jember recordings.",
    )
    parser.add_argument(
        "--max-segments-per-recording", type=int,
        help="Read only the first N TSV rows per Jember recording.",
    )
    parser.add_argument(
        "--max-development-clips", type=int,
        help="Deprecated alias for --development-clips.",
    )
    parser.add_argument(
        "--preview-per-dataset", type=int,
        help="Prepare this many clips from each dataset.",
    )
    parser.add_argument(
        "--jember-clips", type=int,
        help="Limit overlapping Jember TSV-row windows.",
    )
    parser.add_argument(
        "--development-clips", type=int,
        help="Limit pre-segmented Indonesian development clips.",
    )
    parser.add_argument("--min-clip-seconds", type=float, default=3.0)
    parser.add_argument("--max-clip-seconds", type=float, default=40.0)
    parser.add_argument("--jember-min-clip-seconds", type=float)
    parser.add_argument("--jember-max-clip-seconds", type=float)
    parser.add_argument("--development-min-clip-seconds", type=float)
    parser.add_argument("--development-max-clip-seconds", type=float)
    parser.add_argument(
        "--max-clips", type=int, help="Deprecated final combined cap."
    )
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument(
        "--jember-stride-overlap-rows",
        type=int,
        default=2,
        help="Jember rows retained between consecutive expanding blocks.",
    )
    parser.add_argument(
        "--prepare-workers", type=int, default=1,
        help="Number of Jember recordings prepared concurrently in stage 1.",
    )
    parser.add_argument("--reuse-existing", action="store_true")

    # Stage 2: Wav2Vec2 matching of Jember transcript starts.
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--word-search-amount", type=int, default=13)
    parser.add_argument("--match-words", type=int, default=2)
    parser.add_argument(
        "--closeness-metric", choices=("levenshtein",),
        default="levenshtein",
    )
    parser.add_argument("--start-match-model", default=START_MATCH_MODEL)
    parser.add_argument(
        "--start-match-device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument(
        "--start-match-limit", type=int,
        help="Process only the first N Jember clips in stage 2.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")

    # Stage 3: additive room-noise augmentation.
    parser.add_argument(
        "--noise-dir", type=Path, default=Path("indonesian_data/room_noises")
    )
    parser.add_argument("--noise-copies-per-clip", type=int, default=1)
    parser.add_argument(
        "--noise-augmentation-fraction",
        type=float,
        default=1.0,
        help="Fraction of stage-2 rows that receive additive-noise copies.",
    )
    parser.add_argument(
        "--jember-augmentation-multiplier",
        type=float,
        help="Additional stage-3 noisy copies per Jember original.",
    )
    parser.add_argument(
        "--development-augmentation-multiplier",
        type=float,
        help="Additional stage-3 noisy copies per Indonesian-dev original.",
    )
    parser.add_argument("--commonvoice-augmentation-multiplier", type=float)
    parser.add_argument("--min-noise-fraction", type=float, default=0.25)
    parser.add_argument("--max-noise-fraction", type=float, default=0.75)
    parser.add_argument("--min-snr-db", type=float, default=5.0)
    parser.add_argument("--max-snr-db", type=float, default=20.0)
    parser.add_argument("--augmentation-seed", type=int, default=1337)
    parser.add_argument(
        "--augment-workers", type=int, default=1,
        help="Number of concurrent audio workers used by stage 3.",
    )

    parser.add_argument(
        "--training-output-dir", type=Path,
        help=(
            "Stage-5 output directory when no `train.output_dir` is supplied "
            "by --config; defaults beneath --output-dir."
        ),
    )
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()

    train_values: dict[str, object] = {}
    if args.config is not None:
        train_values, _ = load_section(args.config, "train")

    # OpenAI Whisper invokes an executable literally named `ffmpeg`. When the
    # CUDA/SageMaker caller supplies an explicit binary (for example, an
    # isolated Conda installation), expose it to child stages through PATH.
    # CPU and MPS runs retain their existing local environment unchanged.
    ffmpeg_path = Path(args.ffmpeg).expanduser()
    cuda_requested = (
        args.start_match_device == "cuda"
        or train_values.get("device") == "cuda"
    )
    if cuda_requested and ffmpeg_path.parent != Path("."):
        ffmpeg_directory = str(ffmpeg_path.resolve().parent)
        os.environ["PATH"] = ffmpeg_directory + os.pathsep + os.environ.get("PATH", "")

    configured_training_output: Path | None = None
    if train_values.get("output_dir") is not None:
        configured_training_output = Path(train_values["output_dir"])
    training_output_dir = (
        args.training_output_dir
        or configured_training_output
        or args.output_dir / "05_train"
    )

    segment_manifest = args.output_dir / "01_segments.csv"
    processed_manifest = args.output_dir / "02_processed.csv"
    augmented_manifest = args.output_dir / "03_dataset_with_augmented.csv"
    augmented_audio_dir = args.output_dir / "03_augmented_audio"
    perturbed_manifest = args.output_dir / "03_1_dataset_with_perturbations.csv"
    perturbed_audio_dir = args.output_dir / "03_1_perturbed_audio"
    perturbation_comparison = args.output_dir / "03_1_perturbation_comparison.csv"
    match_review = args.output_dir / "clip_start_matches.json"

    stage_1_args = [
        "--jember-manifest", str(args.jember_manifest),
        "--jember-audio-dir", str(args.jember_audio_dir),
        "--development-manifest", str(args.development_manifest),
        "--development-audio-dir", str(args.development_audio_dir),
        "--commonvoice-indonesian-manifest", str(args.commonvoice_indonesian_manifest),
        "--commonvoice-indonesian-audio-dir", str(args.commonvoice_indonesian_audio_dir),
        "--commonvoice-javanese-manifest", str(args.commonvoice_javanese_manifest),
        "--commonvoice-javanese-audio-dir", str(args.commonvoice_javanese_audio_dir),
        "--commonvoice-seed", str(args.commonvoice_seed),
        "--output-dir", str(args.output_dir / "01_segments"),
        "--manifest", str(segment_manifest),
        "--ffmpeg", args.ffmpeg,
        "--ffprobe", args.ffprobe,
        "--min-clip-seconds", str(args.min_clip_seconds),
        "--max-clip-seconds", str(args.max_clip_seconds),
        "--min-words", str(args.min_words),
        "--jember-stride-overlap-rows", str(args.jember_stride_overlap_rows),
        "--prepare-workers", str(args.prepare_workers),
    ]
    optional_stage_1_args = (
        (
            "--development-corrected-long-clips",
            args.development_corrected_long_clips,
        ),
        (
            "--development-corrected-long-audio-dir",
            args.development_corrected_long_audio_dir,
        ),
        ("--jember-min-clip-seconds", args.jember_min_clip_seconds),
        ("--jember-max-clip-seconds", args.jember_max_clip_seconds),
        ("--development-min-clip-seconds", args.development_min_clip_seconds),
        ("--development-max-clip-seconds", args.development_max_clip_seconds),
        ("--commonvoice-code-switch-samples", args.commonvoice_code_switch_samples),
        ("--commonvoice-clips-per-dataset", args.commonvoice_clips_per_dataset),
        ("--max-recordings", args.max_recordings),
        ("--max-segments-per-recording", args.max_segments_per_recording),
        ("--max-development-clips", args.max_development_clips),
        ("--preview-per-dataset", args.preview_per_dataset),
        ("--jember-clips", args.jember_clips),
        ("--development-clips", args.development_clips),
        ("--max-clips", args.max_clips),
    )
    for option, value in optional_stage_1_args:
        if value is not None:
            stage_1_args += [option, str(value)]
    if args.reuse_existing:
        stage_1_args.append("--reuse-existing")
    run("01_prepare_indonesian_segments.py", stage_1_args)

    stage_2_args = [
        "--clips", str(segment_manifest),
        "--tsv", str(args.jember_manifest),
        "--output", str(match_review),
        "--output-csv", str(processed_manifest),
        "--audio-seconds", str(args.audio_seconds),
        "--word-search-amount", str(args.word_search_amount),
        "--match-words", str(args.match_words),
        "--closeness-metric", args.closeness_metric,
        "--model", args.start_match_model,
        "--device", args.start_match_device,
        "--ffmpeg", args.ffmpeg,
    ]
    if args.start_match_limit is not None:
        stage_2_args += ["--limit", str(args.start_match_limit)]
    run("02_match_predicted_clip_start.py", stage_2_args)

    stage_3_args = [
        "--input", str(processed_manifest),
        "--output", str(augmented_manifest),
        "--output-dir", str(augmented_audio_dir),
        "--noise-dir", str(args.noise_dir),
        "--copies-per-clip", str(args.noise_copies_per_clip),
        "--augmentation-fraction", str(args.noise_augmentation_fraction),
        "--min-noise-fraction", str(args.min_noise_fraction),
        "--max-noise-fraction", str(args.max_noise_fraction),
        "--min-snr-db", str(args.min_snr_db),
        "--max-snr-db", str(args.max_snr_db),
        "--seed", str(args.augmentation_seed),
        "--workers", str(args.augment_workers),
    ]
    if args.jember_augmentation_multiplier is not None:
        stage_3_args += [
            "--jember-augmentation-multiplier",
            str(args.jember_augmentation_multiplier),
        ]
    if args.development_augmentation_multiplier is not None:
        stage_3_args += [
            "--development-augmentation-multiplier",
            str(args.development_augmentation_multiplier),
        ]
    if args.commonvoice_augmentation_multiplier is not None:
        stage_3_args += ["--commonvoice-augmentation-multiplier", str(args.commonvoice_augmentation_multiplier)]
    run("03_augment_audio_with_noise.py", stage_3_args)

    stage_3_1_args = [
        "--input", str(augmented_manifest),
        "--output", str(perturbed_manifest),
        "--output-dir", str(perturbed_audio_dir),
        "--comparison-output", str(perturbation_comparison),
        "--rir-dir", str(args.noise_dir),
    ]
    if args.config is not None:
        stage_3_1_args += ["--config", str(args.config)]
    run("03_1_speed_pitch_volume_perturbations.py", stage_3_1_args)

    # Stage 5: model-specific LoRA fine-tuning. The selected trainer loads the
    # train section directly; the stage-3.1 manifest is always wired explicitly
    # so processing and training cannot accidentally diverge.
    training_backend = str(train_values.get("backend", "whisper"))
    training_scripts = {
        "whisper": "05_train.py",
        "qwen3_asr": "05_train_qwen3_asr.py",
    }
    if training_backend not in training_scripts:
        raise ValueError(
            f"Unsupported train.backend {training_backend!r}; choose one of "
            f"{', '.join(sorted(training_scripts))}"
        )
    stage_5_args = ["--manifest", str(perturbed_manifest)]
    if args.config is not None:
        stage_5_args += ["--config", str(args.config)]
    if args.training_output_dir is not None:
        stage_5_args += ["--output-dir", str(args.training_output_dir)]
    elif args.config is None:
        stage_5_args += ["--output-dir", str(training_output_dir)]
    run(training_scripts[training_backend], stage_5_args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset.json").write_text(json.dumps({
        "title": f"Indonesian processing output — {args.output_dir.name}",
        "paths": {
            "segments_manifest": "01_segments.csv",
            "segments_audio_dir": "01_segments/",
            "processed_manifest": "02_processed.csv",
            "clip_start_matches": "clip_start_matches.json",
            "augmented_manifest": "03_dataset_with_augmented.csv",
            "augmented_audio_dir": "03_augmented_audio/",
            "perturbed_manifest": "03_1_dataset_with_perturbations.csv",
            "perturbed_audio_dir": "03_1_perturbed_audio/",
            "perturbation_comparison": "03_1_perturbation_comparison.csv",
            "training_output_dir": str(training_output_dir),
        },
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
