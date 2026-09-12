#!/usr/bin/env python3
"""Split >30s Indonesian-dev clips at silence and align their transcript boundary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from urllib.parse import quote

import librosa
import numpy as np
import soundfile as sf


MODEL_ID = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
SAMPLE_RATE = 16_000
MAX_CLIP_SECONDS = 30.0
FIELDS = [
    "original_audio", "speaker", "duration_seconds", "split_seconds",
    "silence_duration_seconds", "split_method", "boundary_word_index",
    "average_distance", "joint_score", "clip_1_audio", "clip_1_transcript",
    "clip_1_prediction", "clip_1_boundary_phrase", "clip_1_distance",
    "clip_2_audio", "clip_2_transcript", "clip_2_prediction",
    "clip_2_boundary_phrase", "clip_2_distance", "candidate_scores_json",
]


def normalize_word(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z]", "", value)


def normalize_phrase(words: list[str]) -> str:
    return " ".join(filter(None, (normalize_word(word) for word in words)))


def normalized_levenshtein(first: str, second: str) -> float:
    previous = list(range(len(second) + 1))
    for row, first_char in enumerate(first, start=1):
        current = [row]
        for column, second_char in enumerate(second, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (first_char != second_char),
            ))
        previous = current
    return previous[-1] / max(len(first), len(second), 1)


def load_audio(ffmpeg: str, source: Path) -> np.ndarray:
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(source), "-f", "f32le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def find_split(
    audio: np.ndarray,
    *,
    top_db: float,
    minimum_silence_seconds: float,
    minimum_clip_seconds: float,
) -> tuple[int, float, str]:
    """Return a valid split sample at the midpoint of the longest silence."""
    duration_seconds = len(audio) / SAMPLE_RATE
    lower_seconds = max(minimum_clip_seconds, duration_seconds - MAX_CLIP_SECONDS)
    upper_seconds = min(MAX_CLIP_SECONDS - 0.05, duration_seconds - minimum_clip_seconds)
    if lower_seconds >= upper_seconds:
        raise ValueError(
            f"No split can make both halves valid for {duration_seconds:.3f}s audio"
        )
    lower = round(lower_seconds * SAMPLE_RATE)
    upper = round(upper_seconds * SAMPLE_RATE)
    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=2048,
        hop_length=256,
    )
    silence_ranges: list[tuple[int, int]] = []
    previous_end = 0
    for start, end in intervals:
        if start > previous_end:
            silence_ranges.append((previous_end, int(start)))
        previous_end = int(end)
    if previous_end < len(audio):
        silence_ranges.append((previous_end, len(audio)))

    eligible = []
    minimum_silence_samples = round(minimum_silence_seconds * SAMPLE_RATE)
    for start, end in silence_ranges:
        clipped_start, clipped_end = max(start, lower), min(end, upper)
        if clipped_end - clipped_start >= minimum_silence_samples:
            eligible.append((end - start, clipped_start, clipped_end))
    if eligible:
        full_width, start, end = max(eligible)
        return (start + end) // 2, full_width / SAMPLE_RATE, "largest_silence"

    # If no interval meets the requested silence length, use the quietest
    # 200ms window so every long clip can still be inspected in the experiment.
    window = max(1, round(minimum_silence_seconds * SAMPLE_RATE))
    hop = max(1, window // 4)
    best_start, best_rms = lower, float("inf")
    for start in range(lower, max(lower + 1, upper - window + 1), hop):
        region = audio[start:start + window]
        rms = float(np.sqrt(np.mean(np.square(region), dtype=np.float64)))
        if rms < best_rms:
            best_start, best_rms = start, rms
    return min(best_start + window // 2, upper), window / SAMPLE_RATE, "quietest_window_fallback"


def transcribe(audio: np.ndarray, processor, model, device: str) -> str:
    import torch

    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    model_inputs = {"input_values": inputs.input_values.to(device)}
    if getattr(inputs, "attention_mask", None) is not None:
        model_inputs["attention_mask"] = inputs.attention_mask.to(device)
    with torch.inference_mode():
        logits = model(**model_inputs).logits
    return processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()


def choose_boundary(
    transcript: str,
    first_prediction: str,
    second_prediction: str,
    match_words: int,
    expected_boundary: float,
) -> tuple[dict, list[dict]]:
    words = transcript.split()
    first_words = first_prediction.split()[-match_words:]
    second_words = second_prediction.split()[:match_words]
    predicted_left = normalize_phrase(first_words)
    predicted_right = normalize_phrase(second_words)
    if not predicted_left or not predicted_right:
        raise ValueError("Wav2Vec2 returned an empty boundary prediction")
    candidates = []
    for boundary in range(match_words, len(words) - match_words + 1):
        actual_left_words = words[boundary - match_words:boundary]
        actual_right_words = words[boundary:boundary + match_words]
        actual_left = normalize_phrase(actual_left_words)
        actual_right = normalize_phrase(actual_right_words)
        if not actual_left or not actual_right:
            continue
        left_distance = normalized_levenshtein(predicted_left, actual_left)
        right_distance = normalized_levenshtein(predicted_right, actual_right)
        average_distance = (left_distance + right_distance) / 2
        candidates.append({
            "boundary_word_index": boundary,
            "clip_1_phrase": " ".join(actual_left_words),
            "clip_2_phrase": " ".join(actual_right_words),
            "clip_1_distance": left_distance,
            "clip_2_distance": right_distance,
            "average_distance": average_distance,
            "joint_score": 1 - average_distance,
            "distance_from_time_estimate": abs(boundary - expected_boundary),
        })
    if not candidates:
        raise ValueError("Transcript has too few usable words for boundary matching")
    candidates.sort(key=lambda item: (
        item["average_distance"], item["distance_from_time_estimate"]
    ))
    return candidates[0], candidates


def relative_to_csv(path: Path, csv_path: Path) -> str:
    return os.path.relpath(path.resolve(), csv_path.parent.resolve()).replace(os.sep, "/")


def viewer_url(csv_path: Path) -> str | None:
    repository = Path(__file__).resolve().parents[2]
    viewer = Path(__file__).with_name("review.html")
    try:
        viewer_relative = viewer.resolve().relative_to(repository).as_posix()
        csv_relative = csv_path.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    return (
        f"http://127.0.0.1:8000/{viewer_relative}"
        f"?data={quote('/' + csv_relative, safe='/')}"
    )


def main() -> None:
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("indonesian_data/indonesian_dev/metadata.tsv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("indonesian_data/indonesian_dev/clips"))
    parser.add_argument("--output-dir", type=Path, default=experiment_dir / "output")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--top-db", type=float, default=35.0)
    parser.add_argument("--minimum-silence-seconds", type=float, default=0.2)
    parser.add_argument("--minimum-clip-seconds", type=float, default=3.0)
    parser.add_argument("--match-words", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Process only the first N long clips.")
    args = parser.parse_args()
    if not args.metadata.is_file() or not args.audio_dir.is_dir():
        parser.error("The metadata or development audio directory does not exist")
    if shutil.which(args.ffmpeg) is None:
        parser.error(f"ffmpeg is not available: {args.ffmpeg}")
    if args.match_words < 1 or args.minimum_silence_seconds <= 0:
        parser.error("Match words and silence duration must be positive")

    try:
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2Processor
    except ImportError as error:
        raise SystemExit("Run with: uv run --project processing ...") from error
    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is unavailable")

    with args.metadata.open(encoding="utf-8", newline="") as file:
        metadata_rows = list(csv.DictReader(file, delimiter="\t"))
    selected = []
    for row in metadata_rows:
        source = args.audio_dir / row["audio_filename"]
        if not source.is_file():
            continue
        audio = load_audio(args.ffmpeg, source)
        if len(audio) / SAMPLE_RATE > MAX_CLIP_SECONDS:
            selected.append((row, source, audio))
            if args.limit is not None and len(selected) >= args.limit:
                break
    print(f"Found {len(selected)} development clips over 30 seconds")
    if not selected:
        raise SystemExit("No clips were selected")

    print(f"Loading {args.model} on {device}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device).eval()
    audio_output = args.output_dir / "audio"
    audio_output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "split_alignment_results.csv"
    results = []
    for index, (metadata, source, audio) in enumerate(selected, start=1):
        split_sample, silence_duration, split_method = find_split(
            audio,
            top_db=args.top_db,
            minimum_silence_seconds=args.minimum_silence_seconds,
            minimum_clip_seconds=args.minimum_clip_seconds,
        )
        first_audio, second_audio = audio[:split_sample], audio[split_sample:]
        first_prediction = transcribe(first_audio, processor, model, device)
        second_prediction = transcribe(second_audio, processor, model, device)
        words = metadata["transcript"].split()
        expected_boundary = len(words) * split_sample / len(audio)
        selected_boundary, candidates = choose_boundary(
            metadata["transcript"], first_prediction, second_prediction,
            args.match_words, expected_boundary,
        )
        boundary = int(selected_boundary["boundary_word_index"])
        first_path = audio_output / f"{source.stem}__part_1.wav"
        second_path = audio_output / f"{source.stem}__part_2.wav"
        sf.write(first_path, first_audio, SAMPLE_RATE, subtype="PCM_16")
        sf.write(second_path, second_audio, SAMPLE_RATE, subtype="PCM_16")
        results.append({
            "original_audio": str(source),
            "speaker": metadata.get("speaker", ""),
            "duration_seconds": f"{len(audio) / SAMPLE_RATE:.4f}",
            "split_seconds": f"{split_sample / SAMPLE_RATE:.4f}",
            "silence_duration_seconds": f"{silence_duration:.4f}",
            "split_method": split_method,
            "boundary_word_index": str(boundary),
            "average_distance": f"{selected_boundary['average_distance']:.6f}",
            "joint_score": f"{selected_boundary['joint_score']:.6f}",
            "clip_1_audio": relative_to_csv(first_path, csv_path),
            "clip_1_transcript": " ".join(words[:boundary]),
            "clip_1_prediction": first_prediction,
            "clip_1_boundary_phrase": selected_boundary["clip_1_phrase"],
            "clip_1_distance": f"{selected_boundary['clip_1_distance']:.6f}",
            "clip_2_audio": relative_to_csv(second_path, csv_path),
            "clip_2_transcript": " ".join(words[boundary:]),
            "clip_2_prediction": second_prediction,
            "clip_2_boundary_phrase": selected_boundary["clip_2_phrase"],
            "clip_2_distance": f"{selected_boundary['clip_2_distance']:.6f}",
            "candidate_scores_json": json.dumps(candidates, ensure_ascii=False),
        })
        print(
            f"[{index}/{len(selected)}] {source.name}: split="
            f"{split_sample / SAMPLE_RATE:.2f}s, boundary={boundary}/{len(words)}, "
            f"joint={selected_boundary['joint_score']:.3f}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    diagnostics_path = args.output_dir / "split_alignment_results.json"
    diagnostics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(results)} aligned split pairs to {csv_path}")
    print(f"Wrote full alignment diagnostics to {diagnostics_path}")
    link = viewer_url(csv_path)
    if link:
        print(f"View split alignment experiment: {link}")


if __name__ == "__main__":
    main()
