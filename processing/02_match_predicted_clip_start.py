#!/usr/bin/env python3
"""Correct Jember clip-start transcripts and write the stage-2 manifest."""

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
from typing import Callable
from urllib.parse import quote

from run_config import apply_defaults, load_section
from text_normalisation import normalize_text


MODEL_ID = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
Metric = Callable[[str, str], tuple[float, dict[str, float | int]]]


def normalize_word(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z]", "", value)


def levenshtein_metric(predicted: str, actual: str) -> tuple[float, dict[str, float | int]]:
    """Return normalized Levenshtein distance; lower is closer."""
    previous = list(range(len(actual) + 1))
    for row, predicted_char in enumerate(predicted, start=1):
        current = [row]
        for column, actual_char in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (predicted_char != actual_char),
            ))
        previous = current
    edits = previous[-1]
    normalized = edits / max(len(predicted), len(actual), 1)
    return normalized, {"edit_distance": edits, "normalized_distance": normalized}


# Add new plug-and-play metrics here. Each function returns (distance, details);
# lower distance must mean a closer match.
METRICS: dict[str, Metric] = {
    "levenshtein": levenshtein_metric,
}


def viewer_url(viewer: str, parameter: str, artifact: Path) -> str | None:
    """Build a localhost viewer URL when the artifact is inside the repo."""
    repository = Path(__file__).resolve().parent.parent
    try:
        relative = artifact.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    value = quote(f"../{relative}", safe="/")
    return f"http://127.0.0.1:8000/analysis_indonesia/{viewer}?{parameter}={value}"


def raw_words(text: str) -> list[str]:
    return text.split()


def nearby_candidates(rows: list[dict[str, str]], current_row: int,
                      search_amount: int) -> list[dict]:
    candidates: list[dict] = []
    if current_row > 0:
        previous_words = raw_words(rows[current_row - 1]["text"])
        first_index = max(0, len(previous_words) - search_amount)
        for word_index in range(first_index, len(previous_words)):
            candidates.append({
                "word": previous_words[word_index],
                "normalized_word": normalize_word(previous_words[word_index]),
                "row_number": current_row,
                "word_index": word_index,
                "boundary_offset": word_index - len(previous_words),
                "location": "previous_row",
            })
    # Treat the nominal row and following rows as one continuous stream. If
    # dropping words exhausts a short current row, matching continues at the
    # beginning of the next row until search_amount positions are available.
    forward_offset = 0
    for row_index in range(current_row, len(rows)):
        for word_index, word in enumerate(raw_words(rows[row_index]["text"])):
            if forward_offset >= search_amount:
                break
            candidates.append({
                "word": word,
                "normalized_word": normalize_word(word),
                "row_number": row_index + 1,
                "word_index": word_index,
                "boundary_offset": forward_offset,
                "location": "current_row" if row_index == current_row else "next_row",
            })
            forward_offset += 1
        if forward_offset >= search_amount:
            break
    return [candidate for candidate in candidates if candidate["normalized_word"]]


def words_from_match(rows: list[dict[str, str]], match: dict,
                     display_words: int) -> list[str]:
    row_index = int(match["row_number"]) - 1
    result = raw_words(rows[row_index]["text"])[int(match["word_index"]):]
    for later_row in rows[row_index + 1:]:
        result.extend(raw_words(later_row["text"]))
        if len(result) >= display_words:
            break
    return result[:display_words]


def transcript_words_from_match(rows: list[dict[str, str]], match: dict,
                                final_row_number: int) -> list[str]:
    """Return the corrected transcript from the match through the clip end.

    Row numbers in ``match`` and ``final_row_number`` are one-based TSV row
    numbers. A match may fall in the preceding row, the nominal first row, or
    a following row when the nominal row has been exhausted.
    """
    match_row_index = int(match["row_number"]) - 1
    final_row_index = max(match_row_index, final_row_number - 1)
    result = raw_words(rows[match_row_index]["text"])[int(match["word_index"]):]
    for later_row in rows[match_row_index + 1:final_row_index + 1]:
        result.extend(raw_words(later_row["text"]))
    return result


def phrase_from_match(rows: list[dict[str, str]], match: dict,
                      match_words: int) -> list[str]:
    """Read consecutive TSV words from a candidate, crossing row boundaries."""
    return words_from_match(rows, match, match_words)


def choose_match(predicted_phrase: str, candidates: list[dict],
                 metric: Metric) -> tuple[dict, list[dict]]:
    scored = []
    for candidate in candidates:
        distance, details = metric(predicted_phrase, candidate["normalized_phrase"])
        scored.append({**candidate, "distance": distance, "metric_details": details})
    if not scored:
        raise ValueError("no nearby TSV words were available")
    # For identical metric distances, prefer the word closest to the nominal
    # boundary: current first word (0), previous final word (-1), then outward.
    scored.sort(key=lambda item: (
        item["distance"], abs(int(item["boundary_offset"])),
        0 if item["location"] == "current_row" else 1,
    ))
    return scored[0], scored


def load_audio(ffmpeg: str, source: Path, seconds: float):
    import torch

    result = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(source), "-t", f"{seconds:.3f}",
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        check=True, stdout=subprocess.PIPE,
    )
    return torch.frombuffer(bytearray(result.stdout), dtype=torch.float32)


def output_relative_audio_path(audio_path: Path, output: Path) -> str:
    """Return an audio path relative to the generated JSON file."""
    resolved = audio_path if audio_path.is_absolute() else Path.cwd() / audio_path
    return os.path.relpath(resolved.resolve(), output.parent.resolve())


def jember_clip_metadata(row: dict[str, str]) -> tuple[str, list[int]] | None:
    """Return the Jember recording and TSV rows represented by a manifest row."""
    try:
        utterance_ids = json.loads(row.get("utterance_ids", "[]"))
    except json.JSONDecodeError:
        utterance_ids = []
    jember_ids = [
        value for value in utterance_ids
        if isinstance(value, str) and value.startswith("jember:")
    ]
    if row.get("dataset") not in (None, "", "jember") or not jember_ids:
        return None
    parts = [value.split(":") for value in jember_ids]
    recording = parts[0][1]
    row_numbers = [int(value[2]) for value in parts if value[1] == recording]
    return (recording, row_numbers) if row_numbers else None


def load_clips(clips_path: Path, output: Path) -> tuple[list[dict], int]:
    """Load legacy preview JSON or the stage-1 CSV and normalize its fields."""
    if clips_path.suffix.lower() == ".json":
        payload = json.loads(clips_path.read_text(encoding="utf-8"))
        clips = []
        for clip in payload["clips"]:
            source = clips_path.parent / clip["audio_path"]
            clips.append({
                **clip,
                "audio_path": output_relative_audio_path(source, output),
                "_input_audio_path": str(source),
                "full_transcript": (
                    clip.get("full_transcript")
                    or clip.get("core_transcript")
                    or clip.get("nominal_transcript", "")
                ),
            })
        return clips, 0

    if clips_path.suffix.lower() != ".csv":
        raise ValueError("--clips must be a stage-1 CSV or legacy preview JSON")
    with clips_path.open(encoding="utf-8", newline="") as file:
        source_rows = list(csv.DictReader(file))
    clips, skipped = [], 0
    for row in source_rows:
        metadata = jember_clip_metadata(row)
        if metadata is None:
            skipped += 1
            continue
        recording, row_numbers = metadata
        source = Path(row["audio_path"])
        clips.append({
            "clip_id": source.stem,
            "recording": recording,
            "audio_path": output_relative_audio_path(source, output),
            "_input_audio_path": str(source),
            "duration_seconds": (
                (int(row["end_ms"]) - int(row["start_ms"])) / 1000
            ),
            "core_row_numbers": row_numbers,
            "full_transcript": row.get("transcript", ""),
        })
    return clips, skipped


def write_processed_manifest(source: Path, output: Path,
                             results: list[dict]) -> tuple[int, int]:
    """Write start-corrected rows with every transcript normalized."""
    with source.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        source_fields = list(reader.fieldnames or [])
        source_rows = list(reader)

    result_by_clip = {result["clip_id"]: result for result in results}
    metadata_fields = [
        "original_transcript", "start_match_status", "start_match_metric",
        "start_match_distance", "start_match_row_number",
        "start_match_word_index", "start_match_location",
        "start_match_predicted_phrase", "word_count",
    ]
    output_rows: list[dict[str, str]] = []
    processed_jember = unchanged_non_jember = 0
    for source_row in source_rows:
        row = dict(source_row)
        original = row.get("transcript", "")
        row["original_transcript"] = original
        if jember_clip_metadata(row) is None:
            row.update({
                "start_match_status": "not_applicable",
                "start_match_metric": "",
                "start_match_distance": "",
                "start_match_row_number": "",
                "start_match_word_index": "",
                "start_match_location": "",
                "start_match_predicted_phrase": "",
                "word_count": str(len(raw_words(original))),
            })
            output_rows.append(row)
            unchanged_non_jember += 1
            continue

        result = result_by_clip.get(Path(row["audio_path"]).stem)
        # A scoped --limit/--clip-id run must not silently pass unprocessed
        # Jember clips into the next pipeline stage as if they were corrected.
        if result is None:
            continue
        match = result.get("matched_word") or {}
        corrected = result.get("corrected_starting_transcript", "")
        if corrected:
            row["transcript"] = corrected
        words = raw_words(row["transcript"])
        row.update({
            "word_langids": json.dumps(
                [{"word": word, "langid": "other"} for word in words],
                ensure_ascii=False,
            ),
            "language_counts": json.dumps({"other": len(words)}),
            "start_match_status": result["status"],
            "start_match_metric": result["closeness_metric"],
            "start_match_distance": (
                f"{float(match['distance']):.6f}" if "distance" in match else ""
            ),
            "start_match_row_number": str(match.get("row_number", "")),
            "start_match_word_index": str(match.get("word_index", "")),
            "start_match_location": str(match.get("location", "")),
            "start_match_predicted_phrase": result["predicted_start_phrase"],
            "word_count": str(len(words)),
        })
        output_rows.append(row)
        processed_jember += 1

    for row in output_rows:
        normalized = normalize_text(row.get("transcript", "")).strip()
        if not normalized:
            raise ValueError(
                "Transcript normalization produced an empty transcript for "
                f"{row.get('audio_path', 'unknown audio')}"
            )
        words = normalized.split()
        row.update({
            "transcript": normalized,
            "word_langids": json.dumps(
                [{"word": word, "langid": "other"} for word in words],
                ensure_ascii=False,
            ),
            "language_counts": json.dumps({"other": len(words)}),
            "word_count": str(len(words)),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*source_fields]
    for field in metadata_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    return processed_jember, unchanged_non_jember


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config", type=Path, help="YAML named-run configuration file."
    )
    config_args, _ = config_parser.parse_known_args()
    config_values, _ = load_section(config_args.config, "match")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="YAML named-run configuration file."
    )
    parser.add_argument(
        "--clips", type=Path, default=Path("processed_indonesia/01_segments.csv"),
        help="Stage-1 CSV containing Jember and Indonesian development clips.",
    )
    parser.add_argument("--tsv", type=Path, default=Path("indonesian_data/Jember Javanese Spontaneous Speech Corpus/Jember Javanese Spontaneous Speech Corpus - 1-200.tsv"))
    parser.add_argument("--output", type=Path, help="Review JSON; defaults beside --clips.")
    parser.add_argument(
        "--output-csv", type=Path,
        help="Processed manifest; defaults to 02_processed.csv beside --clips.",
    )
    parser.add_argument("--clip-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--audio-seconds", type=float, default=5.0)
    parser.add_argument("--word-search-amount", type=int, default=13)
    parser.add_argument("--match-words", type=int, default=2, help="Consecutive predicted and TSV words compared by the distance metric.")
    parser.add_argument(
        "--display-words", type=int, default=15,
        help="Deprecated compatibility option; corrected transcripts now span the full clip.",
    )
    parser.add_argument("--closeness-metric", choices=sorted(METRICS), default="levenshtein")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()
    if args.clips.suffix.lower() != ".csv":
        parser.error("--clips must be the stage-1 CSV so non-Jember rows can pass through")
    if args.output is None:
        args.output = args.clips.parent / "clip_start_matches.json"
    if args.output_csv is None:
        args.output_csv = args.clips.parent / "02_processed.csv"
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if shutil.which(args.ffmpeg) is None:
        parser.error("ffmpeg must be installed")
    if args.audio_seconds <= 0 or args.word_search_amount < 1 or args.match_words < 1:
        parser.error("audio seconds and word counts must be positive")

    try:
        import torch
        from transformers import AutoModelForCTC, Wav2Vec2Processor
    except ImportError as error:
        raise SystemExit("Run with: uv run --project processing ...") from error

    try:
        selected, non_jember_count = load_clips(args.clips, args.output)
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"Could not read --clips: {error}")
    if non_jember_count:
        print(f"Passing through {non_jember_count} non-Jember clips unchanged")
    if args.clip_id:
        wanted = set(args.clip_id)
        selected = [clip for clip in selected if clip["clip_id"] in wanted]
        missing = wanted.difference(clip["clip_id"] for clip in selected)
        if missing:
            parser.error(f"clip IDs not found: {', '.join(sorted(missing))}")
    if args.limit is not None:
        selected = selected[:args.limit]
    with args.tsv.open(encoding="utf-8", newline="") as file:
        all_rows = list(csv.DictReader(file, delimiter="\t"))
    by_recording: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        by_recording.setdefault(row["Audio file name"], []).append(row)

    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "--device cuda was requested, but torch.cuda.is_available() is false"
        )
    print(f"Loading {args.model} on {device}")
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device).eval()
    model_device = next(model.parameters()).device
    print(f"Wav2Vec2 model parameters loaded on {model_device}")
    if device == "cuda" and model_device.type != "cuda":
        raise RuntimeError(
            f"Wav2Vec2 was requested on CUDA but loaded on {model_device}"
        )
    metric = METRICS[args.closeness_metric]
    results = []

    for index, clip in enumerate(selected, start=1):
        rows = by_recording[str(clip["recording"])]
        current_row = int(clip["core_row_numbers"][0]) - 1
        audio_path = Path(clip["_input_audio_path"])
        audio = load_audio(args.ffmpeg, audio_path, args.audio_seconds)
        inputs = processor(audio.numpy(), sampling_rate=16000, return_tensors="pt")
        with torch.inference_mode():
            logits = model(inputs.input_values.to(device)).logits
        prediction = processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()
        predicted_words = prediction.split()
        predicted_start_words = predicted_words[:args.match_words]
        predicted_start = " ".join(normalize_word(word) for word in predicted_start_words)
        candidate_words = nearby_candidates(rows, current_row, args.word_search_amount)
        for candidate in candidate_words:
            phrase_words = phrase_from_match(rows, candidate, args.match_words)
            candidate["phrase_words"] = phrase_words
            candidate["phrase"] = " ".join(phrase_words)
            candidate["normalized_phrase"] = " ".join(
                normalize_word(word) for word in phrase_words
            )
        candidate_words = [
            candidate for candidate in candidate_words
            if len(candidate["phrase_words"]) == args.match_words
            and all(candidate["normalized_phrase"].split())
        ]
        if len(predicted_start_words) == args.match_words and all(predicted_start.split()):
            match, scored = choose_match(predicted_start, candidate_words, metric)
            corrected_words = transcript_words_from_match(
                rows, match, max(int(value) for value in clip["core_row_numbers"])
            )
            status = "matched"
        else:
            match, scored, corrected_words = None, [], []
            status = "empty_prediction"
        results.append({
            "clip_id": clip["clip_id"],
            "recording": clip["recording"],
            "audio_path": clip["audio_path"],
            "clip_duration_seconds": clip["duration_seconds"],
            "audio_transcribed_seconds": min(args.audio_seconds, audio.numel() / 16000),
            "nominal_row_number": current_row + 1,
            "final_row_number": max(int(value) for value in clip["core_row_numbers"]),
            "nominal_transcript": rows[current_row]["text"],
            "full_transcript": clip.get("full_transcript", ""),
            "predicted_transcript": prediction,
            "predicted_first_word": predicted_words[0] if predicted_words else "",
            "predicted_start_words": predicted_start_words,
            "predicted_start_phrase": " ".join(predicted_start_words),
            "normalized_predicted_start_phrase": predicted_start,
            "closeness_metric": args.closeness_metric,
            "status": status,
            "matched_word": match,
            "corrected_starting_transcript": " ".join(corrected_words),
            "candidate_words": scored,
        })
        selected_text = "none" if match is None else (
            f"{match['phrase']} (row {match['row_number']}, "
            f"distance {match['distance']:.3f})"
        )
        print(
            f"[{index}/{len(selected)}] {clip['clip_id']}: "
            f"ASR start={' '.join(predicted_start_words) if predicted_start_words else '<empty>'!r} "
            f"-> {selected_text}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "model": args.model,
        "audio_seconds": args.audio_seconds,
        "word_search_amount": args.word_search_amount,
        "match_words": args.match_words,
        "closeness_metric": args.closeness_metric,
        "available_metrics": sorted(METRICS),
        "source_clips": str(args.clips),
        "processed_manifest": str(args.output_csv),
        "clips": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} clip-start matches to {args.output}")
    processed_jember, unchanged_non_jember = write_processed_manifest(
        args.clips, args.output_csv, results
    )
    print(
        f"Wrote {processed_jember} corrected Jember clips and "
        f"{unchanged_non_jember} non-Jember clips with normalized transcripts "
        f"to {args.output_csv}"
    )
    link = viewer_url("clip_start_word_match_review.html", "data", args.output)
    if link:
        print(f"View clip-start matches: {link}")
    else:
        print("Clip-start viewer link unavailable: output JSON is outside the repository.")


if __name__ == "__main__":
    main()
