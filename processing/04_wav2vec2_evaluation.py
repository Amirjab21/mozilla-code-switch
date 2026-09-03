#!/usr/bin/env python3
"""Transcribe approved clips with Indonesian NLP Wav2Vec2 and calculate WER."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import quote


MODEL_ID = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"


def relative_path(path: Path, base: Path) -> str:
    """Return a browser-friendly path from *base* to *path*."""
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def viewer_url(config: Path) -> str | None:
    """Build a localhost evaluation-viewer URL for a repository config."""
    repository = Path(__file__).resolve().parent.parent
    try:
        relative = config.resolve().relative_to(repository).as_posix()
    except ValueError:
        return None
    value = quote(f"../{relative}", safe="/")
    return (
        "http://127.0.0.1:8000/analysis/"
        f"whisper-evaluation.html?config={value}"
    )


def evaluation_helpers():
    source = Path(__file__).with_name("04_whisper_evaluation.py")
    spec = importlib.util.spec_from_file_location("whisper_evaluation", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluation helpers from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/02_vad.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("processed_indonesia/04_wav2vec2_evaluation.csv"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--limit", type=int, help="Run only the first N retained clips.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    try:
        import torch
        import whisper
        from transformers import AutoModelForCTC, Wav2Vec2Processor
    except ImportError as error:
        raise SystemExit("Install inference dependencies first: uv sync --project processing") from error

    helpers = evaluation_helpers()
    device = helpers.choose_device(args.device, torch)
    # The repository also ships a KenLM decoder, but direct CTC decoding is the
    # model card's documented path and does not require that optional runtime.
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(device).eval()
    with args.manifest.open(encoding="utf-8", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("filtered_out", "False").casefold() != "true"]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error(f"No retained clips found in {args.manifest}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        reference = helpers.normalise_words(row["transcript"])
        labels = helpers.reference_labels(row, reference)
        audio = whisper.load_audio(row["audio_path"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(
                inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device) if inputs.attention_mask is not None else None,
            ).logits
        prediction = processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()
        hypothesis = helpers.normalise_words(prediction)
        alignment, scores = helpers.align_words(reference, hypothesis, labels)
        output_rows.append({
            "audio_path": row["audio_path"], "transcript": row["transcript"], "asr_transcript": prediction,
            "reference_language_ids": json.dumps(sorted(set(labels))), "reference_dominant_language": helpers.dominant_reference(labels),
            "wer": f"{scores['wer']:.6f}" if scores["wer"] is not None else "",
            "wer_by_langid": json.dumps(scores["by_langid"], ensure_ascii=False, sort_keys=True),
            "alignment": json.dumps(alignment, ensure_ascii=False), "model_backend": "wav2vec2", "model_name": args.model, "device": device,
        })
        print(f"[{index}/{len(rows)}] {Path(row['audio_path']).name}: WER={scores['wer']:.1%}")

    fields = ["audio_path", "transcript", "asr_transcript", "reference_language_ids", "reference_dominant_language", "wer", "wer_by_langid", "alignment", "model_backend", "model_name", "device"]
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} Wav2Vec2 transcription evaluations to {args.output_csv}")

    config = args.output_csv.parent / "wav2vec2_review.json"
    audio_directory = Path(rows[0]["audio_path"]).parent
    config.write_text(json.dumps({
        "title": f"Wav2Vec2 evaluation — {args.output_csv.parent.name}",
        "paths": {
            "vad_audio_dir": relative_path(audio_directory, config.parent) + "/",
            "whisper_evaluation": relative_path(args.output_csv, config.parent),
        },
    }, indent=2) + "\n", encoding="utf-8")
    link = viewer_url(config)
    if link:
        print(f"View Wav2Vec2 results: {link}")
    else:
        print("Wav2Vec2 viewer link unavailable: output is outside the repository.")


if __name__ == "__main__":
    main()
