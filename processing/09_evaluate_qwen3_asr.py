#!/usr/bin/env python3
"""Evaluate a Qwen3-ASR LoRA adapter with greedy autoregressive decoding."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from wer_metrics import word_error_rate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = Path(
    "processed_indonesia/first_full_run_data_mix_commonvoice_jember_1200/"
    "05_train_qwen3_asr_1_7b"
)
OUTPUT_FIELDS = (
    "audio_file_path",
    "ground_truth_transcript",
    "language_labels",
    "predicted_transcript",
    "predicted_language",
    "predicted_language_labels",
    "wer",
    "reference_words",
    "word_errors",
    "substitutions",
    "deletions",
    "insertions",
    "decoding",
)


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def resolve_audio_path(value: str, csv_path: Path) -> Path:
    requested = Path(value).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.extend((REPOSITORY_ROOT / requested, csv_path.parent / requested))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Audio file not found: {value}")


def load_adapter(
    path: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    attention_implementation: str,
):
    try:
        from peft import PeftModel
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise ImportError(
            "Run `uv sync --project processing` to install Qwen3-ASR and PEFT."
        ) from error

    metadata_path = path / "qwen3_asr_lora_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Qwen adapter metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    options = {
        "dtype": dtype,
        "device_map": None,
        "max_new_tokens": max_new_tokens,
    }
    if attention_implementation != "auto":
        options["attn_implementation"] = attention_implementation
    wrapper = Qwen3ASRModel.from_pretrained(metadata["base_model"], **options)
    wrapper.model.thinker = PeftModel.from_pretrained(
        wrapper.model.thinker, str(path), is_trainable=False
    )
    wrapper.model.to(device).eval()
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_RUN_DIR / "best_wer")
    parser.add_argument(
        "--evaluation-csv",
        type=Path,
        default=DEFAULT_RUN_DIR / "best_wer_eval_predictions.csv",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attention-implementation", choices=("auto", "eager", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.adapter_dir.is_dir():
        parser.error(f"Adapter directory not found: {args.adapter_dir}")
    if not args.evaluation_csv.is_file():
        parser.error(f"Evaluation CSV not found: {args.evaluation_csv}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    with args.evaluation_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"audio_file_path", "ground_truth_transcript"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        parser.error(
            f"{args.evaluation_csv} is missing columns: {', '.join(sorted(missing))}"
        )
    if args.limit is not None:
        rows = rows[:args.limit]

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    wrapper = load_adapter(
        args.adapter_dir,
        device,
        dtype,
        args.max_new_tokens,
        args.attention_implementation,
    )
    output_rows = []
    total_substitutions = total_deletions = total_insertions = 0
    total_reference_words = 0
    for index, row in enumerate(rows, start=1):
        audio_path = resolve_audio_path(row["audio_file_path"], args.evaluation_csv)
        result = wrapper.transcribe(audio=str(audio_path), language=None)[0]
        prediction = result.text.strip()
        substitutions, deletions, insertions, reference_words = word_error_rate(
            row["ground_truth_transcript"], prediction
        )
        errors = substitutions + deletions + insertions
        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions
        total_reference_words += reference_words
        output_rows.append(
            {
                "audio_file_path": row["audio_file_path"],
                "ground_truth_transcript": row["ground_truth_transcript"],
                "language_labels": row.get("language_labels", "[]"),
                "predicted_transcript": prediction,
                "predicted_language": result.language,
                "predicted_language_labels": "[]",
                "wer": errors / reference_words if reference_words else math.nan,
                "reference_words": reference_words,
                "word_errors": errors,
                "substitutions": substitutions,
                "deletions": deletions,
                "insertions": insertions,
                "decoding": "Qwen3-ASR greedy autoregressive, no teacher forcing",
            }
        )
        print(
            f"[{index}/{len(rows)}] WER="
            f"{errors / reference_words if reference_words else math.nan:.2%} "
            f"— {audio_path.name}"
        )

    output_csv = args.output_csv or args.evaluation_csv.with_name(
        "qwen_greedy_eval_predictions.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    total_errors = total_substitutions + total_deletions + total_insertions
    aggregate_wer = (
        total_errors / total_reference_words if total_reference_words else math.nan
    )
    metrics_path = output_csv.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "clips": len(output_rows),
                "reference_words": total_reference_words,
                "substitutions": total_substitutions,
                "deletions": total_deletions,
                "insertions": total_insertions,
                "word_errors": total_errors,
                "wer": aggregate_wer,
                "teacher_forcing": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Aggregate WER: {aggregate_wer:.2%}")
    print(f"Wrote predictions to {output_csv}")
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
