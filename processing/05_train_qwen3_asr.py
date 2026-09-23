#!/usr/bin/env python3
"""Fine-tune Qwen3-ASR with all-linear LoRA on the Indonesian data mix."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import types
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import torch
import wandb
from dotenv import load_dotenv
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from qwen_training_data import (
    cap_jember_training_rows,
    qwen_target,
    read_rows,
    split_proportional_augmented_evaluation,
    split_rows,
    write_manifest,
)
from run_config import apply_defaults, load_section
from wer_metrics import word_error_rate


IGNORE_INDEX = -100
EVAL_FIELDS = [
    "audio_file_path",
    "ground_truth_transcript",
    "language_labels",
    "predicted_transcript",
    "predicted_language",
    "predicted_language_labels",
    "wer",
    "reference_words",
    "word_errors",
    "loss",
    "token_loss",
    "language_id_loss",
]


class QwenASRDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "audio": row["audio_path"],
            "target": qwen_target(row),
            "row": row,
        }


class QwenASRCollator:
    def __init__(self, processor: Any, sampling_rate: int = 16_000):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def prefix_text(self, audio: None = None) -> str:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio}]},
        ]
        rendered = self.processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False
        )
        return rendered[0] if isinstance(rendered, list) else rendered

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        prefix = self.prefix_text()
        prefix_texts = [prefix] * len(features)
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [
            prefix_text + feature["target"] + eos
            for prefix_text, feature in zip(prefix_texts, features)
        ]
        audios = [
            librosa.load(feature["audio"], sr=self.sampling_rate, mono=True)[0]
            for feature in features
        ]
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        labels = full_inputs["input_ids"].clone()
        for index, prefix_length in enumerate(
            prefix_inputs["attention_mask"].sum(dim=1).tolist()
        ):
            labels[index, :prefix_length] = IGNORE_INDEX
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = IGNORE_INDEX
        full_inputs["labels"] = labels
        full_inputs["rows"] = [feature["row"] for feature in features]
        return full_inputs


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def move_model_inputs(
    batch: dict[str, Any], device: torch.device, dtype: torch.dtype
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = batch.pop("rows")
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype=dtype)
        moved[key] = value
    return moved, rows


def add_embedding_shim(model: nn.Module) -> None:
    """Expose embeddings for PEFT versions that require the Transformers API."""
    if hasattr(model, "get_input_embeddings"):
        return
    embeddings = [
        module for module in model.modules() if isinstance(module, nn.Embedding)
    ]
    if not embeddings:
        raise RuntimeError("Qwen3-ASR thinker contains no token embedding")
    embedding = max(embeddings, key=lambda module: module.num_embeddings)
    model.get_input_embeddings = types.MethodType(  # type: ignore[attr-defined]
        lambda _self: embedding, model
    )


def all_linear_target_modules(model: nn.Module) -> list[str]:
    """Select every linear layer in both the audio encoder and text decoder."""
    targets = [
        name for name, module in model.named_modules()
        if name and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise RuntimeError("No linear modules were found in the Qwen3-ASR thinker")
    return targets


def load_qwen_lora(
    base_model: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    alpha: int,
    dropout: float,
    attention_implementation: str,
    gradient_checkpointing: bool,
    max_new_tokens: int,
) -> tuple[Any, Any, Any, list[str]]:
    try:
        from peft import LoraConfig, get_peft_model
        from qwen_asr import Qwen3ASRModel
    except ImportError as error:
        raise ImportError(
            "Qwen training requires qwen-asr and peft. "
            "Run `uv sync --project processing` in the CUDA environment."
        ) from error

    load_options: dict[str, Any] = {
        "dtype": dtype,
        "device_map": None,
        "max_new_tokens": max_new_tokens,
    }
    if attention_implementation != "auto":
        load_options["attn_implementation"] = attention_implementation
    wrapper = Qwen3ASRModel.from_pretrained(base_model, **load_options)
    outer_model = wrapper.model
    thinker = outer_model.thinker
    add_embedding_shim(thinker)
    targets = all_linear_target_modules(thinker)
    adapter = get_peft_model(
        thinker,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=targets,
            ensure_weight_tying=True,
        ),
    )
    outer_model.thinker = adapter
    outer_model.to(device)
    if gradient_checkpointing:
        adapter.gradient_checkpointing_enable()
        if hasattr(adapter, "enable_input_require_grads"):
            adapter.enable_input_require_grads()
    if hasattr(adapter.config, "use_cache"):
        adapter.config.use_cache = False
    return wrapper, outer_model, adapter, targets


def save_adapter(
    adapter: Any,
    processor: Any,
    path: Path,
    *,
    base_model: str,
    target_modules: list[str],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    adapter.save_pretrained(str(path), safe_serialization=True)
    processor.save_pretrained(str(path))
    (path / "qwen3_asr_lora_metadata.json").write_text(
        json.dumps(
            {
                "format": "qwen3-asr-all-linear-lora-v1",
                "base_model": base_model,
                "target_modules": target_modules,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def write_evaluation(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_loss(
    model: Any,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    for batch in tqdm(loader, desc="Held-out loss", leave=False):
        inputs, _ = move_model_inputs(batch, device, dtype)
        outputs = model(**inputs)
        total_loss += float(outputs.loss)
        batches += 1
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate_transcripts(
    wrapper: Any,
    rows: list[dict[str, str]],
    loss: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    substitutions = deletions = insertions = reference_words = 0
    for row in tqdm(rows, desc="Held-out WER", leave=False):
        result = wrapper.transcribe(
            audio=row["audio_path"],
            language=None,
        )[0]
        predicted_text = result.text.strip()
        s, d, i, words = word_error_rate(row["transcript"], predicted_text)
        substitutions += s
        deletions += d
        insertions += i
        reference_words += words
        predictions.append(
            {
                "audio_file_path": row["audio_path"],
                "ground_truth_transcript": row["transcript"],
                "language_labels": row.get("word_langids", "[]"),
                "predicted_transcript": predicted_text,
                "predicted_language": result.language,
                "predicted_language_labels": "[]",
                "wer": (s + d + i) / words if words else 0.0,
                "reference_words": words,
                "word_errors": s + d + i,
                "loss": loss,
                "token_loss": loss,
                "language_id_loss": "",
            }
        )
    metrics = {
        "eval/loss": loss,
        "eval/token_loss": loss,
        "eval/wer": (
            (substitutions + deletions + insertions) / reference_words
            if reference_words else float("nan")
        ),
        "eval/reference_words": float(reference_words),
        "eval/clips": float(len(rows)),
    }
    return metrics, predictions


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "eval_every_steps": args.eval_every_steps,
        "eval_decode_batch_size": args.eval_decode_batch_size,
        "eval_total_clips": args.eval_total_clips,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        parser.error(f"These settings must be positive: {', '.join(invalid)}")
    if not 0 <= args.lora_dropout < 1:
        parser.error("--lora-dropout must be in [0, 1)")
    if not 0 <= args.eval_jember_proportion <= 100:
        parser.error("--eval-jember-proportion must be between 0 and 100")
    if not 0 <= args.eval_commonvoice_proportion <= 100:
        parser.error("--eval-commonvoice-proportion must be between 0 and 100")
    if args.eval_jember_proportion + args.eval_commonvoice_proportion > 100:
        parser.error("Evaluation proportions cannot total more than 100")
    if args.backend != "qwen3_asr":
        parser.error("This trainer requires `backend: qwen3_asr`")


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, _ = config_parser.parse_known_args()
    config_values, config_run_name = load_section(config_args.config, "train")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", default="qwen3_asr")
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia/05_train_qwen3_asr"))
    parser.add_argument("--base-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--language", default="id", help="Compatibility setting; dataset-specific Qwen language prefixes are used.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-total-clips", type=int, default=100)
    parser.add_argument("--eval-jember-proportion", type=float, default=0.0)
    parser.add_argument("--eval-commonvoice-proportion", type=float, default=0.0)
    parser.add_argument("--eval-every-steps", type=int, default=250)
    parser.add_argument("--eval-decode-batch-size", type=int, default=1)
    parser.add_argument("--jember-max-training-clips", type=int)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attention-implementation", choices=("auto", "eager", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--wandb-project", default="indonesian-qwen3-asr-lora")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--overwrite-output-dir", action="store_true")
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()
    validate_args(parser, args)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        parser.error(
            f"{args.output_dir} is not empty; use --overwrite-output-dir only "
            "when intentionally replacing its checkpoints"
        )
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be positive")
    load_dotenv(args.env_file)
    if args.wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
        parser.error(f"WANDB_API_KEY was not found in {args.env_file}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    rows = read_rows(args.manifest)
    if args.eval_total_clips:
        train_rows, eval_rows, eval_distribution = split_proportional_augmented_evaluation(
            rows,
            args.eval_total_clips,
            args.eval_jember_proportion,
            args.eval_commonvoice_proportion,
            args.seed,
        )
    else:
        train_rows, eval_rows = split_rows(rows, args.seed)
        eval_distribution = dict(Counter(row["dataset"] for row in eval_rows))
    train_rows = cap_jember_training_rows(
        train_rows, args.jember_max_training_clips, args.seed
    )
    if not train_rows or not eval_rows:
        raise RuntimeError("The selected split contains an empty partition")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train_split.csv", train_rows)
    write_manifest(args.output_dir / "test_split.csv", eval_rows)
    print(f"Training rows: {len(train_rows)}; evaluation rows: {len(eval_rows)}")
    print(f"Evaluation distribution: {eval_distribution}")
    print(f"Training distribution: {dict(Counter(row['dataset'] for row in train_rows))}")

    wrapper, model, adapter, target_modules = load_qwen_lora(
        args.base_model,
        device=device,
        dtype=dtype,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        attention_implementation=args.attention_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
        max_new_tokens=args.max_new_tokens,
    )
    processor = wrapper.processor
    adapter.print_trainable_parameters()
    print(
        f"Applied LoRA to all {len(target_modules)} linear modules in the "
        "Qwen3-ASR audio encoder and text decoder."
    )

    train_dataset = QwenASRDataset(train_rows)
    eval_dataset = QwenASRDataset(eval_rows)
    collator = QwenASRCollator(processor)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_decode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
    )

    def train_loader(epoch: int) -> DataLoader:
        generator = torch.Generator().manual_seed(args.seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=collator,
        )

    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    load_dotenv(args.env_file)
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or config_run_name,
        mode=args.wandb_mode,
        config={
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "device": str(device),
            "resolved_dtype": str(dtype),
            "train_clips": len(train_rows),
            "eval_clips": len(eval_rows),
            "lora_target_count": len(target_modules),
        },
    )

    global_step = 0
    best_loss = float("inf")
    best_wer = float("inf")
    last_eval_step = -1

    def run_evaluation() -> None:
        nonlocal best_loss, best_wer, last_eval_step
        if hasattr(adapter.config, "use_cache"):
            adapter.config.use_cache = True
        loss = evaluate_loss(adapter, eval_loader, device, dtype)
        metrics, predictions = evaluate_transcripts(
            wrapper, eval_rows, loss
        )
        write_evaluation(args.output_dir / "eval_predictions.csv", predictions)
        if metrics["eval/loss"] < best_loss:
            best_loss = metrics["eval/loss"]
            save_adapter(
                adapter,
                processor,
                args.output_dir / "best_lora",
                base_model=args.base_model,
                target_modules=target_modules,
            )
            write_evaluation(
                args.output_dir / "best_eval_predictions.csv", predictions
            )
        if metrics["eval/wer"] < best_wer:
            best_wer = metrics["eval/wer"]
            save_adapter(
                adapter,
                processor,
                args.output_dir / "best_wer",
                base_model=args.base_model,
                target_modules=target_modules,
            )
            write_evaluation(
                args.output_dir / "best_wer_eval_predictions.csv", predictions
            )
        metrics["eval/best_loss"] = best_loss
        metrics["eval/best_wer"] = best_wer
        wandb.log(metrics, step=global_step)
        print(
            f"step {global_step}: loss={metrics['eval/loss']:.4f}, "
            f"WER={metrics['eval/wer']:.3%}, best WER={best_wer:.3%}"
        )
        last_eval_step = global_step
        if hasattr(adapter.config, "use_cache"):
            adapter.config.use_cache = False
        adapter.train()

    try:
        optimizer.zero_grad(set_to_none=True)
        reached_limit = False
        for epoch in range(1, args.epochs + 1):
            adapter.train()
            loader = train_loader(epoch)
            progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch_number, batch in enumerate(progress, start=1):
                inputs, _ = move_model_inputs(batch, device, dtype)
                autocast = (
                    torch.autocast("cuda", dtype=dtype)
                    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
                    else contextlib.nullcontext()
                )
                with autocast:
                    outputs = adapter(**inputs)
                    loss = outputs.loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
                should_step = (
                    batch_number % args.gradient_accumulation_steps == 0
                    or batch_number == len(loader)
                )
                if not should_step:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                reported_loss = float(loss.detach()) * args.gradient_accumulation_steps
                wandb.log(
                    {"train/loss": reported_loss, "epoch": epoch},
                    step=global_step,
                )
                progress.set_postfix(loss=f"{reported_loss:.3f}")
                if global_step % args.eval_every_steps == 0:
                    run_evaluation()
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    reached_limit = True
                    break
            save_adapter(
                adapter,
                processor,
                args.output_dir / "last_lora",
                base_model=args.base_model,
                target_modules=target_modules,
            )
            if reached_limit:
                break
        if global_step != last_eval_step:
            run_evaluation()
        run.summary["best_eval_loss"] = best_loss
        run.summary["best_eval_wer"] = best_wer
        run.summary["best_lora_adapter"] = str(args.output_dir / "best_lora")
        run.summary["best_wer_adapter"] = str(args.output_dir / "best_wer")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
