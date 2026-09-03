#!/usr/bin/env python3
"""Fine-tune Indonesian/Javanese Wav2Vec2 CTC with a LoRA adapter."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path

import torch
import wandb
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor
from wer_metrics import normalise_for_wer, word_error_rate


MODEL_ID = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"


def training_text(text: str) -> str:
    """Make references compatible with the model's lowercase Latin CTC vocabulary."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", text)).strip()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    rows = [row for row in rows if Path(row.get("audio_path", "")).is_file() and training_text(row.get("transcript", ""))]
    if len(rows) < 2:
        raise ValueError("Training requires at least two valid rows")
    return rows


def filter_jember_001_first_ten(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Select the ten nested Jember-001 windows ending at TSV rows 1–10.

    Stage 01 names these clips ``jember_001_0001-0001.wav`` through
    ``jember_001_0001-0010.wav``.  Keeping this selection in a function makes
    the tiny overfitting/smoke test explicit and prevents an accidental random
    ten-row sample from being used instead.
    """
    pattern = re.compile(r"^jember_001_0001-(\d{4})$")
    selected: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        match = pattern.fullmatch(Path(row.get("audio_path", "")).stem)
        if match and 1 <= int(match.group(1)) <= 10:
            selected.append((int(match.group(1)), row))
    selected.sort(key=lambda item: item[0])
    if len(selected) != 10:
        found = [Path(row["audio_path"]).stem for _, row in selected]
        raise ValueError(
            "Expected all 10 Jember-001 preview windows ending at rows 1–10; "
            f"found {len(selected)}: {found}"
        )
    return [row for _, row in selected]


def split_preview_rows(
    rows: list[dict[str, str]], test_size: int = 2,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Use the final windows as a stable holdout for a tiny diagnostic run."""
    if not 0 < test_size < len(rows):
        raise ValueError("Preview test size must leave non-empty train and test sets")
    return rows[:-test_size], rows[-test_size:]


def split_by_development_speaker(
    rows: list[dict[str, str]], test_speaker_count: int, seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    """Hold out complete Indonesian-development speakers for final testing."""
    development_speakers = sorted({
        row.get("speakers", "").strip()
        for row in rows
        if row.get("dataset") == "indonesian_development"
        and row.get("speakers", "").strip()
        and row.get("speakers", "").strip().casefold() != "unknown"
    })
    if len(development_speakers) < test_speaker_count:
        raise ValueError(
            f"Requested {test_speaker_count} test speakers, but the manifest "
            f"contains only {len(development_speakers)} usable Indonesian "
            "development speakers"
        )
    random.Random(seed).shuffle(development_speakers)
    held_out_speakers = sorted(development_speakers[:test_speaker_count])
    held_out = set(held_out_speakers)
    test = [
        row for row in rows
        if row.get("dataset") == "indonesian_development"
        and row.get("speakers", "").strip() in held_out
    ]
    train = [
        row for row in rows
        if not (
            row.get("dataset") == "indonesian_development"
            and row.get("speakers", "").strip() in held_out
        )
    ]
    if not train or not test:
        raise ValueError("Speaker split must produce non-empty train and test sets")
    return train, test, held_out_speakers


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AudioDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], processor: Wav2Vec2Processor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        import whisper

        row = self.rows[index]
        audio = whisper.load_audio(row["audio_path"])
        input_values = self.processor(audio, sampling_rate=16000).input_values[0]
        label_ids = self.processor.tokenizer(training_text(row["transcript"])).input_ids
        return {"input_values": input_values, "label_ids": label_ids, "row": row}


class CTCDataCollator:
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor

    def __call__(self, items: list[dict]) -> dict:
        inputs = self.processor.pad(
            [{"input_values": item["input_values"]} for item in items],
            padding=True,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": item["label_ids"]} for item in items],
            padding=True,
            return_tensors="pt",
        )
        inputs["labels"] = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        inputs["rows"] = [item["row"] for item in items]
        return inputs


@torch.no_grad()
def evaluate(model, loader: DataLoader, processor: Wav2Vec2Processor, device: torch.device) -> tuple[float, float, list[dict[str, str]]]:
    model.eval()
    total_loss = 0.0
    batches = substitutions = deletions = insertions = reference_words = 0
    predictions: list[dict[str, str]] = []
    for batch in tqdm(loader, desc="Evaluation", leave=False):
        rows = batch.pop("rows")
        tensors = {key: value.to(device) for key, value in batch.items()}
        output = model(**tensors)
        total_loss += float(output.loss)
        batches += 1
        decoded = processor.batch_decode(output.logits.argmax(dim=-1))
        for row, prediction in zip(rows, decoded):
            clip_substitutions, clip_deletions, clip_insertions, clip_words = (
                word_error_rate(row["transcript"], prediction)
            )
            clip_errors = clip_substitutions + clip_deletions + clip_insertions
            substitutions += clip_substitutions
            deletions += clip_deletions
            insertions += clip_insertions
            reference_words += clip_words
            predictions.append({
                "audio_path": row["audio_path"],
                "reference": row["transcript"],
                "training_reference": training_text(row["transcript"]),
                "prediction": prediction.strip(),
                "substitutions": str(clip_substitutions),
                "deletions": str(clip_deletions),
                "insertions": str(clip_insertions),
                "reference_words": str(clip_words),
                "word_errors": str(clip_errors),
                "wer": f"{clip_errors / clip_words:.6f}" if clip_words else "0.000000",
            })
    errors = substitutions + deletions + insertions
    corpus_wer = errors / reference_words if reference_words else float("nan")
    return total_loss / max(batches, 1), corpus_wer, predictions


def write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "audio_path", "reference", "training_reference", "prediction",
            "substitutions", "deletions", "insertions", "reference_words",
            "word_errors", "wer",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("processed_indonesia/02_processed.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia/05_wav2vec2_lora"))
    parser.add_argument("--base-model", default=MODEL_ID)
    parser.add_argument(
        "--jember-001-first-ten", action="store_true",
        help=(
            "Restrict the run to jember_001_0001-0001 through "
            "jember_001_0001-0010 and use a deterministic 8/2 diagnostic split."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--test-speaker-count", type=int, default=2,
        help="Number of complete Indonesian-development speakers held out for testing.",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Limit training rows after the speaker-disjoint test split.",
    )
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument(
        "--eval-every-steps", type=int, default=0,
        help="Run intermediate held-out evaluation every N optimizer steps; 0 disables it.",
    )
    parser.add_argument(
        "--eval-size", type=int, default=100,
        help="Maximum held-out clips used for each intermediate evaluation.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", default="indonesian-wav2vec2-lora")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.gradient_accumulation, args.test_speaker_count) < 1:
        parser.error("epochs, batch size, gradient accumulation, and test speaker count must be positive")
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples must be at least 2")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be at least 1")
    if args.eval_every_steps < 0:
        parser.error("--eval-every-steps must be zero or greater")
    if args.eval_size < 1:
        parser.error("--eval-size must be at least 1")

    load_dotenv(args.env_file)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    rows = read_rows(args.manifest)
    if args.jember_001_first_ten:
        rows = filter_jember_001_first_ten(rows)
        train_rows, test_rows = split_preview_rows(rows)
        test_speakers: list[str] = []
        split_rule = (
            "diagnostic only: first eight nested Jember-001 windows train; "
            "final two windows test"
        )
    else:
        train_rows, test_rows, test_speakers = split_by_development_speaker(
            rows, args.test_speaker_count, args.seed
        )
        split_rule = (
            "all clips from the selected Indonesian-development speakers are "
            "test data; every other row is training data"
        )
    eval_rows = test_rows[:args.eval_size]
    if args.max_samples is not None:
        random.Random(args.seed).shuffle(train_rows)
        train_rows = train_rows[:args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "train_split.csv", train_rows)
    write_rows(args.output_dir / "test_split.csv", test_rows)
    (args.output_dir / "split_metadata.json").write_text(json.dumps({
        "source_manifest": str(args.manifest),
        "seed": args.seed,
        "test_speakers": test_speakers,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "intermediate_eval_rows": len(eval_rows),
        "split_rule": split_rule,
    }, indent=2) + "\n", encoding="utf-8")
    print(
        f"Training rows: {len(train_rows)}; test rows: {len(test_rows)}; "
        f"held-out development speakers: {', '.join(test_speakers) or 'none (diagnostic split)'}"
    )

    processor = Wav2Vec2Processor.from_pretrained(args.base_model)
    base_model = AutoModelForCTC.from_pretrained(args.base_model)
    base_model.config.ctc_loss_reduction = "mean"
    base_model.config.ctc_zero_infinity = True
    base_model.freeze_feature_encoder()
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        # Adapt every linear layer in the encoder, including all attention
        # projections and both feed-forward projections. PEFT excludes the
        # output head here; it is trained in full through modules_to_save.
        target_modules="all-linear",
        modules_to_save=["lm_head"],
    )
    model = get_peft_model(base_model, lora).to(device)
    model.print_trainable_parameters()

    collator = CTCDataCollator(processor)
    train_loader = DataLoader(AudioDataset(train_rows, processor), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collator)
    test_loader = DataLoader(AudioDataset(test_rows, processor), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collator)
    eval_loader = DataLoader(AudioDataset(eval_rows, processor), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collator)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    wandb_run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "device": str(device),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "test_speakers": test_speakers,
            "wer_rules": "case-folded; punctuation removed; accents preserved",
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    )
    global_step = 0
    try:
        def run_intermediate_evaluation(step: int) -> None:
            eval_loss, eval_wer, evaluation_rows = evaluate(
                model, eval_loader, processor, device
            )
            prediction_path = args.output_dir / "eval_predictions.csv"
            write_predictions(prediction_path, evaluation_rows)
            wandb_run.log({
                "eval/loss": eval_loss,
                "eval/wer": eval_wer,
                "eval/clips": len(eval_rows),
            }, step=step)
            print(
                f"step {step}: eval loss={eval_loss:.4f}, "
                f"WER={eval_wer:.2%} ({len(eval_rows)} clips)"
            )
            model.train()

        stop = False
        accumulated_loss = 0.0
        accumulated_batches = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch_number, batch in enumerate(progress, start=1):
                batch.pop("rows")
                tensors = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    raw_loss = model(**tensors).loss
                    loss = raw_loss / args.gradient_accumulation
                scaler.scale(loss).backward()
                accumulated_loss += float(raw_loss.detach())
                accumulated_batches += 1
                if batch_number % args.gradient_accumulation == 0 or batch_number == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    mean_loss = accumulated_loss / accumulated_batches
                    metrics = {
                        "train/loss": mean_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                        "train/epoch_progress": (
                            epoch - 1 + batch_number / len(train_loader)
                        ),
                    }
                    wandb_run.log(metrics, step=global_step)
                    progress.set_postfix(loss=f"{mean_loss:.4f}", step=global_step)
                    accumulated_loss = 0.0
                    accumulated_batches = 0
                    if (
                        args.eval_every_steps
                        and global_step % args.eval_every_steps == 0
                    ):
                        run_intermediate_evaluation(global_step)
                    if args.max_train_steps is not None and global_step >= args.max_train_steps:
                        stop = True
                        break

            if stop:
                break

        adapter_dir = args.output_dir / "final_adapter"
        model.save_pretrained(adapter_dir)
        processor.save_pretrained(adapter_dir)
        test_loss, test_wer, prediction_rows = evaluate(
            model, test_loader, processor, device
        )
        prediction_path = args.output_dir / "test_predictions.csv"
        write_predictions(prediction_path, prediction_rows)
        wandb_run.log({
            "test/loss": test_loss,
            "test/wer": test_wer,
            "test/clips": len(test_rows),
        }, step=global_step)
        wandb_run.summary["test_loss"] = test_loss
        wandb_run.summary["test_wer"] = test_wer
        wandb_run.summary["test_speakers"] = test_speakers
        wandb_run.summary["final_adapter"] = str(adapter_dir)
        wandb_run.summary["test_predictions"] = str(prediction_path)
        (adapter_dir / "training_metadata.json").write_text(json.dumps({
            "base_model": args.base_model,
            "test_loss": test_loss,
            "test_wer": test_wer,
            "test_speakers": test_speakers,
            "training_steps": global_step,
            "wandb_project": args.wandb_project,
            "wandb_run_id": wandb_run.id,
            "wandb_run_name": wandb_run.name,
            "wandb_run_url": wandb_run.url,
            "text_rules": "lowercase ASCII letters; accents folded; punctuation removed",
            "wer_rules": "case-folded; punctuation removed; accents preserved",
        }, indent=2) + "\n", encoding="utf-8")
        print(
            f"Final adapter: {adapter_dir}; held-out test loss={test_loss:.4f}, "
            f"WER={test_wer:.2%}"
        )
        if wandb_run.url:
            print(f"W&B run: {wandb_run.url}")
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
