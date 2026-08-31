#!/usr/bin/env python3
"""Fine-tune the local Whisper Small token-language fork on Miami clips."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import wandb
import whisper
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from whisper.tokenizer import get_tokenizer

# `python processing/05_train.py` puts `processing/` rather than the repository
# root on sys.path, so make the local fork importable without installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from models.whisper_lid import load_token_lid_model, save_token_lid_checkpoint
from models.whisper_lid.decode import decode_with_token_language
from models.whisper_lid.labels import IGNORE_INDEX, make_token_language_targets
from run_config import apply_defaults, load_section


# Training defaults. Command-line arguments may override these for an individual run.
TRAIN_FRACTION = 0.90
SPLIT_SEED = 1337
EVAL_SET_SIZE = 100
EVAL_EVERY_STEPS = 250
LID_LOSS_WEIGHT = 0.30


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise_for_wer(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.casefold()).split()


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Return substitutions, deletions, insertions, and reference word count."""
    ref, hyp = normalise_for_wer(reference), normalise_for_wer(hypothesis)
    table = [[(0, 0, 0)] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (0, i, 0)
    for j in range(1, len(hyp) + 1):
        table[0][j] = (0, 0, j)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = table[i - 1][j - 1]
                continue
            candidates = [
                tuple(table[i - 1][j - 1][k] + (1 if k == 0 else 0) for k in range(3)),
                tuple(table[i - 1][j][k] + (1 if k == 1 else 0) for k in range(3)),
                tuple(table[i][j - 1][k] + (1 if k == 2 else 0) for k in range(3)),
            ]
            table[i][j] = min(candidates, key=sum)
    substitutions, deletions, insertions = table[-1][-1]
    return substitutions, deletions, insertions, len(ref)


def read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = {"audio_path", "transcript", "word_langids"} - fields
        if missing:
            raise ValueError(
                f"{manifest} is missing {sorted(missing)}. Re-run stages 01–03 so the final manifest retains word_langids."
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{manifest} contains no rows")
    for row in rows:
        if not Path(row["audio_path"]).exists():
            raise FileNotFoundError(f"Audio file listed in manifest does not exist: {row['audio_path']}")
        try:
            labels = json.loads(row["word_langids"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid word_langids JSON for {row['audio_path']}") from error
        if not labels:
            raise ValueError(f"No word_langids for {row['audio_path']}")
    return rows


def recording_key(row: dict[str, str]) -> str:
    """Derive the original recording ID from a `recording_0001.wav` output name."""
    stem = Path(row["audio_path"]).stem
    return stem.rsplit("_", 1)[0]


def split_rows(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Make a stable 90/10 split, holding entire recordings out when possible."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[recording_key(row)].append(row)
    rng = random.Random(seed)
    target_test_rows = max(1, round(len(rows) * (1 - TRAIN_FRACTION)))
    if len(grouped) >= 2:
        keys = list(grouped)
        rng.shuffle(keys)
        test_keys: set[str] = set()
        count = 0
        for key in keys:
            if count >= target_test_rows:
                break
            test_keys.add(key)
            count += len(grouped[key])
        train = [row for row in rows if recording_key(row) not in test_keys]
        test = [row for row in rows if recording_key(row) in test_keys]
        return train, test, "recording"

    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[target_test_rows:], shuffled[:target_test_rows], "clip (one recording available)"


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MiamiDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tokenizer, n_mels: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.n_mels = n_mels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        word_langids = json.loads(row["word_langids"])
        text_tokens, lid_targets = make_token_language_targets(
            row["transcript"], word_langids, self.tokenizer
        )
        audio = whisper.pad_or_trim(whisper.load_audio(row["audio_path"]))
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.n_mels)
        return {
            "mel": mel,
            "text_tokens": text_tokens,
            "lid_targets": lid_targets,
            "row": row,
        }


def collate_batch(items: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    prefix = list(tokenizer.sot_sequence_including_notimestamps)
    sequences: list[tuple[list[int], list[int], list[int]]] = []
    for item in items:
        text_tokens, lid_targets = item["text_tokens"], item["lid_targets"]
        decoder_input = prefix + text_tokens
        asr_target = [IGNORE_INDEX] * (len(prefix) - 1) + text_tokens + [tokenizer.eot]
        lid_target = [IGNORE_INDEX] * (len(prefix) - 1) + lid_targets + [IGNORE_INDEX]
        if len(decoder_input) != len(asr_target) or len(decoder_input) != len(lid_target):
            raise RuntimeError("Decoder input and target lengths do not agree")
        sequences.append((decoder_input, asr_target, lid_target))

    max_length = max(len(sequence[0]) for sequence in sequences)
    inputs = torch.full((len(items), max_length), tokenizer.eot, dtype=torch.long)
    asr_targets = torch.full_like(inputs, IGNORE_INDEX)
    lid_targets = torch.full_like(inputs, IGNORE_INDEX)
    for index, (decoder_input, asr_target, lid_target) in enumerate(sequences):
        length = len(decoder_input)
        inputs[index, :length] = torch.tensor(decoder_input)
        asr_targets[index, :length] = torch.tensor(asr_target)
        lid_targets[index, :length] = torch.tensor(lid_target)
    return {
        "mels": torch.stack([item["mel"] for item in items]),
        "inputs": inputs,
        "asr_targets": asr_targets,
        "lid_targets": lid_targets,
        "rows": [item["row"] for item in items],
    }


def model_losses(model, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    mels = batch["mels"].to(device)
    inputs = batch["inputs"].to(device)
    asr_targets = batch["asr_targets"].to(device)
    lid_targets = batch["lid_targets"].to(device)
    audio_features = model.encoder(mels)
    token_logits, language_logits = model.logits_with_language(inputs, audio_features)
    asr_loss = F.cross_entropy(token_logits.transpose(1, 2), asr_targets, ignore_index=IGNORE_INDEX)
    valid_lid = lid_targets.ne(IGNORE_INDEX)
    lid_count = int(valid_lid.sum().item())
    if lid_count:
        lid_loss = F.cross_entropy(language_logits.transpose(1, 2), lid_targets, ignore_index=IGNORE_INDEX)
        correct = int((language_logits.argmax(dim=-1)[valid_lid] == lid_targets[valid_lid]).sum().item())
    else:
        lid_loss = torch.zeros((), device=device)
        correct = 0
    return asr_loss, lid_loss, asr_loss + LID_LOSS_WEIGHT * lid_loss, correct, lid_count


@torch.no_grad()
def per_example_loss_values(model, batch: dict[str, Any], device: torch.device) -> list[dict[str, float]]:
    """Calculate the three training objectives independently for every clip in a batch."""
    mels = batch["mels"].to(device)
    inputs = batch["inputs"].to(device)
    asr_targets = batch["asr_targets"].to(device)
    lid_targets = batch["lid_targets"].to(device)
    audio_features = model.encoder(mels)
    token_logits, language_logits = model.logits_with_language(inputs, audio_features)

    def mean_per_clip(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=IGNORE_INDEX, reduction="none")
        valid = targets.ne(IGNORE_INDEX)
        return (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

    token_losses = mean_per_clip(token_logits, asr_targets)
    language_losses = mean_per_clip(language_logits, lid_targets)
    values = []
    for token_loss, language_loss in zip(token_losses.tolist(), language_losses.tolist()):
        values.append({
            "token_loss": token_loss,
            "language_id_loss": language_loss,
            "loss": token_loss + LID_LOSS_WEIGHT * language_loss,
        })
    return values


@torch.no_grad()
def evaluate(model, loader: DataLoader, dataset: MiamiDataset, device: torch.device) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()
    total_asr_loss = total_lid_loss = 0.0
    batches = correct_lid = total_lid = 0
    losses_by_audio_path: dict[str, dict[str, float]] = {}
    for batch in tqdm(loader, desc="Held-out loss", leave=False):
        asr_loss, lid_loss, _, correct, count = model_losses(model, batch, device)
        total_asr_loss += float(asr_loss)
        total_lid_loss += float(lid_loss)
        batches += 1
        correct_lid += correct
        total_lid += count
        for row, values in zip(batch["rows"], per_example_loss_values(model, batch, device)):
            losses_by_audio_path[row["audio_path"]] = values

    substitutions = deletions = insertions = reference_words = 0
    predictions: list[dict[str, str]] = []
    for item in tqdm(dataset, desc="Held-out WER", leave=False):
        result = decode_with_token_language(model, item["mel"], language=None)
        row = item["row"]
        s, d, i, words = word_error_rate(row["transcript"], result.text)
        substitutions += s
        deletions += d
        insertions += i
        reference_words += words
        row_losses = losses_by_audio_path[row["audio_path"]]
        predictions.append({
            "audio_file_path": row["audio_path"],
            "ground_truth_transcript": row["transcript"],
            "language_labels": row["word_langids"],
            "predicted_transcript": result.text,
            "predicted_language_labels": json.dumps(result.words, ensure_ascii=False),
            "wer": (s + d + i) / words if words else 0.0,
            "reference_words": words,
            "word_errors": s + d + i,
            **row_losses,
        })
    token_loss = total_asr_loss / max(batches, 1)
    language_id_loss = total_lid_loss / max(batches, 1)
    return {
        "eval/loss": token_loss + LID_LOSS_WEIGHT * language_id_loss,
        "eval/token_loss": token_loss,
        "eval/language_id_loss": language_id_loss,
        "eval/lid_accuracy": correct_lid / total_lid if total_lid else float("nan"),
        "eval/wer": (substitutions + deletions + insertions) / reference_words if reference_words else float("nan"),
        "eval/reference_words": reference_words,
        "eval/clips": len(dataset),
    }, predictions


EVAL_CSV_FIELDS = [
    "audio_file_path",
    "ground_truth_transcript",
    "language_labels",
    "predicted_transcript",
    "predicted_language_labels",
    "wer",
    "reference_words",
    "word_errors",
    "loss",
    "token_loss",
    "language_id_loss",
]


def write_evaluation_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVAL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_html(path: Path, rows: list[dict[str, str]], run_name: str, best_eval_loss: float) -> None:
    """Write a static local review page; no HTTP server or CSV fetch is required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page_rows = []
    for row in rows:
        audio_path = Path(row["audio_file_path"]).resolve()
        page_rows.append({
            **row,
            "audio_src": os.path.relpath(audio_path, path.parent.resolve()).replace(os.sep, "/"),
        })
    payload = json.dumps(page_rows, ensure_ascii=False).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training evaluation — __RUN_NAME__</title>
<style>
body{margin:0;background:#10151f;color:#e9edf3;font:16px system-ui,-apple-system,sans-serif}.wrap{max-width:980px;margin:0 auto;padding:28px}
h1{margin:0 0 6px}.sub{color:#aebbd0;margin:0 0 20px}.controls{display:flex;gap:10px;align-items:center;margin:18px 0}.controls button{padding:8px 12px}.controls input{flex:1}
.card{background:#182131;border:1px solid #2d3c55;border-radius:10px;padding:20px}.label{display:block;color:#aebbd0;font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin:18px 0 5px}
.text{white-space:pre-wrap;line-height:1.45}.labels{white-space:pre-wrap;overflow:auto;background:#0d131e;padding:12px;border-radius:6px;font:12px ui-monospace,monospace}audio{width:100%;margin-top:8px}
</style></head><body><main class="wrap"><h1>Held-out training evaluation</h1><p class="sub">Run: __RUN_NAME__ · Best evaluation loss: __BEST_LOSS__ · <span id="count"></span></p>
<div class="controls"><label>Sort <select id="sort"><option value="original">Original order</option><option value="wer-desc">WER: highest first</option><option value="wer-asc">WER: lowest first</option><option value="loss-desc">Loss: highest first</option><option value="loss-asc">Loss: lowest first</option><option value="language_id_loss-desc">Language-ID loss: highest first</option><option value="language_id_loss-asc">Language-ID loss: lowest first</option><option value="token_loss-desc">Token loss: highest first</option><option value="token_loss-asc">Token loss: lowest first</option></select></label><button id="previous">← Previous</button><input id="position" type="range" min="0" value="0"><button id="next">Next →</button></div>
<section class="card"><strong id="title"></strong><audio id="audio" controls preload="metadata"></audio><span class="label">Clip metrics</span><div class="text" id="metrics"></div><span class="label">Ground-truth transcript</span><div class="text" id="reference"></div><span class="label">Predicted transcript</span><div class="text" id="prediction"></div><span class="label">Ground-truth language labels</span><div class="labels" id="labels"></div><span class="label">Predicted language labels</span><div class="labels" id="predictedLabels"></div></section>
</main><script>let rows=__DATA__;const originalRows=rows.slice();let current=0;const $=id=>document.getElementById(id);const format=value=>Number.isFinite(Number(value))?Number(value).toFixed(4):'not available';function show(){const r=rows[current];$('position').value=current;$('title').textContent=`Clip ${current+1} of ${rows.length}: ${r.audio_file_path}`;$('audio').src=r.audio_src;$('metrics').textContent=`WER: ${format(r.wer)} · Loss: ${format(r.loss)} · Language-ID loss: ${format(r.language_id_loss)} · Token loss: ${format(r.token_loss)}`;$('reference').textContent=r.ground_truth_transcript;$('prediction').textContent=r.predicted_transcript;$('labels').textContent=r.language_labels;$('predictedLabels').textContent=r.predicted_language_labels}function sortRows(){const [field,direction]=$('sort').value.split('-');rows=originalRows.slice();current=0;if(field!=='original'){const multiplier=direction==='desc'?-1:1;rows.sort((a,b)=>multiplier*((Number(a[field])||0)-(Number(b[field])||0)))}show()}$('position').max=Math.max(rows.length-1,0);$('count').textContent=`${rows.length} clips`;$('previous').onclick=()=>{current=(current+rows.length-1)%rows.length;show()};$('next').onclick=()=>{current=(current+1)%rows.length;show()};$('position').oninput=e=>{current=Number(e.target.value);show()};$('sort').onchange=sortRows;if(rows.length)show();</script></body></html>"""
    path.write_text(
        template.replace("__DATA__", payload)
        .replace("__RUN_NAME__", run_name)
        .replace("__BEST_LOSS__", f"{best_eval_loss:.5f}"),
        encoding="utf-8",
    )


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    config_args, _ = config_parser.parse_known_args()
    config_values, config_run_name = load_section(config_args.config, "train")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML named-run configuration file.")
    parser.add_argument("--manifest", type=Path, default=Path("processed/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed/05_train"))
    parser.add_argument("--base-model", default="small")
    parser.add_argument("--download-root", type=Path, default=Path("models/whisper"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, help="Use a deterministic subset of this many manifest rows for a preview run.")
    parser.add_argument("--max-train-steps", type=int, help="Stop after this many optimizer steps, while still running final evaluation and checkpointing.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--eval-set-size", type=int, default=EVAL_SET_SIZE)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", default="miami-whisper-token-lid")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis/training_run_eval"))
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()
    if config_run_name and args.wandb_run_name is None:
        args.wandb_run_name = config_run_name
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every_steps < 1 or args.eval_set_size < 1:
        parser.error("epochs, batch size, evaluation interval, and evaluation size must all be positive")
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples must be at least 2 so both train and test partitions are non-empty")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be positive")
    if not 0 < TRAIN_FRACTION < 1:
        raise RuntimeError("TRAIN_FRACTION must be between zero and one")

    load_dotenv(args.env_file)
    if args.wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
        parser.error(f"WANDB_API_KEY was not found in {args.env_file}; use --wandb-mode offline to test without uploading")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    rows = read_rows(args.manifest)
    if args.max_samples is not None:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
    train_rows, test_rows, split_unit = split_rows(rows, args.seed)
    if not train_rows or not test_rows:
        raise RuntimeError("The split produced an empty train or test partition")
    eval_rows = test_rows[: min(args.eval_set_size, len(test_rows))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train_split.csv", train_rows)
    write_manifest(args.output_dir / "test_split.csv", test_rows)

    model = load_token_lid_model(args.base_model, device=device, download_root=str(args.download_root))
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages, language="en", task="transcribe")
    train_dataset = MiamiDataset(train_rows, tokenizer, model.dims.n_mels)
    eval_dataset = MiamiDataset(eval_rows, tokenizer, model.dims.n_mels)
    collate = lambda items: collate_batch(items, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config={
            **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "device": str(device),
            "train_fraction": TRAIN_FRACTION,
            "split_unit": split_unit,
            "train_clips": len(train_dataset),
            "test_clips": len(test_rows),
            "held_out_eval_clips": len(eval_dataset),
            "language_labels": model.language_labels,
            "lid_loss_weight": LID_LOSS_WEIGHT,
        },
    )
    global_step = 0
    last_eval_step = -1
    reached_step_limit = False
    best_eval_loss = float("inf")
    best_eval_rows: list[dict[str, str]] = []

    def run_evaluation(step: int) -> dict[str, float]:
        nonlocal best_eval_loss, best_eval_rows
        metrics, evaluation_rows = evaluate(model, eval_loader, eval_dataset, device)
        write_evaluation_csv(args.output_dir / "eval_predictions.csv", evaluation_rows)
        metrics["eval/best_loss"] = min(best_eval_loss, metrics["eval/loss"])
        if metrics["eval/loss"] < best_eval_loss:
            best_eval_loss = metrics["eval/loss"]
            best_eval_rows = evaluation_rows
            save_token_lid_checkpoint(model, args.output_dir / "best.pt")
            write_evaluation_csv(args.output_dir / "best_eval_predictions.csv", best_eval_rows)
        metrics["eval/best_loss"] = best_eval_loss
        wandb.log(metrics, step=step)
        print(
            f"step {step}: loss={metrics['eval/loss']:.4f}, WER={metrics['eval/wer']:.3%}, "
            f"LID accuracy={metrics['eval/lid_accuracy']:.3%}, best loss={best_eval_loss:.4f}"
        )
        model.train()
        return metrics

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for batch in progress:
                optimizer.zero_grad(set_to_none=True)
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
                with autocast:
                    asr_loss, lid_loss, loss, correct, lid_count = model_losses(model, batch, device)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1
                train_metrics = {
                    "train/loss": float(loss.detach()),
                    "train/token_loss": float(asr_loss.detach()),
                    "train/language_id_loss": float(lid_loss.detach()),
                    "train/lid_accuracy": correct / lid_count if lid_count else float("nan"),
                    "epoch": epoch,
                }
                wandb.log(train_metrics, step=global_step)
                progress.set_postfix(loss=f"{train_metrics['train/loss']:.3f}")

                if global_step % args.eval_every_steps == 0:
                    run_evaluation(global_step)
                    last_eval_step = global_step
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    reached_step_limit = True
                    break
            save_token_lid_checkpoint(model, args.output_dir / "last.pt")
            if reached_step_limit:
                break

        if global_step != last_eval_step:
            run_evaluation(global_step)
        if not best_eval_rows:
            raise RuntimeError("No held-out evaluation rows were produced")
        save_token_lid_checkpoint(model, args.output_dir / "final.pt")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_label = re.sub(r"[^A-Za-z0-9_-]+", "_", run.name or args.wandb_run_name or "local")
        html_path = args.analysis_dir / f"{timestamp}_run_{run_label}.html"
        write_evaluation_html(html_path, best_eval_rows, run_label, best_eval_loss)
        run.summary["best_eval_loss"] = best_eval_loss
        run.summary["evaluation_csv"] = str(args.output_dir / "best_eval_predictions.csv")
        run.summary["evaluation_html"] = str(html_path)
        print(f"Wrote best held-out evaluation review to {html_path}")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
