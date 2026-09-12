#!/usr/bin/env python3
"""Fine-tune the local Whisper Small token-language fork on Miami clips."""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import gc
import heapq
import html
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

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

from models.whisper_lid import (
    load_token_lid_lora_adapter,
    load_token_lid_lora_model,
    save_token_lid_lora_adapter,
)
from models.whisper_lid.decode import (
    decode_batch_with_token_language,
    decode_with_token_language,
)
from models.whisper_lid.labels import IGNORE_INDEX, make_token_language_targets
from run_config import apply_defaults, load_section
from wer_metrics import normalise_for_wer, word_error_rate


# Training defaults. Command-line arguments may override these for an individual run.
TRAIN_FRACTION = 0.90
SPLIT_SEED = 1337
EVAL_SET_SIZE = 100
EVAL_EVERY_STEPS = 250
HIGHEST_TRAINING_LOSSES = 20
# Keep the auxiliary language-ID head available for diagnostics, but do not use
# its loss to update the model during the current ASR-only fine-tuning runs.
LID_LOSS_WEIGHT = 0.0


def ask_yes_no(question: str) -> bool:
    """Ask a strict interactive yes/no question, accepting y/yes and n/no."""
    while True:
        answer = input(f"{question} [y/N] ").strip().casefold()
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer Y or N.")


def resolve_existing_output(output_dir: Path) -> tuple[bool, bool]:
    """Return (resume weights, resume exact progress), or safely replace/abort."""
    if not output_dir.exists():
        return False, False
    checkpoint = output_dir / "best_lora"
    resume = ask_yes_no(
        f"This folder already exists: {output_dir}. Would you like to continue "
        "from the checkpoint saved under best_lora?"
    )
    if resume:
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Cannot resume because the checkpoint folder does not exist: {checkpoint}"
            )
        progress_path = checkpoint / "training_progress.json"
        exact = False
        if progress_path.is_file():
            exact = ask_yes_no(
                "training_progress.json exists. Continue from the exact saved epoch "
                "and batch position?"
            )
        else:
            print(
                "No training_progress.json was found; loading the adapter weights "
                "and starting with a newly shuffled training order."
            )
        return True, exact
    if ask_yes_no(f"Would you like to overwrite the training folder {output_dir}?"):
        resolved = output_dir.resolve()
        if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT not in resolved.parents:
            raise ValueError(f"Refusing to recursively replace unsafe output directory: {resolved}")
        shutil.rmtree(resolved)
        return False, False
    raise SystemExit("Training cancelled; the existing output folder was left unchanged.")


def save_training_progress(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    *,
    epoch: int,
    batch_in_epoch: int,
    batches_per_epoch: int,
    global_step: int,
    configured_epochs: int,
    best_eval_loss: float,
    best_eval_wer: float,
    batch_size: int,
    seed: int,
    training_clips: int,
    current_loss_heap: list[tuple[float, str]] | None = None,
    completed_high_losses: list[dict[str, Any]] | None = None,
) -> None:
    """Persist progress and optimizer state corresponding to saved adapter weights."""
    progress = {
        "epoch": epoch,
        "completed_epochs": epoch if batch_in_epoch >= batches_per_epoch else epoch - 1,
        "batch_in_epoch": batch_in_epoch,
        "batches_per_epoch": batches_per_epoch,
        "epoch_progress": batch_in_epoch / max(batches_per_epoch, 1),
        "global_step": global_step,
        "configured_epochs": configured_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "training_clips": training_clips,
        "best_eval_loss": best_eval_loss,
        "best_eval_wer": best_eval_wer,
        "current_loss_heap": current_loss_heap or [],
        "completed_high_losses": completed_high_losses or [],
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (checkpoint_dir / "training_progress.json").write_text(
        json.dumps(progress, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )


def load_training_progress(
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    progress = json.loads(
        (checkpoint_dir / "training_progress.json").read_text(encoding="utf-8")
    )
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"Exact resume requires optimizer state, but it is missing: {state_path}"
        )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)
    scaler.load_state_dict(state.get("scaler_state_dict", {}))
    return progress


def combined_training_loss(
    token_loss: torch.Tensor | float,
    language_loss: torch.Tensor | float,
) -> torch.Tensor | float:
    """Return the configured objective without linking a disabled LID loss."""
    if LID_LOSS_WEIGHT == 0:
        return token_loss
    return token_loss + LID_LOSS_WEIGHT * language_loss


def select_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_evaluation_memory(device: torch.device) -> None:
    """Release accelerator allocations left cached by autoregressive evaluation."""
    gc.collect()
    if device.type == "mps":
        # Finish outstanding kernels before releasing cached unified memory.
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


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


def filter_jember_001_first_ten(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Select nested Jember-001 windows ending at TSV rows 1 through 10."""
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
    """Use the final windows as a stable holdout for the diagnostic run."""
    if not 0 < test_size < len(rows):
        raise ValueError("Preview test size must leave non-empty train and test sets")
    return rows[:-test_size], rows[-test_size:]


def jember_recording_key(row: dict[str, str]) -> str:
    """Identify the source Jember recording shared by windows/augmentations."""
    for field in ("augmentation_source_audio", "audio_path"):
        match = re.search(r"jember_(\d+)_", Path(row.get(field, "")).stem)
        if match:
            return match.group(1)
    source = row.get("source_audio", "").strip()
    if source:
        return Path(source).stem
    raise ValueError(
        f"Cannot identify Jember recording for {row.get('audio_path', '<unknown>')}"
    )


def balance_jember_training_percentage(
    rows: list[dict[str, str]], percentage: float, seed: int
) -> tuple[list[dict[str, str]], int, int]:
    """Remove whole Jember recordings to approach a target row percentage."""
    jember_rows = [row for row in rows if row.get("dataset") == "jember"]
    development_rows = [
        row for row in rows
        if row.get("dataset") == "indonesian_development"
    ]
    other_rows = [
        row for row in rows
        if row.get("dataset") not in {"jember", "indonesian_development"}
    ]
    if other_rows:
        raise ValueError(
            "jember_percentage only supports Jember and Indonesian-development rows"
        )
    original_count = len(jember_rows)
    development_count = len(development_rows)
    if not original_count or percentage == 100:
        return rows, original_count, original_count
    if not development_count:
        if percentage == 100:
            return rows, original_count, original_count
        raise ValueError(
            "Cannot set jember_percentage without Indonesian-development training rows"
        )

    current_percentage = 100 * original_count / (original_count + development_count)
    if current_percentage <= percentage:
        # Removing Jember can only lower its share, never raise it.
        return rows, original_count, original_count

    target_count = (
        0 if percentage == 0
        else round(percentage * development_count / (100 - percentage))
    )
    by_recording: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in jember_rows:
        by_recording[jember_recording_key(row)].append(row)
    recording_ids = list(by_recording)
    random.Random(seed).shuffle(recording_ids)
    retained_recordings = set(recording_ids)
    retained_count = original_count
    for recording_id in recording_ids:
        reduced_count = retained_count - len(by_recording[recording_id])
        if abs(reduced_count - target_count) <= abs(retained_count - target_count):
            retained_recordings.remove(recording_id)
            retained_count = reduced_count

    filtered = [
        row for row in rows
        if row.get("dataset") != "jember"
        or jember_recording_key(row) in retained_recordings
    ]
    return filtered, original_count, retained_count


def composition_bucket(row: dict[str, str]) -> str:
    """Map source dataset labels into Jember, Common Voice, or development."""
    dataset = row.get("dataset", "")
    if dataset == "jember":
        return "jember"
    if dataset in {"cv_indonesian", "cv_javanese", "commonvoice_code_switch"}:
        return "commonvoice"
    if dataset == "indonesian_development":
        return "indonesian_development"
    raise ValueError(f"Unsupported dataset for composition: {dataset!r}")


def balance_training_composition(rows: list[dict[str, str]], jember: float, commonvoice: float, seed: int) -> list[dict[str, str]]:
    """Downsample the three source buckets to the requested row composition."""
    requested = {"jember": jember, "commonvoice": commonvoice, "indonesian_development": 100 - jember - commonvoice}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[composition_bucket(row)].append(row)
    active = {name: percent for name, percent in requested.items() if percent > 0}
    missing = [name for name in active if not groups[name]]
    if missing:
        raise ValueError(f"Requested composition has no rows for: {', '.join(missing)}")
    total = min(len(groups[name]) * 100 / percent for name, percent in active.items())
    selected: list[dict[str, str]] = []
    for offset, name in enumerate(("jember", "commonvoice", "indonesian_development")):
        candidates = groups[name][:]
        random.Random(seed + offset).shuffle(candidates)
        selected.extend(candidates[:round(total * requested[name] / 100)])
    random.Random(seed).shuffle(selected)
    return selected


def split_balanced_dataset_evaluation(
    rows: list[dict[str, str]],
    development_count: int,
    jember_count: int,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Hold out deterministic clip samples from both Indonesian datasets."""
    # Evaluation stays clean. When a selected clean clip has an augmented twin,
    # exclude the twin from training as well to prevent direct leakage.
    clean_rows = [
        row for row in rows
        if row.get("augmented", "false").casefold() != "true"
        and row.get("speed_augmented", "false").casefold() != "true"
    ]
    development = [row for row in clean_rows if row.get("dataset") == "indonesian_development"]
    jember = [row for row in clean_rows if row.get("dataset") == "jember"]
    if len(development) < development_count or len(jember) < jember_count:
        raise ValueError(
            "Balanced evaluation requested "
            f"{development_count} Indonesian-development and {jember_count} Jember clips, "
            f"but only {len(development)} and {len(jember)} are available"
        )
    rng = random.Random(seed)
    rng.shuffle(development)
    rng.shuffle(jember)
    test = development[:development_count] + jember[:jember_count]
    test_paths = {row["audio_path"] for row in test}
    train = [
        row for row in rows
        if row["audio_path"] not in test_paths
        and row.get("augmentation_source_audio", row["audio_path"]) not in test_paths
    ]
    if not train:
        raise ValueError("Balanced evaluation split left no training clips")
    return train, test


def augmentation_family_key(row: dict[str, str]) -> str:
    """Return the clean source shared by an original and all its augmentations."""
    return row.get("augmentation_source_audio", "").strip() or row["audio_path"]


def split_proportional_augmented_evaluation(
    rows: list[dict[str, str]],
    total_clips: int,
    jember_proportion: float,
    commonvoice_proportion: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int, int]:
    """Select an exact mixed-dataset eval set from the augmented row pool.

    At most one row is selected from each augmentation family. Every other
    member of a selected family is removed from training to prevent a clean
    clip and its noisy copies leaking across the split.
    """
    jember_count = round(total_clips * jember_proportion / 100)
    commonvoice_count = round(total_clips * commonvoice_proportion / 100)
    development_count = total_clips - jember_count - commonvoice_count
    requested = {
        "jember": jember_count,
        "commonvoice": commonvoice_count,
        "indonesian_development": development_count,
    }
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    for dataset, count in requested.items():
        if count == 0:
            continue
        candidates = [row for row in rows if composition_bucket(row) == dataset]
        rng.shuffle(candidates)
        dataset_selected: list[dict[str, str]] = []
        seen_families: set[str] = set()
        for row in candidates:
            family = augmentation_family_key(row)
            if family in seen_families:
                continue
            seen_families.add(family)
            dataset_selected.append(row)
            if len(dataset_selected) == count:
                break
        if len(dataset_selected) < count:
            raise ValueError(
                f"Proportional evaluation requested {count} {dataset} clips, "
                f"but only {len(dataset_selected)} independent augmentation "
                "families are available"
            )
        selected.extend(dataset_selected)

    selected_families = {augmentation_family_key(row) for row in selected}
    train = [
        row for row in rows
        if augmentation_family_key(row) not in selected_families
    ]
    if not train:
        raise ValueError("Proportional evaluation split left no training clips")
    return train, selected, jember_count, commonvoice_count, development_count


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


def weighted_asr_loss(
    token_logits: torch.Tensor,
    targets: torch.Tensor,
    eos_token_id: int,
    eos_loss_weight: float,
    *,
    per_example: bool = False,
) -> torch.Tensor:
    """Cross entropy with explicit weighting for the end-of-transcript target."""
    losses = F.cross_entropy(
        token_logits.transpose(1, 2),
        targets,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    valid = targets.ne(IGNORE_INDEX)
    weights = torch.ones_like(losses)
    weights = torch.where(targets.eq(eos_token_id), eos_loss_weight, weights)
    weights = weights * valid
    if per_example:
        return (losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    return (losses * weights).sum() / weights.sum().clamp_min(1)


def model_losses(
    model,
    batch: dict[str, Any],
    device: torch.device,
    eos_token_id: int,
    eos_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    mels = batch["mels"].to(device)
    inputs = batch["inputs"].to(device)
    asr_targets = batch["asr_targets"].to(device)
    lid_targets = batch["lid_targets"].to(device)
    audio_features = model.encoder(mels)
    token_logits, language_logits = model.logits_with_language(inputs, audio_features)
    asr_loss = weighted_asr_loss(
        token_logits, asr_targets, eos_token_id, eos_loss_weight
    )
    per_example_losses = weighted_asr_loss(
        token_logits,
        asr_targets,
        eos_token_id,
        eos_loss_weight,
        per_example=True,
    )
    valid_lid = lid_targets.ne(IGNORE_INDEX)
    lid_count = int(valid_lid.sum().item())
    if lid_count:
        lid_loss = F.cross_entropy(language_logits.transpose(1, 2), lid_targets, ignore_index=IGNORE_INDEX)
        correct = int((language_logits.argmax(dim=-1)[valid_lid] == lid_targets[valid_lid]).sum().item())
    else:
        lid_loss = torch.zeros((), device=device)
        correct = 0
    return (
        asr_loss,
        lid_loss,
        combined_training_loss(asr_loss, lid_loss),
        correct,
        lid_count,
        per_example_losses,
    )


@torch.no_grad()
def per_example_loss_values(
    model,
    batch: dict[str, Any],
    device: torch.device,
    eos_token_id: int,
    eos_loss_weight: float,
) -> list[dict[str, float]]:
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

    token_losses = weighted_asr_loss(
        token_logits,
        asr_targets,
        eos_token_id,
        eos_loss_weight,
        per_example=True,
    )
    language_losses = mean_per_clip(language_logits, lid_targets)
    values = []
    for token_loss, language_loss in zip(token_losses.tolist(), language_losses.tolist()):
        values.append({
            "token_loss": token_loss,
            "language_id_loss": language_loss,
            "loss": combined_training_loss(token_loss, language_loss),
        })
    return values


@torch.no_grad()
def evaluate(
    model,
    loss_loader: DataLoader,
    decode_loader: DataLoader,
    dataset: MiamiDataset,
    device: torch.device,
    eos_token_id: int,
    eos_loss_weight: float,
    language: str | None = None,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()
    total_asr_loss = total_lid_loss = 0.0
    batches = correct_lid = total_lid = 0
    losses_by_audio_path: dict[str, dict[str, float]] = {}
    for batch in tqdm(loss_loader, desc="Held-out loss", leave=False):
        asr_loss, lid_loss, _, correct, count, _ = model_losses(
            model, batch, device, eos_token_id, eos_loss_weight
        )
        total_asr_loss += float(asr_loss)
        total_lid_loss += float(lid_loss)
        batches += 1
        correct_lid += correct
        total_lid += count
        per_clip_losses = per_example_loss_values(
            model, batch, device, eos_token_id, eos_loss_weight
        )
        for row, values in zip(batch["rows"], per_clip_losses):
            losses_by_audio_path[row["audio_path"]] = values

    substitutions = deletions = insertions = reference_words = 0
    predictions: list[dict[str, str]] = []
    for batch in tqdm(decode_loader, desc="Held-out WER", leave=False):
        results = decode_batch_with_token_language(
            model, batch["mels"], language=language
        )
        for row, result in zip(batch["rows"], results):
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
        "eval/loss": combined_training_loss(token_loss, language_id_loss),
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

# One row per named run.  This is deliberately separate from the clip-level
# evaluation CSVs so that runs can be compared without loading their examples.
RUN_RESULTS_FIELDS = [
    "run_name",
    "evaluation_step",
    "evaluated_at",
    "evaluation_csv",
    "clips",
    "reference_words",
    "word_errors",
    "wer",
    "loss",
    "token_loss",
    "language_id_loss",
    "lid_accuracy",
]
HIGH_LOSS_FIELDS = [
    "audio_path",
    "epoch",
    "loss",
    "predicted_transcript",
    "actual_transcript",
]


def write_evaluation_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=EVAL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_highest_training_losses(
    path: Path, rows: list[dict[str, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=HIGH_LOSS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def transcribe_highest_training_losses(
    candidates: list[dict[str, Any]],
    rows_by_audio_path: dict[str, dict[str, str]],
    model,
    n_mels: int,
    language: str,
    device: torch.device,
) -> list[dict[str, str]]:
    """Decode each unique selected clip once using the final trained model."""
    predictions: dict[str, str] = {}
    unique_paths = list(dict.fromkeys(item["audio_path"] for item in candidates))
    model.eval()
    for audio_path in tqdm(unique_paths, desc="Highest-loss transcripts"):
        audio = whisper.pad_or_trim(whisper.load_audio(audio_path))
        mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        predictions[audio_path] = decode_with_token_language(
            model, mel, language=language
        ).text
    model.train()
    release_evaluation_memory(device)
    output = []
    for item in sorted(
        candidates, key=lambda value: (int(value["epoch"]), -float(value["loss"]))
    ):
        audio_path = str(item["audio_path"])
        output.append({
            "audio_path": audio_path,
            "epoch": str(item["epoch"]),
            "loss": f"{float(item['loss']):.10f}",
            "predicted_transcript": predictions[audio_path],
            "actual_transcript": rows_by_audio_path[audio_path]["transcript"],
        })
    return output


def upsert_run_results(
    path: Path,
    *,
    run_name: str,
    evaluation_step: int,
    evaluation_csv: Path,
    metrics: dict[str, float],
) -> None:
    """Record the best held-out aggregate metrics for one named training run."""
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            existing = list(csv.DictReader(file))
    record = {
        "run_name": run_name,
        "evaluation_step": str(evaluation_step),
        "evaluated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evaluation_csv": str(evaluation_csv),
        "clips": str(int(metrics["eval/clips"])),
        "reference_words": str(int(metrics["eval/reference_words"])),
        "word_errors": str(round(metrics["eval/wer"] * metrics["eval/reference_words"])),
        "wer": f"{metrics['eval/wer']:.10f}",
        "loss": f"{metrics['eval/loss']:.10f}",
        "token_loss": f"{metrics['eval/token_loss']:.10f}",
        "language_id_loss": f"{metrics['eval/language_id_loss']:.10f}",
        "lid_accuracy": f"{metrics['eval/lid_accuracy']:.10f}",
    }
    # A named configuration represents one comparable run; replace its row if
    # the run is resumed and finds a better checkpoint.
    existing = [row for row in existing if row.get("run_name") != run_name]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RUN_RESULTS_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(record)


def write_evaluation_html(
    path: Path,
    rows: list[dict[str, str]],
    run_name: str,
    best_eval_loss: float,
    highest_training_losses: list[dict[str, str]],
) -> None:
    """Write a static local review page; no HTTP server or CSV fetch is required."""
    path.parent.mkdir(parents=True, exist_ok=True)
    page_rows = []
    for row in rows:
        audio_path = Path(row["audio_file_path"]).resolve()
        reference_words = normalise_for_wer(row["ground_truth_transcript"])
        predicted_words = normalise_for_wer(row["predicted_transcript"])
        reference_diff: list[str] = []
        prediction_diff: list[str] = []
        for operation, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, reference_words, predicted_words
        ).get_opcodes():
            reference_text = html.escape(" ".join(reference_words[i1:i2]))
            prediction_text = html.escape(" ".join(predicted_words[j1:j2]))
            css_class = {
                "equal": "correct",
                "replace": "substitution",
                "delete": "deletion",
                "insert": "insertion",
            }[operation]
            if reference_text:
                reference_diff.append(f'<span class="{css_class}">{reference_text}</span>')
            if prediction_text:
                prediction_diff.append(f'<span class="{css_class}">{prediction_text}</span>')
        page_rows.append({
            **row,
            "audio_src": os.path.relpath(audio_path, path.parent.resolve()).replace(os.sep, "/"),
            "reference_diff": " ".join(reference_diff),
            "prediction_diff": " ".join(prediction_diff),
        })
    payload = json.dumps(page_rows, ensure_ascii=False).replace("</", "<\\/")
    training_page_rows = [
        {
            **row,
            "audio_src": os.path.relpath(
                Path(row["audio_path"]).resolve(), path.parent.resolve()
            ).replace(os.sep, "/"),
        }
        for row in highest_training_losses
    ]
    training_payload = json.dumps(
        training_page_rows, ensure_ascii=False
    ).replace("</", "<\\/")
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training evaluation — __RUN_NAME__</title>
<style>
body{margin:0;background:#10151f;color:#e9edf3;font:16px system-ui,-apple-system,sans-serif}.wrap{max-width:980px;margin:0 auto;padding:28px}
h1{margin:0 0 6px}.sub{color:#aebbd0;margin:0 0 20px}.controls{display:flex;gap:10px;align-items:center;margin:18px 0}.controls button{padding:8px 12px}.controls input{flex:1}
.card{background:#182131;border:1px solid #2d3c55;border-radius:10px;padding:20px;margin-bottom:28px}.label{display:block;color:#aebbd0;font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;margin:18px 0 5px}
.text{white-space:pre-wrap;line-height:1.55}.diff span{display:inline-block;padding:1px 3px;margin:1px;border-radius:3px}.correct{color:#c9d4e5}.substitution{background:#71313b;color:#fff}.deletion{background:#71313b;color:#fff;text-decoration:line-through}.insertion{background:#24577a;color:#fff}.legend{color:#aebbd0;font-size:.85rem}.labels{white-space:pre-wrap;overflow:auto;background:#0d131e;padding:12px;border-radius:6px;font:12px ui-monospace,monospace}audio{width:100%;margin-top:8px}
</style></head><body><main class="wrap"><h1>Held-out training evaluation</h1><p class="sub">Run: __RUN_NAME__ · Evaluation loss: __BEST_LOSS__ · <span id="count"></span></p>
<div class="controls"><label>Sort <select id="sort"><option value="original">Original order</option><option value="wer-desc">WER: highest first</option><option value="wer-asc">WER: lowest first</option><option value="loss-desc">Loss: highest first</option><option value="loss-asc">Loss: lowest first</option><option value="language_id_loss-desc">Language-ID loss: highest first</option><option value="language_id_loss-asc">Language-ID loss: lowest first</option><option value="token_loss-desc">Token loss: highest first</option><option value="token_loss-asc">Token loss: lowest first</option></select></label><button id="previous">← Previous</button><input id="position" type="range" min="0" value="0"><button id="next">Next →</button></div>
<section class="card"><strong id="title"></strong><audio id="audio" controls preload="metadata"></audio><span class="label">Clip metrics</span><div class="text" id="metrics"></div><span class="label">Scored word differences</span><div class="legend">Red = substitution/deletion · Blue = insertion</div><div class="text diff" id="referenceDiff"></div><div class="text diff" id="predictionDiff"></div><span class="label">Ground-truth transcript</span><div class="text" id="reference"></div><span class="label">Predicted transcript</span><div class="text" id="prediction"></div><span class="label">Ground-truth language labels</span><div class="labels" id="labels"></div><span class="label">Predicted language labels</span><div class="labels" id="predictedLabels"></div></section>
</main><script>let rows=__DATA__;const originalRows=rows.slice();let current=0;const $=id=>document.getElementById(id);const format=value=>Number.isFinite(Number(value))?Number(value).toFixed(4):'not available';function show(){const r=rows[current];$('position').value=current;$('title').textContent=`Clip ${current+1} of ${rows.length}: ${r.audio_file_path}`;$('audio').src=r.audio_src;$('metrics').textContent=`WER: ${format(r.wer)} · Word errors: ${r.word_errors}/${r.reference_words} · Loss: ${format(r.loss)} · Token loss: ${format(r.token_loss)}`;$('referenceDiff').innerHTML=`Reference: ${r.reference_diff}`;$('predictionDiff').innerHTML=`Prediction: ${r.prediction_diff}`;$('reference').textContent=r.ground_truth_transcript;$('prediction').textContent=r.predicted_transcript;$('labels').textContent=r.language_labels;$('predictedLabels').textContent=r.predicted_language_labels}function sortRows(){const [field,direction]=$('sort').value.split('-');rows=originalRows.slice();current=0;if(field!=='original'){const multiplier=direction==='desc'?-1:1;rows.sort((a,b)=>multiplier*((Number(a[field])||0)-(Number(b[field])||0)))}show()}$('position').max=Math.max(rows.length-1,0);$('count').textContent=`${rows.length} clips`;$('previous').onclick=()=>{current=(current+rows.length-1)%rows.length;show()};$('next').onclick=()=>{current=(current+1)%rows.length;show()};$('position').oninput=e=>{current=Number(e.target.value);show()};$('sort').onchange=sortRows;if(rows.length)show();</script></body></html>"""
    training_section = """
<h2>Highest training losses</h2><p class="sub">The 20 highest-loss examples retained independently for every epoch.</p>
<div class="controls"><label>Epoch <select id="lossEpoch"></select></label><button id="lossPrevious">← Previous</button><input id="lossPosition" type="range" min="0" value="0"><button id="lossNext">Next →</button></div>
<section class="card"><strong id="lossTitle"></strong><audio id="lossAudio" controls preload="metadata"></audio><span class="label">Training loss</span><div class="text" id="trainingLoss"></div><span class="label">Actual transcript</span><div class="text" id="lossActual"></div><span class="label">Final-model predicted transcript</span><div class="text" id="lossPredicted"></div></section>
"""
    training_script = """
const trainingRows=__TRAINING_DATA__;let lossRows=[],lossCurrent=0;
const epochs=[...new Set(trainingRows.map(row=>row.epoch))].sort((a,b)=>Number(a)-Number(b));
$('lossEpoch').innerHTML=epochs.map(epoch=>`<option value="${epoch}">Epoch ${epoch}</option>`).join('');
function selectLossEpoch(){lossRows=trainingRows.filter(row=>row.epoch===$('lossEpoch').value);lossCurrent=0;$('lossPosition').max=Math.max(lossRows.length-1,0);showLoss()}
function showLoss(){if(!lossRows.length){$('lossTitle').textContent='No training-loss examples available';return}const row=lossRows[lossCurrent];$('lossPosition').value=lossCurrent;$('lossTitle').textContent=`Example ${lossCurrent+1} of ${lossRows.length}: ${row.audio_path}`;$('lossAudio').src=row.audio_src;$('trainingLoss').textContent=`Epoch ${row.epoch} · Loss: ${format(row.loss)}`;$('lossActual').textContent=row.actual_transcript;$('lossPredicted').textContent=row.predicted_transcript}
$('lossEpoch').onchange=selectLossEpoch;$('lossPrevious').onclick=()=>{lossCurrent=(lossCurrent+lossRows.length-1)%lossRows.length;showLoss()};$('lossNext').onclick=()=>{lossCurrent=(lossCurrent+1)%lossRows.length;showLoss()};$('lossPosition').oninput=event=>{lossCurrent=Number(event.target.value);showLoss()};if(epochs.length)selectLossEpoch();else showLoss();
"""
    template = template.replace("</main><script>", training_section + "</main><script>")
    template = template.replace(
        "</script></body></html>", training_script + "</script></body></html>"
    )
    path.write_text(
        template.replace("__DATA__", payload)
        .replace("__TRAINING_DATA__", training_payload)
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
    parser.add_argument("--manifest", type=Path, default=Path("processed_indonesia/train.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_indonesia/05_train"))
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate an existing LoRA adapter without resuming or running training.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        help="Saved LoRA adapter directory to load when --eval-only is enabled.",
    )
    parser.add_argument("--base-model", default="small")
    parser.add_argument(
        "--language", default="en",
        help="Whisper language code used in the training prompt (use 'id' for Indonesian).",
    )
    parser.add_argument(
        "--jember-001-first-ten", action="store_true",
        help=(
            "Restrict the run to jember_001_0001-0001 through "
            "jember_001_0001-0010 and use a deterministic 8/2 diagnostic split."
        ),
    )
    parser.add_argument(
        "--eval-indonesian-dev-clips", type=int, default=0,
        help="Hold out exactly this many Indonesian-development clips.",
    )
    parser.add_argument(
        "--eval-jember-clips", type=int, default=0,
        help="Hold out exactly this many Jember clips.",
    )
    parser.add_argument(
        "--eval-total-clips",
        type=int,
        help="Total rows in the proportional post-augmentation evaluation set.",
    )
    parser.add_argument(
        "--eval-jember-proportion",
        type=float,
        help=(
            "Percentage of --eval-total-clips drawn from Jember; for example, "
            "60 means 60%% Jember and 40%% Indonesian-dev."
        ),
    )
    parser.add_argument("--eval-commonvoice-proportion", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--download-root", type=Path, default=Path("models/whisper"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--jember-percentage",
        type=float,
        help=(
            "Target percentage of training rows contributed by Jember. Whole "
            "Jember recordings are removed deterministically to approach it. "
            "Omit this argument to retain the complete training split."
        ),
    )
    parser.add_argument("--commonvoice-percentage", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, help="Use a deterministic subset of this many manifest rows for a preview run.")
    parser.add_argument("--max-train-steps", type=int, help="Stop after this many optimizer steps, while still running final evaluation and checkpointing.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--eos-loss-weight", type=float, default=1.0,
        help="Relative cross-entropy weight for the end-of-transcript token.",
    )
    parser.add_argument("--eval-every-steps", type=int, default=EVAL_EVERY_STEPS)
    parser.add_argument("--eval-set-size", type=int, default=EVAL_SET_SIZE)
    parser.add_argument(
        "--eval-decode-batch-size",
        type=int,
        default=1,
        help="Batch size for autoregressive transcript generation during evaluation.",
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", default="miami-whisper-token-lid")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("analysis_indonesia/training_run_eval"))
    parser.add_argument(
        "--run-results-csv",
        type=Path,
        default=Path("analysis_indonesia/training_run_results.csv"),
        help="Run-level aggregate evaluation table, updated when this run improves.",
    )
    apply_defaults(parser, config_values, config_args.config)
    args = parser.parse_args()
    if config_run_name and args.wandb_run_name is None:
        args.wandb_run_name = config_run_name
    if args.epochs < 1 or args.batch_size < 1 or args.eval_every_steps < 1 or args.eval_set_size < 1:
        parser.error("epochs, batch size, evaluation interval, and evaluation size must all be positive")
    if args.eval_decode_batch_size < 1:
        parser.error("--eval-decode-batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 2:
        parser.error("--max-samples must be at least 2 so both train and test partitions are non-empty")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max-train-steps must be positive")
    if args.eos_loss_weight <= 0:
        parser.error("--eos-loss-weight must be positive")
    if args.eval_only and args.adapter_dir is None:
        parser.error("--adapter-dir is required when --eval-only is enabled")
    if not args.eval_only and args.adapter_dir is not None:
        parser.error("--adapter-dir can only be used with --eval-only")
    if args.jember_percentage is not None and not 0 <= args.jember_percentage <= 100:
        parser.error("--jember-percentage must be between 0 and 100")
    if not 0 <= args.commonvoice_percentage <= 100:
        parser.error("--commonvoice-percentage must be between 0 and 100")
    if args.jember_percentage is not None and args.jember_percentage + args.commonvoice_percentage > 100:
        parser.error("--jember-percentage plus --commonvoice-percentage cannot exceed 100")
    if args.eval_indonesian_dev_clips < 0 or args.eval_jember_clips < 0:
        parser.error("Balanced evaluation clip counts cannot be negative")
    proportional_evaluation_requested = (
        args.eval_total_clips is not None
        or args.eval_jember_proportion is not None
    )
    if proportional_evaluation_requested and (
        args.eval_total_clips is None or args.eval_jember_proportion is None
    ):
        parser.error(
            "Use --eval-total-clips and --eval-jember-proportion together"
        )
    if args.eval_total_clips is not None and args.eval_total_clips < 1:
        parser.error("--eval-total-clips must be at least 1")
    if (
        args.eval_jember_proportion is not None
        and not 0 <= args.eval_jember_proportion <= 100
    ):
        parser.error("--eval-jember-proportion must be between 0 and 100")
    if not 0 <= args.eval_commonvoice_proportion <= 100:
        parser.error("--eval-commonvoice-proportion must be between 0 and 100")
    if args.eval_jember_proportion is not None and args.eval_jember_proportion + args.eval_commonvoice_proportion > 100:
        parser.error("Evaluation Jember and Common Voice proportions cannot exceed 100")
    if bool(args.eval_indonesian_dev_clips) != bool(args.eval_jember_clips):
        parser.error(
            "Use --eval-indonesian-dev-clips and --eval-jember-clips together"
        )
    if args.jember_001_first_ten and args.eval_indonesian_dev_clips:
        parser.error(
            "--jember-001-first-ten cannot be combined with the balanced evaluation split"
        )
    if proportional_evaluation_requested and (
        args.jember_001_first_ten
        or args.eval_indonesian_dev_clips
    ):
        parser.error(
            "The proportional augmented evaluation split cannot be combined "
            "with another explicit evaluation split"
        )
    if not 0 < TRAIN_FRACTION < 1:
        raise RuntimeError("TRAIN_FRACTION must be between zero and one")

    if args.eval_only:
        resume_adapter = resume_exact = False
        if not args.adapter_dir.is_dir():
            parser.error(f"Adapter directory not found: {args.adapter_dir}")
    else:
        resume_adapter, resume_exact = resolve_existing_output(args.output_dir)

    load_dotenv(args.env_file)
    if args.wandb_mode == "online" and not os.getenv("WANDB_API_KEY"):
        parser.error(f"WANDB_API_KEY was not found in {args.env_file}; use --wandb-mode offline to test without uploading")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    rows = read_rows(args.manifest)
    if args.jember_001_first_ten:
        rows = filter_jember_001_first_ten(rows)
        train_rows, test_rows = split_preview_rows(rows)
        split_unit = "diagnostic Jember windows (first eight train, final two test)"
    elif proportional_evaluation_requested:
        (
            train_rows,
            test_rows,
            evaluation_jember_count,
            evaluation_commonvoice_count,
            evaluation_development_count,
        ) = split_proportional_augmented_evaluation(
            rows,
            args.eval_total_clips,
            args.eval_jember_proportion,
            args.eval_commonvoice_proportion,
            args.seed,
        )
        split_unit = (
            f"post-augmentation rows ({evaluation_jember_count} Jember, "
            f"{evaluation_commonvoice_count} Common Voice, "
            f"{evaluation_development_count} Indonesian-development; "
            "augmentation families held entirely out)"
        )
    elif args.eval_indonesian_dev_clips:
        train_rows, test_rows = split_balanced_dataset_evaluation(
            rows,
            args.eval_indonesian_dev_clips,
            args.eval_jember_clips,
            args.seed,
        )
        split_unit = (
            f"balanced clips ({args.eval_indonesian_dev_clips} Indonesian-development, "
            f"{args.eval_jember_clips} Jember)"
        )
    elif args.max_samples is not None:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
        train_rows, test_rows, split_unit = split_rows(rows, args.seed)
    else:
        train_rows, test_rows, split_unit = split_rows(rows, args.seed)
    if args.jember_percentage is not None and args.commonvoice_percentage:
        train_rows = balance_training_composition(train_rows, args.jember_percentage, args.commonvoice_percentage, args.seed + 10_000)
        print(f"Training composition selected: Jember={args.jember_percentage:g}%, Common Voice={args.commonvoice_percentage:g}%, Indonesian-dev={100-args.jember_percentage-args.commonvoice_percentage:g}%")
    elif args.jember_percentage is not None:
        train_rows, original_jember_count, retained_jember_count = (
            balance_jember_training_percentage(
                train_rows, args.jember_percentage, args.seed + 10_000
            )
        )
        retained_development_count = sum(
            row.get("dataset") == "indonesian_development" for row in train_rows
        )
        achieved_percentage = (
            100 * retained_jember_count
            / max(retained_jember_count + retained_development_count, 1)
        )
        print(
            f"Jember percentage filter retained {retained_jember_count}/"
            f"{original_jember_count} Jember rows; achieved "
            f"{achieved_percentage:.2f}% (target {args.jember_percentage:g}%)"
        )
    if not train_rows or not test_rows:
        raise RuntimeError("The split produced an empty train or test partition")
    dataset_labels = (
        "indonesian_development", "indonesian_dev", "development",
        "cv_indonesian", "cv_javanese", "commonvoice_code_switch",
    )
    distribution = Counter(row.get("dataset", "") for row in train_rows)
    print("Training dataset distribution before model initialization:")
    for label in dataset_labels:
        count = distribution[label]
        print(f"  {label}: {count} ({100 * count / len(train_rows):.2f}%)")
    other = {label: count for label, count in distribution.items() if label not in dataset_labels}
    for label, count in sorted(other.items()):
        print(f"  {label}: {count} ({100 * count / len(train_rows):.2f}%)")
    synthetic_count = distribution["commonvoice_code_switch"]
    nonsynthetic_count = len(train_rows) - synthetic_count
    commonvoice_count = sum(
        distribution[label]
        for label in ("cv_indonesian", "cv_javanese", "commonvoice_code_switch")
    )
    print(
        "Synthetic composition: "
        f"synthetic={synthetic_count} ({100 * synthetic_count / len(train_rows):.2f}% of training), "
        f"non-synthetic={nonsynthetic_count} ({100 * nonsynthetic_count / len(train_rows):.2f}% of training); "
        f"synthetic Common Voice={100 * synthetic_count / commonvoice_count:.2f}%"
        if commonvoice_count else
        "Synthetic composition: synthetic=0; no Common Voice rows"
    )
    eval_rows = (
        test_rows
        if proportional_evaluation_requested
        else test_rows[: min(args.eval_set_size, len(test_rows))]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "train_split.csv", train_rows)
    write_manifest(args.output_dir / "test_split.csv", test_rows)

    if args.eval_only:
        model = load_token_lid_lora_adapter(
            args.adapter_dir,
            device=device,
            download_root=str(args.download_root),
        )
        print(f"Loaded LoRA adapter for evaluation from {args.adapter_dir}")
    elif resume_adapter:
        model = load_token_lid_lora_adapter(
            args.output_dir / "best_lora",
            device=device,
            download_root=str(args.download_root),
        )
        print(f"Loaded LoRA adapter from {args.output_dir / 'best_lora'}")
    else:
        model = load_token_lid_lora_model(
            args.base_model,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            device=device,
            download_root=str(args.download_root),
        )
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=args.language,
        task="transcribe",
    )
    train_dataset = MiamiDataset(train_rows, tokenizer, model.dims.n_mels)
    eval_dataset = MiamiDataset(eval_rows, tokenizer, model.dims.n_mels)
    collate = lambda items: collate_batch(items, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    eval_decode_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_decode_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    if args.eval_only:
        evaluation_jember_rows = sum(
            row.get("dataset") == "jember" for row in eval_rows
        )
        evaluation_development_rows = sum(
            row.get("dataset") == "indonesian_development" for row in eval_rows
        )
        print(
            "Evaluation set composition: "
            f"Jember={evaluation_jember_rows}, "
            f"Indonesian-dev={evaluation_development_rows}"
        )
        metrics, evaluation_rows = evaluate(
            model,
            eval_loader,
            eval_decode_loader,
            eval_dataset,
            device,
            tokenizer.eot,
            args.eos_loss_weight,
            language=args.language,
        )
        evaluation_csv = args.output_dir / "eval_predictions.csv"
        metrics_path = args.output_dir / "evaluation_metrics.json"
        write_evaluation_csv(evaluation_csv, evaluation_rows)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )
        run_label = str(config_run_name or args.wandb_run_name or args.adapter_dir.parent.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_label = re.sub(r"[^A-Za-z0-9_-]+", "_", run_label)
        html_path = args.analysis_dir / f"{timestamp}_run_{safe_run_label}.html"
        write_evaluation_html(
            html_path,
            evaluation_rows,
            run_label,
            metrics["eval/loss"],
            [],
        )
        print(
            f"Evaluation only: clips={int(metrics['eval/clips'])}, "
            f"loss={metrics['eval/loss']:.4f}, WER={metrics['eval/wer']:.3%}, "
            f"LID accuracy={metrics['eval/lid_accuracy']:.3%}"
        )
        print(f"Wrote evaluation predictions to {evaluation_csv}")
        print(f"Wrote evaluation metrics to {metrics_path}")
        print(f"Wrote evaluation review to {html_path}")
        try:
            relative_html = html_path.resolve().relative_to(REPOSITORY_ROOT)
        except ValueError:
            print(f"Evaluation review: {html_path.resolve().as_uri()}")
        else:
            print(
                "Evaluation review: "
                f"http://127.0.0.1:8000/{quote(relative_html.as_posix())}"
            )
        try:
            relative_output = args.output_dir.resolve().relative_to(
                REPOSITORY_ROOT / "processed_indonesia"
            )
        except ValueError:
            pass
        else:
            print(
                "Reusable evaluation viewer: "
                "http://127.0.0.1:8000/analysis_indonesia/training_evaluation.html"
                f"?folder={quote(relative_output.as_posix())}"
            )
        release_evaluation_memory(device)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    def make_train_loader(epoch: int) -> DataLoader:
        # A per-epoch seed makes the shuffled order reproducible for exact resume.
        generator = torch.Generator().manual_seed(args.seed + epoch)
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=collate,
        )

    resume_progress: dict[str, Any] = {}
    if resume_adapter:
        progress_path = args.output_dir / "best_lora" / "training_progress.json"
        if resume_exact:
            resume_progress = load_training_progress(
                args.output_dir / "best_lora", optimizer, scaler, device
            )
        elif progress_path.is_file():
            resume_progress = json.loads(progress_path.read_text(encoding="utf-8"))

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
    global_step = int(resume_progress.get("global_step", 0)) if resume_exact else 0
    last_eval_step = -1
    reached_step_limit = False
    best_eval_loss = float(resume_progress.get("best_eval_loss", float("inf")))
    best_wer_progress_path = args.output_dir / "best_wer" / "training_progress.json"
    if resume_adapter and best_wer_progress_path.is_file():
        best_wer_progress = json.loads(best_wer_progress_path.read_text(encoding="utf-8"))
        best_eval_wer = float(best_wer_progress.get("best_eval_wer", float("inf")))
    else:
        best_eval_wer = float(resume_progress.get("best_eval_wer", float("inf")))
    best_eval_rows: list[dict[str, str]] = []
    best_wer_rows: list[dict[str, str]] = []
    best_csv_path = args.output_dir / "best_eval_predictions.csv"
    if resume_adapter and best_csv_path.is_file():
        with best_csv_path.open(encoding="utf-8", newline="") as file:
            best_eval_rows = list(csv.DictReader(file))
    best_wer_csv_path = args.output_dir / "best_wer_eval_predictions.csv"
    if resume_adapter and best_wer_csv_path.is_file():
        with best_wer_csv_path.open(encoding="utf-8", newline="") as file:
            best_wer_rows = list(csv.DictReader(file))
    latest_eval_loss = float("inf")
    latest_eval_rows: list[dict[str, str]] = []
    run_label = str(run.name or args.wandb_run_name or "local")
    completed_high_losses: list[dict[str, Any]] = (
        list(resume_progress.get("completed_high_losses", []))
        if resume_exact else []
    )
    current_loss_heap: list[tuple[float, str]] = (
        [
            (float(loss), str(audio_path))
            for loss, audio_path in resume_progress.get("current_loss_heap", [])
        ]
        if resume_exact else []
    )

    start_epoch = 1
    resume_batch_in_epoch = 0
    if resume_exact:
        saved_epoch = int(resume_progress["epoch"])
        completed_batches = int(resume_progress["batch_in_epoch"])
        saved_batches = int(resume_progress["batches_per_epoch"])
        expected_values = {
            "batch_size": args.batch_size,
            "seed": args.seed,
            "training_clips": len(train_dataset),
        }
        mismatches = [
            f"{key}: saved={resume_progress.get(key)!r}, current={value!r}"
            for key, value in expected_values.items()
            if resume_progress.get(key) != value
        ]
        current_batches = len(make_train_loader(saved_epoch))
        if current_batches != saved_batches:
            mismatches.append(
                f"batches_per_epoch: saved={saved_batches}, current={current_batches}"
            )
        if mismatches:
            raise ValueError(
                "Cannot resume from the exact batch position because the training "
                "configuration changed (" + "; ".join(mismatches) + "). Choose "
                "weight-only resume instead."
            )
        if completed_batches >= saved_batches:
            if current_loss_heap:
                completed_high_losses.extend(
                    {
                        "audio_path": audio_path,
                        "epoch": saved_epoch,
                        "loss": loss_value,
                    }
                    for loss_value, audio_path in sorted(
                        current_loss_heap, reverse=True
                    )
                )
                current_loss_heap = []
            start_epoch = saved_epoch + 1
        else:
            start_epoch = saved_epoch
            resume_batch_in_epoch = completed_batches
        print(
            f"Resuming at epoch {start_epoch}/{args.epochs}, after batch "
            f"{resume_batch_in_epoch}, global step {global_step}."
        )

    def run_evaluation(
        step: int, epoch: int, batch_in_epoch: int, batches_per_epoch: int
    ) -> dict[str, float]:
        nonlocal best_eval_loss, best_eval_wer, best_eval_rows, best_wer_rows
        nonlocal latest_eval_loss, latest_eval_rows
        metrics, evaluation_rows = evaluate(
            model,
            eval_loader,
            eval_decode_loader,
            eval_dataset,
            device,
            tokenizer.eot,
            args.eos_loss_weight,
            language=args.language,
        )
        latest_eval_loss = metrics["eval/loss"]
        latest_eval_rows = evaluation_rows
        write_evaluation_csv(args.output_dir / "eval_predictions.csv", evaluation_rows)
        improved_loss = metrics["eval/loss"] < best_eval_loss
        improved_wer = metrics["eval/wer"] < best_eval_wer
        if improved_loss:
            best_eval_loss = metrics["eval/loss"]
            best_eval_rows = evaluation_rows
        if improved_wer:
            best_eval_wer = metrics["eval/wer"]
            best_wer_rows = evaluation_rows
        metrics["eval/best_loss"] = best_eval_loss
        metrics["eval/best_wer"] = best_eval_wer
        if improved_loss:
            save_token_lid_lora_adapter(model, args.output_dir / "best_lora")
            save_training_progress(
                args.output_dir / "best_lora",
                optimizer,
                scaler,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                batches_per_epoch=batches_per_epoch,
                global_step=step,
                configured_epochs=args.epochs,
                best_eval_loss=best_eval_loss,
                best_eval_wer=best_eval_wer,
                batch_size=args.batch_size,
                seed=args.seed,
                training_clips=len(train_dataset),
                current_loss_heap=current_loss_heap,
                completed_high_losses=completed_high_losses,
            )
            best_csv_path = args.output_dir / "best_eval_predictions.csv"
            write_evaluation_csv(best_csv_path, best_eval_rows)
            upsert_run_results(
                args.run_results_csv,
                run_name=run_label,
                evaluation_step=step,
                evaluation_csv=best_csv_path,
                metrics=metrics,
            )
        if improved_wer:
            best_wer_dir = args.output_dir / "best_wer"
            save_token_lid_lora_adapter(model, best_wer_dir)
            save_training_progress(
                best_wer_dir,
                optimizer,
                scaler,
                epoch=epoch,
                batch_in_epoch=batch_in_epoch,
                batches_per_epoch=batches_per_epoch,
                global_step=step,
                configured_epochs=args.epochs,
                best_eval_loss=best_eval_loss,
                best_eval_wer=best_eval_wer,
                batch_size=args.batch_size,
                seed=args.seed,
                training_clips=len(train_dataset),
                current_loss_heap=current_loss_heap,
                completed_high_losses=completed_high_losses,
            )
            write_evaluation_csv(best_wer_csv_path, best_wer_rows)
        wandb.log(metrics, step=step)
        print(
            f"step {step}: loss={metrics['eval/loss']:.4f}, WER={metrics['eval/wer']:.3%}, "
            f"LID accuracy={metrics['eval/lid_accuracy']:.3%}, "
            f"best loss={best_eval_loss:.4f}, best WER={best_eval_wer:.3%}"
        )
        model.train()
        release_evaluation_memory(device)
        if device.type in {"mps", "cuda"}:
            print(f"Released cached {device.type.upper()} evaluation memory")
        return metrics

    training_jember_count = sum(
        row.get("dataset") == "jember" for row in train_rows
    )
    training_development_count = sum(
        row.get("dataset") == "indonesian_development" for row in train_rows
    )
    training_total = training_jember_count + training_development_count
    print(
        "Training set composition: "
        f"Jember={training_jember_count}, "
        f"Indonesian-dev={training_development_count}, "
        f"Jember share={training_jember_count / max(training_total, 1):.2%}"
    )

    try:
        current_epoch = max(1, start_epoch)
        current_batch = 0
        current_batches_per_epoch = len(make_train_loader(current_epoch))
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            if not (
                resume_exact
                and epoch == start_epoch
                and resume_batch_in_epoch > 0
            ):
                current_loss_heap = []
            model.train()
            train_loader = make_train_loader(epoch)
            current_batches_per_epoch = len(train_loader)
            skip_batches = resume_batch_in_epoch if epoch == start_epoch else 0
            train_iterator = iter(train_loader)
            for _ in range(skip_batches):
                next(train_iterator)
            progress = tqdm(
                train_iterator,
                desc=f"Epoch {epoch}/{args.epochs}",
                initial=skip_batches,
                total=current_batches_per_epoch,
            )
            for batch_number, batch in enumerate(progress, start=skip_batches + 1):
                current_batch = batch_number
                optimizer.zero_grad(set_to_none=True)
                autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()
                with autocast:
                    (
                        asr_loss,
                        lid_loss,
                        loss,
                        correct,
                        lid_count,
                        per_example_losses,
                    ) = model_losses(
                        model,
                        batch,
                        device,
                        tokenizer.eot,
                        args.eos_loss_weight,
                    )
                for row, example_loss in zip(
                    batch["rows"], per_example_losses.detach().cpu().tolist()
                ):
                    candidate = (float(example_loss), row["audio_path"])
                    if len(current_loss_heap) < HIGHEST_TRAINING_LOSSES:
                        heapq.heappush(current_loss_heap, candidate)
                    elif candidate > current_loss_heap[0]:
                        heapq.heapreplace(current_loss_heap, candidate)
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
                    run_evaluation(
                        global_step, epoch, batch_number, current_batches_per_epoch
                    )
                    last_eval_step = global_step
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    reached_step_limit = True
                    break
            save_token_lid_lora_adapter(model, args.output_dir / "last_lora")
            if current_batch >= current_batches_per_epoch:
                completed_high_losses.extend(
                    {
                        "audio_path": audio_path,
                        "epoch": epoch,
                        "loss": loss_value,
                    }
                    for loss_value, audio_path in sorted(
                        current_loss_heap, reverse=True
                    )
                )
                current_loss_heap = []
            if reached_step_limit:
                break

        if global_step != last_eval_step:
            run_evaluation(
                global_step,
                current_epoch,
                current_batch,
                current_batches_per_epoch,
            )
        if not best_eval_rows:
            raise RuntimeError("No held-out evaluation rows were produced")
        high_loss_candidates = completed_high_losses + [
            {
                "audio_path": audio_path,
                "epoch": current_epoch,
                "loss": loss_value,
            }
            for loss_value, audio_path in sorted(current_loss_heap, reverse=True)
        ]
        rows_by_audio_path = {row["audio_path"]: row for row in train_rows}
        highest_training_losses = transcribe_highest_training_losses(
            high_loss_candidates,
            rows_by_audio_path,
            model,
            model.dims.n_mels,
            args.language,
            device,
        )
        highest_losses_csv = args.output_dir / "highest_training_losses.csv"
        write_highest_training_losses(
            highest_losses_csv, highest_training_losses
        )
        print(f"Wrote highest training losses to {highest_losses_csv}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run_label = re.sub(r"[^A-Za-z0-9_-]+", "_", run_label)
        html_path = args.analysis_dir / f"{timestamp}_run_{safe_run_label}.html"
        write_evaluation_html(
            html_path,
            latest_eval_rows,
            run_label,
            latest_eval_loss,
            highest_training_losses,
        )
        run.summary["best_eval_loss"] = best_eval_loss
        run.summary["best_eval_wer"] = best_eval_wer
        run.summary["best_lora_adapter"] = str(args.output_dir / "best_lora")
        run.summary["best_wer_adapter"] = str(args.output_dir / "best_wer")
        run.summary["evaluation_csv"] = str(args.output_dir / "eval_predictions.csv")
        run.summary["highest_training_losses_csv"] = str(highest_losses_csv)
        run.summary["evaluation_html"] = str(html_path)
        print(f"Wrote final held-out evaluation review to {html_path}")
        try:
            relative_html = html_path.resolve().relative_to(REPOSITORY_ROOT)
        except ValueError:
            print(f"Evaluation review: {html_path.resolve().as_uri()}")
        else:
            print(
                "Evaluation review: "
                f"http://127.0.0.1:8000/{quote(relative_html.as_posix())}"
            )
        try:
            relative_output = args.output_dir.resolve().relative_to(
                REPOSITORY_ROOT / "processed_indonesia"
            )
        except ValueError:
            pass
        else:
            print(
                "Reusable evaluation viewer: "
                "http://127.0.0.1:8000/analysis_indonesia/training_evaluation.html"
                f"?folder={quote(relative_output.as_posix())}"
            )
    finally:
        run.finish()


if __name__ == "__main__":
    main()
