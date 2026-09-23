"""Dataset selection helpers shared by Qwen3-ASR training runs."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


TRAIN_FRACTION = 0.90


def read_rows(manifest: Path) -> list[dict[str, str]]:
    """Load and validate the fields required for ASR fine-tuning."""
    with manifest.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        missing = {"audio_path", "transcript", "dataset"} - fields
        if missing:
            raise ValueError(f"{manifest} is missing required fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{manifest} contains no rows")
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Audio file listed in manifest does not exist: {audio_path}"
            )
        if not row["transcript"].strip():
            raise ValueError(f"Empty transcript for {audio_path}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def composition_bucket(row: dict[str, str]) -> str:
    dataset = row.get("dataset", "")
    if dataset == "jember":
        return "jember"
    if dataset in {"cv_indonesian", "cv_javanese", "commonvoice_code_switch"}:
        return "commonvoice"
    if dataset in {"indonesian_development", "indonesian_dev", "development"}:
        return "indonesian_development"
    raise ValueError(f"Unsupported dataset for composition: {dataset!r}")


def augmentation_family_key(row: dict[str, str]) -> str:
    """Return the clean source shared by an original and its augmentations."""
    return row.get("augmentation_source_audio", "").strip() or row["audio_path"]


def split_proportional_augmented_evaluation(
    rows: list[dict[str, str]],
    total_clips: int,
    jember_proportion: float,
    commonvoice_proportion: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    """Reproduce the existing leakage-safe mixed-dataset evaluation split."""
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
                f"Requested {count} {dataset} evaluation clips, but only "
                f"{len(dataset_selected)} independent augmentation families exist"
            )
        selected.extend(dataset_selected)

    selected_families = {augmentation_family_key(row) for row in selected}
    train = [
        row for row in rows
        if augmentation_family_key(row) not in selected_families
    ]
    if not train:
        raise ValueError("The evaluation split left no training clips")
    return train, selected, requested


def recording_key(row: dict[str, str]) -> str:
    stem = Path(row["audio_path"]).stem
    return stem.rsplit("_", 1)[0]


def split_rows(
    rows: list[dict[str, str]], seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Make the legacy deterministic 90/10 recording-level split."""
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
        return train, test

    shuffled = rows[:]
    rng.shuffle(shuffled)
    return shuffled[target_test_rows:], shuffled[:target_test_rows]


def cap_jember_training_rows(
    rows: list[dict[str, str]], maximum: int | None, seed: int
) -> list[dict[str, str]]:
    if maximum is None:
        return rows
    jember_rows = [row for row in rows if row.get("dataset") == "jember"]
    retained = random.Random(seed + 10_000).sample(
        jember_rows, min(maximum, len(jember_rows))
    )
    retained_paths = {row["audio_path"] for row in retained}
    return [
        row for row in rows
        if row.get("dataset") != "jember" or row["audio_path"] in retained_paths
    ]


def qwen_language_name(row: dict[str, str]) -> str:
    """Return only language names officially supported by Qwen3-ASR."""
    if row.get("dataset") in {"indonesian_development", "indonesian_dev", "development", "cv_indonesian"}:
        return "Indonesian"
    # Qwen3-ASR does not advertise Javanese support, and a code-switched clip
    # cannot be represented faithfully by its single utterance-level prefix.
    return "None"


def qwen_target(row: dict[str, str]) -> str:
    return f"language {qwen_language_name(row)}<asr_text>{row['transcript'].strip()}"
