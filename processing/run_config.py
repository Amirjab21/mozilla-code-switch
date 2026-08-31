"""Shared loading and validation for named processing/training YAML run configurations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_section(path: Path | None, section: str) -> tuple[dict[str, Any], str | None]:
    """Read one named configuration section and the optional top-level run name."""
    if path is None:
        return {}, None
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    values = config.get(section, {})
    if not isinstance(values, dict):
        raise ValueError(f"{path}: `{section}` must be a YAML mapping")
    return {str(key).replace("-", "_"): value for key, value in values.items()}, config.get("run_name")


def apply_defaults(parser: argparse.ArgumentParser, values: dict[str, Any], path: Path | None) -> None:
    """Apply config values only when they match an argument destination."""
    known = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - known)
    if unknown:
        location = str(path) if path else "configuration"
        parser.error(f"Unknown setting(s) in {location}: {', '.join(unknown)}")
    parser.set_defaults(**values)
