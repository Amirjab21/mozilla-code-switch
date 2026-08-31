"""Fine-tuning backends for the local Whisper token-language fork.

LoRA and full-parameter training share one interface so the training loop can
load and checkpoint either variant without method-specific branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional

import torch

from .model import load_token_lid_model, save_token_lid_checkpoint
from .model_lora import load_token_lid_lora_model, save_token_lid_lora_adapter

CheckpointKind = Literal["best", "last", "final"]


class FineTuneMethod(ABC):
    """Load a token-LID model and persist named training checkpoints."""

    name: str
    wandb_best_key: str

    @abstractmethod
    def load_model(
        self,
        name: str,
        *,
        device: Optional[str | torch.device] = None,
        download_root: Optional[str] = None,
    ) -> Any:
        """Return a trainable token-LID model for this fine-tuning method."""

    @abstractmethod
    def save_checkpoint(self, model: Any, path: Path) -> None:
        """Write one named checkpoint in this method's native format."""

    @abstractmethod
    def checkpoint_path(self, output_dir: Path, kind: CheckpointKind) -> Path:
        """Return the path used for a best, last, or final checkpoint."""

    def save_named(self, model: Any, output_dir: Path, kind: CheckpointKind) -> Path:
        path = self.checkpoint_path(output_dir, kind)
        self.save_checkpoint(model, path)
        return path


class LoraFineTune(FineTuneMethod):
    name = "lora"
    wandb_best_key = "best_lora_adapter"

    def __init__(self, *, rank: int = 16, alpha: int = 32, dropout: float = 0.05):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

    def load_model(
        self,
        name: str,
        *,
        device: Optional[str | torch.device] = None,
        download_root: Optional[str] = None,
    ):
        return load_token_lid_lora_model(
            name,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            device=device,
            download_root=download_root,
        )

    def save_checkpoint(self, model: Any, path: Path) -> None:
        save_token_lid_lora_adapter(model, path)

    def checkpoint_path(self, output_dir: Path, kind: CheckpointKind) -> Path:
        return output_dir / f"{kind}_lora"


class FullFineTune(FineTuneMethod):
    name = "full"
    wandb_best_key = "best_checkpoint"

    def load_model(
        self,
        name: str,
        *,
        device: Optional[str | torch.device] = None,
        download_root: Optional[str] = None,
    ):
        return load_token_lid_model(name, device=device, download_root=download_root)

    def save_checkpoint(self, model: Any, path: Path) -> None:
        save_token_lid_checkpoint(model, path)

    def checkpoint_path(self, output_dir: Path, kind: CheckpointKind) -> Path:
        return output_dir / f"{kind}.pt"


def make_finetune_method(
    method: str,
    *,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> FineTuneMethod:
    """Instantiate the LoRA or full-parameter fine-tuning backend."""
    if method == "lora":
        return LoraFineTune(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    if method == "full":
        return FullFineTune()
    raise ValueError(f"Unknown fine-tuning method {method!r}; expected 'lora' or 'full'")
