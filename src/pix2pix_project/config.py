from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    data_dir: str = "cityscapes"
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 20
    num_workers: int = 0
    learning_rate: float = 2e-4
    beta1: float = 0.5
    lambda_l1: float = 100.0
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    augment: bool = False
    crop_margin: int = 0
    gan_mode: str = "bce"
    generator_base_channels: int = 64
    discriminator_base_channels: int = 64
    residual_blocks: int = 0
    save_every: int = 1
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    mixed_precision: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def get_preset_config(name: str) -> ExperimentConfig:
    normalized = name.strip().lower()
    if normalized == "baseline":
        return ExperimentConfig(name="baseline")
    if normalized == "improved":
        return ExperimentConfig(
            name="improved",
            augment=True,
            crop_margin=16,
            gan_mode="lsgan",
            residual_blocks=2,
        )
    raise ValueError(f"Unknown preset: {name}")
