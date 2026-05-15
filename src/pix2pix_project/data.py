from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import ExperimentConfig


@dataclass(slots=True)
class Sample:
    condition: torch.Tensor
    target: torch.Tensor
    path: str


class PairedImageDataset(Dataset[Sample]):
    def __init__(
        self,
        image_paths: list[Path],
        image_size: int,
        augment: bool,
        crop_margin: int,
        seed: int,
    ) -> None:
        self.image_paths = image_paths
        self.image_size = image_size
        self.augment = augment
        self.crop_margin = crop_margin
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_paths)

    def _resize(self, image: Image.Image, is_label: bool) -> Image.Image:
        interpolation = InterpolationMode.NEAREST if is_label else InterpolationMode.BICUBIC
        return TF.resize(image, [self.image_size, self.image_size], interpolation=interpolation)

    def _paired_augment(self, condition: Image.Image, target: Image.Image, index: int) -> tuple[Image.Image, Image.Image]:
        rng = random.Random(self.seed + index)
        if rng.random() < 0.5:
            condition = TF.hflip(condition)
            target = TF.hflip(target)

        if self.crop_margin > 0:
            resize_to = self.image_size + self.crop_margin
            condition = TF.resize(condition, [resize_to, resize_to], interpolation=InterpolationMode.NEAREST)
            target = TF.resize(target, [resize_to, resize_to], interpolation=InterpolationMode.BICUBIC)
            top = rng.randint(0, self.crop_margin)
            left = rng.randint(0, self.crop_margin)
            condition = TF.crop(condition, top, left, self.image_size, self.image_size)
            target = TF.crop(target, top, left, self.image_size, self.image_size)
        else:
            condition = self._resize(condition, is_label=True)
            target = self._resize(target, is_label=False)

        return condition, target

    def __getitem__(self, index: int) -> Sample:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        midpoint = width // 2

        target = image.crop((0, 0, midpoint, height))
        condition = image.crop((midpoint, 0, width, height))

        if self.augment:
            condition, target = self._paired_augment(condition, target, index)
        else:
            condition = self._resize(condition, is_label=True)
            target = self._resize(target, is_label=False)

        condition_tensor = TF.to_tensor(condition) * 2.0 - 1.0
        target_tensor = TF.to_tensor(target) * 2.0 - 1.0
        return Sample(condition=condition_tensor, target=target_tensor, path=str(path))


def collate_samples(batch: list[Sample]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "condition": torch.stack([sample.condition for sample in batch]),
        "target": torch.stack([sample.target for sample in batch]),
        "path": [sample.path for sample in batch],
    }


def discover_images(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    paths = sorted(path for path in root.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No JPG images found in {root}")
    return paths


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def build_dataloaders(config: ExperimentConfig) -> tuple[dict[str, DataLoader], dict[str, int]]:
    paths = discover_images(config.data_dir)
    splits = split_paths(paths, config.train_ratio, config.val_ratio, config.seed)
    datasets = {
        split: PairedImageDataset(
            image_paths=split_paths_list,
            image_size=config.image_size,
            augment=config.augment and split == "train",
            crop_margin=config.crop_margin,
            seed=config.seed,
        )
        for split, split_paths_list in splits.items()
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            num_workers=config.num_workers,
            pin_memory=True,
            collate_fn=collate_samples,
        )
        for split, dataset in datasets.items()
    }
    sizes = {split: len(dataset) for split, dataset in datasets.items()}
    return loaders, sizes
