from __future__ import annotations

from pathlib import Path
import json
import random

from PIL import Image
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.detach().cpu().clamp(-1.0, 1.0)
    image = image.add(1.0).div(2.0)
    image = image.mul(255.0).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(image)
