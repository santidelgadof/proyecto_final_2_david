from __future__ import annotations

from pathlib import Path
import argparse

from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .config import ExperimentConfig
from .models import GeneratorUNet
from .utils import ensure_dir, tensor_to_image


def load_generator(checkpoint_path: str | Path, device: torch.device) -> tuple[GeneratorUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_dict = checkpoint["config"]
    config = ExperimentConfig(**config_dict)
    model = GeneratorUNet(
        base_channels=config.generator_base_channels,
        residual_blocks=config.residual_blocks,
    ).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, config_dict


def extract_condition(image: Image.Image, input_kind: str) -> Image.Image:
    if input_kind == "label":
        return image.convert("RGB")
    if input_kind == "paired":
        width, height = image.size
        midpoint = width // 2
        return image.crop((midpoint, 0, width, height)).convert("RGB")
    raise ValueError(f"Unsupported input_kind: {input_kind}")


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    resized = TF.resize(image, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    tensor = TF.to_tensor(resized) * 2.0 - 1.0
    return tensor.unsqueeze(0)


def run_inference(checkpoint: str | Path, input_path: str | Path, output_path: str | Path, device_name: str, input_kind: str) -> Path:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    model, config = load_generator(checkpoint, device)
    image = Image.open(input_path)
    condition = extract_condition(image, input_kind=input_kind)
    tensor = preprocess(condition, image_size=config["image_size"]).to(device)

    with torch.no_grad():
        prediction = model(tensor)[0]

    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    tensor_to_image(prediction).save(output_file)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pix2Pix inference on a label map or paired sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--input-kind", choices=["paired", "label"], default="paired")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_inference(
        checkpoint=args.checkpoint,
        input_path=args.input,
        output_path=args.output,
        device_name=args.device,
        input_kind=args.input_kind,
    )
    print(output)


if __name__ == "__main__":
    main()
