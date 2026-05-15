from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import contextlib
import json
import time

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .config import ExperimentConfig, get_preset_config
from .data import build_dataloaders
from .metrics import batch_metrics
from .models import GeneratorUNet, PatchDiscriminator
from .utils import ensure_dir, save_json, set_seed, tensor_to_image


class GANLoss(nn.Module):
    def __init__(self, gan_mode: str) -> None:
        super().__init__()
        if gan_mode == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported gan_mode: {gan_mode}")

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_value = 1.0 if target_is_real else 0.0
        target_tensor = torch.full_like(prediction, fill_value=target_value)
        return self.loss(prediction, target_tensor)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_preview(
    output_dir: Path,
    split: str,
    epoch: int,
    condition: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    preview_dir = ensure_dir(output_dir / "previews" / split)
    samples = torch.cat([condition[:4], prediction[:4], target[:4]], dim=0)
    grid = make_grid(samples, nrow=min(4, condition.shape[0]), normalize=True, value_range=(-1, 1))
    tensor_to_image(grid).save(preview_dir / f"epoch_{epoch:03d}.png")


def evaluate(
    generator: GeneratorUNet,
    discriminator: PatchDiscriminator,
    loader: DataLoader,
    gan_loss: GANLoss,
    l1_loss: nn.Module,
    device: torch.device,
    config: ExperimentConfig,
    output_dir: Path | None = None,
    split: str = "val",
    epoch: int = 0,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()
    aggregates = {"g_total": 0.0, "g_gan": 0.0, "g_l1": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0}
    batches = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if config.max_eval_batches is not None and batch_index >= config.max_eval_batches:
                break

            condition = batch["condition"].to(device)
            target = batch["target"].to(device)
            prediction = generator(condition)
            disc_fake = discriminator(condition, prediction)
            gan_component = gan_loss(disc_fake, True)
            l1_component = l1_loss(prediction, target)
            total_loss = gan_component + config.lambda_l1 * l1_component

            metrics = batch_metrics(prediction, target)
            aggregates["g_total"] += float(total_loss.item())
            aggregates["g_gan"] += float(gan_component.item())
            aggregates["g_l1"] += float(l1_component.item())
            for name, value in metrics.items():
                aggregates[name] += value
            batches += 1

            if output_dir is not None and batch_index == 0:
                save_preview(output_dir, split, epoch, condition, prediction, target)

    if batches == 0:
        raise RuntimeError(f"No evaluation batches available for split {split}")
    return {name: value / batches for name, value in aggregates.items()}


def train_one_epoch(
    generator: GeneratorUNet,
    discriminator: PatchDiscriminator,
    loader: DataLoader,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    gan_loss: GANLoss,
    l1_loss: nn.Module,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    generator.train()
    discriminator.train()
    aggregates = {"d_total": 0.0, "g_total": 0.0, "g_gan": 0.0, "g_l1": 0.0}
    batches = 0
    autocast_context = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if device.type == "cuda" and config.mixed_precision
        else contextlib.nullcontext
    )

    for batch_index, batch in enumerate(loader):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break

        condition = batch["condition"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        discriminator_optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            prediction = generator(condition)
            disc_real = discriminator(condition, target)
            disc_fake = discriminator(condition, prediction.detach())
            d_loss_real = gan_loss(disc_real, True)
            d_loss_fake = gan_loss(disc_fake, False)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        if scaler is not None:
            scaler.scale(d_loss).backward()
            scaler.step(discriminator_optimizer)
        else:
            d_loss.backward()
            discriminator_optimizer.step()

        generator_optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            prediction = generator(condition)
            disc_fake = discriminator(condition, prediction)
            g_gan = gan_loss(disc_fake, True)
            g_l1 = l1_loss(prediction, target)
            g_loss = g_gan + config.lambda_l1 * g_l1

        if scaler is not None:
            scaler.scale(g_loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()
        else:
            g_loss.backward()
            generator_optimizer.step()

        aggregates["d_total"] += float(d_loss.item())
        aggregates["g_total"] += float(g_loss.item())
        aggregates["g_gan"] += float(g_gan.item())
        aggregates["g_l1"] += float(g_l1.item())
        batches += 1

    if batches == 0:
        raise RuntimeError("No training batches were processed")
    return {name: value / batches for name, value in aggregates.items()}


def save_checkpoint(
    path: Path,
    generator: GeneratorUNet,
    discriminator: PatchDiscriminator,
    generator_optimizer: Adam,
    discriminator_optimizer: Adam,
    config: ExperimentConfig,
    epoch: int,
    best_val_psnr: float,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(config),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
            "best_val_psnr": best_val_psnr,
        },
        path,
    )


def run_training(config: ExperimentConfig, output_dir: str | Path, device_name: str = "auto") -> dict[str, object]:
    output_path = ensure_dir(output_dir)
    set_seed(config.seed)
    config.save_json(output_path / "config.json")

    device = get_device(device_name)
    loaders, split_sizes = build_dataloaders(config)
    generator = GeneratorUNet(
        base_channels=config.generator_base_channels,
        residual_blocks=config.residual_blocks,
    ).to(device)
    discriminator = PatchDiscriminator(base_channels=config.discriminator_base_channels).to(device)
    generator_optimizer = Adam(generator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    discriminator_optimizer = Adam(discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
    gan_loss = GANLoss(config.gan_mode)
    l1_loss = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.mixed_precision)

    history: list[dict[str, object]] = []
    best_val_psnr = float("-inf")
    start_time = time.time()

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            generator,
            discriminator,
            loaders["train"],
            generator_optimizer,
            discriminator_optimizer,
            gan_loss,
            l1_loss,
            scaler if scaler.is_enabled() else None,
            device,
            config,
        )
        val_metrics = evaluate(
            generator,
            discriminator,
            loaders["val"],
            gan_loss,
            l1_loss,
            device,
            config,
            output_dir=output_path,
            split="val",
            epoch=epoch,
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        save_json({"history": history}, output_path / "history.json")

        latest_path = output_path / "latest.pt"
        save_checkpoint(
            latest_path,
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            config,
            epoch,
            best_val_psnr,
        )

        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            save_checkpoint(
                output_path / "best.pt",
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                config,
                epoch,
                best_val_psnr,
            )

    best_checkpoint = torch.load(output_path / "best.pt", map_location=device, weights_only=False)
    generator.load_state_dict(best_checkpoint["generator"])
    test_metrics = evaluate(
        generator,
        discriminator,
        loaders["test"],
        gan_loss,
        l1_loss,
        device,
        config,
        output_dir=output_path,
        split="test",
        epoch=config.epochs,
    )

    summary = {
        "config": config.to_dict(),
        "device": str(device),
        "split_sizes": split_sizes,
        "history": history,
        "test": test_metrics,
        "training_minutes": round((time.time() - start_time) / 60.0, 2),
    }
    save_json(summary, output_path / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pix2Pix experiments on paired image datasets.")
    parser.add_argument("--preset", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_preset_config(args.preset)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.output_dir is None:
        output_dir = Path("runs") / config.name
    else:
        output_dir = Path(args.output_dir)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.max_train_batches is not None:
        config.max_train_batches = args.max_train_batches
    if args.max_eval_batches is not None:
        config.max_eval_batches = args.max_eval_batches

    summary = run_training(config, output_dir=output_dir, device_name=args.device)
    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
