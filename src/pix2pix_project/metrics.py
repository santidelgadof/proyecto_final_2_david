from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def denormalize(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(-1.0, 1.0).add(1.0).div(2.0)


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target))


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = denormalize(prediction)
    tgt = denormalize(target)
    mse = torch.mean((pred - tgt) ** 2)
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10(torch.tensor(1.0, device=prediction.device) / mse)


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = denormalize(prediction)
    tgt = denormalize(target)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_pred = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_tgt = F.avg_pool2d(tgt, kernel_size=3, stride=1, padding=1)

    sigma_pred = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_pred.pow(2)
    sigma_tgt = F.avg_pool2d(tgt * tgt, kernel_size=3, stride=1, padding=1) - mu_tgt.pow(2)
    sigma_cross = F.avg_pool2d(pred * tgt, kernel_size=3, stride=1, padding=1) - mu_pred * mu_tgt

    numerator = (2 * mu_pred * mu_tgt + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred.pow(2) + mu_tgt.pow(2) + c1) * (sigma_pred + sigma_tgt + c2)
    score = numerator / denominator
    return score.mean()


def batch_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "mae": float(mae(prediction, target).item()),
        "psnr": float(psnr(prediction, target).item()),
        "ssim": float(ssim(prediction, target).item()),
    }
