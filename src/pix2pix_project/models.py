from __future__ import annotations

import torch
from torch import nn


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64, residual_blocks: int = 0) -> None:
        super().__init__()
        ch = base_channels
        self.down1 = DownBlock(in_channels, ch, normalize=False)
        self.down2 = DownBlock(ch, ch * 2)
        self.down3 = DownBlock(ch * 2, ch * 4)
        self.down4 = DownBlock(ch * 4, ch * 8)
        self.down5 = DownBlock(ch * 8, ch * 8)
        self.down6 = DownBlock(ch * 8, ch * 8)
        self.down7 = DownBlock(ch * 8, ch * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            *[ResidualBlock(ch * 8) for _ in range(residual_blocks)],
        )

        self.up1 = UpBlock(ch * 8, ch * 8, dropout=0.5)
        self.up2 = UpBlock(ch * 16, ch * 8, dropout=0.5)
        self.up3 = UpBlock(ch * 16, ch * 8, dropout=0.5)
        self.up4 = UpBlock(ch * 16, ch * 8)
        self.up5 = UpBlock(ch * 16, ch * 4)
        self.up6 = UpBlock(ch * 8, ch * 2)
        self.up7 = UpBlock(ch * 4, ch)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ch * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        bottleneck = self.bottleneck(d7)

        u1 = self.up1(bottleneck)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        return self.final(torch.cat([u7, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 64) -> None:
        super().__init__()
        ch = base_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 8, 1, kernel_size=4, stride=1, padding=1),
        )
        self.apply(init_weights)

    def forward(self, condition: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([condition, target], dim=1)
        return self.model(x)
