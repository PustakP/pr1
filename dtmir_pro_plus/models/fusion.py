"""
Multi-Weather Feature Fusion block (MWFF), kept as in DTMIR-Pro: fuses a
pair of weather-stream features both globally (via FFT) and locally (via
multi-scale gated depthwise convolutions), so the restoration network can
handle compound degradations like "rain + haze" together.

We are NOT changing this block in this pass -- it stays as in the original
paper. Included here so the full pipeline in dtmir_plus.py runs end-to-end.
"""

import torch
import torch.nn as nn


class GlobalFusion(nn.Module):
    """FFT2D -> 1x1 conv -> IFFT2D, applied to the concatenated pair."""

    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Conv2d(channels * 2, channels * 2, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, b], dim=1)
        x_freq = torch.fft.rfft2(x, norm="ortho")
        real = self.mix(x_freq.real)
        imag = self.mix(x_freq.imag)
        x_freq = torch.complex(real, imag)
        out = torch.fft.irfft2(x_freq, s=x.shape[-2:], norm="ortho")
        return out


class GatedDepthwiseConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels)
        self.gate = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * torch.sigmoid(self.gate(x))


class LocalFusion(nn.Module):
    """Multi-scale gated depthwise convolutions over the concatenated pair,
    to capture weather artifacts of different physical sizes (fine rain
    streaks vs. large snow blobs, etc.)."""

    def __init__(self, channels: int, kernel_sizes=(1, 3, 5)):
        super().__init__()
        self.pre = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.branches = nn.ModuleList([
            GatedDepthwiseConv(channels, k) for k in kernel_sizes
        ])
        self.post = nn.Conv2d(channels * len(kernel_sizes), channels, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = self.pre(torch.cat([a, b], dim=1))
        branch_outs = [branch(x) for branch in self.branches]
        return self.post(torch.cat(branch_outs, dim=1))


class WeatherFusionBlock(nn.Module):
    """Combines one pair of weather-stream features (e.g. haze + rain)."""

    def __init__(self, channels: int):
        super().__init__()
        self.global_fusion = GlobalFusion(channels)
        self.local_fusion = LocalFusion(channels)
        self.project = nn.Conv2d(channels * 2 + channels, channels, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = self.global_fusion(a, b)
        l = self.local_fusion(a, b)
        return self.project(torch.cat([g, l], dim=1))


class MultiWeatherFusion(nn.Module):
    """Produces the three pairwise fusions used by the restoration network:
    haze+rain, rain+snow, haze+snow."""

    def __init__(self, channels: int):
        super().__init__()
        self.hr = WeatherFusionBlock(channels)
        self.rs = WeatherFusionBlock(channels)
        self.hs = WeatherFusionBlock(channels)

    def forward(self, f_haze: torch.Tensor, f_rain: torch.Tensor, f_snow: torch.Tensor):
        return self.hr(f_haze, f_rain), self.rs(f_rain, f_snow), self.hs(f_haze, f_snow)
