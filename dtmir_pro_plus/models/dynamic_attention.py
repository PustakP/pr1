"""
Dynamic Attention Block (DAB), as described in DTMIR-Pro: keys/values come
from ordinary depthwise convolutions, but the *query* comes from a "dynamic
convolution" -- a per-input weighted combination of N parallel kernels
(weights produced by a squeeze-and-excitation style gate).

Kept close to the paper for now. This is one of the two places flagged as
a possible upgrade later (swap for MIRAGE's conv+attention+MLP channel
split) -- not done in this pass, see README.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # per-channel logits, NOT yet passed through sigmoid


class DynamicConv(nn.Module):
    """Attentive aggregation of N parallel conv kernels, per DTMIR-Pro Eq. 4-5."""

    def __init__(self, channels: int, num_kernels: int = 4, kernel_size: int = 3):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
            for _ in range(num_kernels)
        ])
        self.gate = SqueezeExcite(channels, reduction=8)
        self.gate_proj = nn.Conv2d(channels, num_kernels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_feat = self.gate(x)                       # (B, C, 1, 1)
        weights = self.gate_proj(gate_feat)             # (B, N, 1, 1)
        weights = torch.softmax(weights, dim=1)

        out = 0.0
        for n, conv in enumerate(self.kernels):
            out = out + weights[:, n:n + 1] * conv(x)
        return F.relu(out, inplace=True)


class DynamicAttentionBlock(nn.Module):
    """NOTE: full pixel-to-pixel spatial attention (every pixel attending to
    every other pixel) is O((H*W)^2) and runs out of memory at real image
    resolutions (confirmed: a naive first version of this block crashed at
    256x256 during our own sanity check). Real image-restoration
    transformers (e.g. Restormer) instead attend *across channels*, which
    is O(C^2) and stays cheap regardless of image size -- that's what we
    use here, keeping DTMIR-Pro's "dynamic convolution as query" idea.
    """

    def __init__(self, channels: int, num_kernels: int = 4):
        super().__init__()
        self.query_conv = DynamicConv(channels, num_kernels=num_kernels)
        self.key_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.value_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query_conv(x).flatten(2)   # (B, C, HW)
        k = self.key_conv(x).flatten(2)
        v = self.value_conv(x).flatten(2)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.softmax((q @ k.transpose(1, 2)) * self.temperature, dim=-1)  # (B, C, C)
        out = attn @ v            # (B, C, HW)
        out = out.reshape(b, c, h, w)
        return x + out
