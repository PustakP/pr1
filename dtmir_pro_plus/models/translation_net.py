"""
Domain translation network: turns one degraded input image into three
weather-translated versions (haze-look, rain-look, snow-look), using a
shared encoder-decoder guided by Weather Coded Prompts at each scale.

Our change lives here as a simple but important switch: `freeze_translator`.
- DTMIR-Pro baseline: train this network once, then call
  `.freeze()` and never update it again while training the restoration
  network.
- Our version: leave it trainable (`freeze_translator=False` in the
  config), so gradients from the restoration loss flow back into it.
  See `dtmir_plus.py` for where that wiring happens.
"""

import torch
import torch.nn as nn

from .dynamic_attention import DynamicAttentionBlock
from .weather_prompt import WeatherCodedPrompt


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class TranslationEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_scales: int = 4, code_dim: int = 16):
        super().__init__()
        self.stem = ConvBlock(in_channels, base_channels)

        self.wcp_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()

        ch = base_channels
        for _ in range(num_scales):
            self.wcp_blocks.append(WeatherCodedPrompt(
                in_channels=ch, prompt_dim=ch, prompt_size=16, code_dim=code_dim))
            self.attn_blocks.append(DynamicAttentionBlock(ch))
            self.down_blocks.append(ConvBlock(ch, ch * 2, stride=2))
            ch *= 2

        self.out_channels = ch

    def forward(self, x: torch.Tensor, weather_code: torch.Tensor):
        feat = self.stem(x)
        skips = []
        for wcp, attn, down in zip(self.wcp_blocks, self.attn_blocks, self.down_blocks):
            prompt = wcp(feat, weather_code)
            feat = feat + prompt           # fuse weather-coded prompt into features
            feat = attn(feat)
            skips.append(feat)
            feat = down(feat)
        return feat, skips


class TranslationDecoder(nn.Module):
    def __init__(self, in_channels: int, num_scales: int = 4, out_channels: int = 3):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        ch = in_channels
        for _ in range(num_scales):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch //= 2
        self.to_rgb = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, feat: torch.Tensor, skips):
        for up, skip in zip(self.up_blocks, reversed(skips)):
            feat = up(feat)
            if feat.shape[-2:] != skip.shape[-2:]:
                feat = nn.functional.interpolate(feat, size=skip.shape[-2:], mode="nearest")
            feat = feat + skip
        return torch.tanh(self.to_rgb(feat))


class TranslationNetwork(nn.Module):
    """Single shared network for haze/rain/snow translation (unlike the
    original domain-translation paper [40] this builds on, which trains a
    separate network per weather type)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32,
                 num_scales: int = 4, code_dim: int = 16):
        super().__init__()
        self.encoder = TranslationEncoder(in_channels, base_channels, num_scales, code_dim)
        self.decoder = TranslationDecoder(self.encoder.out_channels, num_scales, in_channels)

    def forward(self, x: torch.Tensor, weather_code: torch.Tensor):
        feat, skips = self.encoder(x, weather_code)
        translated = self.decoder(feat, skips)
        # `feat` is the LAST encoder layer (bottleneck) = DT_E, which SBFR routes
        # (paper Eq. 8). The skips are only used internally by the decoder.
        return translated, feat

    def set_frozen(self, frozen: bool):
        """Baseline = True (freeze after pretraining). Ours = False."""
        for p in self.parameters():
            p.requires_grad = not frozen
