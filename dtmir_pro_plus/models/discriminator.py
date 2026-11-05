"""
Discriminators for adversarially training the translation network (one per
weather domain: haze, rain, snow), matching DTMIR-Pro's stated setup
("trained adversarially having discriminator networks D_i").

A standard PatchGAN discriminator -- classifies overlapping patches of the
image as real/fake rather than the whole image at once, which is the usual
choice for this kind of image-to-image translation and is cheap to train.
"""

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_channels
        for i in range(1, num_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, next_ch, 4, stride=2, padding=1),
                nn.InstanceNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]  # per-patch real/fake logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 1, H', W') -- NOT passed through sigmoid (use BCEWithLogits)


class WeatherDiscriminators(nn.Module):
    """One discriminator per weather domain, bundled together for convenience."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.haze = PatchDiscriminator(in_channels)
        self.rain = PatchDiscriminator(in_channels)
        self.snow = PatchDiscriminator(in_channels)

    def forward(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return getattr(self, name)(x)
