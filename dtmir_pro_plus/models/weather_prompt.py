"""
Weather Coded Prompt (WCP) block.

DTMIR-Pro baseline: the "weather code" fed into this block is a fixed
one-hot vector -> [1,0,0] for haze, [0,1,0] for rain, [0,0,1] for snow.

Our change: replace the one-hot code with a *continuous* embedding learned
from the input image itself, so a single image can express "a bit hazy AND
a bit rainy" instead of being forced into one of three buckets.

Both modes share the same downstream block (CBIN-based fusion of the code
into a learnable prompt tensor), so switching `code_mode` should be a
drop-in change for the rest of the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CentralBiasingInstanceNorm(nn.Module):
    """CBIN: instance-normalizes the prompt, then re-biases it using an
    affine transform of the weather code. Matches the CBIN formula used in
    DTMIR-Pro / SingleGAN."""

    def __init__(self, num_channels: int, code_dim: int):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(num_channels, affine=False)
        self.code_to_bias = nn.Linear(code_dim, num_channels)

    def forward(self, prompt: torch.Tensor, weather_code: torch.Tensor) -> torch.Tensor:
        # prompt: (B, C, H, W), weather_code: (B, code_dim)
        normed = self.inorm(prompt)
        bias = torch.tanh(self.code_to_bias(weather_code))  # (B, C)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return normed + bias


class WeatherEmbeddingNet(nn.Module):
    """Our replacement for the fixed one-hot weather code.

    Looks at the input image's own features and produces a continuous
    vector describing its weather condition(s). Trained jointly with the
    rest of the pipeline -- nothing here is hand-labeled.
    """

    def __init__(self, in_channels: int, code_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.to_code = nn.Linear(64, code_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).flatten(1)
        # no softmax / one-hot constraint -- a free continuous vector
        return self.to_code(feat)


class WeatherCodedPrompt(nn.Module):
    """WCP block: fuses learnable prompt tensors with the input features,
    modulated by the (one-hot or continuous) weather code via CBIN.
    """

    def __init__(self, in_channels: int, prompt_dim: int, prompt_size: int,
                 code_dim: int):
        super().__init__()
        self.prompt = nn.Parameter(torch.randn(1, prompt_dim, prompt_size, prompt_size) * 0.02)
        self.to_weights = nn.Conv2d(in_channels, prompt_dim, kernel_size=1)
        self.cbin = CentralBiasingInstanceNorm(prompt_dim, code_dim)

    def forward(self, features: torch.Tensor, weather_code: torch.Tensor) -> torch.Tensor:
        b = features.shape[0]
        gap = F.adaptive_avg_pool2d(features, 1)          # (B, C, 1, 1)
        weights = torch.softmax(self.to_weights(gap), dim=1)  # (B, prompt_dim, 1, 1)

        prompt = self.prompt.expand(b, -1, -1, -1)
        prompt = F.interpolate(prompt, size=features.shape[-2:], mode="bilinear",
                                align_corners=False)
        weighted_prompt = weights * prompt
        return self.cbin(weighted_prompt, weather_code)


def build_weather_code(mode: str, onehot_index: torch.Tensor = None,
                        embedding_net: WeatherEmbeddingNet = None,
                        image: torch.Tensor = None, num_classes: int = 3,
                        code_dim: int = 16) -> torch.Tensor:
    """Convenience switch used by translation_net.py.

    mode="onehot"     -> baseline DTMIR-Pro behaviour (needs onehot_index)
    mode="continuous" -> our change (needs embedding_net + image)
    """
    if mode == "onehot":
        assert onehot_index is not None
        return F.one_hot(onehot_index, num_classes=num_classes).float()
    elif mode == "continuous":
        assert embedding_net is not None and image is not None
        return embedding_net(image)
    else:
        raise ValueError(f"Unknown weather-code mode: {mode}")
