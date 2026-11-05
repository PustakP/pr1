"""
Two routing strategies for deciding which translation-encoder features to hand to
the restoration network.

Per the DTMIR-Pro paper, Similarity-Based Encoder Routing (SBFR) routes the
translation network's **last encoder layer** (DT_E), a single feature map (paper
Eq. 6-8). That single routed feature is later upsampled at every restoration
decoder level (Eq. 11). Both routers below therefore output one feature map, not a
per-scale list.

`HardSimilarityRouting`  -- DTMIR-Pro baseline (SBFR): computes SSIM on the Y
    channel between the input and each translated image, keeps ONLY the encoder
    features of the translation with the *lowest* SSIM (most different from the
    input -> dominant degradation), and discards the other two. Not differentiable
    end-to-end (argmin).

`SoftFeatureRouting` -- our replacement: learns a similarity score for each of the
    three streams and turns them into softmax weights, so all three streams
    contribute (weighted) and the routing step is differentiable.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure


def _rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    # img in [-1, 1], shape (B, 3, H, W) -> Y channel, per DTMIR-Pro Eq. 6
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _to_01(x: torch.Tensor) -> torch.Tensor:
    # torchmetrics SSIM expects [0, 1]; our images live in [-1, 1]
    return (x + 1) / 2


def _per_sample_ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample SSIM on the luminance channel.

    `reduction=None` returns one value per batch element, which is what the routing
    decision needs (the library default returns a single mean).
    """
    a = _to_01(_rgb_to_y(a))
    b = _to_01(_rgb_to_y(b))
    return structural_similarity_index_measure(a, b, data_range=1.0, reduction=None)


class HardSimilarityRouting(nn.Module):
    """Baseline SBFR — kept for controlled comparisons against our change."""

    def forward(self, input_img: torch.Tensor, translated: List[torch.Tensor],
                encoder_feats: List[torch.Tensor]) -> torch.Tensor:
        scores = torch.stack(
            [_per_sample_ssim(input_img, t) for t in translated], dim=1
        )  # (B, 3)
        chosen = torch.argmin(scores, dim=1)  # lowest SSIM = most different = dominant degradation

        # gather the chosen stream's last-encoder-layer feature (not differentiable
        # w.r.t. the choice itself — that limitation is what SoftFeatureRouting fixes)
        stacked = torch.stack(encoder_feats, dim=1)  # (B, 3, C, H, W)
        idx = chosen.view(-1, 1, 1, 1, 1).expand(-1, 1, *stacked.shape[2:])
        return stacked.gather(1, idx).squeeze(1)


class SoftFeatureRouting(nn.Module):
    """Our change: a small learned similarity head produces one score per stream;
    softmax turns those into blending weights so every stream contributes, and the
    whole routing step is differentiable. Still outputs a single routed feature.
    """

    def __init__(self, channels: int):
        super().__init__()
        # compares each translated stream's last-encoder features against the
        # input's own (bottleneck-scale) features to produce a per-stream score
        self.score_head = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, input_feat: torch.Tensor,
                encoder_feats: List[torch.Tensor]) -> tuple:
        scores = []
        for feat in encoder_feats:
            pair = torch.cat([input_feat, feat], dim=1)
            scores.append(self.score_head(pair).flatten(1))  # (B, 1)
        scores = torch.cat(scores, dim=1)                     # (B, 3)
        weights = torch.softmax(scores, dim=1)                # (B, 3)

        stacked = torch.stack(encoder_feats, dim=1)           # (B, 3, C, H, W)
        w = weights.view(*weights.shape, 1, 1, 1)
        return (stacked * w).sum(dim=1), weights              # blended feat + weights (logging)


class RoutingInputStem(nn.Module):
    """Small independent feature extractor that describes the *input* image at the
    same channels/resolution as the translation encoder's last layer (DT_E).

    This replaces the earlier shortcut of re-using the translator encoder with a
    weather code of class 0, so the soft router's "input features" are a clean
    descriptor of the degraded input.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 24,
                 num_scales: int = 3):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for _ in range(num_scales):
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
            ch *= 2
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.act(self.layers[0](x))
        for layer in self.layers[1:]:
            feat = self.act(layer(feat))
        return feat
