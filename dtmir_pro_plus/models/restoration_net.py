"""
Restoration network, following the DTMIR-Pro paper (Sec. 3.2):

  1. Each domain-translated image (haze/rain/snow) -> conv features F_H, F_R, F_S.
  2. Multi-Weather Feature Fusion (MWFF) -> pairwise fused features F_HR, F_RS, F_HS.
  3. Each fused feature passes through (shared) encoder layers of dynamic attention +
     strided convolution, producing per-level features E_HR^i, E_RS^i, E_HS^i.
  4. The three encoded bottlenecks are concatenated and decoded.
  5. At every decoder level, the MWF encoder features are concatenated with the
     SBFR-routed feature E_DT (upsampled to that level) via a 3x3 conv — paper Eq. 11:

       F_Skip_i = Conv_{3x3}(<E_HR^i, E_RS^i, E_HS^i, up(E_DT)>)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_attention import DynamicAttentionBlock
from .fusion import MultiWeatherFusion


class FeatureExtractor(nn.Module):
    """Turns a single translated RGB image into a feature map (F_H / F_R / F_S)."""

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderStage(nn.Module):
    """One level of the (shared) restoration encoder: DAB + strided conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn = DynamicAttentionBlock(channels)
        self.down = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.attn(x)
        return self.down(x), x  # downsampled, skip


class EncoderBranch(nn.Module):
    """Shared multi-level encoder applied to each MWF-fused feature."""

    def __init__(self, feat_channels: int, num_stages: int):
        super().__init__()
        self.stages = nn.ModuleList()
        ch = feat_channels
        for _ in range(num_stages):
            self.stages.append(EncoderStage(ch))
            ch *= 2

    def forward(self, x: torch.Tensor):
        skips = []
        for stage in self.stages:
            x, skip = stage(x)
            skips.append(skip)
        return x, skips


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_in_channels: int):
        super().__init__()
        out_ch = in_channels // 2
        self.up = nn.ConvTranspose2d(in_channels, out_ch, 4, stride=2, padding=1)
        self.skip_proj = nn.Conv2d(skip_in_channels, out_ch, 3, padding=1)
        self.attn = DynamicAttentionBlock(out_ch)

    def forward(self, x: torch.Tensor, branch_skips, e_dt: torch.Tensor):
        x = self.up(x)
        if e_dt.shape[-2:] != x.shape[-2:]:
            e_dt = F.interpolate(e_dt, size=x.shape[-2:], mode="bilinear",
                                 align_corners=False)
        skip = self.skip_proj(torch.cat([*branch_skips, e_dt], dim=1))
        x = x + skip
        return self.attn(x)


class RestorationNetwork(nn.Module):
    def __init__(self, feat_channels: int = 32, num_stages: int = 3,
                 routed_channels: int = None):
        super().__init__()
        self.num_stages = num_stages

        self.extract_h = FeatureExtractor(3, feat_channels)
        self.extract_r = FeatureExtractor(3, feat_channels)
        self.extract_s = FeatureExtractor(3, feat_channels)

        self.mwff = MultiWeatherFusion(feat_channels)

        self.encoder = EncoderBranch(feat_channels, num_stages)

        bottleneck_ch = feat_channels * (2 ** num_stages)
        routed_channels = routed_channels or bottleneck_ch
        self.agg = nn.Conv2d(bottleneck_ch * 3, bottleneck_ch, kernel_size=1)

        self.dec_stages = nn.ModuleList()
        ch = bottleneck_ch
        for level in range(num_stages):
            branch_ch = feat_channels * (2 ** (num_stages - 1 - level))
            skip_in_ch = 3 * branch_ch + routed_channels
            self.dec_stages.append(DecoderStage(ch, skip_in_ch))
            ch //= 2

        self.to_rgb = nn.Conv2d(ch, 3, kernel_size=3, padding=1)

    def forward(self, img_haze: torch.Tensor, img_rain: torch.Tensor, img_snow: torch.Tensor,
                routed_feat: torch.Tensor):
        f_h = self.extract_h(img_haze)
        f_r = self.extract_r(img_rain)
        f_s = self.extract_s(img_snow)

        f_hr, f_rs, f_hs = self.mwff(f_h, f_r, f_s)

        x_hr, skips_hr = self.encoder(f_hr)
        x_rs, skips_rs = self.encoder(f_rs)
        x_hs, skips_hs = self.encoder(f_hs)

        x = self.agg(torch.cat([x_hr, x_rs, x_hs], dim=1))

        for level, stage in enumerate(self.dec_stages):
            i = self.num_stages - 1 - level
            branch_skips = [skips_hr[i], skips_rs[i], skips_hs[i]]
            x = stage(x, branch_skips, routed_feat)

        return torch.tanh(self.to_rgb(x))
