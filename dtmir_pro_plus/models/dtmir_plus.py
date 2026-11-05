"""
Top-level model. Wires together:
  input image
    -> weather code (one-hot baseline OR our continuous embedding)
    -> translation network (frozen baseline OR our jointly-trained version)
    -> [haze-look, rain-look, snow-look] images + last-encoder features (DT_E)
    -> routing (hard SBFR baseline OR our soft blending)
    -> restoration network
    -> restored image

Every one of our three proposed changes is a config flag here, so the baseline and
our version are literally the same class with different settings -- makes ablations
straightforward later.
"""

import torch
import torch.nn as nn

from .weather_prompt import WeatherEmbeddingNet, build_weather_code
from .translation_net import TranslationNetwork
from .routing import HardSimilarityRouting, SoftFeatureRouting, RoutingInputStem
from .restoration_net import RestorationNetwork


class DTMIRPlus(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        code_dim = config.get("code_dim", 16)
        base_channels = config.get("base_channels", 24)
        num_scales = config.get("num_scales", 3)

        self.weather_code_mode = config.get("weather_code", "continuous")  # "onehot" | "continuous"
        self.routing_mode = config.get("routing", "soft")                  # "hard" | "soft"
        self.freeze_translator = config.get("freeze_translator", False)

        if self.weather_code_mode == "continuous":
            self.weather_embed = WeatherEmbeddingNet(in_channels=3, code_dim=code_dim)
            self.onehot_proj = None
        else:
            self.weather_embed = None
            # one-hot codes are only 3-dim (haze/rain/snow); project up to the
            # same code_dim used everywhere else so both modes are drop-in compatible
            self.onehot_proj = nn.Linear(3, code_dim)

        self.translator = TranslationNetwork(
            in_channels=3, base_channels=base_channels,
            num_scales=num_scales, code_dim=code_dim,
        )
        self.translator.set_frozen(self.freeze_translator)

        # last encoder layer channels (DT_E) = base_channels * 2^num_scales
        self.routed_channels = self.translator.encoder.out_channels

        if self.routing_mode == "hard":
            self.router = HardSimilarityRouting()
            self.input_stem = None
        else:
            self.router = SoftFeatureRouting(channels=self.routed_channels)
            # independent descriptor of the input image for the soft router
            self.input_stem = RoutingInputStem(3, base_channels, num_scales)

        # num_stages must match num_scales so the routed feature upsamples across
        # the same number of decoder levels as the translation encoder scales
        self.restorer = RestorationNetwork(
            feat_channels=base_channels, num_stages=num_scales,
            routed_channels=self.routed_channels,
        )

    def _weather_code(self, batch_size: int, img: torch.Tensor, class_idx: torch.Tensor,
                       device) -> torch.Tensor:
        raw_code = build_weather_code(
            mode=self.weather_code_mode,
            onehot_index=class_idx,
            embedding_net=self.weather_embed,
            image=img,
            num_classes=3,
            code_dim=self.config.get("code_dim", 16),
        )
        if self.weather_code_mode == "onehot":
            raw_code = self.onehot_proj(raw_code)  # -> code_dim, matches CBIN's expectations
        return raw_code

    def forward(self, img: torch.Tensor):
        device = img.device
        b = img.shape[0]

        outputs = {"translated": {}}
        translated_imgs = []
        encoder_feats = []

        for name, idx in (("haze", 0), ("rain", 1), ("snow", 2)):
            class_idx = torch.full((b,), idx, dtype=torch.long, device=device)
            code = self._weather_code(b, img, class_idx, device)

            if self.freeze_translator:
                with torch.no_grad():
                    t_img, t_feat = self.translator(img, code)
            else:
                t_img, t_feat = self.translator(img, code)

            outputs["translated"][name] = t_img
            translated_imgs.append(t_img)
            encoder_feats.append(t_feat)

        if self.routing_mode == "hard":
            routed_feat = self.router(img, translated_imgs, encoder_feats)
            outputs["routing_weights"] = None  # not applicable — hard choice
        else:
            input_feat = self.input_stem(img)
            routed_feat, weights = self.router(input_feat, encoder_feats)
            outputs["routing_weights"] = weights

        restored = self.restorer(
            outputs["translated"]["haze"],
            outputs["translated"]["rain"],
            outputs["translated"]["snow"],
            routed_feat,
        )
        outputs["restored"] = restored
        return outputs
