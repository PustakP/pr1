"""
Stage 1: pretrain the translation network adversarially (matches
DTMIR-Pro's "translation network is trained for 100 epochs" step, before
the restoration network is trained on top of it).

Usage: python train_translation.py --config configs/config.yaml

After this finishes, `train.py` (stage 2) loads the resulting checkpoint
and either freezes it (baseline) or keeps it trainable (our change,
`freeze_translator: false` in the config).
"""

import argparse

import torch
import yaml
from torch.utils.data import DataLoader

from data.translation_dataset import MultiWeatherTranslationDataset
from losses import TranslationLoss
from models.discriminator import WeatherDiscriminators
from models.translation_net import TranslationNetwork
from models.weather_prompt import WeatherEmbeddingNet, build_weather_code


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    code_dim = config.get("code_dim", 16)

    translator = TranslationNetwork(
        in_channels=3, base_channels=config.get("base_channels", 32),
        num_scales=config.get("num_scales", 4), code_dim=code_dim,
    ).to(device)
    discriminators = WeatherDiscriminators().to(device)

    weather_code_mode = config.get("weather_code", "continuous")
    if weather_code_mode == "continuous":
        weather_embed = WeatherEmbeddingNet(in_channels=3, code_dim=code_dim).to(device)
        onehot_proj = None
        gen_params = list(translator.parameters()) + list(weather_embed.parameters())
    else:
        weather_embed = None
        onehot_proj = torch.nn.Linear(3, code_dim).to(device)
        gen_params = list(translator.parameters()) + list(onehot_proj.parameters())

    criterion = TranslationLoss(use_vgg=config.get("use_vgg_loss", True)).to(device)
    opt_g = torch.optim.Adam(gen_params, lr=config["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminators.parameters(), lr=config["lr"], betas=(0.5, 0.999))

    dataset = MultiWeatherTranslationDataset(config["data_root"], config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    domain_index = {"haze": 0, "rain": 1, "snow": 2}

    for epoch in range(config["epochs_translation"]):
        for batch in loader:
            x = batch["input"].to(device)
            gt = {name: batch[name].to(device) for name in domain_index}
            b = x.shape[0]

            translated = {}
            for name, idx in domain_index.items():
                class_idx = torch.full((b,), idx, dtype=torch.long, device=device)
                code = build_weather_code(
                    mode=weather_code_mode, onehot_index=class_idx,
                    embedding_net=weather_embed, image=x,
                    num_classes=3, code_dim=code_dim,
                )
                if weather_code_mode == "onehot":
                    code = onehot_proj(code)
                t_img, _ = translator(x, code)
                translated[name] = t_img

            # ---- discriminator step ----
            real_scores = {n: discriminators(n, gt[n]) for n in domain_index}
            fake_scores = {n: discriminators(n, translated[n].detach()) for n in domain_index}
            d_loss = criterion.discriminator_loss(real_scores, fake_scores)

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # ---- generator (translator) step ----
            fake_scores_for_g = {n: discriminators(n, translated[n]) for n in domain_index}
            g_losses = criterion.generator_loss(translated, gt, discriminators, fake_scores_for_g)

            opt_g.zero_grad()
            g_losses["total"].backward()
            opt_g.step()

        print(f"epoch {epoch}: g_total={g_losses['total'].item():.4f} "
              f"adv={g_losses['adversarial'].item():.4f} "
              f"contrastive={g_losses['contrastive'].item():.4f} "
              f"d_loss={d_loss.item():.4f}")

    checkpoint = {
        "translator": translator.state_dict(),
        "weather_embed": weather_embed.state_dict() if weather_embed is not None else None,
        "onehot_proj": onehot_proj.state_dict() if onehot_proj is not None else None,
        "config": config,
    }
    torch.save(checkpoint, "translator_pretrained.pt")
    print("Saved translator_pretrained.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
