"""
Stage 2: train the restoration network (and, optionally, fine-tune the
translator) on paired degraded/clean images.

Before training it loads the stage-1 translator checkpoint, so stage 2 no
longer starts from a fresh translator (the earlier gap). The translator is
then either frozen (baseline) or left trainable (our joint fine-tuning
change) according to `freeze_translator` in the config.

Also provides:
  - cosine-annealing learning rate (paper's schedule)
  - periodic + best-model checkpointing
  - PSNR/SSIM validation on a held-out split
  - CSV logging of losses and metrics
"""

import argparse
import csv
import os

import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import structural_similarity_index_measure

from data.dataset import PairedRestorationDataset
from losses import RestorationLoss
from models.dtmir_plus import DTMIRPlus


def to_01(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = to_01(a), to_01(b)
    mse = F.mse_loss(a, b)
    return float(10 * torch.log10(1.0 / (mse + 1e-8)))


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = to_01(a), to_01(b)
    return float(structural_similarity_index_measure(a, b, data_range=1.0))


def load_stage1(model: DTMIRPlus, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    if "translator" in ckpt:
        model.translator.load_state_dict(ckpt["translator"])
        if model.weather_embed is not None and ckpt.get("weather_embed") is not None:
            model.weather_embed.load_state_dict(ckpt["weather_embed"])
        if model.onehot_proj is not None and ckpt.get("onehot_proj") is not None:
            model.onehot_proj.load_state_dict(ckpt["onehot_proj"])
    else:
        # bare state dict (older stage-1 format)
        model.translator.load_state_dict(ckpt)
    return model


def validate(model, loader, device):
    model.eval()
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            restored = model(degraded)["restored"]
            psnr_sum += psnr(restored, clean) * degraded.shape[0]
            ssim_sum += ssim(restored, clean) * degraded.shape[0]
            n += degraded.shape[0]
    model.train()
    return psnr_sum / n, ssim_sum / n


def main(config_path: str, checkpoint_path: str, outdir: str, epochs_override: int = None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DTMIRPlus(config).to(device)
    if os.path.exists(checkpoint_path):
        model = load_stage1(model, checkpoint_path, device)
        print(f"Loaded stage-1 checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: stage-1 checkpoint not found at {checkpoint_path}; "
              "starting stage 2 from a random translator (not recommended)")

    criterion = RestorationLoss(
        config.get("l1_weight", 1.0),
        config.get("contrastive_weight", 0.75),
        use_vgg=config.get("use_vgg_loss", True),
    ).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"]
    )

    dataset = PairedRestorationDataset(config["data_root"], config["image_size"])
    val_len = max(1, int(len(dataset) * config.get("val_split", 0.1)))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)

    epochs = epochs_override or config.get("epochs_restoration", 300)
    steps_per_epoch = len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

    log_path = os.path.join(outdir, "metrics.csv")
    fieldnames = ["epoch", "total", "l1", "contrastive", "val_psnr", "val_ssim", "lr"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best_psnr = -1.0
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)

            outputs = model(degraded)
            losses = criterion(outputs, clean)

            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            scheduler.step()

        val_psnr, val_ssim = validate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]

        row = {
            "epoch": epoch,
            "total": losses["total"].item(),
            "l1": losses["l1"].item(),
            "contrastive": losses["contrastive"].item(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "lr": lr,
        }
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        print(f"epoch {epoch}: total={row['total']:.4f} l1={row['l1']:.4f} "
              f"contrastive={row['contrastive']:.4f} val_psnr={val_psnr:.2f} "
              f"val_ssim={val_ssim:.4f} lr={lr:.2e}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_psnr": val_psnr, "config": config},
                       os.path.join(outdir, "best_model.pt"))

        if (epoch + 1) % config.get("checkpoint_every", 10) == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "optimizer": optimizer.state_dict(), "config": config},
                       os.path.join(outdir, f"checkpoint_epoch_{epoch}.pt"))

    print(f"Training finished. Best val PSNR: {best_psnr:.2f} dB. Log: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="translator_pretrained.pt")
    parser.add_argument("--outdir", default="runs")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.outdir, args.epochs)
