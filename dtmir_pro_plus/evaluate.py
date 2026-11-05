"""
Evaluate a trained DTMIR-Pro+ checkpoint and produce:
  - a qualitative side-by-side image grid  (degraded | restored | clean)
  - quantitative metrics: PSNR, SSIM, LPIPS, FID

Usage:
    python evaluate.py --checkpoint runs/ablation/B1/best_model.pt \
        --data-root data/sots/restoration --image-size 256 \
        --outdir runs/visual --num 50 --grid-rows 6

The config is read from the checkpoint itself (train.py saves it inside best_model.pt),
so you only need to pass the checkpoint path.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import structural_similarity_index_measure

from data.dataset import PairedRestorationDataset
from models.dtmir_plus import DTMIRPlus


def to_01(x):
    return (x + 1) / 2


def to_pil(x):
    arr = ((x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255)
    return arr.clip(0, 255).astype("uint8")


def psnr(a, b):
    a, b = to_01(a), to_01(b)
    mse = F.mse_loss(a, b)
    return float(10 * torch.log10(1.0 / (mse + 1e-8)))


def ssim(a, b):
    a, b = to_01(a), to_01(b)
    return float(structural_similarity_index_measure(a, b, data_range=1.0))


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    config = ckpt.get("config")
    if config is None:
        raise SystemExit("checkpoint has no 'config'; pass --config explicitly")
    model = DTMIRPlus(config).to(device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model, config


def save_grid(panels, outpath):
    n = len(panels)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        for c, img in enumerate(panels[r]):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(["Degraded", "Restored", "Clean"][c], fontsize=12)
    fig.tight_layout(pad=0.5)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative grid: {outpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--outdir", default="runs/visual")
    p.add_argument("--num", type=int, default=50, help="images to evaluate")
    p.add_argument("--grid-rows", type=int, default=6, help="rows in the saved image grid")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, config = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"config: weather={config.get('weather_code')} "
          f"translator={'trainable' if not config.get('freeze_translator') else 'frozen'} "
          f"routing={config.get('routing')}")

    dataset = PairedRestorationDataset(args.data_root, args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    psnr_vals, ssim_vals = [], []
    panels = []
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if count >= args.num:
                break
            degraded = batch["degraded"].to(device)
            clean = batch["clean"].to(device)
            restored = model(degraded)["restored"]

            psnr_vals.append(psnr(restored, clean))
            ssim_vals.append(ssim(restored, clean))
            lpips_metric.update(to_01(restored), to_01(clean))
            fid.update(to_01(clean), real=True)
            fid.update(to_01(restored), real=False)

            if len(panels) < args.grid_rows:
                panels.append((to_pil(degraded), to_pil(restored), to_pil(clean)))
            count += 1

    lpips_val = float(lpips_metric.compute())
    fid_val = float(fid.compute())
    avg_psnr = sum(psnr_vals) / len(psnr_vals)
    avg_ssim = sum(ssim_vals) / len(ssim_vals)

    results = {"psnr": avg_psnr, "ssim": avg_ssim, "lpips": lpips_val, "fid": fid_val}
    with open(os.path.join(args.outdir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results.keys()))
        w.writeheader()
        w.writerow(results)

    save_grid(panels, os.path.join(args.outdir, "qualitative.png"))

    print(f"PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | "
          f"LPIPS: {lpips_val:.4f} | FID: {fid_val:.2f}")
    print(f"Saved metrics to {os.path.join(args.outdir, 'metrics.csv')}")


if __name__ == "__main__":
    main()
