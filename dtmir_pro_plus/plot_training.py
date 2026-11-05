"""
Reads a training run's metrics.csv and plots loss + validation curves.

Usage:
    python plot_training.py --csv runs/smoke/metrics.csv --out runs/smoke/curves.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="DTMIR-Pro+ smoke training")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["epoch"], df["total"], label="total", color="tab:blue")
    axes[0].plot(df["epoch"], df["l1"], label="L1", color="tab:orange")
    axes[0].plot(df["epoch"], df["contrastive"], label="contrastive", color="tab:green")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["val_psnr"], marker="o", color="tab:red")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")

    axes[2].plot(df["epoch"], df["val_ssim"], marker="o", color="tab:purple")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
