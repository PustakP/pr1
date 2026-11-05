"""
Automates the B0-B4 ablation ladder (the core comparable result):

  B0  one-hot     frozen    hard   (baseline reproduction)
  B1  one-hot     trainable hard   (unfreeze translator only)
  B2  continuous  frozen    hard   (continuous embedding only)
  B3  one-hot     frozen    soft   (soft routing only)
  B4  continuous  trainable soft   (full proposal)

For each unique weather-code mode (one-hot / continuous) it pretrains the
stage-1 translator once, then runs stage 2 for every row that uses that
mode. Results are collected into a single summary CSV + Markdown table.

Usage:
    python run_ablation.py --data-root data/synthetic \
        --epochs-stage1 2 --epochs-stage2 10 --outdir runs/ablation --use-vgg 0
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROWS = [
    ("B0", "onehot", True, "hard"),
    ("B1", "onehot", False, "hard"),
    ("B2", "continuous", True, "hard"),
    ("B3", "onehot", True, "soft"),
    ("B4", "continuous", False, "soft"),
]

BASE = dict(code_dim=16, base_channels=26, num_scales=3,
            image_size=256, batch_size=4, lr=1.0e-3,
            l1_weight=1.0, contrastive_weight=0.75)


def run(cmd, cwd="."):
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def write_yaml(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def stage1_checkpoint(weather_code, args, workdir):
    path = workdir / f"translator_{weather_code}.pt"
    if path.exists():
        print(f"Reusing {path}")
        return path

    cfg = dict(BASE)
    cfg.update(
        weather_code=weather_code,
        freeze_translator=False,
        routing="soft",
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs_translation=args.epochs_stage1,
        use_vgg_loss=bool(args.use_vgg),
        data_root=str(args.data_root / "translation"),
    )
    cfg_path = workdir / f"translation_{weather_code}.yaml"
    write_yaml(cfg_path, cfg)

    run([sys.executable, "train_translation.py", "--config", str(cfg_path)], cwd=args.repo)
    os.replace(args.repo / "translator_pretrained.pt", path)
    return path


def best_metrics(rundir):
    best = {"val_psnr": -1.0, "val_ssim": 0.0}
    with open(rundir / "metrics.csv") as f:
        for row in csv.DictReader(f):
            if float(row["val_psnr"]) > best["val_psnr"]:
                best = {"val_psnr": float(row["val_psnr"]),
                        "val_ssim": float(row["val_ssim"])}
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs-stage1", type=int, default=100)
    p.add_argument("--epochs-stage2", type=int, default=300)
    p.add_argument("--use-vgg", type=int, default=1)
    p.add_argument("--outdir", type=Path, default=Path("runs/ablation"))
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--rows", default=None,
                   help="comma-separated rows to run, e.g. B1 or B0,B1 (default: all)")
    args = p.parse_args()

    rows = ROWS
    if args.rows:
        wanted = {r.strip().upper() for r in args.rows.split(",")}
        rows = [r for r in ROWS if r[0] in wanted]

    workdir = args.outdir
    workdir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, weather_code, freeze, routing in rows:
        ckpt = stage1_checkpoint(weather_code, args, workdir)
        rundir = workdir / name
        rundir.mkdir(exist_ok=True)

        cfg = dict(BASE)
        cfg.update(
            weather_code=weather_code,
            freeze_translator=freeze,
            routing=routing,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs_restoration=args.epochs_stage2,
            use_vgg_loss=bool(args.use_vgg),
            val_split=0.1,
            checkpoint_every=10,
            data_root=str(args.data_root / "restoration"),
        )
        cfg_path = workdir / f"stage2_{name}.yaml"
        write_yaml(cfg_path, cfg)

        run([sys.executable, "train.py", "--config", str(cfg_path),
             "--checkpoint", str(ckpt), "--outdir", str(rundir),
             "--epochs", str(args.epochs_stage2)], cwd=args.repo)

        best = best_metrics(rundir)
        results.append({"row": name, "weather": weather_code,
                        "translator": "trainable" if not freeze else "frozen",
                        "routing": routing, "psnr": best["val_psnr"],
                        "ssim": best["val_ssim"]})

    summary = workdir / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n=== Ablation summary ===")
    print("| Row | Weather code | Translator | Routing | PSNR (dB) | SSIM |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['row']} | {r['weather']} | {r['translator']} | {r['routing']} "
              f"| {r['psnr']:.2f} | {r['ssim']:.4f} |")
    print(f"\nSaved {summary}")


if __name__ == "__main__":
    main()
