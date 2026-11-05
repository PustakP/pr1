"""
Builds the folder layout this repo expects from a SOTS (RESIDE) folder.

SOTS on disk (and the Hugging Face mirror) looks like:
    <sots_root>/gt/0001.png          (clean ground truth)
    <sots_root>/hazy/0001_0.8_0.2.jpg  (hazy version, name has haze params)

Note the hazy name carries a "_<beta>_<A>" suffix and a different extension, so the
two folders do NOT share exact filenames. We pair them by the leading stem.

This script produces:
    <outdir>/restoration/degraded/   <- hazy images (resized)
    <outdir>/restoration/clean/      <- clean ground truth (resized)
    <outdir>/translation/input/      <- hazy images
    <outdir>/translation/haze_gt/    <- hazy images (identity for haze)
    <outdir>/translation/rain_gt/    <- SYNTHESIZED rain on the clean image
    <outdir>/translation/snow_gt/    <- SYNTHESIZED snow on the clean image

NOTE: rain/snow translation targets are synthesized (not real ORD/CSD). This is a
documented simplification so a first SOTS-only restoration curve can be produced;
for the real paper-comparable run use actual ORD and CSD in the translation stage.

Usage:
    python -m data.prepare_sots --sots-root /path/to/SOTS/outdoor --outdir data/sots \
        --size 256 --num 400
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .make_synthetic_data import rain, snow  # reuse the simple degraders

_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _load_resized(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype("uint8")


def _image_names(directory):
    return sorted(p.name for p in directory.glob("*") if p.suffix.lower() in _IMG_EXTS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sots-root", required=True, help="folder containing hazy/ and gt/")
    p.add_argument("--outdir", default="data/sots")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num", type=int, default=400, help="max image pairs to use")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.sots_root)
    hazy_dir = root / "hazy"
    gt_dir = root / "gt"
    assert hazy_dir.is_dir() and gt_dir.is_dir(), \
        f"expected {hazy_dir} and {gt_dir} to exist"

    out = Path(args.outdir)
    trans = out / "translation"
    resto = out / "restoration"
    for d in [trans / "input", trans / "haze_gt", trans / "rain_gt", trans / "snow_gt",
              resto / "degraded", resto / "clean"]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    hazy_names = _image_names(hazy_dir)
    gt_names = _image_names(gt_dir)

    pairs = []
    for gt_name in gt_names:
        stem = Path(gt_name).stem  # e.g. "0001"
        matches = [h for h in hazy_names if h.startswith(stem + "_")]
        if matches:
            pairs.append((gt_name, matches[0]))

    pairs = pairs[:args.num]
    if not pairs:
        raise SystemExit(
            f"Could not pair any hazy/gt images under {root}. "
            f"Found {len(gt_names)} gt and {len(hazy_names)} hazy files.")

    for i, (gt_name, hazy_name) in enumerate(pairs):
        stem = Path(gt_name).stem
        out_name = f"{stem}.png"

        hazy = _load_resized(hazy_dir / hazy_name, args.size)
        clean = _load_resized(gt_dir / gt_name, args.size)

        Image.fromarray(hazy).save(trans / "input" / out_name)
        Image.fromarray(hazy).save(trans / "haze_gt" / out_name)
        Image.fromarray(rain(clean, rng)).save(trans / "rain_gt" / out_name)
        Image.fromarray(snow(clean, rng)).save(trans / "snow_gt" / out_name)

        Image.fromarray(hazy).save(resto / "degraded" / out_name)
        Image.fromarray(clean).save(resto / "clean" / out_name)

    print(f"Prepared {len(pairs)} SOTS pairs into {out}")


if __name__ == "__main__":
    main()
