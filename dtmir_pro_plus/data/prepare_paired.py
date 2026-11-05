"""
Builds the repo's expected layout from a paired dataset whose input/ and target/
folders already share filenames (e.g. Rain13K, or any derain/dehaze pair set).

Produces:
    <outdir>/restoration/degraded/   <- input (degraded) images
    <outdir>/restoration/clean/      <- target (clean) images
    <outdir>/translation/input/      <- input images
    <outdir>/translation/haze_gt/    <- identity if identity-domain=haze, else synthesized
    <outdir>/translation/rain_gt/    <- identity if identity-domain=rain, else synthesized
    <outdir>/translation/snow_gt/    <- identity if identity-domain=snow, else synthesized

Usage:
    python -m data.prepare_paired --input-dir /path/to/input --target-dir /path/to/target \
        --outdir data/rain --size 256 --num 98 --identity-domain rain
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from .make_synthetic_data import haze, rain, snow

_IMG_EXTS = (".png", ".jpg", ".jpeg")


def _load_resized(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img).astype("uint8")


def _names(directory):
    return sorted(p.name for p in directory.glob("*") if p.suffix.lower() in _IMG_EXTS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--target-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num", type=int, default=98)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--identity-domain", default="rain", choices=("haze", "rain", "snow"))
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    assert input_dir.is_dir() and target_dir.is_dir()

    out = Path(args.outdir)
    trans = out / "translation"
    resto = out / "restoration"
    for d in [trans / "input", trans / "haze_gt", trans / "rain_gt", trans / "snow_gt",
              resto / "degraded", resto / "clean"]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    input_names = _names(input_dir)
    target_names = set(_names(target_dir))

    pairs = [n for n in input_names if n in target_names][:args.num]
    if not pairs:
        raise SystemExit("No matching input/target filenames found")

    for name in pairs:
        inp = _load_resized(input_dir / name, args.size)
        tgt = _load_resized(target_dir / name, args.size)

        domains = {}
        domains["haze"] = inp if args.identity_domain == "haze" else haze(tgt, rng)
        domains["rain"] = inp if args.identity_domain == "rain" else rain(tgt, rng)
        domains["snow"] = inp if args.identity_domain == "snow" else snow(tgt, rng)

        Image.fromarray(inp).save(trans / "input" / name)
        Image.fromarray(domains["haze"]).save(trans / "haze_gt" / name)
        Image.fromarray(domains["rain"]).save(trans / "rain_gt" / name)
        Image.fromarray(domains["snow"]).save(trans / "snow_gt" / name)

        Image.fromarray(inp).save(resto / "degraded" / name)
        Image.fromarray(tgt).save(resto / "clean" / name)

    print(f"Prepared {len(pairs)} paired samples into {out} (identity domain: {args.identity_domain})")


if __name__ == "__main__":
    main()
