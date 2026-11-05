"""
Generates a small synthetic dataset so the full stage-1 -> stage-2 pipeline
can be smoke-tested end-to-end without downloading real data.

Layout produced:
  <outdir>/
    translation/
      input/      mixed degraded image
      haze_gt/    clean image rendered with haze
      rain_gt/    clean image rendered with rain
      snow_gt/    clean image rendered with snow
    restoration/
      degraded/   mixed degraded image
      clean/      ground-truth clean image

Nothing here is photorealistic — it exists purely to exercise the data
loaders, losses, checkpointing and metric logging before real datasets are
pointed at the same scripts.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype("uint8"))


def _clean_sample(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """A random but smooth-ish 'scene' (gradient + a few shapes)."""
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    base = (
        rng.uniform(0.2, 0.9) * (xx / w)[..., None]
        + rng.uniform(0.2, 0.9) * (yy / h)[..., None]
    )
    base = base * rng.uniform(0.5, 1.0, size=(1, 1, 3)) * 255
    img = np.clip(base, 0, 255).astype("uint8")

    # a few random rectangles/circles for structure
    for _ in range(rng.integers(2, 5)):
        if rng.random() < 0.5:
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            r = rng.integers(4, max(5, h // 5))
            color = rng.integers(0, 255, size=3)
            y0, y1 = max(0, cy - r), min(h, cy + r)
            x0, x1 = max(0, cx - r), min(w, cx + r)
            img[y0:y1, x0:x1] = color
        else:
            x0, y0 = rng.integers(0, w // 2), rng.integers(0, h // 2)
            x1, y1 = rng.integers(w // 2, w), rng.integers(h // 2, h)
            color = rng.integers(0, 255, size=3)
            img[y0:y1, x0:x1] = color
    return img


def haze(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.35, 0.65)
    a = rng.uniform(0.7, 0.95)
    return np.clip(img.astype("float32") * t + 255 * a * (1 - t), 0, 255).astype("uint8")


def rain(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(rng.integers(15, 40)):
        x = rng.integers(0, w)
        y = rng.integers(0, h // 2)
        length = rng.integers(5, max(6, h // 6))
        thickness = rng.integers(1, 2)
        dx = rng.integers(-2, 2)
        intensity = rng.integers(160, 240)
        for t in range(length):
            yy = min(h - 1, y + t)
            xx = min(w - 1, max(0, x + dx * t // 2))
            out[yy - thickness:yy + thickness, xx - thickness:xx + thickness] = intensity
    return out


def snow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(rng.integers(30, 80)):
        r = rng.integers(1, 3)
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        out[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = rng.integers(200, 255)
    return out


def mixed(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hz = haze(img, rng)
    return rain(hz, rng) if rng.random() < 0.5 else snow(hz, rng)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="data/synthetic")
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    root = Path(args.outdir)
    trans = root / "translation"
    resto = root / "restoration"
    for d in [trans / "input", trans / "haze_gt", trans / "rain_gt", trans / "snow_gt",
              resto / "degraded", resto / "clean"]:
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for i in range(args.num):
        name = f"{i:04d}.png"
        clean = _clean_sample(args.size, args.size, rng)
        hz = haze(clean, rng)
        rn = rain(clean, rng)
        sn = snow(clean, rng)
        inp = mixed(clean, rng)

        _to_img(inp).save(trans / "input" / name)
        _to_img(hz).save(trans / "haze_gt" / name)
        _to_img(rn).save(trans / "rain_gt" / name)
        _to_img(sn).save(trans / "snow_gt" / name)

        _to_img(inp).save(resto / "degraded" / name)
        _to_img(clean).save(resto / "clean" / name)

    print(f"Wrote {args.num} synthetic samples to {root}")


if __name__ == "__main__":
    main()
