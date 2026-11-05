"""
Dataset stub. Expects a directory structure like:

    root/
      degraded/  xxx.png
      clean/     xxx.png   (same filename, paired ground truth)

TODO: point `root` at whichever of SOTS / Outdoor-Rain / Comprehensive
Snow Dataset we start with, add resizing/augmentation to match the paper
(random crop, flips), and add a weather-type label if we want to log
per-weather-type metrics during validation.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedRestorationDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.degraded_dir = self.root / "degraded"
        self.clean_dir = self.root / "clean"
        self.filenames = sorted(p.name for p in self.degraded_dir.glob("*.png"))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1], matches tanh outputs
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        degraded = Image.open(self.degraded_dir / name).convert("RGB")
        clean = Image.open(self.clean_dir / name).convert("RGB")
        return {
            "degraded": self.transform(degraded),
            "clean": self.transform(clean),
        }
