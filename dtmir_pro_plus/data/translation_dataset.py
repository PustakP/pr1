"""
Dataset stub for translation-stage training. Unlike the restoration
dataset (one degraded image + one clean target), the translation network
needs an input image plus a *ground-truth version of it in each weather
domain* (haze/rain/snow), matching how DTMIR-Pro trains its translator:
SOTS for haze, Outdoor-Rain for rain, Comprehensive Snow Dataset for snow.

TODO: this assumes matching filenames across three separate ground-truth
folders (only realistic for synthetic datasets where the same clean scene
was rendered with each degradation type). Point `root` at wherever we've
assembled that once we pick a dataset combination.
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiWeatherTranslationDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.input_dir = self.root / "input"
        self.domain_dirs = {
            "haze": self.root / "haze_gt",
            "rain": self.root / "rain_gt",
            "snow": self.root / "snow_gt",
        }
        self.filenames = sorted(p.name for p in self.input_dir.glob("*.png"))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        item = {"input": self.transform(Image.open(self.input_dir / name).convert("RGB"))}
        for domain, d in self.domain_dirs.items():
            item[domain] = self.transform(Image.open(d / name).convert("RGB"))
        return item
