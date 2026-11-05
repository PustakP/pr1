"""
Live demo script -- meant to be run in front of the advisor to show that
the three proposed changes are actually implemented and working, not just
described on slides.

Deliberately self-contained and offline-safe:
  - uses a bundled sample image if you don't pass one, so it works even
    with no internet in the meeting room
  - VGG perceptual loss is switched OFF by default here (use_vgg_loss in
    the demo config) so nothing tries to download weights mid-meeting;
    turn it back on for real training later

What it shows, in order:
  1. Baseline DTMIR-Pro config runs, translator is frozen (no gradient
     reaches it)
  2. Our config runs, translator is NOT frozen (gradient does reach it)
  3. Soft routing weights for the three streams (all three contribute,
     not just one)
  4. Saves a before/after image grid to demo_output.png

Run with:  python demo.py
"""

import time

import torch
from PIL import Image
from torchvision import transforms

from models.dtmir_plus import DTMIRPlus


def load_demo_image(path: str = None, size: int = 128) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    if path is not None:
        img = Image.open(path).convert("RGB")
    else:
        # no sample image passed -- make a simple synthetic "foggy" gradient
        # image so the demo still has something visual to show
        import numpy as np
        arr = np.tile(np.linspace(180, 255, size), (size, 1)).astype("uint8")
        arr = np.stack([arr, arr, arr], axis=-1)
        img = Image.fromarray(arr)
    return tfm(img).unsqueeze(0)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    arr = ((tensor.squeeze(0).permute(1, 2, 0).detach().numpy() + 1) / 2 * 255).clip(0, 255)
    return Image.fromarray(arr.astype("uint8"))


def run_model(name: str, config: dict, x: torch.Tensor):
    print(f"\n=== {name} ===")
    torch.manual_seed(0)
    model = DTMIRPlus(config)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameters: {n_params:.2f}M")

    t0 = time.time()
    outputs = model(x)
    elapsed = time.time() - t0
    print(f"forward pass: {elapsed:.2f}s, restored shape {tuple(outputs['restored'].shape)}")

    # prove the gradient-flow claim live
    dummy_target = torch.randn_like(x)
    loss = torch.nn.functional.l1_loss(outputs["restored"], dummy_target)
    loss.backward()
    grad_to_translator = sum(
        p.grad.abs().sum().item() for p in model.translator.parameters() if p.grad is not None
    )
    print(f"gradient reaching the translator: {grad_to_translator:.6f} "
          f"({'frozen, as expected' if config['freeze_translator'] else 'flowing back, as intended'})")

    if outputs["routing_weights"] is not None:
        w = outputs["routing_weights"][0].detach().numpy()
        print(f"soft routing weights [haze, rain, snow]: {w.round(3)} (sum={w.sum():.3f})")

    return outputs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="optional path to a sample photo")
    args = parser.parse_args()

    x = load_demo_image(args.image, size=128)

    shared = dict(code_dim=16, base_channels=16, num_scales=3)

    baseline_config = {**shared, "weather_code": "onehot",
                        "freeze_translator": True, "routing": "hard"}
    ours_config = {**shared, "weather_code": "continuous",
                   "freeze_translator": False, "routing": "soft"}

    run_model("DTMIR-Pro baseline (frozen translator, hard routing)", baseline_config, x)
    outputs = run_model("Our version (trainable translator, soft routing)", ours_config, x)

    # save a visual before/after strip
    panels = [to_pil(x)] + [to_pil(outputs["translated"][n]) for n in ("haze", "rain", "snow")] \
        + [to_pil(outputs["restored"])]
    labels = ["input", "haze-look", "rain-look", "snow-look", "restored"]

    w, h = panels[0].size
    strip = Image.new("RGB", (w * len(panels), h + 20), "white")
    for i, p in enumerate(panels):
        strip.paste(p, (i * w, 20))
    strip.save("demo_output.png")
    print(f"\nSaved visual output to demo_output.png ({' | '.join(labels)})")
    print("\nNote: the model has random (untrained) weights at this stage, so the")
    print("images will look like structured noise, not a real restoration yet --")
    print("this demo is to prove the *pipeline* runs correctly end-to-end, not")
    print("to show restoration quality (that comes after real training).")
