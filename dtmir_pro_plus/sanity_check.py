"""
Quick sanity check: builds the model in both baseline mode and our
proposed mode, runs a random batch through it, checks shapes, and checks
that gradients actually reach the translation network when
freeze_translator=False (confirming the joint fine-tuning change is wired
correctly).

Run with: python sanity_check.py
"""

import torch

from models.dtmir_plus import DTMIRPlus


def run_config(name: str, config: dict, image_size: int = 64):
    print(f"\n--- {name} ---")
    model = DTMIRPlus(config)
    x = torch.randn(2, 3, image_size, image_size)

    outputs = model(x)
    restored = outputs["restored"]
    print("restored shape:", tuple(restored.shape))
    assert restored.shape == x.shape, "restored image shape must match input"

    target = torch.randn_like(x)
    loss = torch.nn.functional.l1_loss(restored, target)
    loss.backward()

    translator_grad_norm = sum(
        p.grad.abs().sum().item() for p in model.translator.parameters() if p.grad is not None
    )
    print(f"gradient reaching translator: {translator_grad_norm:.6f} "
          f"(expected {'> 0' if not config['freeze_translator'] else '== 0'})")

    if config["freeze_translator"]:
        assert translator_grad_norm == 0.0
    else:
        assert translator_grad_norm > 0.0

    if outputs["routing_weights"] is not None:
        print("routing weights (per-stream, should sum to ~1):",
              outputs["routing_weights"][0].detach().numpy())


if __name__ == "__main__":
    base_config = dict(code_dim=8, base_channels=8, num_scales=2)

    baseline_config = {**base_config, "weather_code": "onehot",
                        "freeze_translator": True, "routing": "hard"}
    ours_config = {**base_config, "weather_code": "continuous",
                   "freeze_translator": False, "routing": "soft"}

    run_config("DTMIR-Pro baseline (onehot / frozen / hard routing)", baseline_config)
    run_config("Our version (continuous / trainable / soft routing)", ours_config)

    print("\nAll sanity checks passed.")
