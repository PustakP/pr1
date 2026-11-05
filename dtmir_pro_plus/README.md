# DTMIR-Pro+ (working title)

Improved multi-weather image restoration pipeline built from scratch (DTMIR-Pro's
official repo has no released code). Our three proposed changes are each a config
toggle, so we can run controlled ablations.

## What changed this update

- **Restoration net corrected to match the paper** — after reading the actual
  DTMIR-Pro paper: SBFR routes the translation encoder's *last* layer (`DT_E`, a
  single feature map), which is upsampled into each decoder level alongside three
  per-level MWF encoder branches (paper Eq. 8 and 11). Rewrote the restorer to match
  (the earlier version was a simplified single-branch decoder).
- **Stage-1 checkpoint loading** — `train.py` loads `translator_pretrained.pt`
  (translator + weather embedding/projection) instead of starting stage 2 from a
  fresh translator.
- **Parameter budget matched** — baseline config now sits at **3.93M params**
  (paper: 3.9M) via `base_channels: 26`, `num_scales: 3`.
- **Library SSIM** — torchmetrics replaces the hand-written version in routing.
- **Cleaner soft-router input** — a small independent `RoutingInputStem` replaces the
  old "reuse the translator encoder" shortcut.
- **Real training loop** — cosine-annealing LR, checkpointing, PSNR/SSIM validation,
  CSV logging (`train.py`).
- **Ablation runner** — `run_ablation.py` automates the B0–B4 ladder and writes one
  comparable summary table.
- **Colab notebook** — `colab_train.ipynb` for the GPU run.

## Status

This is a **working scaffold**, not a trained model. The full pipeline (stage 1 →
stage 2) has been smoke-tested end-to-end on synthetic data and learns (validation
PSNR rises on the smoke run). No real dataset has been touched yet — that is the next
step and needs a GPU.

## Quick start

```
pip install -r requirements.txt
python sanity_check.py               # fast correctness check, no data needed
python demo.py                       # live demo (baseline vs. ours)
```

## Running the smoke test (CPU, no dataset)

```
python data/make_synthetic_data.py --num 16 --size 64
python train_translation.py --config configs/config_translation_smoke.yaml
python train.py --config configs/config_restoration_smoke.yaml \
    --checkpoint translator_pretrained.pt --outdir runs/smoke --epochs 20
python plot_training.py --csv runs/smoke/metrics.csv --out runs/smoke/curves.png
```

## Running the ablation ladder

```
python run_ablation.py --data-root data/synthetic --image-size 64 \
    --epochs-stage1 1 --epochs-stage2 2 --use-vgg 0 --outdir runs/ablation_smoke
```

Use larger epochs, `--image-size 256`, and `--use-vgg 1` for a real run (GPU).

## Layout

```
dtmir_pro_plus/
├── models/
│   ├── weather_prompt.py      # WCP — one-hot (baseline) or continuous embedding
│   ├── dynamic_attention.py   # Dynamic Attention Block (channel attention)
│   ├── translation_net.py     # Domain translation network (shared encoder-decoder)
│   ├── discriminator.py       # Per-domain PatchGAN discriminators
│   ├── routing.py             # Hard SBFR (baseline) + soft routing (ours)
│   ├── fusion.py              # Multi-Weather Feature Fusion (MWFF)
│   ├── restoration_net.py     # Restoration network (3 MWF branches + upsampled SBFR)
│   └── dtmir_plus.py          # Wires everything into one end-to-end model
├── data/
│   ├── dataset.py             # Restoration dataset (degraded/clean pairs)
│   ├── translation_dataset.py # Translation dataset (input + per-domain GT)
│   ├── make_synthetic_data.py # Synthetic smoke-test data generator
│   └── prepare_sots.py        # Builds the expected layout from a SOTS folder
├── configs/
│   ├── config.yaml            # Full config (baseline ~3.93M params)
│   ├── config_translation_smoke.yaml
│   └── config_restoration_smoke.yaml
├── losses.py                  # L1 + contrastive + adversarial losses
├── train_translation.py       # Stage 1: pretrain translator adversarially
├── train.py                   # Stage 2: restoration / joint fine-tuning
├── run_ablation.py            # B0–B4 ablation ladder
├── plot_training.py           # Plots metrics.csv
├── colab_train.ipynb          # Colab GPU notebook
├── demo.py                    # Self-contained live demo
└── sanity_check.py            # Fast correctness checks
```

## Known gaps (flagged for the update)

- Contrastive loss uses a `pos/neg` ratio that can blow up when translated streams are
  too similar; needs re-checking against the paper's formulation (paper Eq. 9-11 gives
  weights 0.5/1.0 translation and 1.0/0.75 restoration, but not the exact contrastive
  form).
- Continuous weather embedding has no auxiliary supervision yet.
- Real-world (RTTS/RID/Snow-Realistic) NIQE/Entropy/BRISQUE evaluation not yet done.
