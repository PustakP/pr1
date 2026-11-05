# DTMIR-Pro+: Multi-Weather Image Restoration

An improved multi-weather image restoration framework building upon **DTMIR-Pro** (*Domain Translation with Prompt-based Latent-Space Generalization for Multi-Weather Image Restoration*, Kulkarni et al., WACV 2026).

---

## 🌟 Overview & Key Contributions

DTMIR-Pro+ extends the baseline multi-weather restoration architecture with three novel enhancements designed to improve representation blending, weather conditioning, and feature alignment across adverse weather domains (dehaze, derain, desnow):

1. **Soft (Differentiable) Feature Routing**: Replaces hard `argmin` routing with a learned continuous softmax blend over domain-translated representations.
2. **Continuous Visual Weather Fingerprint**: Replaces fixed one-hot weather codes with a learned visual embedding extracted directly from the degraded input image.
3. **End-to-End Joint Domain Translation**: Unfreezes the domain translator during the restoration phase, enabling joint optimization of translation and multi-scale restoration.

---

## 📂 Repository Structure

```
pr1/
├── dtmir_pro_plus/                  # Core PyTorch source package
│   ├── configs/                     # Experiment YAML configurations
│   ├── data/                        # Dataset loaders & synthetic data generator
│   ├── models/                      # Architectural modules
│   │   ├── weather_prompt.py        # One-hot & continuous weather embeddings
│   │   ├── translation_net.py       # Domain translation network
│   │   ├── routing.py               # Hard vs. Soft routing
│   │   ├── fusion.py                # Multi-Weather Feature Fusion (MWFF)
│   │   ├── restoration_net.py       # Multi-scale restoration network
│   │   └── dtmir_plus.py            # End-to-end framework
│   ├── losses.py                    # Multi-task training loss definitions
│   ├── train_translation.py         # Stage 1 translator training
│   ├── train.py                     # Stage 2 restoration training
│   ├── run_ablation.py              # Automated B0–B4 ablation suite
│   ├── evaluate.py                  # Evaluation metrics (PSNR, SSIM, LPIPS, FID)
│   └── demo.py                      # Interactive demo script
├── notebookb0225ea4d9.ipynb         # Kaggle GPU training & visual evaluation notebook
├── dtmir_update_slides.tex          # LaTeX presentation slides
├── pipeline_flowchart_standalone.tex# Standalone TikZ pipeline flowchart
├── HANDOFF.md                       # Detailed project handoff & status report
└── figures & diagrams               # Visual pipeline architectures & flowcharts
```

---

## 🚀 Quick Start

### Installation

```bash
cd dtmir_pro_plus
pip install -r requirements.txt
```

### Sanity Check

```bash
python sanity_check.py
```

### Synthetic Smoke Test (CPU)

```bash
# Generate synthetic paired data
python data/make_synthetic_data.py --num 16 --size 64

# Stage 1: Pretrain translator
python train_translation.py --config configs/config_translation_smoke.yaml

# Stage 2: Restoration training
python train.py --config configs/config_restoration_smoke.yaml \
    --checkpoint translator_pretrained.pt --outdir runs/smoke --epochs 20

# Plot loss curves
python plot_training.py --csv runs/smoke/metrics.csv --out runs/smoke/curves.png
```

---

## 📊 Evaluation & Ablation Study

Run the automated ablation ladder across configurations (B0 through B4):

```bash
python run_ablation.py --data-root data/synthetic --image-size 64 \
    --epochs-stage1 1 --epochs-stage2 2 --outdir runs/ablation_smoke
```
