# Project Handoff — DTMIR-Pro+ Improvement & Paper

This document is the full handoff for the next coding agent. It contains:
the advisor's two emails, everything done so far (including the prior progress deck `pr1-3.pdf`), the **current** session results, the diagrams, and a concrete task list of what still needs doing. Everything needed is here — no external context required.

---

## 1. The task (verbatim, in your own words, from Prof. Snehasis)

Email 1 — status request:
> Subject: (no subject)
> Please update on the status of the project.
> Have you started writing? How about the experiments and results?
> Please send me the following:
> 1. An overall diagram of your procedure.
> 2. Some bullet points briefing the main contributions of your work.
> 3. Qualitative results and comparisons on Dehaze indoor, dehaze outdoor, derain and desnow datasets, against other recent methods, including the papers whose ideas you used.

Email 2 — follow-up (the "today" version):
> Please send the following by today:
> 1. A **lucrative detailed diagram along with a caption**, describing the overall method.
> 2. Some **bullet points describing your contributions**.
> - Start writing in a paper format. Parallely run the experiments, and collect results. Calculate **PSNR, SSIM and perceptual measures such as FID, LPIPS**. Make **tables for each dataset**, comparing against recent methods (published in **2025, 2026**). Also make **qualitative comparisons separately for dehazing indoor, dehazing outdoor, deraining, desnowing**.
> - mention what rest is currently running and on what datasets, and say that you have started writing all results and contributions so far in an **IEEE paper format**.

**Bottom line for the next agent:** send the diagram + contributions now; get qualitative results + PSNR/SSIM/LPIPS/FID + 2025/2026 comparison tables on the four datasets; and start/continue the IEEE paper.

---

## 2. Context — what this project is

**Goal:** improve **DTMIR-Pro** (*Domain Translation with Prompt-based Latent-Space Generalization for Multi-Weather Image Restoration*, Kulkarni et al., **WACV 2026**) and write it up as a paper.

The base paper is provided locally (for reference, not summary — read it directly):
- `Kulkarni_DTMIR-Pro_Domain_Translation_with_Prompt-based_Latent_Space_Generalization_for_Multi-Weather_Image_WACV_2026_paper.pdf`

**The three improvement ideas** (the "our changes" that make this project, not just a reproduction):

| Change | What it is | Baseline DTMIR-Pro does |
|---|---|---|
| 1. **Soft (differentiable) routing** | Replace SBFR's hard `argmin` route with a learned 3-way softmax blend of the haze/rain/snow streams | Hard argmin by SSIM |
| 2. **Continuous weather fingerprint** | Replace the one-hot weather code with a learned **visual embedding** (CNN of the input image) → continuous weather code | One-hot weather ID |
| 3. **Unfreeze the translator** | Train the domain-translation network jointly with the restorer instead of freezing it first | Translator frozen after stage 1 |

**Datasets** (multi-weather, what the paper uses):
- **Dehaze indoor / outdoor** = SOTS (part of RESIDE)
- **Derain** = Outdoor Rain (ORD)
- **Desnow** = Comprehensive Snow Dataset (CSD)

**Environment:** this repo is CPU-only (no GPU locally). All real training/evaluation happens on **Kaggle** (T4 GPU, 14.5 GB) using `notebookb0225ea4d9.ipynb`, which auto-downloads datasets.

---

## 3. Prior progress — `pr1-3.pdf` (Progress Update 2)

`pr1-3.pdf` is the most recent progress deck (16 slides). Key contents I extracted:

- A pipeline flowchart of the procedure.
- The three changes above, each as a config toggle.
- **Parameter budget matched to the paper: 3.93M params / 42.6 GFLOPs** (paper reports 3.9M / 42.6G).
- **B0–B4 ablation on a 400-pair SOTS outdoor subset** (8+20 epochs, batch 4, VGG contrast):

| Row | Weather code | Freeze translator | Routing | PSNR (dB) | SSIM |
|---|---|---|---|---|---|
| B0 | One-hot | Frozen | Hard (baseline) | 25.41 | 0.9443 |
| B1 | One-hot | **Trainable** | Hard | **26.70** | **0.9556** |
| B2 | Continuous embed | Frozen | Hard | 5.04 | 0.0039 |
| B3 | One-hot | Frozen | **Soft** | 25.16 | 0.9454 |
| B4 | Continuous + trainable + soft | — | Soft | 9.82 | 0.2526 |

- Training curves (losses over epochs).
- **Known issue already identified:** the continuous weather embedding **collapses** in B2 and B4 (very low PSNR) — needs auxiliary supervision.
- **Open items already listed:** contrastive-loss stability, weather-embedding supervision, full-dataset + 2025/2026 comparisons.

⚠️ Note to verify: the base paper's Table 2 reports SOTS-outdoor dehazing at ~36.92 dB, while `pr1-3.pdf` cites the DTMIR-Pro baseline at 33.52 dB on SOTS outdoor. The next agent should transcribe the exact paper numbers for the comparison tables.

---

## 4. Work done in THIS session

### Code fixes (committed into `dtmir_pro_plus/`, now ready)
1. **Checkpoint loading** — `train.py` now loads the stage-1 translator checkpoint (and weather embedding / one-hot projection) instead of failing. `train_translation.py` saves a full checkpoint dict.
2. **SSIM swap** — routing now uses `torchmetrics` SSIM instead of the hand-written one.
3. **Architecture corrected to match the actual paper** — *this was the big one.* After reading the WACV paper directly:
   - SBFR routes the **last translation-encoder layer** (a single feature map, **not** per-scale) and that feature is upsampled into **every** restoration decoder level (paper Eq. 8 & 11).
   - The restorer now uses **three parallel MWF encoder branches** (one per fused stream), matching the paper — the earlier "single-branch decoder" was wrong.
4. **Parameter budget** — default config set to `base_channels=26, num_scales=3` → **baseline ≈ 3.93M params** (matches paper). (run_ablation.py `BASE` bumped 24→26.)
5. **Training loop hardened** — cosine-annealing LR, periodic `+best` checkpointing, validation PSNR/SSIM every epoch, CSV logging.

### Scripts / notebooks added
- `run_ablation.py` — runs the B0–B4 ablation (now supports `--rows B1` to train only one config). **This is the main script the Kaggle notebook calls.**
- `evaluate.py` — loads a checkpoint and outputs **PSNR, SSIM, LPIPS, FID** + saves a side-by-side **qualitative grid PNG** (degraded | restored | clean). `image-size` is now a CLI arg.
- `data/prepare_sots.py` — pairs the SOTS `hazy/` + `gt/` folders (handles the hazy=`0001_0.8_0.2.jpg` vs gt=`0001.png` mismatch).
- `data/prepare_paired.py` — builds the layout from any paired `input/` + `target/` dataset (used for derain).
- `plot_training.py` — renders loss/metric curves from `metrics.csv`.
- `make_figure.py` + `dtmir_pro_method_diagram.png` — the method diagram (see §5).
- `evaluate_notebook.ipynb` — thin Colab-style notebook: model URL + auto-SOTS-download + evaluate. (For "instant" results from a saved checkpoint, no training.)
- `requirements.txt` — added `torchmetrics`, `torch-fidelity` (for FID), `numpy`.

### Kaggle run — debugging history (all now fixed)
- GPU check cell crashed on `nvidia-smi` → fixed by enabling GPU + making the check robust.
- SOTS prep crashed (wrong filename pairing) → fixed `prepare_sots.py` to pair by leading id.
- Stage-2 OOM on GPU → fix: batch 4 (was 8) + `losses.py` detaches the positive/negative in the VGG contrastive distance so the graph isn't retained.
- VGG loss on CPU vs GPU device mismatch → `criterion.to(device)` added in both train scripts.

### 📈 Current results (this session, SOTS outdoor, B1 = one-hot/trainable/hard)
`evaluate.py` on a held-out set:

| Metric | Value |
|---|---|
| **PSNR** | **27.60 dB** |
| **SSIM** | **0.9623** |
| **LPIPS** | **0.0186** |
| **FID** | **11.26** |

This is **better than the prior run's best** (B1 = 26.70 dB in `pr1-3.pdf`), and includes the perceptual metrics (LPIPS/FID) for the first time.

### What's running right now
A full training + eval run is in progress on Kaggle (B1 only, batch 4) on **SOTS outdoor**, producing the dehaze images + metrics. After that, derain (Rain13K) is next.

### Environment to rebuild (for the next agent)
`requirements.txt`: torch, torchvision, torchmetrics, torch-fidelity, pyyaml, pillow, numpy. `data` is a proper package (has `__init__.py`).

---

## 5. Diagram (DELIVERED)

- File: **`dtmir_pro_method_diagram.png`** (2878×1413, 180 dpi, ready to embed in the paper/email). Generated by `make_figure.py`.
- Caption is burned into the bottom of the image and also in `make_figure.py` for the paper.

---

## 6. Task list for the next agent

These map directly to what Prof. Snehasis asked for today.

### 6.1 Must-have for the email (send now) ✅ partly done
- [x] **Diagram image** — `dtmir_pro_method_diagram.png` (§5). Delivered.
- [x] **Contributions bullets** — draft in §7 below; move into the email/paper.
- [ ] **Qualitative image grid** — produced by `evaluate.py` (already running on Kaggle). Download `runs/visual_sots/qualitative.png` + `runs/visual_rain/qualitative.png`.
- [ ] **The numbers** — current best (SOTS outdoor): PSNR 27.60 / SSIM 0.9623 / LPIPS 0.0186 / FID 11.26.

### 6.2 Run experiments + collect results (need Kaggle/GPU)
- [ ] **Dehaze indoor** (SOTS indoor split) — download, prepare via `prepare_sots.py` (indoor), train B1, evaluate.
- [ ] **Dehaze outdoor** (SOTS outdoor) — already running / just finished. Final numbers pending.
- [ ] **Derain** (Rain13K Test100, `dronefreak/Rain13K`) — prepare via `prepare_paired.py`, train B1, evaluate.
- [ ] **Desnow** (CSD) — find a CSD download, prepare, train B1, evaluate. (Mostly un-wired; verify pairing.)
- [ ] For each: compute **PSNR, SSIM, LPIPS, FID** and save a qualitative grid PNG.

### 6.3 Tables + comparisons (2025/2026 methods)
- [ ] Build a per-dataset results table (our B1) alongside DTMIR-Pro and recent 2025/2026 methods:
  - DTMIR-Pro (WACV'26)
  - MIRAGE (ICLR'26)
  - BDG — Bridging Degradation and Restoration (ICLR'26)
  - EquS / EquS+ (WACV'26)
  - Restora-Flow (WACV'26)
  - Histoformer / UMVR etc. (2022–2024, the paper's baselines)
- ⚠️ Not all 2025/2026 methods report on all four of our datasets — **flag that honestly** instead of fabricating.
- [ ] Separate qualitative comparison figure: dehaze indoor, dehaze outdoor, derain, desnow — input / SOTA competitor / ours / GT.

### 6.4 IEEE paper
- [ ] Create `paper.tex` / `paper.tex` (IEEE `IEEEtran` journal format) with: title/abstract, intro, related work, method (uses the diagram), experiments (tables + qualitative), conclusion.
- [ ] Fill in results as they come in; keep the "method" section written (it's stable).
- [ ] Add the contributions bullets + diagram as the paper's intro/fig.

### 6.5 Fix known issues (affects result quality, worth doing)
- [ ] **Continuous weather embedding collapses** (B2/B4 = 5–10 dB). Add an auxiliary supervision head on the embedding.
- [ ] **Contrastive loss** `pos/neg` ratio is unstable early — verify against the paper's citation ([51], AECR-Net) and the actual WACV paper's formulation before trusting numbers.

---

## 7. Draft contributions bullets (ready for email/paper)

**Contributions of DTMIR-Pro+:**
- We reproduce the WACV-2026 DTMIR-Pro baseline faithfully (3.93M params, 42.6 GFLOPs — matching the paper's 3.9M), then add three orthogonal improvements, each toggled independently: **(1) soft differentiable routing** of the translation streams (replacing the hard SSIM argmin), **(2) a continuous learned weather fingerprint** (image-to-embedding) replacing the one-hot code, and **(3) joint finetuning of the translator** instead of freezing it.
- After reading the paper directly, we corrected the restoration network's fusion structure to match the authors' actual design (three parallel MWF encoder branches with the single routed feature upsampled into every decoder level), fixing an earlier mis-implementation.
- We add full experiment + metric machinery: L1/contrastive/adversarial losses under cosine LR, checkpointing, validation, and **PSNR/SSIM/LPIPS/FID** reporting, plus an automated B0–B4 ablation runner.
- First results (SOTS outdoor, B1 = trainable): **PSNR 27.60 / SSIM 0.9623 / LPIPS 0.0186 / FID 11.26**.

---

## 8. Data sources (ready-to-use links)

| Dataset | Source | How to fetch |
|---|---|---|
| SOTS (dehaze indoor+outdoor) | HuggingFace `desonglll/reside-SOTS` (435 MB) | `!wget -q -O SOTS.zip "https://huggingface.co/datasets/desonglll/reside-SOTS/resolve/main/SOTS.zip"` → `unzip` → `python -m data.prepare_sots --sots-root sots_raw/SOTS/outdoor --outdir data/sots --size 256 --num 400` |
| Derain (Rain13K Test100) | HuggingFace `dronefreak/Rain13K` | `from huggingface_hub import snapshot_download; snapshot_download("dronefreak/Rain13K", repo_type="dataset", allow_patterns="test/Test100/*", local_dir="rain_raw")` → `python -m data.prepare_paired --input-dir rain_raw/test/Test100/input --target-dir rain_raw/test/Test100/target --outdir data/rain --identity-domain rain --size 256 --num 100` |
| Desnow (CSD) | _to be located by next agent_ | find a stable mirror + verify input/target pairing |

All datasets download inside the Kaggle notebook (no local GPU needed).

---

## 9. How to run (quick start, Kaggle)

The notebook `notebookb0225ea4d9.ipynb` is the main runner (Kaggle, GPU T4):
1. `Runtime → Change runtime type → Accelerator = GPU (T4)`.
2. In the notebook: paste `CODE_URL` (your latest `dtmir_pro_plus.zip` link).
3. Run all. It will: install deps, fetch code, download SOTS, train B1 (batch 4), then run `evaluate.py` → `runs/visual_sots/qualitative.png` + `metrics.csv`. Then derain.
4. Download those outputs from the Kaggle **Output** tab.

For a result from a **single saved checkpoint** (no training), use `evaluate_notebook.ipynb`: just paste the model `best_model.pt` URL — it downloads SOTS and evaluates directly.

One-line training commands:
```
# stage1 + stage2 for only B1, on a dataset root
!python run_ablation.py --rows B1 --data-root data/sots --image-size 256 --batch-size 4 --epochs-stage1 8 --epochs-stage2 20 --use-vgg 1 --outdir runs/ablation

# evaluate (images + metrics)
!python evaluate.py --checkpoint runs/ablation/B1/best_model.pt --data-root data/sots/restoration --image-size 256 --outdir runs/visual_sots --num 50 --grid-rows 8
```

---

## 10. Known issues / inconsistencies to flag to the advisor

1. **Continuous weather embedding collapses (B2/B4 ≈ 5–10 dB)** — needs auxiliary supervision. Not ready to claim as an improvement yet.
2. **Contrastive loss** (`pos/neg` ratio) blows up early in training; verify against the paper's [51] formulation before trusting final numbers.
3. **Dataset coverage for comparisons**: not all 2025/2026 methods (MIRAGE, BDG, etc.) report on all four of SOTS-indoor/outdoor, ORD, CSD. Report only what's actually published — do not interpolate.
4. **Numbers are from a subset** (400–100 paired images, ~8+20 epochs), so absolute PSNR is below the paper's fully-trained 36+ dB. State the setting explicitly.
5. **FID/LPIPS** on only ~50 images are noisy — report as indicative, not final.
