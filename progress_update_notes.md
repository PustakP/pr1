# DTMIR-Pro+ — Progress Update Notes

## What I did this session

Everything below is verified to run on this machine (CPU-only, no GPU).

### Code fixes (the handoff's "next 5 tasks" + key gaps)
1. **Stage-1 checkpoint loading (gap 1)** — `train.py` now loads
   `translator_pretrained.pt` (translator + weather embedding/projection) instead of
   starting stage 2 from a fresh translator. `train_translation.py` now saves the full
   checkpoint.
2. **Parameter budget (gap 2)** — matched the paper. Baseline is now **3.93M params**
   (paper: 3.9M) with `base_channels: 26`, `num_scales: 3`. The old `13M` figure was the
   default `base_channels: 32`, `num_scales: 4`; it was a config issue, not a code bug.
3. **Library SSIM (gap 3)** — hand-written SSIM replaced with torchmetrics
   (`requirements.txt` updated).
4. **Restoration net corrected against the actual paper (gap 4 + 8)** — after reading the
   uploaded paper, I found the handoff's "routes at every scale" note was a misreading.
   The paper routes the translation encoder's **last** layer (`DT_E`, a single feature map,
   Eq. 8) and upsamples that one feature into each decoder level next to three per-level
   MWF encoder branches (Eq. 11). Rewrote the restorer to match (3 shared branches + one
   upsampled routed feature).
5. **Soft-router input cleanup (gap 5)** — replaced the "reuse translator encoder with
   weather class 0" shortcut with a small independent `RoutingInputStem`.
6. **Real training loop** — `train.py` now has cosine-annealing LR, periodic + best-model
   checkpointing, PSNR/SSIM validation, and CSV logging. (Also fixed a missing
   `use_vgg` argument in the old `train.py`.)

### New tooling
- `data/make_synthetic_data.py` — generates a synthetic degraded/clean + translation
  dataset for smoke tests (no download needed).
- `run_ablation.py` — automates the **B0–B4 ablation ladder**, pretrains the translator
  once per weather-code mode, runs all five rows, and writes one comparable summary table.
- `plot_training.py` — plots `metrics.csv` (loss + PSNR + SSIM curves).
- `colab_train.ipynb` — GPU notebook for the real run.

### Visual results produced
- **Smoke training curve** (`runs/smoke/curves.png`): full stage-1 → stage-2 pipeline
  learns on 16 synthetic images — validation PSNR **~8.9 → 12.7 dB**, SSIM **0.23 → 0.48**.
  This proves the pipeline is real and learns, not that restoration is good yet.
- **Ablation summary table** (`runs/ablation_smoke/summary.csv`): B0–B4 all run
  end-to-end and write comparable rows (synthetic numbers only, not comparable to paper).
- Updated Beamer deck `dtmir_update_slides.tex` (see below).

### Slides
`dtmir_update_slides.tex` now: fixes the two loose attributions (Change 2 is our own
proposal; Change 3 soft-blending is our own design, MIRAGE only offers the SPD/covariance
similarity idea), adds the advisor's two extensions as dashed "future" nodes, and adds
slides for the code fixes, the smoke training curve (self-contained TikZ plot), and the
ablation ladder. No LaTeX is installed on this machine, so compile it in Overleaf.

## Inconsistencies / problems found (flagged for the update)

1. **The handoff misread the SBFR routing (now fixed).** The handoff said "the paper routes
   at every scale (E_DT at each skip level)". The actual paper (Eq. 8) routes only the
   translation encoder's **last** layer, a single feature map, and upsamples that one feature
   at each decoder level (Eq. 11). The three per-level features are the *MWF encoder*
   branches, not routed features. I corrected the restorer to match the paper.
2. **Contrastive loss instability** — `losses.py` computes `pos_dist / neg_dist`. When the
   three translated streams are too similar (common early in training), `neg_dist` is near
   zero and the loss spikes into the millions (seen in the smoke runs). The paper cites a
   specific contrastive formulation [51] and only gives the weighting (0.5/1.0 translation,
   1.0/0.75 restoration), so this needs re-checking before real training.
3. **`data/` package name collision** — the repo's `data/` folder had no `__init__.py`, so
   `import data.dataset` was silently resolving to an unrelated installed `data` package
   instead of ours. Added `data/__init__.py`.
4. **Parameter count** — the handoff's "full config ~13M vs paper 3.9M" was real but it was
   just the default width/depth; the fix is config tuning, not architecture surgery.
5. **Weather embedding has no supervision (gap 9, still open)** — nothing currently forces
   the continuous embedding to be meaningful for mixed weather. Flagged; needs an auxiliary
   objective or a sanity visualization (t-SNE) once trained.
6. **torchmetrics FutureWarning** — a harmless upstream deprecation warning appears on
   import; it does not affect results.

## What I need from you next (GPU / data)

I cannot run the real training here (no GPU, and the datasets aren't local). To get actual
comparable numbers:

1. Get the datasets — **SOTS** (haze), **Outdoor-Rain ORD**, **Comprehensive Snow CSD** —
   a small subset first is fine for a first curve.
2. Run **`colab_train.ipynb`** on a Colab GPU (any T4 is enough; the paper used a GTX 1080
   8GB). It installs deps, uploads the code, and runs `run_ablation.py` to produce the
   B0–B4 PSNR/SSIM table.
3. Decide with the advisor: is a reproduction *slightly under* the paper's numbers
   acceptable as the baseline row, or do we compare only against our own re-trained
   baselines?

Once those numbers exist, I can drop them straight into the ablation slide next to the
paper's SOTS reference (36.92 / 0.981).
