# DTMIR-Pro Improvement Project — Handoff / Progress Update

Paste this whole file into the new chat first, then upload the files listed in Section 9.

---

## 1. Project in one paragraph

CSD493 Project-1 (Monsoon 2026), advisor Dr. Snehasis Mukherjee, solo (Pustak Pathak). Title: **Joint Fine-Tuning and Continuous Weather Embeddings for Prompt-Based Multi-Weather Image Restoration.** Baseline is **DTMIR-Pro** (WACV 2026): a two-stage pipeline where a single prompt-conditioned *translation network* turns a degraded image into haze/rain/snow-look versions, and a *restoration network* uses those to produce the clean image. Proposal was signed 31 Jul 2026. The advisor's brief (okok-1.pdf): understand DTMIR-Pro fully, try running its code, see if ideas from the other reading-list papers can improve it.

Important fact: **DTMIR-Pro's official GitHub repo is empty**, so we are implementing from scratch from the paper.

## 2. Reading list and what each paper contributes

| Paper | Venue | Role in our project |
|---|---|---|
| DTMIR-Pro | WACV'26 | Baseline. 3.9M params, 28.6 GFLOPs. Reported PSNR/SSIM: SOTS 36.92/0.981, CSD 33.94/0.953, ORD 32.88/0.954. Real-world (RTTS/RID/Snow-Realistic): NIQE/Entropy/BRISQUE |
| MIRAGE | ICLR'26 | Idea source for (a) channel-split conv+attention+MLP backbone, (b) SPD-covariance contrastive alignment between shallow and latent features |
| BDG | ICLR'26 | Idea source for a continuous learned degradation descriptor (MAS-GLCM) and staged alignment instead of a discrete label |
| Restora-Flow | WACV'26 | Considered, deferred (needs a flow/diffusion restorer) |
| EquS / EquS+ | WACV'26 | Considered, deferred (diffusion sampling trick) |

## 3. The three problems and three proposed changes

| # | DTMIR-Pro weakness | Our change | Inspired by | Sanity-check verdict |
|---|---|---|---|---|
| 1 | Translator frozen after pretraining | Keep trainable; restoration loss gradients flow back | Own proposal | Strongest, clearly justified |
| 2 | Weather code is one of 3 one-hot labels | Continuous learned weather embedding from the image | BDG idea | Real gap, but mixed degradations is a known issue |
| 3 | SBFR hard-picks 1 of 3 streams via SSIM argmin | Learned softmax-weighted blend of all 3 streams | MIRAGE / own | Most debatable; must be measured, may give small gain |

Optional extensions from the advisor meeting: (A) replace the Dynamic Attention Block with MIRAGE's conv+attention+MLP channel split; (B) use MIRAGE-style SPD similarity or BDG-style staged alignment as the routing score.

## 4. What has been delivered so far

1. **Slide deck** (LaTeX Beamer, 12 slides, `dtmir_improvement_flowchart.pdf`, source `dtmir_update_slides.tex`): recap, colour-coded flowchart (blue unchanged / orange modified / green new, dashed red feedback arrow), one slide per change (replaces / instead / source paper / why), "considered but not used" slide, summary table, code status.
2. **Starter code** (`dtmir_pro_plus_starter_code.zip`, PyTorch). Layout:
   - `models/weather_prompt.py` — Weather Coded Prompt with CBIN; `WeatherEmbeddingNet` (continuous) vs one-hot (+ projection to `code_dim`)
   - `models/dynamic_attention.py` — dynamic-conv query + **channel attention**
   - `models/translation_net.py` — shared encoder-decoder, `set_frozen()` toggle
   - `models/discriminator.py` — per-domain PatchGAN
   - `models/routing.py` — `HardSimilarityRouting` (baseline SBFR) and `SoftFeatureRouting` (ours)
   - `models/fusion.py` — MWFF (FFT global + multi-scale gated depthwise local), kept as in paper
   - `models/restoration_net.py`, `models/dtmir_plus.py` — wiring, all three changes are config flags
   - `losses.py` — L1 + VGG-feature contrastive + adversarial (VGG falls back to random init if weights can't download)
   - `train_translation.py` (stage 1, adversarial), `train.py` (stage 2), `data/` dataset stubs, `configs/config.yaml`, `sanity_check.py`, `demo.py`
3. **Demo for the advisor meeting**: `python demo.py` (offline after one setup run) prints params, timing, translator-gradient proof (0 in baseline, non-zero in ours) and routing weights, and saves `demo_output.png`.

## 5. What has actually been verified

- Baseline config and our config both run end-to-end; output shape matches input.
- Gradient reaches translator only when `freeze_translator: false` (checked in `sanity_check.py` and `demo.py`).
- Forward+backward at 256x256 works (full config ~13M params; demo config ~1.8M).
- `train_translation.py` and `train.py` each ran 2 epochs on tiny random data (G step, D step, VGG loss) without errors.
- One real bug found and fixed: original attention did full pixel-to-pixel attention and ran out of memory at 256x256 -> switched to channel attention.

## 6. Honest status: what is NOT done / known gaps

**No real data has been touched. No result exists.** Output images in the demo are noise because weights are untrained. Specific gaps to fix before any numbers are reported:

1. `train.py` does **not load** `translator_pretrained.pt` from stage 1 — stage 2 currently starts from a fresh translator. Must load the checkpoint (frozen for baseline, trainable for ours).
2. Param count: full config is ~13M vs paper's 3.9M. Channel widths/blocks need tuning to be comparable.
3. SSIM in `routing.py` is a hand-written single-scale version; replace with a library implementation (e.g. torchmetrics).
4. Soft routing uses only the last (lowest-resolution) translator encoder features, projected/upsampled; the paper routes at every scale (E_DT at each skip level). Simplified.
5. Soft router's "input features" reuse the translator encoder with weather code of class 0 — a shortcut that should be cleaned up.
6. Contrastive loss: confirm it matches the paper's formulation (we use L1 distance ratio in VGG feature space); loss weights from the paper (0.5/1.0 translation, 1.0/0.75 restoration) are only partly wired.
7. Dataset loaders are stubs; translation dataset assumes aligned per-domain ground truth which real datasets may not provide.
8. Restoration net is a simplified version of the paper's decoder (the paper concatenates MWFF + SBFR features at every level).
9. Weather embedding currently has no supervision beyond the end loss; nothing enforces it to be meaningful for mixed weather.

**Slide-deck wording to tighten before presenting again:**
- Change 3's "MIRAGE shows combining streams with a learned similarity step works better" is a loose attribution. MIRAGE does not do stream routing; what it offers is the SPD/covariance similarity idea (per the advisor's slide 9). Reword to "MIRAGE's covariance-based similarity could serve as the routing score" or call soft routing our own design.
- Change 2's claim that "every paper avoids freezing a component" and the EquS example is not accurate; say simply that it is our own proposal.
- The deck does not yet show the two advisor extensions (MIRAGE backbone swap, BDG staged alignment) as flowchart nodes; add them as dashed "future" nodes.

## 7. Future directions (framed so each step produces a comparable result)

Principle: every stage ends with a table the advisor can compare against the DTMIR-Pro paper's numbers.

**Stage A — Reproduce baseline (target: a number comparable to the paper)**
- Fix gaps 1-3 above, then train baseline config (onehot / frozen / hard) on the paper's data: SOTS (5000 train / 500 test), Outdoor-Rain ORD (9000 / 750), Comprehensive Snow CSD (8000 / 2000).
- Paper's schedule: translator 100 epochs, restorer 300 epochs, lr 1e-3 cosine annealing.
- Deliverable: PSNR/SSIM on SOTS, ORD, CSD, plus params/GFLOPs, next to the paper's 36.92/0.981, 32.88/0.954, 33.94/0.953. Even landing a bit below is a valid "our reproduction" row. Start with a small subset to get a first curve quickly.

**Stage B — Ablation ladder (the core comparable result)**

| Row | Weather code | Translator | Routing |
|---|---|---|---|
| B0 | one-hot | frozen | hard (reproduction) |
| B1 | one-hot | **trainable** | hard |
| B2 | **continuous** | frozen | hard |
| B3 | one-hot | frozen | **soft** |
| B4 | continuous | trainable | soft (full proposal) |

Same data, same schedule, same seeds (ideally 2-3 seeds), PSNR/SSIM on SOTS/ORD/CSD. This directly answers "which of the three problems was worth solving", including the possibility that routing change does little.

**Stage C — Mixed-weather evidence (where our changes should matter most)**
- ORD (rain+haze) and CSD (snow+haze) are already mixed; report them separately.
- Qualitative: show soft routing weights and embedding behaviour on mixed vs pure images (e.g. plot weights per dataset, t-SNE of embeddings coloured by dataset). A figure showing weights are non-uniform and shift with weather type is strong evidence.
- Real-world: RTTS, RID, Snow-Realistic with NIQE / Entropy / BRISQUE, same as paper's Table 1.

**Stage D — Advisor's extensions (one at a time, each as new rows)**
- D1: MIRAGE channel-split block replacing the Dynamic Attention Block -> report params/FLOPs and PSNR (target: same or better with fewer params).
- D2: SPD (covariance) similarity as the routing score, or BDG-style staged alignment of the weather embedding -> compare with B4.

**Stage E — Write-up**
- Final deck/report: flowchart, ablation table, routing-weight figure, honest discussion (including null results), comparison to the paper.

**Needs to be decided with the advisor / resources**
- GPU access (paper used a GTX 1080 8GB, so modest hardware is enough).
- Whether a reproduction slightly under the paper's numbers is acceptable as the baseline.
- Dataset access and whether to evaluate against reported numbers or re-trained baselines only.
- Deadlines (not stated in the files; add here).

## 8. Suggested immediate next 5 tasks

1. Load stage-1 checkpoint in `train.py`; swap in library SSIM; match param budget toward 3.9M.
2. Download SOTS (small subset first) and run baseline for a few epochs to get a first loss curve.
3. Fix the three slide wordings and add the future-extension nodes; regenerate the deck.
4. Make routing multi-scale as in the paper (gap 4).
5. Run the B0-B4 ladder on the small subset to check the pipeline before full training.

## 9. Files to upload to the new chat

- This handoff file
- `dtmir_pro_plus_starter_code.zip`
- `dtmir_update_slides.tex` (and `dtmir_improvement_flowchart.pdf`)
- `okok-1.pdf` (advisor meeting slides) and `ProjectProposalForm-PustakPathak-1.pdf`
- The five papers if the new chat needs to re-check details (DTMIR-Pro especially)

## 10. Message to paste to start the new chat

> I'm continuing my DTMIR-Pro improvement project (CSD493, advisor Dr. Snehasis Mukherjee). Full context is in the attached handoff file; code zip and slides attached. Please start from Section 8 (next 5 tasks). Keep explanations in simple, non-technical language, give complete ready-to-use files, and tell me directly if something I've written is wrong.

---

## 11. Demo runbook for the advisor meeting (corrected)

**Before the meeting (once, with internet, ideally the night before)**
1. Unzip `dtmir_pro_plus_starter_code.zip`, open a terminal in the folder.
2. `pip install -r requirements.txt` (this is the only step that needs internet; PyTorch is a big download).
3. Run `python demo.py` once to confirm it works on that exact laptop. (Correction to an earlier note: `demo.py` does NOT download VGG weights; it doesn't use the VGG loss at all. VGG weights are only fetched by the training scripts.)
4. Optional: `python demo.py --image somephoto.jpg` with a real hazy/rainy photo for a more convincing picture.

**In the meeting**
1. `cd` into the folder, say "let me show you this actually runs", type `python demo.py` (takes seconds).
2. Point out, in order:
   - It runs two setups: baseline DTMIR-Pro, then ours.
   - "gradient reaching the translator": **0.000000** for baseline vs a **real nonzero number** for ours. This is live proof the "unfreeze the translator" change is wired in.
   - The routing weights line (about `[0.333, 0.333, 0.333]`): proof the soft blending combines all three streams instead of picking one. (They are equal only because the model is untrained; after training they should differ per image.)
3. Show `demo_output.png` briefly (see Section 12).
4. Say plainly: the model is untrained, so images are noise; this run proves the pipeline works end to end, not restoration quality.

**If something goes wrong in the room**
- No internet: fine, demo needs none after setup.
- Different laptop: do the setup and one test run before the meeting starts.
- Live error: don't debug live. Open the `demo_output.png` and printed numbers from your earlier successful run.

## 12. How to explain demo_output.png

Five panels, left to right: (1) input (a fake foggy gradient unless you pass a real photo), (2-4) haze-look, rain-look, snow-look versions from the translator, (5) final restored output. Panels 2-5 look like static / flat colour because the weights are random, not trained. It is only evidence that "five distinct stages run and nothing crashes". Don't lead with it. The two printed numbers (translator gradient, routing weights) are the stronger evidence. If asked "does it produce images": show it, say the one sentence above, move on.

## 13. Plain-language walkthrough of the code (for explaining to anyone)

Two stages, as in the paper: **Stage 1 translator** (one bad-weather photo -> three fake weather versions) and **Stage 2 restorer** (those three versions -> clean photo), each with its own training script.

- `weather_prompt.py` — how the network is told what weather to imagine. Old way: rigid label (haze OR rain OR snow). Ours (change 1): a small piece looks at the photo and invents a flexible description.
- `dynamic_attention.py` — helps the network focus on important image regions. Real bug fixed here: first version compared every pixel with every other pixel and would crash on real-sized photos; now compares channels instead, as professional restoration models do.
- `translation_net.py` — the translator. Holds the on/off switch for change 2 (allowed to keep learning, or locked after first training).
- `discriminator.py` — a small judge network that guesses "real or fake rainy photo?"; the translator trains against it. Standard technique, not our invention.
- `routing.py` — change 3. Old way picks exactly one of the three versions and discards two; ours blends all three with learned weights.
- `fusion.py` — combines pairs of weather versions (e.g. haze+rain) for photos with several problems. Kept exactly as in the paper.
- `restoration_net.py` — the restorer. `dtmir_plus.py` — pure wiring; reads the config and connects everything in the right order.
- `data/` — only describes the folder layout each stage expects; no photos included.
- `losses.py` — the scoring rules used in training. `train_translation.py` / `train.py` — the two training scripts (both test-run on tiny fake data). `demo.py` — the meeting demo. `sanity_check.py` — quick developer correctness check. `configs/config.yaml` — one settings file with the three old-vs-ours switches.

**Real vs placeholder (lead with this if asked "is it really working")**
- Proven to work: the pipeline's plumbing, all three changes wired in, both training scripts running.
- Not yet meaningful: everything was tested only on random fake images, so no loss value or output means anything until training on real datasets.
- Simplified / TODO: see Section 6 (hand-written SSIM, simplified routing and decoder, stage-1 checkpoint not loaded, etc.).

## 14. Communication preferences for the new chat
Explain in simple, non-technical language; give complete ready-to-use files (full LaTeX, full code), not partial diffs; point out directly when something stated is wrong; don't omit content from source material without being told.
