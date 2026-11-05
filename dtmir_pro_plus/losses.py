"""
Loss functions, following DTMIR-Pro's stated objective:
  - Translation stage: adversarial loss (per-domain discriminators) +
    a contrastive loss between translated images (keeps haze/rain/snow
    outputs visually distinct from each other)
  - Restoration stage: L1 reconstruction loss + a contrastive loss that
    pulls the restored image toward the ground truth and pushes it away
    from the (still-degraded) translated images

The contrastive losses use VGG features as the comparison space (standard
for this kind of "contrastive regularization" in low-level vision, and
what DTMIR-Pro itself uses) with a safe fallback: if pretrained VGG
weights can't be downloaded (e.g. no internet, as in this sandbox), we
fall back to a randomly-initialized VGG of the same shape so the code
still runs -- swap back to pretrained weights before any real training,
since a random-init "feature extractor" isn't meaningful for comparing
images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGFeatures(nn.Module):
    """Frozen VGG16 features up to a mid-level layer, used as the
    comparison space for contrastive losses."""

    def __init__(self, layer_idx: int = 16):
        super().__init__()
        try:
            vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
            self._pretrained = True
        except Exception:
            # no internet access to fetch pretrained weights (e.g. sandboxed dev
            # environment) -- fall back to random init so the pipeline still runs.
            # DO NOT use this fallback for real training/evaluation.
            vgg = torchvision.models.vgg16(weights=None)
            self._pretrained = False
        self.features = vgg.features[:layer_idx].eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x expected in [-1, 1] (matches the tanh outputs elsewhere in this repo)
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        return self.features(x)


class ContrastiveRestorationLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module = None, margin: float = 0.0):
        super().__init__()
        self.margin = margin
        self.features = feature_extractor  # None -> falls back to raw pixels

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.features is not None:
            # detach b (positive/negative) so we only backprop through the anchor;
            # this keeps the contrastive loss from blowing up GPU memory
            a, b = self.features(a), self.features(b.detach())
        return F.l1_loss(a, b, reduction="none").mean(dim=[1, 2, 3])

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor,
                negatives: list) -> torch.Tensor:
        pos_dist = self._dist(anchor, positive)
        neg_dists = torch.stack([self._dist(anchor, neg) for neg in negatives], dim=1).mean(dim=1)
        loss = pos_dist / (neg_dists + 1e-6)
        return loss.mean()


class RestorationLoss(nn.Module):
    def __init__(self, l1_weight: float = 1.0, contrastive_weight: float = 0.75,
                 use_vgg: bool = True):
        super().__init__()
        self.l1_weight = l1_weight
        self.contrastive_weight = contrastive_weight
        feat = VGGFeatures() if use_vgg else None
        self.contrastive = ContrastiveRestorationLoss(feature_extractor=feat)

    def forward(self, outputs: dict, ground_truth: torch.Tensor) -> dict:
        restored = outputs["restored"]
        negatives = list(outputs["translated"].values())

        l1 = F.l1_loss(restored, ground_truth)
        cl = self.contrastive(restored, ground_truth, negatives)
        total = self.l1_weight * l1 + self.contrastive_weight * cl
        return {"total": total, "l1": l1, "contrastive": cl}


class TranslationLoss(nn.Module):
    """DTMIR-Pro's translation-stage objective: adversarial loss (per
    domain) + a contrastive loss that keeps the three translated domains
    visually distinct from one another (anchor = translated image,
    positive = that domain's ground truth, negatives = the other two
    translated domains)."""

    def __init__(self, contrastive_weight: float = 1.0, adv_weight: float = 0.5,
                 use_vgg: bool = True):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.adv_weight = adv_weight
        feat = VGGFeatures() if use_vgg else None
        self.contrastive = ContrastiveRestorationLoss(feature_extractor=feat)
        self.bce = nn.BCEWithLogitsLoss()

    def generator_loss(self, translated: dict, ground_truth: dict,
                        discriminators, disc_scores: dict) -> dict:
        # adversarial: translator wants each discriminator to say "real"
        adv = sum(self.bce(disc_scores[name], torch.ones_like(disc_scores[name]))
                  for name in translated)

        # contrastive: keep haze/rain/snow translations distinct from each other
        cl = 0.0
        names = list(translated.keys())
        for name in names:
            negatives = [translated[n] for n in names if n != name]
            cl = cl + self.contrastive(translated[name], ground_truth[name], negatives)

        total = self.adv_weight * adv + self.contrastive_weight * cl
        return {"total": total, "adversarial": adv, "contrastive": cl}

    def discriminator_loss(self, real_scores: dict, fake_scores: dict) -> torch.Tensor:
        loss = 0.0
        for name in real_scores:
            real_loss = self.bce(real_scores[name], torch.ones_like(real_scores[name]))
            fake_loss = self.bce(fake_scores[name], torch.zeros_like(fake_scores[name]))
            loss = loss + 0.5 * (real_loss + fake_loss)
        return loss

