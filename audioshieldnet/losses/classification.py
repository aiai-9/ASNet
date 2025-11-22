# audioShieldNet/asnet_6/audioshieldnet/losses/classification.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, Tuple

import torch
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _to_device(x, device):
    if x is None:
        return None
    return x.to(device) if hasattr(x, "to") else x

def effective_num_weights(n0: float, n1: float, beta: float = 0.9999, device: str = "cuda"):
    """Effective Number of Samples (Cui et al., CVPR 2019) smoother class weights."""
    def eff(n): return (1.0 - beta) / (1.0 - beta ** max(n, 1.0))
    w0, w1 = eff(n0), eff(n1)
    s = w0 + w1
    return torch.tensor([w0 / s, w1 / s], dtype=torch.float32, device=device)



def compute_class_weights_from_counts(real_count: float, fake_count: float, device: str):
    """
    Class-balanced weights that sum to ~1 and down-weight the majority.
      w0 -> weight used on class 0 (real)
      w1 -> weight used on class 1 (fake)
    """
    n0 = float(real_count)
    n1 = float(fake_count)
    n  = max(1.0, n0 + n1)
    w0 = n / (2.0 * max(n0, 1.0))
    w1 = n / (2.0 * max(n1, 1.0))
    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    return w


def _maybe_flip_labels(y: torch.Tensor, pos_is_fake: bool) -> torch.Tensor:
    """Ensure target=1 means 'fake' for the internal loss."""
    y = y.float()
    return y if pos_is_fake else (1.0 - y)


@dataclass
class LossInfo:
    name: str
    details: Dict[str, Any]


# -----------------------------
# Losses
# -----------------------------
class BCEWithClassWeights(torch.nn.Module):
    """
    BCE with per-class weights applied per-sample (robust for imbalanced data).
    - Supports label smoothing.
    - Works with/without weighted sampler (no double counting like pos_weight).
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0, pos_is_fake: bool = True):
        super().__init__()
        self.class_weights = class_weights  # tensor([w0, w1]) on device
        self.label_smoothing = float(label_smoothing)
        self.pos_is_fake = bool(pos_is_fake)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> torch.Tensor:
        y = _maybe_flip_labels(targets, self.pos_is_fake)
        if self.label_smoothing > 0:
            eps = self.label_smoothing
            y = y * (1 - eps) + 0.5 * eps

        # per-sample BCE
        bce_per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")

        # per-sample class weights
        if self.class_weights is not None:
            cw0, cw1 = self.class_weights[0], self.class_weights[1]
            w = torch.where(y > 0.5, cw1, cw0)
            bce_per = bce_per * w

        if sample_weights is not None:
            bce_per = bce_per * sample_weights

        return bce_per.mean()


class FocalLoss(torch.nn.Module):
    """
    Binary focal loss (gamma>0 focuses on hard examples).
    alpha can be:
      - scalar in [0,1] (weight for positive class)
      - 2-vector [alpha_neg, alpha_pos]
      - None → no alpha weighting
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor | float] = 0.25,
                 pos_is_fake: bool = True):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.pos_is_fake = bool(pos_is_fake)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> torch.Tensor:
        y = _maybe_flip_labels(targets, self.pos_is_fake)
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        fl = (1 - pt).pow(self.gamma) * ce

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor) and self.alpha.numel() == 2:
                a0, a1 = self.alpha[0], self.alpha[1]
                a = torch.where(y > 0.5, a1, a0)
            else:
                a = float(self.alpha)
                a = torch.where(y > 0.5, torch.tensor(a, device=logits.device),
                                torch.tensor(1.0 - a, device=logits.device))
            fl = a * fl

        if sample_weights is not None:
            fl = fl * sample_weights

        return fl.mean()


class AsymmetricFocalLoss(torch.nn.Module):
    """
    Asymmetric Focal Loss (ASL) for extreme imbalance.
    Reference: "ASL for Multi-Label Classification" (Ben-Baruch et al.)
    Here adapted for binary: different focusing for pos/neg.
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0,
                 clip: float = 0.05, eps: float = 1e-8, pos_is_fake: bool = True):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.eps = float(eps)
        self.pos_is_fake = bool(pos_is_fake)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> torch.Tensor:
        y = _maybe_flip_labels(targets, self.pos_is_fake)
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # asymmetric clipping
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # basic CE
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # asymmetric focusing
        p_t = xs_pos * y + xs_neg * (1 - y)
        gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        loss *= (1 - p_t) ** gamma

        loss = -loss
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()


class LogitAdjustedBCE(torch.nn.Module):
    """
    Logit-Adjusted Cross Entropy (Menon et al., NeurIPS'20).
    Adjust logits by log-priors to counter label imbalance:
      logits' = logits + log(p/(1-p))
    """
    def __init__(self, pos_prior: float, tau: float = 1.0, pos_is_fake: bool = True):
        super().__init__()
        pos_prior = min(max(1e-6, float(pos_prior)), 1 - 1e-6)
        bias = torch.log(torch.tensor(pos_prior / (1 - pos_prior)))
        self.register_buffer("bias", bias * float(tau))
        self.pos_is_fake = bool(pos_is_fake)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> torch.Tensor:
        y = _maybe_flip_labels(targets, self.pos_is_fake)
        logits_adj = logits + self.bias
        loss = F.binary_cross_entropy_with_logits(logits_adj, y, reduction="none")
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()


# -----------------------------
# Loss Wrapper / Factory
# -----------------------------
class WarmUpSwitcher(torch.nn.Module):
    """
    Switch between two losses after warm-up epochs.
    E.g., focal for the first 2 epochs, then class-balanced BCE.
    """
    def __init__(self, warmup_epochs: int, first: torch.nn.Module, second: torch.nn.Module):
        super().__init__()
        self.warmup_epochs = int(warmup_epochs)
        self.first = first
        self.second = second

    def forward(self, logits, targets, sample_weights: Optional[torch.Tensor] = None,
                epoch: Optional[int] = None) -> torch.Tensor:
        if epoch is None or epoch < self.warmup_epochs:
            return self.first(logits, targets, sample_weights=sample_weights, epoch=epoch)
        return self.second(logits, targets, sample_weights=sample_weights, epoch=epoch)


def _read_counts_from_cfg(cfg: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    pc = (cfg.get("data", {}) or {}).get("prior_counts", None)
    if pc is None:
        return None
    return float(pc.get("real", 1.0)), float(pc.get("fake", 1.0))


def build_classification_loss(cfg: Dict[str, Any], device: str) -> Tuple[Callable, LossInfo]:
    """
    Factory: chooses an imbalance-aware loss according to cfg.
    Supported:
      train.loss:
        name: ["balanced_bce", "focal", "asl", "logit_adjusted_bce"]
        label_smoothing: 0.0
        focal:
          gamma: 2.0
          alpha: 0.25 or [a_neg, a_pos]
        asl:
          gamma_pos: 0.0
          gamma_neg: 4.0
          clip: 0.05
        logit_adjusted:
          tau: 1.0
        warmup:
          enable: true
          epochs: 2
          first: "focal" | "asl"
    Also uses:
      train.pos_class   (default "fake")
      data.prior_counts (to compute class weights or priors)
    """
    tl = (cfg.get("train", {}) or {})
    loss_name = str((tl.get("loss", {}) or {}).get("name", "balanced_bce")).lower()
    label_smoothing = float((tl.get("loss", {}) or {}).get("label_smoothing", 0.0))
    pos_is_fake = str(tl.get("pos_class", "fake")).lower() == "fake"

    counts = _read_counts_from_cfg(cfg)
    tloss = (tl.get("loss", {}) or {})

    if counts is not None:
        n_real, n_fake = float(counts[0]), float(counts[1])
        total = max(1.0, n_real + n_fake)
        pos_prior = n_fake / total

        # Prefer explicit class-balanced weights unless user *really* wants effective_num
        use_eff = bool(tloss.get("use_effective_num", False))
        if use_eff:
            beta = float(tloss.get("effective_beta", 0.9999))
            w = effective_num_weights(n_real, n_fake, beta=beta, device=device)
        else:
            w = compute_class_weights_from_counts(n_real, n_fake, device=device)

        # Visibility: print exactly what we’ll use
        try:
            print(f"[LOSS][priors] counts(real={int(n_real)}, fake={int(n_fake)}) "
                  f"→ pos_prior(fake)={pos_prior:.4f}  "
                  f"→ class_weights(w0={float(w[0]):.4f}, w1={float(w[1]):.4f})  "
                  f"(using {'effective_num' if use_eff else 'balanced'} weights)")
        except Exception:
            pass
    else:
        # No priors found: fall back to equal weights, but say it clearly once
        w = torch.tensor([0.5, 0.5], device=device)
        pos_prior = 0.5
        try:
            print("[LOSS][priors][WARN] No prior_counts in cfg.data; using equal weights (0.5/0.5).")
        except Exception:
            pass


    # Build base losses
    balanced_bce = BCEWithClassWeights(class_weights=w, label_smoothing=label_smoothing, pos_is_fake=pos_is_fake)

    if loss_name == "focal":
        fcfg = (tl.get("loss", {}) or {}).get("focal", {}) or {}
        gamma = float(fcfg.get("gamma", 2.0))
        alpha = fcfg.get("alpha", 0.25)
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        focal = FocalLoss(gamma=gamma, alpha=alpha, pos_is_fake=pos_is_fake)
        return focal, LossInfo(name="focal", details={"gamma": gamma, "alpha": alpha})

    if loss_name == "asl":
        acfg = (tl.get("loss", {}) or {}).get("asl", {}) or {}
        asl = AsymmetricFocalLoss(
            gamma_pos=float(acfg.get("gamma_pos", 0.0)),
            gamma_neg=float(acfg.get("gamma_neg", 4.0)),
            clip=float(acfg.get("clip", 0.05)),
            pos_is_fake=pos_is_fake
        )
        return asl, LossInfo(name="asl", details=acfg)

    if loss_name == "logit_adjusted_bce":
        lacfg = (tl.get("loss", {}) or {}).get("logit_adjusted", {}) or {}
        tau = float(lacfg.get("tau", 1.0))
        la = LogitAdjustedBCE(pos_prior=pos_prior, tau=tau, pos_is_fake=pos_is_fake)
        return la, LossInfo(name="logit_adjusted_bce", details={"tau": tau, "pos_prior": pos_prior})

    # default: balanced_bce, possibly with warmup focal/asl
    wcfg = (tl.get("loss", {}) or {}).get("warmup", {}) or {}
    if bool(wcfg.get("enable", False)):
        warm_epochs = int(wcfg.get("epochs", 2))
        first_name = str(wcfg.get("first", "focal")).lower()
        if first_name == "asl":
            first = AsymmetricFocalLoss(pos_is_fake=pos_is_fake)
            tag = "asl→balanced_bce"
        else:
            first = FocalLoss(pos_is_fake=pos_is_fake)
            tag = "focal→balanced_bce"
        wrapped = WarmUpSwitcher(warmup_epochs=warm_epochs, first=first, second=balanced_bce)
        return wrapped, LossInfo(name=tag, details={
            "warmup_epochs": warm_epochs,
            "class_weights": (float(w[0].item()), float(w[1].item())),
            "label_smoothing": label_smoothing
        })

    info = {
        "class_weights": (float(w[0].item()), float(w[1].item())),
        "label_smoothing": label_smoothing
    }
    try:
        print(f"[LOSS] Using {loss_name if 'loss_name' in locals() else 'balanced_bce'} "
              f"with {info}")
    except Exception:
        pass
    return balanced_bce, LossInfo(name="balanced_bce", details=info)

