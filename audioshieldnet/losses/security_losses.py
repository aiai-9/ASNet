# audioShieldNet/asnet_6/audioshieldnet/losses/security_losses.py
# -*- coding: utf-8 -*-
"""
Security-oriented auxiliary losses:
- CMRA (cross-modal robustness alignment) losses
- Energy-gated consistency objectives for adversarial/OOD branches
"""

from typing import Optional
import torch
import torch.nn.functional as F


# ---------------------------
# CMRA: simple hinge version
# ---------------------------
def cmra_loss(
    z_spec: torch.Tensor,
    z_pros: torch.Tensor,
    margin: float = 0.7,
) -> torch.Tensor:
    """
    Penalize cosine similarity only when it exceeds 'margin':
        sim = cos(z_spec, z_pros)
        loss = mean( max(0, sim - margin) )
    """
    if z_spec is None or z_pros is None:
        raise ValueError("cmra_loss expects both z_spec and z_pros.")
    if z_spec.shape != z_pros.shape:
        raise ValueError(f"cmra_loss: shape mismatch {z_spec.shape} vs {z_pros.shape}")

    sim = F.cosine_similarity(z_spec, z_pros, dim=1)  # [-1, 1]
    return torch.clamp(sim - margin, min=0.0).mean()


# --------------------------------------
# CMRA corridor (alignment + repulsion)
# --------------------------------------
def cmra_corridor_loss(
    z_spec: torch.Tensor,
    z_pros: torch.Tensor,
    s_align: float = 0.25,
    s_max: float = 0.55,
    w_align: float = 1.0,
    w_repel: float = 0.6,
) -> torch.Tensor:
    """
    Keep cosine similarity in [s_align, s_max]:
      L_align = ReLU(s_align - sim)   (too dissimilar)
      L_repel = ReLU(sim - s_max)     (too similar / collapse)
      L = w_align*L_align + w_repel*L_repel
    Inputs are L2-normalized internally for stability.
    """
    if z_spec is None or z_pros is None:
        raise ValueError("cmra_corridor_loss expects both z_spec and z_pros.")
    if z_spec.shape != z_pros.shape:
        raise ValueError(f"cmra_corridor_loss: shape mismatch {z_spec.shape} vs {z_pros.shape}")

    z_spec = F.normalize(z_spec, dim=1)
    z_pros = F.normalize(z_pros, dim=1)
    sim = (z_spec * z_pros).sum(dim=1)  # [-1, 1]

    align = torch.clamp(s_align - sim, min=0.0)
    repel = torch.clamp(sim - s_max,   min=0.0)
    return (w_align * align + w_repel * repel).mean()


# --------------------------------------
# Energy gate + consistency objectives
# --------------------------------------
def _flatten_1d(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape [B]."""
    if x.dim() == 0:
        return x.unsqueeze(0)
    if x.dim() > 1:
        return x.view(x.size(0))
    return x


def energy_gate(
    E_id: torch.Tensor,
    w_g: float = 1.0,
    b_g: float = 0.0,
    min_g: float = 0.3,
    max_g: float = 1.0,
    detach_energy: bool = True,
) -> torch.Tensor:
    """
    Per-sample robustness weight from energy:
      g = clamp( sigmoid(w_g * E + b_g), [min_g, max_g] )
    Accepts E as [B] or [B,1].
    """
    if E_id is None:
        raise ValueError("energy_gate expects E_id.")

    E = _flatten_1d(E_id)
    if detach_energy:
        E = E.detach()

    g = torch.sigmoid(w_g * E + b_g)
    g = torch.clamp(g, min=min_g, max=max_g)
    return g


def _check_broadcastable(a: torch.Tensor, b: torch.Tensor) -> None:
    """
    Raise when shapes are not broadcastable in PyTorch sense.
    """
    try:
        torch.empty(0).new_empty(())  # no-op to ensure torch import isn't stripped
        _ = a + b  # will throw if not broadcastable
    except Exception as e:
        raise ValueError(f"Shapes not broadcastable: {tuple(a.shape)} vs {tuple(b.shape)}") from e


def energy_weighted_adv_consistency(
    logits_clean: torch.Tensor,
    logits_adv: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean-squared logit consistency, weighted by gates:
      per = mean( (logits_clean - logits_adv)^2, dim=logit )
      return mean( gates * per ) if gates provided else mean(per)
    """
    if logits_clean.shape != logits_adv.shape:
        raise ValueError(f"adv_consistency: shape mismatch {logits_clean.shape} vs {logits_adv.shape}")

    lc = logits_clean.view(logits_clean.size(0), -1)
    la = logits_adv.view(logits_adv.size(0), -1)
    per = (lc - la).pow(2).mean(dim=1)  # [B]

    if gates is not None:
        gates = _flatten_1d(gates)
        _check_broadcastable(per, gates)
        per = per * gates

    return per.mean()


def energy_weighted_ood_consistency(
    E_id: torch.Tensor,
    E_ood: torch.Tensor,
    gates: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Encourage separation between OOD and ID energies:
      per = (E_ood - E_id)^2
      return mean( gates * per ) if gates provided else mean(per)
    Accepts [B] or [B,1] inputs.
    """
    E_id = _flatten_1d(E_id)
    E_ood = _flatten_1d(E_ood)
    if E_id.shape != E_ood.shape:
        raise ValueError(f"ood_consistency: shape mismatch {E_id.shape} vs {E_ood.shape}")

    per = (E_ood - E_id).pow(2)  # [B]

    if gates is not None:
        gates = _flatten_1d(gates)
        _check_broadcastable(per, gates)
        per = per * gates

    return per.mean()


__all__ = [
    "cmra_loss",
    "cmra_corridor_loss",
    "energy_gate",
    "energy_weighted_adv_consistency",
    "energy_weighted_ood_consistency",
]
