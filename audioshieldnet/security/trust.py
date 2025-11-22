# audioshieldnet/security/trust.py
import torch
from .ood import energy_score

def suspicious_flags(logits: torch.Tensor, energy_thr: float, temp_scaler=None, tau=(0.4, 0.6)):
    """
    Returns (probs, energy, suspicious_mask)
    suspicious_mask is True for samples considered "Suspicious" (abstain).
    """
    if temp_scaler is not None:
        logits = temp_scaler(logits)
    probs = torch.sigmoid(logits)
    E = energy_score(logits)
    is_ood = E > energy_thr
    is_lowconf = (probs > tau[0]) & (probs < tau[1])
    suspicious = is_ood | is_lowconf
    return probs, E, suspicious
