# audioshieldnet/security/ood.py 
import torch
import torch.nn.functional as F

def energy_score(logits, T=1.0, binary_logit_is_fake=True):
    """
    Returns energy (higher → more OOD).
    - If logits is shape [B,2]: standard -T*logsumexp(logits/T, dim=-1)
    - If logits is shape [B]: treat as single binary logit.
      If binary_logit_is_fake=True, we assume logits = logit(p_fake) ≡ [0, z].
      Else logits = logit(p_real) ≡ [z, 0].
    """
    if logits.ndim == 2 and logits.size(-1) > 1:
        return -T * torch.logsumexp(logits / T, dim=-1)
    # single-logit path:
    z = logits / T
    if binary_logit_is_fake:
        return -F.softplus(z) * T       # -log(1 + e^{z})
    else:
        return -F.softplus(-z) * T      # -log(1 + e^{-z})
