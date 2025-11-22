# audioShieldNet/asnet_6/audioshieldnet/losses/ood_energy.py

import torch
import torch.nn.functional as F
from audioshieldnet.security.ood import energy_score

def energy_margin_loss(logits_id, logits_ood, margin=0.5, T=1.0):
    """
    Encourage energy separation: E_ood - E_id >= margin (hinge loss)
    → pushes OOD energies up while keeping ID stable.
    """
    E_id = energy_score(logits_id, T=T)
    E_ood = energy_score(logits_ood, T=T)
    diff = E_ood - E_id
    loss = F.relu(margin - diff).mean()  # hinge: max(0, margin - (E_ood - E_id))
    return loss, E_id.detach(), E_ood.detach()
