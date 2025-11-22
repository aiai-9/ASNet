# audioshieldnet/security/calibrate.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TempScaler(nn.Module):
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.logT = nn.Parameter(torch.log(torch.tensor(float(T))))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.exp(self.logT)

@torch.no_grad()
def calibrate_temperature(model_forward, loader, device, init_T=1.0, steps=200, lr=1e-2):
    """
    Simple temperature scaling fit: expects model_forward(x) -> logits (raw).
    We perform a small optimization on a held-out loader.
    """
    scaler = TempScaler(init_T).to(device)
    opt = torch.optim.Adam([scaler.logT], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model_forward_device = model_forward

    model_forward_device_eval = lambda x: model_forward(x.to(device)).detach()

    model_forward_device_eval  # convenience
    model_forward_device  # no-op

    for _ in range(steps):
        for wav, y, *rest in loader:
            wav = wav.to(device)
            y = y.to(device).float()
            logits = model_forward(wav)
            logits = scaler(logits)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return scaler
