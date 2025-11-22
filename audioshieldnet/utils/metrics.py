# audioShieldNet/asnet_1/audioshieldnet/utils/metrics.py

import numpy as np, torch
def energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    if logits.dim() == 1:
        two = torch.stack([-logits, logits], dim=1)
    else:
        two = logits
    return -T * torch.logsumexp(two / T, dim=1)

def expected_calibration_error(y_true, y_prob, n_bins=15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i+1])
        if not np.any(m): continue
        acc = (y_true[m] == (y_prob[m] >= 0.5)).mean()
        conf = y_prob[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)

def fnr_at_tpr95(y_true, y_score) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(1.0 - tpr[idx])
