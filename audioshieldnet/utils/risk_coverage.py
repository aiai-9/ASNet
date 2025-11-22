# audioShieldNet/asnet_1/audioshieldnet/utils/risk_coverage.py

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import wandb

def _eer_from_scores(y_true, y_prob):
    """Compute EER from probs (1D arrays). Returns float in [0,1]."""
    if (y_true.size == 0) or (np.unique(y_true).size < 2):
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob.astype(float))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)

def _risk_fn(y_true, y_prob, metric="eer"):
    if metric == "1-auc":
        # safe when both classes present
        if (np.unique(y_true).size < 2):
            return float("nan")
        return 1.0 - float(roc_auc_score(y_true.astype(int), y_prob.astype(float)))
    # default: EER
    return _eer_from_scores(y_true, y_prob)

def risk_coverage_from_energy(y_true, y_prob, energies, metric="eer", qs=None):
    """
    Sweep thresholds over energy quantiles (low energy = keep).
    Returns (covs, risks, AURC)
    covs: coverage fractions
    risks: risk values at those coverages (EER or 1-AUC)
    AURC: area under the risk-coverage curve (lower better)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    energies = np.asarray(energies)

    assert y_true.size == y_prob.size == energies.size
    if qs is None:
        qs = np.linspace(0.0, 1.0, 21)  # 0%,5%,...,100%

    covs, risks = [], []
    for q in qs:
        tau = np.quantile(energies, q, method="higher") if energies.size else np.nan
        keep = energies < tau
        cov = float(keep.mean()) if keep.size else np.nan
        if keep.any():
            r = _risk_fn(y_true[keep], y_prob[keep], metric=metric)
        else:
            r, cov = float("nan"), 0.0
        covs.append(cov)
        risks.append(r)

    covs = np.array(covs)
    risks = np.array(risks)

    # Fill NaNs in risks using forward fill then final value, to allow trapz
    if np.any(~np.isfinite(risks)):
        valid = np.isfinite(risks)
        if valid.any():
            last = None
            for i in range(len(risks)):
                if valid[i]:
                    last = risks[i]
                else:
                    risks[i] = last if last is not None else 0.0
        else:
            # all NaNs
            AURC = float("nan")
            return covs, risks, AURC

    # integrate over coverage (sorted)
    order = np.argsort(covs)
    AURC = float(np.trapz(risks[order], covs[order]))

    return covs, risks, AURC

def ece_on_subset(y_true, y_prob, n_bins=15):
    """Simple ECE (expected calibration error) on a subset."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        conf = float(y_prob[m].mean())
        acc  = float((y_true[m] == (y_prob[m] >= 0.5)).mean())
        ece += (np.sum(m) / y_true.size) * abs(acc - conf)
    return float(ece)


def wandb_log_risk_coverage(wb, coverages, risks, title="RC", max_points=1000):
    """Log risk–coverage curve without pushing big ndarrays into wandb.log."""
    if wb is None:
        return
    covs = np.asarray(coverages).ravel()
    rks  = np.asarray(risks).ravel()
    if covs.size == 0 or rks.size == 0:
        return

    # downsample to at most max_points to keep tables light
    n = covs.size
    idx = np.linspace(0, n - 1, min(n, max_points)).astype(int)

    tbl = wandb.Table(
        data=[[float(covs[i]), float(rks[i])] for i in idx],
        columns=["coverage", "risk"]
    )
    wb.log({f"val/rc_{title}": wandb.plot.line(tbl, "coverage", "risk", title=title)}, commit=False)