# audioShieldNet/asnet_1/audioshieldnet/security/conformal.py
import numpy as np

def fit_tau_conformal(energies_cal, target_coverage=0.90):
    """
    Conformal-style one-sided threshold for ID acceptance.
    We accept if E < tau. To hit coverage C, choose tau at the C-quantile of energies.
    """
    energies_cal = np.asarray(energies_cal).reshape(-1)
    if energies_cal.size == 0:
        return None
    target_coverage = float(np.clip(target_coverage, 0.01, 0.999))
    tau = float(np.quantile(energies_cal, target_coverage, method="higher"))
    return tau

def auto_tau_from_val_energies(energies_val, q=0.95):
    e = np.asarray(energies_val).reshape(-1)
    if e.size == 0:
        return None
    return float(np.quantile(e, q, method="higher"))
