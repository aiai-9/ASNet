

# audioshieldnet/metrics/far_frr.py

import os
from typing import Dict, Any, Tuple

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def compute_far_frr(
    y_true,
    y_score,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """
    Compute FAR (false-accept rate), FRR (false-reject rate),
    FER (average of FAR & FRR) over all thresholds, plus EER & AUC.

    Args:
        y_true: 1D array-like of ground-truth labels (0=real, 1=fake by convention).
        y_score: 1D array-like of scores/probabilities (higher = more fake).
        pos_label: label treated as "positive" (fake).

    Returns:
        dict with:
            - far: array of FAR values
            - frr: array of FRR values
            - fer: array of FER values
            - thresholds: thresholds corresponding to each point
            - eer: equal-error rate (FAR ~ FRR)
            - eer_threshold: threshold at EER
            - auc_far_vs_1_minus_frr: AUROC over FAR vs (1-FRR)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # Standard ROC: fpr = FAR, tpr = TPR
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)

    far = fpr
    frr = 1.0 - tpr
    fer = 0.5 * (far + frr)

    # EER: find point where |FAR - FRR| is minimal
    idx_eer = int(np.argmin(np.abs(far - frr)))
    eer = float(0.5 * (far[idx_eer] + frr[idx_eer]))
    eer_threshold = float(thresholds[idx_eer])

    # AUC over FAR vs (1 - FRR)  (equivalent to ROC AUC)
    auc_far_vs_1_minus_frr = float(metrics.auc(far, 1.0 - frr))

    return {
        "far": far,
        "frr": frr,
        "fer": fer,
        "thresholds": thresholds,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "auc_far_vs_1_minus_frr": auc_far_vs_1_minus_frr,
    }


def plot_far_frr_curve(
    far: np.ndarray,
    frr: np.ndarray,
    out_path: str,
    title: str = "FAR–FRR Curve",
) -> str:
    """
    Plot FAR vs FRR (FAR on x-axis, FRR on y-axis) and save to out_path.

    Args:
        far: 1D array of FAR values.
        frr: 1D array of FRR values.
        out_path: where to save the PNG.
        title: plot title.

    Returns:
        out_path (for convenience).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.plot(far, frr, linewidth=2)
    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("False Rejection Rate (FRR)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return out_path


def add_far_frr_to_report(
    report: Dict[str, Any],
    section: str,
    far_frr_stats: Dict[str, Any],
) -> None:
    """
    Convenience helper: stuff scalar FAR/FRR/FER-at-EER style stats into
    a given section of the main report dict (e.g., 'clean' or 'adversarial').

    Args:
        report: top-level report dict from evaluator.
        section: 'clean' or 'adversarial' etc.
        far_frr_stats: output from compute_far_frr().
    """
    sec = report.setdefault(section, {})

    eer = float(far_frr_stats.get("eer", float("nan")))
    sec["far_frr"] = {
        "eer": eer,  # at EER, FAR ≈ FRR ≈ eer
        "fer_at_eer": eer,  # FER at EER is also ≈ EER
        "eer_threshold": float(far_frr_stats.get("eer_threshold", float("nan"))),
        "auc_far_vs_1_minus_frr": float(
            far_frr_stats.get("auc_far_vs_1_minus_frr", float("nan"))
        ),
    }
