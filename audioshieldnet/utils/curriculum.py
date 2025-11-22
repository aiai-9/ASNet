# audioshieldnet/utils/curriculum.py

from typing import Any, Dict, Tuple

def as_epoch(val: Any, total_epochs: int, default_if_missing: int) -> int:
    """
    Convert a config value to an absolute epoch index.

    Rules:
      - If val is None/missing → use default_if_missing
      - If 0 < val <= 1.0 (float) → interpret as fraction of total_epochs
      - Else → interpret as absolute epoch (int)
    """
    if val is None:
        return int(default_if_missing)
    try:
        v = float(val)
    except (TypeError, ValueError):
        return int(default_if_missing)

    # Fractional gate (e.g., 0.6 means 60% of total epochs)
    if 0.0 < v <= 1.0:
        return max(0, int(round(v * total_epochs)))
    # Absolute epoch
    return max(0, int(round(v)))


def resolve_curriculum(train_cfg: Dict[str, Any], total_epochs: int) -> Dict[str, int]:
    """
    Read fractional or absolute keys from train_cfg, compute absolute epoch gates,
    write them back into the dict using the traditional *_start_epoch keys, and return them.

    Supported keys (either/or):
      - stability_warmup_epochs  | stability_warmup_frac
      - sam_start_epoch          | sam_start_frac
      - energy_start_epoch       | energy_start_frac
      - ood_start_epoch          | ood_start_frac
      - adv_start_epoch          | adv_start_frac

    Precedence: explicit *_start_epoch wins over *_start_frac.
    """
    t = train_cfg

    stability = as_epoch(
        # if explicit epochs given, use; else use frac; else keep existing epochs default (or 0)
        t.get("stability_warmup_epochs", t.get("stability_warmup_frac", None)),
        total_epochs=total_epochs,
        default_if_missing=t.get("stability_warmup_epochs", 0),
    )

    sam_ep = as_epoch(
        t.get("sam_start_epoch", t.get("sam_start_frac", None)),
        total_epochs=total_epochs,
        default_if_missing=t.get("sam_start_epoch", 0),
    )

    energy_ep = as_epoch(
        t.get("energy_start_epoch", t.get("energy_start_frac", None)),
        total_epochs=total_epochs,
        default_if_missing=t.get("energy_start_epoch", 9999),
    )

    ood_ep = as_epoch(
        t.get("ood_start_epoch", t.get("ood_start_frac", None)),
        total_epochs=total_epochs,
        default_if_missing=t.get("ood_start_epoch", 9999),
    )

    adv_ep = as_epoch(
        t.get("adv_start_epoch", t.get("adv_start_frac", None)),
        total_epochs=total_epochs,
        default_if_missing=t.get("adv_start_epoch", t.get("adv_warmup_epochs", 0)),
    )

    # Write normalized values back so the rest of the code can keep using *_start_epoch
    t["stability_warmup_epochs"] = stability
    t["sam_start_epoch"] = sam_ep
    t["energy_start_epoch"] = energy_ep
    t["ood_start_epoch"] = ood_ep
    t["adv_start_epoch"] = adv_ep

    return {
        "stability_warmup_epochs": stability,
        "sam_start_epoch": sam_ep,
        "energy_start_epoch": energy_ep,
        "ood_start_epoch": ood_ep,
        "adv_start_epoch": adv_ep,
    }
