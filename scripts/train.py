#!/usr/bin/env python3
"""
AudioShieldNet — Unified Trainer Entrypoint
-------------------------------------------
Supports both single- and multi-dataset training.
Automatically handles:
  • YAML parsing & placeholder resolution
  • Reproducibility & environment setup
  • Model + consistency loss initialization
  • Dataloader construction with injected priors
  • Optional warm-start checkpoints
  • Trainer launch with W&B logging
"""

import sys, os, yaml, warnings, argparse, torch, shutil
from itertools import islice
from typing import Dict, Any, Optional

# -------------------------------------------------------------
# ✅ Dynamic repo root resolution
# -------------------------------------------------------------
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

# -------------------------------------------------------------
# ✅ Core imports
# -------------------------------------------------------------
from audioshieldnet.engine.trainer import Trainer
from audioshieldnet.data import build_dataloaders
from audioshieldnet.models.asn import build_model
from audioshieldnet.losses.asn_consistency import ASNConsistencyLoss
from audioshieldnet.utils.ema import EMA
from audioshieldnet.utils.seed import fix_seed
from audioshieldnet.utils.cudnn import tune_cudnn
from audioshieldnet.utils.config_utils import resolve_placeholders


# =============================================================
# Utility Helpers
# =============================================================

def _normalize_class_info(class_info: Any) -> Dict[str, int]:
    """Normalize to {'real': N_real, 'fake': N_fake} no matter dict/tuple/namespace."""
    try:
        if isinstance(class_info, dict):
            return {
                "real": int(class_info.get("real", 0)),
                "fake": int(class_info.get("fake", 0)),
            }
        if hasattr(class_info, "real") and hasattr(class_info, "fake"):
            return {"real": int(class_info.real), "fake": int(class_info.fake)}
        if isinstance(class_info, (list, tuple)) and len(class_info) == 2:
            return {"real": int(class_info[0]), "fake": int(class_info[1])}
    except Exception:
        pass
    return {}


def _snapshot_run_files(cfg_path: str, outdir: str) -> None:
    """
    Save critical run-time files (config, trainer, and train script)
    into ckpt_dir/checkpoints for reproducibility.
    """
    os.makedirs(outdir, exist_ok=True)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    def _copy(src: str, dst_name: Optional[str] = None):
        try:
            dst = os.path.join(ckpt_dir, dst_name or os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"[SNAPSHOT] Saved → {dst}")
        except Exception as e:
            print(f"[SNAPSHOT][WARN] Could not copy {src}: {e}")

    _copy(cfg_path)
    _copy(os.path.abspath(__file__), "train.py")
    _copy(os.path.join(PKG_ROOT, "audioshieldnet", "engine", "trainer.py"), "trainer.py")


def _estimate_class_balance(dl_tr, device: str) -> Dict[str, Any]:
    """
    Estimate approximate real/fake ratio and pos_weight for BCE losses.
    """
    n0 = n1 = 0
    for i, batch in enumerate(islice(dl_tr, 50)):  # scan first ~50 mini-batches
        labels = batch[1].detach().reshape(-1)
        n1 += int((labels == 1).sum().item())
        n0 += int((labels == 0).sum().item())
        if n0 + n1 >= 20000:
            break
    total = max(1, n0 + n1)
    frac_fake = n1 / total
    pos_weight_val = max(1.0, float(n0) / max(float(n1), 1.0))
    return {
        "real": n0,
        "fake": n1,
        "frac_fake": frac_fake,
        "pos_weight_tensor": torch.tensor([pos_weight_val], device=device),
        "pos_weight_value": float(pos_weight_val),
    }


# =============================================================
# Main Entrypoint
# =============================================================
def main():
    # -------------------------------
    # 1. Parse arguments & config
    # -------------------------------
    ap = argparse.ArgumentParser(description="AudioShieldNet Training Entrypoint")
    ap.add_argument("--config", required=True, help="Path to YAML config file")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", message="StreamingMediaDecoder has been deprecated")

    cfg = yaml.safe_load(open(args.config))
    _ckpt = cfg.get("ckpt_dir", "run")
    cfg = resolve_placeholders(cfg, {"ckpt_dir": _ckpt})

    # -------------------------------
    # 2. Output directory & snapshot
    # -------------------------------
    outdir = cfg["log"]["outdir"]
    _snapshot_run_files(args.config, outdir)

    # -------------------------------
    # 3. Device, W&B, determinism
    # -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wcfg = cfg["log"].get("wandb", {}) or {}
    wcfg["config_path"] = os.path.abspath(args.config)
    cfg["log"]["wandb"] = wcfg
    use_wandb = bool(wcfg.get("enable", False))

    fix_seed(cfg.get("train", {}).get("seed", 42))
    tune_cudnn()

    # -------------------------------
    # 4. Build model + consistency loss
    # -------------------------------
    feats, net = build_model(cfg, device)
    asn_cfg = cfg.get("asn", {}) or {}
    asn_crit = ASNConsistencyLoss(
        win_lengths=tuple(asn_cfg.get("win_ms", [80, 160])),
        hop=asn_cfg.get("hop_ms", 40),
        margin=asn_cfg.get("margin", 0.2),
        spoof_weight=asn_cfg.get("spoof_weight", 0.5),
        sr=cfg["data"]["sr"],
        hop_length=cfg["data"]["hop"],
        tv_weight=asn_cfg.get("tv_weight", 0.0),
        mel_sample=asn_cfg.get("mel_sample", 16),
        max_time_windows=asn_cfg.get("max_time_windows", 64),
    ).to(device)

    ema = EMA(net, decay=float(cfg.get("ema", {}).get("decay", 0.999))) \
        if cfg.get("ema", {}).get("use_ema", True) else None

    # -------------------------------
    # 5. Build dataloaders + priors
    # -------------------------------
    dl_tr, dl_va, dl_cal = build_dataloaders(cfg)
    priors = _normalize_class_info((cfg.get("data", {}) or {}).get("prior_counts", {}))
    if priors:
        print(f"[INFO] Class priors (train split): {priors}")
    else:
        print("[WARN] Could not infer priors — loss will assume equal weights.")

    # Compute empirical imbalance for reporting / BCE pos_weight
    imb = _estimate_class_balance(dl_tr, device)
    cfg.setdefault("train", {})
    cfg["train"].update({
        "pos_weight_tensor": imb["pos_weight_tensor"],
        "pos_weight_value": imb["pos_weight_value"],
        "y1_frac_est": imb["frac_fake"],
        "class_info_counts": {"real": imb["real"], "fake": imb["fake"]},
    })
    print(f"[INFO] Imbalance → real={imb['real']} fake={imb['fake']} "
          f"fake_frac={imb['frac_fake']:.4f} pos_weight={imb['pos_weight_value']:.3f}")

    # -------------------------------
    # 6. Optional warm-start checkpoint
    # -------------------------------
    init_ckpt = (cfg.get("train", {}) or {}).get("init_ckpt", None)
    if init_ckpt and os.path.isfile(init_ckpt):
        try:
            print(f"[INIT] Loading pretrained weights: {init_ckpt}")
            payload = torch.load(init_ckpt, map_location="cpu")
            state = payload.get("model_state") or payload.get("state_dict") or payload
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = net.load_state_dict(state, strict=False)
            print(f"[INIT] Model loaded (missing={len(missing)}, unexpected={len(unexpected)})")

            if ema and "ema_state" in payload:
                ema_state = payload["ema_state"]
                if any(k.startswith("module.") for k in ema_state.keys()):
                    ema_state = {k.replace("module.", "", 1): v for k, v in ema_state.items()}
                ema.load_state_dict(ema_state, strict=False)
                print("[INIT] EMA state loaded.")
        except Exception as e:
            print(f"[INIT][WARN] Warm-start failed: {e}")
    else:
        print("[INIT] No warm-start (train.init_ckpt not set or file missing).")

    # -------------------------------
    # 7. Launch training
    # -------------------------------
    trainer = Trainer(
        cfg=cfg,
        device=device,
        feats=feats,
        net=net,
        asn_crit=asn_crit,
        ema=ema,
        class_info=priors,
        use_wandb=use_wandb,
    )
    trainer.run(dl_tr, dl_va, dl_cal)


# =============================================================
if __name__ == "__main__":
    main()
# =============================================================