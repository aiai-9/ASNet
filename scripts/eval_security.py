"""
audioShieldNet/asnet_6/scripts/eval_security.py

Generalized one-shot evaluator for AudioShieldNet models.

Selector (from YAML):
  eval.use_test: true  → evaluate on TEST (auto-prepared if needed)
  eval.use_test: false → evaluate on VAL (no silent TEST fallback)

Other features:
- Resolves loaders via audioshieldnet.data.loader_dispatch (aliases & multi-source).
- When use_test=True: build_testloader→(auto-prepare via build_dataloaders)→retry→error if still missing.
- When use_test=False: build_dataloaders to obtain VAL; error if VAL missing.
- Snapshots audioshieldnet/engine/evaluator.py for repro.
- Writes report.json + figures and prints concise console summary.
- Appends a one-line summary to results/result_summary.csv.
- NEW: if eval.ckpt_choose == "auto" and --ckpt is NOT given, evaluates
  all of: best, last, topk1, topk2, topk3 (skipping missing ones).

CLI:
  python audioShieldNet/asnet_6/scripts/eval_security.py \
      --config audioShieldNet/asnet_6/configs/asn_librisevoc_split.yaml \
      --ckpt   audioShieldNet/asnet_6/experiments/lsv_tcndeep_1/checkpoints/best.ckpt \
      --device cuda
"""

from __future__ import annotations
import os
import sys
import json
import yaml
import shutil
import argparse
import warnings
from typing import Any, Dict, Optional, List, Tuple

import torch

# Ensure audioshieldnet package is importable when running as a script
PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

# Project imports
from audioshieldnet.models.asn import build_model
from audioshieldnet.utils.config_utils import resolve_placeholders
from audioshieldnet.utils.seed import fix_seed
from audioshieldnet.utils.cudnn import tune_cudnn
from audioshieldnet.data.loader_dispatch import resolve_data_builders
from audioshieldnet.engine.evaluator import load_checkpoint_into_model, evaluate_on_loader
from audioshieldnet.utils.checkpoints import resolve_checkpoint_by_choice


# -----------------------------
# Utilities
# -----------------------------
def _quiet_third_party_warnings() -> None:
    """Suppress noisy dependency warnings for a cleaner eval log."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
    warnings.filterwarnings("ignore", message="StreamingMediaDecoder has been deprecated")
    # Some envs spam pydantic field warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", category=Warning, message="The 'repr' attribute")
    warnings.filterwarnings("ignore", category=Warning, message="The 'frozen' attribute")


def _snapshot_evaluator_only(outdir: str) -> None:
    """
    Save exactly one file: audioshieldnet/engine/evaluator.py into <outdir>/checkpoints.
    Keeps eval artifacts reproducible without duplicating other sources.
    """
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    src = os.path.join(PKG_ROOT, "audioshieldnet", "engine", "evaluator.py")
    dst = os.path.join(ckpt_dir, "evaluator.py")
    try:
        shutil.copy2(src, dst)
        print(f"[SNAPSHOT] Saved evaluator.py → {dst}")
    except Exception as e:
        print(f"[SNAPSHOT][WARN] Could not copy evaluator.py: {e}")


def _pick_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fmt_num(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[config] Not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"[config] YAML did not parse to a dict: {path}")
    return cfg


def _resolve_output_dirs(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns:
      - outdir:       experiment root, e.g. audioShieldNet/asnet_6/experiments/lsv_2
      - ckpt_dir:     outdir/checkpoints
      - results_root: outdir/results      (all eval runs go here)
    """
    outdir = ((cfg.get("log") or {}).get("outdir"))
    if not outdir:
        raise RuntimeError("[config] log.outdir is missing; cannot place eval outputs.")

    ckpt_dir = os.path.join(outdir, "checkpoints")
    results_root = os.path.join(outdir, "results")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    return {"outdir": outdir, "ckpt_dir": ckpt_dir, "results_root": results_root}


# -----------------------------
# Loader Selection (TEST vs VAL)
# -----------------------------
def _build_test_loader_strict(
    cfg: Dict[str, Any],
    build_dataloaders,
    build_testloader,
):
    """
    Strict TEST path:
      1) Try build_testloader(cfg)
      2) If None, call build_dataloaders(cfg) to let modules auto-populate test_csv, then RETRY build_testloader(cfg)
      3) If still None → raise with guidance
    """
    dl_te, dl_cal = build_testloader(cfg)
    if dl_te is not None:
        print("[eval][INFO] Using TEST loader (module provided).")
        return dl_te, dl_cal

    # Trigger auto-prepare side-effects (e.g., FOR writes data.test_csv)
    try:
        _dl_tr, _dl_va, _dl_cal2 = build_dataloaders(cfg)
        dl_te2, dl_cal2 = build_testloader(cfg)
        if dl_te2 is not None:
            print("[eval][INFO] TEST loader became available after build_dataloaders() (auto-prepare).")
            return dl_te2, (dl_cal2 or dl_cal or _dl_cal2)
    except Exception as e:
        print(f"[eval][WARN] build_dataloaders() did not yield a TEST loader yet: {e}")

    raise RuntimeError(
        "[eval] Requested use_test=True but no TEST loader is available.\n"
        "Fix options:\n"
        "  • Provide data.test_csv in YAML, OR\n"
        "  • Ensure your dataset module populates cfg['data']['test_csv'] during build_dataloaders(), OR\n"
        "  • Implement build_testloader(cfg) in your dataset module."
    )


def _build_val_loader_strict(
    cfg: Dict[str, Any],
    build_dataloaders,
):
    """
    Strict VAL path:
      1) Call build_dataloaders(cfg) and take VAL.
      2) If VAL is None → raise with guidance (we DO NOT silently swap to TEST here).
    """
    try:
        dl_tr, dl_va, dl_cal = build_dataloaders(cfg)
        if dl_va is not None:
            print("[eval][INFO] Using VAL loader.")
            return dl_va, dl_cal
    except Exception as e:
        print(f"[eval][WARN] build_dataloaders() failed to produce VAL: {e}")

    raise RuntimeError(
        "[eval] Requested use_test=False but no VAL loader is available.\n"
        "Fix options:\n"
        "  • Provide data.val_csv in YAML (dataset-specific), OR\n"
        "  • Ensure your dataset module creates a VAL split in build_dataloaders(cfg)."
    )


def _choose_eval_loader(
    cfg: Dict[str, Any],
    build_dataloaders,
    build_testloader,
):
    """
    Deterministically choose VAL or TEST based on eval.use_test boolean.
    Returns:
      dl_eval, dl_cal, tag
    Where tag is "test" or "test_val".
    """
    eval_cfg = (cfg.get("eval", {}) or {})
    use_test = bool(eval_cfg.get("use_test", True))
    if use_test:
        dl_te, dl_cal = _build_test_loader_strict(cfg, build_dataloaders, build_testloader)
        return dl_te, dl_cal, "test"
    else:
        dl_va, dl_cal = _build_val_loader_strict(cfg, build_dataloaders)
        return dl_va, dl_cal, "test_val"


def _print_console_summary(report: Dict[str, Any]) -> None:
    clean = report.get("clean", {})
    adv   = report.get("adversarial", {})
    abst  = report.get("abstain_energy", {})

    print("\n== TEST SUMMARY ==")
    print(
        f"AUC={_fmt_num(clean.get('auc'))}  "
        f"EER={_fmt_num(clean.get('eer'))}  "
        f"ECE={_fmt_num(clean.get('ece'))}"
    )
    print(
        f"Robust(AUC)={_fmt_num(adv.get('auc'))}  "
        f"Robust(EER)={_fmt_num(adv.get('eer'))}  "
        f"FNR@TPR95={_fmt_num(adv.get('fnr_at_tpr95'))}"
    )

    abst_frac = abst.get("abstain_frac")
    if isinstance(abst_frac, (int, float)) and abst_frac == abst_frac:
        abst_pct = float(abst_frac) * 100.0
    else:
        abst_pct = float("nan")

    print(
        f"Abstain%={_fmt_num(abst_pct)}  "
        f"AUC_kept={_fmt_num(abst.get('auc_kept'))}  "
        f"EER_kept={_fmt_num(abst.get('eer_kept'))}"
    )


def _append_result_summary_csv(
    results_root: str,
    ckpt_dir_name: str,
    split_dir: str,
    ckpt_label: str,
    ckpt_path: str,
    report: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Append a one-line summary to results/result_summary.csv (compact version).
    Removes ckpt_path/result_dir and rounds floats to 3 decimals.
    Returns the written row dict.
    """
    import datetime
    import csv
    import math

    summary_path = os.path.join(results_root, "result_summary.csv")
    os.makedirs(results_root, exist_ok=True)

    clean   = report.get("clean", {})
    adv     = report.get("adversarial", {})
    abst    = report.get("abstain_energy", {})
    counts  = report.get("counts", {})
    timestamp = report.get("timestamp") or datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    def r3(x):
        try:
            if isinstance(x, (float, int)):
                if math.isnan(float(x)):
                    return ""
            return round(float(x), 3)
        except Exception:
            return ""

    abst_frac = abst.get("abstain_frac")
    abst_pct = (
        r3(float(abst_frac) * 100.0)
        if isinstance(abst_frac, (int, float)) and abst_frac == abst_frac
        else ""
    )

    row = {
        "ckpt_dir_name": ckpt_dir_name,
        "split": split_dir,             # "test" or "val"
        "ckpt_label": ckpt_label,       # best / last / topk2 / override / ...
        "auc": r3(clean.get("auc")),
        "eer": r3(clean.get("eer")),
        "ece": r3(clean.get("ece")),
        "robust_auc": r3(adv.get("auc")),
        "robust_eer": r3(adv.get("eer")),
        "fnr_at_tpr95": r3(adv.get("fnr_at_tpr95")),
        "abstain_pct": abst_pct,
        "auc_kept": r3(abst.get("auc_kept")),
        "eer_kept": r3(abst.get("eer_kept")),
        "N": counts.get("N"),
        "pos": counts.get("pos"),
        "neg": counts.get("neg"),
        # "timestamp": timestamp,
    }

    fieldnames = list(row.keys())
    write_header = not os.path.isfile(summary_path)

    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[SUMMARY] Appended compact row → {summary_path}")
    return row


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generalized evaluator for AudioShieldNet.")
    ap.add_argument(
        "--no_result_summary",
        action="store_true",
        help="Skip writing to results/result_summary.csv (for test_multi runs).",
    )
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--ckpt", default=None, help="Override checkpoint path (optional).")
    ap.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Compute device preference."
    )
    ap.add_argument(
        "--result_tag",
        default=None,
        help=(
            "Optional extra tag for result subdir. "
            "Final path: OUTDIR/results/<test|val>/<ckpt_label>[_<result_tag>]/"
        ),
    )

    args = ap.parse_args()
    _quiet_third_party_warnings()

    # Load + resolve placeholders (e.g., ${ckpt_dir})
    cfg = _load_yaml_config(args.config)
    ckpt_dir_name = cfg.get("ckpt_dir", "run")  # e.g. "lsv_2"
    cfg = resolve_placeholders(cfg, {"ckpt_dir": ckpt_dir_name})

    # Output dirs and snapshot evaluator
    paths = _resolve_output_dirs(cfg)
    outdir       = paths["outdir"]
    ckpt_dir     = paths["ckpt_dir"]
    results_root = paths["results_root"]
    _snapshot_evaluator_only(outdir)

    # Builders (dataset-specific)
    build_dataloaders, build_testloader = resolve_data_builders(cfg)

    # Determinism + CUDA setup
    fix_seed((cfg.get("train", {}) or {}).get("seed", 42))
    tune_cudnn()

    # Choose VAL or TEST strictly as requested
    dl_eval, _dl_cal, tag = _choose_eval_loader(cfg, build_dataloaders, build_testloader)

    # Result split dir (test / val)
    if tag == "test":
        split_dir = "test"
    else:
        split_dir = "val"

    split_root = os.path.join(results_root, split_dir)
    os.makedirs(split_root, exist_ok=True)

    # Build model once; we'll reload different checkpoints into it
    device = _pick_device(args.device)
    feats, net = build_model(cfg, device)
    net.to(device)

    # Criterion for adversarial eval (once)
    from audioshieldnet.losses.classification import build_classification_loss
    crit, _ = build_classification_loss(cfg, device)

    eval_cfg = (cfg.get("eval", {}) or {})
    choice   = str(eval_cfg.get("ckpt_choose", "auto")).lower().strip()

    # ----------------------------------------------------
    # Decide which checkpoints to evaluate in this run
    # ----------------------------------------------------
    ckpt_choices: List[Tuple[str, str]] = []

    if args.ckpt:
        # Explicit override via CLI: single run labeled "override"
        ckpt_choices.append((args.ckpt, "override"))
    else:
        if choice == "auto":
            # NEW BEHAVIOR:
            # Evaluate all of: best, last, topk1, topk2, topk3 (if they exist)
            labels_to_try = ["best", "last", "topk1", "topk2", "topk3"]
            for lab in labels_to_try:
                tmp_eval_cfg = dict(eval_cfg)
                tmp_eval_cfg["ckpt_choose"] = lab
                try:
                    path, label = resolve_checkpoint_by_choice(
                        ckpt_dir=ckpt_dir,
                        override=None,
                        eval_cfg=tmp_eval_cfg,
                    )
                    if (path, label) not in ckpt_choices:
                        ckpt_choices.append((path, label))
                except Exception as e:
                    print(f"[eval][WARN] Could not resolve ckpt for '{lab}': {e}")
            if not ckpt_choices:
                raise FileNotFoundError(
                    f"[eval] ckpt_choose=auto but none of best/last/topk1-3 "
                    f"found in {ckpt_dir}."
                )
        else:
            # Non-auto: just use resolve_checkpoint_by_choice once
            path, label = resolve_checkpoint_by_choice(
                ckpt_dir=ckpt_dir,
                override=None,
                eval_cfg=eval_cfg,
            )
            ckpt_choices.append((path, label))

    # --- Temperature scaling (per-run, shared temperature.json per ckpt_dir) ---
    use_ts       = bool(eval_cfg.get("use_temperature_scaling", True))
    reuse_saved  = bool(eval_cfg.get("reuse_saved_temperature", True))
    temp_path    = os.path.join(ckpt_dir, "temperature.json")

    def _save_temp(T: float):
        try:
            with open(temp_path, "w") as f:
                json.dump({"temperature": float(T)}, f)
            print(f"[CAL] Saved temperature={T:.4f} → {temp_path}")
        except Exception as e:
            print(f"[CAL][WARN] Could not save temperature: {e}")

    def _load_temp() -> Optional[float]:
        if os.path.isfile(temp_path):
            try:
                with open(temp_path, "r") as f:
                    obj = json.load(f)
                T = float(obj.get("temperature", 1.0))
                print(f"[CAL] Loaded saved temperature={T:.4f} from {temp_path}")
                return T
            except Exception as e:
                print(f"[CAL][WARN] Could not load temperature: {e}")
        return None

    # ----------------------------------------------------
    # Loop over all selected checkpoints
    # ----------------------------------------------------
    for ckpt_path, ckpt_label in ckpt_choices:
        print(f"\n[eval] === Evaluating checkpoint ({ckpt_label}): {ckpt_path} ===")
        if not os.path.isfile(ckpt_path):
            print(f"[eval][WARN] Skipping missing checkpoint: {ckpt_path}")
            continue

        # Load weights into model
        _ = load_checkpoint_into_model(ckpt_path, net, device=device)

        # Decide result subdir name for this ckpt
        if args.result_tag:
            rt = args.result_tag.strip()
            result_subdir = f"{ckpt_label}_{rt}"
        else:
            result_subdir = ckpt_label

        # Temperature for this ckpt (start from 1.0 each time)
        temperature = 1.0

        if use_ts and reuse_saved:
            T_saved = _load_temp()
            if T_saved is not None:
                temperature = T_saved

        # If we still have T=1.0 and TS is requested, try to fit on a calibration/VAL loader
        if use_ts and abs(temperature - 1.0) < 1e-6:
            cal_loader = _dl_cal
            if cal_loader is None:
                # Try to fetch VAL from build_dataloaders
                try:
                    _dl_tr2, _dl_va2, _dl_cal2 = build_dataloaders(cfg)
                    cal_loader = _dl_va2 or _dl_cal2
                except Exception:
                    cal_loader = None

            if cal_loader is not None:
                from audioshieldnet.engine.evaluator import _fit_temperature_on_loader

                temperature = _fit_temperature_on_loader(
                    feats,
                    net,
                    cal_loader,
                    device,
                    max_iter=int(eval_cfg.get("temperature_max_iter", 200)),
                    init_T=float(eval_cfg.get("temperature_init", 1.0)),
                )
                _save_temp(temperature)
            else:
                print(
                    "[CAL][WARN] No calibration/VAL loader available; "
                    "proceeding with temperature=1.0"
                )

        # Run full evaluation for this checkpoint:
        #   results saved under:
        #     OUTDIR/results/<test|val>/<ckpt_label>[_<result_tag>]/
        report = evaluate_on_loader(
            cfg,
            feats,
            net,
            dl_eval,
            device,
            split_root,           # root directory for this split
            crit,
            result_subdir=result_subdir,
            temperature=temperature,
        )

        # Persist + print a compact console summary
        print("\n== REPORT.JSON ==")
        print(json.dumps(report, indent=2))
        print("Saved:", os.path.join(report["files"]["result_dir"], "report.json"))

        _print_console_summary(report)

        # Append / create results/result_summary.csv
        if not args.no_result_summary:
            _append_result_summary_csv(
                results_root=results_root,
                ckpt_dir_name=ckpt_dir_name,
                split_dir=split_dir,
                ckpt_label=ckpt_label,
                ckpt_path=ckpt_path,
                report=report,
            )


if __name__ == "__main__":
    main()
