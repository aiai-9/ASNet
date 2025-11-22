# audioShieldNet/asnet_1/audioshieldnet/engine/trainer.py

import os, json, torch, random, uuid, warnings
import torch.nn as nn
os.environ.pop("WANDB_START_METHOD", None)



import numpy as np
import sklearn.metrics as skm
from contextlib import nullcontext
from tqdm import tqdm
import wandb

# =========================
# 🔒 Security & Evaluation
# =========================
from audioshieldnet.security.attacks import fgsm_attack, pgd_attack
from audioshieldnet.security.ood import energy_score
from audioshieldnet.security.sanitize import mp3_roundtrip
from audioshieldnet.security.plots import (
    log_val_confusion_metrics,
    log_curriculum_schedule,
)
from audioshieldnet.security.conformal import fit_tau_conformal, auto_tau_from_val_energies
from audioshieldnet.security.pseudo_ood import PseudoOODSampler

# =========================
# ⚙️ Core Utilities
# =========================
from audioshieldnet.utils.augs import add_white_noise, spec_augment
from audioshieldnet.utils.metrics import expected_calibration_error
from audioshieldnet.utils.risk_coverage import (
    risk_coverage_from_energy, ece_on_subset, wandb_log_risk_coverage
)
from audioshieldnet.utils.wand_setup import (
    to_serializable, wandb_log_safe, save_npz_artifact, init_wandb
)
from audioshieldnet.utils.checkpoints import save_checkpoint, try_resume, metric_better, prune_topk
from audioshieldnet.utils.opt_sched import (
    build_optimizer, build_scheduler,
    swa_is_active, swa_make_model, swa_should_update, swa_update_bn,
    schedule_adv_eps
)
from audioshieldnet.utils.curriculum import resolve_curriculum

# =========================
# 🧠 Engine / Loss
# =========================
from audioshieldnet.engine.evaluator import (
    collect_probs, adversarial_eval, suspicious_fraction, auc_eer
)
from audioshieldnet.losses.classification import build_classification_loss

# Silence harmless torch scheduler warning
warnings.filterwarnings(
    "ignore",
    message=".*epoch parameter in `scheduler.step.*",
    category=UserWarning,
    module="torch.optim.lr_scheduler",
)

# === Helper: set BN layers to train/eval mode ===

def set_bn_train_mode(m, train: bool):
    for mod in m.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.train(train)
   
# --- Stronger determinism (helps stabilize AUC/EER between runs) ---
def _set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_set_global_seed(int(os.environ.get("ASNET_SEED", "42")))

# --- NaN guard (don’t let a single bad batch nuke metrics) ---
def _safe(t):
    if isinstance(t, torch.Tensor):
        t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
    return t


class Trainer:
    def __init__(self, cfg, device, feats, net, asn_crit, ema, class_info, use_wandb=False):
        self.cfg, self.device = cfg, device
        self.feats, self.net, self.asn_crit, self.ema = feats, net, asn_crit, ema
        self.cons_weight = float(cfg.get('asn', {}).get('cons_weight', 0.2))
        self.use_amp = bool(cfg['train'].get('amp', True))
        if self.use_amp and torch.cuda.is_available():
            # New API keeps mixed precision stable; avoids the old deprecation path
            self.amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        else:
            self.amp_ctx = nullcontext()


        # Security/aug configs
        self.sec = cfg.get('security', {}) or {}
        self.aug = cfg.get('augs', {}) or {}

        # Energy triage/regularization
        self.T_energy = float(self.sec.get('triage', {}).get('T', 1.0))
        self.tau_susp = float(self.sec.get('triage', {}).get('tau_susp_energy', -0.2))
        self.use_Ereg = bool(self.sec.get('energy_reg', True))

        # Anchored energy (calibration-preserving)
        Ecfg = (self.sec.get('energy', {}) or {})
        self.energy_anchor_tau = float(Ecfg.get('tau_id', -0.10))      # ID energy anchor target
        self.energy_anchor_lambda = float(Ecfg.get('lambda_id', 1e-5)) # small, stable weight
        self.energy_anchor_warmup = int(Ecfg.get('warmup_epochs', 0))  # optional delay

        # OOD push (gentle hinge)
        self.ood_cfg = (self.sec.get('ood_push', {}) or {})
        self.ood_tau = float(self.ood_cfg.get('tau_target', 0.10))
        self.ood_lambda = float(self.ood_cfg.get('weight', 1.0e-5))

        # Adversarial cadence
        self.adv_every = int(self.sec.get('adv_every', 0))

        # Augs
        self.add_noise_p = float(self.aug.get('add_noise_p', 0.3))
        self.snr_lo, self.snr_hi = self.aug.get('snr_db', [10, 20])
        self.use_specaug = bool(self.aug.get('use_specaug', True))
        self.Tmask = int(self.aug.get('time_mask_T', 30))
        self.Pt = float(self.aug.get('time_mask_p', 0.3))
        self.Fmask = int(self.aug.get('freq_mask_F', 8))
        self.Pf = float(self.aug.get('freq_mask_p', 0.3))

        # --- score polarity state (auto-calibrated) ---
        self._polarity_fixed = None   # None | True (invert) | False (as-is)
        self._polarity_buffer = []    # rolling last-K decisions to stabilize
        self._polarity_K = 3          # require K consecutive same decisions



        # Pseudo-OOD sampler (multi-type, curriculum-ready)
        self.pseudo_ood = None
        try:
            sr_from_cfg = int(cfg.get('data', {}).get('sr', 16000))
        except Exception:
            sr_from_cfg = 16000
        if bool(self.ood_cfg.get('use', False)):
            enabled = self.ood_cfg.get('types', None)
            curriculum = bool(self.ood_cfg.get('curriculum', True))
            self.pseudo_ood = PseudoOODSampler(sr=sr_from_cfg, enabled_types=enabled, curriculum=curriculum)

        # Optimizers, SWA, loss
        self.opt, self.use_sam = build_optimizer(cfg, self.net)
        self.use_swa = swa_is_active(cfg)
        self.swa_model = swa_make_model(self.net) if self.use_swa else None
        # ---- Provide counts and polarity where the loss factory actually reads them ----
        # Robust handling even if class_info=None or missing keys
        if class_info is None:
            class_info = {}
        elif not isinstance(class_info, dict):
            try:
                class_info = {
                    "num_real": int(getattr(class_info, "num_real", 0)),
                    "num_fake": int(getattr(class_info, "num_fake", 0)),
                }
            except Exception:
                class_info = {}

        # fallback to cfg if prior counts exist
        prior_counts = (cfg.get("data", {}) or {}).get("prior_counts", {})
        n_real = int(class_info.get("num_real", prior_counts.get("real", 0)))
        n_fake = int(class_info.get("num_fake", prior_counts.get("fake", 0)))
        prior_fake = n_fake / max(1, (n_real + n_fake))

        # save into cfg for downstream losses
        self.cfg.setdefault("data", {}).setdefault("prior_counts", {"real": n_real, "fake": n_fake})

        print(f"[LOSS] prior(fake)={prior_fake:.4f}  (real={n_real}, fake={n_fake})")


        # (B) set train.pos_class so internal losses know which label is "positive"
        #     1 => fake positive (default), 0 => real positive
        loss_root = self.cfg.setdefault("train", {}).setdefault("loss", {})
        pos_label = int(loss_root.get("pos_label", 1))
        self.cfg["train"]["pos_class"] = "fake" if pos_label == 1 else "real"

        print(f"[LOSS] prior(fake)={prior_fake:.4f}  pos_label={pos_label}  pos_class={self.cfg['train']['pos_class']}")

        # Now build the loss (balanced_bce/focal/asl/logit_adjusted) with the correct cfg paths
        self.criterion, self.loss_info = build_classification_loss(self.cfg, self.device)
        print(f"[LOSS] Using {self.loss_info.name} with {self.loss_info.details}")



        # Dirs
        self.ckpt_dir = os.path.join(cfg['log']['outdir'], "checkpoints")
        self.metrics_dir = os.path.join(cfg['log']['outdir'], "metrics")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ckpt_dir, "topk"), exist_ok=True)

        # W&B
        self.wandb = None
        if use_wandb:
            wcfg = cfg['log'].get('wandb', {})
            outdir = cfg['log']['outdir']
            os.makedirs(outdir, exist_ok=True)

            # Stable run-id
            run_id_file = os.path.join(outdir, ".wandb_run_id")
            if os.path.isfile(run_id_file):
                run_id = open(run_id_file, "r").read().strip()
            else:
                run_id = str(uuid.uuid4())
                with open(run_id_file, "w") as f: f.write(run_id)

            settings = wandb.Settings(init_timeout=300, code_dir=None)

            try:
                self.wandb = wandb.init(
                    project=wcfg.get('project', 'audioshieldnet'),
                    entity=wcfg.get('entity', None),
                    name=wcfg.get('run_name', None),
                    config=cfg,
                    dir=outdir,
                    id=run_id,
                    resume="allow",
                    settings=settings,
                )
            except Exception as e:
                os.environ["WANDB_MODE"] = "offline"
                print(f"[W&B] Online init failed ({e}). Falling back to OFFLINE mode.")
                cfg_serializable = to_serializable(cfg)
                self.wandb = wandb.init(
                    project=wcfg.get('project', 'audioshieldnet'),
                    entity=wcfg.get('entity', None),
                    name=wcfg.get('run_name', None),
                    config=cfg_serializable,
                    dir=outdir,
                    id=run_id,
                    resume="allow",
                    settings=settings,
                )

            try:
                wandb.log_code(
                    root=os.getcwd(),
                    include_fn=lambda p: any(
                        p.startswith(x) for x in [
                            "audioShieldNet/asnet_1/audioshieldnet",
                            "audioShieldNet/asnet_1/scripts",
                            "audioShieldNet/asnet_1/configs",
                        ]
                    )
                )
            except Exception:
                pass

            wandb.watch(self.net, log=None)

            cfg_art = wandb.Artifact("train_config", type="config")
            try:
                cfg_path = wcfg.get('config_path', None)
                if cfg_path and os.path.isfile(cfg_path):
                    cfg_art.add_file(cfg_path, name="config.yaml")
                else:
                    tmp_cfg = os.path.join(outdir, "config_dump.json")
                    with open(tmp_cfg, "w") as f: json.dump(cfg, f, indent=2)
                    cfg_art.add_file(tmp_cfg, name="config.json")
            except Exception:
                pass
            self.wandb.log_artifact(cfg_art, aliases=["latest"])

        # SAM switch tracking
        self._sam_prev = False
        self._sam_just_switched = False
        
    def _choose_polarity(self, auc_p: float, auc_ip: float) -> bool:
        inv_cfg = (self.cfg.get('eval', {}) or {}).get('force_invert', None)
        if inv_cfg is True:  return True
        if inv_cfg is False: return False

        # if already stabilized
        if self._polarity_fixed is not None:
            return bool(self._polarity_fixed)

        # margin helps avoid jitter around 0.5 AUC
        invert_now = bool(auc_ip > auc_p + 1e-3)

        self._polarity_buffer.append(invert_now)
        if len(self._polarity_buffer) > self._polarity_K:
            self._polarity_buffer.pop(0)

        # Require K consecutive same decisions to fix polarity
        if len(self._polarity_buffer) == self._polarity_K:
            if all(self._polarity_buffer):
                self._polarity_fixed = True
                print(f"[Polarity] Fixed → INVERT after {self._polarity_K} stable wins (AUC_ip={auc_ip:.3f} > AUC_p={auc_p:.3f})")
            elif not any(self._polarity_buffer):
                self._polarity_fixed = False
                print(f"[Polarity] Fixed → AS-IS after {self._polarity_K} stable wins (AUC_p={auc_p:.3f} ≥ AUC_ip={auc_ip:.3f})")

        return invert_now


    def run(self, dl_tr, dl_va, dl_cal=None):
        cfg = self.cfg
        best_metric_name = str(cfg['train'].get('best_metric', 'auc')).lower()
        best_mode = str(cfg['train'].get('best_mode', 'max')).lower()
        if best_metric_name in ['eer','ece','fnr95_fgsm','susp_frac']: best_mode = 'min'
        keep_top_k = int(cfg['train'].get('keep_top_k', 3))
        save_every_epochs = int(cfg['train'].get('save_every_epochs', 1))
        steps_per_epoch = cfg['train'].get('steps_per_epoch', None)
        total_epochs = int(cfg['train']['epochs'])
        global_step, best_value = 0, None
        last_path = os.path.join(self.ckpt_dir, "last.ckpt")

        # Curriculum gates
        gates = resolve_curriculum(self.cfg["train"], total_epochs)
        # ---- Once, at the top of run() after resolve_curriculum ----
        use_sam     = bool(self.cfg.get("optim", {}).get("use_sam", False))
        use_energy  = bool(self.cfg.get("security", {}).get("energy_reg", False))
        use_ood     = bool((self.cfg.get("security", {}).get("ood_push", {}) or {}).get("use", False))
        use_adv     = bool(self.cfg.get("security", {}).get("use_adv", False))

        sam_start_epoch     = gates["sam_start_epoch"]     if use_sam    else float("inf")
        energy_start_epoch  = gates["energy_start_epoch"]  if use_energy else float("inf")
        ood_start_epoch     = gates["ood_start_epoch"]     if use_ood    else float("inf")
        adv_start_epoch     = gates["adv_start_epoch"]     if use_adv    else float("inf")
        stability_epochs    = gates["stability_warmup_epochs"]


        # SAM switch behaviors
        self.lr_drop_on_switch = float(self.cfg['train'].get('lr_drop_on_switch', 0.5))

        print(  f"Curriculum Gates → total_epochs={total_epochs} | "
                f"SAM={'∞' if not use_sam else sam_start_epoch}, "
                f"ENERGY={'∞' if not use_energy else energy_start_epoch}, "
                f"OOD={'∞' if not use_ood else ood_start_epoch}, "
                f"ADV={'∞' if not use_adv else adv_start_epoch}, "
                f"STABILITY={stability_epochs}"
            )

        start_epoch, global_step, best_value = try_resume(last_path, self.net, self.opt, self.ema, self.device)
        if start_epoch > 0:
            print(f"[RESUME] {last_path} → epoch={start_epoch}, global_step={global_step}, best={best_value}")

        try:
            # print("\n")
            with tqdm(total=total_epochs - start_epoch, desc="Epochs", unit="epoch", leave=True) as tepochs:
                total_steps = steps_per_epoch if steps_per_epoch is not None else len(dl_tr)
                self.scheduler, self.sched_per_batch = build_scheduler(cfg, self.opt, total_steps, total_epochs)

                # Optional conformal τ from calibration split
                tau_conformal = None
                if self.sec.get('triage', {}).get('use_conformal', False) and (dl_cal is not None):
                    calE = []
                    with torch.no_grad():
                        for wav, *_ in dl_cal:
                            lm, pm = self.feats(wav.to(self.device, non_blocking=True))
                            logits, _ = self.net(lm, pm, target=None)
                            calE.append(energy_score(logits, T=self.T_energy).cpu().numpy())
                    calE = np.concatenate(calE) if calE else np.array([])
                    if calE.size:
                        tau_conformal = fit_tau_conformal(
                            calE,
                            target_coverage=float(self.sec['triage'].get('target_coverage', 0.90))
                        )
                        # Optionally freeze: self.tau_susp = float(tau_conformal)

                for epoch in range(start_epoch, total_epochs):
                    # print("\n")
                    # ---- TRAIN ----
                    self.net.train()
                    tot = 0.0

       
                    # curriculum gates
                    # If a defense is disabled in cfg, push its gate to ∞ so it never triggers
                    if not use_sam:    sam_start_epoch    = float("inf")
                    if not use_energy: energy_start_epoch = float("inf")
                    if not use_ood:    ood_start_epoch    = float("inf")
                    if not use_adv:    adv_start_epoch    = float("inf")

                    # ---- DEFINE PER-EPOCH ENABLE FLAGS (used below) ----
                    sam_enabled    = (use_sam    and (epoch >= sam_start_epoch))
                    energy_enabled = (use_energy and (epoch >= energy_start_epoch))
                    ood_enabled    = (use_ood    and (epoch >= ood_start_epoch))
                    adv_enabled    = (use_adv    and (epoch >= adv_start_epoch))   # <-- correct name

                    # SAM switch handling (BN freeze + LR nudge)
                    self._sam_just_switched = (sam_enabled and not self._sam_prev)
                    if self._sam_just_switched and self.lr_drop_on_switch > 0:
                        for g in self.opt.param_groups:
                            g["lr"] = float(g["lr"]) * float(self.lr_drop_on_switch)
                        set_bn_train_mode(self.net, False)  # freeze BN this epoch
                    self._sam_prev = sam_enabled


                    with tqdm(total=total_steps, desc=f"Epoch {epoch}/{total_epochs} [train]", unit="batch", leave=False) as pbar:
                        for batch in dl_tr:
                            # support (wav, yb, *_) or (wav, yb)
                            if len(batch) >= 2:
                                wav, yb = batch[0], batch[1]
                            else:
                                raise RuntimeError("Train loader must yield at least (wav, y).")

                            global_step += 1
                            wav = wav.to(self.device, non_blocking=True)
                            yb  = yb.to(self.device, non_blocking=True)

                            if random.random() < self.add_noise_p:
                                snr = random.uniform(self.snr_lo, self.snr_hi)
                                wav = add_white_noise(wav, snr)

                            logmel, phmel = self.feats(wav)
                            if self.use_specaug:
                                if yb.float().mean() < 0.5:
                                    logmel = spec_augment(logmel, T=self.Tmask + 10, p_t=min(1.0, self.Pt + 0.1),
                                                          F=self.Fmask, p_f=self.Pf)
                                else:
                                    logmel = spec_augment(logmel, T=self.Tmask, p_t=self.Pt,
                                                          F=self.Fmask, p_f=self.Pf)
                            if self.use_specaug:
                                phmel = spec_augment(phmel, T=self.Tmask, p_t=self.Pt, F=self.Fmask, p_f=self.Pf)


                            with self.amp_ctx:
                                logits, _ = self.net(logmel, phmel, target=None)
                                bce_loss = self.criterion(logits, yb, epoch=epoch)

                            # adversarial cadence
                            adv_enabled_now = adv_enabled   # already computed above with the correct gate
                            do_adv = bool(adv_enabled_now and self.adv_every > 0 and (global_step % self.adv_every == 0))


                            bce_adv = torch.tensor(0.0, device=self.device)
                            wav_adv = None
                            if do_adv:
                                # total_train_steps = (steps_per_epoch or len(dl_tr)) * total_epochs
                                remaining_epochs = (total_epochs - start_epoch)
                                total_train_steps = (steps_per_epoch or len(dl_tr)) * remaining_epochs
                                eps_use = schedule_adv_eps(self.cfg, global_step, total_train_steps)
                                alpha = float(self.sec.get('adv_alpha', 0.0)) or (eps_use / 4.0)
                                attack_loss = lambda lg, tgt: self.criterion(lg, tgt, epoch=epoch)

                                if int(self.sec.get('adv_steps', 1)) <= 1:
                                    wav_adv = fgsm_attack(wav, yb, self.net, self.feats, attack_loss, eps=eps_use)
                                else:
                                    wav_adv = pgd_attack(
                                        wav, yb, self.net, self.feats, attack_loss,
                                        eps=eps_use, alpha=alpha, steps=int(self.sec.get('adv_steps', 1))
                                    )

                                with self.amp_ctx:
                                    logmel_adv, phmel_adv = self.feats(wav_adv)
                                    logits_adv, _ = self.net(logmel_adv, phmel_adv, target=None)
                                    bce_adv = self.criterion(logits_adv, yb, epoch=epoch)

                            A_map = logmel.unsqueeze(1).float()
                            P_map = phmel.unsqueeze(1).float()
                            asn_loss, coh = self.asn_crit(A_map, P_map, yb)

                            # Anchored energy (ID calibration)
                            E_anchor = torch.tensor(0.0, device=self.device)
                            if energy_enabled and (epoch >= self.energy_anchor_warmup):
                                E_id = energy_score(logits, T=self.T_energy)  # [B]
                                E_anchor = self.energy_anchor_lambda * (E_id - self.energy_anchor_tau).pow(2).mean()

                            # OOD hinge (push energies upward for pseudo-OOD)
                            ood_loss = torch.tensor(0.0, device=self.device)
                            if ood_enabled:
                                # training OOD branch
                                with torch.no_grad():
                                    if self.pseudo_ood is not None:
                                        wav_ood, ood_chosen = self.pseudo_ood(wav, step=global_step)
                                    else:
                                        wav_ood, ood_chosen = mp3_roundtrip(wav), "mp3"
                                    wav_ood = wav_ood.to(self.device, non_blocking=True)  # <-- add this

                                with self.amp_ctx:
                                    logmel_o, phmel_o = self.feats(wav_ood)
                                    logits_o, _ = self.net(logmel_o, phmel_o, target=None)
                                    E_ood = energy_score(logits_o, T=self.T_energy)

                                ood_loss = self.ood_lambda * torch.relu(self.ood_tau - E_ood).mean()

                                if self.wandb and (global_step % 200 == 0):
                                    try:
                                        self.wandb.log({"train/ood_type": str(ood_chosen)}, commit=False)
                                    except Exception:
                                        pass

                            total_loss = (
                                bce_loss + bce_adv
                                + self.cons_weight * asn_loss
                                + E_anchor
                                + ood_loss
                            )

                            if sam_enabled:
                                # SAM in fp32 to avoid AMP jitter
                                if hasattr(torch.cuda, "amp"):
                                    autocast_off = torch.amp.autocast("cuda", enabled=False)
                                else:
                                    from contextlib import nullcontext as autocast_off
                                with autocast_off:
                                    self.opt.zero_grad(set_to_none=True)
                                    # total_loss.backward()
                                    _safe(total_loss).backward()
                                    self.opt.first_step(zero_grad=True)

                                    logits2, _ = self.net(logmel, phmel, target=None)
                                    bce2 = self.criterion(logits2, yb, epoch=epoch)
                                    if do_adv and (wav_adv is not None):
                                        logmel_adv2, phmel_adv2 = self.feats(wav_adv)
                                        logits_adv2, _ = self.net(logmel_adv2, phmel_adv2, target=None)
                                        bce_adv2 = self.criterion(logits_adv2, yb, epoch=epoch)
                                    else:
                                        bce_adv2 = torch.tensor(0.0, device=self.device)

                                    asn2, _ = self.asn_crit(A_map, P_map, yb)
                                    if energy_enabled and (epoch >= self.energy_anchor_warmup):
                                        E_id2 = energy_score(logits2, T=self.T_energy)
                                        E_anchor2 = self.energy_anchor_lambda * (E_id2 - self.energy_anchor_tau).pow(2).mean()
                                    else:
                                        E_anchor2 = torch.tensor(0.0, device=self.device)

                                    loss2 = bce2 + bce_adv2 + self.cons_weight * asn2 + E_anchor2
                                    _safe(loss2).backward()
                                    gc = float(self.cfg['train'].get('grad_clip', 1.0))
                                    if gc and gc > 0:
                                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), gc)
                                    self.opt.second_step(zero_grad=True)


                                    loss_for_log = 0.5 * (total_loss.detach() + loss2.detach())
                            else:
                                self.opt.zero_grad(set_to_none=True)
                                _safe(total_loss).backward()

                                gc = float(self.cfg['train'].get('grad_clip', 1.0))  # default to 1.0 if unset
                                if gc and gc > 0:
                                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), gc)

                                base_opt = getattr(self.opt, "base_optimizer", None)
                                if base_opt is None:
                                    self.opt.step()
                                else:
                                    base_opt.step()


                                loss_for_log = total_loss.detach()

                            if self.ema is not None:
                                self.ema.update(self.net)

                            if self.scheduler is not None and self.sched_per_batch:
                                self.scheduler.step()

                            if self.wandb:
                                for i, g in enumerate(self.opt.param_groups):
                                    try:
                                        self.wandb.log({f"lr/group{i}": float(g["lr"])}, commit=False)
                                    except Exception:
                                        pass

                            grad_norm = None
                            if (pbar.n % 100 == 0):
                                total_sq = 0.0
                                for p in self.net.parameters():
                                    if p.grad is not None:
                                        g = p.grad.detach()
                                        total_sq += float(g.pow(2).sum().item())
                                grad_norm = (total_sq ** 0.5) if total_sq > 0 else 0.0

                            if self.wandb:
                                bce_only = float(bce_loss.item())
                                if do_adv and (bce_adv is not None):
                                    bce_only += float(bce_adv.item())

                                log_payload = {
                                    "train/loss_batch": float(loss_for_log.item()),
                                    "train/loss_total_raw": float(total_loss.item()),
                                    "train/bce_only_batch": bce_only,
                                    "train/asn_batch": float(asn_loss.item()),
                                    "train/coh_batch": float(coh.mean().item()),
                                    "train/adv_enabled": float(do_adv),
                                }
                                if grad_norm is not None:
                                    log_payload["train/grad_norm"] = grad_norm
                                self.wandb.log(log_payload, commit=False)

                            tot += float(loss_for_log.item())
                            pbar.set_postfix(loss=f"{tot / (pbar.n + 1):.4f}",
                                            bce=f"{bce_loss.item():.3f}",
                                            asn=f"{asn_loss.item():.3f}",
                                            coh=f"{coh.mean().item():.3f}")
                            pbar.update(1)
                            if steps_per_epoch is not None and (pbar.n >= steps_per_epoch):
                                break

                    epoch_loss = tot / max(1, len(dl_tr))

                    # ---- VAL ----
                    self.net.eval()
                    if self.ema is not None:
                        self.ema.apply_to(self.net)

                    # 1) Clean predictions + base metrics (consistent inversion)
                    # REVISED (trainer.py — inside Trainer.run(), validation section)
                    inv_cfg = (self.cfg.get('eval', {}) or {}).get('force_invert', None)

                    # Get both (these may be logits depending on your evaluator)
                    auc_p,  eer_p,  (yt_p,  yp_p)  = auc_eer(self.net, self.feats, dl_va, self.device, invert=False)
                    auc_ip, eer_ip, (yt_ip, yp_ip) = auc_eer(self.net, self.feats, dl_va, self.device, invert=True)

                    # Stable polarity decision (or honor cfg)
                    invert = self._choose_polarity(auc_p, auc_ip) if inv_cfg is None else bool(inv_cfg)
                    y_true = (yt_ip if invert else yt_p)
                    y_prob = (yp_ip if invert else yp_p)

                    # → Convert to numpy and clamp/sigmoid if they look like logits
                    if hasattr(y_prob, "detach"): y_prob = y_prob.detach().cpu().numpy()
                    if hasattr(y_true, "detach"): y_true = y_true.detach().cpu().numpy()
                    y_true = np.asarray(y_true).astype(int)
                    y_prob = np.asarray(y_prob).astype(float)

                    if (y_prob.min() < 0.0) or (y_prob.max() > 1.0):
                        y_prob = 1.0 / (1.0 + np.exp(-np.clip(y_prob, -20, 20)))  # stable sigmoid
                    
                    # --- NEW: quick sanity plots for probability distribution & class balance ---
                    if self.wandb and y_prob.size:
                        try:
                            self.wandb.log({
                                "val/y_prob_hist": wandb.Histogram(y_prob.tolist()),
                                "val/y_true_ratio_fake": float((y_true == 1).mean())
                            }, commit=False)
                        except Exception:
                            pass


                    # **Authoritative** metrics recomputed from the normalized probabilities
                    if np.unique(y_true).size > 1:
                        fpr, tpr, thr = skm.roc_curve(y_true.astype(int), y_prob.astype(float))
                        fnr = 1.0 - tpr
                        idx = np.nanargmin(np.abs(fnr - fpr))
                        eer = float((fpr[idx] + fnr[idx]) / 2.0) if fpr.size else float("nan")
                        auc = skm.roc_auc_score(y_true.astype(int), y_prob.astype(float))
                    else:
                        eer, auc = float("nan"), float("nan")




                    # Ensure all downstream metrics (ECE, energy/triage, CSV) use y_prob above
                    if self.wandb:
                        self.wandb.log({
                            "val/auc_as_is":      float(auc_p),
                            "val/auc_inverted":   float(auc_ip),
                            "val/polarity_invert": float(invert),
                            "val/polarity_fixed":  float(self._polarity_fixed is not None),
                        }, commit=False)


                    # Clean BCE on val (to track optimization quality)
                    with torch.no_grad():
                        val_bce_accum, n_val = 0.0, 0
                        for batch in dl_va:
                            if len(batch) >= 2:
                                wav, yb = batch[0], batch[1]
                            else:
                                raise RuntimeError("Val loader must yield at least (wav, y).")
                            wav = wav.to(self.device, non_blocking=True)
                            yb  = yb.to(self.device, non_blocking=True)
                            logmel, phmel = self.feats(wav)
                            logits, _ = self.net(logmel, phmel, target=None)
                            bce_per = self.criterion(logits, yb, epoch=None)  # mean over batch
                            bs = yb.numel()
                            val_bce_accum += float(bce_per.item()) * bs
                            n_val += bs
                        val_bce = val_bce_accum / max(1, n_val)


                    # 2) Core calibration
                    p_safe = np.clip(y_prob, 1e-5, 1.0 - 1e-5)
                    ece = expected_calibration_error(y_true, p_safe)

                    # 3) Adversarial robustness
                    with torch.no_grad():
                        rob_auc, rob_eer, fnr95 = adversarial_eval(
                            self.feats, self.net, dl_va, self.device, self.criterion, self.sec
                        )

                    # 4) Energy-based OOD (pseudo-OOD via codec) + AUROC
                    with torch.no_grad():
                        id_E, ood_E = [], []
                        for batch in dl_va:
                            wav = batch[0].to(self.device, non_blocking=True)
                            lm, pm = self.feats(wav)
                            logits, _ = self.net(lm, pm, target=None)
                            id_E.append(energy_score(logits, T=self.T_energy).cpu().numpy())

                            if bool(self.ood_cfg.get('use', False)):
                                wav_ood = mp3_roundtrip(wav)
                                wav_ood = wav_ood.to(self.device, non_blocking=True)
                                lm_o, pm_o = self.feats(wav_ood)
                                logits_o, _ = self.net(lm_o, pm_o, target=None)
                                ood_E.append(energy_score(logits_o, T=self.T_energy).cpu().numpy())

                        id_E = np.concatenate(id_E) if len(id_E) else np.array([])
                        ood_E = np.concatenate(ood_E) if len(ood_E) else np.array([])

                    ood_auroc = float("nan")
                    if id_E.size and ood_E.size and not np.allclose(id_E, ood_E):
                        y_mix = np.concatenate([np.zeros_like(id_E), np.ones_like(ood_E)])
                        s_mix = np.concatenate([id_E, ood_E])  # higher ⇒ more OOD
                        ood_auroc = skm.roc_auc_score(y_mix, s_mix)

                    # Per-type OOD AUROC (optional)
                    per_type_auroc = {}
                    if bool(self.ood_cfg.get('use', False)) and (self.pseudo_ood is not None) and id_E.size:
                        from audioshieldnet.security.pseudo_ood import OODS
                        try:
                            from audioshieldnet.security.pseudo_ood import ALIASES
                        except Exception:
                            ALIASES = {}

                        ood_types_eval = self.ood_cfg.get('types', list(OODS.keys()))
                        with torch.no_grad():
                            for t in ood_types_eval:
                                t_eval = ALIASES.get(t, t)
                                if t_eval not in OODS:
                                    continue
                                oE = []
                                for batch in dl_va:
                                    wav = batch[0].to(self.device, non_blocking=True)
                                    wav_t = OODS[t_eval](wav, getattr(self.pseudo_ood, "sr", 16000), 0)
                                    wav_t = wav_t.to(self.device, non_blocking=True) 
                                    lm_o, pm_o = self.feats(wav_t)
                                    logits_o, _ = self.net(lm_o, pm_o, target=None)
                                    oE.append(energy_score(logits_o, T=self.T_energy).cpu().numpy())
                                oE = np.concatenate(oE) if oE else np.array([])
                                if oE.size:
                                    y_mix = np.concatenate([np.zeros_like(id_E), np.ones_like(oE)])
                                    s_mix = np.concatenate([id_E, oE])
                                    try:
                                        per_type_auroc[t] = float(skm.roc_auc_score(y_mix, s_mix))
                                    except Exception:
                                        per_type_auroc[t] = float("nan")

                    # 5) Abstain-aware metrics
                    abstain_tau = float(self.tau_susp)
                    auc_kept, eer_kept, abstain_frac = float("nan"), float("nan"), float("nan")
                    if id_E.size and y_true.size and y_prob.size:
                        keep_mask = (id_E < abstain_tau)
                        if keep_mask.any():
                            yk, pk = y_true[keep_mask], y_prob[keep_mask]
                            if (yk.size > 1) and (np.unique(yk).size > 1):
                                auc_kept = skm.roc_auc_score(yk.astype(int), pk.astype(float))
                                fpr_k, tpr_k, _ = skm.roc_curve(yk.astype(int), pk.astype(float))
                                fnr_k = 1.0 - tpr_k
                                idx_k = np.nanargmin(np.abs(fnr_k - fpr_k))
                                eer_kept = float((fpr_k[idx_k] + fnr_k[idx_k]) / 2.0)
                            abstain_frac = 1.0 - float(keep_mask.mean())


                    # 6) Energy τ auto-tune + Risk–Coverage + ECE_kept
                    energies = id_E
                    tau_auto = None
                    if energies.size and epoch >= 5:
                        tau_auto = auto_tau_from_val_energies(energies, q=0.95)
                    if tau_auto is not None:
                        self.tau_susp = max(float(tau_auto), -0.25)
                        self.cfg['security']['triage']['tau_susp_energy'] = self.tau_susp
                        if self.wandb:
                            self.wandb.log({"val/auto_tau_susp_energy": self.tau_susp}, commit=False)

                    if energies.size and y_true.size and y_prob.size:
                        covs,  risks,  AURC  = risk_coverage_from_energy(y_true, y_prob, energies, metric="eer")
                        covs2, risks2, AURC2 = risk_coverage_from_energy(y_true, y_prob, energies, metric="1-auc")
                        if self.wandb:
                            self.wandb.log({"val/aurc_eer": AURC, "val/aurc_1minusauc": AURC2}, commit=False)
                            wandb_log_risk_coverage(self.wandb, covs,  risks,  title="RC(EER)")
                            wandb_log_risk_coverage(self.wandb, covs2, risks2, title="RC(1-AUC)")

                        keep_mask_current = (energies < float(self.tau_susp))
                        ece_kept = ece_on_subset(y_true[keep_mask_current], y_prob[keep_mask_current])
                        if self.wandb:
                            self.wandb.log({"val/ece_kept": ece_kept}, commit=False)
                    else:
                        AURC, AURC2, ece_kept = float("nan"), float("nan"), float("nan")

                    # 7) Suspicious fraction
                    susp_frac = suspicious_fraction(self.feats, self.net, dl_va, self.device, self.T_energy, self.tau_susp)

                    # Pretty print
                    print(
                        f"[{epoch}] loss={epoch_loss:.4f}  valBCE={val_bce:.4f}  AUC={auc:.3f}  EER={eer:.3f}  "
                        f"ECE={ece:.3f}  FNR@95(FGSM)={fnr95:.3f}  Susp%={susp_frac*100:.1f}  "
                        f"OOD-AUROC={ood_auroc:.3f}  AUC_kept={auc_kept:.3f}  EER_kept={eer_kept:.3f}  "
                        f"Abstain%={abstain_frac*100:.1f}  τ_auto={self.tau_susp:.3f}"
                    )
                    
                    # ---- LR scheduler (after all monitored metrics are computed) ----
                    if self.scheduler is not None and (not self.sched_per_batch):
                        try:
                            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                # choose the correct metric to monitor based on best_metric_name
                                monitored = {
                                    "auc": auc, "eer": eer, "ece": ece,
                                    "fnr95_fgsm": fnr95, "susp_frac": susp_frac
                                }.get(best_metric_name, auc)

                                # Guard against NaNs (plateau schedulers dislike NaN)
                                if monitored is None or (isinstance(monitored, float) and (np.isnan(monitored) or np.isinf(monitored))):
                                    print(f"[WARN] Scheduler skipped at epoch {epoch}: monitored metric '{best_metric_name}' is NaN/Inf.")
                                else:
                                    self.scheduler.step(monitored)
                            else:
                                # Step-based / cosine etc.
                                self.scheduler.step()
                        except Exception as e:
                            print(f"[WARN] Scheduler step skipped at epoch {epoch}: {e}")


                    # ============================================================
                    # Confusion Matrix + Precision/Recall/F1 at EER (clean unified version)
                    # ============================================================
                    try:
                        

                        # --- Compute EER threshold once ---
                        fpr, tpr, thr = skm.roc_curve(y_true.astype(int), y_prob.astype(float))
                        fnr = 1.0 - tpr
                        idx = np.nanargmin(np.abs(fnr - fpr))
                        thr_eer = float(thr[idx]) if (thr is not None and thr.size) else 0.5

                        # --- Predictions + core metrics ---
                        y_hat = (y_prob >= thr_eer).astype(int)
                        tn, fp, fn, tp = skm.confusion_matrix(y_true.astype(int), y_hat, labels=[0, 1]).ravel()
                        prec_eer, rec_eer, f1_eer, _ = skm.precision_recall_fscore_support(
                            y_true.astype(int), y_hat, average="binary", zero_division=0
                        )

                        # print(
                        #     f"[VAL][EER] thr={thr_eer:.3f}  TP={tp} FP={fp} TN={tn} FN={fn}  "
                        #     f"Prec={prec_eer:.3f}  Rec={rec_eer:.3f}  F1={f1_eer:.3f}"
                        # )

                        # --- Log scalars + confusion matrix to W&B ---
                        if self.wandb:
                            self.wandb.log({
                                "val/conf_thr_eer": thr_eer,
                                "val/conf_tp": tp, "val/conf_fp": fp, "val/conf_tn": tn, "val/conf_fn": fn,
                                "val/prec_eer": prec_eer,
                                "val/rec_eer": rec_eer,
                                "val/f1_eer": f1_eer,
                            }, commit=False)

                        # Save + log confusion matrix (EER threshold only)
                        log_val_confusion_metrics(
                            y_true=y_true,
                            y_prob=y_prob,
                            metrics_dir=self.metrics_dir,
                            epoch=epoch,
                            wandb_run=self.wandb,
                            threshold=thr_eer,
                            # also_log_fixed_05=False,  # optionally True if you want both
                            label_names=["real", "fake"]
                        )

                        # --- Optional: ROC/PR curves for visualization only ---
                        prec_curve, rec_curve, _ = skm.precision_recall_curve(y_true.astype(int), y_prob.astype(float))
                        if self.wandb:
                            roc_tbl = wandb.Table(data=list(zip(fpr.tolist(), tpr.tolist())), columns=["fpr", "tpr"])
                            pr_tbl  = wandb.Table(data=list(zip(rec_curve.tolist(), prec_curve.tolist())), columns=["recall", "precision"])
                            self.wandb.log({
                                "val/roc_curve": wandb.plot.line(roc_tbl, "fpr", "tpr", title="ROC"),
                                "val/pr_curve":  wandb.plot.line(pr_tbl, "recall", "precision", title="PR")
                            }, commit=False)

                    except Exception as e:
                        print(f"[WARN] Validation confusion/curve logging failed: {e}")

                    # ============================================================
                    # Curriculum schedule visualization (unchanged)
                    # ============================================================
                    # use the derived flags + start epochs from section (1)
                    sam_on, energy_on, ood_on, adv_on = sam_enabled, energy_enabled, ood_enabled, adv_enabled


                    bar = (f"[Epoch {epoch:02d}] Curriculum: "
                        f"SAM{'✓' if sam_on else '✗'}  "
                        f"ENERGY{'✓' if energy_on else '✗'}  "
                        f"OOD{'✓' if ood_on else '✗'}  "
                        f"ADV{'✓' if adv_on else '✗'}")
                    print(bar)
                    if self.wandb:
                        self.wandb.log({"val/curriculum_bar": bar}, commit=False)


                    if epoch == 0 or sam_on or energy_on or ood_on or adv_on:
                        try:
                            
                            log_curriculum_schedule(
                                cfg=self.cfg,
                                total_epochs=total_epochs,
                                metrics_dir=self.metrics_dir,
                                epoch=epoch,
                                wandb_run=self.wandb
                            )
                        except Exception as e:
                            print(f"[WARN] Curriculum plot skipped: {e}")

                    # Restore EMA to normal net after val
                    if self.ema is not None:
                        self.ema.restore(self.net)

                                                
                                            

                    # 9) W&B logging (no duplicate ROC/PR plotting here)
                    if self.wandb:
                        payload = {
                            "train/loss_epoch": epoch_loss,
                            "val/bce": val_bce,
                            "val/auc": auc,
                            "val/eer": eer,
                            "val/ece": ece,
                            "val/fnr95_fgsm": fnr95,
                            "val/rob_auc": rob_auc,
                            "val/rob_eer": rob_eer,
                            "val/ood_auroc": ood_auroc,
                            "val/auc_kept": auc_kept,
                            "val/eer_kept": eer_kept,
                            "val/abstain_frac": abstain_frac,
                            "val/susp_frac": susp_frac,
                            "val/aurc_eer": AURC if 'AURC' in locals() else float("nan"),
                            "val/aurc_1minusauc": AURC2 if 'AURC2' in locals() else float("nan"),
                            "val/ece_kept": ece_kept if 'ece_kept' in locals() else float("nan"),
                            "epoch": epoch
                        }
                        for k, v in (per_type_auroc or {}).items():
                            payload[f"val/ood_auroc/{k}"] = v
                        wandb_log_safe(self.wandb, payload, commit=False)
                        self.wandb.log({}, commit=True)
                        

                    # ---- Checkpointing & best-model selection ----
                    if (epoch % save_every_epochs) == 0:
                        last_p = os.path.join(self.ckpt_dir, "last.ckpt")
                        save_checkpoint(last_p, self.net, self.opt, self.ema, epoch, 0, None, self.cfg)
                        if self.wandb:
                            last_art = wandb.Artifact("ckpt_last", type="model")
                            last_art.add_file(last_p)
                            self.wandb.log_artifact(last_art, aliases=["latest"])

                    stats = {
                        'epoch': epoch, 'auc': float(auc), 'eer': float(eer), 'ece': float(ece),
                        'fnr95_fgsm': float(fnr95), 'susp_frac': float(susp_frac)
                    }
                    current_value = float(stats[best_metric_name])
                    if metric_better(current_value, best_value, best_mode):
                        best_value = current_value
                        best_p = os.path.join(self.ckpt_dir, "best.ckpt")
                        save_checkpoint(best_p, self.net, self.opt, self.ema, epoch, 0, best_value, self.cfg)
                        

                        with open(os.path.join(self.metrics_dir, "val_best.json"), "w") as f:
                            json.dump({
                                "best_metric": best_metric_name,
                                "best_mode": best_mode,
                                **stats,
                                "invert_polarity": bool(self._polarity_fixed if self._polarity_fixed is not None else invert)
                            }, f, indent=2)


                        snap = f"epoch{epoch:03d}_{best_metric_name}{best_value:.4f}.ckpt"
                        topk_p = os.path.join(self.ckpt_dir, "topk", snap)
                        save_checkpoint(topk_p, self.net, self.opt, self.ema, epoch, 0, best_value, self.cfg)
                        prune_topk(self.ckpt_dir, keep_top_k)
                        print(f"[CHECKPOINT] New best model (by {best_metric_name}={best_value:.4f}) saved.")

                        if self.wandb:
                            best_art = wandb.Artifact("ckpt_best", type="model")
                            best_art.add_file(best_p)
                            self.wandb.log_artifact(best_art, aliases=["best"])

                            topk_art = wandb.Artifact("ckpt_topk", type="model")
                            topk_art.add_dir(os.path.join(self.ckpt_dir, "topk"))
                            self.wandb.log_artifact(topk_art, aliases=["topk", f"epoch{epoch:03d}"])

                    # ---- SWA update ----
                    if self.use_swa and self.swa_model is not None:
                        if swa_should_update(self.cfg, epoch, total_epochs):
                            self.swa_model.update_parameters(self.net)

                    # Unfreeze BN after the first SAM epoch (one-epoch freeze)
                    if self._sam_just_switched:
                        set_bn_train_mode(self.net, True)
                        self._sam_just_switched = False

                    tepochs.set_postfix(current_epoch=epoch, best=f"{best_metric_name}:{best_value:.4f}")
                    tepochs.update(1)
                print("\n")       
            

        finally:
            # ---- SWA finalize ----
            if self.use_swa and self.swa_model is not None:
                try:
                    print("[SWA] Updating BN statistics on train loader...")
                    from audioshieldnet.utils.opt_sched import swa_update_bn_two_input
                    swa_update_bn_two_input(dl_tr, self.swa_model, self.feats, device=self.device)

                    # Export SWA as a separate artifact; don't clobber live/EMA model here.
                    swa_p = os.path.join(self.ckpt_dir, "swa.ckpt")
                    save_checkpoint(
                        swa_p,
                        (self.swa_model.module if hasattr(self.swa_model, "module") else self.swa_model),
                        self.opt, self.ema,
                        epoch if 'epoch' in locals() else start_epoch,
                        0, None, self.cfg
                    )
                    if self.wandb:
                        swa_art = wandb.Artifact("ckpt_swa", type="model")
                        swa_art.add_file(swa_p)
                        self.wandb.log_artifact(swa_art, aliases=["swa"])

                except Exception as e:
                    print(f"[SWA] finalize skipped: {e}")

            # Rolling last checkpoint
            last_p = os.path.join(self.ckpt_dir, "last.ckpt")
            save_checkpoint(last_p, self.net, self.opt, self.ema, epoch if 'epoch' in locals() else start_epoch, 0, best_value, self.cfg)

            if self.wandb:
                try:
                    last_art = wandb.Artifact("ckpt_last", type="model")
                    last_art.add_file(last_p)
                    self.wandb.log_artifact(last_art, aliases=["latest"])
                except Exception:
                    pass
                self.wandb.finish()

        print(f"[FINISH] Training completed. Best {best_metric_name}={best_value:.4f}.")
