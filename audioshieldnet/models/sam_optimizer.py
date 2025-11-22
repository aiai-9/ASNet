# audioShieldNet/asnet_6/audioshieldnet/models/sam_optimizer.py
import torch

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021)."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False):
        if not isinstance(base_optimizer, torch.optim.Optimizer):
            raise TypeError("base_optimizer must be an instance of torch.optim.Optimizer")

        # Canonical defaults
        defaults = dict(rho=float(rho), adaptive=bool(adaptive))
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer

        # 🔒 Ensure the BASE optimizer's groups have SAM keys
        for g in self.base_optimizer.param_groups:
            g.setdefault("rho", float(rho))
            g.setdefault("adaptive", bool(adaptive))

        # Share groups
        self.param_groups = self.base_optimizer.param_groups

        # Scheduler-friendly step counter
        self._step_count = getattr(self, "_step_count", 0)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        scale = self._calc_scale()
        if scale == 0.0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        for group in self.param_groups:
            rho = float(group.get("rho", self.defaults.get("rho", 0.05)))
            adaptive = bool(group.get("adaptive", self.defaults.get("adaptive", False)))
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.abs(p) if adaptive else 1.0) * p.grad
                self.state[p]["e_w"] = e_w
                p.add_(e_w, alpha=(rho / scale))
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            rho = float(group.get("rho", self.defaults.get("rho", 0.05)))
            scale = self._calc_scale() or 1.0
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = self.state[p].get("e_w", None)
                if e_w is not None:
                    p.sub_(e_w, alpha=(rho / scale))

        self.base_optimizer.step()
        self._step_count = getattr(self, "_step_count", 0) + 1

        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, *args, **kwargs):
        raise RuntimeError("Use first_step(...); second_step(...) with SAM.")

    def _calc_scale(self) -> float:
        norms = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    norms.append(torch.norm(p.grad))
        if not norms:
            return 0.0
        return float(torch.norm(torch.stack(norms)))
