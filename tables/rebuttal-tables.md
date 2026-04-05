**Table R1b. Intermediate stage diagnostics on LibriSeVoc.**

| Stage Transition | Property Delivered | Diagnostic | Before → After Stage |
|:---|:---|:---|:---|
| Baseline → +SAM | Loss landscape smoothing | SAM perturbation norm ‖∇L‖₂ | 0.41 → 0.18 |
| +SAM → +Energy | Cross-domain energy stability | Var[E(x<sub>clean</sub>) − E(x<sub>aug</sub>)] | 0.089 → 0.021 |
| +Energy → +OOD | Energy distribution alignment | KL(E<sub>clean</sub> ‖ E<sub>codec</sub>) | 0.31 → 0.09 |
| +OOD → +Adversarial | Logit drift under PGD | ‖f(x) − f(x<sub>adv</sub>)‖₂ | 0.44 → 0.17 |


**Table R2. Joint coefficient perturbation on LibriSeVoc.** Each trial perturbs $\lambda_{\mathrm{cmra}}$, $\lambda_{\mathrm{eng}}$, $\lambda_{\mathrm{adv}}$, and $\lambda_{\mathrm{ood}}$ **simultaneously** by $\pm 50\%$. The goal is not strict invariance, but stability within the same performance regime without retuning.

| Trial | $\lambda_{\mathrm{cmra}}$ | $\lambda_{\mathrm{eng}}$ | $\lambda_{\mathrm{adv}}$ | $\lambda_{\mathrm{ood}}$ | Rob. AUC ↑ | OOD-AUROC ↑ | ECE ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Default | 1.0 | 1.0 | 1.0 | 1.0 | 0.90 | 0.71 | 0.12 |
| P1 | 0.5 | 1.0 | 1.5 | 0.5 | 0.89 | 0.67 | 0.13 |
| P2 | 1.5 | 0.5 | 1.0 | 1.5 | 0.88 | 0.70 | 0.15 |
| P3 | 0.5 | 1.5 | 0.5 | 1.0 | 0.86 | 0.68 | 0.13 |
| P4 | 1.0 | 0.5 | 1.5 | 0.5 | 0.89 | 0.66 | 0.14 |
| **Range** | — | — | — | — | **0.04** | **0.05** | **0.03** |


**Table R4. Hyperparameter audit for ASNet.**

| Category | Parameters | Count | How Set |
|:---|:---|:---:|:---|
| **Learned** (no manual tuning) | ECRM gate w<sub>g</sub>, b<sub>g</sub> | 2 | Gradient-based optimization |
| **Data-driven** (auto from validation) | Energy anchor τ<sub>id</sub> | 1 | Initialized from the median energy of correctly classified validation samples and refreshed periodically (App. B.4, Eq. 10) |
| **Fixed from validation statistics** | CMRA margin m = 0.8 | 1 | Set once from the high-quantile of the validation cosine-similarity distribution (App. B.5); not retuned per dataset |
| **Standard practice** (shared with all baselines) | SAM radius ρ, PGD budget ε, PGD step α, learning rate, weight decay, batch size | 6 | Standard ranges from the SAM and PGD literature; identical values used for all baselines |
| **ASNet-specific, low-sensitivity** | Curriculum timing (20%/40%/60%) | 3 | Fixed; ΔRobust AUC ≤ 0.02 under ±10% perturbation (Table 8, App. B.6) |
| **ASNet-specific, low-sensitivity** | Loss coefficients $\lambda_{\mathrm{cmra}}$, $\lambda_{\mathrm{eng}}$, $\lambda_{\mathrm{adv}}$, and $\lambda_{\mathrm{ood}}$ | 4 | Default 1.0; ΔRobust AUC ≤ 0.04 under ±50% joint perturbation (Table R2) |
