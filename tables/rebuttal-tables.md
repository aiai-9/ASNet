**Table R1. Curriculum permutation ablation on LibriSeVoc.** All rows contain the same four components (SAM, Energy, OOD, Adversarial), the same architecture, and the same 80-epoch budget. Only activation order differs.

| Ordering | AUC ↑ | EER ↓ | ECE ↓ | Rob. AUC ↑ | Rob. EER ↓ | OOD-AUROC ↑ |
|:---|---:|---:|---:|---:|---:|---:|
| SAM→Eng→OOD→Adv **(ours)** | 0.998 | 0.012 | 0.12 | 0.90 | 0.16 | 0.71 |
| Eng→SAM→OOD→Adv | 0.994 | 0.023 | 0.15 | 0.86 | 0.21 | 0.68 |
| OOD→Eng→SAM→Adv | 0.991 | 0.027 | 0.19 | 0.84 | 0.23 | 0.63 |
| Adv→SAM→Eng→OOD | 0.992 | 0.026 | 0.20 | 0.83 | 0.22 | 0.64 |
| Adv→OOD→Eng→SAM | 0.985 | 0.040 | 0.23 | 0.78 | 0.28 | 0.58 |
| All simultaneous | 0.983 | 0.044 | 0.24 | 0.76 | 0.29 | 0.57 |

---

**Table R1b. Intermediate stage diagnostics on LibriSeVoc.**

| Stage Transition | Property Delivered | Diagnostic | Before → After Stage |
|:---|:---|:---|:---|
| Baseline → +SAM | Loss landscape smoothing | SAM perturbation norm ‖∇L‖₂ | 0.41 → 0.18 |
| +SAM → +Energy | Cross-domain energy stability | Var[E(x<sub>clean</sub>) − E(x<sub>aug</sub>)] | 0.089 → 0.021 |
| +Energy → +OOD | Energy distribution alignment | KL(E<sub>clean</sub> ‖ E<sub>codec</sub>) | 0.31 → 0.09 |
| +OOD → +Adversarial | Logit drift under PGD | ‖f(x) − f(x<sub>adv</sub>)‖₂ | 0.44 → 0.17 |

---

**Table R2. Joint coefficient perturbation on LibriSeVoc.** Each trial perturbs $\lambda_{\mathrm{cmra}}$, $\lambda_{\mathrm{eng}}$, $\lambda_{\mathrm{adv}}$, and $\lambda_{\mathrm{ood}}$ **simultaneously** by $\pm 50\%$. The goal is not strict invariance, but stability within the same performance regime without retuning.

| Trial | $\lambda_{\mathrm{cmra}}$ | $\lambda_{\mathrm{eng}}$ | $\lambda_{\mathrm{adv}}$ | $\lambda_{\mathrm{ood}}$ | Rob. AUC ↑ | OOD-AUROC ↑ | ECE ↓ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Default | 1.0 | 1.0 | 1.0 | 1.0 | 0.90 | 0.71 | 0.12 |
| P1 | 0.5 | 1.0 | 1.5 | 0.5 | 0.89 | 0.67 | 0.13 |
| P2 | 1.5 | 0.5 | 1.0 | 1.5 | 0.88 | 0.70 | 0.15 |
| P3 | 0.5 | 1.5 | 0.5 | 1.0 | 0.86 | 0.68 | 0.13 |
| P4 | 1.0 | 0.5 | 1.5 | 0.5 | 0.89 | 0.66 | 0.14 |
| **Range** | — | — | — | — | **0.04** | **0.05** | **0.03** |

---

**Table R3. Hyperparameter audit for ASNet.**

| Category | Parameters | Count | How Set |
|:---|:---|:---:|:---|
| **Learned** (no manual tuning) | ECRM gate w<sub>g</sub>, b<sub>g</sub> | 2 | Gradient-based optimization |
| **Data-driven** (auto from validation) | Energy anchor τ<sub>id</sub> | 1 | Initialized from the median energy of correctly classified validation samples and refreshed periodically (App. B.4, Eq. 10) |
| **Fixed from validation statistics** | CMRA margin m = 0.8 | 1 | Set once from the high-quantile of the validation cosine-similarity distribution (App. B.5); not retuned per dataset |
| **Standard practice** (shared with all baselines) | SAM radius ρ, PGD budget ε, PGD step α, learning rate, weight decay, batch size | 6 | Standard ranges from the SAM and PGD literature; identical values used for all baselines |
| **ASNet-specific, low-sensitivity** | Curriculum timing (20%/40%/60%) | 3 | Fixed; ΔRobust AUC ≤ 0.02 under ±10% perturbation (Table 8, App. B.6) |
| **ASNet-specific, low-sensitivity** | Loss coefficients $\lambda_{\mathrm{cmra}}$, $\lambda_{\mathrm{eng}}$, $\lambda_{\mathrm{adv}}$, and $\lambda_{\mathrm{ood}}$ | 4 | Default 1.0; ΔRobust AUC ≤ 0.04 under ±50% joint perturbation (Table R2) |


---

**Table R4. Simplified ASNet variants on LibriSeVoc.**

| Variant | What's Kept | What's Removed | Rob. AUC ↑ | OOD-AUROC ↑ | ECE ↓ |
|:---|:---|:---|---:|---:|---:|
| ASNet-Lite | 1 encoder + SAM + Adv | Prosody, CMRA, gated fusion, ECRM, OOD, Energy | 0.76 | 0.58 | 0.23 |
| ASNet-Lite + Prosody† | 2 encoders (concat) + SAM + Adv | CMRA, gated fusion, ECRM, OOD, Energy | 0.81 | 0.60 | 0.21 |
| ASNet-Mid | 2 encoders + concat + SAM + Adv + Energy | CMRA, gated fusion, ECRM | 0.84 | 0.64 | 0.17 |
| Full ASNet | All | — | 0.90 | 0.71 | 0.12 |

---

**Table R4. Complexity–performance trade-off of simplified ASNet variants.**  
All variants use the same single-pass inference procedure at test time; only the training-time robustness components differ. Complexity is reported **relative to Full ASNet** (so **1.00× = same as Full ASNet**, and smaller values indicate lower cost).

| Variant | What's Kept | What's Removed | Rob. AUC ↑ | OOD-AUROC ↑ | ECE ↓ | Params (vs Full ASNet) ↓ | Train cost (vs Full ASNet) ↓ | Inference cost (vs Full ASNet) ↓ | Abstention available |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| ASNet-Lite | 1 encoder + SAM + Adv | Prosody, CMRA, gated fusion, ECRM, OOD, Energy | 0.76 | 0.58 | 0.23 | 0.62× | 0.58× | 0.73× | No |
| ASNet-Lite + Prosody† | 2 encoders (concat) + SAM + Adv | CMRA, gated fusion, ECRM, OOD, Energy | 0.81 | 0.60 | 0.21 | 0.79× | 0.70× | 0.86× | No |
| ASNet-Mid | 2 encoders + concat + SAM + Adv + Energy | CMRA, gated fusion, ECRM | 0.84 | 0.64 | 0.17 | 0.87× | 0.81× | 0.92× | Yes |
| Full ASNet | All | — | 0.90 | 0.71 | 0.12 | 1.00× | 1.00× | 1.00× | Yes |

---

**Table R5. Is the residual blind spot specific to logit-derived confidence? A matched comparison of confidence signals for robustness modulation and abstention.**

| Confidence / control signal | Signal type | Rob. AUC ↑ | OOD-AUROC ↑ | ECE ↓ | Misclassified caught by abstention ↑ | Residual blind spot ↓ | AUC_kept @20% ↑ | EER_kept @20% ↓ |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| Logit margin | logit-derived | 0.821 | 0.683 | 0.061 | 0.70 | 0.30 | 0.898 | 0.121 |
| Predictive entropy | logit-derived | 0.828 | 0.691 | 0.058 | 0.73 | 0.27 | 0.906 | 0.114 |
| Mahalanobis on fused feature \(z^*\) | non-logit feature-space | 0.832 | 0.700 | 0.056 | 0.74 | 0.26 | 0.913 | 0.108 |
| Auxiliary confidence head on \(z^*\) | non-logit learned | 0.838 | 0.706 | 0.053 | 0.76 | 0.24 | 0.921 | 0.101 |
| Energy (ours) | logit-derived, class-symmetric | **0.857** | **0.734** | **0.041** | **0.81** | **0.19** | **0.949** | **0.079** |

---

