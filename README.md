

# 🛡️ AudioShieldNet (ASNet)

### **A Defense-in-Depth Framework for Secure and Trustworthy Audio Deepfake Detection**

<p align="center">
  <img src="figures/asnet_model.png" />
</p>

---

## 📘 Abstract

Modern audio deepfake detectors remain brittle under unseen vocoders, codec distortions, and small adversarial perturbations.
**AudioShieldNet (ASNet)** introduces the first *defense-in-depth* audio security framework combining:

* Spectral + prosodic dual encoders
* Energy-gated cross-fusion
* Cross-Modal Robustness Alignment (CMRA)
* A staged robustness curriculum integrating SAM, OOD consistency, and adversarial ECRM alignment

Under a scoped digital threat model, ASNet maintains competitive in-domain accuracy while significantly improving unseen-vocoder, codec-shift, and adversarial robustness.

Full architecture, curriculum, evaluations, and ablations are described in the associated CVPR 2026 submission.

---

## 🚀 Key Features

### **🔹 Dual-View Architecture**

* **Spectral encoder:** 80-bin log-mels → micro-acoustic spoof cues
* **Prosody encoder:** f0, energy, ZCR, flux → macro-temporal cues

### **🔹 Energy-Gated Fusion**

* Learns adaptive reliability weighting
* Down-weights unreliable modalities under codec/channel shift
* Produces both **confidence** and **abstention** scores

### **🔹 Cross-Modal Robustness Alignment (CMRA)**

* Stabilizes representations across modalities
* Reduces drift under perturbations

### **🔹 Security-Aware Training Curriculum**

1. Base detector
2. SAM (sharpness-aware optimization)
3. Energy calibration
4. OOD consistency
5. PGD-based adversarial alignment with ECRM

### **🔹 Strong Robustness (Summary)**

* Improved **AUC/EER across 5 corpora**
* Higher adversarial robustness (PGD / CW / AutoAttack)
* Improved codec/channel **OOD-AUROC**
* Lower overconfidence under attack (ECE↓)

---

## 🧱 Architecture Overview

<p align="center">
  <img src="figures/asnet_architecture.png" width="750"/>
</p>

ASNet integrates:

* **Dual encoders (spectral + prosody)**
* **Energy-weighted fusion gate**
* **CMRA alignment block**
* **Security-aware multi-stage head** yielding:

  * Real/Fake probability
  * Energy-based OOD score
  * Abstention logit

---

## 📊 Key Metrics (from paper)

| Metric                    | ASNet | Best Baseline | Gain             |
| ------------------------- | ----- | ------------- | ---------------- |
| **Cross-vocoder AUC**     | ↑     | –             | **+4–8%**        |
| **Codec-shift AUROC**     | ↑     | –             | **+7–12%**       |
| **Adversarial EER (PGD)** | ↓     | –             | **30–50% lower** |
| **Energy-calibrated ECE** | ↓     | –             | **Reduced 2–3×** |

(Full tables available in the paper PDF included in the repo.)

---

## 📦 Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/aiai-9/ASNet.git
cd ASNet
```

### **2. Create Environment**

```bash
conda create -n asnet python=3.10 -y
conda activate asnet
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ How to Run ASNet

### **Train**

```bash
python audioshieldnet/engine/trainer.py \
    --config configs/asnet_base.yaml
```

### **Evaluate**

```bash
python audioshieldnet/engine/evaluator.py \
    --config configs/asnet_base.yaml \
    --checkpoint ckpts/asnet_best.pt
```

### **Run Security Benchmarks**

```bash
python audioshieldnet/security/attacks.py \
    --config configs/asnet_base.yaml
```

---

## 📁 Repository Structure

```
ASNet/
│
├── audioshieldnet/
│   ├── data/               → loaders + splits
│   ├── engine/             → trainer/evaluator
│   ├── models/             → ASNet encoders + fusion
│   ├── losses/             → CMRA, OOD, security losses
│   ├── security/           → attacks, ECRM, calibrations
│   └── utils/              → scheduler, seed, metrics
│
├── configs/                → YAML configs
├── figures/                → architecture & paper figures
├── requirements.txt
└── README.md
```

---


