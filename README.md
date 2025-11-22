

````markdown
# 🛡️ AudioShieldNet (ASNet)
### A Defense-in-Depth Framework for Secure and Trustworthy Audio Deepfake Detection
<!-- 
<p align="center">
  <img src="figures/asnet.png" width="700"/>
</p> -->

---

## 📄 Abstract
Modern audio deepfake detectors remain brittle under unseen vocoders, codec distortions, and small adversarial perturbations. **ASNet** introduces a **security-aware, dual-encoder architecture** with:

- Spectral + prosodic encoders  
- Energy-gated cross-fusion  
- Cross-Modal Robustness Alignment (CMRA)  
- A staged robustness curriculum integrating SAM, OOD consistency, and adversarial logit alignment  

Under a scoped digital threat model, ASNet maintains competitive in-domain accuracy while improving unseen-domain AUC, codec robustness, adversarial stability, and calibrated abstention across multiple corpora.

Full method details, architecture, curriculum, and results are described in the associated CVPR 2026 submission. :contentReference[oaicite:1]{index=1}

---

## 🚀 Key Features

### **Dual-View Architecture**
- **Spectral encoder:** 80-bin log-mels → micro-acoustic spoof cues  
- **Prosody encoder:** f0, energy, ZCR, flux → macro-temporal cues  

### **Energy-Gated Fusion**
- Learns adaptive view weighting  
- Down-weights unreliable modalities under codec/channel shift  
- Provides both **confidence** and **abstention**

### **Cross-Modal Robustness Alignment (CMRA)**
- Encourages stable, complementary representations  
- Controls drift under perturbations

### **Security-Aware Training Curriculum**
Stages:
1. Base detector  
2. SAM (sharpness-aware optimization)  
3. Energy calibration  
4. OOD consistency  
5. PGD-based adversarial alignment with ECRM  

### **Strong Robustness**
- Improved AUC/EER across 5 corpora  
- Higher adversarial robustness (PGD / CW / AutoAttack)  
- Improved codec/channel OOD-AUROC  
- Best selective risk–coverage curves

---

## 📊 Highlighted Results

| Setting | Metric | ASNet | Strong Baseline (Ren et al.) |
|--------|--------|--------|-------------------------------|
| **ASVspoof21 (unseen)** | AUC ↑ | **0.862** | 0.842 |
| **CodecFake OOD-AUROC** | ↑ | **0.71** | 0.66 |
| **Robust AUC (PGD-10)** | ↑ | **0.56** | 0.53 |
| **Selective (20% abstention)** | EER ↓ | **0.079** | 0.103 |

ASNet consistently lies on the **Pareto frontier** of accuracy vs robustness.

---

## 📦 Installation

### **1. Clone Repository**
```bash
git clone https://github.com/aiai-9/ASNet.git
cd ASNet
````

### **2. Create Conda Environment**

```bash
conda create -n asnet python=3.10 -y
conda activate asnet
```

### **3. Install Requirements**

```bash
pip install -r requirements.txt
```

### **4. (Optional) Install GPU-accelerated libraries**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🗂️ Project Structure

```
ASNet/
│
├── figures/
│    └── asnet.png
│
├── asnet/
│    ├── models/           # spectral encoder, prosody encoder, fusion, CMRA
│    ├── training/         # curriculum, PGD, SAM, ECRM
│    ├── data/             # dataset loaders + preprocessing
│    ├── utils/            # metrics, logging, visualization
│    └── config/           # YAML configs for each corpus
│
├── scripts/
│    ├── train_asnet.py
│    ├── eval_asnet.py
│    └── adversarial_attack.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🏋️ Training ASNet

### **Train on LibriSeVoc (default)**

```bash
python scripts/train_asnet.py \
    --config config/asnet_lsv.yaml
```

The YAML includes:

* Curriculum schedule
* Encoder configs
* Loss weights (CMRA, energy, OOD, adv)
* Augmentation settings
* PGD parameters

---

## 🧪 Evaluation

### **1. Evaluate on clean data**

```bash
python scripts/eval_asnet.py \
    --config config/asnet_lsv.yaml \
    --checkpoint checkpoints/asnet_best.pt
```

### **2. Evaluate adversarial robustness**

```bash
python scripts/adversarial_attack.py \
    --checkpoint checkpoints/asnet_best.pt \
    --epsilon 0.001 --steps 10
```

### **3. Evaluate codec robustness**

```bash
python scripts/eval_asnet.py \
    --codec_test mp3_64
```

### **4. Selective classification (energy-based abstention)**

```bash
python scripts/eval_asnet.py \
    --abstain_threshold 0.5
```

---

## 📈 Reproducibility Checklist

* All experiments run on a single **A100-80GB**
* Training uses **80 epochs**, batch size 128
* Sliding-window inference (6 s windows, 50% overlap)
* 3 seeds: {0, 1, 2}
* Adversarial evaluation: PGD-10, CW-50, AutoAttack

---

## 📚 Citation

If you use ASNet in academic work:

```bibtex
@article{asnet2026,
  title={AudioShieldNet: A Defense-in-Depth Framework for Secure and Trustworthy Audio Deepfake Detection},
  author={Anonymous},
  journal={CVPR},
  year={2026}
}
```

---

## 🛡 License

MIT License.

---

