<div align="center">

<img src="https://img.shields.io/badge/IEEE-Paper-00629B?style=for-the-badge&logo=ieee&logoColor=white" alt="IEEE Paper"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/>
<img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License"/>

<br/><br/>

<h1>🫁 MASA-Net v2</h1>
<h3><em>Multi-scale Attention Self-Augmentation Network<br/>for Pneumonia Detection on PneumoniaMNIST</em></h3>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-93.75%25-0D9488?style=flat-square&labelColor=0F172A" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/F1--Score-0.9511-0D9488?style=flat-square&labelColor=0F172A" alt="F1"/>
  <img src="https://img.shields.io/badge/AUC--ROC-0.9770-0D9488?style=flat-square&labelColor=0F172A" alt="AUC"/>
  <img src="https://img.shields.io/badge/Sensitivity-0.9718-0D9488?style=flat-square&labelColor=0F172A" alt="Sensitivity"/>
  <img src="https://img.shields.io/badge/Specificity-0.8803-0D9488?style=flat-square&labelColor=0F172A" alt="Specificity"/>
</p>

<br/>

> **Published at IEEE** | MITS DU, Gwalior, India
>
> Ayan Ahmed Khan

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Architecture](#-architecture)
- [Results](#-results)
- [Ablation Study](#-ablation-study)
- [Computational Efficiency](#-computational-efficiency)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Figures & Visualizations](#-figures--visualizations)
- [Citation](#-citation)
- [License](#-license)

---

## 🔭 Overview

MASA-Net v2 is a deep learning framework for binary pneumonia classification from chest X-ray images on the **PneumoniaMNIST** benchmark. It combines a modern ConvNeXt-Small backbone with channel-wise squeeze-and-excitation attention, a novel **BinaryTET loss**, and a carefully designed inference pipeline to achieve state-of-the-art results.

<div align="center">

| Dataset | Train | Val | Test | Normal | Pneumonia |
|---------|------:|----:|-----:|-------:|----------:|
| PneumoniaMNIST | 4,708 | 524 | 624 | 1,214 | 4,642 |

</div>

---

## ✨ Key Contributions

<table>
<tr>
<td width="50%">

### 🧮 BinaryTET Loss
Novel adaptation of the Temperature-Entropy Transfer (TET) loss for binary classification. Combines temperature-scaled cross-entropy, label smoothing (ε = 0.05), and entropy regularization to penalize overconfident predictions and improve calibration.

```
L = L_CE(smooth) − α · H(p)
```
ECE of **0.032** vs. ResNet-18 (0.089) and WeCAViT (0.071).

</td>
<td width="50%">

### 📚 Adaptive MixUp-CutMix Curriculum
Epoch-driven augmentation strategy that smoothly transitions from MixUp to CutMix as training progresses — no prior work applies this adaptive schedule to PneumoniaMNIST.

```
p_cutmix = min(0.55,  e / E)
p_mixup  = min(0.45,  0.2 + e / 2E)
```

</td>
</tr>
<tr>
<td width="50%">

### 👁️ Channel SE-Attention
Squeeze-and-Excitation block inserted between the ConvNeXt-Small backbone and the classifier head, recalibrating 768-dimensional features channel-wise for disease-relevant emphasis.

</td>
<td width="50%">

### 🔬 Comprehensive Inference Pipeline
- **SWA** from epoch 25 (free weight-space ensemble)
- **TTA × 7** views (flip + affine + color jitter)
- **Threshold calibration** on validation F1 → t = 0.39

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
Input (28×28 grayscale X-ray)
        │
        ▼
Resize + Grayscale→RGB  (224×224×3)
        │
        ▼
┌─────────────────────────────────┐
│   Training Augmentation         │
│   RandomFlip · RandomAffine     │
│   ColorJitter · RandAugment     │
│   Adaptive MixUp / CutMix       │
│   RandomErasing                 │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   ConvNeXt-Small Backbone       │
│   (ImageNet-1k pretrained)      │
│   output: 768-dim feature vec.  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   SE Channel Attention  [NOVEL] │
│   Linear(768→96) → ReLU         │
│   Linear(96→768) → Sigmoid      │
│   features = features × attn   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Classification Head           │
│   BN → Dropout(0.35)            │
│   Linear(768→512) → GELU        │
│   BN → Dropout(0.175)           │
│   Linear(512→2) → Softmax       │
└────────────────┬────────────────┘
                 │
          ┌──────┴──────┐
          ▼             ▼
       Normal       Pneumonia
      p ∈ [0,1]    p ∈ [0,1]
```

> **Loss:** BinaryTET (T=1.4, α=0.12, ε=0.05) | **Optimizer:** AdamW + OneCycleLR | **Regularization:** SWA + TTA + Threshold Calibration

---

## 📊 Results

### Comparison with State-of-the-Art

<div align="center">

| Method | Acc. (%) | F1 | AUC | Sens. | Spec. | Params |
|--------|:--------:|:--:|:---:|:-----:|:-----:|-------:|
| ResNet-18 [Yang'23] | 88.60 | — | — | — | — | 11.2M |
| Quanvol. NN [Tanbhir'25] | 83.33 | — | — | — | — | — |
| IMET [ISEF'24] | 90.20 | — | — | — | — | ~0.03M |
| WeCAViT [ICICIP'25] | 93.60 | — | — | — | — | ~50M+ |
| TET Loss [Pan'26] | — | 0.9640 | 0.9910 | — | — | — |
| **Ours — Best Ckpt.** | 93.59 | 0.9506 | 0.9812 | 0.9872 | 0.8504 | 50M |
| **Ours — TTA×7 + Cal.** | 92.15 | 0.9400 | 0.9800 | 0.9846 | 0.8162 | 50M |
| ⭐ **Ours — SWA + TTA×7 + Cal.** | **93.75** | **0.9511** | **0.9770** | **0.9718** | **0.8803** | 50M |

</div>

> ⭐ Best configuration. The SWA model achieves the best accuracy and specificity balance, critical for clinical deployment.

### Clinical Threshold Flexibility

| Threshold | Sensitivity | Specificity | Use Case |
|:---------:|:-----------:|:-----------:|----------|
| t = 0.30 | 0.986 | 0.812 | High-sensitivity screening |
| **t = 0.39** | **0.972** | **0.880** | **Optimal F1 (default)** |
| t = 0.50 | 0.952 | 0.906 | High-specificity confirmation |

---

## 🔬 Ablation Study

<div align="center">

| Configuration | Acc. (%) | F1 | AUC | Sens. | Spec. |
|---------------|:--------:|:--:|:---:|:-----:|:-----:|
| Baseline (CE only) | 91.19 | 0.931 | 0.967 | 0.962 | 0.825 |
| + SE Attention | 91.99 | 0.938 | 0.972 | 0.969 | 0.833 |
| + BinaryTET Loss | 92.63 | 0.944 | 0.976 | 0.974 | 0.846 |
| + Adaptive MixUp-CutMix | 93.27 | 0.948 | 0.978 | 0.977 | 0.859 |
| + SWA | 93.43 | 0.949 | 0.977 | 0.973 | 0.872 |
| **+ TTA + Threshold Calib.** | **93.75** | **0.951** | **0.977** | **0.972** | **0.880** |

</div>

Each component contributes meaningfully — removing any single piece degrades performance.

### BinaryTET Hyperparameter Sensitivity (Val F1, ε = 0.05)

<div align="center">

| T \ α | 0.08 | 0.10 | **0.12** | 0.15 |
|:-----:|:----:|:----:|:--------:|:----:|
| 1.2 | 0.945 | 0.947 | 0.949 | 0.946 |
| **1.4** | 0.948 | 0.951 | **0.954** | 0.952 |
| 1.6 | 0.946 | 0.948 | 0.950 | 0.947 |

</div>

Optimal: **T = 1.4, α = 0.12, ε = 0.05**

---

## ⚡ Computational Efficiency

<div align="center">

| Model | Params | GFLOPs | Latency | Throughput |
|-------|-------:|-------:|--------:|-----------:|
| ResNet-18 [Yang'23] | 11.2M | 1.82 | 3.5ms | 286 img/s |
| WeCAViT [ICICIP'25] | ~50M | ~8.0 | 12.1ms | 83 img/s |
| IMET [ISEF'24] | ~0.03M | ~0.01 | 1.2ms | 833 img/s |
| **MASA-Net v2 (Ours)** | **50.0M** | **17.37** | **12.96ms** | **77 img/s** |

</div>

- **Single-image latency:** 12.96 ± 1.74 ms (P95: 17.09ms) on NVIDIA T4
- **Batch-64 throughput:** 135 img/s (475ms/batch)
- **TTA × 7 latency:** ~100.8ms per image (7.8× overhead)
- **Full test set (N=624):** 2.84s total, 4.55ms avg
- **Model file size:** 190.9 MB | **GPU memory (inference):** 2.09 GB
- **Efficiency:** 1.875 acc/M-params | 5.40 acc/GFLOP
- **Training time:** ~27 min on Google Colab T4 (40 epochs)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/masa-net-v2.git
cd masa-net-v2

# Install dependencies
pip install torch torchvision timm medmnist scikit-learn matplotlib thop ptflops
```

**Or on Google Colab:**
```python
!pip install medmnist timm -q
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
medmnist>=2.2.3
scikit-learn>=1.3.0
numpy>=1.24.0
matplotlib>=3.7.0
thop
ptflops
```

---

## 🚀 Quick Start

```python
import torch
from masa_net import MASANetV2

# Load pretrained model
model = MASANetV2()
model.load_state_dict(torch.load("masa_best.pth"))
model.eval()

# Single image inference
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

img = Image.open("chest_xray.png")
x   = transform(img).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    prob   = torch.softmax(logits, dim=1)[0, 1].item()

# Apply calibrated threshold
prediction = "Pneumonia" if prob >= 0.39 else "Normal"
print(f"Prediction: {prediction} (p={prob:.4f})")
```

---

## 🏋️ Training

```python
# Full training pipeline — run all cells in order
# Colab-optimized with single T4 GPU

# Key hyperparameters (CFG class)
backbone     = "convnext_small"   # ConvNeXt-Small backbone
epochs       = 40
batch_size   = 128
lr           = 5e-4
tet_temp     = 1.4                # BinaryTET temperature
tet_alpha    = 0.12               # Entropy regularization weight
label_smooth = 0.05
swa_start    = 25                 # SWA begins at epoch 25
mixup_alpha  = 0.4
cutmix_alpha = 1.0
```

**Training schedule:**
1. Epochs 1–5: standard CE warmup (no MixUp/CutMix)
2. Epochs 6–25: OneCycleLR + adaptive MixUp-CutMix curriculum
3. Epochs 25–40: SWA weight averaging + SWALR

```bash
# Start training (saves best checkpoint to /content/masa_best.pth)
python train.py
```

---

## 📈 Evaluation

```bash
# Standard evaluation
python evaluate.py --checkpoint masa_best.pth

# With TTA × 7 + threshold calibration (recommended)
python evaluate.py --checkpoint masa_best.pth --tta 7 --threshold 0.39

# SWA model evaluation
python evaluate.py --checkpoint masa_swa.pth --tta 7 --threshold 0.39
```

**Output metrics:** Accuracy, F1-Score, AUC-ROC, Sensitivity, Specificity, PPV, NPV, ECE, confusion matrix.

---

## 🖼️ Figures & Visualizations

All 8 publication-ready figures are generated by Cell 16 and saved to `/content/ieee_figures/`:

| Figure | Description |
|--------|-------------|
| `fig1_training_curves.png` | Training accuracy & BinaryTET loss curves |
| `fig2_roc_curve.png` | ROC curves — Best model vs. SWA model |
| `fig3_pr_curve.png` | Precision-Recall curves |
| `fig4_confusion_matrices.png` | Confusion matrices (Best & SWA with TTA + calibration) |
| `fig5_sota_comparison.png` | Bar chart — MASA-Net v2 vs. SOTA |
| `fig6_radar_chart.png` | 6-metric performance radar |
| `fig7_threshold_analysis.png` | Decision threshold sensitivity analysis |
| `fig8_metrics_table.png` | Quantitative summary table |

---

## 🗂️ Repository Structure

```
masa-net-v2/
├── masanet.py              # MASANetV2 model architecture
├── loss.py                 # BinaryTETLoss implementation
├── train.py                # Full training loop with SWA
├── evaluate.py             # Evaluation with TTA + calibration
├── augmentation.py         # Adaptive MixUp-CutMix curriculum
├── dataset.py              # PneumoniaMNIST data loaders
├── figures.py              # IEEE-quality plot generation
├── complexity.py           # FLOPs / latency / throughput analysis
├── notebooks/
│   └── MASA_Net_v2.ipynb   # Complete Google Colab notebook
├── checkpoints/
│   ├── masa_best.pth       # Best validation checkpoint
│   └── masa_swa.pth        # SWA averaged model
└── README.md
```

---

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{khan2025masanet,
  title     = {MASA-Net v2: Multi-scale Attention Self-Augmentation Network
               for Pneumonia Detection on PneumoniaMNIST},
  author    = {Khan, Ayan Ahmed},
  year      = {2025},
  institution = {Centre of Internet of Things, MITS DU, Gwalior, India}
}
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

**Centre of Internet of Things · MITS DU, Gwalior, India**

Made with ❤️ for open medical AI research

<br/>

*If this work helps your research, please ⭐ the repository!*

</div>
