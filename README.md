<div align="center">

# 🌙 SleepMapper

**On-device binary sleep apnea detection using multimodal deep learning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

SleepMapper classifies 30-second audio clips recorded from a bedside smartphone as **Normal** or **Apnea** by fusing mel-spectrogram features with blood oxygen (SpO₂) signals — achieving **0.932 AUC-ROC** on held-out patient data.

[Architecture](#architecture) · [Quick Start](#quick-start) · [Results](#results) · [Dataset](#dataset) · [Citation](#citation)

</div>

---

## Highlights

- 🎯 **0.932 AUC-ROC** on patient-level validation split (no data leakage)
- 🔬 **Multimodal fusion** — EfficientNet-B0 (audio) + LSTM (SpO₂) outperforms audio-only by **+0.18 AUC**
- 📱 **Privacy-first** — designed for ONNX export and on-device mobile inference, no cloud audio storage
- ⚡ **Mixed-precision training** with automatic mel-spectrogram caching for fast iteration
- 🏥 **Clinically motivated** — sensitivity-prioritized threshold (0.35) to minimize missed apnea events

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MultimodalSleepMapper                         │
│                                                                  │
│   ┌─────────────────────┐     ┌──────────────────────┐          │
│   │   Audio Branch       │     │    SpO₂ Branch        │          │
│   │                     │     │                      │          │
│   │  Mel Spectrogram    │     │  50-sample SpO₂      │          │
│   │  (1, 128, T)        │     │  (1Hz × 50s window)  │          │
│   │       ↓             │     │       ↓              │          │
│   │  Resize → (3,224,224)│     │  2-layer LSTM        │          │
│   │  + ImageNet Norm    │     │  (hidden=32)         │          │
│   │       ↓             │     │       ↓              │          │
│   │  EfficientNet-B0    │     │  Last hidden state   │          │
│   │  (pretrained)       │     │       ↓              │          │
│   │       ↓             │     │  (batch, 32)         │          │
│   │  (batch, 1280)      │     │                      │          │
│   └────────┬────────────┘     └──────────┬───────────┘          │
│            │           Concatenate        │                      │
│            └──────────┬──────────────────┘                      │
│                       ↓                                          │
│              (batch, 1312)                                       │
│                       ↓                                          │
│           Linear(1312, 256) → ReLU → Dropout(0.4)               │
│                       ↓                                          │
│              Linear(256, 1) → raw logit                          │
└──────────────────────────────────────────────────────────────────┘
```

The SpO₂ branch uses a **20-second lookahead window** (30s audio + 20s ahead = 50 samples) to compensate for the ~15–30s physiological lag between apnea onset and blood oxygen desaturation.

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (trained on NVIDIA RTX 4050, 8 GB VRAM)

### Installation

```bash
https://github.com/sudo-stvk/SleepMapper.git
cd SleepMapper/sleepmapper
pip install -r requirements.txt
```

### Data Setup

Place PSG patient data under `data/raw/psg/` with the following structure:

```
data/raw/psg/
├── <patient_id>/
│   ├── <patient_id>_phone.wav           # 16kHz mono smartphone audio
│   ├── <patient_id>_annotation.json     # AASM-standard sleep event labels
│   └── <patient_id>_SpO2.csv           # 1Hz SpO₂ readings
├── <patient_id>/
│   └── ...
```

### Training

```bash
# (Optional) Pre-compute mel cache to disk — speeds up all subsequent epochs
python -m src.training.train_multimodal data/raw/psg --dataset psg --precompute

# Train the multimodal model
python -m src.training.train_multimodal data/raw/psg --dataset psg
```

### Evaluation

```bash
python eval_multimodal.py
```

This loads `outputs/checkpoints/best_multimodal.pth`, runs inference on the validation set, and saves a confusion matrix to `outputs/plots/`.

---

## Results

### Model Comparison

| Model | Val AUC-ROC | Epochs | Status |
|:------|:----------:|:------:|:------:|
| **Multimodal CNN** (EfficientNet-B0 + SpO₂ LSTM) | **0.932** | 19 | ✅ Active |
| Audio-Only EfficientNet-B0 | 0.623 | 20 | Deprecated |
| Wav2Vec2 (fine-tuned) | 0.458 | 1 | Abandoned |
| BiLSTM + Attention | 0.343 | 3 | Abandoned |

### SpO₂ Fusion Ablation

At the best checkpoint (epoch 16), zeroing the SpO₂ branch during inference isolates each modality's contribution:

| Configuration | AUC-ROC |
|:---|:---:|
| Audio + SpO₂ (full model) | **0.932** |
| Audio-only (SpO₂ zeroed) | 0.755 |
| **SpO₂ contribution** | **+0.177** |

### Best Checkpoint Metrics (Epoch 16, threshold = 0.35)

| Metric | Value |
|:---|:---:|
| AUC-ROC | 0.932 |
| F1 Score | 0.765 |
| Precision | 0.689 |
| Recall | 0.861 |
| Accuracy | 0.855 |

---

## Repository Structure

```
sleepmapper/
├── configs/
│   └── config.yaml                        # Global hyperparameters
├── data/
│   ├── raw/psg/                           # Patient data (not committed)
│   └── cache/                             # Pre-computed mel spectrograms
├── src/
│   ├── preprocessing/
│   │   ├── tao_dataset.py                 # TaoMultimodalDataset + patient-level splits
│   │   └── slice_dataset.py              # PSG slicing pipeline (multiprocessing)
│   ├── models/
│   │   ├── multimodal_cnn.py              # MultimodalSleepMapper (active model)
│   │   └── model_utils.py                # ONNX export, checkpoint utilities
│   └── training/
│       ├── train_multimodal.py            # Training script (active)
│       └── evaluate_multimodal.py         # Evaluation utilities
├── eval_multimodal.py                     # Top-level evaluation entry point
├── requirements.txt
└── README.md
```

---

## Dataset

This project uses polysomnography (PSG) data from:

> **Tao et al.** "A multimodal dataset for training deep learning models aimed at detecting and analyzing sleep apnea", *Scientific Data* (2025).
> DOI: [10.57760/sciencedb.19070](https://doi.org/10.57760/sciencedb.19070)

**Key properties:**

| Property | Value |
|:---|:---|
| Patients | 50 (patient-level split: 35 train / 8 val / 7 test) |
| Audio | 16kHz mono, 30s clips with 15s hop (50% overlap) |
| SpO₂ | 1Hz, clinical range 70–100% normalized to [0, 1] |
| Labels | Obstructive / Central / Mixed Apnea, Hypopnea → 1; else → 0 |
| Split seed | `random_state=42` for reproducibility |

---

## Training Details

| Setting | Value |
|:---|:---|
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | OneCycleLR (10% warmup, cosine annealing) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | Patience 15, monitors val AUC-ROC |
| Mixed precision | AMP with GradScaler on CUDA |
| Batch size | 32 |

**Differential learning rates:**

| Parameter Group | Learning Rate |
|:---|:---:|
| Lower CNN blocks (EfficientNet features 0–5) | 1e-5 |
| Upper CNN blocks (features 6+) | 5e-5 |
| SpO₂ encoder + fusion head | 3e-4 |

**Class imbalance handling** (all three applied together):
1. `WeightedRandomSampler` — inverse class frequency sampling
2. `BCEWithLogitsLoss` with `pos_weight` computed from data (never hardcoded)
3. Classification threshold of 0.35 at inference (sensitivity-prioritized)

---

## Dependencies

```
torch, torchaudio, librosa, numpy, pandas, scikit-learn,
matplotlib, tqdm, onnx, onnxruntime, wfdb, transformers, pyyaml
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this work, please cite the underlying dataset:

```bibtex
@article{tao2025multimodal,
  title   = {A multimodal dataset for training deep learning models aimed at detecting and analyzing sleep apnea},
  author  = {Tao, Xueyu and others},
  journal = {Scientific Data},
  year    = {2025},
  doi     = {10.57760/sciencedb.19070}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
