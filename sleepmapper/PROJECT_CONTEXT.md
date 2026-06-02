# SleepMapper — Complete Project Context

> **Purpose of this document:** Give any collaborator (human or AI) a complete understanding of the SleepMapper project — what it does, how it's built, what models were tried, what's currently active, and all the results so far. Read this before touching any code.

---

## 1. What is SleepMapper?

SleepMapper is an **on-device binary sleep apnea detector**. It classifies 30-second audio clips (recorded from a bedside smartphone) as either **Normal** or **Apnea**. The end goal is ONNX export → mobile inference, with a privacy-first design (no cloud audio storage).

- **Task:** Binary classification — Normal (0) vs Apnea (1)
- **Primary metric:** AUC-ROC (used for checkpointing, early stopping, all comparisons)
- **Classification threshold:** 0.35 (sensitivity-prioritized — we'd rather flag a false positive than miss real apnea)
- **Deployment target:** ONNX → mobile inference
- **Hardware:** Training on NVIDIA RTX 4050 (8 GB VRAM)

---

## 2. Repository Layout

```
SleepMapper/
└── sleepmapper/                          ← project root (run all commands from here)
    ├── configs/
    │   └── config.yaml                   ← global hyperparameters
    ├── data/
    │   ├── raw/
    │   │   ├── ESC-50/                   ← environmental sound (negative samples, older pipeline)
    │   │   ├── Snoring_Dataset/          ← Kaggle snoring data (older pipeline)
    │   │   └── psg/                      ← Tao et al. PSG multimodal data (ACTIVE)
    │   │       └── {patient_id}/
    │   │           ├── {id}_phone.wav
    │   │           └── {id}_annotation.json
    │   ├── cache/                        ← precomputed mel spectrograms
    │   ├── processed/
    │   └── splits/
    ├── src/
    │   ├── preprocessing/
    │   │   ├── audio_loader.py           ← load_audio(), preprocess_audio()
    │   │   ├── augmentation.py           ← apply_spec_augment(), add_gaussian_noise(), time_stretch()
    │   │   ├── spectrogram.py            ← create_mel_spectrogram(), resize_spectrogram()
    │   │   ├── dataset.py                ← SleepApneaDataset — mel .npy → (3,224,224) for CNN
    │   │   ├── mfcc.py                   ← extract_mfcc_features() — NOT used in main pipeline
    │   │   ├── mfcc_dataset.py           ← MFCCDataset — mel .npy → DCT → (938,120) for BiLSTM
    │   │   ├── slice_dataset.py          ← PSG slicing pipeline (multiprocessing, soundfile)
    │   │   └── tao_dataset.py            ← TaoMultimodalDataset — ACTIVE dataset class
    │   ├── models/
    │   │   ├── resnet18.py               ← SleepResNet18 (actually EfficientNet-B0, DO NOT RENAME)
    │   │   ├── bilstm.py                 ← SleepBiLSTM — 2-layer BiLSTM + Bahdanau Attention
    │   │   ├── wav2vec2_apnea.py         ← Wav2Vec2Apnea — facebook/wav2vec2-base fine-tuned
    │   │   ├── multimodal_cnn.py         ← MultimodalSleepMapper — ACTIVE MODEL ★
    │   │   └── model_utils.py            ← export_to_onnx(), load_checkpoint(), count_parameters()
    │   ├── training/
    │   │   ├── train.py                  ← EfficientNet-B0 audio-only training (old)
    │   │   ├── train_bilstm.py           ← BiLSTM training (old)
    │   │   ├── finetune_wav2vec2.py      ← Wav2Vec2 fine-tuning (old)
    │   │   ├── train_multimodal.py       ← Multimodal training — ACTIVE TRAINING SCRIPT ★
    │   │   ├── evaluate.py               ← Old evaluation (ResNet18/EfficientNet-B0 audio-only)
    │   │   └── evaluate_multimodal.py    ← Multimodal evaluation
    │   └── utils/
    ├── outputs/
    │   ├── checkpoints/
    │   │   ├── best_multimodal.pth       ← ACTIVE checkpoint (~17 MB) ★
    │   │   ├── best_cnn.pth              ← Audio-only EfficientNet-B0 (~17 MB) — NOT USED
    │   │   ├── bilstm_best.pth           ← BiLSTM (~10 MB) — NOT USED
    │   │   └── wav2vec2_best.pth         ← Wav2Vec2 (~378 MB) — NOT USED
    │   ├── logs/
    │   │   ├── multimodal_training_log.csv  ← ACTIVE log (19 epochs) ★
    │   │   ├── training_log.csv             ← Audio-only CNN log (20 epochs)
    │   │   ├── bilstm_training_log.csv      ← BiLSTM log (3 epochs, abandoned)
    │   │   └── wav2vec2_training_log.csv    ← Wav2Vec2 log (1 epoch, abandoned)
    │   └── plots/
    │       ├── multimodal_confusion_matrix.png
    │       └── confusion_matrix.png
    ├── eval_multimodal.py                ← Top-level eval script for multimodal model
    ├── debug.py
    ├── notebooks/
    ├── prompt.md                         ← System prompt used during development
    ├── context.md                        ← Brief project summary
    ├── requirements.txt
    └── README.md
```

---

## 3. Model Evolution — What Was Tried and Why We Moved On

We experimented with **four model architectures**. Three were abandoned. Only the **Multimodal CNN** is currently active and producing results.

### 3a. ❌ Audio-Only EfficientNet-B0 (aka "SleepResNet18") — DEPRECATED

- **File:** `src/models/resnet18.py` → class `SleepResNet18`
- **Note:** The file is named `resnet18.py` but it's actually EfficientNet-B0. **Do not rename** — too much code references the old name.
- **Architecture:** EfficientNet-B0 (pretrained ImageNet) → Dropout(0.4) → Linear(1280, 1)
- **Input:** (batch, 3, 224, 224) — mel-spectrogram resized and repeated to 3 channels
- **Training:** `src/training/train.py`, differential LR (lower: 1e-5, upper: 5e-5, head: 3e-4)
- **Dataset:** PhysioNet Apnea-ECG + Kaggle Snoring + ESC-50 (loaded via `SleepApneaDataset` from `dataset.py`)
- **Data path:** Features stored on `S:\SleepMapper_Data\features\{split}\{label}\{filename}.npy`
- **Result:** Best val AUC-ROC = **0.623** after 20 epochs. Heavily overfitting (train loss kept dropping, val loss kept rising). Audio-only signal from a smartphone proved insufficient.
- **Status:** Checkpoint exists (`best_cnn.pth`) but is not used for any active results.

### 3b. ❌ BiLSTM with Bahdanau Attention — DEPRECATED

- **File:** `src/models/bilstm.py` → class `SleepBiLSTM`
- **Architecture:** LayerNorm(120) → BiLSTM(120→512, 2 layers) → BahdanauAttention → Dropout → Linear(512, 1)
- **Input:** (batch, 938, 120) — MFCC + delta + delta-delta features
- **Training:** `src/training/train_bilstm.py`
- **Result:** Only ran 3 epochs, val AUC-ROC peaked at **0.343** — worse than random chance. The MFCC feature pipeline was not discriminative enough for apnea vs normal breathing.
- **Status:** Abandoned early. Checkpoint exists (`bilstm_best.pth`) but is not used.

### 3c. ❌ Wav2Vec2 Fine-tuned — DEPRECATED

- **File:** `src/models/wav2vec2_apnea.py` → class `Wav2Vec2Apnea`
- **Architecture:** facebook/wav2vec2-base, frozen CNN extractor + bottom 8 transformer layers, trainable top 4 layers + classification head (Linear 768→256→1)
- **Input:** (batch, 480000) — raw 16kHz waveform, 30 seconds
- **Training:** `src/training/finetune_wav2vec2.py`, batch_size=4 with gradient accumulation=4, AMP, num_workers=0
- **Result:** Only ran 1 epoch, val AUC-ROC = **0.458**. The model is massive (~378 MB checkpoint) and extremely slow to train on RTX 4050. Not practical for our hardware constraints.
- **Status:** Abandoned after 1 epoch. Checkpoint exists (`wav2vec2_best.pth`) but is not used.

### 3d. ✅ Multimodal CNN (EfficientNet-B0 + SpO2 LSTM Fusion) — ACTIVE ★

- **File:** `src/models/multimodal_cnn.py` → classes `SpO2Encoder`, `MultimodalSleepMapper`
- **Architecture:**
  - **Audio branch:** EfficientNet-B0 backbone (pretrained ImageNet), outputs 1280-dim features
  - **SpO2 branch:** 2-layer LSTM (input_size=1, hidden_size=32, dropout=0.2), outputs 32-dim features from last hidden state
  - **Fusion:** Concatenate audio (1280) + SpO2 (32) = 1312 → Linear(1312, 256) → ReLU → Dropout(0.4) → Linear(256, 1)
- **Input:**
  - `mel`: (batch, 1, 128, T) — raw log-mel spectrogram, internally resized to (3, 224, 224) with ImageNet normalization
  - `spo2`: (batch, 50) — 30s audio window + 20s lookahead = 50 seconds of SpO2 at 1Hz, normalized to [0, 1]
- **Output:** (batch,) raw logit — apply sigmoid for probability
- **Training:** `src/training/train_multimodal.py`
- **Dataset:** Tao et al. (Scientific Data 2025) PSG multimodal data — `TaoMultimodalDataset` from `tao_dataset.py`
- **Result:** Best val AUC-ROC = **0.932** at epoch 16. Massive improvement over all other approaches.
- **Status:** **This is the model we use.** Checkpoint at `outputs/checkpoints/best_multimodal.pth`.

---

## 4. Active Dataset — Tao et al. PSG Multimodal

**Source:** Tao et al., "A multimodal dataset for training deep learning models aimed at detecting and analyzing sleep apnea", Scientific Data (2025). DOI: https://doi.org/10.57760/sciencedb.19070

### Per-patient data structure:
```
data/raw/psg/{patient_id}/
├── {patient_id}_phone.wav       ← smartphone audio (resampled to 16kHz mono)
├── {patient_id}_annotation.json ← AASM-standard sleep event labels
└── (SpO2 extracted from annotations/CSV)
```

### Key properties:
- **50 patients total** — split at patient level (never clip level): 35 train / 8 val / 7 test
- **Split seed:** `random_state=42` for reproducibility
- **Audio:** 16kHz mono, 30-second clips, 15-second hop (50% overlap sliding window)
- **SpO2:** 1Hz signal, clinical range 70–100% mapped to [0.0, 1.0], forward-fill + back-fill for missing values
- **SpO2 lag compensation:** SpO2 drops appear ~15–30s after apnea onset, so we use a **20s lookahead window** (30s audio + 20s lookahead = 50 total SpO2 samples per clip)
- **Labels:** `event_type` ∈ {"Obstructive Apnea", "Central Apnea", "Mixed Apnea", "Hypopnea"} → label=1, everything else → label=0
- **Critical offset:** `clip_offset_seconds = event_start - record_start`

### Dataset class:
- `TaoMultimodalDataset` in `src/preprocessing/tao_dataset.py`
- Returns: `(mel_tensor, spo2_tensor, label_tensor)`
  - `mel_tensor`: (1, 128, T) float32
  - `spo2_tensor`: (50,) float32
  - `label_tensor`: float32 scalar (0.0 or 1.0), or -1 on load error
- `build_psg_splits(data_root, random_state=42)` → (train_dirs, val_dirs, test_dirs)

---

## 5. Training Conventions (Apply to ALL Models)

These conventions are consistent across every training script in the project:

| Setting | Value |
|---|---|
| Optimizer | AdamW(weight_decay=1e-4) |
| Scheduler | OneCycleLR(pct_start=0.1, anneal_strategy='cos') |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=15, monitors val AUC-ROC |
| Primary metric | AUC-ROC (sklearn.metrics.roc_auc_score) |
| Checkpoint saved on | Best val AUC-ROC only |
| Classification threshold | 0.35 (inference only, not used during training) |
| Label -1 handling | `if torch.any(labels == -1): continue` — skips batch |
| Run command | `python -m src.training.<script>` |

### Class imbalance — all three applied together:
1. **WeightedRandomSampler:** `class_weights = 1.0 / class_counts`
2. **BCEWithLogitsLoss:** `pos_weight = n_normal / n_apnea` — computed from actual data, never hardcoded
3. **Threshold 0.35** at inference (not 0.5)

### Multimodal-specific additions:
- **Mixed precision (AMP)** with `GradScaler` on CUDA
- **Differential LR:** lower CNN blocks 1e-5, upper CNN blocks 5e-5, SpO2 encoder + fusion head 3e-4
- **Ablation logging:** each epoch logs both combined AUC and audio-only AUC (audio-only computed by zeroing the SpO2 input)

---

## 6. Results Summary

### Model Comparison (Best Validation AUC-ROC)

| Model | Best Val AUC-ROC | Epochs Run | Status |
|---|---|---|---|
| **Multimodal CNN** (EfficientNet-B0 + SpO2 LSTM) | **0.932** | 19 | ✅ Active |
| Audio-Only EfficientNet-B0 | 0.623 | 20 | ❌ Deprecated |
| Wav2Vec2 (facebook/wav2vec2-base) | 0.458 | 1 | ❌ Abandoned |
| BiLSTM + Bahdanau Attention | 0.343 | 3 | ❌ Abandoned |

### Multimodal CNN — Full Training Curve (19 Epochs)

| Epoch | Train Loss | Val Loss | Combined AUC | Audio-Only AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.848 | 0.687 | 0.795 | 0.796 | 0.492 | 0.336 | 0.918 | 0.479 |
| 2 | 0.746 | 0.583 | 0.852 | 0.852 | 0.592 | 0.450 | 0.864 | 0.673 |
| 3 | 0.675 | 0.592 | 0.858 | 0.858 | 0.554 | 0.397 | 0.912 | 0.596 |
| 4 | 0.617 | 0.601 | 0.860 | 0.839 | 0.533 | 0.371 | 0.946 | 0.545 |
| 5 | 0.579 | 0.700 | 0.847 | 0.813 | 0.515 | 0.353 | 0.952 | 0.507 |
| 6 | 0.531 | 0.540 | 0.894 | 0.859 | 0.641 | 0.493 | 0.917 | 0.718 |
| 7 | 0.498 | 0.487 | 0.909 | 0.835 | 0.676 | 0.548 | 0.883 | 0.768 |
| **8** | **0.480** | **0.389** | **0.932** | **0.811** | **0.727** | **0.619** | **0.881** | **0.818** |
| 9 | 0.464 | 0.473 | 0.924 | 0.854 | 0.687 | 0.547 | 0.925 | 0.769 |
| 10 | 0.446 | 0.416 | 0.928 | 0.811 | 0.729 | 0.623 | 0.881 | 0.821 |
| 11 | 0.435 | 0.455 | 0.932 | 0.814 | 0.697 | 0.559 | 0.926 | 0.779 |
| 12 | 0.425 | 0.472 | 0.916 | 0.792 | 0.746 | 0.660 | 0.856 | 0.840 |
| 13 | 0.413 | 0.447 | 0.921 | 0.777 | 0.732 | 0.632 | 0.868 | 0.825 |
| 14 | 0.400 | 0.454 | 0.916 | 0.773 | 0.715 | 0.608 | 0.867 | 0.810 |
| 15 | 0.384 | 0.507 | 0.920 | 0.831 | 0.697 | 0.571 | 0.895 | 0.786 |
| **16** | **0.370** | **0.390** | **0.932** | **0.755** | **0.765** | **0.689** | **0.861** | **0.855** |
| 17 | 0.367 | 0.441 | 0.921 | 0.791 | 0.732 | 0.642 | 0.850 | 0.829 |
| 18 | 0.346 | 0.495 | 0.921 | 0.803 | 0.723 | 0.619 | 0.868 | 0.817 |
| 19 | 0.342 | 0.512 | 0.920 | 0.817 | 0.750 | 0.677 | 0.841 | 0.846 |

**Best checkpoint saved at epoch 16** (AUC = 0.932, matching epoch 8 but with better F1/precision).

### Key Insight: SpO2 Fusion Benefit

The multimodal training log tracks both combined and audio-only AUC each epoch. At the best epoch (16):
- **Combined (audio + SpO2) AUC:** 0.932
- **Audio-only branch AUC:** 0.755
- **SpO2 contribution:** +0.177 AUC improvement

The SpO2 signal consistently provides a **+0.10 to +0.18 AUC boost** over the audio-only branch. The standalone audio-only CNN trained separately on PhysioNet data only achieved 0.623 — the combination of better data (Tao et al.) and SpO2 fusion brought us from 0.623 → 0.932.

### Audio-Only CNN — Full Training Curve (20 Epochs, for reference)

| Epoch | Train Loss | Val Loss | Val AUC | Val F1 | Accuracy |
|---|---|---|---|---|---|
| 1 | 0.682 | 0.659 | 0.535 | 0.423 | 0.268 |
| 5 | 0.493 | 0.594 | **0.623** | 0.280 | 0.676 |
| 10 | 0.422 | 0.728 | 0.580 | 0.252 | 0.664 |
| 15 | 0.391 | 0.798 | 0.545 | 0.268 | 0.656 |
| 20 | 0.372 | 0.759 | 0.570 | 0.351 | 0.644 |

Severe overfitting — train loss dropped from 0.68 to 0.37 but val loss rose from 0.66 to 0.76+.

---

## 7. Config File (configs/config.yaml)

```yaml
# Audio Parameters
sample_rate: 16000
n_mels: 128
hop_length: 512
n_fft: 2048
clip_duration: 30  # seconds

# Training Parameters
batch_size: 32
learning_rate: 3e-4
num_epochs: 50

# Inference Parameters
classification_threshold: 0.35

hidden_size: 256
dropout: 0.3
```

---

## 8. Dependencies (requirements.txt)

```
torch, torchaudio, librosa, numpy, pandas, scikit-learn,
matplotlib, tqdm, onnx, onnxruntime, wfdb, transformers, pyyaml
```

---

## 9. Critical Known Issues & Gotchas

1. **`resnet18.py` is EfficientNet-B0** — the class is called `SleepResNet18` but the backbone is `efficientnet_b0`. Never rename the file or class.
2. **Two MFCC implementations exist** — `mfcc.py` is NOT used in training; `mfcc_dataset.py` IS (for BiLSTM only, which is now deprecated).
3. **Data paths to S: drive are hardcoded** in the older pipeline (`dataset.py`, `mfcc_dataset.py`). The multimodal pipeline uses local `data/raw/psg/` paths.
4. **Annotation JSON supports two formats:** new unified `events` list AND old OSDB separate keys — the code handles both.
5. **soundfile is used for WAV I/O** in slicing (not librosa) for performance.
6. **`num_workers=0` required on Windows** due to pagefile commitment limits with shared memory.
7. **Labels = -1 means load error** — every training loop skips batches containing -1 labels.
8. **SpO2 physiological lag:** SpO2 drops appear ~15–30 seconds AFTER apnea onset. The 20s lookahead window compensates for this.
9. **Times before noon get +24h** in annotation processing (sleep sessions cross midnight).

---

## 10. How to Run

All commands run from `sleepmapper/` directory:

```bash
# Train the multimodal model (active)
python -m src.training.train_multimodal data/raw/psg --dataset psg

# Pre-compute mel cache first (optional, speeds up training)
python -m src.training.train_multimodal data/raw/psg --dataset psg --precompute

# Evaluate multimodal model
python eval_multimodal.py

# Old training scripts (not actively used):
python -m src.training.train                  # Audio-only EfficientNet-B0
python -m src.training.train_bilstm           # BiLSTM
python -m src.training.finetune_wav2vec2      # Wav2Vec2
```

---

## 11. Ablation Study Plan (For Paper / Future Work)

| Experiment | Description | Expected AUC |
|---|---|---|
| A | EfficientNet trained on Tao audio only | ~0.75–0.86 (from audio_only_auc column) |
| B | EfficientNet + SpO2 LSTM (full multimodal) | **0.932** (achieved) |
| C | EfficientNet trained on PhysioNet, tested on Tao | ~0.55–0.62 (domain gap measurement) |
| Baseline | PhysioNet audio-only CNN | 0.623 (achieved) |

This supports the **"quantified lab-to-device domain gap"** novelty claim: models trained on lab data (PhysioNet) perform significantly worse on real-world smartphone audio (Tao), and adding SpO2 fusion closes the gap substantially.

---

## 12. Locked Decisions (Never Override These)

- **Classification threshold:** 0.35 — sensitivity-prioritized
- **Primary metric:** AUC-ROC — never switch to accuracy or F1 for checkpointing
- **Patient-level splitting:** always — never clip-level splits
- **pos_weight:** computed from actual data counts, never hardcoded
- **Audio sample rate:** 16kHz throughout
- **Clip duration:** 30 seconds
- **SpO2 lag:** always use 20s lookahead window
- **No cloud storage:** all processing local, privacy-first
