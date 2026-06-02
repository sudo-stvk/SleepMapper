"""
SleepMapper — Multimodal Training Script (EfficientNet-B0 + SpO2 LSTM)
Uses Tao et al. (Scientific Data 2025) multimodal dataset.
Follows train.py conventions exactly — same optimizer, scheduler,
early stopping, checkpointing, and logging patterns.

Extras vs train.py:
  - Takes (mel, spo2, label) tuples instead of (image, label)
  - Differential LR on the audio encoder (same as train.py)
  - SpO2 encoder and fusion head have their own LR group
  - Logs both audio-only AUC and combined AUC per epoch for ablation tracking

Run:
    python -m src.training.train_multimodal <path_to_tao_data_root>
"""

import os
import csv
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast

from src.preprocessing.tao_dataset import TaoMultimodalDataset, build_tao_splits, build_psg_splits
from src.models.multimodal_cnn import MultimodalSleepMapper

# ─── CONFIG ───────────────────────────────────────────────────────────────────

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_metrics(y_true, y_pred_probs, threshold=0.35):
    """Same implementation as train.py and train_bilstm.py."""
    if len(np.unique(y_true)) == 1:
        auc_roc = 0.5
    else:
        auc_roc = roc_auc_score(y_true, y_pred_probs)

    y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    return {
        "auc_roc":   auc_roc,
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "accuracy":  accuracy_score(y_true, y_pred),
    }


def create_weighted_sampler(dataset):
    """
    WeightedRandomSampler for class-balanced batches.
    Mirrors the pattern in train.py / train_bilstm.py.
    dataset.labels must be a list of int (0 or 1) — TaoMultimodalDataset exposes this.
    """
    labels = np.array(dataset.labels)
    class_counts  = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ─── TRAIN EPOCH ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = 0.0

    for mel, spo2, labels in tqdm(loader, desc="  Train", leave=False):
        # Skip any batch that contains a load-error sample (label == -1)
        if torch.any(labels == -1):
            continue

        mel    = mel.to(device)
        spo2   = spo2.to(device)
        labels = labels.to(device).float()   # (batch,) — model returns (batch,) logit

        optimizer.zero_grad()

        with autocast(device.type, enabled=scaler is not None):
            logits = model(mel, spo2)            # (batch,)
            loss   = criterion(logits, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * mel.size(0)

    return total_loss / len(loader.dataset)


# ─── EVALUATE EPOCH ───────────────────────────────────────────────────────────

def evaluate_epoch(model, loader, criterion, device, threshold=0.35):
    """
    Returns val loss + metrics dict.
    Also returns audio_only_probs separately for the ablation AUC log.

    To compute audio-only AUC without a separate model, we run the audio encoder
    alone by zeroing out the SpO2 input. This gives a fair ablation because
    the audio branch weights are shared — we just suppress the SpO2 signal.
    """
    model.eval()
    total_loss     = 0.0
    all_labels     = []
    all_probs      = []
    audio_only_probs = []

    with torch.no_grad():
        for mel, spo2, labels in tqdm(loader, desc="  Val  ", leave=False):
            if torch.any(labels == -1):
                continue

            mel    = mel.to(device)
            spo2   = spo2.to(device)
            labels = labels.to(device).float()

            # Combined (audio + SpO2) forward pass
            logits = model(mel, spo2)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * mel.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

            # Audio-only ablation: zero the SpO2 tensor so LSTM sees a flat signal
            # This isolates the audio branch contribution without loading a second model
            spo2_zeroed  = torch.zeros_like(spo2)
            logits_audio = model(mel, spo2_zeroed)
            probs_audio  = torch.sigmoid(logits_audio).cpu().numpy()
            audio_only_probs.extend(probs_audio)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    metrics  = calculate_metrics(all_labels, all_probs, threshold)

    # Compute audio-only AUC for the ablation log
    if len(np.unique(all_labels)) > 1:
        metrics["audio_only_auc"] = roc_auc_score(all_labels, audio_only_probs)
    else:
        metrics["audio_only_auc"] = 0.5

    return avg_loss, metrics


# ─── MAIN TRAIN FUNCTION ──────────────────────────────────────────────────────

def train(data_root, dataset_type="psg"):
    """
    Args:
        data_root   : path to the dataset root
                      PSG  → sleepmapper/data/raw/psg/
                      Tao  → path to Tao et al. data root
        dataset_type: 'psg' (default) or 'tao'
    """
    config    = load_config()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = config.get("classification_threshold", 0.35)
    print(f"\nDevice: {device}")
    print(f"Dataset: {dataset_type.upper()} — {data_root}")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    # ── Patient-level split (never clip-level) ──
    if dataset_type == "psg":
        train_dirs, val_dirs, _ = build_psg_splits(data_root, random_state=42)
    else:
        train_dirs, val_dirs, _ = build_tao_splits(data_root, random_state=42)

    train_ds = TaoMultimodalDataset(train_dirs, split="train", augment=True, cache_audio=False)
    val_ds   = TaoMultimodalDataset(val_dirs,   split="val",   augment=False, cache_audio=False)

    # Weighted sampler: class_weights = 1.0 / class_counts (same as all other loaders)
    train_sampler = create_weighted_sampler(train_ds)

    # Dynamically select workers (Windows pagefile commitment limits shared memory, requiring 0 workers)
    num_workers = 0 if os.name == "nt" else 4

    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("batch_size", 32),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("batch_size", 32),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ── Loss: pos_weight from actual class ratio — never hardcode this ──
    labels_arr   = np.array(train_ds.labels)
    n_neg        = (labels_arr == 0).sum()
    n_pos        = (labels_arr == 1).sum()
    computed_pw  = n_neg / n_pos
    print(f"pos_weight: {computed_pw:.4f}  (Normal={n_neg}, Apnea={n_pos})")
    pos_weight = torch.tensor([computed_pw]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Model ──
    model = MultimodalSleepMapper(pretrained=True).to(device)

    # ── Differential LR — mirrors train.py exactly for the audio encoder ──
    # Groups: lower CNN blocks | upper CNN blocks | fusion/spo2 head
    lower_params  = []   # audio_encoder features.0–5
    upper_params  = []   # audio_encoder features.6+ and avgpool
    head_params   = []   # spo2_encoder + fusion_head

    for name, param in model.named_parameters():
        if "spo2_encoder" in name or "fusion_head" in name:
            head_params.append(param)
        elif any(f"features.{i}" in name for i in range(6)):
            lower_params.append(param)
        else:
            upper_params.append(param)

    optimizer = optim.AdamW([
        {"params": lower_params, "lr": 1e-5},   # lower CNN blocks: very slow (same as train.py)
        {"params": upper_params, "lr": 5e-5},   # upper CNN blocks: medium
        {"params": head_params,  "lr": 3e-4},   # spo2 encoder + fusion head: fast
    ], weight_decay=1e-4)

    # ── OneCycleLR with 10% warmup — identical to train.py ──
    num_epochs  = config.get("num_epochs", 50)
    total_steps = len(train_loader) * num_epochs
    scheduler   = OneCycleLR(
        optimizer,
        max_lr=[1e-5, 5e-5, 3e-4],
        total_steps=total_steps,
        pct_start=0.1,         # 10% warmup
        anneal_strategy="cos",
    )

    # ── Mixed-precision (AMP) — ~1.5–2× faster on CUDA GPUs ──
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        print("Mixed-precision (AMP) enabled")

    # ── Training loop ──
    best_auc          = 0.0
    patience          = 15
    epochs_no_improve = 0

    log_path = "outputs/logs/multimodal_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Extra columns vs train.py: audio_only_auc for ablation (Experiment A vs B)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "val_auc", "audio_only_auc",
            "val_f1", "val_precision", "val_recall", "val_accuracy",
        ])

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss            = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, threshold)

        print(f"  Train Loss    : {train_loss:.4f}")
        print(f"  Val Loss      : {val_loss:.4f}")
        print(f"  AUC (combined): {val_metrics['auc_roc']:.4f}")
        print(f"  AUC (audio Ø) : {val_metrics['audio_only_auc']:.4f}  "
              f"← ablation: audio-only contribution")
        print(f"  Recall        : {val_metrics['recall']:.4f}  "
              f"Precision: {val_metrics['precision']:.4f}  "
              f"F1: {val_metrics['f1']:.4f}  "
              f"Accuracy: {val_metrics['accuracy']:.4f}")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, val_loss,
                val_metrics["auc_roc"], val_metrics["audio_only_auc"],
                val_metrics["f1"], val_metrics["precision"],
                val_metrics["recall"], val_metrics["accuracy"],
            ])

        # Checkpoint on best combined AUC-ROC (never audio-only — that's ablation only)
        if val_metrics["auc_roc"] > best_auc:
            best_auc          = val_metrics["auc_roc"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), "outputs/checkpoints/best_multimodal.pth")
            print(f"  > New best AUC: {best_auc:.4f} - checkpoint saved")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\nTraining complete. Best Val AUC-ROC: {best_auc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SleepMapper multimodal training")
    parser.add_argument("data_root",    type=str,
                        help="Path to dataset root (PSG: data/raw/psg, Tao: Tao data root)")
    parser.add_argument("--dataset",    type=str, default="psg", choices=["psg", "tao"],
                        help="Dataset type: 'psg' (default) or 'tao'")
    parser.add_argument("--precompute", action="store_true",
                        help="Pre-compute mel cache to disk and exit (run once before training)")
    args = parser.parse_args()

    if args.precompute:
        if args.dataset == "psg":
            train_dirs, val_dirs, _ = build_psg_splits(args.data_root, random_state=42)
        else:
            train_dirs, val_dirs, _ = build_tao_splits(args.data_root, random_state=42)

        print("\n-- Pre-computing mel spectrograms to disk cache --")
        train_ds = TaoMultimodalDataset(train_dirs, split="train", augment=False)
        train_ds.precompute_cache()
        val_ds = TaoMultimodalDataset(val_dirs, split="val", augment=False)
        val_ds.precompute_cache()
        print("\nDone! Run again without --precompute to train at full speed.")
    else:
        train(args.data_root, dataset_type=args.dataset)
