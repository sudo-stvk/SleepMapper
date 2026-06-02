"""
Evaluate the best multimodal checkpoint on the validation set.
Produces: confusion matrix plot + classification report.

Run:
    python eval_multimodal.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from torch.utils.data import DataLoader

from src.preprocessing.tao_dataset import TaoMultimodalDataset, build_psg_splits
from src.models.multimodal_cnn import MultimodalSleepMapper

CHECKPOINT = "outputs/checkpoints/best_multimodal.pth"
THRESHOLD  = 0.35
DATA_ROOT  = "data/raw/psg"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load validation data (same split as training) ──
    _, val_dirs, test_dirs = build_psg_splits(DATA_ROOT, random_state=42)

    val_ds = TaoMultimodalDataset(val_dirs, split="val", augment=False, cache_audio=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    # ── Load model ──
    model = MultimodalSleepMapper(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT}")

    # ── Inference ──
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for mel, spo2, labels in val_loader:
            if torch.any(labels == -1):
                continue
            mel    = mel.to(device)
            spo2   = spo2.to(device)
            logits = model(mel, spo2)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= THRESHOLD).astype(int)

    # ── Metrics ──
    auc_val = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    print(f"\nVal AUC-ROC: {auc_val:.4f}")
    print(f"Threshold:   {THRESHOLD}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels.astype(int), all_preds,
                                target_names=["Normal", "Apnea"], zero_division=0))

    # ── Confusion Matrix ──
    cm = confusion_matrix(all_labels.astype(int), all_preds)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Apnea"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Multimodal CNN — Confusion Matrix\nVal AUC: {auc_val:.4f}  |  Threshold: {THRESHOLD}", fontsize=13)

    # Add counts as annotation
    total = cm.sum()
    tn, fp, fn, tp = cm.ravel()
    summary = f"TN={tn}  FP={fp}  FN={fn}  TP={tp}  |  Total={total}"
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, style="italic", color="gray")

    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots/multimodal_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved: {save_path}")


if __name__ == "__main__":
    main()
