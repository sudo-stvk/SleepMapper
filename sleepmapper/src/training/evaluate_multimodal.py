import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.preprocessing.tao_dataset import build_psg_splits, TaoMultimodalDataset
from src.models.multimodal_cnn import MultimodalSleepMapper
from src.training.train import load_config

def evaluate_multimodal():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = config.get("classification_threshold", 0.35)
    
    print(f"Loading checkpoint from outputs/checkpoints/best_multimodal.pth")
    model = MultimodalSleepMapper().to(device)
    
    try:
        model.load_state_dict(torch.load("outputs/checkpoints/best_multimodal.pth", map_location=device, weights_only=True))
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    model.eval()
    
    print("Building dataset...")
    train_dirs, val_dirs, test_dirs = build_psg_splits('data/raw/psg', random_state=42)
    test_ds = TaoMultimodalDataset(test_dirs, split="test", augment=False, cache_audio=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
    
    all_labels = []
    all_probs = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for mel, spo2, label in test_loader:
            if torch.any(label == -1):
                continue
                
            mel = mel.to(device)
            spo2 = spo2.to(device)
            
            logits = model(mel, spo2)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(label.numpy().flatten())
            
    if len(all_labels) == 0:
        print("No valid test data found.")
        return
        
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Apnea"], zero_division=0))
    
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), va='center', ha='center')
            
    ax.set_xticklabels([''] + ["Pred: Normal", "Pred: Apnea"])
    ax.set_yticklabels([''] + ["True: Normal", "True: Apnea"])
    plt.title(f"Confusion Matrix (Threshold: {threshold})", pad=20)
    
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots/confusion_matrix.png"
    plt.savefig(save_path)
    print(f"\nConfusion matrix saved to {save_path}")
    
if __name__ == "__main__":
    evaluate_multimodal()
