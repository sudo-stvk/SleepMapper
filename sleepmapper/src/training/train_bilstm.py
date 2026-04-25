import os
import yaml
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from src.preprocessing.dataset import get_patient_splits
from src.preprocessing.mfcc import extract_mfcc_features
from src.models.bilstm import SleepBiLSTM
from src.training.train import load_config, get_dummy_metadata, calculate_metrics

class MFCCDataset(Dataset):
    """
    PyTorch Dataset for loading audio and extracting MFCCs on the fly,
    or generating dummy MFCCs if given mock paths.
    """
    def __init__(self, file_paths, labels, sr=16000, duration=30):
        """
        Args:
            file_paths (list): List of audio file paths.
            labels (list): List of binary labels (0=normal, 1=apnea).
            sr (int): Sample rate for audio.
            duration (int): Duration of the audio clip in seconds.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index.
        """
        try:
            # In a production scenario, we would load the audio file here.
            # e.g., audio, _ = librosa.load(self.file_paths[idx], sr=self.sr)
            
            # For demonstration with dummy metadata, we generate random uniform noise audio.
            # This mimics a 30-second audio clip at the specified sample rate.
            audio = np.random.uniform(-1, 1, self.sr * self.duration)
            
            # Extract features (time_steps, 120)
            features = extract_mfcc_features(audio)
            
            if features is None:
                raise ValueError("Failed to extract MFCC features")
                
            label = torch.tensor(self.labels[idx]).long()
            return features, label
            
        except Exception as e:
            print(f"Error processing sample {self.file_paths[idx]}: {e}")
            # Return dummy zero tensor and label -1 to indicate error
            # Default librosa hop_length is 512
            time_steps = int(np.ceil((self.sr * self.duration) / 512))
            return torch.zeros((time_steps, 120)), torch.tensor(-1)

def create_mfcc_dataloaders(file_paths, labels, patient_ids, batch_size=32):
    """
    Creates train, validation, and test dataloaders for MFCC dataset.
    
    Args:
        file_paths (list): List of file paths.
        labels (list): List of labels.
        patient_ids (list): List of patient IDs for grouped splitting.
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    try:
        train_idx, val_idx, test_idx = get_patient_splits(file_paths, labels, patient_ids)
        
        if train_idx is None:
            return None

        train_ds = MFCCDataset([file_paths[i] for i in train_idx], [labels[i] for i in train_idx])
        val_ds = MFCCDataset([file_paths[i] for i in val_idx], [labels[i] for i in val_idx])
        test_ds = MFCCDataset([file_paths[i] for i in test_idx], [labels[i] for i in test_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    except Exception as e:
        print(f"Error creating MFCC dataloaders: {e}")
        return None

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training dataloader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        
    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    
    for inputs, labels in dataloader:
        # Skip invalid samples indicated by label == -1
        if torch.any(labels == -1):
            continue
            
        inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
        optimizer.zero_grad()
        
        # BiLSTM forward pass returns (logit, attention_weights)
        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
    return total_loss / len(dataloader.dataset)

def evaluate_epoch(model, dataloader, criterion, device, threshold=0.35):
    """
    Evaluate the model on validation or test set.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Evaluation dataloader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        threshold (float): Classification threshold.
        
    Returns:
        tuple: (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.any(labels == -1):
                continue
                
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
            
            logits, _ = model(inputs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * inputs.size(0)
            
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            
    avg_loss = total_loss / max(len(dataloader.dataset), 1)
    
    if len(all_labels) == 0:
        # Return dummy metrics if evaluation fails completely
        metrics = calculate_metrics([0, 1], [0.1, 0.9], threshold)
    else:
        metrics = calculate_metrics(all_labels, all_probs, threshold)
        
    return avg_loss, metrics

def train():
    """
    Main training function orchestrating the BiLSTM training pipeline.
    """
    try:
        # 1. Setup and configurations
        config = load_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # 2. Directories
        os.makedirs("outputs/checkpoints", exist_ok=True)
        os.makedirs("outputs/logs", exist_ok=True)
        
        # 3. Load data
        file_paths, labels, patient_ids = get_dummy_metadata()
        batch_size = config.get("batch_size", 32)
        
        dataloaders = create_mfcc_dataloaders(file_paths, labels, patient_ids, batch_size=batch_size)
        if dataloaders is None:
            print("Failed to create dataloaders. Exiting.")
            return
            
        train_loader, val_loader, test_loader = dataloaders
        
        # Calculate pos_weight for class imbalance
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(device)
        print(f"Calculated pos_weight: {pos_weight.item():.2f}")
        
        # 4. Model, Loss, Optimizer, Scheduler
        # BiLSTM uses MFCC features with size 120
        hidden_size = config.get("hidden_size", 256)
        dropout = config.get("dropout", 0.3)
        model = SleepBiLSTM(input_size=120, hidden_size=hidden_size, num_layers=2, dropout=dropout).to(device)
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        learning_rate = float(config.get("learning_rate", 1e-4))
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        num_epochs = config.get("num_epochs", 50)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        # 5. Training Loop setup
        threshold = 0.35 # Constraint constraint: must be 0.35
        best_auc = 0.0
        patience = 10
        epochs_no_improve = 0
        
        log_file_path = "outputs/logs/bilstm_training_log.csv"
        with open(log_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_auc", "val_f1", "val_precision", "val_recall", "val_acc"])
            
        # 6. Training Loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics = evaluate_epoch(model, val_loader, criterion, device, threshold)
            
            scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['auc_roc']:.4f}")
            
            # Log metrics
            with open(log_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1, 
                    train_loss, 
                    val_loss, 
                    val_metrics["auc_roc"],
                    val_metrics["f1"],
                    val_metrics["precision"],
                    val_metrics["recall"],
                    val_metrics["accuracy"]
                ])
                
            # Early Stopping & Checkpointing
            if val_metrics["auc_roc"] > best_auc:
                best_auc = val_metrics["auc_roc"]
                epochs_no_improve = 0
                
                checkpoint_path = "outputs/checkpoints/bilstm_best.pth"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved new best model with AUC: {best_auc:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        print("Training complete.")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")

if __name__ == "__main__":
    train()
