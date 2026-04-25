import os
import sys
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pandas as pd
from tqdm import tqdm

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.wav2vec2_apnea import Wav2Vec2Apnea
from src.preprocessing.audio_loader import load_audio, preprocess_audio

class RawAudioDataset(Dataset):
    """
    Dataset for loading raw audio waveforms and applying preprocessing for Wav2Vec2.
    """
    def __init__(self, file_paths, labels, sample_rate=16000, clip_duration=30):
        """
        Initializes the dataset.
        
        Args:
            file_paths (list of str): Paths to audio files.
            labels (list of int/float): Binary labels (0 or 1).
            sample_rate (int): Target sample rate.
            clip_duration (int): Duration in seconds to clip or pad to.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Fetches and preprocesses the audio file at the given index.
        
        Returns:
            tuple: (audio_tensor, label_tensor)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        audio = load_audio(file_path, self.sample_rate)
        audio = preprocess_audio(audio, self.sample_rate, self.clip_duration)
        
        if audio is None:
            # Fallback to zero tensor if loading or preprocessing fails
            audio = np.zeros(int(self.sample_rate * self.clip_duration))
            
        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def load_config(config_path):
    """
    Loads hyperparameters from a YAML configuration file.
    
    Args:
        config_path (str): Path to the config.yaml file.
        
    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_metrics(y_true, y_prob, threshold=0.35):
    """
    Computes classification metrics using a specified threshold.
    
    Args:
        y_true (np.ndarray): True binary labels.
        y_prob (np.ndarray): Predicted probabilities.
        threshold (float): Classification threshold.
        
    Returns:
        tuple: (auc, f1, precision, recall)
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5  # Fallback if only one class is present in the batch
        
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    return auc, f1, precision, recall

def train_model(config_path, train_paths, train_labels, val_paths, val_labels):
    """
    Fine-tunes the Wav2Vec2 model according to the configuration and constraints.
    
    Args:
        config_path (str): Path to the config.yaml file.
        train_paths (list): List of paths to training audio files.
        train_labels (list): List of training labels.
        val_paths (list): List of paths to validation audio files.
        val_labels (list): List of validation labels.
    """
    config = load_config(config_path)
    
    # Extract hyperparameters from config
    sample_rate = config.get('sample_rate', 16000)
    clip_duration = config.get('clip_duration', 30)
    batch_size = config.get('batch_size', 32)
    num_epochs = config.get('num_epochs', 50)
    threshold = config.get('classification_threshold', 0.35)
    
    # Define directories for outputs
    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    
    # Initialize datasets and data loaders
    train_dataset = RawAudioDataset(train_paths, train_labels, sample_rate, clip_duration)
    val_dataset = RawAudioDataset(val_paths, val_labels, sample_rate, clip_duration)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2Apnea().to(device)
    
    print(f"Trainable parameters: {model.count_trainable_params()}")
    
    # Differentiate parameters for custom learning rates
    transformer_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'classification_head' in name:
            head_params.append(param)
        else:
            transformer_params.append(param)
            
    # Two different learning rates
    optimizer = optim.AdamW([
        {'params': transformer_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ])
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Warmup scheduler: 10% of total steps
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    best_val_auc = -1.0
    patience_counter = 0
    patience_limit = 10
    
    log_data = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_probs = []
        train_targets = []
        
        for batch_audio, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_audio = batch_audio.to(device)
            batch_labels = batch_labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            logits = model(batch_audio)
            
            loss = criterion(logits, batch_labels)
            loss.backward()
            
            # Gradient clipping with max_norm=1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            train_probs.extend(probs)
            train_targets.extend(batch_labels.cpu().numpy().flatten())
            
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for batch_audio, batch_labels in val_loader:
                batch_audio = batch_audio.to(device)
                batch_labels = batch_labels.to(device).unsqueeze(1)
                
                logits = model(batch_audio)
                loss = criterion(logits, batch_labels)
                
                val_loss += loss.item()
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_targets.extend(batch_labels.cpu().numpy().flatten())
                
        # Compute metrics
        train_auc, train_f1, train_prec, train_rec = compute_metrics(
            np.array(train_targets), np.array(train_probs), threshold
        )
        val_auc, val_f1, val_prec, val_rec = compute_metrics(
            np.array(val_targets), np.array(val_probs), threshold
        )
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f} | Train AUC: {train_auc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
        
        log_data.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss, 'train_auc': train_auc, 'train_f1': train_f1,
            'val_loss': avg_val_loss, 'val_auc': val_auc, 'val_f1': val_f1,
            'val_precision': val_prec, 'val_recall': val_rec
        })
        
        # Early Stopping and Checkpoint saving based on AUC-ROC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), 'outputs/checkpoints/wav2vec2_best.pth')
            print("Saved new best checkpoint!")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
                
    # Save training logs to CSV
    pd.DataFrame(log_data).to_csv('outputs/logs/wav2vec2_training_log.csv', index=False)
    print("Training finished. Logs saved.")

if __name__ == "__main__":
    # Placeholder for running the script directly
    # Adjust config path relative to the script
    # config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'))
    # train_model(config_path, ['dummy_train.wav'], [0], ['dummy_val.wav'], [1])
    pass
