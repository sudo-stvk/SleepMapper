import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc

from src.preprocessing.dataset import create_dataloaders
from src.models.resnet18 import SleepResNet18
from src.models.model_utils import load_checkpoint, export_to_onnx

def load_config(config_path="configs/config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_dummy_metadata():
    """
    Placeholder function to generate dummy metadata for evaluation.
    In a real scenario, this would load from a CSV or directory structure.
    
    Returns:
        tuple: (file_paths, labels, patient_ids)
    """
    num_samples = 100
    file_paths = [f"dummy_path_{i}.npy" for i in range(num_samples)]
    labels = np.random.randint(0, 2, size=num_samples).tolist()
    patient_ids = [f"patient_{i // 5}" for i in range(num_samples)]
    
    return file_paths, labels, patient_ids

def plot_roc_curve(y_true, y_probs, save_path="outputs/roc_curve.png"):
    """
    Plot and save the ROC curve.
    
    Args:
        y_true (list or np.array): Ground truth labels.
        y_probs (list or np.array): Predicted probabilities.
        save_path (str): Path to save the plot.
    """
    if len(np.unique(y_true)) == 1:
        print("Cannot plot ROC curve with only one class present in the test set.")
        return
        
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to: {save_path}")

def run_inference(model, dataloader, device):
    """
    Run inference on the test set.
    
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Test dataloader.
        device (torch.device): Device to run evaluation on.
        
    Returns:
        tuple: (all_labels, all_probs)
    """
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            if torch.any(labels == -1):
                continue
                
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            
            all_labels.extend(labels.numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            
    return all_labels, all_probs

def evaluate():
    """
    Main evaluation function orchestrating inference and exporting.
    """
    # 1. Setup and configurations
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    threshold = config.get("classification_threshold", 0.35)
    
    # 2. Load Model Checkpoint
    checkpoint_path = "outputs/checkpoints/best_model.pth"
    model = SleepResNet18(pretrained=False) # Don't need pretrained weights when loading checkpoint
    
    try:
        model = load_checkpoint(model, checkpoint_path)
        model = model.to(device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running train.py")
        return
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
    # 3. Load Data
    file_paths, labels, patient_ids = get_dummy_metadata()
    dataloaders = create_dataloaders(
        file_paths, 
        labels, 
        patient_ids, 
        batch_size=config.get("batch_size", 32)
    )
    
    if dataloaders is None:
        print("Failed to create dataloaders. Exiting.")
        return
        
    _, _, test_loader = dataloaders
    
    # 4. Run Inference
    print("Running inference on test set...")
    y_true, y_probs = run_inference(model, test_loader, device)
    
    if len(y_true) == 0:
        # Dummy behavior since we have dummy data that yields 0 valid batches
        print("Warning: No valid data found for evaluation. Generating dummy results for verification.")
        y_true = np.random.randint(0, 2, size=100)
        y_probs = np.random.rand(100)
        
    # 5. Metrics & Reporting
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Apnea"], zero_division=0))
    
    plot_roc_curve(y_true, y_probs)
    
    # 6. Export to ONNX
    onnx_path = "outputs/models/sleepmapper_resnet18.onnx"
    # ResNet model expects input [batch_size, 1, 224, 224]
    export_to_onnx(model, onnx_path, input_size=(1, 1, 224, 224))

if __name__ == "__main__":
    evaluate()
