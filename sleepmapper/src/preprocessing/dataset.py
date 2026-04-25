import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

class SleepApneaDataset(Dataset):
    """
    PyTorch Dataset for loading SleepMapper spectrograms and labels.
    """
    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list): List of paths to .npy spectrogram files.
            labels (list): List of binary labels (0=normal, 1=apnea).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            # Load spectrogram
            spectrogram = np.load(self.file_paths[idx])
            label = self.labels[idx]

            # Convert to float32 and add channel dimension for ResNet [1, 224, 224]
            spectrogram = torch.from_numpy(spectrogram).float().unsqueeze(0)
            label = torch.tensor(label).long()

            if self.transform:
                spectrogram = self.transform(spectrogram)

            return spectrogram, label
        except Exception as e:
            print(f"Error loading sample {self.file_paths[idx]}: {e}")
            # Return a zero tensor and label -1 as an error indicator (or handle differently)
            return torch.zeros((1, 224, 224)), torch.tensor(-1)

def get_patient_splits(file_paths, labels, patient_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Perform patient-level train/val/test split.
    
    Args:
        file_paths (list): List of file paths.
        labels (list): List of labels.
        patient_ids (list): List of patient IDs corresponding to each file.
        train_ratio (float): Ratio for training set.
        val_ratio (float): Ratio for validation set.
        test_ratio (float): Ratio for test set.
        
    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    try:
        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        # First split: Train vs (Val + Test)
        gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=42)
        train_idx, temp_idx = next(gss_train.split(file_paths, labels, groups=patient_ids))

        # Second split: Val vs Test from the temp set
        # Calculate relative ratio for validation
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        
        # We need to filter paths, labels, and patient_ids for the second split
        temp_paths = [file_paths[i] for i in temp_idx]
        temp_labels = [labels[i] for i in temp_idx]
        temp_patients = [patient_ids[i] for i in temp_idx]
        
        gss_val = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=42)
        val_rel_idx, test_rel_idx = next(gss_val.split(temp_paths, temp_labels, groups=temp_patients))
        
        # Map back to original indices
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]

        return train_idx, val_idx, test_idx
    except Exception as e:
        print(f"Error splitting dataset: {e}")
        return None, None, None

def create_dataloaders(file_paths, labels, patient_ids, batch_size=32):
    """
    Helper to create Train, Val, and Test DataLoaders.
    """
    train_idx, val_idx, test_idx = get_patient_splits(file_paths, labels, patient_ids)
    
    if train_idx is None:
        return None

    # Create datasets
    train_ds = SleepApneaDataset([file_paths[i] for i in train_idx], [labels[i] for i in train_idx])
    val_ds = SleepApneaDataset([file_paths[i] for i in val_idx], [labels[i] for i in val_idx])
    test_ds = SleepApneaDataset([file_paths[i] for i in test_idx], [labels[i] for i in test_idx])

    # Create loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
