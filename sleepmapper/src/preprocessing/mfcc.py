import librosa
import numpy as np
import torch
import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """
    Load configuration from a YAML file.
    """
    try:
        # Resolve path relative to project root if needed
        # Assuming script is run from project root
        if not os.path.exists(config_path):
            # Try to find it relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(base_dir, "configs", "config.yaml")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        return {}

def extract_mfcc_features(audio, n_mfcc=40):
    """
    Extract MFCC features along with delta and delta-delta coefficients.
    
    Args:
        audio (np.ndarray): Input audio signal.
        n_mfcc (int): Number of MFCC coefficients to extract.
        
    Returns:
        torch.Tensor: Feature tensor of shape (time_steps, 120).
    """
    try:
        if audio is None:
            raise ValueError("Input audio is None")

        # Load sample rate from config
        config = load_config()
        sr = config.get('sample_rate', 16000)
        
        # 1. Extract MFCCs
        # Using default n_fft and hop_length from librosa or could pull from config
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # 2. Compute Delta features
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 3. Compute Delta-Delta features
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate: (3 * n_mfcc, time_steps)
        combined = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)
        
        # Transpose to (time_steps, 3 * n_mfcc) -> (time_steps, 120)
        features = combined.T
        
        # Convert to torch tensor
        return torch.from_numpy(features).float()
        
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    dummy_audio = np.random.uniform(-1, 1, 16000 * 30) # 30s dummy audio
    features = extract_mfcc_features(dummy_audio)
    if features is not None:
        print(f"Feature shape: {features.shape}") # Expected (time_steps, 120)
