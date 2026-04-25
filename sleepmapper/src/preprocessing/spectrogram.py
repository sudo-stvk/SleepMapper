import librosa
import numpy as np
import torch
import torch.nn.functional as F

def create_mel_spectrogram(audio, sample_rate=16000, n_mels=128, hop_length=512, n_fft=2048):
    """
    Convert audio to a mel spectrogram with log scaling.
    
    Args:
        audio (np.ndarray): Input audio signal.
        sample_rate (int): Sample rate.
        n_mels (int): Number of Mel bands.
        hop_length (int): Hop length.
        n_fft (int): FFT size.
        
    Returns:
        np.ndarray: Log-mel spectrogram.
    """
    try:
        # Mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=sample_rate, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels
        )
        
        # Log scaling (convert power to dB)
        log_S = librosa.power_to_db(S, ref=np.max)
        
        return log_S
    except Exception as e:
        print(f"Error creating spectrogram: {e}")
        return None

def resize_spectrogram(spectrogram, target_shape=(224, 224)):
    """
    Resize spectrogram to target shape (e.g., 224x224 for ResNet).
    
    Args:
        spectrogram (np.ndarray): Input spectrogram.
        target_shape (tuple): Target (height, width).
        
    Returns:
        np.ndarray: Resized spectrogram.
    """
    try:
        # Convert to torch tensor and add batch/channel dims: [1, 1, H, W]
        tensor_spec = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)
        
        # Interpolate
        resized_spec = F.interpolate(
            tensor_spec, 
            size=target_shape, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to numpy and remove extra dims
        return resized_spec.squeeze().numpy()
    except Exception as e:
        print(f"Error resizing spectrogram: {e}")
        return None

def save_spectrogram(spectrogram, output_path):
    """
    Save spectrogram as a .npy file.
    
    Args:
        spectrogram (np.ndarray): Spectrogram to save.
        output_path (str): Target file path.
    """
    try:
        np.save(output_path, spectrogram)
    except Exception as e:
        print(f"Error saving spectrogram to {output_path}: {e}")
