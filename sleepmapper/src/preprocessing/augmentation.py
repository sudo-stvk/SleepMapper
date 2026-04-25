import numpy as np
import librosa
import random

def apply_spec_augment(spectrogram, max_time_mask=80, max_freq_mask=27):
    """
    Apply SpecAugment (time and frequency masking) to a spectrogram.
    
    Args:
        spectrogram (np.ndarray): Input spectrogram.
        max_time_mask (int): Maximum frames to mask in time.
        max_freq_mask (int): Maximum bins to mask in frequency.
        
    Returns:
        np.ndarray: Augmented spectrogram.
    """
    try:
        augmented = spectrogram.copy()
        n_mels, n_frames = augmented.shape
        
        # Frequency masking
        f = random.randint(0, max_freq_mask)
        f0 = random.randint(0, n_mels - f)
        augmented[f0:f0+f, :] = np.mean(augmented)
        
        # Time masking
        t = random.randint(0, max_time_mask)
        t0 = random.randint(0, n_frames - t)
        augmented[:, t0:t0+t] = np.mean(augmented)
        
        return augmented
    except Exception as e:
        print(f"Error applying SpecAugment: {e}")
        return spectrogram

def add_gaussian_noise(audio, noise_level=0.005):
    """
    Add Gaussian noise to audio signal.
    
    Args:
        audio (np.ndarray): Input audio signal.
        noise_level (float): Amplitude of noise.
        
    Returns:
        np.ndarray: Noisy audio signal.
    """
    try:
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_level * noise
        return augmented_audio
    except Exception as e:
        print(f"Error adding Gaussian noise: {e}")
        return audio

def time_stretch(audio, rate=None):
    """
    Stretch or compress audio signal in time.
    
    Args:
        audio (np.ndarray): Input audio signal.
        rate (float): Stretch factor (0.8 to 1.2). If None, random.
        
    Returns:
        np.ndarray: Time-stretched audio signal.
    """
    try:
        if rate is None:
            rate = random.uniform(0.8, 1.2)
        
        # librosa.effects.time_stretch changes the length
        # Note: If we want to maintain 30s length, we might need to pad/clip again
        # but usually stretching is done before padding/clipping or as a data augment step
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return stretched
    except Exception as e:
        print(f"Error applying time stretch: {e}")
        return audio
