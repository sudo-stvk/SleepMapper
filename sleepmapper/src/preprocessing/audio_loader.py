import librosa
import numpy as np
import os

def load_audio(file_path, sample_rate=16000):
    """
    Load a .wav file at a specific sample rate in mono.
    
    Args:
        file_path (str): Path to the audio file.
        sample_rate (int): Target sample rate. Default is 16000.
        
    Returns:
        np.ndarray: Loaded audio signal.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        return audio
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None

def preprocess_audio(audio, sample_rate=16000, duration=30):
    """
    Clip or pad audio to exactly a target duration and normalize amplitude.
    
    Args:
        audio (np.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the audio.
        duration (int): Target duration in seconds.
        
    Returns:
        np.ndarray: Preprocessed audio signal.
    """
    if audio is None:
        return None
        
    try:
        target_length = sample_rate * duration
        
        # Clip or pad
        if len(audio) > target_length:
            # Clip
            audio = audio[:target_length]
        else:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            
        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
            
        return audio
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None
