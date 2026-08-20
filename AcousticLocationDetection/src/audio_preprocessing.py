import os
import librosa
import numpy as np

TARGET_SR = 16000

def preprocess_audio(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = librosa.load(
        file_path,
        sr=TARGET_SR,
        mono=True
    )

    # Remove DC Offset
    audio = audio - np.mean(audio)

    # Pre-emphasis
    alpha = 0.97
    audio = np.append(
        audio[0],
        audio[1:] - alpha * audio[:-1]
    )

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    return audio, sr