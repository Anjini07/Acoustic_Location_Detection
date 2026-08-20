

import numpy as np
from audio_preprocessing import load_audio_any_format
from chunk_audio import SAMPLE_RATE, CLIP_SECONDS

for path in [
    "Acoustic_Location_Detection/raw_recordings/nature/yt_nature_001.wav",
    "Acoustic_Location_Detection/raw_recordings/cafeteria/yt_cafeteria_002.wav",
    "Acoustic_Location_Detection/raw_recordings/office/yt_office_000.wav",
    "Acoustic_Location_Detection/raw_recordings/traffic/yt_traffic_001.wav",
]:
    audio, sr = load_audio_any_format(path, SAMPLE_RATE)
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    print(f"{path}: overall RMS={rms:.5f}  peak={peak:.4f}")