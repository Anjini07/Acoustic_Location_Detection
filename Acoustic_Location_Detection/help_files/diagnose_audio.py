import librosa
import numpy as np
import matplotlib.pyplot as plt

def inspect_audio_quality(file_path):
    audio, sr = librosa.load(file_path, sr=16000, mono=True)

    print(f"File: {file_path}")
    print(f"  Duration: {len(audio)/sr:.1f}s")
    print(f"  Peak amplitude: {np.max(np.abs(audio)):.4f}  (near 1.0 = likely clipping)")
    print(f"  RMS (avg loudness): {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"  Silent fraction: {np.mean(np.abs(audio) < 0.01):.2%}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(audio)
    axes[0].set_title("Waveform")
    axes[1].specgram(audio, Fs=sr)
    axes[1].set_title("Spectrogram")
    plt.tight_layout()
    plt.savefig(f"Acoustic_Location_Detection/quality_check_{file_path.split('/')[-1]}.png")
    print(f"  Saved visual to quality_check_{file_path.split('/')[-1]}.png\n")


inspect_audio_quality("Acoustic_Location_Detection/incoming/Office_test1.wav")