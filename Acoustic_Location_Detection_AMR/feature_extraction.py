import librosa
import numpy as np
from scipy.stats import skew

AMR_FMIN = 300
AMR_FMAX = 3400


def extract_features(audio, sr):

    # Short Time Fourier Transform (STFT)

    stft = librosa.stft(
        y=audio,
        n_fft=512,
        hop_length=160,
        window="hann"
    )
    
    # Power Spectrogram

    power_spec = np.abs(stft) ** 2

    # Mel Spectrogram (restricted to AMR-NB's narrowband passband)

    mel_spec = librosa.feature.melspectrogram(
        S=power_spec,
        sr=sr,
        n_mels=40,
        fmin=AMR_FMIN,
        fmax=AMR_FMAX
    )

    # Log Mel Spectrogram

    log_mel = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # MFCC

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=20
    )

    # Statistical Feature Extraction

    features = []

    for coeff in mfcc:

        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(skew(coeff))
        features.append(np.percentile(coeff, 10))
        features.append(np.percentile(coeff, 90))

    # Global Audio Features

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # Spectral Centroid
    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr
        )
    )

    # Spectral Flatness
    flatness = np.mean(
        librosa.feature.spectral_flatness(
            y=audio
        )
    )

    # RMS Energy
    rms = np.mean(
        librosa.feature.rms(
            y=audio
        )
    )

    features.extend([
        zcr,
        centroid,
        flatness,
        rms
    ])

    return features
