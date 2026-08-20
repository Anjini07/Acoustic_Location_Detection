import librosa
import numpy as np
from scipy.stats import skew


def extract_features(audio, sr):

    # =====================================================
    # STEP 1 : Short Time Fourier Transform (STFT)
    # =====================================================

    stft = librosa.stft(
        y=audio,
        n_fft=512,
        hop_length=160,
        window="hann"
    )

    # =====================================================
    # STEP 2 : Power Spectrogram
    # =====================================================

    power_spec = np.abs(stft) ** 2

    # =====================================================
    # STEP 3 : Mel Spectrogram
    # =====================================================

    mel_spec = librosa.feature.melspectrogram(
        S=power_spec,
        sr=sr,
        n_mels=40
    )

    # =====================================================
    # STEP 4 : Log Mel Spectrogram
    # =====================================================

    log_mel = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # =====================================================
    # STEP 5 : MFCC
    # =====================================================

    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=20
    )

    # =====================================================
    # STEP 6 : Statistical Feature Extraction
    # =====================================================

    features = []

    for coeff in mfcc:

        features.append(np.mean(coeff))
        features.append(np.std(coeff))
        features.append(skew(coeff))
        features.append(np.percentile(coeff, 10))
        features.append(np.percentile(coeff, 90))

    # =====================================================
    # STEP 7 : Global Audio Features
    # =====================================================

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