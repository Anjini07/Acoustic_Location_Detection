"""
feature_extraction.py
Lightweight, fast feature set computed per 5s window. Deliberately not a
104-dim MFCC vector -- these are frame-averaged scalar features, cheap to
compute and small to store (9 floats per window vs. a full spectral
fingerprint), which is what makes the streaming pipeline both fast and
storage-light.
"""

import numpy as np

EPS = 1e-10

FEATURE_NAMES = [
    "noise_floor_rms",
    "speech_rms",
    "snr_estimate_db",
    "spectral_flatness_mean",
    "spectral_centroid_mean",
    "zcr_mean",
    "zcr_std",
    "energy_dynamic_range",
    "band_energy_ratio_mean",
]

BAND_LOW_HZ = 300
BAND_HIGH_HZ = 1000


def frame_signal(signal: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))

    num_frames = 1 + (len(signal) - frame_len) // hop_len
    return np.stack([signal[i * hop_len: i * hop_len + frame_len] for i in range(num_frames)])


def _energy(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames ** 2, axis=1) + EPS)


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs, axis=1))
    return np.mean(crossings, axis=1) / 2.0


def _spectral_flatness(frames: np.ndarray) -> np.ndarray:
    window = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) + EPS
    geo_mean = np.exp(np.mean(np.log(spec), axis=1))
    arith_mean = np.mean(spec, axis=1)
    return geo_mean / arith_mean


def _spectral_centroid(frames: np.ndarray, sr: int) -> np.ndarray:
    window = np.hanning(frames.shape[1])
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) + EPS
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    return np.sum(spec * freqs, axis=1) / np.sum(spec, axis=1)


def _band_energy_ratio(frames: np.ndarray, sr: int, band_low: float = BAND_LOW_HZ,
                        band_high: float = BAND_HIGH_HZ) -> np.ndarray:
    window = np.hanning(frames.shape[1])
    power_spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 + EPS
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    return np.sum(power_spec[:, band_mask], axis=1) / np.sum(power_spec, axis=1)


def _voice_activity_mask(energy: np.ndarray) -> np.ndarray:
    threshold = np.percentile(energy, 60)
    return energy > threshold


def extract_features(signal: np.ndarray, sr: int) -> dict:
    frames = frame_signal(signal, sr)

    energy = _energy(frames)
    zcr = _zero_crossing_rate(frames)
    flatness = _spectral_flatness(frames)
    centroid = _spectral_centroid(frames, sr)
    band_ratio = _band_energy_ratio(frames, sr)

    voice_mask = _voice_activity_mask(energy)
    non_voice_mask = ~voice_mask
    if not np.any(non_voice_mask):
        non_voice_mask = np.ones_like(voice_mask, dtype=bool)
    if not np.any(voice_mask):
        voice_mask = np.ones_like(non_voice_mask, dtype=bool)

    noise_floor_rms = float(np.mean(energy[non_voice_mask]))
    speech_rms = float(np.mean(energy[voice_mask]))
    snr_estimate_db = float(20 * np.log10((speech_rms + EPS) / (noise_floor_rms + EPS)))

    return {
        "noise_floor_rms": noise_floor_rms,
        "speech_rms": speech_rms,
        "snr_estimate_db": snr_estimate_db,
        "spectral_flatness_mean": float(np.mean(flatness[non_voice_mask])),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "energy_dynamic_range": float(np.percentile(energy, 95) - np.percentile(energy, 5)),
        "band_energy_ratio_mean": float(np.mean(band_ratio[non_voice_mask])),
    }
