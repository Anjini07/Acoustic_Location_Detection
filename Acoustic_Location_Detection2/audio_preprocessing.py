import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Point pydub directly at the local ffmpeg copy bundled in this project --
# avoids any dependence on Windows PATH or terminal restarts.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_FFMPEG_PATH = os.path.join(_PROJECT_ROOT, "ffmpeg", "bin", "ffmpeg.exe")
_FFPROBE_PATH = os.path.join(_PROJECT_ROOT, "ffmpeg", "bin", "ffprobe.exe")

if os.path.exists(_FFMPEG_PATH):
    AudioSegment.converter = _FFMPEG_PATH
    AudioSegment.ffprobe = _FFPROBE_PATH
else:
    print(f"WARNING: local ffmpeg not found at {_FFMPEG_PATH} -- AMR loading will fail.")

TARGET_SR = 16000
VALID_EXTENSIONS = (".wav", ".amr")


def load_audio_any_format(file_path: str, target_sr: int = TARGET_SR):
    """
    Loads WAV via soundfile, or AMR via pydub -> ffmpeg.
    soundfile (libsndfile) cannot decode AMR at all -- that format has to go
    through pydub/ffmpeg instead, which is why this branches by extension.
    Returns mono float32 samples in [-1, 1] at target_sr.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".amr":
        seg = AudioSegment.from_file(file_path, format="amr")
        seg = seg.set_channels(1).set_frame_rate(target_sr)
        audio = np.array(seg.get_array_of_samples()).astype(np.float32)
        audio = audio / float(1 << (8 * seg.sample_width - 1))  # normalize 16-bit PCM to [-1, 1]
        sr = target_sr

    elif ext == ".wav":
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    else:
        raise ValueError(f"Unsupported audio format: {ext}. Expected .wav or .amr")

    return audio, sr


def preprocess_audio(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sr = load_audio_any_format(file_path, TARGET_SR)

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
