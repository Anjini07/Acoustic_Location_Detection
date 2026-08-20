"""
simulate_amr_roundtrip.py
Encodes existing WAV recordings into REAL AMR files -- an actual AMR-NB
encode via ffmpeg's libopencore_amrnb, not a renamed WAV -- so they pick up
the codec's real voice-activity detection and narrowband (~300-3400Hz)
filtering before being used for training/testing. This is what actually
recreates the "loud and moderate look similar in AMR" problem instead of
sidestepping it with clean WAV data.

AMR-NB requires exactly 8000 Hz mono input -- anything else is resampled
first automatically. Output lands in --output_dir as genuine .amr files,
ready to feed straight into build_features_streaming.py exactly like any
other real AMR recording from the device -- no separate "decode back" step
needed, since load_audio_any_format() already decodes AMR on read.

IMPORTANT: only use this to round-trip files you plan to use for TRAINING
(e.g. bulking up your under-represented 'noisy' class). Your held-out TEST
set should stay genuine, device-recorded AMR only -- round-tripped WAV
approximates real AMR degradation closely, but "closely" isn't the same as
"actually came from the device", and test numbers should reflect the real
thing.
"""

import os
import glob
import argparse
from pydub import AudioSegment

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_FFMPEG_PATH = os.path.join(_PROJECT_ROOT, "ffmpeg", "bin", "ffmpeg.exe")
_FFPROBE_PATH = os.path.join(_PROJECT_ROOT, "ffmpeg", "bin", "ffprobe.exe")

if os.path.exists(_FFMPEG_PATH):
    AudioSegment.converter = _FFMPEG_PATH
    AudioSegment.ffprobe = _FFPROBE_PATH
else:
    print(f"WARNING: local ffmpeg not found at {_FFMPEG_PATH} -- AMR encoding will fail.")

AMR_SAMPLE_RATE = 8000  # AMR-NB is fixed at 8kHz -- not configurable, codec requirement
AMR_BITRATE = "12.2k"   # highest-quality AMR-NB mode (least aggressive compression of the valid modes)


def roundtrip_to_amr(wav_path: str, output_dir: str) -> str:
    """Real encode: WAV -> AMR-NB via ffmpeg. The resulting .amr file has
    actually been through the codec's VAD + narrowband filtering, unlike a
    simple resample or bandpass-filter simulation."""
    seg = AudioSegment.from_file(wav_path)
    seg = seg.set_channels(1).set_frame_rate(AMR_SAMPLE_RATE)

    base = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_dir, f"{base}_amr.amr")

    seg.export(out_path, format="amr", bitrate=AMR_BITRATE)
    return out_path


def main(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))

    if not wav_files:
        raise ValueError(f"No .wav files found in {input_dir}")

    written = 0
    for wav_path in wav_files:
        try:
            out_path = roundtrip_to_amr(wav_path, output_dir)
            print(f"  [ok] {os.path.basename(wav_path)} -> {os.path.basename(out_path)}")
            written += 1
        except Exception as e:
            print(f"  [skip] {wav_path} -- {e}")

    print(f"\nRound-tripped {written}/{len(wav_files)} file(s) to real AMR in {output_dir}")
    print("These are genuinely AMR-degraded now -- feed them into build_features_streaming.py "
          "like any other .amr recording. Keep them out of your held-out test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-trip WAV files through a real AMR-NB encode")
    parser.add_argument("--input_dir", required=True, help="Acoustic_Location_Detection/raw_recordings/metro")
    parser.add_argument("--output_dir", required=True, help="Acoustic_Location_Detection2/raw_recordings_unlabeled")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
