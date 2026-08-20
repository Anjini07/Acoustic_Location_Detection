"""
extract_clip_for_listening.py
Decodes ONE source recording (named in the 'source_files' column of your
features CSV) into a throwaway WAV, purely so you can listen to it. This is
the deliberate tradeoff of the streaming pipeline: instead of persisting
every 5s window as its own file up front (the storage cost we're avoiding),
you decode on demand, only for the handful of files you actually want to
spot-check (e.g. the "most typical example per cluster" that
cluster_features.py prints).

Note: for a window built from merged short recordings, 'source_files' may
list more than one file (e.g. "short_0.wav+short_1.wav"). This tool previews
each named file in full rather than trying to reconstruct the exact
sub-second slice that went into that window -- reconstructing the precise
slice isn't worth the complexity for a spot-check whose only job is "does
this sound roughly like what the label says".

Usage:
    python3 extract_clip_for_listening.py raw_recordings/Cafe/some_file.amr
    python3 extract_clip_for_listening.py raw_recordings/Cafe/a.amr raw_recordings/Cafe/b.amr --output_dir previews/
"""

import os
import argparse
import soundfile as sf

from audio_preprocessing import load_audio_any_format


def extract(source_path: str, output_dir: str = ".") -> str:
    audio, sr = load_audio_any_format(source_path)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(source_path))[0]
    out_path = os.path.join(output_dir, f"{base}_preview.wav")
    sf.write(out_path, audio, sr)

    print(f"  {source_path} -> {out_path} ({len(audio)/sr:.2f}s) -- temporary, delete after listening")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode one or more raw recordings to WAV for spot-checking")
    parser.add_argument("source_files", nargs="+",
                         help="Path(s) to raw AMR/WAV file(s), e.g. from a 'source_files' CSV column "
                              "(split multi-file entries like 'a.amr+b.amr' on '+' first)")
    parser.add_argument("--output_dir", default=".", help="Where to write the preview WAV(s)")
    args = parser.parse_args()

    for source_path in args.source_files:
        extract(source_path, args.output_dir)
