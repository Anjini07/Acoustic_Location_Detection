import os
import glob
import numpy as np
import soundfile as sf

from audio_preprocessing import load_audio_any_format

RAW_DIR = "Acoustic_Location_Detection/raw_recordings"
TARGET_SR = 16000


def group_amr_files_by_session(location_dir):
    """Groups AMR files that share a common prefix (same recording
    session) so they can be concatenated in order. Assumes filenames
    sort correctly in chronological order (e.g. clip_001.amr, clip_002.amr)."""
    amr_files = sorted(glob.glob(os.path.join(location_dir, "*.amr")))

    sessions = {}
    for f in amr_files:
        base = os.path.splitext(os.path.basename(f))[0]
        # Strip trailing numeric suffix to find the session prefix
        # e.g. "Home1_clip003" -> "Home1_clip", or adjust this rule
        # to match your actual device's naming convention
        prefix = base.rstrip("0123456789_")
        sessions.setdefault(prefix, []).append(f)

    return sessions


def merge_session(file_list, output_path):
    """Concatenates a list of short AMR clips (in order) into one
    longer WAV file."""
    all_audio = []
    for f in file_list:
        audio, sr = load_audio_any_format(f, target_sr=TARGET_SR)
        all_audio.append(audio)

    merged = np.concatenate(all_audio)
    sf.write(output_path, merged, TARGET_SR)
    print(f"  Merged {len(file_list)} clips -> {output_path} ({len(merged)/TARGET_SR:.1f}s)")


def main():
    for location_name in os.listdir(RAW_DIR):
        location_dir = os.path.join(RAW_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        sessions = group_amr_files_by_session(location_dir)
        if not sessions:
            continue

        print(f"Location: {location_name}")
        for session_prefix, file_list in sessions.items():
            if len(file_list) < 2:
                continue  # nothing to merge, only one clip in this "session"

            output_path = os.path.join(location_dir, f"{session_prefix}_merged.wav")
            merge_session(file_list, output_path)


if __name__ == "__main__":
    main()