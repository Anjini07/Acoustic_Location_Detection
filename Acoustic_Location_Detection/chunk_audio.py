import os
import glob
import json
import argparse
import numpy as np
import soundfile as sf

from audio_preprocessing import load_audio_any_format, VALID_EXTENSIONS, TARGET_SR

CLIP_SECONDS = 5   # default window; override per-run with --clip-seconds
SAMPLE_RATE = TARGET_SR
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE
SILENCE_AMPLITUDE = 0.004
SILENCE_FRACTION_LIMIT = 0.92
CROSSFADE_SAMPLES = int(0.02 * SAMPLE_RATE)  # 20ms crossfade at file-merge seams only
DEAD_AIR_RMS = 0.0015

# CHANGE: raw source files can arrive at wildly different loudness levels
# (e.g. YouTube downloads mastered/normalized differently per video -- one
# location's source measured ~50x quieter in RMS than others, peaking at
# 0.047 vs 1.0). The fixed silence thresholds above assume roughly comparable
# loudness across files, so an entire quiet-but-real recording could get
# misclassified as dead air, chunk by chunk, wiping the location out of any
# downstream test/train split. Peak-normalizing right after load fixes the
# silence check without touching the raw files on disk, and keeps behavior
# for already-normal-volume files effectively unchanged (a file already near
# NORMALIZE_TARGET_PEAK barely moves).
NORMALIZE_TARGET_PEAK = 0.95
NORMALIZE_MIN_PEAK = 1e-6  # guard against a true all-zero/silent buffer


def normalize_audio(audio, target_peak=NORMALIZE_TARGET_PEAK):
    """Peak-normalizes audio to target_peak. A true silent/all-zero buffer
    is returned unchanged (nothing to scale, and is_mostly_silent's
    DEAD_AIR_RMS check will correctly still catch it)."""
    peak = np.max(np.abs(audio))
    if peak < NORMALIZE_MIN_PEAK:
        return audio
    return audio * (target_peak / peak)


def is_mostly_silent(audio, threshold=SILENCE_FRACTION_LIMIT, silence_amplitude=SILENCE_AMPLITUDE):
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < DEAD_AIR_RMS:
        return True  # genuinely dead air -- mic capped, no signal at all
    silent_fraction = np.mean(np.abs(audio) < silence_amplitude)
    return silent_fraction > threshold

def load_ledger():
    if os.path.exists(LEDGER_FILE):
        with open(LEDGER_FILE, "r") as f:
            return json.load(f)
    return {}


RAW_DIR = "Acoustic_Location_Detection/raw_recordings"
OUTPUT_BASE = "Acoustic_Location_Detection/dataset"
LEDGER_FILE = "Acoustic_Location_Detection/chunked_files.json"
CLIP_PROVENANCE_FILE = "Acoustic_Location_Detection/clip_provenance.json"


def save_ledger(ledger):
    with open(LEDGER_FILE, "w") as f:
        json.dump(ledger, f, indent=2)


def load_provenance():
    if os.path.exists(CLIP_PROVENANCE_FILE):
        with open(CLIP_PROVENANCE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_provenance(provenance):
    with open(CLIP_PROVENANCE_FILE, "w") as f:
        json.dump(provenance, f, indent=2)


def _crossfade_seam(buffer, seam_position, fade_samples=CROSSFADE_SAMPLES):
    """Applies a short linear crossfade centered on a file-merge seam inside
    the buffer, to remove the abrupt sample-level discontinuity that occurs
    when two unrelated recordings are concatenated directly. Only called at
    real seams (where a new raw file's audio was just appended), not inside
    a single file's own continuous audio."""
    start = max(0, seam_position - fade_samples // 2)
    end = min(len(buffer), seam_position + fade_samples // 2)
    n = end - start
    if n <= 1:
        return buffer
    fade = np.linspace(0, 1, n)
    pre = buffer[start:seam_position]
    post = buffer[seam_position:end]
    mid = min(len(pre), len(post))
    if mid > 0:
        buffer[seam_position - mid:seam_position] *= (1 - fade[:mid])
        buffer[seam_position:seam_position + mid] *= fade[:mid]
    return buffer


def chunk_location(location_name, raw_paths, output_dir, ledger, provenance,
                    skip_silent=True, clip_seconds=CLIP_SECONDS):
    """
    Processes every raw file for one location as a single continuous stream:
    long files get split into clip_seconds-long pieces, short files get merged
    with whatever's carried over from the previous file in the same location.
    Tracks which raw file(s) contributed to each output clip, and smooths
    the sample-level discontinuity at each file-merge seam.
    """
    os.makedirs(output_dir, exist_ok=True)
    clip_samples = clip_seconds * SAMPLE_RATE

    buffer = np.array([], dtype=np.float32)
    buffer_sources = []          # (source_file, num_samples_contributed) pairs still in buffer
    seam_positions = []          # sample indices in buffer where a new file started
    clip_index = 0
    written = 0
    skipped_silent = 0
    consumed_files = []

    for raw_path in raw_paths:
        try:
            audio, sr = load_audio_any_format(raw_path, SAMPLE_RATE)
            audio = normalize_audio(audio)
        except Exception as e:
            print(f"  [skip] {raw_path} -- {e}")
            continue

        if len(buffer) > 0:
            seam_positions.append(len(buffer))  # mark where this file's audio begins

        buffer = np.concatenate([buffer, audio])
        buffer_sources.append([raw_path, len(audio)])
        consumed_files.append(raw_path)

        # Smooth any seam that now falls inside the buffer
        for seam in seam_positions:
            buffer = _crossfade_seam(buffer, seam)
        seam_positions = [s for s in seam_positions if s >= clip_samples]  # drop seams already cut past

        while len(buffer) >= clip_samples:
            clip = buffer[:clip_samples]
            buffer = buffer[clip_samples:]
            seam_positions = [s - clip_samples for s in seam_positions if s - clip_samples >= 0]

            contributing = []
            remaining = clip_samples
            while buffer_sources and remaining > 0:
                src, n = buffer_sources[0]
                take = min(n, remaining)
                contributing.append(src)
                remaining -= take
                if take == n:
                    buffer_sources.pop(0)
                else:
                    buffer_sources[0][1] = n - take

            if skip_silent and is_mostly_silent(clip):
                skipped_silent += 1
            else:
                clip_name = f"{location_name}_{clip_index:04d}.wav"
                out_path = os.path.join(output_dir, clip_name)
                sf.write(out_path, clip, SAMPLE_RATE)
                provenance[os.path.join(output_dir, clip_name)] = {
                    "location": location_name,
                    "source_files": contributing,
                    "merged": len(set(contributing)) > 1,
                }
                written += 1

            clip_index += 1

    leftover_seconds = len(buffer) / SAMPLE_RATE
    if leftover_seconds > 0:
        print(f"  (dropping {leftover_seconds:.2f}s leftover at end of {location_name} -- nothing left to merge into)")

    print(f"  {location_name}: {len(consumed_files)} raw file(s) -> {written} clips, {skipped_silent} skipped (mostly silent)")

    for f in consumed_files:
        ledger[f] = {"location": location_name}

    return written


def main():
    parser = argparse.ArgumentParser(description="Chunk raw location recordings into fixed-length clips.")
    parser.add_argument("--clip-seconds", type=int, default=CLIP_SECONDS,
                         help=f"Clip window length in seconds (default {CLIP_SECONDS}). "
                              f"Use 10 to match the original project design; 5 for finer-grained clips.")
    args = parser.parse_args()

    ledger = load_ledger()
    provenance = load_provenance()
    total_written = 0
    total_new_files = 0

    for location_name in sorted(os.listdir(RAW_DIR)):
        location_dir = os.path.join(RAW_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        all_paths = sorted(
            p for p in glob.glob(os.path.join(location_dir, "*"))
            if p.lower().endswith(VALID_EXTENSIONS)
        )
        new_paths = [p for p in all_paths if p not in ledger]

        if not new_paths:
            continue

        print(f"Chunking: {location_name} ({len(new_paths)} new file(s))")
        out_dir = os.path.join(OUTPUT_BASE, location_name)
        written = chunk_location(location_name, new_paths, out_dir, ledger, provenance,
                                  clip_seconds=args.clip_seconds)

        total_written += written
        total_new_files += len(new_paths)

    if total_new_files == 0:
        print("No new raw recordings found -- nothing to chunk.")
    else:
        print(f"\nChunked {total_new_files} new raw recording(s) into {total_written} clip(s) total "
              f"({args.clip_seconds}s windows).")

    save_ledger(ledger)
    save_provenance(provenance)


# ---- Single-file chunking, used by predict_incoming.py and verify_test_audio.py
# for scoring/testing one arbitrary file. No ledger, no location grouping --
# just cuts one file into clip_seconds windows.
#
# CHANGE: a file that is entirely shorter than one window (e.g. a 3-10s
# downloaded test clip, vs a full clip_seconds window) is no longer silently
# dropped. It's used whole, as a single clip, with no padding -- the feature
# extractor's stats (mean/std/etc over the clip) work fine on a shorter clip,
# and padding would distort loudness-based features exactly like merging
# already avoids elsewhere in this file. Leftover *tails* after full windows
# have already been cut (e.g. a 47s file at 10s windows leaving a 7s tail)
# are still dropped, same as before -- that's a true leftover with nothing to
# merge it into, distinct from a clip that was simply always short.
def chunk_file(file_path, output_dir, prefix="clip", skip_silent=True, clip_seconds=CLIP_SECONDS):
    os.makedirs(output_dir, exist_ok=True)
    clip_samples = clip_seconds * SAMPLE_RATE
    audio, sr = load_audio_any_format(file_path, SAMPLE_RATE)
    audio = normalize_audio(audio)

    written_paths = []

    if len(audio) < clip_samples:
        duration = len(audio) / SAMPLE_RATE
        if skip_silent and is_mostly_silent(audio):
            print(f"  (skipping {file_path} -- mostly silent, {duration:.2f}s)")
            return written_paths
        out_path = os.path.join(output_dir, f"{prefix}_0000.wav")
        sf.write(out_path, audio, SAMPLE_RATE)
        written_paths.append(out_path)
        print(f"  {file_path}: {duration:.2f}s < {clip_seconds}s window -- used whole clip as-is (no padding)")
        return written_paths

    clip_index = 0
    pos = 0
    while pos + clip_samples <= len(audio):
        clip = audio[pos:pos + clip_samples]
        pos += clip_samples
        if skip_silent and is_mostly_silent(clip):
            clip_index += 1
            continue
        out_path = os.path.join(output_dir, f"{prefix}_{clip_index:04d}.wav")
        sf.write(out_path, clip, SAMPLE_RATE)
        written_paths.append(out_path)
        clip_index += 1

    leftover = len(audio) - pos
    if leftover > 0:
        print(f"  (dropping {leftover / SAMPLE_RATE:.2f}s leftover tail of {file_path} -- "
              f"single-file mode has nothing to merge it into)")

    return written_paths


if __name__ == "__main__":
    main()