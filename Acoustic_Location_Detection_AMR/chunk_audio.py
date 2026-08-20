import os
import glob
import json
import numpy as np
import soundfile as sf

from audio_preprocessing import load_audio_any_format, VALID_EXTENSIONS, TARGET_SR

CLIP_SECONDS = 5
SAMPLE_RATE = TARGET_SR
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE
SILENCE_AMPLITUDE = 0.004
SILENCE_FRACTION_LIMIT = 0.92
CROSSFADE_SAMPLES = int(0.02 * SAMPLE_RATE)  # 20ms crossfade at file-merge seams only
DEAD_AIR_RMS = 0.0015    

RAW_DIR = "Acoustic_Location_Detection_AMR/raw_recordings_amr"
OUTPUT_BASE = "Acoustic_Location_Detection_AMR/dataset"
LEDGER_FILE = "Acoustic_Location_Detection_AMR/chunked_files.json"
CLIP_PROVENANCE_FILE = "Acoustic_Location_Detection_AMR/clip_provenance.json"


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
    start = max(0, seam_position - fade_samples // 2)
    end = min(len(buffer), seam_position + fade_samples // 2)
    n = end - start
    if n <= 1:
        return buffer
    fade = np.linspace(0, 1, n)
    # Blend the pre-seam tail down and post-seam head up across the fade
    # window -- softens the click without meaningfully altering either
    # recording's own content outside the seam.
    pre = buffer[start:seam_position]
    post = buffer[seam_position:end]
    mid = min(len(pre), len(post))
    if mid > 0:
        buffer[seam_position - mid:seam_position] *= (1 - fade[:mid])
        buffer[seam_position:seam_position + mid] *= fade[:mid]
    return buffer


def chunk_location(location_name, raw_paths, output_dir, ledger, provenance, skip_silent=True):
   
    os.makedirs(output_dir, exist_ok=True)

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
        seam_positions = [s for s in seam_positions if s >= CLIP_SAMPLES]  # drop seams already cut past

        while len(buffer) >= CLIP_SAMPLES:
            clip = buffer[:CLIP_SAMPLES]
            buffer = buffer[CLIP_SAMPLES:]
            seam_positions = [s - CLIP_SAMPLES for s in seam_positions if s - CLIP_SAMPLES >= 0]

            # Figure out which source file(s) contributed to this clip
            consumed_here = min(CLIP_SAMPLES, sum(n for _, n in buffer_sources))
            contributing = []
            remaining = CLIP_SAMPLES
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
        written = chunk_location(location_name, new_paths, out_dir, ledger, provenance)

        total_written += written
        total_new_files += len(new_paths)

    if total_new_files == 0:
        print("No new raw recordings found -- nothing to chunk.")
    else:
        print(f"\nChunked {total_new_files} new raw recording(s) into {total_written} clip(s) total.")

    save_ledger(ledger)
    save_provenance(provenance)


def chunk_file(file_path, output_dir, prefix="clip", skip_silent=True):
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = load_audio_any_format(file_path, SAMPLE_RATE)

    written_paths = []
    clip_index = 0
    pos = 0
    while pos + CLIP_SAMPLES <= len(audio):
        clip = audio[pos:pos + CLIP_SAMPLES]
        pos += CLIP_SAMPLES
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
