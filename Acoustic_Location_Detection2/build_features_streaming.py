import os
import glob
import csv
import argparse
from collections import deque
import numpy as np

from audio_preprocessing import load_audio_any_format, VALID_EXTENSIONS, TARGET_SR
from feature_extraction import extract_features, FEATURE_NAMES

CLIP_SECONDS = 5
SAMPLE_RATE = TARGET_SR
CLIP_SAMPLES = CLIP_SECONDS * SAMPLE_RATE
SILENCE_AMPLITUDE = 0.01
SILENCE_FRACTION_LIMIT = 0.7


def is_mostly_silent(audio: np.ndarray) -> bool:
    return np.mean(np.abs(audio) < SILENCE_AMPLITUDE) > SILENCE_FRACTION_LIMIT


def process_location(location_name: str, raw_paths: list, label: str = None) -> list:
    """
    Streams every raw file for one location through decode -> window ->
    feature extraction -> discard audio. Long files get split into 5s
    pieces; short files get merged with whatever's left over from the
    previous file in the same location, using a queue of (source_file,
    remaining_audio) so each window keeps an accurate record of exactly
    which source file(s) it was built from.
    """
    queue = deque()  # items: [file_path, remaining_audio_array]
    rows = []
    clip_index = 0
    skipped_silent = 0
    files_used = 0

    for raw_path in raw_paths:
        try:
            audio, sr = load_audio_any_format(raw_path, SAMPLE_RATE)
        except Exception as e:
            print(f"  [skip] {raw_path} -- {e}")
            continue

        files_used += 1
        queue.append([raw_path, audio])

        while sum(len(a) for _, a in queue) >= CLIP_SAMPLES:
            parts = []
            contributing_files = []
            needed = CLIP_SAMPLES

            while needed > 0:
                path, arr = queue[0]
                take = min(needed, len(arr))
                parts.append(arr[:take])
                name = os.path.basename(path)
                if name not in contributing_files:
                    contributing_files.append(name)
                needed -= take

                if take == len(arr):
                    queue.popleft()
                else:
                    queue[0][1] = arr[take:]

            window = np.concatenate(parts)

            if is_mostly_silent(window):
                skipped_silent += 1
                clip_index += 1
                continue

            feats = extract_features(window, SAMPLE_RATE)
            feats["location"] = location_name
            feats["source_files"] = "+".join(contributing_files)
            feats["clip_index"] = clip_index
            if label is not None:
                feats["label"] = label

            rows.append(feats)
            clip_index += 1

    leftover_seconds = sum(len(a) for _, a in queue) / SAMPLE_RATE
    if leftover_seconds > 0:
        print(f"  (dropping {leftover_seconds:.2f}s leftover at end of {location_name} -- nothing left to merge into)")

    print(f"  {location_name}: {files_used} raw file(s) -> {len(rows)} feature rows, {skipped_silent} skipped (mostly silent)")
    return rows


def build(input_dir: str, output_csv: str, flat: bool = False, location_as_label: bool = False) -> None:
    all_rows = []

    if flat:
        raw_paths = sorted(
            p for p in glob.glob(os.path.join(input_dir, "*"))
            if p.lower().endswith(VALID_EXTENSIONS)
        )
        if not raw_paths:
            raise ValueError(f"No .wav/.amr files found in {input_dir}")
        print(f"Processing {len(raw_paths)} unlabeled file(s)")
        all_rows.extend(process_location("unlabeled", raw_paths, label=None))

    else:
        locations = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        )
        if not locations:
            raise ValueError(f"No subfolders found in {input_dir}. Use --flat if recordings aren't sorted yet.")

        for location_name in locations:
            location_dir = os.path.join(input_dir, location_name)
            raw_paths = sorted(
                p for p in glob.glob(os.path.join(location_dir, "*"))
                if p.lower().endswith(VALID_EXTENSIONS)
            )
            if not raw_paths:
                continue

            print(f"Processing: {location_name} ({len(raw_paths)} file(s))")
            label = location_name if location_as_label else None
            all_rows.extend(process_location(location_name, raw_paths, label=label))

    if not all_rows:
        raise ValueError("No feature rows produced -- check input_dir and file formats")

    base_cols = ["location", "source_files", "clip_index"]
    fieldnames = base_cols + FEATURE_NAMES
    if any("label" in r for r in all_rows):
        fieldnames.append("label")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} feature rows to {output_csv}")
    print("No chunk audio was written to disk -- only this CSV and your original raw files exist now.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream raw recordings straight to a features CSV, no chunk files")
    parser.add_argument("--input_dir", default="raw_recordings",
                         help="Folder of raw recordings -- with location subfolders, or flat if --flat is set")
    parser.add_argument("--output_csv", default="features.csv")
    parser.add_argument("--flat", action="store_true",
                         help="Set this if input_dir has no subfolders (recordings not sorted/labeled yet)")
    parser.add_argument("--location_as_label", action="store_true",
                         help="Use each subfolder name directly as the label column (e.g. quiet/moderate/noisy "
                              "folders you already sorted by hand). Skip this if folder names are locations, "
                              "not conditions, and you want cluster_features.py to discover labels instead.")
    args = parser.parse_args()

    build(args.input_dir, args.output_csv, args.flat, args.location_as_label)