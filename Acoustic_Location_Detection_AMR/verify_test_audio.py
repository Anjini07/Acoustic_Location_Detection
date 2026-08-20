import os
import glob
import json
import random
import shutil
import numpy as np

from audio_preprocessing import list_audio_files, preprocess_audio
from feature_extraction import extract_features
from drift_detection import (
    load_global_stats,
    normalize,
    cosine_similarity,
    classify_similarity,
)
from chunk_audio import chunk_file, CLIP_SECONDS

RAW_DIR = "Acoustic_Location_Detection_AMR/raw_recordings_amr"
MULTI_BASELINES_FILE = "Acoustic_Location_Detection_AMR/multi_baselines.json"
CLIP_PROVENANCE_FILE = "Acoustic_Location_Detection_AMR/clip_provenance.json"
SPLIT_FILE = "Acoustic_Location_Detection_AMR/test_split.json"
OUTPUT_FILE = "Acoustic_Location_Detection_AMR/verification_report.json"
TEMP_CHUNK_DIR = "Acoustic_Location_Detection_AMR/_verify_chunks"

TEST_FRACTION = 0.20
RANDOM_SEED = 42  


def load_multi_baselines():
    with open(MULTI_BASELINES_FILE, "r") as f:
        return json.load(f)


def score_against_location(test_vector, location_baselines):
    scores = [
        cosine_similarity(test_vector, np.array(centroid))
        for centroid in location_baselines
    ]
    return max(scores)


def load_or_create_split():

    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, "r") as f:
            return json.load(f)

    rng = random.Random(RANDOM_SEED)
    split = {}

    for location_name in sorted(os.listdir(RAW_DIR)):
        location_dir = os.path.join(RAW_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        all_files = list_audio_files(location_dir)
        if not all_files:
            continue

        shuffled = all_files[:]
        rng.shuffle(shuffled)

        n_test = max(1, round(len(shuffled) * TEST_FRACTION))
        split[location_name] = shuffled[:n_test]

    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Created new test_split.json ({TEST_FRACTION:.0%} per location, seed={RANDOM_SEED}).")
    return split


def check_leakage(split):
    
    if not os.path.exists(CLIP_PROVENANCE_FILE):
        print("NOTE: no clip_provenance.json found yet -- skipping leakage check "
              "(this is expected if dataset/ hasn't been built via chunk_audio.py).")
        return

    with open(CLIP_PROVENANCE_FILE, "r") as f:
        provenance = json.load(f)

    chunked_sources = set()
    for entry in provenance.values():
        for src in entry.get("source_files", []):
            chunked_sources.add(os.path.normpath(os.path.abspath(src)))

    leaked = []
    for location_name, files in split.items():
        for f in files:
            if os.path.normpath(os.path.abspath(f)) in chunked_sources:
                leaked.append((location_name, f))

    if leaked:
        print(f"\n{'!'*60}")
        print(f"LEAKAGE WARNING: {len(leaked)} held-out test file(s) were already")
        print(f"chunked into dataset/ and used to build multi_baselines.json.")
        print(f"Accuracy below is OPTIMISTIC, not honest, for these files:")
        for location_name, f in leaked:
            print(f"  [{location_name}] {f}")
        print(f"To get an honest number: remove these files' chunks from dataset/,")
        print(f"rebuild global_stats.json + multi_baselines.json without them, then")
        print(f"rerun this script. Or delete test_split.json to redraw a split and")
        print(f"manually keep the redrawn files out of chunk_audio.py's input next time.")
        print(f"{'!'*60}\n")
    else:
        print("Leakage check passed: none of the held-out test files were used to build the baselines.\n")


def evaluate_file(file_path, expected_location, multi_baselines, global_mean, global_std):
    chunk_paths = chunk_file(file_path, TEMP_CHUNK_DIR, prefix="verify", skip_silent=True)

    if not chunk_paths:
        print(f"  WARNING: no usable (non-silent) audio in {file_path} -- skipping")
        return None

    all_window_scores = {loc: [] for loc in multi_baselines}
    for chunk_path in chunk_paths:
        audio, sr = preprocess_audio(chunk_path)
        vector = normalize(extract_features(audio, sr), global_mean, global_std)
        for location_name, location_baselines in multi_baselines.items():
            all_window_scores[location_name].append(score_against_location(vector, location_baselines))

    for p in chunk_paths:
        os.remove(p)

    avg_scores = {loc: float(np.mean(scores)) for loc, scores in all_window_scores.items()}
    best_match = max(avg_scores, key=avg_scores.get)
    best_score = avg_scores[best_match]
    status, action = classify_similarity(best_score)

    expected_score = avg_scores.get(expected_location)
    expected_status, expected_action = classify_similarity(expected_score)

    sorted_scores = sorted(avg_scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    return {
        "expected_location": expected_location,
        "num_windows": len(chunk_paths),
        "all_scores": avg_scores,
        "best_match": best_match,
        "best_score": best_score,
        "best_status": status,
        "expected_score": expected_score,
        "expected_status": expected_status,
        "correct_match": best_match == expected_location,
        "margin": margin,
    }


def main():
    global_mean, global_std = load_global_stats()
    multi_baselines = load_multi_baselines()

    split = load_or_create_split()
    check_leakage(split)

    os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)
    report = {}

    for expected_location, file_paths in split.items():
        if expected_location not in multi_baselines:
            print(f"WARNING: '{expected_location}' has no entry in multi_baselines.json -- skipping its test files")
            continue

        for file_path in file_paths:
            file_key = f"{expected_location}/{os.path.basename(file_path)}"
            print(f"\nTesting: {file_key}")

            result = evaluate_file(file_path, expected_location, multi_baselines, global_mean, global_std)
            if result is None:
                continue

            print(f"  Split into {result['num_windows']} window(s) of {CLIP_SECONDS}s")
            print(f"  Best match: {result['best_match']} ({result['best_score']:.4f}) -> {result['best_status']}")
            print(f"  Vs expected '{expected_location}': {result['expected_score']:.4f} -> {result['expected_status']}")

            report[file_key] = result

    shutil.rmtree(TEMP_CHUNK_DIR, ignore_errors=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    total = len(report)
    correct = sum(1 for r in report.values() if r["correct_match"])
    print(f"\n{'='*50}")
    print(f"Summary: {correct}/{total} held-out test files matched their expected location")
    print(f"(20% split per location, from raw_recordings_amr, seed={RANDOM_SEED} -- see test_split.json)")
    print(f"Full report saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
