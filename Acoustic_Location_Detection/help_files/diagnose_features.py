import os
import glob
import json
import numpy as np

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features

DATASET_DIR = "Acoustic_Location_Detection/dataset"


def main():
    # 1. Count recordings per location
    print("=== Recording counts per location ===")
    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue
        wav_files = glob.glob(os.path.join(location_dir, "*.wav"))
        print(f"  {location_name}: {len(wav_files)} recording(s)")

    # 2. Check global_stats.json for near-zero std values
    with open("Acoustic_Location_Detection/global_stats.json", "r") as f:
        stats = json.load(f)
    std = np.array(stats["std"])
    print(f"\n=== global_stats.json ===")
    print(f"num_files_used: {stats['num_files_used']}")
    print(f"Smallest std values (these dominate z-score if too small):")
    smallest_idx = np.argsort(std)[:10]
    for idx in smallest_idx:
        print(f"  feature[{idx}] std = {std[idx]:.6f}")

    # 3. Compare RAW (non-normalized) cosine similarity vs baseline
    print("\n=== Raw cosine similarity (no z-score normalization) ===")
    with open("Acoustic_Location_Detection/baselines.json", "r") as f:
        baselines = json.load(f)

    test_file = "Acoustic_Location_Detection/test_audio/home/Home3.wav"  # change as needed
    audio, sr = preprocess_audio(test_file)
    raw_vector = np.array(extract_features(audio, sr))

    for location_name, data in baselines.items():
        baseline_vector = np.array(data["vector"])
        dot = np.dot(raw_vector, baseline_vector)
        norms = np.linalg.norm(raw_vector) * np.linalg.norm(baseline_vector)
        raw_score = dot / (norms + 1e-10)
        print(f"  vs {location_name}: {raw_score:.4f}")


if __name__ == "__main__":
    main()