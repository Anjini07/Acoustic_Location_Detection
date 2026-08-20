import os
import glob
import json
import numpy as np
from sklearn.cluster import KMeans

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize

DATASET_DIR = "Acoustic_Location_Detection/dataset"
OUTPUT_FILE = "Acoustic_Location_Detection/multi_baselines.json"
MAX_CLUSTERS_PER_LOCATION = 3   # e.g. quiet / normal / busy


def choose_k(num_samples, max_k):
    """Don't cluster into more groups than makes sense for how much data
    exists for this location."""
    if num_samples < 6:
        return 1
    return min(max_k, num_samples // 6)  # roughly 6+ clips per cluster minimum


def main():
    global_mean, global_std = load_global_stats()
    multi_baselines = {}

    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        vectors = []
        from audio_preprocessing import list_audio_files
        for wav_path in list_audio_files(location_dir):
            audio, sr = preprocess_audio(wav_path)
            features = extract_features(audio, sr)
            vectors.append(normalize(features, global_mean, global_std))

        vectors = np.array(vectors)
        k = choose_k(len(vectors), MAX_CLUSTERS_PER_LOCATION)

        print(f"{location_name}: {len(vectors)} clips -> {k} sub-baseline(s)")

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(vectors)

        # Store each cluster centroid as a separate baseline for this location
        multi_baselines[location_name] = [
            centroid.tolist() for centroid in kmeans.cluster_centers_
        ]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(multi_baselines, f, indent=2)

    print(f"\nSaved multi-baselines to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()