import os
import glob
import json
import numpy as np

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize

DATASET_DIR = "Acoustic_Location_Detection/dataset"
OUTPUT_FILE = "Acoustic_Location_Detection/feature_weights.json"


def main():
    global_mean, global_std = load_global_stats()

    # Collect normalized feature vectors, grouped by location
    location_vectors = {}
    for location_name in os.listdir(DATASET_DIR):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        vectors = []
        for wav_path in glob.glob(os.path.join(location_dir, "*.wav")):
            audio, sr = preprocess_audio(wav_path)
            features = extract_features(audio, sr)
            vectors.append(normalize(features, global_mean, global_std))

        location_vectors[location_name] = np.array(vectors)

    num_features = 104
    fisher_scores = np.zeros(num_features)

    overall_mean = np.mean(
        [v.mean(axis=0) for v in location_vectors.values()], axis=0
    )

    for f in range(num_features):
        between_var = 0.0
        within_var = 0.0

        for loc, vectors in location_vectors.items():
            loc_values = vectors[:, f]
            loc_mean = loc_values.mean()
            n = len(loc_values)

            between_var += n * (loc_mean - overall_mean[f]) ** 2
            within_var += ((loc_values - loc_mean) ** 2).sum()

        fisher_scores[f] = between_var / (within_var + 1e-8)

    # Normalize scores to use as weights (0 to 1 range)
    weights = fisher_scores / (fisher_scores.max() + 1e-8)

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"weights": weights.tolist()}, f, indent=2)

    print("Top 10 most discriminative features (indices):")
    for idx in np.argsort(fisher_scores)[::-1][:10]:
        print(f"  feature[{idx}] score={fisher_scores[idx]:.4f}")

    print("\n10 least useful features (indices):")
    for idx in np.argsort(fisher_scores)[:10]:
        print(f"  feature[{idx}] score={fisher_scores[idx]:.4f}")

    print(f"\nSaved weights to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()