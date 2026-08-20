import os
import glob
import json
import numpy as np

from audio_preprocessing import preprocess_audio
from feature_extraction import extract_features
from drift_detection import load_global_stats, normalize, cosine_similarity

DATASET_DIR = "Acoustic_Location_Detection/dataset"
MULTI_BASELINES_FILE = "Acoustic_Location_Detection/multi_baselines.json"

def load_multi_baselines():
    with open(MULTI_BASELINES_FILE, "r") as f:
        return json.load(f)


def score_against_location(test_vector, location_baselines):
    """A location can have multiple baselines (different conditions).
    The location's score is the BEST match among its own sub-baselines —
    the test clip only needs to resemble ONE known condition for that
    location, not all of them. Centroids in multi_baselines.json are
    already normalized (they were built from normalized vectors), so no
    extra normalize() call is needed here."""
    scores = [
        cosine_similarity(test_vector, np.array(centroid))
        for centroid in location_baselines
    ]
    return max(scores)

def main():
    global_mean, global_std = load_global_stats()
    baselines = load_multi_baselines()

    same_scores = []
    diff_scores = []

    print(f"{'File':45s} {'Own location':>13s} {'Best other':>12s} {'Gap':>8s}")
    print("-" * 82)

    for location_name in sorted(os.listdir(DATASET_DIR)):
        location_dir = os.path.join(DATASET_DIR, location_name)
        if not os.path.isdir(location_dir):
            continue

        for wav_path in sorted(glob.glob(os.path.join(location_dir, "*.wav"))):
            audio, sr = preprocess_audio(wav_path)
            vector = normalize(extract_features(audio, sr), global_mean, global_std)

            own_score = None
            other_scores = {}
            for loc, location_baselines in baselines.items():
                score = score_against_location(vector, location_baselines)
                if loc == location_name:
                    own_score = score
                    same_scores.append(score)
                else:
                    other_scores[loc] = score
                    diff_scores.append(score)

            best_other = max(other_scores.values())
            gap = own_score - best_other
            flag = "  <-- WRONG (own score lower than a different location)" if gap < 0 else ""

            fname = f"{location_name}/{os.path.basename(wav_path)}"
            print(f"{fname:45s} {own_score:13.4f} {best_other:12.4f} {gap:8.4f}{flag}")

    same_scores = np.array(same_scores)
    diff_scores = np.array(diff_scores)
    print(f"\nSame-location:      count={len(same_scores)}  mean={same_scores.mean():.4f}  std={same_scores.std():.4f}")
    print(f"Different-location: count={len(diff_scores)}  mean={diff_scores.mean():.4f}  std={diff_scores.std():.4f}")


if __name__ == "__main__":
    main()